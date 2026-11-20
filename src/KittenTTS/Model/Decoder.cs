using KittenTTS.Audio;
using KittenTTS.Numerics;

namespace KittenTTS.Model;

/// <summary>
/// The StyleTTS 2 decoder: F0 / energy downsampling convs, the <c>encode</c> and four <c>decode</c>
/// <c>AdainResBlk1d</c> blocks (the last upsampling ×2), then the iSTFTNet <see cref="Generator"/>. One class for
/// both variants: <c>istftnet</c> (mini: 1024 / 512 channels, three resblocks per stage, softplus magnitude) and
/// <c>istftnet_mini_2</c> (nano / pico: narrower, two resblocks, exp magnitude, affine instance norms); the widths
/// come from the weight shapes.
/// </summary>
internal sealed class Decoder : IDisposable
{
    internal readonly Conv1d _f0Conv, _nConv, _asrRes;
    internal readonly AdainResBlk1d _encode;
    internal readonly AdainResBlk1d[] _decode;
    internal readonly Generator _generator;

    public Decoder(WeightReader r, Architecture arch, bool onnxArithmetic)
    {
        _f0Conv = Conv1d.Load(r, "decoder.F0_conv", stride: 2, padding: 1);
        _nConv = Conv1d.Load(r, "decoder.N_conv", stride: 2, padding: 1);
        _asrRes = Conv1d.Load(r, "decoder.asr_res.0");
        _encode = new AdainResBlk1d(r, "decoder.encode", false);
        _decode = [new(r, "decoder.decode.0", false), new(r, "decoder.decode.1", false),
                   new(r, "decoder.decode.2", false), new(r, "decoder.decode.3", true)];
        _generator = new Generator(r, arch.Decoder, onnxArithmetic);
    }

    /// <summary>[frames, hidden] aligned text features, F0 and energy at 2 × frames → waveform (600 samples per frame).</summary>
    public float[] Forward(Act asr, float[] f0Curve, float[] n, ReadOnlySpan<float> s, SynthesisNoise noise, Parallelism par)
    {
        var f0 = _f0Conv.Forward(new Act(f0Curve.Length, 1, f0Curve), par);
        var nn = _nConv.Forward(new Act(n.Length, 1, n), par);
        var x = _encode.Forward(Act.ConcatChannels(asr, f0, nn), s, par);
        var asrRes = _asrRes.Forward(asr, par);
        bool res = true;
        foreach (var block in _decode)
        {
            if (res) x = Act.ConcatChannels(x, asrRes, f0, nn);
            bool up = ReferenceEquals(block, _decode[^1]);
            x = block.Forward(x, s, par);
            if (up) res = false;
        }
        return _generator.Forward(x, s, f0Curve, noise, par);
    }

    /// <summary>The decoder's instance-norm modules in execution order (the ONNX graphs store their affine
    /// parameters unnamed, in this order).</summary>
    public static List<string> InstanceNormModules(Architecture arch)
    {
        var list = new List<string>();
        void Block(string m) { list.Add(m + ".norm1.norm"); list.Add(m + ".norm2.norm"); }
        void Res(string m) { for (int j = 0; j < 3; j++) { list.Add($"{m}.adain1.{j}.norm"); list.Add($"{m}.adain2.{j}.norm"); } }
        Block("decoder.encode");
        for (int i = 0; i < 4; i++) Block($"decoder.decode.{i}");
        var d = arch.Decoder;
        for (int i = 0; i < d.UpsampleRates.Length; i++)
        {
            Res($"decoder.generator.noise_res.{i}");
            for (int j = 0; j < d.ResblockKernelSizes.Length; j++) Res($"decoder.generator.resblocks.{i * d.ResblockKernelSizes.Length + j}");
        }
        return list;
    }

    public void Dispose()
    {
        _f0Conv.Dispose(); _nConv.Dispose(); _asrRes.Dispose(); _encode.Dispose();
        foreach (var b in _decode) b.Dispose();
        _generator.Dispose();
    }
}

/// <summary>
/// The iSTFTNet generator with a harmonic-plus-noise source (hn-NSF): the F0 curve drives a sine source of nine
/// harmonics whose STFT is injected at every upsampling stage; the last convolution predicts a log/softplus magnitude
/// and a phase (through sin) for an inverse STFT of n_fft 20, hop 5.
/// </summary>
internal sealed class Generator : IDisposable
{
    private const int Harmonics = 9;            // harmonic_num 8 + the fundamental
    private const float SineAmp = 0.1f, NoiseStd = 0.003f, VoicedThreshold = 10f;
    private const int SampleRate = 24000;

    internal readonly ConvTranspose1d[] _ups;
    internal readonly Conv1d[] _noiseConvs;
    internal readonly AdaInResBlock1[] _noiseRes;
    internal readonly AdaInResBlock1[] _resblocks;
    internal readonly Conv1d _convPost;
    internal readonly float[] _lLinearW;
    internal readonly float _lLinearB;
    internal readonly int _kernels, _nFft, _hop, _upsampleScale;
    internal readonly bool _softplus;
    internal readonly bool _fusedInterpolation;
    internal readonly Stft _stft;

    /// <param name="r">Weights.</param>
    /// <param name="d">Decoder shape.</param>
    /// <param name="onnxArithmetic">Repeat the ONNX graph's operation order where it differs from PyTorch's in an
    /// ill-conditioned spot (the sine source's phase interpolation): set for weights read from the ONNX graph.</param>
    public Generator(WeightReader r, DecoderArchitecture d, bool onnxArithmetic)
    {
        _fusedInterpolation = !onnxArithmetic;
        int n = d.UpsampleRates.Length;
        _kernels = d.ResblockKernelSizes.Length;
        _nFft = d.IstftNFft;
        _hop = d.IstftHop;
        _upsampleScale = d.UpsampleRates.Aggregate(1, (a, b) => a * b) * _hop;
        _softplus = d.Type == "istftnet";
        _ups = new ConvTranspose1d[n];
        _noiseConvs = new Conv1d[n];
        _noiseRes = new AdaInResBlock1[n];
        _resblocks = new AdaInResBlock1[n * _kernels];
        for (int i = 0; i < n; i++)
        {
            int u = d.UpsampleRates[i], k = d.UpsampleKernelSizes[i];
            _ups[i] = ConvTranspose1d.Load(r, $"decoder.generator.ups.{i}", stride: u, padding: (k - u) / 2);
            for (int j = 0; j < _kernels; j++)
                _resblocks[i * _kernels + j] = new AdaInResBlock1(r, $"decoder.generator.resblocks.{i * _kernels + j}",
                                                                  d.ResblockKernelSizes[j], d.ResblockDilationSizes[j]);
            if (i + 1 < n)
            {
                int strideF0 = d.UpsampleRates.Skip(i + 1).Aggregate(1, (a, b) => a * b);
                _noiseConvs[i] = Conv1d.Load(r, $"decoder.generator.noise_convs.{i}", stride: strideF0, padding: (strideF0 + 1) / 2);
                _noiseRes[i] = new AdaInResBlock1(r, $"decoder.generator.noise_res.{i}", 7, [1, 3, 5]);
            }
            else
            {
                _noiseConvs[i] = Conv1d.Load(r, $"decoder.generator.noise_convs.{i}");
                _noiseRes[i] = new AdaInResBlock1(r, $"decoder.generator.noise_res.{i}", 11, [1, 3, 5]);
            }
        }
        _convPost = Conv1d.Load(r, "decoder.generator.conv_post", padding: 3);
        _lLinearW = r.Get("decoder.generator.m_source.l_linear.weight", 1, Harmonics);
        _lLinearB = r.Get("decoder.generator.m_source.l_linear.bias", 1)[0];
        _stft = new Stft(_nFft, _hop);
    }

    public float[] Forward(Act x, ReadOnlySpan<float> s, float[] f0Curve, SynthesisNoise noise, Parallelism par)
    {
        var (_, har) = HarmonicSource(f0Curve, noise, par);
        int n = _ups.Length;
        for (int i = 0; i < n; i++)
        {
            x = F.LeakyRelu(x, 0.1f);
            var xSource = _noiseConvs[i].Forward(har, par);
            xSource = _noiseRes[i].Forward(xSource, s, par);
            x = _ups[i].Forward(x, par);
            if (i == n - 1) x = F.ReflectPadLeft1(x);
            F.AddInPlace(x, xSource);
            Act? xs = null;
            for (int j = 0; j < _kernels; j++)
            {
                var y = _resblocks[i * _kernels + j].Forward(x, s, par);
                if (xs == null) xs = y;
                else F.AddInPlace(xs, y);
            }
            F.ScaleInPlace(xs!, 1f / _kernels);
            x = xs!;
        }
        x = F.LeakyRelu(x, 0.01f, inPlace: true);
        return Istft(_convPost.Forward(x, par));
    }

    /// <summary>conv_post output [frames, n_fft + 2] → waveform: exp / softplus magnitude, sin phase, iSTFT.</summary>
    internal float[] Istft(Act x)
    {
        int bins = _nFft / 2 + 1;
        var mag = new float[(long)x.T * bins];
        var phase = new float[(long)x.T * bins];
        for (int t = 0; t < x.T; t++)
        {
            var row = x.Row(t);
            for (int b = 0; b < bins; b++)
            {
                mag[t * bins + b] = _softplus ? F.Softplus(row[b]) : MathF.Exp(row[b]);
                phase[t * bins + b] = MathF.Sin(row[bins + b]);
            }
        }
        return _stft.Inverse(mag, phase, x.T);
    }

    /// <summary>
    /// <c>SourceModuleHnNSF</c> followed by the STFT of its output: [frames + 1, n_fft + 2] (magnitudes, then phases).
    /// The sine generator works at the F0 frame rate exactly as <c>SineGen._f02sine</c> does: per-sample normalized
    /// frequency, linear downsampling by the upsample scale, cumulative phase, linear upsampling back. (Its random
    /// initial phase lands on sample 0, which the downsampling never reads, so it has no effect and is omitted.)
    /// </summary>
    internal (float[] Source, Act Har) HarmonicSource(float[] f0Curve, SynthesisNoise noise, Parallelism par)
    {
        int frames = f0Curve.Length;                    // F0 at twice the decoder frame rate
        int scale = _upsampleScale;
        int len = frames * scale;
        // rad_values[f, h] = (f0 · (h + 1) / sr) % 1 is constant over each frame's `scale` samples, so the linear
        // downsampling (taps at 300 d + 149 and + 150, both inside frame d) returns it unchanged.
        // torch.cumsum (and the ONNX graph's CumSum, behind a Cast to double) accumulates in double.
        var phase = new float[(long)frames * Harmonics];
        var cum = new double[Harmonics];
        for (int f = 0; f < frames; f++)
            for (int h = 0; h < Harmonics; h++)
            {
                float fn = f0Curve[f] * (h + 1);
                float rad = fn / SampleRate;
                rad -= MathF.Floor(rad);
                cum[h] += rad;
                phase[f * Harmonics + h] = (float)cum[h] * (2f * MathF.PI) * scale;
            }

        // linear upsampling ×scale (align_corners=False), sin, voicing, noise, merge (Linear 9 → 1 + tanh)
        float inv = (float)(1.0 / scale);
        var source = new float[len];
        var eps = noise.Normal(len * Harmonics);
        par.ForRange(len, 4096, (s0, s1) =>
        {
        for (int i = s0; i < s1; i++)
        {
            float src = (i + 0.5f) * inv - 0.5f;
            if (src < 0) src = 0;
            int i0 = Math.Min((int)MathF.Floor(src), frames - 1);
            int i1 = i0 < frames - 1 ? i0 + 1 : i0;
            float l1 = Math.Clamp(src - i0, 0f, 1f), l0 = 1f - l1;
            float f0 = f0Curve[i / scale];           // nearest upsampling of F0
            bool voiced = f0 > VoicedThreshold;
            float uv = voiced ? 1f : 0f;
            float noiseAmp = uv * NoiseStd + (1f - uv) * SineAmp / 3f;
            float acc = 0;
            for (int h = 0; h < Harmonics; h++)
            {
                // The phase reaches 10⁴ rad and its float rounding goes straight into the sine, so the interpolation
                // repeats the reference's exact operations: PyTorch's kernel compiles to fma(p0, w0, p1 · w1), the
                // ONNX graph runs separate Mul / Mul / Add nodes.
                float p = _fusedInterpolation
                    ? MathF.FusedMultiplyAdd(phase[i0 * Harmonics + h], l0, phase[i1 * Harmonics + h] * l1)
                    : phase[i0 * Harmonics + h] * l0 + phase[i1 * Harmonics + h] * l1;
                float sine = (float)Math.Sin(p) * SineAmp;
                float v = sine * uv + noiseAmp * (eps.Length == 0 ? 0f : eps[(long)i * Harmonics + h]);
                acc = MathF.FusedMultiplyAdd(v, _lLinearW[h], acc);
            }
            source[i] = MathF.Tanh(acc + _lLinearB);
        }
        });
        var (mag, ph, stftFrames) = _stft.Forward(source);
        int bins = _nFft / 2 + 1;
        var har = new Act(stftFrames, 2 * bins);
        for (int t = 0; t < stftFrames; t++)
        {
            var row = har.Row(t);
            for (int b = 0; b < bins; b++)
            {
                row[b] = mag[t * bins + b];
                row[bins + b] = ph[t * bins + b];
            }
        }
        return (source, har);
    }

    public void Dispose()
    {
        foreach (var u in _ups) u.Dispose();
        foreach (var c in _noiseConvs) c.Dispose();
        foreach (var b in _noiseRes) b.Dispose();
        foreach (var b in _resblocks) b.Dispose();
        _convPost.Dispose();
    }
}
