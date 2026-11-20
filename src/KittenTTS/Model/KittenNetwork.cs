using KittenTTS.IO;
using KittenTTS.Numerics;

namespace KittenTTS.Model;

/// <summary>The output of one forward pass: the waveform (float32, 24 kHz) and every token's duration in frames
/// (600 samples per frame).</summary>
public sealed record ModelOutput(float[] Waveform, int[] Durations);

/// <summary>
/// One KittenTTS 2 nano / pico / mini network, computing what the ONNX graph and the PyTorch backend compute:
/// <c>input_ids [T], style [256], speed → waveform, durations</c>. <c>style[..128]</c> conditions the decoder (timbre),
/// <c>style[128..]</c> the prosody predictor.
/// <code>
/// tokens ─► TextEncoder ─────────────────────────────────────────────┐ (aligned by the durations)
///        ─► PL-BERT ─► bert_encoder ─► DurationEncoder(s_p) ─► LSTM ─► durations
///                                          └──(aligned)─► F0Ntrain(s_p) ─► F0, N
///                                                                         └─► Decoder(s_a) ─► waveform
/// </code>
/// Thread-safe: weights are read-only and every call allocates its own activations.
/// </summary>
public sealed class KittenNetwork : IDisposable
{
    private readonly TextEncoder _textEncoder;
    private readonly Albert _bert;
    private readonly Linear _bertEncoder;
    private readonly ProsodyPredictor _predictor;
    private readonly Decoder _decoder;

    /// <summary>Stage outputs, for parity tests: <c>t_en</c>, <c>bert</c>, <c>d_en</c>, <c>d</c>, <c>durations</c>,
    /// <c>F0</c>, <c>N</c>, <c>asr</c>, as float arrays in channels-last layout.</summary>
    public delegate void TraceCallback(string name, float[] data, int[] shape);

    public KittenNetwork(ITensorSource weights, Architecture architecture)
    {
        Architecture = architecture;
        if (weights is OnnxWeights onnx)
        {
            var modules = Decoder.InstanceNormModules(architecture);
            if (onnx.AffinePairs > 0) onnx.AssignAffine(modules);
        }
        var r = new WeightReader(weights);
        _textEncoder = new TextEncoder(r, architecture.TextEncoderKernelSize);
        _bert = new Albert(r, architecture.AlbertHeads, architecture.AlbertLayers, architecture.AlbertLayerNormEps);
        _bertEncoder = Linear.Load(r, "bert_encoder");
        _predictor = new ProsodyPredictor(r);
        _decoder = new Decoder(r, architecture, onnxArithmetic: weights is OnnxWeights);
    }

    public Architecture Architecture { get; }

    /// <summary>Samples per duration frame (the product of the upsampling rates and the iSTFT hop, then ×2 for the
    /// decoder's last upsampling block): 600 for every released model.</summary>
    public int SamplesPerFrame => 2 * Architecture.Decoder.UpsampleRates.Aggregate(1, (a, b) => a * b) * Architecture.Decoder.IstftHop;

    /// <summary>Maximum number of tokens per call (PL-BERT's position table).</summary>
    public int MaxTokens => _bert.MaxPositions;

    public ModelOutput Forward(int[] tokens, ReadOnlySpan<float> style, float speed, SynthesisNoise noise, Parallelism par,
                               TraceCallback? trace = null)
    {
        int sd = Architecture.StyleDim;
        if (style.Length != 2 * sd) throw new ArgumentException($"The style vector must have {2 * sd} values, got {style.Length}.");
        if (tokens.Length == 0) throw new ArgumentException("No tokens.");
        if (speed <= 0) throw new ArgumentOutOfRangeException(nameof(speed), "Speed must be positive.");
        var sAcoustic = style[..sd].ToArray();
        var sProsody = style[sd..].ToArray();

        Profiler.Start();
        var tEn = _textEncoder.Forward(tokens, par);
        Profiler.Mark("text_encoder");
        var bert = _bert.Forward(tokens, par);
        Profiler.Mark("bert");
        var dEn = _bertEncoder.Forward(bert, par);
        var d = _predictor.EncodeText(dEn, sProsody, par);
        var durations = _predictor.Durations(d, speed, par);
        Profiler.Mark("durations");
        var en = F.Expand(d, durations);
        var (f0, n) = _predictor.F0N(en, sProsody, par);
        DebugDump.Write("F0", f0, [f0.Length]);
        Profiler.Mark("f0n");
        var asr = F.Expand(tEn, durations);
        var wav = _decoder.Forward(asr, f0, n, sAcoustic, noise, par);
        Profiler.Mark("decoder");

        if (trace != null)
        {
            trace("t_en", tEn.Data, [tEn.T, tEn.C]);
            trace("bert", bert.Data, [bert.T, bert.C]);
            trace("d_en", dEn.Data, [dEn.T, dEn.C]);
            trace("d", d.Data, [d.T, d.C]);
            trace("durations", durations.Select(x => (float)x).ToArray(), [durations.Length]);
            trace("F0", f0, [f0.Length]);
            trace("N", n, [n.Length]);
            trace("waveform", wav, [wav.Length]);
        }
        return new ModelOutput(wav, durations);
    }

    /// <summary>The decoder alone, on given text features ([T, hidden]), durations, F0 and energy curves (parity tests
    /// feed it the reference's own intermediates, which isolates the decoder from upstream float rounding).</summary>
    internal float[] Decode(float[] tEn, int hidden, int[] durations, float[] f0, float[] n, ReadOnlySpan<float> style,
                            SynthesisNoise noise, Parallelism par)
    {
        var asr = F.Expand(new Act(tEn.Length / hidden, hidden, tEn), durations);
        return _decoder.Forward(asr, f0, n, style[..Architecture.StyleDim], noise, par);
    }

    public void Dispose()
    {
        _textEncoder.Dispose();
        _bert.Dispose();
        _bertEncoder.Dispose();
        _predictor.Dispose();
        _decoder.Dispose();
    }
}
