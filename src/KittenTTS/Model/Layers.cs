using System.Numerics.Tensors;
using KittenTTS.Numerics;

namespace KittenTTS.Model;

/// <summary><c>nn.Linear</c>: <c>y = x · Wᵀ + b</c> on the packed GEMM.</summary>
internal sealed unsafe class Linear : IDisposable
{
    private readonly PackedMatrix _w;
    private readonly float[]? _bias;

    public Linear(float[] w, float[]? bias, int outFeatures, int inFeatures)
    {
        if (w.Length != (long)outFeatures * inFeatures) throw new ArgumentException("Linear weight size mismatch.");
        In = inFeatures;
        Out = outFeatures;
        fixed (float* p = w) _w = PackedMatrix.FromLinear(p, outFeatures, inFeatures, hugePages: false);
        _bias = bias;
    }

    public static Linear Load(WeightReader r, string module)
    {
        var shape = r.Shape(module + ".weight");
        return new Linear(r.Get(module + ".weight"), r.TryGet(module + ".bias"), shape[0], shape[1]);
    }

    public int In { get; }
    public int Out { get; }

    public Act Forward(Act x, Parallelism par, Activation act = Activation.None)
    {
        if (x.C != In) throw new ArgumentException($"Linear expects {In} features, got {x.C}.");
        var y = new Act(x.T, Out);
        fixed (float* a = x.Data, c = y.Data, b = _bias)
            Gemm.Multiply(a, In, x.T, _w, c, Out, par, b, accumulate: false, act);
        return y;
    }

    public void Dispose() => _w.Dispose();
}

/// <summary>A linear layer applied to one vector (AdaIN / AdaLayerNorm style projections): plain float dot products.</summary>
internal sealed class VectorLinear(float[] w, float[]? bias, int outFeatures, int inFeatures)
{
    public static VectorLinear Load(WeightReader r, string module)
    {
        var shape = r.Shape(module + ".weight");
        return new VectorLinear(r.Get(module + ".weight"), r.TryGet(module + ".bias"), shape[0], shape[1]);
    }

    public int Out => outFeatures;

    public float[] Forward(ReadOnlySpan<float> x)
    {
        if (x.Length != inFeatures) throw new ArgumentException($"Expected {inFeatures} features, got {x.Length}.");
        var y = new float[outFeatures];
        for (int o = 0; o < outFeatures; o++)
        {
            // float accumulation in input order, like a float32 GEMV
            float s = 0;
            var row = w.AsSpan(o * inFeatures, inFeatures);
            for (int i = 0; i < inFeatures; i++) s = MathF.FusedMultiplyAdd(row[i], x[i], s);
            y[o] = s + (bias?[o] ?? 0f);
        }
        return y;
    }
}

/// <summary>
/// <c>nn.Conv1d</c> (groups = 1) on the packed GEMM, with no im2col buffer: the input is copied once into a zero-padded
/// buffer, and for dilation 1 the GEMM's A operand is that buffer with overlapping rows (row <c>t</c> starts at padded
/// step <c>t · stride</c> and spans <c>K · Cin</c> values, exactly the receptive field in tap-major order). Dilated
/// convolutions run one accumulating GEMM per tap instead.
/// </summary>
internal sealed unsafe class Conv1d : IDisposable
{
    private readonly PackedMatrix? _full;
    private readonly PackedMatrix[]? _taps;
    private readonly float[]? _bias;

    public Conv1d(float[] w, float[]? bias, int cout, int cin, int k, int stride = 1, int padding = 0, int dilation = 1)
    {
        if (w.Length != (long)cout * cin * k) throw new ArgumentException("Conv1d weight size mismatch.");
        if (dilation > 1 && stride != 1) throw new NotSupportedException("Dilated strided convolutions are not supported.");
        Cin = cin; Cout = cout; K = k; Stride = stride; Padding = padding; Dilation = dilation;
        _bias = bias;
        if (dilation == 1)
        {
            // [Cout, Cin, K] -> [Cout, K * Cin] (tap-major rows)
            var r = new float[w.Length];
            for (int o = 0; o < cout; o++)
                for (int i = 0; i < cin; i++)
                    for (int t = 0; t < k; t++) r[((long)o * k + t) * cin + i] = w[((long)o * cin + i) * k + t];
            fixed (float* p = r) _full = PackedMatrix.FromLinear(p, cout, k * cin, hugePages: false);
        }
        else
        {
            _taps = new PackedMatrix[k];
            var tap = new float[(long)cout * cin];
            for (int t = 0; t < k; t++)
            {
                for (int o = 0; o < cout; o++)
                    for (int i = 0; i < cin; i++) tap[(long)o * cin + i] = w[((long)o * cin + i) * k + t];
                fixed (float* p = tap) _taps[t] = PackedMatrix.FromLinear(p, cout, cin, hugePages: false);
            }
        }
    }

    public static Conv1d Load(WeightReader r, string module, int stride = 1, int padding = 0, int dilation = 1)
    {
        var shape = r.Shape(module + ".weight");
        return new Conv1d(r.Get(module + ".weight"), r.TryGet(module + ".bias"), shape[0], shape[1], shape[2], stride, padding, dilation);
    }

    public int Cin { get; }
    public int Cout { get; }
    public int K { get; }
    public int Stride { get; }
    public int Padding { get; }
    public int Dilation { get; }

    public int OutputLength(int t) => (t + 2 * Padding - Dilation * (K - 1) - 1) / Stride + 1;

    public Act Forward(Act x, Parallelism par)
    {
        if (x.C != Cin) throw new ArgumentException($"Conv1d expects {Cin} channels, got {x.C}.");
        int tout = OutputLength(x.T);
        if (tout <= 0) throw new ArgumentException($"Conv1d input of {x.T} steps is too short.");
        int tp = x.T + 2 * Padding;
        float[] xp;
        if (Padding == 0) xp = x.Data;
        else
        {
            xp = new float[(long)tp * Cin];
            x.Data.AsSpan().CopyTo(xp.AsSpan(Padding * Cin));
        }
        var y = new Act(tout, Cout);
        fixed (float* a = xp, c = y.Data, b = _bias)
        {
            if (_full != null)
            {
                Gemm.Multiply(a, (long)Stride * Cin, tout, _full, c, Cout, par, b);
            }
            else
            {
                for (int t = 0; t < K; t++)
                    Gemm.Multiply(a + (long)t * Dilation * Cin, Cin, tout, _taps![t], c, Cout, par, t == 0 ? b : null, accumulate: t > 0);
            }
        }
        return y;
    }

    public void Dispose()
    {
        _full?.Dispose();
        if (_taps != null) foreach (var t in _taps) t.Dispose();
    }
}

/// <summary><c>nn.ConvTranspose1d</c> (groups = 1): one GEMM computes every tap's contribution of every input step
/// (<c>Z[t, k·Cout + o] = Σ_i x[t, i] · w[i, o, k]</c>), then an overlap-add places them at
/// <c>t · stride − padding + k</c>.</summary>
internal sealed unsafe class ConvTranspose1d : IDisposable
{
    private readonly PackedMatrix _w;
    private readonly float[]? _bias;

    public ConvTranspose1d(float[] w, float[]? bias, int cin, int cout, int k, int stride, int padding, int outputPadding = 0)
    {
        if (w.Length != (long)cin * cout * k) throw new ArgumentException("ConvTranspose1d weight size mismatch.");
        Cin = cin; Cout = cout; K = k; Stride = stride; Padding = padding; OutputPadding = outputPadding;
        _bias = bias;
        // [Cin, Cout, K] -> B[Cin, K * Cout]
        var b = new float[w.Length];
        for (int i = 0; i < cin; i++)
            for (int o = 0; o < cout; o++)
                for (int t = 0; t < k; t++) b[(long)i * k * cout + t * cout + o] = w[((long)i * cout + o) * k + t];
        fixed (float* p = b) _w = PackedMatrix.FromKN(p, cin, k * cout, hugePages: false);
    }

    public static ConvTranspose1d Load(WeightReader r, string module, int stride, int padding, int outputPadding = 0)
    {
        var shape = r.Shape(module + ".weight");
        return new ConvTranspose1d(r.Get(module + ".weight"), r.TryGet(module + ".bias"), shape[0], shape[1], shape[2], stride, padding, outputPadding);
    }

    public int Cin { get; }
    public int Cout { get; }
    public int K { get; }
    public int Stride { get; }
    public int Padding { get; }
    public int OutputPadding { get; }

    public Act Forward(Act x, Parallelism par)
    {
        if (x.C != Cin) throw new ArgumentException($"ConvTranspose1d expects {Cin} channels, got {x.C}.");
        int tout = (x.T - 1) * Stride - 2 * Padding + K + OutputPadding;
        int n = K * Cout;
        var z = new float[(long)x.T * n];
        fixed (float* a = x.Data, c = z) Gemm.Multiply(a, Cin, x.T, _w, c, n, par);
        var y = new Act(tout, Cout);
        var yd = y.Data;
        // Gather form (each output row sums its contributing taps in k order), parallel over output rows.
        par.ForRange(tout, 64, (o0, o1) =>
        {
            for (int to = o0; to < o1; to++)
            {
                var row = yd.AsSpan(to * Cout, Cout);
                if (_bias != null) _bias.AsSpan().CopyTo(row);
                int p = to + Padding;                       // = t * stride + k
                int kFirst = p % Stride;
                for (int k = kFirst; k < K; k += Stride)
                {
                    int t = (p - k) / Stride;
                    if (t < 0) break;
                    if (t >= x.T) continue;
                    TensorPrimitives.Add(row, z.AsSpan((int)((long)t * n + k * Cout), Cout), row);
                }
            }
        });
        return y;
    }

    public void Dispose() => _w.Dispose();
}

/// <summary>The depthwise <c>nn.ConvTranspose1d(C, C, 3, stride=2, groups=C, padding=1, output_padding=1)</c> that
/// upsamples ×2 inside <c>AdainResBlk1d</c>.</summary>
internal sealed class DepthwiseUpsample(float[] w, float[]? bias, int channels)
{
    public static DepthwiseUpsample Load(WeightReader r, string module)
    {
        var shape = r.Shape(module + ".weight");   // [C, 1, 3]
        if (shape[1] != 1 || shape[2] != 3) throw new InvalidDataException($"{module}: unexpected shape [{string.Join(", ", shape)}].");
        return new DepthwiseUpsample(r.Get(module + ".weight"), r.TryGet(module + ".bias"), shape[0]);
    }

    public Act Forward(Act x)
    {
        int c = channels;
        var y = new Act(2 * x.T, c);
        for (int to = 0; to < y.T; to++)
        {
            // to = 2t - 1 + k: even outputs take k = 1 (t = to/2), odd outputs k = 0 (t = (to+1)/2) and k = 2 (t = (to-1)/2)
            var row = y.Row(to);
            for (int ch = 0; ch < c; ch++)
            {
                float s = bias?[ch] ?? 0f;
                if ((to & 1) == 0) s += x[to / 2, ch] * w[ch * 3 + 1];
                else
                {
                    int t0 = (to + 1) / 2, t2 = (to - 1) / 2;
                    if (t0 < x.T) s += x[t0, ch] * w[ch * 3 + 0];
                    s += x[t2, ch] * w[ch * 3 + 2];
                }
                row[ch] = s;
            }
        }
        return y;
    }
}

/// <summary>A single-layer bidirectional <c>nn.LSTM</c> (batch_first). The input projection of both directions is one
/// GEMM; the recurrences run concurrently, one per direction, as float dot products.</summary>
internal sealed unsafe class BiLstm : IDisposable
{
    private readonly PackedMatrix _wih;           // [I, 8H]: forward gates then reverse gates
    private readonly float[] _bias;               // b_ih + b_hh, both directions
    private readonly float[][] _whh = new float[2][];   // [4H, H] per direction

    public BiLstm(WeightReader r, string module)
    {
        var shape = r.Shape(module + ".weight_ih_l0");
        H = shape[0] / 4;
        Input = shape[1];
        var wih = new float[(long)8 * H * Input];
        _bias = new float[8 * H];
        for (int d = 0; d < 2; d++)
        {
            string s = d == 0 ? "l0" : "l0_reverse";
            r.Get($"{module}.weight_ih_{s}", 4 * H, Input).CopyTo(wih.AsSpan(d * 4 * H * Input));
            _whh[d] = r.Get($"{module}.weight_hh_{s}", 4 * H, H);
            var bih = r.Get($"{module}.bias_ih_{s}", 4 * H);
            var bhh = r.Get($"{module}.bias_hh_{s}", 4 * H);
            for (int i = 0; i < 4 * H; i++) _bias[d * 4 * H + i] = bih[i] + bhh[i];
        }
        fixed (float* p = wih) _wih = PackedMatrix.FromLinear(p, 8 * H, Input, hugePages: false);
    }

    public int H { get; }
    public int Input { get; }

    /// <summary>[T, I] → [T, 2H] (forward hidden states, then reverse).</summary>
    public Act Forward(Act x, Parallelism par)
    {
        if (x.C != Input) throw new ArgumentException($"LSTM expects {Input} features, got {x.C}.");
        int T = x.T, G = 8 * H;
        var gates = new float[(long)T * G];
        fixed (float* a = x.Data, c = gates, b = _bias) Gemm.Multiply(a, Input, T, _wih, c, G, par, b);
        var y = new Act(T, 2 * H);
        if (par.MaxThreads > 1 && T > 8)
            Parallel.Invoke(() => Recur(gates, y, 0), () => Recur(gates, y, 1));
        else
        {
            Recur(gates, y, 0);
            Recur(gates, y, 1);
        }
        if (DebugDump.Enabled) DebugDump.Write(DebugDump.Seq("lstm"), y);
        return y;
    }

    private void Recur(float[] gates, Act y, int dir)
    {
        int T = y.T, H4 = 4 * H, G = 8 * H;
        var whh = _whh[dir];
        var h = new float[H];
        var c = new float[H];
        var g = new float[H4];
        for (int step = 0; step < T; step++)
        {
            int t = dir == 0 ? step : T - 1 - step;
            var gin = gates.AsSpan(t * G + dir * H4, H4);
            for (int j = 0; j < H4; j++) g[j] = gin[j] + TensorPrimitives.Dot(whh.AsSpan(j * H, H), h);
            var outRow = y.Data.AsSpan(t * 2 * H + dir * H, H);
            for (int j = 0; j < H; j++)
            {
                float ig = Sigmoid(g[j]);
                float fg = Sigmoid(g[H + j]);
                float gg = MathF.Tanh(g[2 * H + j]);
                float og = Sigmoid(g[3 * H + j]);
                c[j] = fg * c[j] + ig * gg;
                h[j] = og * MathF.Tanh(c[j]);
                outRow[j] = h[j];
            }
        }
    }

    private static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));

    public void Dispose() => _wih.Dispose();
}
