using System.Numerics.Tensors;
using System.Runtime.Intrinsics;
using KittenTTS.Numerics;

namespace KittenTTS.Model;

/// <summary>Element-wise and normalization ops over channels-last activations. Statistics are accumulated in double,
/// so they agree with PyTorch's float32 reductions to within float rounding.</summary>
internal static class F
{
    /// <summary><c>F.layer_norm</c> over the channel axis of every step, with optional affine parameters.</summary>
    public static Act LayerNorm(Act x, float[]? gamma, float[]? beta, float eps)
    {
        var y = new Act(x.T, x.C);
        int c = x.C;
        for (int t = 0; t < x.T; t++)
        {
            var src = x.Row(t);
            var dst = y.Row(t);
            double s = 0;
            for (int i = 0; i < c; i++) s += src[i];
            double mean = s / c;
            double q = 0;
            for (int i = 0; i < c; i++) { double d = src[i] - mean; q += d * d; }
            float inv = (float)(1.0 / Math.Sqrt(q / c + eps));
            float m = (float)mean;
            for (int i = 0; i < c; i++)
            {
                float v = (src[i] - m) * inv;
                if (gamma != null) v *= gamma[i];
                if (beta != null) v += beta[i];
                dst[i] = v;
            }
        }
        return y;
    }

    /// <summary>
    /// AdaIN: <c>(1 + γ) · InstanceNorm1d(x) + β</c>, the instance norm over time per channel (biased variance,
    /// eps 1e-5), with its own affine parameters when the module has them (<c>affine=True</c> in the nano / pico
    /// decoders), folded into one scale and shift per channel; optionally followed by <see cref="Snake"/> in the same
    /// pass. The statistics are two-pass and double-accumulated (parallel over row blocks); the apply step runs on
    /// blocks of rows against tiled per-channel vectors, so every vector op spans many rows.
    /// </summary>
    public static Act AdaIn(Act x, float[] gammaBeta, float[]? normWeight, float[]? normBias, Parallelism par,
                            float[]? snakeAlpha = null, float eps = 1e-5f)
    {
        int c = x.C, T = x.T;
        var mean = ColumnSums(x, null, par);
        for (int i = 0; i < c; i++) mean[i] /= T;
        var sq = ColumnSums(x, mean, par);
        var scale = new float[c];
        var shift = new float[c];
        for (int i = 0; i < c; i++)
        {
            float inv = (float)(1.0 / Math.Sqrt(sq[i] / T + eps));
            float a = inv, b = (float)-mean[i] * inv;            // norm(x) = a x + b
            if (normWeight != null) { a *= normWeight[i]; b *= normWeight[i]; }
            if (normBias != null) b += normBias[i];
            float g = 1f + gammaBeta[i], be = gammaBeta[c + i];
            scale[i] = g * a;
            shift[i] = g * b + be;
        }
        var y = Act.Uninit(T, c);
        ApplyBlocks(x, y, scale, shift, snakeAlpha, par);
        return y;
    }

    private const int Block = 32;

    private static float[] Tile(float[] v, int rows)
    {
        var t = new float[v.Length * rows];
        for (int r = 0; r < rows; r++) v.CopyTo(t, r * v.Length);
        return t;
    }

    /// <summary>y = x · scale + shift, then snake when <paramref name="alpha"/> is given.</summary>
    private static void ApplyBlocks(Act x, Act y, float[] scale, float[] shift, float[]? alpha, Parallelism par)
    {
        int c = x.C, T = x.T;
        var sc = Tile(scale, Block);
        var sh = Tile(shift, Block);
        float[]? al = null, inv = null;
        if (alpha != null)
        {
            al = Tile(alpha, Block);
            inv = new float[al.Length];
            for (int i = 0; i < al.Length; i++) inv[i] = 1f / al[i];
        }
        int blocks = (T + Block - 1) / Block;
        par.ForRange(blocks, 8, (b0, b1) =>
        {
            var tmp = alpha != null ? new float[Block * c] : null;
            for (int b = b0; b < b1; b++)
            {
                int r0 = b * Block, n = Math.Min(Block, T - r0) * c;
                var src = x.Data.AsSpan(r0 * c, n);
                var dst = y.Data.AsSpan(r0 * c, n);
                TensorPrimitives.MultiplyAdd(src, sc.AsSpan(0, n), sh.AsSpan(0, n), dst);
                if (tmp != null) SnakeSpan(dst, al!.AsSpan(0, n), inv!.AsSpan(0, n), tmp.AsSpan(0, n));
            }
        });
    }

    /// <summary>In place: v + (1 / α) · sin(α · v)², the same operation order as the PyTorch expression.</summary>
    private static void SnakeSpan(Span<float> v, ReadOnlySpan<float> alpha, ReadOnlySpan<float> inv, Span<float> tmp)
    {
        TensorPrimitives.Multiply(v, alpha, tmp);
        TensorPrimitives.Sin(tmp, tmp);
        TensorPrimitives.Multiply(tmp, tmp, tmp);
        TensorPrimitives.MultiplyAdd(tmp, inv, v, v);
    }

    /// <summary>Per-channel Σ x (or Σ (x − mean)² when <paramref name="mean"/> is given), in double.</summary>
    private static unsafe double[] ColumnSums(Act x, double[]? mean, Parallelism par)
    {
        int c = x.C, T = x.T;
        int chunks = Math.Max(1, Math.Min(par.MaxThreads, T / 256));
        var partial = new double[chunks][];
        int per = (T + chunks - 1) / chunks;
        par.For(chunks, k =>
        {
            var acc = new double[c];
            int t0 = k * per, t1 = Math.Min(T, t0 + per);
            fixed (float* xp = x.Data)
            fixed (double* ap = acc)
            fixed (double* mp = mean)
            {
                for (int t = t0; t < t1; t++)
                {
                    float* row = xp + (long)t * c;
                    int i = 0;
                    if (Vector256.IsHardwareAccelerated)
                    {
                        for (; i + 8 <= c; i += 8)
                        {
                            var (lo, hi) = Vector256.Widen(Vector256.Load(row + i));
                            if (mp != null)
                            {
                                lo -= Vector256.Load(mp + i);
                                hi -= Vector256.Load(mp + i + 4);
                                lo *= lo;
                                hi *= hi;
                            }
                            Vector256.Store(Vector256.Load(ap + i) + lo, ap + i);
                            Vector256.Store(Vector256.Load(ap + i + 4) + hi, ap + i + 4);
                        }
                    }
                    for (; i < c; i++)
                    {
                        double v = row[i];
                        if (mp != null) { v -= mp[i]; v *= v; }
                        ap[i] += v;
                    }
                }
            }
            partial[k] = acc;
        });
        var sum = partial[0];
        for (int k = 1; k < chunks; k++)
            for (int i = 0; i < c; i++) sum[i] += partial[k][i];
        return sum;
    }

    public static Act LeakyRelu(Act x, float slope, bool inPlace = false)
    {
        var y = inPlace ? x : Act.Uninit(x.T, x.C);
        var s = x.Data;
        var d = y.Data;
        for (long i = 0; i < s.LongLength; i++) { float v = s[i]; d[i] = v >= 0 ? v : v * slope; }
        return y;
    }

    /// <summary>Snake: <c>x + (1 / α) · sin(α · x)²</c> with one α per channel.</summary>
    public static Act Snake(Act x, float[] alpha, Parallelism par)
    {
        var y = Act.Uninit(x.T, x.C);
        var ones = Enumerable.Repeat(1f, x.C).ToArray();
        ApplyBlocks(x, y, ones, new float[x.C], alpha, par);
        return y;
    }

    /// <summary>Nearest-neighbour ×2 upsampling over time (<c>F.interpolate(scale_factor=2, mode='nearest')</c>).</summary>
    public static Act Upsample2(Act x)
    {
        var y = Act.Uninit(2 * x.T, x.C);
        for (int t = 0; t < y.T; t++) x.Row(t / 2).CopyTo(y.Row(t));
        return y;
    }

    /// <summary><c>nn.ReflectionPad1d((1, 0))</c>: prepends x[1].</summary>
    public static Act ReflectPadLeft1(Act x)
    {
        var y = new Act(x.T + 1, x.C);
        x.Row(1).CopyTo(y.Row(0));
        x.Data.AsSpan().CopyTo(y.Data.AsSpan(x.C));
        return y;
    }

    public static void AddInPlace(Act acc, Act x)
    {
        if (acc.T != x.T || acc.C != x.C) throw new ArgumentException($"Shape mismatch: [{acc.T}, {acc.C}] + [{x.T}, {x.C}].");
        TensorPrimitives.Add(acc.Data, x.Data, acc.Data);
    }

    public static void ScaleInPlace(Act x, float s) => TensorPrimitives.Multiply(x.Data, s, x.Data);

    /// <summary>Repeats row <c>i</c> of <paramref name="x"/> <c>durations[i]</c> times (the token → frame alignment).</summary>
    public static Act Expand(Act x, int[] durations)
    {
        int frames = durations.Sum();
        var y = new Act(frames, x.C);
        int f = 0;
        for (int i = 0; i < x.T; i++)
            for (int r = 0; r < durations[i]; r++) x.Row(i).CopyTo(y.Row(f++));
        return y;
    }

    /// <summary><c>F.softplus</c> (threshold 20).</summary>
    public static float Softplus(float x) => x > 20f ? x : (float)Math.Log(1.0 + Math.Exp(x));
}
