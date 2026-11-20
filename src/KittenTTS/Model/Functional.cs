using System.Numerics.Tensors;

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
    /// decoders). Folded into one scale and shift per channel.
    /// </summary>
    public static Act AdaIn(Act x, float[] gammaBeta, float[]? normWeight, float[]? normBias, float eps = 1e-5f)
    {
        int c = x.C, T = x.T;
        var mean = new double[c];
        var sq = new double[c];
        for (int t = 0; t < T; t++)
        {
            var row = x.Row(t);
            for (int i = 0; i < c; i++) mean[i] += row[i];
        }
        for (int i = 0; i < c; i++) mean[i] /= T;
        for (int t = 0; t < T; t++)
        {
            var row = x.Row(t);
            for (int i = 0; i < c; i++) { double d = row[i] - mean[i]; sq[i] += d * d; }
        }
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
        var y = new Act(T, c);
        for (int t = 0; t < T; t++)
        {
            var src = x.Row(t);
            var dst = y.Row(t);
            TensorPrimitives.MultiplyAdd(src, scale, shift, dst);
        }
        return y;
    }

    public static Act LeakyRelu(Act x, float slope, bool inPlace = false)
    {
        var y = inPlace ? x : new Act(x.T, x.C);
        var s = x.Data;
        var d = y.Data;
        for (long i = 0; i < s.LongLength; i++) { float v = s[i]; d[i] = v >= 0 ? v : v * slope; }
        return y;
    }

    /// <summary>Snake: <c>x + (1 / α) · sin(α · x)²</c> with one α per channel.</summary>
    public static Act Snake(Act x, float[] alpha)
    {
        int c = x.C;
        var y = new Act(x.T, c);
        var inv = new float[c];
        for (int i = 0; i < c; i++) inv[i] = 1f / alpha[i];
        var tmp = new float[c];
        for (int t = 0; t < x.T; t++)
        {
            var src = x.Row(t);
            var dst = y.Row(t);
            TensorPrimitives.Multiply(src, alpha, tmp);
            TensorPrimitives.Sin(tmp, tmp);
            TensorPrimitives.Multiply(tmp, tmp, tmp);
            TensorPrimitives.MultiplyAdd(tmp, inv, src, dst);
        }
        return y;
    }

    /// <summary>Nearest-neighbour ×2 upsampling over time (<c>F.interpolate(scale_factor=2, mode='nearest')</c>).</summary>
    public static Act Upsample2(Act x)
    {
        var y = new Act(2 * x.T, x.C);
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
