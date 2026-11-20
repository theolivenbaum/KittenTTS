using KittenTTS.Audio;
using KittenTTS.Model;
using KittenTTS.Numerics;

namespace KittenTTS.Tests;

/// <summary>The numerical building blocks against straightforward double-precision references (no model needed, so
/// these always run, also in CI). Run with <c>DOTNET_EnableAVX512F=0</c> / <c>DOTNET_EnableHWIntrinsic=0</c> to check
/// the AVX2 and scalar paths.</summary>
public sealed unsafe class KernelTests
{
    private static float[] Random(int n, int seed, float scale = 1f)
    {
        var r = new Random(seed);
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(r.NextDouble() * 2 - 1) * scale;
        return a;
    }

    private static void AssertClose(double[] expected, float[] actual, double tol)
    {
        Assert.Equal(expected.Length, actual.Length);
        double maxErr = 0, maxRef = 1e-30;
        for (int i = 0; i < expected.Length; i++)
        {
            maxErr = Math.Max(maxErr, Math.Abs(expected[i] - actual[i]));
            maxRef = Math.Max(maxRef, Math.Abs(expected[i]));
        }
        Assert.True(maxErr <= tol * maxRef, $"max error {maxErr:E2} (reference max {maxRef:E2})");
    }

    [Theory]
    [InlineData(1, 1, 1)]
    [InlineData(3, 17, 5)]
    [InlineData(13, 33, 257)]
    [InlineData(137, 96, 1408)]
    [InlineData(500, 22, 48)]
    public void GemmMatchesReference(int m, int n, int k)
    {
        foreach (var threads in new[] { 1, 4 })
        {
            var a = Random(m * k, 1);
            var w = Random(n * k, 2);
            var bias = Random(n, 3);
            var c = new float[m * n];
            fixed (float* wp = w, ap = a, cp = c, bp = bias)
            {
                using var packed = PackedMatrix.FromLinear(wp, n, k);
                using var par = new Parallelism(threads);
                Gemm.Multiply(ap, k, m, packed, cp, n, par, bp);
            }
            var expected = new double[m * n];
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                {
                    double s = bias[j];
                    for (int p = 0; p < k; p++) s += (double)a[i * k + p] * w[j * k + p];
                    expected[i * n + j] = s;
                }
            AssertClose(expected, c, 1e-5);
        }
    }

    /// <summary>nn.Conv1d semantics: y[o, t] = b[o] + Σ_i Σ_k w[o, i, k] · x[i, t·s + k·d − p].</summary>
    [Theory]
    [InlineData(7, 5, 3, 1, 1, 1, 50)]      // conv1 / conv2 of the residual blocks
    [InlineData(6, 6, 7, 1, 9, 3, 80)]      // dilated
    [InlineData(4, 3, 11, 1, 25, 5, 120)]   // dilation 5
    [InlineData(22, 9, 12, 6, 3, 1, 61)]    // the strided noise conv
    [InlineData(1, 1, 3, 2, 1, 1, 20)]      // F0_conv
    public void Conv1dMatchesReference(int cin, int cout, int k, int stride, int pad, int dil, int T)
    {
        var w = Random(cout * cin * k, 4);
        var b = Random(cout, 5);
        var x = Random(T * cin, 6);         // channels-last [T, Cin]
        using var conv = new Conv1d(w, b, cout, cin, k, stride, pad, dil);
        using var par = new Parallelism(4);
        var y = conv.Forward(new Act(T, cin, x), par);
        int tout = (T + 2 * pad - dil * (k - 1) - 1) / stride + 1;
        Assert.Equal(tout, y.T);
        var expected = new double[tout * cout];
        for (int t = 0; t < tout; t++)
            for (int o = 0; o < cout; o++)
            {
                double s = b[o];
                for (int i = 0; i < cin; i++)
                    for (int kk = 0; kk < k; kk++)
                    {
                        int src = t * stride + kk * dil - pad;
                        if (src >= 0 && src < T) s += (double)w[(o * cin + i) * k + kk] * x[src * cin + i];
                    }
                expected[t * cout + o] = s;
            }
        AssertClose(expected, y.Data, 1e-5);
    }

    /// <summary>nn.ConvTranspose1d semantics: y[o, t·s − p + k] += w[i, o, k] · x[i, t].</summary>
    [Theory]
    [InlineData(8, 4, 20, 10, 5, 30)]       // ups.0
    [InlineData(6, 3, 12, 6, 3, 21)]        // ups.1
    public void ConvTranspose1dMatchesReference(int cin, int cout, int k, int stride, int pad, int T)
    {
        var w = Random(cin * cout * k, 7);
        var b = Random(cout, 8);
        var x = Random(T * cin, 9);
        using var conv = new ConvTranspose1d(w, b, cin, cout, k, stride, pad);
        using var par = new Parallelism(4);
        var y = conv.Forward(new Act(T, cin, x), par);
        int tout = (T - 1) * stride - 2 * pad + k;
        var expected = new double[tout * cout];
        for (int t = 0; t < tout; t++) for (int o = 0; o < cout; o++) expected[t * cout + o] = b[o];
        for (int t = 0; t < T; t++)
            for (int i = 0; i < cin; i++)
                for (int o = 0; o < cout; o++)
                    for (int kk = 0; kk < k; kk++)
                    {
                        int dst = t * stride - pad + kk;
                        if (dst >= 0 && dst < tout) expected[dst * cout + o] += (double)w[(i * cout + o) * k + kk] * x[t * cin + i];
                    }
        AssertClose(expected, y.Data, 1e-5);
    }

    [Fact]
    public void DepthwiseUpsampleMatchesReference()
    {
        int c = 5, T = 9;
        var w = Random(c * 3, 10);
        var b = Random(c, 11);
        var x = Random(T * c, 12);
        var y = new DepthwiseUpsample(w, b, c).Forward(new Act(T, c, x));
        var expected = new double[2 * T * c];
        for (int t = 0; t < 2 * T; t++) for (int ch = 0; ch < c; ch++) expected[t * c + ch] = b[ch];
        for (int t = 0; t < T; t++)
            for (int ch = 0; ch < c; ch++)
                for (int k = 0; k < 3; k++)
                {
                    int dst = 2 * t - 1 + k;     // stride 2, padding 1, output_padding 1
                    if (dst >= 0 && dst < 2 * T) expected[dst * c + ch] += (double)w[ch * 3 + k] * x[t * c + ch];
                }
        AssertClose(expected, y.Data, 1e-6);
    }

    [Fact]
    public void StftRoundTrips()
    {
        // torch.istft(torch.stft(x)) reconstructs x for the generator's Hann / hop-5 transform
        var x = Random(600, 13, 0.5f);
        var stft = new Stft(20, 5);
        var (mag, phase, frames) = stft.Forward(x);
        Assert.Equal(600 / 5 + 1, frames);
        var y = stft.Inverse(mag, phase, frames);
        Assert.Equal(x.Length, y.Length);
        AssertClose(x.Select(v => (double)v).ToArray(), y, 1e-5);
    }

    [Fact]
    public void AdaInSnakeMatchesReference()
    {
        int T = 300, c = 7;
        var x = Random(T * c, 14);
        var gb = Random(2 * c, 15, 0.3f);
        var alpha = Random(c, 16).Select(v => 0.5f + Math.Abs(v)).ToArray();
        using var par = new Parallelism(4);
        var y = F.AdaIn(new Act(T, c, x), gb, null, null, par, alpha);
        var expected = new double[T * c];
        for (int ch = 0; ch < c; ch++)
        {
            double mean = 0, var = 0;
            for (int t = 0; t < T; t++) mean += x[t * c + ch];
            mean /= T;
            for (int t = 0; t < T; t++) var += Math.Pow(x[t * c + ch] - mean, 2);
            var /= T;
            for (int t = 0; t < T; t++)
            {
                double v = (1 + gb[ch]) * (x[t * c + ch] - mean) / Math.Sqrt(var + 1e-5) + gb[c + ch];
                expected[t * c + ch] = v + Math.Pow(Math.Sin(alpha[ch] * v), 2) / alpha[ch];
            }
        }
        AssertClose(expected, y.Data, 1e-5);
    }
}
