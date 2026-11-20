using KittenTTS.Numerics;

namespace KittenTTS.Model;

/// <summary><c>AdaIN1d</c>: instance norm whose scale and shift come from the style vector (<c>fc(s) → γ, β</c>).</summary>
internal sealed class AdaIn1d
{
    private readonly VectorLinear _fc;
    private readonly float[]? _normWeight, _normBias;

    public AdaIn1d(WeightReader r, string module)
    {
        _fc = VectorLinear.Load(r, module + ".fc");
        _normWeight = r.TryGet(module + ".norm.weight");
        _normBias = r.TryGet(module + ".norm.bias");
        Channels = _fc.Out / 2;
    }

    public int Channels { get; }

    /// <summary>γ and β for a style vector (constant over the utterance, computed once per call).</summary>
    public float[] Style(ReadOnlySpan<float> s) => _fc.Forward(s);

    public Act Forward(Act x, float[] gammaBeta, Parallelism par, float[]? snakeAlpha = null) =>
        F.AdaIn(x, gammaBeta, _normWeight, _normBias, par, snakeAlpha);
}

/// <summary>
/// <c>AdainResBlk1d</c> (prosody predictor and decoder):
/// <c>(conv2(lrelu(norm2(conv1(pool(lrelu(norm1(x))))))) + conv1x1(upsample(x))) / √2</c>, leaky slope 0.2,
/// where pool / upsample double the length in the upsampling blocks.
/// </summary>
internal sealed class AdainResBlk1d : IDisposable
{
    private readonly AdaIn1d _norm1, _norm2;
    private readonly Conv1d _conv1, _conv2;
    private readonly Conv1d? _conv1x1;
    private readonly DepthwiseUpsample? _pool;
    private readonly bool _upsample;

    public AdainResBlk1d(WeightReader r, string module, bool upsample)
    {
        _norm1 = new AdaIn1d(r, module + ".norm1");
        _norm2 = new AdaIn1d(r, module + ".norm2");
        _conv1 = Conv1d.Load(r, module + ".conv1", padding: 1);
        _conv2 = Conv1d.Load(r, module + ".conv2", padding: 1);
        if (r.Has(module + ".conv1x1.weight")) _conv1x1 = Conv1d.Load(r, module + ".conv1x1");
        _upsample = upsample;
        if (upsample) _pool = DepthwiseUpsample.Load(r, module + ".pool");
        In = _conv1.Cin;
        Out = _conv2.Cout;
    }

    public int In { get; }
    public int Out { get; }

    public Act Forward(Act x, ReadOnlySpan<float> s, Parallelism par)
    {
        var h = _norm1.Forward(x, _norm1.Style(s), par);
        F.LeakyRelu(h, 0.2f, inPlace: true);
        if (_pool != null) h = _pool.Forward(h);
        h = _conv1.Forward(h, par);
        h = _norm2.Forward(h, _norm2.Style(s), par);
        F.LeakyRelu(h, 0.2f, inPlace: true);
        h = _conv2.Forward(h, par);

        var sc = _upsample ? F.Upsample2(x) : x;
        if (_conv1x1 != null) sc = _conv1x1.Forward(sc, par);
        F.AddInPlace(h, sc);
        F.ScaleInPlace(h, 1f / MathF.Sqrt(2f));
        return h;
    }

    public void Dispose()
    {
        _conv1.Dispose();
        _conv2.Dispose();
        _conv1x1?.Dispose();
    }
}

/// <summary>
/// <c>AdaINResBlock1</c> of the iSTFTNet generator: three residual units
/// <c>x += conv2(snake₂(adain2(conv1_d(snake₁(adain1(x))))))</c> with dilations 1, 3, 5 in conv1.
/// </summary>
internal sealed class AdaInResBlock1 : IDisposable
{
    private readonly Conv1d[] _convs1 = new Conv1d[3], _convs2 = new Conv1d[3];
    private readonly AdaIn1d[] _adain1 = new AdaIn1d[3], _adain2 = new AdaIn1d[3];
    private readonly float[][] _alpha1 = new float[3][], _alpha2 = new float[3][];

    public AdaInResBlock1(WeightReader r, string module, int kernel, int[] dilations)
    {
        for (int j = 0; j < 3; j++)
        {
            int d = dilations[j];
            _convs1[j] = Conv1d.Load(r, $"{module}.convs1.{j}", padding: (kernel * d - d) / 2, dilation: d);
            _convs2[j] = Conv1d.Load(r, $"{module}.convs2.{j}", padding: (kernel - 1) / 2);
            _adain1[j] = new AdaIn1d(r, $"{module}.adain1.{j}");
            _adain2[j] = new AdaIn1d(r, $"{module}.adain2.{j}");
            _alpha1[j] = r.Get($"{module}.alpha1.{j}");
            _alpha2[j] = r.Get($"{module}.alpha2.{j}");
        }
    }

    public Act Forward(Act x, ReadOnlySpan<float> s, Parallelism par)
    {
        for (int j = 0; j < 3; j++)
        {
            // AdaIN and snake fused into one pass
            var xt = _adain1[j].Forward(x, _adain1[j].Style(s), par, _alpha1[j]);
            xt = _convs1[j].Forward(xt, par);
            xt = _adain2[j].Forward(xt, _adain2[j].Style(s), par, _alpha2[j]);
            xt = _convs2[j].Forward(xt, par);
            F.AddInPlace(xt, x);
            x = xt;
        }
        return x;
    }

    public void Dispose()
    {
        foreach (var c in _convs1) c.Dispose();
        foreach (var c in _convs2) c.Dispose();
    }
}
