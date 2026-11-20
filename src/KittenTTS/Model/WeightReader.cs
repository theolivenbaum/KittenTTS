using KittenTTS.IO;

namespace KittenTTS.Model;

/// <summary>Reads module parameters by PyTorch name from a checkpoint, folding weight normalization
/// (<c>weight = g · v / ‖v‖</c>, the norm taken over every dimension but the first, as <c>torch.nn.utils.weight_norm</c>
/// with <c>dim=0</c>) when the checkpoint stores <c>weight_g</c> / <c>weight_v</c> instead of <c>weight</c>.</summary>
internal sealed class WeightReader(ITensorSource source)
{
    public ITensorSource Source { get; } = source;

    public bool Has(string name) => Source.Contains(name) || (name.EndsWith(".weight", StringComparison.Ordinal)
        && Source.Contains(name + "_g") && Source.Contains(name + "_v"));

    public Tensor Tensor(string name)
    {
        if (Source.Contains(name)) return Source.Get(name);
        if (name.EndsWith(".weight", StringComparison.Ordinal) && Source.Contains(name + "_g") && Source.Contains(name + "_v"))
            return FoldWeightNorm(Source.Get(name + "_g"), Source.Get(name + "_v"));
        throw new KeyNotFoundException($"The checkpoint has no tensor '{name}'.");
    }

    public float[] Get(string name, params int[] expectedShape)
    {
        var t = Tensor(name);
        if (expectedShape.Length > 0 && !t.Shape.SequenceEqual(expectedShape))
            throw new InvalidDataException($"{name}: shape {t.ShapeString}, expected [{string.Join(", ", expectedShape)}].");
        return t.ToArray();
    }

    public float[]? TryGet(string name) => Has(name) ? Tensor(name).ToArray() : null;

    public int[] Shape(string name) => Tensor(name).Shape;

    private static Tensor FoldWeightNorm(Tensor g, Tensor v)
    {
        int rows = v.Shape[0];
        long per = v.Length / rows;
        var vs = v.Span;
        var gs = g.Span;
        var w = new float[v.Length];
        for (int r = 0; r < rows; r++)
        {
            double ss = 0;
            for (long i = 0; i < per; i++) { double x = vs[(int)(r * per + i)]; ss += x * x; }
            // PyTorch computes the norm and the scale in float32: w = v * (g / ||v||)
            float norm = (float)Math.Sqrt(ss);
            float scale = gs[r] / norm;
            for (long i = 0; i < per; i++) w[r * per + i] = vs[(int)(r * per + i)] * scale;
        }
        return new Tensor(v.Shape, w);
    }
}
