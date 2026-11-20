using System.Text.Json;
using KittenTTS.IO;
using Xunit.Abstractions;

namespace KittenTTS.Tests;

/// <summary>
/// Locates a model repository and the reference dumps written by <c>scripts/torch_reference.py</c>:
/// <list type="bullet">
/// <item><c>KITTENTTS_MODEL_DIR</c>: a local KittenTTS 2 nano / pico / mini repository (config.json, vocab.json, the
/// language's safetensors and int8 ONNX graph, voices_*.npz, g2p/)</item>
/// <item><c>KITTENTTS_REFERENCE_DIR</c>: the reference dump of that repository (manifest.json + one directory per case)</item>
/// </list>
/// Tests that need them print <c>SKIPPED</c> and pass when they are missing, so the kernel tests run anywhere.
/// </summary>
public static class Fixtures
{
    public static string? ModelDir => Existing(Environment.GetEnvironmentVariable("KITTENTTS_MODEL_DIR"));
    public static string? ReferenceDir => Existing(Environment.GetEnvironmentVariable("KITTENTTS_REFERENCE_DIR"));

    private static string? Existing(string? p) => !string.IsNullOrEmpty(p) && Directory.Exists(p) ? p : null;

    public static bool Available(ITestOutputHelper output, bool reference = true)
    {
        var missing = new List<string>();
        if (ModelDir == null) missing.Add("KITTENTTS_MODEL_DIR");
        if (reference && ReferenceDir == null) missing.Add("KITTENTTS_REFERENCE_DIR");
        if (missing.Count == 0) return true;
        output.WriteLine($"SKIPPED: set {string.Join(", ", missing)} (see README, 'Validation').");
        return false;
    }

    public sealed record Case(string Name, string Text, string Phonemes, int[] Tokens);
    public sealed record Manifest(string Model, string Language, string Voice, float Speed, List<Case> Cases, string? Weights = null)
    {
        /// <summary>The weights the reference ran (fp32, or int8: PyTorch with the int8 graph's dequantized weights).</summary>
        public WeightFormat Format => Weights == "int8" ? WeightFormat.Int8 : WeightFormat.Float32;
    }

    public static Manifest LoadManifest() =>
        JsonSerializer.Deserialize<Manifest>(File.ReadAllText(Path.Combine(ReferenceDir!, "manifest.json")),
                                             new JsonSerializerOptions { PropertyNameCaseInsensitive = true })!;

    public static NpyArray Ref(string caseName, string stage) => Npy.Read(Path.Combine(ReferenceDir!, caseName, stage + ".npy"));

    public static bool HasRef(string caseName, string stage) => File.Exists(Path.Combine(ReferenceDir!, caseName, stage + ".npy"));

    public static (double RelL2, double MaxAbs) Error(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual)
    {
        double num = 0, den = 0, mx = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            double d = (double)actual[i] - expected[i];
            num += d * d;
            den += (double)expected[i] * expected[i];
            mx = Math.Max(mx, Math.Abs(d));
        }
        return (Math.Sqrt(num / Math.Max(den, 1e-30)), mx);
    }

    /// <summary>Asserts closeness and reports max-abs / relative-L2 error.</summary>
    public static (double RelL2, double MaxAbs) AssertClose(ITestOutputHelper output, string label, ReadOnlySpan<float> expected,
                                                            ReadOnlySpan<float> actual, double maxRelL2)
    {
        Assert.Equal(expected.Length, actual.Length);
        var (rel, mx) = Error(expected, actual);
        output.WriteLine($"{label,-32} n={expected.Length,9} maxAbs={mx:E2} relL2={rel:E2}");
        Assert.True(rel <= maxRelL2, $"{label}: relative L2 error {rel:E3} > {maxRelL2:E1}");
        return (rel, mx);
    }
}
