using KittenTTS.Model;
using KittenTTS.Numerics;
using Xunit.Abstractions;

namespace KittenTTS.Tests;

/// <summary>
/// The C# network against the original implementation on the token ids, style vectors and synthesis noise recorded
/// in the reference dump: float32 weights against PyTorch stage by stage, int8 weights against ONNX Runtime running
/// the int8 graph (with basic graph optimizations, i.e. its float32 semantics). Durations must be identical and every
/// stage up to F0 / energy agrees to ~1e-6.
/// <para>
/// The waveform bounds are looser for two reasons that are properties of the model, not of the port: the sine source
/// integrates F0 into a phase of up to 10⁴ rad, so a 1e-7 difference in F0 becomes ~1e-3 rad of phase; and the
/// harmonic STFT's phase is ill-conditioned wherever a bin's magnitude is near zero (frame 0 is exactly symmetric, so
/// its imaginary parts are zero up to rounding and the angle flips between ±π on the sign of that rounding, which
/// differs between PyTorch's MKL FFT, ONNX Runtime's conv and the exact DFT here). Fed the reference's own F0 and text
/// features, the decoder alone agrees to ~1e-5 (nano) / ~3e-4 (mini).
/// </para>
/// </summary>
public sealed class ParityTests(ITestOutputHelper output)
{
    private const double EndToEnd = 2e-2;
    private const double DecoderOnly = 1e-3;

    [Fact]
    public void Float32MatchesPyTorch()
    {
        if (!Fixtures.Available(output)) return;
        var manifest = Fixtures.LoadManifest();
        var repo = ModelRepository.Open(Fixtures.ModelDir!);
        using var weights = repo.OpenWeights(manifest.Language, WeightFormat.Float32);
        using var model = new KittenModel(weights, repo.Architecture(manifest.Language));
        using var par = new Parallelism(0);
        foreach (var c in manifest.Cases)
        {
            var tokens = Fixtures.Ref(c.Name, "tokens").Data.Select(x => (int)x).ToArray();
            var style = Fixtures.Ref(c.Name, "style").Data;
            var stages = new Dictionary<string, float[]>();
            var noise = SynthesisNoise.FromSamples(Fixtures.Ref(c.Name, "noise").Data);
            var r = model.Forward(tokens, style, manifest.Speed, noise, par, (n, d, _) => stages[n] = d);
            output.WriteLine($"{c.Name}: {tokens.Length} tokens, {r.Waveform.Length} samples");
            Assert.Equal(Fixtures.Ref(c.Name, "durations").Data, stages["durations"]);
            Fixtures.AssertClose(output, "  t_en", Fixtures.Ref(c.Name, "t_en").Data, stages["t_en"], 1e-5);
            Fixtures.AssertClose(output, "  bert", Fixtures.Ref(c.Name, "bert").Data, stages["bert"], 1e-5);
            Fixtures.AssertClose(output, "  d_en", Fixtures.Ref(c.Name, "d_en").Data, stages["d_en"], 1e-5);
            Fixtures.AssertClose(output, "  d", Fixtures.Ref(c.Name, "d").Data, stages["d"], 1e-5);
            Fixtures.AssertClose(output, "  F0", Fixtures.Ref(c.Name, "F0").Data, stages["F0"], 1e-4);
            Fixtures.AssertClose(output, "  N", Fixtures.Ref(c.Name, "N").Data, stages["N"], 1e-4);
            Fixtures.AssertClose(output, "  waveform", Fixtures.Ref(c.Name, "waveform").Data, r.Waveform, EndToEnd);
        }
    }

    /// <summary>The decoder on PyTorch's own text features, durations, F0 and energy: isolates it from the float
    /// rounding of the stages before it (which the sine source integrates into its phase, see the README).</summary>
    [Fact]
    public void Float32DecoderMatchesPyTorch()
    {
        if (!Fixtures.Available(output)) return;
        var manifest = Fixtures.LoadManifest();
        var repo = ModelRepository.Open(Fixtures.ModelDir!);
        var arch = repo.Architecture(manifest.Language);
        using var weights = repo.OpenWeights(manifest.Language, WeightFormat.Float32);
        using var model = new KittenModel(weights, arch);
        using var par = new Parallelism(0);
        foreach (var c in manifest.Cases)
        {
            var tEn = Fixtures.Ref(c.Name, "t_en");
            var durations = Fixtures.Ref(c.Name, "durations").Data.Select(x => (int)x).ToArray();
            var noise = SynthesisNoise.FromSamples(Fixtures.Ref(c.Name, "noise").Data);
            var wav = model.Decode(tEn.Data, tEn.Shape[1], durations, Fixtures.Ref(c.Name, "F0").Data, Fixtures.Ref(c.Name, "N").Data,
                                   Fixtures.Ref(c.Name, "style").Data, noise, par);
            output.WriteLine($"{c.Name}:");
            Fixtures.AssertClose(output, "  waveform (decoder only)", Fixtures.Ref(c.Name, "waveform").Data, wav, DecoderOnly);
        }
    }

    [Fact]
    public void Int8MatchesOnnxRuntime()
    {
        if (!Fixtures.Available(output)) return;
        var manifest = Fixtures.LoadManifest();
        if (!Fixtures.HasRef(manifest.Cases[0].Name, "onnx_waveform"))
        {
            output.WriteLine("SKIPPED: the reference dump has no ONNX Runtime output (run torch_reference.py with --onnx).");
            return;
        }
        var repo = ModelRepository.Open(Fixtures.ModelDir!);
        using var weights = repo.OpenWeights(manifest.Language, WeightFormat.Int8);
        using var model = new KittenModel(weights, repo.Architecture(manifest.Language));
        using var par = new Parallelism(0);
        foreach (var c in manifest.Cases)
        {
            var tokens = Fixtures.Ref(c.Name, "tokens").Data.Select(x => (int)x).ToArray();
            var style = Fixtures.Ref(c.Name, "style").Data;
            var noise = SynthesisNoise.FromSamples(Fixtures.Ref(c.Name, "onnx_noise").Data);
            var r = model.Forward(tokens, style, manifest.Speed, noise, par);
            output.WriteLine($"{c.Name}: {tokens.Length} tokens, {r.Waveform.Length} samples");
            Assert.Equal(Fixtures.Ref(c.Name, "onnx_durations").Data, r.Durations.Select(x => (float)x).ToArray());
            Fixtures.AssertClose(output, "  waveform (int8)", Fixtures.Ref(c.Name, "onnx_waveform").Data, r.Waveform, EndToEnd);
        }
    }
}
