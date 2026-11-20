using KittenTTS.IO;
using KittenTTS.Model;
using KittenTTS.Numerics;
using Xunit.Abstractions;

namespace KittenTTS.Tests;

/// <summary>
/// Layer by layer against PyTorch: every module of the network (text encoder blocks, LSTMs, the 12 ALBERT layer
/// applications, the prosody predictor's blocks, the decoder's blocks, the generator's source, convolutions, residual
/// blocks and iSTFT) is run <b>on the input PyTorch gave it</b> (recorded by <c>scripts/torch_reference.py --layers</c>)
/// and its output compared with PyTorch's. Each layer is checked in isolation, so an error cannot hide behind (or be
/// blamed on) the layers before it. Float32 weights against PyTorch with the checkpoint; int8 weights (read by the
/// C# ONNX importer) against PyTorch running the int8 graph's dequantized weights, mapped onto the modules
/// independently by <c>torch_reference.py --weights int8</c> (whose output matches ONNX Runtime's durations and
/// waveform; see <see cref="ParityTests"/>).
/// </summary>
public sealed class LayerParityTests(ITestOutputHelper output)
{
    /// <summary>Relative L2 bound of every isolated layer: float32 rounding, with headroom for long reductions.</summary>
    private const double LayerBound = 2e-5;

    private sealed record Result(string Case, string Layer, int N, double RelL2, double MaxAbs, double Bound);

    [Fact]
    public void EveryLayerMatchesPyTorch()
    {
        if (!Fixtures.Available(output)) return;
        var manifest = Fixtures.LoadManifest();
        if (!Directory.Exists(Path.Combine(Fixtures.ReferenceDir!, manifest.Cases[0].Name, "layers")))
        {
            output.WriteLine("SKIPPED: the reference dump has no layer records (run torch_reference.py with --layers).");
            return;
        }
        var repo = ModelRepository.Open(Fixtures.ModelDir!);
        var arch = repo.Architecture(manifest.Language);
        // the reference is PyTorch either way: with the float32 checkpoint, or with the int8 graph's dequantized weights
        using var weights = repo.OpenWeights(manifest.Language, manifest.Format);
        using var net = new KittenNetwork(weights, arch, onnxArithmetic: false);
        output.WriteLine($"{repo.Name}, {manifest.Format} weights");
        using var par = new Parallelism(0);
        var results = new List<Result>();
        foreach (var c in manifest.Cases) results.AddRange(Case(net, arch, c.Name, par));

        output.WriteLine($"{"layer",-48} {"n",9} {"max rel L2",11} {"max |err|",10}");
        foreach (var g in results.GroupBy(r => r.Layer.Split('#')[0]))
            output.WriteLine($"{g.Key,-48} {g.Max(r => r.N),9} {g.Max(r => r.RelL2),11:E2} {g.Max(r => r.MaxAbs),10:E2}");
        output.WriteLine($"{results.Count} layer checks over {manifest.Cases.Count} cases");
        var failed = results.Where(r => r.RelL2 > r.Bound).ToList();
        foreach (var f in failed) output.WriteLine($"FAILED {f.Case} {f.Layer}: rel L2 {f.RelL2:E2} > {f.Bound:E1}");
        Assert.Empty(failed);
    }

    private List<Result> Case(KittenNetwork net, Architecture arch, string caseName, Parallelism par)
    {
        string dir = Path.Combine(Fixtures.ReferenceDir!, caseName, "layers");
        bool Has(string name) => File.Exists(Path.Combine(dir, name + ".npy"));
        NpyArray Load(string name) => Npy.Read(Path.Combine(dir, name + ".npy"));
        Act In(string layer)
        {
            var a = Load(layer + ".in");
            return a.Shape.Length == 2 ? new Act(a.Shape[0], a.Shape[1], a.Data) : new Act(a.Data.Length, 1, a.Data);
        }
        var style = Fixtures.Ref(caseName, "style").Data;
        var sA = style[..arch.StyleDim];
        var sP = style[arch.StyleDim..];
        var tokens = Fixtures.Ref(caseName, "tokens").Data.Select(x => (int)x).ToArray();
        var results = new List<Result>();
        void Check(string layer, float[] actual, string? expectedName = null, double bound = LayerBound)
        {
            var expected = Load(expectedName ?? layer + ".out").Data;
            Assert.True(expected.Length == actual.Length, $"{caseName} {layer}: {actual.Length} values, PyTorch has {expected.Length}");
            var (rel, mx) = Fixtures.Error(expected, actual);
            results.Add(new Result(caseName, layer, expected.Length, rel, mx, bound));
        }

        // text encoder
        var te = net._textEncoder;
        for (int i = 0; i < te.CnnCount; i++) Check($"text_encoder.cnn.{i}", te.CnnBlock(i, In($"text_encoder.cnn.{i}"), par).Data);
        Check("text_encoder.lstm", te._lstm.Forward(In("text_encoder.lstm"), par).Data);

        // PL-BERT: embeddings, projection, every application of the shared layer
        var bert = net._bert;
        Check("bert.embeddings", bert.Embeddings(tokens).Data);
        Check("bert.encoder.embedding_hidden_mapping_in", bert._mapIn.Forward(In("bert.encoder.embedding_hidden_mapping_in"), par).Data);
        for (int k = 0; k < arch.AlbertLayers; k++)
        {
            string key = k == 0 ? "bert.layer" : $"bert.layer#{k}";
            Check(key, bert.Layer(In(key), par).Data);
        }
        Check("bert_encoder", net._bertEncoder.Forward(In("bert_encoder"), par).Data);

        // prosody predictor
        var p = net._predictor;
        Check("predictor.text_encoder", p.EncodeText(In("predictor.text_encoder"), sP, par).Data);
        for (int i = 0; i < 2 * p._encoder.Count; i++)
        {
            string key = $"predictor.text_encoder.lstms.{i}";
            var (lstm, fc) = p._encoder[i / 2];
            Check(key, (i % 2 == 0 ? lstm.Forward(In(key), par) : ProsodyPredictor.AdaLayerNorm(In(key), fc, sP)).Data);
        }
        Check("predictor.lstm", p._lstm.Forward(In("predictor.lstm"), par).Data);
        Check("predictor.duration_proj", p._durationProj.Forward(In("predictor.duration_proj"), par).Data);
        Check("predictor.shared", p._shared.Forward(In("predictor.shared"), par).Data);
        for (int i = 0; i < 3; i++)
        {
            Check($"predictor.F0.{i}", p._f0[i].Forward(In($"predictor.F0.{i}"), sP, par).Data);
            Check($"predictor.N.{i}", p._n[i].Forward(In($"predictor.N.{i}"), sP, par).Data);
        }
        Check("predictor.F0_proj", p._f0Proj.Forward(In("predictor.F0_proj"), par).Data);
        Check("predictor.N_proj", p._nProj.Forward(In("predictor.N_proj"), par).Data);

        // decoder
        var d = net._decoder;
        Check("decoder.F0_conv", d._f0Conv.Forward(In("decoder.F0_conv"), par).Data);
        Check("decoder.N_conv", d._nConv.Forward(In("decoder.N_conv"), par).Data);
        Check("decoder.asr_res.0", d._asrRes.Forward(In("decoder.asr_res.0"), par).Data);
        Check("decoder.encode", d._encode.Forward(In("decoder.encode"), sA, par).Data);
        for (int i = 0; i < d._decode.Length; i++) Check($"decoder.decode.{i}", d._decode[i].Forward(In($"decoder.decode.{i}"), sA, par).Data);

        // generator: harmonic source and its STFT, every convolution and residual block, the iSTFT, the whole
        var g = d._generator;
        var f0 = Load("decoder.generator.f0").Data;
        var noise = SynthesisNoise.FromSamples(Fixtures.Ref(caseName, "noise").Data);
        var (source, har) = g.HarmonicSource(f0, noise, par);
        Check("decoder.generator.m_source", source);
        CheckHarmonics(caseName, har, Load("decoder.generator.har"), results);
        for (int i = 0; i < g._ups.Length; i++)
        {
            Check($"decoder.generator.noise_convs.{i}", g._noiseConvs[i].Forward(In($"decoder.generator.noise_convs.{i}"), par).Data);
            Check($"decoder.generator.noise_res.{i}", g._noiseRes[i].Forward(In($"decoder.generator.noise_res.{i}"), sA, par).Data);
            Check($"decoder.generator.ups.{i}", g._ups[i].Forward(In($"decoder.generator.ups.{i}"), par).Data);
        }
        for (int i = 0; i < g._resblocks.Length; i++)
            Check($"decoder.generator.resblocks.{i}", g._resblocks[i].Forward(In($"decoder.generator.resblocks.{i}"), sA, par).Data);
        Check("decoder.generator.conv_post", g._convPost.Forward(In("decoder.generator.conv_post"), par).Data);
        var post = Load("decoder.generator.conv_post.out");
        Check("decoder.generator.istft", g.Istft(new Act(post.Shape[0], post.Shape[1], post.Data)), "decoder.generator.out");
        // The whole generator chains the source's STFT into every stage; the STFT phase of near-silent bins is
        // ill-conditioned (see CheckHarmonics), so this chained check has a looser bound.
        var gin = Load("decoder.generator.in");
        var noise2 = SynthesisNoise.FromSamples(Fixtures.Ref(caseName, "noise").Data);
        Check("decoder.generator (chained)", g.Forward(new Act(gin.Shape[0], gin.Shape[1], gin.Data), sA, f0, noise2, par),
              "decoder.generator.out", bound: 1e-3);
        return results;
    }

    /// <summary>
    /// The STFT of the harmonic source: magnitudes must agree to float rounding; phases only where they are defined. A
    /// bin whose magnitude is ~1e-7 of the frame's (rounding noise of an exact zero, e.g. the imaginary part of the
    /// symmetric first frame) has an arbitrary angle (±π), which differs between MKL's FFT and the exact DFT here.
    /// </summary>
    private static void CheckHarmonics(string caseName, Act har, NpyArray expected, List<Result> results)
    {
        int bins = har.C / 2, frames = har.T;
        Assert.Equal(expected.Data.Length, har.Data.Length);
        var magE = new List<float>(); var magA = new List<float>();
        var phE = new List<float>(); var phA = new List<float>();
        float peak = 0;
        for (int t = 0; t < frames; t++) for (int b = 0; b < bins; b++) peak = Math.Max(peak, expected.Data[t * har.C + b]);
        for (int t = 0; t < frames; t++)
            for (int b = 0; b < bins; b++)
            {
                float me = expected.Data[t * har.C + b];
                magE.Add(me); magA.Add(har.Data[t * har.C + b]);
                if (me < 1e-3f * peak) continue;
                float pe = expected.Data[t * har.C + bins + b], pa = har.Data[t * har.C + bins + b];
                float diff = pa - pe;
                if (diff > MathF.PI) pa -= 2 * MathF.PI;           // ±π are the same angle
                else if (diff < -MathF.PI) pa += 2 * MathF.PI;
                phE.Add(pe); phA.Add(pa);
            }
        var (rm, mm) = Fixtures.Error(magE.ToArray(), magA.ToArray());
        results.Add(new Result(caseName, "decoder.generator.har (magnitude)", magE.Count, rm, mm, LayerBound));
        var (rp, mp) = Fixtures.Error(phE.ToArray(), phA.ToArray());
        results.Add(new Result(caseName, "decoder.generator.har (phase, defined bins)", phE.Count, rp, mp, 1e-4));
    }
}
