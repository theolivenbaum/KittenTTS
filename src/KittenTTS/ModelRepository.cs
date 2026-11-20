using System.Text.Json;
using KittenTTS.IO;
using KittenTTS.Model;

namespace KittenTTS;

/// <summary>Which weights of a model to run.</summary>
public enum WeightFormat
{
    /// <summary>The int8 ONNX graph (<c>kitten_tts_2_*.onnx</c>): the default of the Python package and of the
    /// KittenTTSSharp.Models.* NuGet packages. Weights are dequantized at load; the arithmetic is float32.</summary>
    Int8,
    /// <summary>The float32 PyTorch weights (<c>*.safetensors</c>), what the Python <c>backend="torch"</c> runs.</summary>
    Float32,
}

/// <summary>A voice of a model repository.</summary>
public sealed record VoiceInfo(string Language, string Key, string Name, string Description);

/// <summary>
/// A local KittenTTS 2 nano / pico / mini repository (<c>"type": "KITTEN2_TINY"</c>), laid out like the Hugging Face
/// repositories <c>KittenML/kitten-tts-2-{nano,pico,mini}</c> (or a subset of one: only the files of the languages and
/// the weight format in use need to be present):
/// <code>
/// config.json      languages → model / weight / voice files, architectures, voices
/// vocab.json       phoneme → token id
/// kitten_tts_2_&lt;family&gt;[_&lt;lang&gt;].onnx / .safetensors
/// voices_&lt;lang&gt;.npz   one float32 [510, 256] style pack per voice
/// g2p/             the English CMUdict lexicon and BART G2P
/// </code>
/// </summary>
public sealed class ModelRepository
{
    public const string TinyType = "KITTEN2_TINY";

    private readonly JsonElement _config;

    private ModelRepository(string directory, JsonElement config)
    {
        Directory = directory;
        _config = config;
        if (!config.TryGetProperty("type", out var t) || t.GetString() != TinyType)
            throw new NotSupportedException($"{directory}: unsupported model type {(config.TryGetProperty("type", out var tt) ? tt.GetString() : "(none)")}; " +
                                            $"this library runs the KittenTTS 2 nano / pico / mini models (type {TinyType}).");
        Name = config.TryGetProperty("name", out var n) ? n.GetString() ?? "" : "";
        Family = config.TryGetProperty("family", out var f) ? f.GetString() ?? "" : "";
        SampleRate = config.TryGetProperty("sample_rate", out var sr) ? sr.GetInt32() : 24000;
        Phonemizer = config.TryGetProperty("phonemizer", out var p) ? p.GetString() ?? "v1" : "v1";
        Languages = config.GetProperty("languages").EnumerateObject().Select(e => e.Name).ToArray();
        DefaultLanguage = config.TryGetProperty("default_language", out var dl) ? dl.GetString() ?? Languages[0] : Languages[0];
        var voices = new List<VoiceInfo>();
        if (config.TryGetProperty("voices", out var vs))
            foreach (var lang in vs.EnumerateObject())
                foreach (var v in lang.Value.EnumerateArray())
                    voices.Add(new VoiceInfo(lang.Name, v.GetProperty("key").GetString()!,
                                             v.TryGetProperty("name", out var vn) ? vn.GetString()! : v.GetProperty("key").GetString()!,
                                             v.TryGetProperty("description", out var vd) ? vd.GetString() ?? "" : ""));
        Voices = voices;
        string vocabFile = config.TryGetProperty("vocab", out var vf) ? vf.GetString()! : "vocab.json";
        Vocab = JsonSerializer.Deserialize<Dictionary<string, int>>(File.ReadAllText(Path.Combine(directory, vocabFile)))!;
        G2PDirectory = Path.Combine(directory, config.TryGetProperty("g2p_dir", out var g) ? g.GetString()! : "g2p");
    }

    /// <summary>Reads <c>config.json</c> and <c>vocab.json</c> of a repository directory.</summary>
    public static ModelRepository Open(string directory)
    {
        string cfg = Path.Combine(directory, "config.json");
        if (!File.Exists(cfg)) throw new FileNotFoundException($"No config.json in {directory}.", cfg);
        using var doc = JsonDocument.Parse(File.ReadAllText(cfg));
        return new ModelRepository(directory, doc.RootElement.Clone());
    }

    public string Directory { get; }
    public string Name { get; }
    /// <summary><c>nano</c>, <c>pico</c> or <c>mini</c>.</summary>
    public string Family { get; }
    public int SampleRate { get; }
    /// <summary>The phonemizer version the models were trained with (<c>v1</c>: nano, pico; <c>v2</c>: mini).</summary>
    public string Phonemizer { get; }
    public IReadOnlyList<string> Languages { get; }
    public string DefaultLanguage { get; }
    public IReadOnlyList<VoiceInfo> Voices { get; }
    public IReadOnlyDictionary<string, int> Vocab { get; }
    public string G2PDirectory { get; }

    private JsonElement LanguageEntry(string language) =>
        _config.GetProperty("languages").TryGetProperty(language, out var e) ? e
            : throw new ArgumentException($"{Name} does not speak '{language}'; languages: {string.Join(", ", Languages)}.");

    /// <summary>The model key of a language (several languages share one model in the mini).</summary>
    public string ModelKey(string language) => LanguageEntry(language).GetProperty("model").GetString()!;

    public Architecture Architecture(string language) =>
        Model.Architecture.FromJson(_config.GetProperty("models").GetProperty(ModelKey(language)).GetProperty("architecture"));

    /// <summary>The relative path of a language's weights in <paramref name="format"/>.</summary>
    public string WeightsFile(string language, WeightFormat format)
    {
        var e = LanguageEntry(language);
        if (format == WeightFormat.Float32)
            return e.TryGetProperty("weights_file", out var w) ? w.GetString()! : throw new NotSupportedException($"{Name} has no float32 weights for '{language}'.");
        if (e.TryGetProperty("model_files", out var mf) && mf.TryGetProperty("int8", out var i8)) return i8.GetString()!;
        return e.GetProperty("model_file").GetString()!;
    }

    public string VoicesFile(string language) => LanguageEntry(language).GetProperty("voices").GetString()!;

    /// <summary>The files a language needs in <paramref name="format"/> (besides config.json, vocab.json and, for
    /// English, g2p/).</summary>
    public IEnumerable<string> RequiredFiles(string language, WeightFormat format) => [WeightsFile(language, format), VoicesFile(language)];

    /// <summary>Opens a language's weights as a tensor source (memory-mapped).</summary>
    public ITensorSource OpenWeights(string language, WeightFormat format)
    {
        string path = Path.Combine(Directory, WeightsFile(language, format));
        if (!File.Exists(path)) throw new FileNotFoundException($"Missing weights file {path}.", path);
        return format == WeightFormat.Float32 ? SafeTensors.Open(path) : OnnxWeights.Open(path);
    }

    /// <summary>The style packs of a language's voices: voice key → [510, 256] float32.</summary>
    public Dictionary<string, NpyArray> LoadVoices(string language)
    {
        string path = Path.Combine(Directory, VoicesFile(language));
        if (!File.Exists(path)) throw new FileNotFoundException($"Missing voice file {path}.", path);
        return Npy.ReadArchive(path);
    }
}
