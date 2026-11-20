using KittenTTS.Audio;
using KittenTTS.IO;
using KittenTTS.Model;
using KittenTTS.Numerics;
using KittenTTS.Text;

namespace KittenTTS;

/// <summary>Options of <see cref="KittenTts"/>.</summary>
public sealed record KittenTtsOptions
{
    /// <summary>Worker threads (0 = all cores, 1 = run on the calling thread).</summary>
    public int Threads { get; init; }

    /// <summary>Which weights to run (default: the int8 graphs, like the Python package).</summary>
    public WeightFormat Weights { get; init; } = WeightFormat.Int8;

    /// <summary>Seed of the synthesis noise (the hn-NSF source's Gaussian noise); null for a random seed per call.</summary>
    public int? Seed { get; init; }

    /// <summary>Add the synthesis noise, as the original does (default). Off gives deterministic output with purely
    /// harmonic voiced parts.</summary>
    public bool Noise { get; init; } = true;
}

/// <summary>
/// Text-to-speech with a KittenTTS 2 nano / pico / mini model: the C# counterpart of the Python
/// <c>KittenTTS("KittenML/kitten-tts-2-nano")</c> (<c>kittentts2_tiny.KittenTTS2Tiny</c>), with the same voices,
/// languages, chunking and post-processing. One instance is thread-safe.
/// <code>
/// using var tts = await KittenTts.LoadAsync(KittenModel.Nano);            // English, int8; downloads on first use
/// float[] audio = tts.Generate("Hello there.", voice: "Eleanor");          // 24 kHz mono
/// tts.GenerateToFile("Guten Morgen.", "out.wav", voice: "Paul");           // a German voice: German
/// </code>
/// </summary>
public sealed class KittenTts : IDisposable
{
    /// <summary>The end-of-text token the models were trained with.</summary>
    public const int EndToken = 10;

    /// <summary>Silence between chunks of long text, in seconds.</summary>
    public const float GapSeconds = 0.15f;

    private readonly KittenTtsOptions _options;
    private readonly Parallelism _par;
    private readonly Frontend _frontend;
    private readonly Dictionary<string, KittenNetwork> _models = new(StringComparer.Ordinal);
    private readonly Dictionary<string, Dictionary<string, NpyArray>> _voices = new(StringComparer.Ordinal);
    private readonly List<IDisposable> _owned = new();
    private readonly object _lock = new();

    private KittenTts(ModelRepository repository, KittenTtsOptions options)
    {
        Repository = repository;
        _options = options;
        _par = new Parallelism(options.Threads);
        _frontend = Frontend.For(repository);
    }

    /// <summary>Opens a local repository directory (see <see cref="ModelRepository"/>).</summary>
    public static KittenTts Load(string repositoryDirectory, KittenTtsOptions? options = null) =>
        new(ModelRepository.Open(repositoryDirectory), options ?? new KittenTtsOptions());

    /// <summary>
    /// Loads a released model for the given languages (default: the model's default language, English): from an
    /// installed KittenTTSSharp.Models.* package, the download cache, or Hugging Face (see <see cref="KittenModels"/>).
    /// </summary>
    public static async Task<KittenTts> LoadAsync(KittenModel model, IReadOnlyCollection<string>? languages = null, KittenTtsOptions? options = null,
                                                  string? cacheDirectory = null, Action<DownloadProgress>? progress = null,
                                                  CancellationToken cancellationToken = default)
    {
        options ??= new KittenTtsOptions();
        string dir = await KittenModels.EnsureFilesAsync(model, languages, options.Weights, cacheDirectory, progress, cancellationToken).ConfigureAwait(false);
        return Load(dir, options);
    }

    public ModelRepository Repository { get; }

    /// <summary>24000 for every released model.</summary>
    public int SampleRate => Repository.SampleRate;

    public IReadOnlyList<string> Languages => Repository.Languages;

    /// <summary>The voices of one language (default: all).</summary>
    public IReadOnlyList<VoiceInfo> Voices(string? language = null) =>
        language == null ? Repository.Voices : Repository.Voices.Where(v => v.Language == CheckLanguage(language)).ToList();

    private string CheckLanguage(string? language)
    {
        if (language == null) return Repository.DefaultLanguage;
        if (!Repository.Languages.Contains(language))
            throw new ArgumentException($"{Repository.Name} does not speak '{language}'; languages: {string.Join(", ", Repository.Languages)}.");
        return language;
    }

    /// <summary>
    /// → (language, voice key). <paramref name="voice"/> is a name in any case ("Anna", "anna"), a key, a description,
    /// or null (the language's first voice); without <paramref name="language"/>, a name only one language has decides
    /// it, and a name several languages share means the default language's voice.
    /// </summary>
    public (string Language, string Key) ResolveVoice(string? voice = null, string? language = null)
    {
        if (voice == null)
        {
            string lang = CheckLanguage(language);
            var first = Repository.Voices.FirstOrDefault(v => v.Language == lang) ?? throw new ArgumentException($"No voices for '{lang}'.");
            return (lang, first.Key);
        }
        var langs = language != null ? [CheckLanguage(language)] : Repository.Languages;
        string want = voice.ToLowerInvariant();
        var hits = Repository.Voices.Where(v => langs.Contains(v.Language)
                                                && (v.Key.ToLowerInvariant() == want || v.Name.ToLowerInvariant() == want || v.Description.ToLowerInvariant() == want))
                                    .ToList();
        if (hits.Count == 0)
            throw new ArgumentException($"No voice '{voice}'{(language != null ? $" for '{language}'" : "")}; voices: " +
                                        string.Join(", ", Repository.Voices.Where(v => langs.Contains(v.Language)).Select(v => $"{v.Name} ({v.Language})")));
        if (hits.Count > 1)
        {
            var def = hits.FirstOrDefault(h => h.Language == Repository.DefaultLanguage);
            if (language == null && def != null) return (def.Language, def.Key);
            throw new ArgumentException($"Voice '{voice}' exists in {string.Join(", ", hits.Select(h => h.Language))}: pass the language.");
        }
        return (hits[0].Language, hits[0].Key);
    }

    /// <summary>text → the phoneme chunks the model will speak.</summary>
    public IReadOnlyList<string> Phonemize(string text, string? language = null) => _frontend.Phonemize(text, CheckLanguage(language)).Chunks;

    /// <summary>Synthesizes <paramref name="text"/>: float32 audio at <see cref="SampleRate"/>, chunks joined by
    /// <see cref="GapSeconds"/> of silence.</summary>
    /// <param name="text">What to say.</param>
    /// <param name="voice">A voice name from <see cref="Voices"/> (any case); default: the language's first voice.</param>
    /// <param name="language">A code from <see cref="Languages"/>; default: the voice's language, else the default language.</param>
    /// <param name="speed">&gt; 1 speaks faster.</param>
    public float[] Generate(string text, string? voice = null, string? language = null, float speed = 1f)
    {
        int gap = (int)(GapSeconds * SampleRate);
        var parts = GenerateStream(text, voice, language, speed).ToList();
        if (parts.Count == 0) return [];
        var result = new float[parts.Sum(p => p.Length) + gap * (parts.Count - 1)];
        int o = 0;
        for (int i = 0; i < parts.Count; i++)
        {
            parts[i].CopyTo(result, o);
            o += parts[i].Length + gap;
        }
        return result;
    }

    /// <summary>Yields the audio of each chunk (long text is cut at sentence ends) as it is produced.</summary>
    public IEnumerable<float[]> GenerateStream(string text, string? voice = null, string? language = null, float speed = 1f)
    {
        var (lang, key) = ResolveVoice(voice, language);
        var chunks = _frontend.Phonemize(text, lang).Chunks;
        for (int i = 0; i < chunks.Count; i++) yield return Chunk(lang, key, chunks[i], speed, keepStart: i == 0);
    }

    /// <summary>Synthesizes to a 16-bit PCM WAV file.</summary>
    public void GenerateToFile(string text, string path, string? voice = null, string? language = null, float speed = 1f) =>
        WaveFile.WriteMono16(path, Generate(text, voice, language, speed), SampleRate);

    /// <summary>Phonemes → token ids: [0] + symbols (parentheses read as commas) + [end, 0]; unknown symbols are dropped.</summary>
    public int[] Tokens(string phonemes)
    {
        var ids = new List<int> { 0 };
        foreach (var r in phonemes.EnumerateRunes())
        {
            string ch = r.ToString();
            if (Repository.Vocab.TryGetValue(ch, out int id)) ids.Add(id);
            else if (ch is "(" or ")") ids.Add(Repository.Vocab[","]);
        }
        ids.Add(EndToken);
        ids.Add(0);
        return [.. ids];
    }

    /// <summary>One chunk, post-processed as in the original: the end token's frames cut (and the start pause of
    /// every chunk but the first), 5 ms fades.</summary>
    private float[] Chunk(string lang, string key, string phonemes, float speed, bool keepStart)
    {
        var tokens = Tokens(phonemes);
        var model = ModelFor(lang);
        if (tokens.Length > model.MaxTokens) throw new ArgumentException($"A chunk of {tokens.Length} tokens exceeds the model's {model.MaxTokens}.");
        var pack = VoicePack(lang, key);
        int rows = pack.Shape[0], dim = pack.Shape[1];
        int row = Math.Min(tokens.Length - 1, rows - 1);
        var style = pack.Data.AsSpan(row * dim, dim);
        var noise = _options.Noise ? SynthesisNoise.Gaussian(_options.Seed) : SynthesisNoise.None;
        var output = model.Forward(tokens, style, speed, noise, _par);

        var w = output.Waveform;
        for (int i = 0; i < w.Length; i++) w[i] = Math.Clamp(w[i], -1f, 1f);
        var dur = output.Durations;
        long total = dur.Sum(d => (long)d);
        int spf = Math.Max(1, (int)Math.Round((double)w.Length / Math.Max(total, 1), MidpointRounding.ToEven));
        int lead = keepStart ? 0 : Math.Max(0, dur[0] - 1) * spf;
        int trail = Math.Max(0, dur[^1] - 1) * spf;
        if (lead + trail + spf < w.Length) w = w[lead..(w.Length - trail)];
        int f = Math.Min(120, w.Length / 8);                       // ~5 ms fades
        if (f > 0)
        {
            for (int i = 0; i < f; i++)
            {
                float g = f == 1 ? 0f : (float)((double)i / (f - 1));   // np.linspace(0, 1, f, dtype=float32)
                w[i] *= g;
                w[w.Length - 1 - i] *= g;
            }
        }
        return w;
    }

    private KittenNetwork ModelFor(string lang)
    {
        string modelKey = Repository.ModelKey(lang);
        lock (_lock)
        {
            if (!_models.TryGetValue(modelKey, out var m))
            {
                var weights = Repository.OpenWeights(lang, _options.Weights);
                _owned.Add(weights);
                m = new KittenNetwork(weights, Repository.Architecture(lang));
                _owned.Add(m);
                _models[modelKey] = m;
            }
            return m;
        }
    }

    private NpyArray VoicePack(string lang, string key)
    {
        lock (_lock)
        {
            if (!_voices.TryGetValue(lang, out var packs)) _voices[lang] = packs = Repository.LoadVoices(lang);
            return packs.TryGetValue(key, out var p) ? p : throw new ArgumentException($"The voice pack of '{lang}' has no voice '{key}'.");
        }
    }

    public void Dispose()
    {
        lock (_lock)
        {
            for (int i = _owned.Count - 1; i >= 0; i--) _owned[i].Dispose();
            _owned.Clear();
            _models.Clear();
        }
        _par.Dispose();
    }
}
