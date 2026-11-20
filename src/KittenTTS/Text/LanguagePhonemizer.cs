using System.Text.RegularExpressions;
using KittenTTS.Text.Espeak;

namespace KittenTTS.Text;

/// <summary>
/// The phonemizer of one non-English language, exactly as the KittenTTS 2 models were trained: espeak-ng through
/// misaki's <see cref="EspeakG2P"/>, then the fold tables of phonemizer <c>v1</c> (nano, pico: <c>phonemize_multi.py</c>)
/// or <c>v2</c> (mini: <c>phonemize_v2.py</c>, which adds Korean through espeak, Hindi danda / flap handling and
/// Arabic question marks). No text normalization. The result can still hold symbols outside a model's vocabulary;
/// the caller drops those. Needs the espeak-ng library (see <see cref="EspeakNative"/>).
/// </summary>
public sealed partial class LanguagePhonemizer
{
    /// <summary>Language code → espeak voice (<c>ESPEAK_VOICE</c> of phonemize_multi.py; v2 adds <c>ko</c>).</summary>
    public static readonly IReadOnlyDictionary<string, string> EspeakVoices = new Dictionary<string, string>
    {
        ["es"] = "es", ["fr"] = "fr-fr", ["it"] = "it", ["pt"] = "pt-br", ["hi"] = "hi", ["de"] = "de", ["nl"] = "nl",
        ["pl"] = "pl", ["ru"] = "ru", ["tr"] = "tr", ["ar"] = "ar", ["he"] = "he", ["sv"] = "sv", ["da"] = "da",
        ["no"] = "nb", ["fi"] = "fi", ["el"] = "el", ["cs"] = "cs", ["hu"] = "hu", ["ro"] = "ro", ["uk"] = "uk",
        ["ms"] = "ms", ["sw"] = "sw",
    };

    /// <summary>FOLD of phonemize_multi.py (v1), in dict order: G2P symbols the 114-symbol vocab has no row for. The
    /// English front end applies it too.</summary>
    public static readonly IReadOnlyList<(string From, string To)> FoldV1 =
    [
        ("̪", ""), ("̩", ""), ("̍", ""), ("̯", ""), ("ɭ", "l"), ("ɫ", "l"),
        ("ɵ", "ø"), ("ε", "ɛ"), ("\x05", ""), ("\x06", ""), ("\x07", ""), ("\x08", ""),
        ("{", ""), ("}", ""), ("[", ""), ("]", ""), ("¿", ""), ("¡", ""), (" ", " "),
    ];

    /// <summary>FOLD_EXTRA of phonemize_v2.py (applied after v1's table; also to English for the v2 models).</summary>
    public static readonly IReadOnlyList<(string From, string To)> FoldExtra =
    [
        (" ", " "), ("\t", " "), ("«", ""), ("»", ""), ("‘", ""), ("’", ""), ("'", ""),
        ("~", ""), ("·", ""), ("#", ""), ("g", "ɡ"),
    ];

    [GeneratedRegex(@"r\.(?=[^\s.,?!;:…])")]
    private static partial Regex HindiFlapRe();

    [GeneratedRegex("[ड़ढ़]|[डढ]़")]
    private static partial Regex HindiFlapTextRe();

    private readonly EspeakG2P _g2p;
    private readonly (string From, string To)[] _fold;

    /// <param name="language">A language code (<c>de</c>, <c>fr</c>, <c>hi</c>, …; <c>ko</c> needs v2).</param>
    /// <param name="phonemizerVersion"><c>v1</c> (nano / pico) or <c>v2</c> (mini), the repository's <c>phonemizer</c>.</param>
    public LanguagePhonemizer(string language, string phonemizerVersion)
    {
        if (phonemizerVersion is not ("v1" or "v2")) throw new ArgumentException($"Unknown phonemizer version '{phonemizerVersion}'.", nameof(phonemizerVersion));
        Language = language;
        Version = phonemizerVersion;
        bool v2 = phonemizerVersion == "v2";
        if (language is "en" or "en-us" or "en-gb")
            throw new ArgumentException("English has its own front end (lexicon + BART G2P), not espeak.", nameof(language));
        if (language is "ja" or "zh" or "vi" || (language == "ko" && !v2))
            throw new NotSupportedException($"Language '{language}' uses misaki's own G2P in the original implementation, which is not ported to C#.");
        string? voice = language == "ko" ? "ko" : EspeakVoices.GetValueOrDefault(language);
        if (voice == null) throw new ArgumentException($"No espeak voice is mapped for language '{language}'.", nameof(language));
        _g2p = new EspeakG2P(voice);
        _fold = v2 ? [.. FoldV1, .. FoldExtra] : [.. FoldV1];
    }

    public string Language { get; }
    public string Version { get; }

    /// <summary>Applies a fold table in order (as the Python loops over the dict).</summary>
    public static string ApplyFolds(string ps, IEnumerable<(string From, string To)> table)
    {
        foreach (var (from, to) in table)
            if (ps.Contains(from, StringComparison.Ordinal)) ps = ps.Replace(from, to, StringComparison.Ordinal);
        return ps;
    }

    /// <summary>The espeak voice in use.</summary>
    public string EspeakVoice => _g2p.Language;

    /// <summary>Text → phoneme string (phonemize_lang of the selected version).</summary>
    public string Phonemize(string text)
    {
        bool v2 = Version == "v2";
        if (v2)
        {
            if (Language == "hi") text = text.Replace("।", ".").Replace("॥", ".");
            else if (Language == "ar") text = text.Replace("؟", "?");
        }
        string ps = EspeakBackend.PyStrip(_g2p.Phonemize(text));
        ps = ApplyFolds(ps, _fold);
        if (v2 && Language == "hi")
        {
            // espeak writes ड़ / ढ़ as "r." / "r.h"; inside a word it is always the flap, at a word end only when the
            // text holds the letter (there "r." could be r and a full stop)
            ps = ps.Replace("r.h", "ɽʰ");
            ps = HindiFlapRe().Replace(ps, "ɽ");
            if (HindiFlapTextRe().IsMatch(text)) ps = ps.Replace("r.", "ɽ");
        }
        return ps;
    }
}
