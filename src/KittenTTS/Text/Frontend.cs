using System.Text.RegularExpressions;
using KittenTTS.Text.English;

namespace KittenTTS.Text;

/// <summary>
/// Text → phoneme chunks for one model repository, exactly as in training
/// (<c>.reference/kittenml/kittentts2_tiny/frontend/__init__.py</c>): English through <see cref="EnglishG2P"/> (filtered
/// with the v1 symbol set; the v2 mini then applies <c>FOLD_EXTRA</c>), every other language through espeak-ng
/// (<see cref="LanguagePhonemizer"/>) with the symbols outside the vocabulary dropped. Text of more than
/// <see cref="MaxPhonemes"/> phonemes is cut at sentence ends, then at spaces, into chunks the model takes in one pass.
/// G2P objects are built on first use of each language. Thread-safe.
/// </summary>
public sealed partial class Frontend
{
    /// <summary>The p90 phoneme length of the training lists: the longest chunk the models are asked to say.</summary>
    public const int MaxPhonemes = 350;

    [GeneratedRegex(@"(?<=[.!?…।])\s+|(?<=[。！？])")]
    private static partial Regex SentenceSplit();

    private readonly string _phonemizer;
    private readonly IReadOnlyDictionary<string, int> _vocab;
    private readonly string _g2pDirectory;
    private readonly Dictionary<string, IG2P> _g2p = new();

    public Frontend(string phonemizerVersion, IReadOnlyDictionary<string, int> vocab, string g2pDirectory)
    {
        _phonemizer = phonemizerVersion;
        _vocab = vocab;
        _g2pDirectory = g2pDirectory;
    }

    public static Frontend For(ModelRepository repo) => new(repo.Phonemizer, repo.Vocab, repo.G2PDirectory);

    /// <summary>The G2P of a language.</summary>
    public IG2P G2P(string language)
    {
        lock (_g2p)
        {
            if (!_g2p.TryGetValue(language, out var g))
            {
                // English always filters with the v1 symbol set (v2 = v1 + U+0289 on the unused id 29), as in training
                g = language == "en"
                    ? new EnglishG2P(_vocab.Keys.Where(k => k != "ʉ"), _g2pDirectory)
                    : new EspeakLanguageG2P(new LanguagePhonemizer(language, _phonemizer), _vocab.Keys);
                _g2p[language] = g;
            }
            return g;
        }
    }

    /// <summary>text → (phoneme chunks, report).</summary>
    public (List<string> Chunks, G2PResult Report) Phonemize(string text, string language)
    {
        var (chunks, rep) = FrontEnd(G2P(language), text ?? "");
        if (language == "en" && _phonemizer == "v2")
            chunks = chunks.Select(c => LanguagePhonemizer.ApplyFolds(c, LanguagePhonemizer.FoldExtra)).ToList();
        return (chunks, rep);
    }

    private static IEnumerable<string> Wrap(string ps)
    {
        while (ps.Length > MaxPhonemes)
        {
            int cut = ps.LastIndexOf(' ', MaxPhonemes - 1, MaxPhonemes);
            cut = cut > 0 ? cut : MaxPhonemes;
            yield return ps[..cut];
            ps = ps[cut..].Trim();
        }
        if (ps.Length > 0) yield return ps;
    }

    private static (List<string>, G2PResult) FrontEnd(IG2P g2p, string text)
    {
        var whole = g2p.Convert(text);
        if (whole.Phonemes.Length <= MaxPhonemes) return (whole.Phonemes.Length > 0 ? [whole.Phonemes] : [], whole);
        var chunks = new List<string>();
        string cur = "";
        var fallback = new List<(string, string, string)>();
        var skipped = new List<string>();
        var dropped = new List<string>();
        foreach (var sent in SentenceSplit().Split(whole.Text))
        {
            if (sent.Trim().Length == 0) continue;
            var r = g2p.Convert(sent, normalise: false);
            foreach (var x in r.Fallback) if (!fallback.Contains(x)) fallback.Add(x);
            foreach (var x in r.Skipped) if (!skipped.Contains(x)) skipped.Add(x);
            foreach (var x in r.Dropped) if (!dropped.Contains(x)) dropped.Add(x);
            foreach (var ps in Wrap(r.Phonemes))
            {
                if (cur.Length > 0 && cur.Length + 1 + ps.Length > MaxPhonemes)
                {
                    chunks.Add(cur);
                    cur = ps;
                }
                else cur = $"{cur} {ps}".Trim();
            }
        }
        if (cur.Length > 0) chunks.Add(cur);
        whole.Phonemes = string.Join(" | ", chunks);
        whole.Fallback = fallback;
        whole.Skipped = skipped;
        whole.Dropped = dropped;
        return (chunks, whole);
    }
}

/// <summary>A non-English language: espeak-ng through <see cref="LanguagePhonemizer"/>, then only vocabulary symbols
/// (no text normalisation, as in training).</summary>
internal sealed class EspeakLanguageG2P(LanguagePhonemizer phonemizer, IEnumerable<string> vocab) : IG2P
{
    private readonly HashSet<char> _vocab = [.. vocab.Where(v => v.Length == 1).Select(v => v[0])];

    public G2PResult Convert(string text, bool normalise = true)
    {
        string ps = phonemizer.Phonemize(text);
        var dropped = ps.Where(c => !_vocab.Contains(c)).Select(c => c.ToString()).Distinct().OrderBy(c => c, StringComparer.Ordinal).ToList();
        if (dropped.Count > 0) ps = new string(ps.Where(_vocab.Contains).ToArray());
        return new G2PResult { Text = text, Phonemes = ps, Dropped = dropped };
    }
}
