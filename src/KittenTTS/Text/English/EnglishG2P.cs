using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

using KittenTTS.Text;

namespace KittenTTS.Text.English;

/// <summary>What the English front end did with one piece of text.</summary>
public sealed class G2PResult
{
    /// <summary>The text after normalisation (what the G2P saw).</summary>
    public required string Text { get; init; }
    /// <summary>The phoneme string, only symbols the model has a row for.</summary>
    public required string Phonemes { get; set; }
    /// <summary>Words the lexicon did not know: (word, phonemes, source = CMUdict / BART).</summary>
    public List<(string Word, string Phonemes, string Source)> Fallback { get; set; } = [];
    /// <summary>Text nothing could pronounce (other scripts, emoji, bare symbols).</summary>
    public List<string> Skipped { get; set; } = [];
    /// <summary>Phoneme symbols outside the vocabulary that were dropped.</summary>
    public List<string> Dropped { get; set; } = [];
}

/// <summary>Anything that turns text into a <see cref="G2PResult"/> (the English pipeline, or espeak for the other
/// languages).</summary>
public interface IG2P
{
    G2PResult Convert(string text, bool normalise = true);
}

/// <summary>
/// The English front end of the nano / pico / mini models (<c>.reference/kittenml/kittentts2_tiny/frontend/english.py</c>):
/// <see cref="TextNormalizer"/> → misaki lexicon (<see cref="MisakiG2P"/>) → CMUdict → BART (<see cref="BartG2P"/>)
/// for the words neither knows → the training's phoneme folds → only symbols of the model's vocabulary.
/// No espeak anywhere, as in training.
/// </summary>
public sealed class EnglishG2P : IG2P
{
    private static readonly string[] Digits = "zero one two three four five six seven eight nine".Split(' ');
    private static readonly Dictionary<char, string> Translit = new()
    {
        ['ø'] = "o", ['Ø'] = "O", ['ß'] = "ss", ['æ'] = "ae", ['Æ'] = "AE", ['œ'] = "oe", ['Œ'] = "OE", ['ł'] = "l", ['Ł'] = "L",
        ['đ'] = "d", ['Đ'] = "D", ['’'] = "'",
    };

    private readonly HashSet<char> _vocab;
    private readonly Dictionary<string, string> _cmu;
    private readonly BartG2P _bart;
    private readonly Regex _pieces;
    private readonly MisakiG2P _misaki;
    private readonly Dictionary<string, string> _memo = new(StringComparer.Ordinal);
    private readonly object _lock = new();

    // per-call reports (calls are serialized by _lock)
    private List<(string, string, string)> _used = [];
    private List<string> _skipped = [];

    /// <param name="vocab">The phoneme symbols the model has rows for.</param>
    /// <param name="g2pDirectory">A repository's <c>g2p/</c> directory (<c>cmu_misaki_us.json</c>, <c>g2p_bart_en_us/</c>).</param>
    public EnglishG2P(IEnumerable<string> vocab, string g2pDirectory)
    {
        _vocab = [.. vocab.Where(v => v.Length == 1).Select(v => v[0])];
        string cmuPath = Path.Combine(g2pDirectory, "cmu_misaki_us.json");
        if (!File.Exists(cmuPath)) throw new FileNotFoundException($"The English front end needs {cmuPath} (the model repository's g2p/ directory).", cmuPath);
        _cmu = JsonSerializer.Deserialize<Dictionary<string, string>>(File.ReadAllText(cmuPath))!;
        _bart = new BartG2P(Path.Combine(g2pDirectory, "g2p_bart_en_us"));
        // escaped like Python's re.escape: Regex.Escape leaves "-" alone, which would make a range inside [...]
        string letters = Regex.Escape(new string([.. _bart.Graphemes])).Replace("-", "\\-").Replace("]", "\\]");
        _pieces = new Regex($@"(\d+)|([{letters}]+)|(.)", RegexOptions.Singleline | RegexOptions.CultureInvariant);
        _misaki = new MisakiG2P(Fallback, unk: "");
    }

    public G2PResult Convert(string text, bool normalise = true)
    {
        text = normalise ? TextNormalizer.Normalize(text) : text;
        lock (_lock)
        {
            _used = [];
            _skipped = [];
            var (ps, tokens) = _misaki.Convert(text);
            var unsaid = tokens.Where(t => t.Phonemes == null).Select(t => t.Text).ToList();
            ps = LanguagePhonemizer.ApplyFolds(ps.Trim(), LanguagePhonemizer.FoldV1);
            var dropped = ps.Where(c => !_vocab.Contains(c)).Select(c => c.ToString()).Distinct().OrderBy(c => c, StringComparer.Ordinal).ToList();
            if (dropped.Count > 0) ps = new string(ps.Where(_vocab.Contains).ToArray());
            var skipped = _skipped.Concat(unsaid).Distinct().ToList();
            return new G2PResult { Text = text, Phonemes = ps, Fallback = [.. _used], Skipped = skipped, Dropped = dropped };
        }
    }

    /// <summary>misaki's fallback hook: CMUdict first, the BART for the rest.</summary>
    private (string?, int?) Fallback(MToken token)
    {
        var (ps, source) = Say(token.Text);
        if (ps.Length == 0)
        {
            _skipped.Add(token.Text);
            return (null, null);
        }
        _used.Add((token.Text, ps, source));
        return (ps, 2);
    }

    private (string, string) Say(string text)
    {
        // fold accents and the typographic apostrophe so "café" and "it’s" stay sayable
        var sb = new StringBuilder();
        foreach (char c in text) sb.Append(Translit.TryGetValue(c, out var r) ? r : c.ToString());
        string t = UnicodeNormalizer.Normalize(sb.ToString(), NormalizationForm.FormKD);
        t = new string(t.Where(c => CharUnicodeInfo.GetUnicodeCategory(c) != UnicodeCategory.NonSpacingMark).ToArray());
        var outList = new List<string>();
        var sources = new SortedSet<string>(StringComparer.Ordinal);
        foreach (Match m in _pieces.Matches(t))
        {
            if (m.Groups[1].Success) outList.Add(string.Join(' ', m.Groups[1].Value.Select(d => Word(Digits[d - '0'], sources))));
            else if (m.Groups[2].Success && m.Groups[2].Value.Trim('\'', '-', '.').Length > 0) outList.Add(Word(m.Groups[2].Value.Trim('\'', '-', '.'), sources));
            else if (m.Groups[3].Success && m.Groups[3].Value.Trim().Length > 0) _skipped.Add(m.Groups[3].Value);
        }
        return (string.Join(' ', outList.Where(p => p.Length > 0)), string.Join('+', sources));
    }

    private string? Cmu(string word)
    {
        string w = word.ToLowerInvariant();
        if (_cmu.TryGetValue(w, out var ps)) return ps;
        if (w.EndsWith("'s") && _cmu.TryGetValue(w[..^2], out var stem))
        {
            char last = stem[^1];
            return stem + ("ptkfθ".Contains(last) ? "s" : "szʃʒʧʤ".Contains(last) ? "ᵻz" : "z");
        }
        return null;
    }

    private string Word(string word, SortedSet<string> sources)
    {
        string? ps = Cmu(word);
        if (ps == null && word.Contains('-'))
            return string.Join(' ', word.Split('-').Where(p => p.Length > 0).Select(p => Word(p, sources)));
        sources.Add(ps != null ? "CMUdict" : "BART");
        return ps ?? Bart(word);
    }

    private string Bart(string word)
    {
        if (!_memo.TryGetValue(word, out var ps))
        {
            // the same final substitution misaki applies to its own lexicon output
            ps = _bart.Pronounce(word).Replace('ɾ', 'T').Replace('ʔ', 't');
            _memo[word] = ps;
        }
        return ps;
    }
}
