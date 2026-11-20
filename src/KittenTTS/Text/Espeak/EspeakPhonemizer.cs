using System.Text;
using System.Text.RegularExpressions;

namespace KittenTTS.Text.Espeak;

/// <summary>
/// phonemizer's <c>EspeakBackend(language, preserve_punctuation=True, with_stress=True, tie='^',
/// language_switch='remove-flags')</c> (phonemizer 3.6) on one line of text, default separators, <c>strip=False</c>:
/// <list type="number">
/// <item>the text is cut into chunks at runs of punctuation (a <c>,</c> or <c>.</c> between two digits is a decimal
/// separator, not a cut), each chunk phonemized alone;</item>
/// <item>chunks separated only by inline marks (quotes, brackets) are phonemized together as well, and that version
/// is used when its words can be split back at the marks (so they are pronounced in context);</item>
/// <item>espeak's output is cleaned (merged lines, doubled separators, language-switch flags such as <c>(en)</c>
/// removed), the tie U+0361 becomes <c>^</c>, and the punctuation is put back.</item>
/// </list>
/// </summary>
public sealed partial class EspeakBackend(string language)
{
    private const string DefaultMarks = ";:,.!?¡¿—…\"«»“”(){}[]";
    private const string InlineMarks = "\"«»“”‘’„‹›()[]{}";
    private const string Tie = "^";

    // \s* (marks)+ \s*, repeated; ',' and '.' are marks unless between two digits
    [GeneratedRegex(@"(\s*(?:[;:!?¡¿—…""«»“”(){}\[\]]|(?<![0-9])[,.]|[,.](?![0-9]))+\s*)+")]
    private static partial Regex MarksRe();

    [GeneratedRegex(@"\(.+?\)")]
    private static partial Regex FlagsRe();

    [GeneratedRegex(@"_+")]
    private static partial Regex UnderscoresRe();

    /// <summary>The espeak language code (e.g. <c>fr-fr</c>).</summary>
    public string Language { get; } = language;

    /// <summary>Phonemizes one line (phonemizer returns it with a trailing word separator; callers strip it).</summary>
    public string Phonemize(string line)
    {
        // preserve(): chunks between punctuation runs, and the line's parts (chunk index or mark)
        var chunks = new List<string>();
        var parts = new List<object>();
        int cursor = 0;
        foreach (Match m in MarksRe().Matches(line))
        {
            if (m.Index > cursor)
            {
                parts.Add(chunks.Count);
                chunks.Add(line[cursor..m.Index]);
            }
            parts.Add(m.Value);
            cursor = m.Index + m.Length;
        }
        if (cursor < line.Length)
        {
            parts.Add(chunks.Count);
            chunks.Add(line[cursor..]);
        }

        var groups = InlineGroups(parts);
        var texts = new List<string>(chunks);
        foreach (var g in groups) texts.Add(string.Join(" ", g.Select(c => chunks[c])));

        var phonemized = texts.Select(t => PostprocessLine(EspeakNative.TextToPhonemes(Language, t))).ToList();
        var chunkPh = phonemized.Take(chunks.Count).ToList();
        var context = groups.Select((g, i) => (Group: g, Text: phonemized[chunks.Count + i])).ToList();
        return Restore(chunkPh, parts, context);
    }

    /// <summary>EspeakBackend._postprocess_line with with_stress, tie '^', strip False, remove-flags.</summary>
    private static string PostprocessLine(string line)
    {
        line = PyStrip(line).Replace("\n", " ").Replace("  ", " ");
        if (line.Contains("__")) line = UnderscoresRe().Replace(line, "_");
        line = line.Replace("_ ", " ");
        if (FlagsRe().IsMatch(line)) line = FlagsRe().Replace(line, "");
        if (line.Length == 0) return "";
        var words = line.Split(' ').Select(w => PyStrip(w).Replace("͡", Tie));
        return string.Join(" ", words) + " ";
    }

    private static bool IsInline(string mark)
    {
        var m = new StringBuilder();
        foreach (var ch in mark) if (!IsPyWhitespace(ch)) m.Append(ch);
        if (m.Length == 0) return false;
        foreach (var ch in m.ToString()) if (!InlineMarks.Contains(ch)) return false;
        return true;
    }

    private static List<List<int>> InlineGroups(List<object> parts)
    {
        var groups = new List<List<int>>();
        var group = new List<int>();
        for (int num = 0; num < parts.Count; num++)
        {
            if (parts[num] is not int part) continue;
            if (group.Count > 0 && num >= 2 && parts[num - 2] is int prev && prev == group[^1] && parts[num - 1] is string mk && IsInline(mk))
                group.Add(part);
            else
            {
                if (group.Count > 1) groups.Add(group);
                group = [part];
            }
        }
        if (group.Count > 1) groups.Add(group);
        return groups;
    }

    private static List<string> Words(string text) => text.Split(' ').Where(w => w.Length > 0).ToList();

    /// <summary>Punctuation._context: a group phonemized as a whole, cut back into its chunks, or null.</summary>
    private static List<string>? Context(List<string> text, List<int> group, string phonemized)
    {
        var counts = group.Select(c => Words(text[c]).Count).ToList();
        if (counts.Any(c => c == 0) || Words(phonemized).Count != counts.Sum()) return null;
        var starts = new HashSet<int>();
        int acc = 0;
        for (int i = 0; i < counts.Count - 1; i++) { acc += counts[i]; starts.Add(acc); }
        var pieces = phonemized.Split(' ');
        var cuts = new List<int> { 0 };
        int nwords = 0;
        for (int num = 0; num < pieces.Length; num++)
        {
            if (pieces[num].Length == 0) continue;
            if (starts.Contains(nwords)) cuts.Add(num);
            nwords++;
        }
        cuts.Add(pieces.Length);
        var result = new List<string>();
        for (int i = 0; i + 1 < cuts.Count; i++) result.Add(string.Join(" ", pieces[cuts[i]..cuts[i + 1]]));
        return result;
    }

    /// <summary>Punctuation.restore for one line, separator word ' ', strip False.</summary>
    private static string Restore(List<string> text, List<object> parts, List<(List<int> Group, string Text)> context)
    {
        var position = new Dictionary<int, int>();
        for (int pos = 0; pos < parts.Count; pos++) if (parts[pos] is int c) position[c] = pos;
        var inContext = new Dictionary<int, string>();
        foreach (var (group, ph) in context)
        {
            bool ok = true;
            for (int i = 0; i + 1 < group.Count && ok; i++)
            {
                if (!position.TryGetValue(group[i], out int a) || !position.TryGetValue(group[i + 1], out int b)) { ok = false; break; }
                ok = b == a + 2 && parts[a + 1] is string mk && IsInline(mk);
            }
            if (!ok || group.Any(g => !position.ContainsKey(g))) continue;
            var pieces = Context(text, group, ph);
            if (pieces != null) for (int i = 0; i < group.Count; i++) inContext[group[i]] = pieces[i];
        }

        var sb = new StringBuilder();
        string? mark = null;
        foreach (var part in parts)
        {
            if (part is string m)
            {
                if (sb.Length > 0 && sb[^1] == ' ') sb.Length--;
                mark = m.Replace(" ", " ");
                sb.Append(mark);
            }
            else
            {
                int c = (int)part;
                string chunk = inContext.TryGetValue(c, out var x) ? x : text[c];
                if (chunk.Length > 0)
                {
                    sb.Append(chunk);
                    mark = null;
                }
            }
        }
        if (mark != null && !mark.EndsWith(' ')) sb.Append(' ');
        return sb.ToString();
    }

    /// <summary>Python's str.isspace() for one character.</summary>
    internal static bool IsPyWhitespace(char c) => char.IsWhiteSpace(c) || c is '\x1c' or '\x1d' or '\x1e' or '\x1f';

    /// <summary>Python's str.strip() (Unicode whitespace).</summary>
    internal static string PyStrip(string s)
    {
        int a = 0, b = s.Length;
        while (a < b && IsPyWhitespace(s[a])) a++;
        while (b > a && IsPyWhitespace(s[b - 1])) b--;
        return s[a..b];
    }
}

/// <summary>
/// misaki's <c>EspeakG2P</c> (version None), used by KittenTTS for every language but English, Japanese and Chinese:
/// brackets are hidden from espeak as angle quotes, the text goes through <see cref="EspeakBackend"/>, two-letter
/// sequences joined by the tie become misaki's single symbols (<c>t^s</c> → <c>ʦ</c>, <c>a^ɪ</c> → <c>I</c>, …),
/// and the remaining ties and hyphens are removed.
/// </summary>
public sealed class EspeakG2P
{
    private static readonly (string From, string To)[] E2M = new (string, string)[]
    {
        ("a^ɪ", "I"), ("a^ʊ", "W"), ("d^z", "ʣ"), ("d^ʒ", "ʤ"), ("e^ɪ", "A"), ("o^ʊ", "O"), ("ə^ʊ", "Q"),
        ("s^s", "S"), ("t^s", "ʦ"), ("t^ʃ", "ʧ"), ("ɔ^ɪ", "Y"),
    }.OrderBy(kv => kv.Item1, StringComparer.Ordinal).ToArray();   // Python sorted(dict.items())

    private readonly EspeakBackend _backend;

    /// <param name="espeakLanguage">An espeak language code (e.g. <c>de</c>, <c>fr-fr</c>, <c>pt-br</c>).</param>
    public EspeakG2P(string espeakLanguage) => _backend = new EspeakBackend(espeakLanguage);

    public string Language => _backend.Language;

    public string Phonemize(string text)
    {
        text = text.Replace('«', '“').Replace('»', '”');
        text = text.Replace('(', '«').Replace(')', '»');
        // phonemizer.phonemize([text]) is empty for empty input
        string ps = EspeakBackend.PyStrip(_backend.Phonemize(text));
        foreach (var (from, to) in E2M) ps = ps.Replace(from, to);
        ps = ps.Replace("^", "").Replace("-", "");
        return ps.Replace('«', '(').Replace('»', ')');
    }
}
