using System.Reflection;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace KittenTTS.Text.English;

/// <summary>A token of the English tokenizer: its text and whether whitespace follows it.</summary>
public sealed record SpacyToken(string Text, string Whitespace);

/// <summary>
/// spaCy's rule-based English tokenizer (<c>en_core_web_sm</c>, spaCy 3.8), which misaki runs before its lexicon:
/// split on spaces, then per chunk the special cases (contractions such as <c>can't → ca + n't</c>, abbreviations),
/// prefix and suffix stripping, the URL pattern and infix splitting, in the order of spaCy's
/// <c>Tokenizer._split_affixes</c> / <c>_attach_tokens</c>. The rules are spaCy's own, exported to
/// <c>Data/spacy_en_tokenizer.json</c> by <c>scripts/export_english_data.py</c>; character ranges outside the Basic
/// Multilingual Plane (emoji, rare CJK) are dropped from the patterns, as .NET regular expressions cannot express them
/// in character classes.
/// </summary>
public sealed partial class SpacyTokenizer
{
    private readonly Dictionary<string, string[]> _special;
    private readonly Regex _prefix, _suffix, _infix;
    private readonly Regex? _url, _tokenMatch;

    private SpacyTokenizer(Stream json)
    {
        using var doc = JsonDocument.Parse(json);
        var r = doc.RootElement;
        _special = new Dictionary<string, string[]>(StringComparer.Ordinal);
        foreach (var p in r.GetProperty("special_cases").EnumerateObject())
            _special[p.Name] = p.Value.EnumerateArray().Select(e => e.GetString()!).ToArray();
        _prefix = Compile(r.GetProperty("prefix").GetString()!);
        _suffix = Compile(r.GetProperty("suffix").GetString()!);
        _infix = Compile(r.GetProperty("infix").GetString()!);
        if (r.TryGetProperty("url", out var u) && u.ValueKind == JsonValueKind.String) _url = Compile(u.GetString()!);
        if (r.TryGetProperty("token_match", out var t) && t.ValueKind == JsonValueKind.String) _tokenMatch = Compile(t.GetString()!);
    }

    private static readonly Lazy<SpacyTokenizer> DefaultInstance = new(() =>
    {
        using var s = Assembly.GetExecutingAssembly().GetManifestResourceStream("KittenTTS.Text.English.Data.spacy_en_tokenizer.json")
                      ?? throw new InvalidOperationException("The spaCy tokenizer data is missing from the assembly.");
        return new SpacyTokenizer(s);
    });

    /// <summary>The tokenizer with spaCy's English rules (embedded in the assembly).</summary>
    public static SpacyTokenizer Default => DefaultInstance.Value;

    [GeneratedRegex(@"\\U([0-9A-Fa-f]{8})(?:-\\U([0-9A-Fa-f]{8}))?")]
    private static partial Regex AstralEscape();

    /// <summary>Translates a Python <c>re</c> pattern to .NET: drops <c>(?u)</c> and astral-plane escapes.</summary>
    internal static Regex Compile(string pattern)
    {
        pattern = pattern.Replace("(?u)", "");
        pattern = AstralEscape().Replace(pattern, m =>
        {
            int a = Convert.ToInt32(m.Groups[1].Value, 16);
            if (a <= 0xFFFF) return m.Groups[2].Success ? $"\\u{a:X4}-\\u{Convert.ToInt32(m.Groups[2].Value, 16):X4}" : $"\\u{a:X4}";
            return "";
        });
        pattern = pattern.Replace("[]", "[^\\s\\S]").Replace("[^]", "[\\s\\S]");
        return new Regex(pattern, RegexOptions.CultureInvariant | RegexOptions.Compiled);
    }

    /// <summary>Tokenizes <paramref name="text"/> like <c>nlp.tokenizer(text)</c>.</summary>
    public List<SpacyToken> Tokenize(string text)
    {
        var tokens = new List<SpacyToken>();
        if (string.IsNullOrEmpty(text)) return tokens;
        // spaCy: chunks separated by single spaces; any other whitespace run becomes a token of its own
        int i = 0;
        int n = text.Length;
        bool inWs = char.IsWhiteSpace(text[0]);
        int start = 0;
        for (i = 0; i <= n; i++)
        {
            bool end = i == n;
            bool ws = !end && char.IsWhiteSpace(text[i]);
            if (end || ws != inWs)
            {
                if (start < i)
                {
                    string span = text[start..i];
                    if (inWs)
                    {
                        // a single space after a token is that token's whitespace, the rest is a whitespace token
                        if (span == " " && tokens.Count > 0 && tokens[^1].Whitespace.Length == 0)
                            tokens[^1] = tokens[^1] with { Whitespace = " " };
                        else
                        {
                            if (tokens.Count > 0 && tokens[^1].Whitespace.Length == 0 && span[0] == ' ')
                            {
                                tokens[^1] = tokens[^1] with { Whitespace = " " };
                                span = span[1..];
                            }
                            if (span.Length > 0) tokens.Add(new SpacyToken(span, ""));
                        }
                    }
                    else TokenizeSpan(span, tokens);
                }
                start = i;
                inWs = ws;
            }
        }
        return tokens;
    }

    private bool IsSpecial(string s) => _special.ContainsKey(s);

    private void TokenizeSpan(string span, List<SpacyToken> tokens)
    {
        if (_special.TryGetValue(span, out var sc))
        {
            foreach (var t in sc) tokens.Add(new SpacyToken(t, ""));
            return;
        }
        var prefixes = new List<string>();
        var suffixes = new List<string>();
        string rest = SplitAffixes(span, prefixes, suffixes);
        foreach (var p in prefixes) tokens.Add(new SpacyToken(p, ""));
        if (rest.Length > 0)
        {
            if (_special.TryGetValue(rest, out var sc2)) foreach (var t in sc2) tokens.Add(new SpacyToken(t, ""));
            else if (_tokenMatch != null && _tokenMatch.IsMatch(rest)) tokens.Add(new SpacyToken(rest, ""));
            else if (_url != null && _url.IsMatch(rest)) tokens.Add(new SpacyToken(rest, ""));
            else
            {
                var matches = _infix.Matches(rest);
                if (matches.Count == 0) tokens.Add(new SpacyToken(rest, ""));
                else
                {
                    int start = 0;
                    foreach (Match m in matches)
                    {
                        int s = m.Index, e = m.Index + m.Length;
                        if (s == 0) continue;
                        if (s != start) tokens.Add(new SpacyToken(rest[start..s], ""));
                        if (s != e) tokens.Add(new SpacyToken(rest[s..e], ""));
                        start = e;
                    }
                    if (start < rest.Length) tokens.Add(new SpacyToken(rest[start..], ""));
                }
            }
        }
        for (int k = suffixes.Count - 1; k >= 0; k--) tokens.Add(new SpacyToken(suffixes[k], ""));
    }

    private int FindPrefix(string s)
    {
        var m = _prefix.Match(s);
        return m.Success && m.Index == 0 ? m.Length : 0;
    }

    private int FindSuffix(string s)
    {
        var m = _suffix.Match(s);
        return m.Success && m.Index + m.Length == s.Length ? m.Length : 0;
    }

    private string SplitAffixes(string s, List<string> prefixes, List<string> suffixes)
    {
        int lastSize = -1;
        while (s.Length > 0 && s.Length != lastSize)
        {
            if (_tokenMatch != null && _tokenMatch.IsMatch(s)) break;
            if (IsSpecial(s)) break;
            lastSize = s.Length;
            int pre = FindPrefix(s);
            string prefix = "", minusPre = "";
            if (pre != 0)
            {
                prefix = s[..pre];
                minusPre = s[pre..];
                if (minusPre.Length > 0 && IsSpecial(minusPre))
                {
                    s = minusPre;
                    prefixes.Add(prefix);
                    break;
                }
            }
            int suf = FindSuffix(s[pre..]);
            string suffix = "", minusSuf = "";
            if (suf != 0)
            {
                suffix = s[^suf..];
                minusSuf = s[..^suf];
                if (minusSuf.Length > 0 && IsSpecial(minusSuf))
                {
                    s = minusSuf;
                    suffixes.Add(suffix);
                    break;
                }
            }
            if (pre != 0 && suf != 0 && pre + suf <= s.Length)
            {
                s = s[pre..^suf];
                prefixes.Add(prefix);
                suffixes.Add(suffix);
            }
            else if (pre != 0)
            {
                s = minusPre;
                prefixes.Add(prefix);
            }
            else if (suf != 0)
            {
                s = minusSuf;
                suffixes.Add(suffix);
            }
            if (s.Length > 0 && IsSpecial(s)) break;
        }
        return s;
    }
}
