using System.Globalization;
using System.IO.Compression;
using System.Reflection;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

using KittenTTS.Text;

namespace KittenTTS.Text.English;

/// <summary>A lexicon entry: one pronunciation, or pronunciations by part of speech (<c>DEFAULT</c>, <c>NOUN</c>,
/// <c>VERB</c>, <c>VBD</c>, ...; a value may be null).</summary>
internal sealed record LexEntry(string? Value, Dictionary<string, string?>? ByTag);

/// <summary>The decoding context misaki passes from right to left: whether the next word starts with a vowel (null at
/// a sentence end) and whether it is "to".</summary>
internal sealed record TokenContext(bool? FutureVowel = null, bool FutureTo = false);

/// <summary>
/// misaki's US English lexicon (<c>misaki/en.py</c>, <c>Lexicon</c>, version 0.9.4): the gold and silver dictionaries
/// with their case variants, special cases for function words, -s / -ed / -ing stemming, numbers through
/// <see cref="NumberWords"/>, currencies and stress rules. A line-by-line port; the dictionaries are misaki's own
/// (Apache-2.0), embedded gzipped.
/// </summary>
internal sealed class Lexicon
{
    public const char Primary = 'ˈ', Secondary = 'ˌ';
    private const string Stresses = "ˌˈ";
    internal static readonly HashSet<char> Vowels = [.. "AIOQWYaiuæɑɒɔəɛɜɪʊʌᵻ"];
    internal static readonly HashSet<char> Consonants = [.. "bdfhjklmnpstvwzðŋɡɹɾʃʒʤʧθ"];
    private static readonly HashSet<char> Diphthongs = [.. "AIOQWYʤʧ"];
    private static readonly HashSet<char> UsTaus = [.. "AIOWYiuæɑəɛɪɹʊʌ"];
    private static readonly HashSet<int> LexiconOrds = [39, 45, .. Enumerable.Range(65, 26), .. Enumerable.Range(97, 26)];
    private static readonly Dictionary<string, (string Unit, string Sub)> Currencies = new()
    {
        ["$"] = ("dollar", "cent"), ["£"] = ("pound", "pence"), ["€"] = ("euro", "cent"),
    };
    private static readonly HashSet<string> OrdinalSuffixes = ["st", "nd", "rd", "th"];
    private static readonly Dictionary<string, string> AddSymbols = new() { ["."] = "dot", ["/"] = "slash" };
    internal static readonly Dictionary<string, string> Symbols = new() { ["%"] = "percent", ["&"] = "and", ["+"] = "plus", ["@"] = "at" };
    private static readonly (float, float) CapStresses = (0.5f, 2f);

    private readonly Dictionary<string, LexEntry> _golds, _silvers;

    private Lexicon(Dictionary<string, LexEntry> golds, Dictionary<string, LexEntry> silvers)
    {
        _golds = golds;
        _silvers = silvers;
    }

    private static readonly Lazy<Lexicon> DefaultInstance = new(() => new Lexicon(Load("us_gold.json.gz"), Load("us_silver.json.gz")));

    public static Lexicon Default => DefaultInstance.Value;

    private static Dictionary<string, LexEntry> Load(string resource)
    {
        using var s = Assembly.GetExecutingAssembly().GetManifestResourceStream("KittenTTS.Text.English.Data." + resource)
                      ?? throw new InvalidOperationException($"The lexicon {resource} is missing from the assembly.");
        using var gz = new GZipStream(s, CompressionMode.Decompress);
        using var doc = JsonDocument.Parse(gz);
        var d = new Dictionary<string, LexEntry>(StringComparer.Ordinal);
        foreach (var p in doc.RootElement.EnumerateObject())
        {
            if (p.Value.ValueKind == JsonValueKind.String) d[p.Name] = new LexEntry(p.Value.GetString(), null);
            else
            {
                var byTag = new Dictionary<string, string?>(StringComparer.Ordinal);
                foreach (var t in p.Value.EnumerateObject()) byTag[t.Name] = t.Value.ValueKind == JsonValueKind.Null ? null : t.Value.GetString();
                d[p.Name] = new LexEntry(null, byTag);
            }
        }
        return GrowDictionary(d);
    }

    /// <summary>Adds the capitalized form of lowercase keys and the lowercase form of capitalized keys (the original
    /// entries win).</summary>
    private static Dictionary<string, LexEntry> GrowDictionary(Dictionary<string, LexEntry> d)
    {
        var e = new Dictionary<string, LexEntry>(StringComparer.Ordinal);
        foreach (var (k, v) in d)
        {
            if (k.Length < 2) continue;
            string lower = Py.Lower(k);
            if (k == lower)
            {
                string cap = Py.Capitalize(k);
                if (k != cap) e[cap] = v;
            }
            else if (k == Py.Capitalize(lower)) e[lower] = v;
        }
        foreach (var (k, v) in d) e[k] = v;
        return e;
    }

    public bool InGolds(string w) => _golds.ContainsKey(w);

    /// <summary>Whether a word's pronunciation depends on its part of speech (a heteronym such as "use", "record").</summary>
    public bool IsHeteronym(string w) =>
        (_golds.TryGetValue(w, out var g) || _golds.TryGetValue(Py.Lower(w), out g)) && g.ByTag != null
        || (_silvers.TryGetValue(w, out var s) || _silvers.TryGetValue(Py.Lower(w), out s)) && s.ByTag != null;
    public bool InSilvers(string w) => _silvers.ContainsKey(w);

    public static int StressWeight(string? ps) => ps == null ? 0 : ps.Sum(c => Diphthongs.Contains(c) ? 2 : 1);

    public static string? ApplyStress(string? ps, float? stress)
    {
        if (ps == null) return null;
        static string Restress(string ps)
        {
            var ips = ps.Select((c, i) => (Pos: (double)i, C: c)).ToList();
            var moves = new List<(int I, int J)>();
            for (int i = 0; i < ips.Count; i++)
            {
                if (!Stresses.Contains(ips[i].C)) continue;
                int j = -1;
                for (int k = i; k < ips.Count; k++) if (Vowels.Contains(ips[k].C)) { j = k; break; }
                if (j < 0) throw new InvalidOperationException("no vowel after a stress mark");
                moves.Add((i, j));
            }
            foreach (var (i, j) in moves) ips[i] = (j - 0.5, ips[i].C);
            return new string(ips.OrderBy(x => x.Pos).ThenBy(x => x.C).Select(x => x.C).ToArray());
        }
        if (stress == null) return ps;
        float s = stress.Value;
        bool hasStress = ps.Any(c => Stresses.Contains(c));
        if (s < -1) return ps.Replace(Primary.ToString(), "").Replace(Secondary.ToString(), "");
        if (s == -1 || ((s == 0 || s == -0.5f) && ps.Contains(Primary)))
            return ps.Replace(Secondary.ToString(), "").Replace(Primary, Secondary);
        if ((s == 0 || s == 0.5f || s == 1) && !hasStress)
        {
            if (!ps.Any(Vowels.Contains)) return ps;
            return Restress(Secondary + ps);
        }
        if (s >= 1 && !ps.Contains(Primary) && ps.Contains(Secondary)) return ps.Replace(Secondary, Primary);
        if (s > 1 && !hasStress)
        {
            if (!ps.Any(Vowels.Contains)) return ps;
            return Restress(Primary + ps);
        }
        return ps;
    }

    private static bool IsDigitString(string t) => t.Length > 0 && t.All(c => c is >= '0' and <= '9');

    public (string? Ps, int? Rating) GetNNP(string word)
    {
        var parts = new List<string>();
        foreach (char c in word)
        {
            if (!char.IsLetter(c)) continue;
            if (!_golds.TryGetValue(Py.Upper(c.ToString()), out var e) || e.Value == null) return (null, null);
            parts.Add(e.Value);
        }
        string? ps = ApplyStress(string.Concat(parts), 0);
        if (ps == null) return (null, null);
        int last = ps.LastIndexOf(Secondary);
        if (last >= 0) ps = ps[..last] + Primary + ps[(last + 1)..];
        return (ps, 3);
    }

    private (string?, int?) GetSpecialCase(string word, string? tag, float? stress, TokenContext ctx)
    {
        tag ??= "";
        if (tag == "ADD" && AddSymbols.TryGetValue(word, out var add)) return Lookup(add, null, -0.5f, ctx);
        if (Symbols.TryGetValue(word, out var sym)) return Lookup(sym, null, null, ctx);
        string stripped = word.Trim('.');
        if (stripped.Contains('.') && Py.IsAlpha(word.Replace(".", "")) && word.Split('.').Max(p => p.Length) < 3) return GetNNP(word);
        if (word is "a" or "A") return (tag == "DT" ? "ɐ" : "ˈA", 4);
        if (word is "am" or "Am" or "AM")
        {
            if (tag.StartsWith("NN")) return GetNNP(word);
            if (ctx.FutureVowel == null || word != "am" || (stress is float st && st > 0)) return (_golds["am"].Value, 4);
            return ("ɐm", 4);
        }
        if (word is "an" or "An" or "AN")
        {
            if (word == "AN" && tag.StartsWith("NN")) return GetNNP(word);
            return ("ɐn", 4);
        }
        if (word == "I" && tag == "PRP") return ($"{Secondary}I", 4);
        if (word is "by" or "By" or "BY" && ParentTag(tag) == "ADV") return ("bˈI", 4);
        if (word is "to" or "To" || (word == "TO" && tag is "TO" or "IN"))
            return (ctx.FutureVowel switch { null => _golds["to"].Value, false => "tə", true => "tʊ" }, 4);
        if (word is "in" or "In" || (word == "IN" && tag != "NNP"))
            return ((ctx.FutureVowel == null || tag != "IN" ? Primary.ToString() : "") + "ɪn", 4);
        if (word is "the" or "The" || (word == "THE" && tag == "DT")) return (ctx.FutureVowel == true ? "ði" : "ðə", 4);
        if (tag == "IN" && Regex.IsMatch(word, @"^(?i:vs\.?)$")) return Lookup("versus", null, null, ctx);
        if (word is "used" or "Used" or "USED")
        {
            var e = _golds["used"].ByTag!;
            if (tag is "VBD" or "JJ" && ctx.FutureTo) return (e["VBD"], 4);
            return (e["DEFAULT"], 4);
        }
        return (null, null);
    }

    public static string? ParentTag(string? tag)
    {
        if (tag == null) return null;
        if (tag.StartsWith("VB")) return "VERB";
        if (tag.StartsWith("NN")) return "NOUN";
        if (tag.StartsWith("ADV") || tag.StartsWith("RB")) return "ADV";
        if (tag.StartsWith("ADJ") || tag.StartsWith("JJ")) return "ADJ";
        return tag;
    }

    public bool IsKnown(string word, string? tag)
    {
        if (_golds.ContainsKey(word) || Symbols.ContainsKey(word) || _silvers.ContainsKey(word)) return true;
        if (!Py.IsAlpha(word) || !word.All(c => LexiconOrds.Contains(c))) return false;
        if (word.Length == 1) return true;
        if (word == Py.Upper(word) && _golds.ContainsKey(Py.Lower(word))) return true;
        return word[1..] == Py.Upper(word[1..]);
    }

    public (string?, int?) Lookup(string word, string? tag, float? stress, TokenContext? ctx)
    {
        bool? isNNP = null;
        if (word == Py.Upper(word) && !_golds.ContainsKey(word))
        {
            word = Py.Lower(word);
            isNNP = tag == "NNP";
        }
        LexEntry? entry = _golds.GetValueOrDefault(word);
        int rating = 4;
        if (entry == null && isNNP != true)
        {
            entry = _silvers.GetValueOrDefault(word);
            rating = 3;
        }
        string? ps = entry?.Value;
        if (entry?.ByTag is { } byTag)
        {
            if (ctx != null && ctx.FutureVowel == null && byTag.ContainsKey("None")) tag = "None";
            else if (tag == null || !byTag.ContainsKey(tag)) tag = ParentTag(tag);
            ps = tag != null && byTag.TryGetValue(tag, out var v) ? v : byTag["DEFAULT"];
        }
        if (ps == null || (isNNP == true && !ps.Contains(Primary)))
        {
            var (nnp, r) = GetNNP(word);
            if (nnp != null) return (nnp, r);
        }
        return (ApplyStress(ps, stress), rating);
    }

    private static string? S(string? stem)
    {
        if (string.IsNullOrEmpty(stem)) return null;
        char l = stem[^1];
        if ("ptkfθ".Contains(l)) return stem + "s";
        if ("szʃʒʧʤ".Contains(l)) return stem + "ᵻz";
        return stem + "z";
    }

    public (string?, int?) StemS(string word, string? tag, float? stress, TokenContext? ctx)
    {
        if (word.Length < 3 || !word.EndsWith('s')) return (null, null);
        string stem;
        if (!word.EndsWith("ss") && IsKnown(word[..^1], tag)) stem = word[..^1];
        else if ((word.EndsWith("'s") || (word.Length > 4 && word.EndsWith("es") && !word.EndsWith("ies"))) && IsKnown(word[..^2], tag)) stem = word[..^2];
        else if (word.Length > 4 && word.EndsWith("ies") && IsKnown(word[..^3] + "y", tag)) stem = word[..^3] + "y";
        else return (null, null);
        var (ps, rating) = Lookup(stem, tag, stress, ctx);
        return (S(ps), rating);
    }

    private static string? Ed(string? stem)
    {
        if (string.IsNullOrEmpty(stem)) return null;
        char l = stem[^1];
        if ("pkfθʃsʧ".Contains(l)) return stem + "t";
        if (l == 'd') return stem + "ᵻd";
        if (l != 't') return stem + "d";
        if (stem.Length < 2) return stem + "ɪd";
        if (UsTaus.Contains(stem[^2])) return stem[..^1] + "ɾᵻd";
        return stem + "ᵻd";
    }

    public (string?, int?) StemEd(string word, string? tag, float? stress, TokenContext? ctx)
    {
        if (word.Length < 4 || !word.EndsWith('d')) return (null, null);
        string stem;
        if (!word.EndsWith("dd") && IsKnown(word[..^1], tag)) stem = word[..^1];
        else if (word.Length > 4 && word.EndsWith("ed") && !word.EndsWith("eed") && IsKnown(word[..^2], tag)) stem = word[..^2];
        else return (null, null);
        var (ps, rating) = Lookup(stem, tag, stress, ctx);
        return (Ed(ps), rating);
    }

    private static string? Ing(string? stem)
    {
        if (string.IsNullOrEmpty(stem)) return null;
        if (stem.Length > 1 && stem[^1] == 't' && UsTaus.Contains(stem[^2])) return stem[..^1] + "ɾɪŋ";
        return stem + "ɪŋ";
    }

    private static readonly Regex DoubledIng = new(@"([bcdgklmnprstvxz])\1ing$|cking$", RegexOptions.CultureInvariant);

    public (string?, int?) StemIng(string word, string? tag, float? stress, TokenContext? ctx)
    {
        if (word.Length < 5 || !word.EndsWith("ing")) return (null, null);
        string stem;
        if (word.Length > 5 && IsKnown(word[..^3], tag)) stem = word[..^3];
        else if (IsKnown(word[..^3] + "e", tag)) stem = word[..^3] + "e";
        else if (word.Length > 5 && DoubledIng.IsMatch(word) && IsKnown(word[..^4], tag)) stem = word[..^4];
        else return (null, null);
        var (ps, rating) = Lookup(stem, tag, stress, ctx);
        return (Ing(ps), rating);
    }

    public (string?, int?) GetWord(string word, string? tag, float? stress, TokenContext ctx)
    {
        var sc = GetSpecialCase(word, tag, stress, ctx);
        if (sc.Item1 != null) return sc;
        string wl = Py.Lower(word);
        if (word.Length > 1 && Py.IsAlpha(word.Replace("'", "")) && word != wl && (tag != "NNP" || word.Length > 7)
            && !_golds.ContainsKey(word) && !_silvers.ContainsKey(word)
            && (word == Py.Upper(word) || word[1..] == Py.Lower(word[1..]))
            && (_golds.ContainsKey(wl) || _silvers.ContainsKey(wl) || StemS(wl, tag, stress, ctx).Item1 != null
                || StemEd(wl, tag, stress, ctx).Item1 != null || StemIng(wl, tag, stress, ctx).Item1 != null))
            word = wl;
        if (IsKnown(word, tag)) return Lookup(word, tag, stress, ctx);
        if (word.EndsWith("s'") && IsKnown(word[..^2] + "'s", tag)) return Lookup(word[..^2] + "'s", tag, stress, ctx);
        if (word.EndsWith('\'') && IsKnown(word[..^1], tag)) return Lookup(word[..^1], tag, stress, ctx);
        var s = StemS(word, tag, stress, ctx);
        if (s.Item1 != null) return s;
        var ed = StemEd(word, tag, stress, ctx);
        if (ed.Item1 != null) return ed;
        var ing = StemIng(word, tag, stress ?? 0.5f, ctx);
        if (ing.Item1 != null) return ing;
        return (null, null);
    }

    private static bool IsCurrency(string word)
    {
        if (!word.Contains('.')) return true;
        if (word.Count(c => c == '.') > 1) return false;
        string cents = word.Split('.')[1];
        return cents.Length < 3;   // (misaki's set(cents) == {0} compares characters with an int: never true)
    }

    private static readonly Regex SuffixRe = new("[a-z']+$", RegexOptions.CultureInvariant);

    public (string?, int?) GetNumber(string word, string? currency, bool isHead, string numFlags)
    {
        var sm = SuffixRe.Match(word);
        string? suffix = sm.Success ? sm.Value : null;
        if (suffix != null) word = word[..^suffix.Length];
        var result = new List<(string? Ps, int? Rating)>();
        if (word.StartsWith('-'))
        {
            result.Add(Lookup("minus", null, null, null));
            word = word[1..];
        }
        void ExtendNum(string num, bool first = true, bool escape = false)
        {
            string text = escape ? num : NumberWords.Cardinal(System.Numerics.BigInteger.Parse(num, CultureInfo.InvariantCulture));
            var splits = Regex.Split(text, "[^a-z]+");
            for (int i = 0; i < splits.Length; i++)
            {
                string w = splits[i];
                if (w != "and" || numFlags.Contains('&'))
                {
                    if (first && i == 0 && splits.Length > 1 && w == "one" && numFlags.Contains('a')) result.Add(("ə", 4));
                    else result.Add(Lookup(w, null, w == "point" ? -2 : null, null));
                }
                else if (w == "and" && numFlags.Contains('n') && result.Count > 0)
                    result[^1] = (result[^1].Ps + "ən", result[^1].Rating);
            }
        }
        if (IsDigitString(word) && suffix != null && OrdinalSuffixes.Contains(suffix))
            ExtendNum(NumberWords.Ordinal(System.Numerics.BigInteger.Parse(word, CultureInfo.InvariantCulture)), escape: true);
        else if (result.Count == 0 && word.Length == 4 && (currency == null || !Currencies.ContainsKey(currency)) && IsDigitString(word))
            ExtendNum(NumberWords.Year(long.Parse(word, CultureInfo.InvariantCulture)), escape: true);
        else if (!isHead && !word.Contains('.'))
        {
            string num = word.Replace(",", "");
            if (num[0] == '0' || num.Length > 3) foreach (char n in num) ExtendNum(n.ToString(), first: false);
            else if (num.Length == 3 && !num.EndsWith("00"))
            {
                ExtendNum(num[0].ToString());
                if (num[1] == '0')
                {
                    result.Add(Lookup("O", null, -2, null));
                    ExtendNum(num[2].ToString(), first: false);
                }
                else ExtendNum(num[1..], first: false);
            }
            else ExtendNum(num);
        }
        else if (word.Count(c => c == '.') > 1 || !isHead)
        {
            bool first = true;
            foreach (var num in word.Replace(",", "").Split('.'))
            {
                if (num.Length == 0) { }
                else if (num[0] == '0' || (num.Length != 2 && num[1..].Any(n => n != '0'))) foreach (char n in num) ExtendNum(n.ToString(), first: false);
                else ExtendNum(num, first: first);
                first = false;
            }
        }
        else if (currency != null && Currencies.TryGetValue(currency, out var units) && IsCurrency(word))
        {
            var parts = word.Replace(",", "").Split('.');
            var pairs = new List<(long Num, string Unit)>();
            string[] unitNames = [units.Unit, units.Sub];
            for (int i = 0; i < Math.Min(parts.Length, 2); i++)
                pairs.Add((parts[i].Length > 0 ? long.Parse(parts[i], CultureInfo.InvariantCulture) : 0, unitNames[i]));
            if (pairs.Count > 1)
            {
                if (pairs[1].Num == 0) pairs = pairs.Take(1).ToList();
                else if (pairs[0].Num == 0) pairs = pairs.Skip(1).ToList();
            }
            for (int i = 0; i < pairs.Count; i++)
            {
                var (num, unit) = pairs[i];
                if (i > 0) result.Add(Lookup("and", null, null, null));
                ExtendNum(num.ToString(CultureInfo.InvariantCulture), first: i == 0);
                result.Add(Math.Abs(num) != 1 && unit != "pence" ? StemS(unit + "s", null, null, null) : Lookup(unit, null, null, null));
            }
        }
        else
        {
            if (IsDigitString(word)) word = NumberWords.Cardinal(System.Numerics.BigInteger.Parse(word, CultureInfo.InvariantCulture));
            else if (!word.Contains('.'))
            {
                var n = System.Numerics.BigInteger.Parse(word.Replace(",", ""), CultureInfo.InvariantCulture);
                word = suffix != null && OrdinalSuffixes.Contains(suffix) ? NumberWords.Ordinal(n) : NumberWords.Cardinal(n);
            }
            else
            {
                word = word.Replace(",", "");
                if (word[0] == '.') word = "point " + string.Join(' ', word[1..].Select(c => NumberWords.Cardinal(c - '0')));
                else word = NumberWords.Float(word);
            }
            ExtendNum(word, escape: true);
        }
        if (result.Count == 0) return (null, null);
        string ps = string.Join(' ', result.Select(r => r.Ps));
        int rating = result.Min(r => r.Rating ?? int.MaxValue);
        if (suffix is "s" or "'s") return (S(ps), rating);
        if (suffix is "ed" or "'d") return (Ed(ps), rating);
        if (suffix == "ing") return (Ing(ps), rating);
        return (ps, rating);
    }

    private string AppendCurrency(string ps, string? currency)
    {
        if (string.IsNullOrEmpty(currency)) return ps;
        if (!Currencies.TryGetValue(currency, out var c)) return ps;
        string? cur = StemS(c.Unit + "s", null, null, null).Item1;
        return cur != null ? $"{ps} {cur}" : ps;
    }

    private static string NumericIfNeeded(char c)
    {
        int d = CharUnicodeInfo.GetDigitValue(c);
        if (d < 0) return c.ToString();
        double n = CharUnicodeInfo.GetNumericValue(c);
        return n == Math.Floor(n) ? ((int)n).ToString(CultureInfo.InvariantCulture) : c.ToString();
    }

    public static bool IsNumber(string word, bool isHead)
    {
        if (!word.Any(c => c is >= '0' and <= '9')) return false;
        foreach (var s in new[] { "ing", "'d", "ed", "'s", "st", "nd", "rd", "th", "s" })
            if (word.EndsWith(s, StringComparison.Ordinal)) { word = word[..^s.Length]; break; }
        for (int i = 0; i < word.Length; i++)
        {
            char c = word[i];
            if (!(c is >= '0' and <= '9' || c is ',' or '.' || (isHead && i == 0 && c == '-'))) return false;
        }
        return true;
    }

    /// <summary>Pronounces one token (<c>Lexicon.__call__</c>).</summary>
    public (string?, int?) Pronounce(MToken tk, TokenContext ctx)
    {
        string word = (tk.Alias ?? tk.Text).Replace('‘', '\'').Replace('’', '\'');
        word = UnicodeNormalizer.Normalize(word, NormalizationForm.FormKC);
        word = string.Concat(word.Select(NumericIfNeeded));
        float? stress = word == Py.Lower(word) ? null : word == Py.Upper(word) ? CapStresses.Item2 : CapStresses.Item1;
        var (ps, rating) = GetWord(word, tk.Tag, stress, ctx);
        if (ps != null) return (ApplyStress(AppendCurrency(ps, tk.Currency), tk.Stress), rating);
        if (IsNumber(word, tk.IsHead))
        {
            var (nps, nr) = GetNumber(word, tk.Currency, tk.IsHead, tk.NumFlags);
            return (ApplyStress(nps, tk.Stress), nr);
        }
        return (null, null);
    }
}

/// <summary>Python string semantics the port relies on.</summary>
internal static class Py
{
    public static string Lower(string s) => s.ToLowerInvariant();
    public static string Upper(string s) => s.ToUpperInvariant();

    /// <summary><c>str.capitalize()</c>: first character upper case, the rest lower case.</summary>
    public static string Capitalize(string s) => s.Length == 0 ? s : char.ToUpperInvariant(s[0]) + s[1..].ToLowerInvariant();

    /// <summary><c>str.isalpha()</c>: non-empty and every character a letter.</summary>
    public static bool IsAlpha(string s) => s.Length > 0 && s.All(char.IsLetter);
}
