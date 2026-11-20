using System.Globalization;
using System.Numerics;
using System.Text;
using System.Text.RegularExpressions;

using KittenTTS.Text;

namespace KittenTTS.Text.English;

/// <summary>
/// English text normalisation before G2P: a port of the training code's <c>tts_norm.normalize</c>
/// (<c>.reference/kittenml/kittentts2_tiny/frontend/tts_norm.py</c>). Dates, times, phone numbers, money, versions,
/// fractions, magnitudes, units, abbreviations, roman numerals, ordinals, ranges, symbols and numbers become words;
/// sentence punctuation (. , ! ? ; and dashes) is kept for prosody. The rules run in the original order.
/// </summary>
public static partial class TextNormalizer
{
    private const RegexOptions Opt = RegexOptions.CultureInvariant;
    private const RegexOptions OptI = RegexOptions.CultureInvariant | RegexOptions.IgnoreCase;

    private static readonly string[] Months = ["January", "February", "March", "April", "May", "June", "July",
                                               "August", "September", "October", "November", "December"];

    private static readonly Dictionary<string, string> Units = new()
    {
        ["km"] = "kilometres", ["mi"] = "miles", ["cm"] = "centimetres", ["mm"] = "millimetres",
        ["kg"] = "kilograms", ["lb"] = "pounds", ["lbs"] = "pounds", ["oz"] = "ounces",
        ["ml"] = "millilitres", ["l"] = "litres", ["ft"] = "feet", ["in"] = "inches",
        ["hr"] = "hours", ["hrs"] = "hours", ["min"] = "minutes", ["mins"] = "minutes",
        ["sec"] = "seconds", ["secs"] = "seconds", ["ms"] = "milliseconds",
        ["gb"] = "gigabytes", ["mb"] = "megabytes", ["kb"] = "kilobytes", ["tb"] = "terabytes",
        ["hz"] = "hertz", ["khz"] = "kilohertz", ["mhz"] = "megahertz", ["ghz"] = "gigahertz",
        ["kw"] = "kilowatts", ["mph"] = "miles per hour", ["kph"] = "kilometres per hour",
    };

    private static readonly Dictionary<string, string> Mag = new() { ["k"] = "thousand", ["m"] = "million", ["b"] = "billion", ["t"] = "trillion" };

    private static readonly Dictionary<(int, int), string> Fractions = new()
    {
        [(1, 2)] = "a half", [(1, 3)] = "a third", [(2, 3)] = "two thirds",
        [(1, 4)] = "a quarter", [(3, 4)] = "three quarters", [(1, 8)] = "an eighth",
    };

    // dict order matters: the Python loops apply these in insertion order
    private static readonly (string, string)[] Abbrev =
    [
        ("mr", "Mister"), ("mrs", "Missus"), ("ms", "Miss"), ("dr", "Doctor"),
        ("prof", "Professor"), ("rev", "Reverend"), ("hon", "Honourable"),
        ("ext", "extension"), ("approx", "approximately"), ("vs", "versus"),
        ("est", "estimated"), ("dept", "department"), ("univ", "university"),
        ("fig", "figure"), ("eq", "equation"), ("ref", "reference"), ("vol", "volume"),
        ("incl", "including"), ("excl", "excluding"), ("min", "minimum"),
        ("max", "maximum"), ("avg", "average"), ("mt", "mount"), ("ft", "fort"),
        ("appt", "appointment"), ("pp", "pages"), ("jr", "Junior"), ("sr", "Senior"),
        ("inc", "Incorporated"), ("ltd", "Limited"), ("corp", "Corporation"),
        ("mins", "minutes"), ("hrs", "hours"), ("wks", "weeks"), ("yrs", "years"),
    ];

    private static readonly (string, string)[] DaysAndMonths =
    [
        ("Mon", "Monday"), ("Tue", "Tuesday"), ("Tues", "Tuesday"), ("Wed", "Wednesday"), ("Weds", "Wednesday"),
        ("Thu", "Thursday"), ("Thur", "Thursday"), ("Thurs", "Thursday"), ("Fri", "Friday"), ("Sat", "Saturday"),
        ("Sun", "Sunday"),
        ("Jan", "January"), ("Feb", "February"), ("Mar", "March"), ("Apr", "April"), ("Jun", "June"), ("Jul", "July"),
        ("Aug", "August"), ("Sep", "September"), ("Sept", "September"), ("Oct", "October"), ("Nov", "November"),
        ("Dec", "December"),
    ];

    private static readonly (string, string)[] Symbols =
    [
        ("%", " percent "), ("&", " and "), ("@", " at "), ("+", " plus "), ("=", " equals "), ("×", " times "),
        ("÷", " divided by "), ("<", " less than "), (">", " greater than "), ("~", " about "), ("°", " degrees "),
        ("€", " euros "), ("£", " pounds "), ("¥", " yen "), ("©", " copyright "), ("™", " trademark "), ("§", " section "),
    ];

    private static readonly Dictionary<string, int> Roman = new()
    {
        ["i"] = 1, ["ii"] = 2, ["iii"] = 3, ["iv"] = 4, ["v"] = 5, ["vi"] = 6, ["vii"] = 7,
        ["viii"] = 8, ["ix"] = 9, ["x"] = 10, ["xi"] = 11, ["xii"] = 12,
    };

    private const string RomanCtx = "(?:chapter|section|part|volume|book|act|phase|appendix|world war|type|class|figure|table|step|stage)";

    private static string Card(BigInteger n)
    {
        try { return NumberWords.Cardinal(n); }
        catch { return n.ToString(CultureInfo.InvariantCulture); }
    }

    private static string Card(string digits) => Card(BigInteger.Parse(digits, CultureInfo.InvariantCulture));

    private static string Year(int y) => (1100 <= y && y <= 1999) || (2000 <= y && y <= 2099) ? NumberWords.Year(y) : Card(y);

    private static int Int(string s) => int.Parse(s, CultureInfo.InvariantCulture);

    private static string Dates(string s)
    {
        s = Regex.Replace(s, @"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", m =>
        {
            int y = Int(m.Groups[1].Value), mo = Int(m.Groups[2].Value), d = Int(m.Groups[3].Value);
            if (!(1 <= mo && mo <= 12 && 1 <= d && d <= 31)) return m.Value;
            return $"{Months[mo - 1]} {NumberWords.Ordinal(d)}, {Year(y)}";
        }, Opt);
        s = Regex.Replace(s, @"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b", m =>
        {
            int a = Int(m.Groups[1].Value), b = Int(m.Groups[2].Value), y = Int(m.Groups[3].Value);
            if (!(1 <= a && a <= 12 && 1 <= b && b <= 31)) return m.Value;
            if (y < 100) y += y < 50 ? 2000 : 1900;
            return $"{Months[a - 1]} {NumberWords.Ordinal(b)}, {Year(y)}";
        }, Opt);
        s = Regex.Replace(s, @"\b(" + string.Join('|', Months) + @")\s+(\d{1,2}),?\s+(\d{4})\b",
                          m => $"{m.Groups[1].Value} {NumberWords.Ordinal(Int(m.Groups[2].Value))}, {Year(Int(m.Groups[3].Value))}", OptI);
        return s;
    }

    /// <summary>Puts back a sentence-final period swallowed by the optional "." of an am/pm group.</summary>
    private static string KeepFinalDot(Match m, string input, string output)
    {
        if (!m.Value.EndsWith('.')) return output;
        string rest = input[(m.Index + m.Length)..];
        string stripped = PyStrip(rest);
        if (stripped.Length == 0 || (rest.Length > 0 && char.IsWhiteSpace(rest[0]) && char.IsUpper(stripped[0]))) return output + ".";
        return output;
    }

    private static string Times(string s)
    {
        string input = s;
        return Regex.Replace(s, @"\b(\d{1,2}):(\d{2})(?:\s*([ap]\.?m\.?))?", m =>
        {
            int h = Int(m.Groups[1].Value), mi = Int(m.Groups[2].Value);
            string ampm = m.Groups[3].Success ? m.Groups[3].Value.ToLowerInvariant() : "";
            if (h > 23 || mi > 59) return m.Value;
            string suffix = "";
            if (ampm.Length > 0) suffix = ampm.StartsWith('a') ? " AM" : " PM";
            else if (h == 0) { h = 12; suffix = " AM"; }
            string spoken;
            if (mi == 0)
                spoken = suffix.Length == 0 ? $"{Card(h <= 12 || ampm.Length > 0 ? h : h)} o'clock" : Card(h <= 12 ? h : h - 12);
            else if (mi < 10)
                spoken = $"{Card(h <= 12 || suffix.Length == 0 ? h : h - 12)} oh {Card(mi)}";
            else
                spoken = $"{Card(h <= 12 || suffix.Length == 0 ? h : h - 12)} {Card(mi)}";
            return KeepFinalDot(m, input, spoken + suffix);
        }, OptI);
    }

    private static string BareHour(string s)
    {
        string input = s;
        return Regex.Replace(s, @"\b(\d{1,2})\s*([ap])\.?m\.?(?![a-z])", m =>
        {
            int h = Int(m.Groups[1].Value);
            if (!(1 <= h && h <= 12)) return m.Value;
            return KeepFinalDot(m, input, $"{Card(h)} {(m.Groups[2].Value.ToLowerInvariant().StartsWith('a') ? "AM" : "PM")}");
        }, OptI);
    }

    private static string Phones(string s) =>
        Regex.Replace(s, @"\b(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]\d{3}[-.\s]?\d{0,4}\b", m =>
        {
            string digits = Regex.Replace(m.Value, @"\D", "");
            if (!(7 <= digits.Length && digits.Length <= 11)) return m.Value;
            var groups = new List<string>();
            int i = 0;
            if (digits.Length == 11 && digits[0] == '1') { groups.Add("one"); i = 1; }
            foreach (var (a, b) in new[] { (i, i + 3), (i + 3, i + 6), (i + 6, digits.Length) })
                if (a < digits.Length)
                    groups.Add(string.Join(' ', digits[a..Math.Min(b, digits.Length)].Select(c => Card(c - '0'))));
            return string.Join(", ", groups.Where(g => g.Length > 0));
        }, Opt);

    private static string Money(string s) =>
        Regex.Replace(s, @"\$\s?(\d[\d,]*(?:\.\d+)?)(?:\s?([KkMmBbTt])\b)?", m =>
        {
            string amt = m.Groups[1].Value.Replace(",", ""), suf = m.Groups[2].Success ? m.Groups[2].Value.ToLowerInvariant() : "";
            if (suf.Length > 0)
            {
                string whole;
                if (amt.Contains('.'))
                {
                    double num = double.Parse(amt, CultureInfo.InvariantCulture);
                    whole = num == Math.Floor(num)
                        ? Card(new BigInteger(num))
                        : $"{Card(new BigInteger(Math.Truncate(num)))} point {string.Join(' ', amt.Split('.')[1].Select(c => Card(c - '0')))}";
                }
                else whole = Card(amt);
                return $"{whole} {Mag[suf]} dollars";
            }
            if (amt.Contains('.'))
            {
                var parts = amt.Split('.', 2);
                string c = (parts[1] + "00")[..2];
                var d = BigInteger.Parse(parts[0], CultureInfo.InvariantCulture);
                string dollars = $"{Card(d)} dollar{(d == 1 ? "" : "s")}";
                int ci = Int(c);
                return ci != 0 ? $"{dollars} and {Card(ci)} cent{(ci == 1 ? "" : "s")}" : dollars;
            }
            var n = BigInteger.Parse(amt, CultureInfo.InvariantCulture);
            return $"{Card(n)} dollar{(n == 1 ? "" : "s")}";
        }, Opt);

    private static string Magnitudes(string s) =>
        Regex.Replace(s, @"\b(\d[\d,]*(?:\.\d+)?)\s*([KMBT])\b", m => $"{DecimalWords(m.Groups[1].Value)} {Mag[m.Groups[2].Value.ToLowerInvariant()]}", Opt);

    private static string DecimalWords(string tok)
    {
        tok = tok.Replace(",", "");
        if (!tok.Any(char.IsDigit)) return tok;
        if (tok.Contains('.'))
        {
            var parts = tok.Split('.', 2);
            return $"{Card(parts[0].Length == 0 ? "0" : parts[0])} point {string.Join(' ', parts[1].Select(c => Card(c - '0')))}";
        }
        return Card(tok);
    }

    private static string FractionsRule(string s) =>
        Regex.Replace(s, @"\b(\d{1,3})/(\d{1,3})\b", m =>
        {
            int a = Int(m.Groups[1].Value), b = Int(m.Groups[2].Value);
            if (Fractions.TryGetValue((a, b), out var f)) return f;
            if (b == 0) return m.Value;
            return $"{Card(a)} {NumberWords.Ordinal(b)}{(a != 1 ? "s" : "")}";
        }, Opt);

    private static string Versions(string s) =>
        Regex.Replace(s, @"\b(\d+)\.(\d+)\.(\d+)\b", m => string.Join(" point ", new[] { m.Groups[1], m.Groups[2], m.Groups[3] }.Select(g => Card(g.Value))), Opt);

    private static string UnitsRule(string s)
    {
        string pat = @"\b(\d[\d,]*(?:\.\d+)?)\s?(" + string.Join('|', Units.Keys.OrderByDescending(k => k.Length)) + @")\b";
        s = Regex.Replace(s, pat, m =>
        {
            string n = m.Groups[1].Value, unit = m.Groups[2].Value.ToLowerInvariant();
            string val = DecimalWords(n);
            string word = Units[unit];
            if (n.Replace(",", "") is "1" or "1.0") word = word.EndsWith('s') ? word.TrimEnd('s') : word;
            return $"{val} {word}";
        }, OptI);
        s = Regex.Replace(s, @"([-−]?[\d.]+)\s?°\s?([CF])\b", m =>
        {
            string raw = m.Groups[1].Value, unit = m.Groups[2].Value.ToUpperInvariant();
            bool neg = raw[0] is '-' or '−';
            string val = DecimalWords(neg ? raw[1..] : raw);
            return $"{(neg ? "minus " : "")}{val} degrees {(unit == "C" ? "Celsius" : "Fahrenheit")}";
        }, Opt);
        return s;
    }

    private static string AbbrevRule(string s)
    {
        s = Regex.Replace(s, @"\bSt\.\s+(?=[A-Z])", "Saint ", Opt);
        s = Regex.Replace(s, @"\bSt\.(?=\s*$)", "Street.", Opt);
        s = Regex.Replace(s, "\\bSt\\.(?=\\s*[\"'\\)])", "Street.", Opt);
        s = Regex.Replace(s, @"\bSt\.(?!\s+[A-Z])", "Street", Opt);
        s = Regex.Replace(s, @"\bAve\.", "Avenue", Opt);
        s = Regex.Replace(s, @"\bRd\.", "Road", Opt);
        s = Regex.Replace(s, @"\bBlvd\.", "Boulevard", Opt);
        s = Regex.Replace(s, @"\bNo\.\s*(?=\d)", "number ", Opt);
        s = Regex.Replace(s, @"\bNos\.\s*(?=\d)", "numbers ", Opt);
        foreach (var (k, v) in DaysAndMonths) s = Regex.Replace(s, $@"\b{k}\.", v, Opt);
        foreach (var (k, v) in Abbrev) s = Regex.Replace(s, $@"\b{k}\.", v, OptI);
        return s;
    }

    private static string Romans(string s) =>
        Regex.Replace(s, $@"\b({RomanCtx})\s+([IVX]{{1,4}})\b", m =>
            Roman.TryGetValue(m.Groups[2].Value.ToLowerInvariant(), out int v) ? $"{m.Groups[1].Value} {Card(v)}" : m.Value, OptI);

    private static string Ordinals(string s) =>
        Regex.Replace(s, @"\b(\d+)(?:st|nd|rd|th)\b", m => NumberWords.Ordinal(BigInteger.Parse(m.Groups[1].Value, CultureInfo.InvariantCulture)), OptI);

    private static string Ranges(string s) =>
        Regex.Replace(s, @"\b(\d+)\s?[-–—]\s?(\d+)\b", m => $"{Card(m.Groups[1].Value)} to {Card(m.Groups[2].Value)}", Opt);

    private static string Numbers(string s)
    {
        s = Regex.Replace(s, @"(?<![\w.])[-−](\d)", "minus $1", Opt);
        return Regex.Replace(s, @"\b\d[\d,]*(?:\.\d+)?\b", m =>
        {
            string tok = m.Value;
            if (!tok.Contains(',') && Regex.IsMatch(tok, @"^\d{4}$", Opt) && Int(tok) is >= 1100 and <= 2099) return Year(Int(tok));
            return DecimalWords(tok);
        }, Opt);
    }

    /// <summary>Python's <c>str.strip()</c> (Unicode whitespace).</summary>
    private static string PyStrip(string s) => s.Trim();

    /// <summary>Expands <paramref name="text"/> to its spoken form. Keeps . , ! ? ; and dashes for prosody.</summary>
    public static string Normalize(string text)
    {
        string s = UnicodeNormalizer.Normalize(text ?? "", NormalizationForm.FormKC);
        s = s.Replace("’", "'").Replace("‘", "'").Replace("“", "\"").Replace("”", "\"")
             .Replace("—", " - ").Replace("–", "-").Replace("…", "...");
        s = Regex.Replace(s, @"https?://\S+|www\.\S+", " link ", Opt);
        s = Regex.Replace(s, @"\b[\w.+-]+@[\w-]+\.[\w.]+\b", " email address ", Opt);
        s = Regex.Replace(s, @"\be\.\s?g\.", "for example,", OptI);
        s = Regex.Replace(s, @"\bi\.\s?e\.", "that is,", OptI);
        s = Regex.Replace(s, @"\betc\.", "etcetera", OptI);
        s = Regex.Replace(s, @"\bvs\.?\b", "versus", OptI);
        s = Dates(s);
        s = Times(s);
        s = BareHour(s);
        s = Phones(s);
        s = Money(s);
        s = Versions(s);
        s = FractionsRule(s);
        s = Magnitudes(s);
        s = UnitsRule(s);
        s = AbbrevRule(s);
        s = Romans(s);
        s = Ordinals(s);
        s = Ranges(s);
        foreach (var (k, v) in Symbols) s = s.Replace(k, v);
        s = Numbers(s);
        s = s.Replace(":", ",");
        s = Regex.Replace(s, @"[_*`^|\\/\[\]{}<>]", " ", Opt);
        s = Regex.Replace(s, @"\s+([.,!?;])", "$1", Opt);
        s = Regex.Replace(s, "([.,!?;])(?=[^\\s\"',.!?;])", "$1 ", Opt);
        s = Regex.Replace(s, @"\s{2,}", " ", Opt).Trim();
        return s;
    }
}
