using System.Globalization;
using System.Numerics;

namespace KittenTTS.Text.English;

/// <summary>
/// English number words exactly as the Python <c>num2words</c> package (lang_EN) writes them, which both the text
/// normaliser and misaki rely on: British-style "and" after hundreds and the final group ("one thousand and one"),
/// commas between groups ("one thousand, two hundred and thirty-four"), hyphenated tens, "oh-five" years.
/// </summary>
public static class NumberWords
{
    private static readonly (BigInteger Value, string Word)[] Cards = BuildCards();

    private static (BigInteger, string)[] BuildCards()
    {
        var list = new List<(BigInteger, string)>();
        // num2words' EU high numwords, largest first (10^33 ... 10^6)
        string[] high = ["dec", "non", "oct", "sept", "sext", "quint", "quadr", "tr", "b", "m"];
        for (int i = 0; i < high.Length; i++) list.Add((BigInteger.Pow(10, 33 - 3 * i), high[i] + "illion"));
        list.Add((1000, "thousand"));
        list.Add((100, "hundred"));
        string[] tens = ["ninety", "eighty", "seventy", "sixty", "fifty", "forty", "thirty"];
        for (int i = 0; i < tens.Length; i++) list.Add((90 - 10 * i, tens[i]));
        string[] low = ["twenty", "nineteen", "eighteen", "seventeen", "sixteen", "fifteen", "fourteen", "thirteen",
                        "twelve", "eleven", "ten", "nine", "eight", "seven", "six", "five", "four", "three", "two", "one", "zero"];
        for (int i = 0; i < low.Length; i++) list.Add((20 - i, low[i]));
        return list.ToArray();
    }

    private static readonly Dictionary<string, string> Ords = new()
    {
        ["one"] = "first", ["two"] = "second", ["three"] = "third", ["four"] = "fourth", ["five"] = "fifth",
        ["six"] = "sixth", ["seven"] = "seventh", ["eight"] = "eighth", ["nine"] = "ninth", ["ten"] = "tenth",
        ["eleven"] = "eleventh", ["twelve"] = "twelfth",
    };

    private abstract record Node;
    private sealed record Leaf(string Text, BigInteger Num) : Node;
    private sealed record Branch(List<Node> Items) : Node;

    /// <summary><c>num2words(n)</c>.</summary>
    public static string Cardinal(BigInteger value)
    {
        string neg = "";
        if (value < 0) { value = -value; neg = "minus "; }
        if (value >= BigInteger.Pow(10, 36)) throw new OverflowException($"{value} is too large to spell.");
        var (text, _) = Clean(SplitNum(value));
        return neg + text;
    }

    public static string Cardinal(long value) => Cardinal(new BigInteger(value));

    /// <summary><c>num2words(n, to='ordinal')</c>.</summary>
    public static string Ordinal(BigInteger value)
    {
        if (value < 0) throw new ArgumentOutOfRangeException(nameof(value), "Ordinals of negative numbers are not defined.");
        var words = Cardinal(value).Split(' ');
        var last = words[^1].Split('-');
        string w = last[^1];
        if (Ords.TryGetValue(w, out var o)) w = o;
        else
        {
            if (w.EndsWith('y')) w = w[..^1] + "ie";
            w += "th";
        }
        last[^1] = w;
        words[^1] = string.Join('-', last);
        return string.Join(' ', words);
    }

    public static string Ordinal(long value) => Ordinal(new BigInteger(value));

    /// <summary><c>num2words(n, to='year')</c> (non-negative years).</summary>
    public static string Year(long value)
    {
        if (value < 0) return Year(-value) + " BC";
        long high = value / 100, low = value % 100;
        if (high == 0 || (high % 10 == 0 && low < 10) || high >= 100) return Cardinal(value);
        string hi = Cardinal(high);
        string lo = low == 0 ? "hundred" : low < 10 ? "oh-" + Cardinal(low) : Cardinal(low);
        return hi + " " + lo;
    }

    /// <summary><c>num2words(float)</c>: "three point one four" (digits after the point, as num2words prints them).</summary>
    public static string Float(string number)
    {
        // num2words converts through float and prints the fractional digits of repr(float); for the decimal strings
        // the front end passes this is the digits as written (trailing zeros dropped).
        if (!decimal.TryParse(number, NumberStyles.Float, CultureInfo.InvariantCulture, out var d)) return number;
        bool neg = d < 0;
        if (neg) d = -d;
        string s = d.ToString(CultureInfo.InvariantCulture);
        int dot = s.IndexOf('.');
        string whole = dot < 0 ? s : s[..dot];
        string frac = dot < 0 ? "0" : s[(dot + 1)..].TrimEnd('0');
        if (frac.Length == 0) frac = "0";
        var words = new List<string> { Cardinal(BigInteger.Parse(whole, CultureInfo.InvariantCulture)), "point" };
        foreach (char c in frac) words.Add(Cardinal(c - '0'));
        return (neg ? "minus " : "") + string.Join(' ', words);
    }

    private static Node SplitNum(BigInteger value)
    {
        foreach (var (elem, word) in Cards)
        {
            if (elem > value) continue;
            var outList = new List<Node>();
            BigInteger div, mod;
            if (value == 0) { div = 1; mod = 0; }
            else div = BigInteger.DivRem(value, elem, out mod);
            if (div == 1) outList.Add(new Leaf("one", 1));
            else outList.Add(SplitNum(div));
            outList.Add(new Leaf(word, elem));
            if (mod != 0) outList.Add(SplitNum(mod));
            return new Branch(outList);
        }
        throw new InvalidOperationException("unreachable");
    }

    private static (string, BigInteger) Clean(Node node)
    {
        if (node is Leaf l) return (l.Text, l.Num);
        var val = ((Branch)node).Items;
        while (val.Count != 1)
        {
            var outList = new List<Node>();
            if (val[0] is Leaf left && val[1] is Leaf right)
            {
                var (t, n) = Merge((left.Text, left.Num), (right.Text, right.Num));
                outList.Add(new Leaf(t, n));
                if (val.Count > 2) outList.Add(new Branch(val.Skip(2).ToList()));
            }
            else
            {
                foreach (var elem in val)
                {
                    if (elem is Branch b)
                    {
                        if (b.Items.Count == 1) outList.Add(b.Items[0]);
                        else { var (t, n) = Clean(b); outList.Add(new Leaf(t, n)); }
                    }
                    else outList.Add(elem);
                }
            }
            val = outList;
        }
        return val[0] is Leaf r ? (r.Text, r.Num) : Clean(val[0]);
    }

    private static (string, BigInteger) Merge((string Text, BigInteger Num) l, (string Text, BigInteger Num) r)
    {
        if (l.Num == 1 && r.Num < 100) return (r.Text, r.Num);
        if (100 > l.Num && l.Num > r.Num) return ($"{l.Text}-{r.Text}", l.Num + r.Num);
        if (l.Num >= 100 && 100 > r.Num) return ($"{l.Text} and {r.Text}", l.Num + r.Num);
        if (r.Num > l.Num) return ($"{l.Text} {r.Text}", l.Num * r.Num);
        return ($"{l.Text}, {r.Text}", l.Num + r.Num);
    }
}
