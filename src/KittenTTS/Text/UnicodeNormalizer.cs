using System.IO.Compression;
using System.Reflection;
using System.Text;
using System.Text.Json;

namespace KittenTTS.Text;

/// <summary>
/// Unicode normalization (NFC, NFD, NFKC, NFKD) that works in every .NET configuration. <see cref="string.Normalize()"/>
/// needs ICU and silently returns its input under <c>InvariantGlobalization</c> (common in containers and AOT apps),
/// which would make the front end keep accents and compatibility characters the Python front end folds. When the
/// runtime cannot normalize, this falls back to tables exported from Python's <c>unicodedata</c>
/// (<c>scripts/export_english_data.py</c>): full decompositions, canonical ordering by combining class, canonical
/// composition, and the algorithmic Hangul syllables.
/// </summary>
public static class UnicodeNormalizer
{
    private static readonly bool RuntimeWorks = Probe();

    private static bool Probe()
    {
        try
        {
            return "é".Normalize(NormalizationForm.FormD).Length == 2 && "ﬁ".Normalize(NormalizationForm.FormKC) == "fi";
        }
        catch (PlatformNotSupportedException)
        {
            return false;
        }
    }

    private sealed class Tables
    {
        public required Dictionary<int, int[]> Nfd, Nfkd;
        public required Dictionary<int, int> Ccc;
        public required Dictionary<(int, int), int> Compose;
    }

    private static readonly Lazy<Tables> Data = new(Load);

    private static Tables Load()
    {
        using var s = Assembly.GetExecutingAssembly().GetManifestResourceStream("KittenTTS.Text.English.Data.unicode_norm.json.gz")
                      ?? throw new InvalidOperationException("The Unicode normalization tables are missing from the assembly.");
        using var gz = new GZipStream(s, CompressionMode.Decompress);
        using var doc = JsonDocument.Parse(gz);
        var r = doc.RootElement;
        static Dictionary<int, int[]> Map(JsonElement e) =>
            e.EnumerateObject().ToDictionary(p => int.Parse(p.Name), p => p.Value.EnumerateArray().Select(x => x.GetInt32()).ToArray());
        return new Tables
        {
            Nfd = Map(r.GetProperty("nfd")),
            Nfkd = Map(r.GetProperty("nfkd")),
            Ccc = r.GetProperty("ccc").EnumerateObject().ToDictionary(p => int.Parse(p.Name), p => p.Value.GetInt32()),
            Compose = r.GetProperty("compose").EnumerateArray().ToDictionary(a => (a[0].GetInt32(), a[1].GetInt32()), a => a[2].GetInt32()),
        };
    }

    /// <summary>Normalizes <paramref name="s"/> to <paramref name="form"/>.</summary>
    public static string Normalize(string s, NormalizationForm form)
    {
        if (string.IsNullOrEmpty(s)) return s ?? "";
        if (RuntimeWorks) return s.Normalize(form);
        if (s.All(c => c < 0x80)) return s;   // ASCII is invariant under every form
        return Managed(s, form);
    }

    /// <summary>The table-driven implementation (public for tests; <see cref="Normalize"/> uses the runtime's when it works).</summary>
    public static string Managed(string s, NormalizationForm form)
    {
        var t = Data.Value;
        bool compat = form is NormalizationForm.FormKC or NormalizationForm.FormKD;
        bool compose = form is NormalizationForm.FormC or NormalizationForm.FormKC;
        var cps = new List<int>(s.Length);
        foreach (var rune in s.EnumerateRunes()) Decompose(rune.Value, compat ? t.Nfkd : t.Nfd, cps);
        // canonical ordering: stable sort of each run of non-starters by combining class
        for (int i = 1; i < cps.Count; i++)
        {
            int c = Ccc(t, cps[i]);
            if (c == 0) continue;
            int j = i;
            while (j > 0 && Ccc(t, cps[j - 1]) > c)
            {
                (cps[j - 1], cps[j]) = (cps[j], cps[j - 1]);
                j--;
            }
        }
        if (compose) cps = Compose(t, cps);
        var sb = new StringBuilder(cps.Count);
        foreach (int cp in cps) sb.Append(char.ConvertFromUtf32(cp));
        return sb.ToString();
    }

    private static int Ccc(Tables t, int cp) => t.Ccc.TryGetValue(cp, out int c) ? c : 0;

    private const int SBase = 0xAC00, LBase = 0x1100, VBase = 0x1161, TBase = 0x11A7, LCount = 19, VCount = 21, TCount = 28;
    private const int NCount = VCount * TCount, SCount = LCount * NCount;

    private static void Decompose(int cp, Dictionary<int, int[]> map, List<int> output)
    {
        if (cp >= SBase && cp < SBase + SCount)
        {
            int si = cp - SBase;
            output.Add(LBase + si / NCount);
            output.Add(VBase + si % NCount / TCount);
            if (si % TCount != 0) output.Add(TBase + si % TCount);
            return;
        }
        if (map.TryGetValue(cp, out var d)) output.AddRange(d);   // the tables hold full (recursive) decompositions
        else output.Add(cp);
    }

    private static List<int> Compose(Tables t, List<int> cps)
    {
        var result = new List<int>(cps.Count);
        int starter = -1;          // index in result of the last starter
        int lastCcc = -1;
        foreach (int cp in cps)
        {
            int ccc = Ccc(t, cp);
            if (starter >= 0)
            {
                int s = result[starter];
                // Hangul LV / LVT
                if (s >= LBase && s < LBase + LCount && cp >= VBase && cp < VBase + VCount && lastCcc == -1)
                {
                    result[starter] = SBase + ((s - LBase) * VCount + (cp - VBase)) * TCount;
                    continue;
                }
                if (s >= SBase && s < SBase + SCount && (s - SBase) % TCount == 0 && cp > TBase && cp < TBase + TCount && lastCcc == -1)
                {
                    result[starter] = s + (cp - TBase);
                    continue;
                }
                bool blocked = lastCcc != -1 && (lastCcc == 0 || lastCcc >= ccc);
                if (!blocked && t.Compose.TryGetValue((s, cp), out int composed))
                {
                    result[starter] = composed;
                    continue;
                }
            }
            if (ccc == 0)
            {
                starter = result.Count;
                lastCcc = -1;
            }
            else lastCcc = ccc;
            result.Add(cp);
        }
        return result;
    }
}
