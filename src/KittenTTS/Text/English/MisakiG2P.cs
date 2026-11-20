using System.Text.RegularExpressions;

namespace KittenTTS.Text.English;

/// <summary>A token as misaki processes it (<c>misaki.token.MToken</c> with its <c>_</c> fields flattened).</summary>
public sealed class MToken
{
    public required string Text { get; set; }
    public string? Tag { get; set; }
    public string Whitespace { get; set; } = "";
    public string? Phonemes { get; set; }
    public bool IsHead { get; set; } = true;
    public string? Alias { get; set; }
    public float? Stress { get; set; }
    public string? Currency { get; set; }
    public string NumFlags { get; set; } = "";
    public bool Prespace { get; set; }
    public int? Rating { get; set; }

    internal MToken Copy() => (MToken)MemberwiseClone();
}

/// <summary>
/// misaki's English G2P (<c>misaki/en.py</c>, <c>G2P.__call__</c>, version 0.9.4, US English): tokenize, merge,
/// split into sub-tokens, then pronounce right to left with the <see cref="Lexicon"/> (so "the" / "to" can see the
/// next word's first sound), with a fallback for words the lexicon cannot say. Tokens come from the
/// <see cref="SpacyTokenizer"/> and tags from <see cref="PosTagger"/> instead of spaCy.
/// </summary>
public sealed partial class MisakiG2P
{
    private static readonly HashSet<char> SubtokenJunks = [.. "',-._‘’/"];
    private static readonly HashSet<char> Puncts = [.. ";:,.!?—…\"“”"];
    private static readonly HashSet<char> NonQuotePuncts = [.. ";:,.!?—…"];
    private static readonly HashSet<string> PunctTags = [".", ",", "-LRB-", "-RRB-", "``", "\"\"", "''", ":", "$", "#", "NFP"];
    private static readonly Dictionary<string, string> PunctTagPhonemes = new()
    {
        ["-LRB-"] = "(", ["-RRB-"] = ")", ["``"] = "“", ["\"\""] = "”", ["''"] = "”",
    };
    private static readonly HashSet<string> CurrencySymbols = ["$", "£", "€"];

    [GeneratedRegex(@"^['‘’]+|\p{Lu}(?=\p{Lu}\p{Ll})|(?:^-)?(?:\d?[,.]?\d)+|[-_]+|['‘’]{2,}|\p{L}*?(?:['‘’]\p{L})*?\p{Ll}(?=\p{Lu})|\p{L}+(?:['‘’]\p{L})*|[^-_\p{L}'‘’\d]|['‘’]+$")]
    private static partial Regex SubtokenRegex();

    private readonly Lexicon _lexicon = Lexicon.Default;
    private readonly SpacyTokenizer _tokenizer = SpacyTokenizer.Default;
    private readonly Func<MToken, (string? Phonemes, int? Rating)>? _fallback;
    private readonly string? _unk;

    /// <param name="fallback">Called with a token the lexicon cannot pronounce; returns phonemes (or null).</param>
    /// <param name="unk">Written for a token nothing could pronounce (null keeps it out of the string).</param>
    public MisakiG2P(Func<MToken, (string? Phonemes, int? Rating)>? fallback = null, string? unk = "")
    {
        _fallback = fallback;
        _unk = unk;
    }

    /// <summary>text → (phoneme string, tokens).</summary>
    public (string Phonemes, List<MToken> Tokens) Convert(string text)
    {
        text = text.TrimStart();
        var spacy = _tokenizer.Tokenize(text);
        var tags = PosTagger.Tag(spacy);
        var tokens = new List<MToken>();
        for (int i = 0; i < spacy.Count; i++) tokens.Add(new MToken { Text = spacy[i].Text, Tag = tags[i], Whitespace = spacy[i].Whitespace });
        if (tokens.Count == 0) return ("", tokens);
        var words = Retokenize(tokens);

        var ctx = new TokenContext();
        for (int wi = words.Count - 1; wi >= 0; wi--)
        {
            var w = words[wi];
            if (w.Count == 1 && !w.IsGroup)
            {
                var tk = w[0];
                if (tk.Phonemes == null)
                {
                    var (ps, r) = _lexicon.Pronounce(tk.Copy(), ctx);
                    tk.Phonemes = ps;
                    tk.Rating = r;
                }
                if (tk.Phonemes == null && _fallback != null)
                {
                    var (ps, r) = _fallback(tk.Copy());
                    tk.Phonemes = ps;
                    tk.Rating = r;
                }
                ctx = TokenContextAfter(ctx, tk.Phonemes, tk);
                continue;
            }
            int left = 0, right = w.Count;
            bool shouldFallback = false;
            while (left < right)
            {
                MToken? merged = null;
                bool anyFixed = false;
                for (int k = left; k < right; k++) if (w[k].Alias != null || w[k].Phonemes != null) { anyFixed = true; break; }
                if (!anyFixed) merged = MergeTokens(w.Items.GetRange(left, right - left), null);
                var (ps, rating) = merged == null ? (null, null) : _lexicon.Pronounce(merged, ctx);
                if (ps != null)
                {
                    w[left].Phonemes = ps;
                    w[left].Rating = rating;
                    for (int k = left + 1; k < right; k++) w[k].Phonemes = "";
                    ctx = TokenContextAfter(ctx, ps, merged!);
                    right = left;
                    left = 0;
                }
                else if (left + 1 < right) left++;
                else
                {
                    right--;
                    var tk = w[right];
                    if (tk.Phonemes == null)
                    {
                        if (tk.Text.All(SubtokenJunks.Contains))
                        {
                            tk.Phonemes = "";
                            tk.Rating = 3;
                        }
                        else if (_fallback != null)
                        {
                            shouldFallback = true;
                            break;
                        }
                    }
                    left = 0;
                }
            }
            if (shouldFallback)
            {
                var tk = MergeTokens(w.Items, null);
                var (ps, r) = _fallback!(tk);
                w[0].Phonemes = ps;
                w[0].Rating = r;
                for (int j = 1; j < w.Count; j++) { w[j].Phonemes = ""; w[j].Rating = r; }
            }
            else ResolveTokens(w.Items);
        }

        var final = words.Select(w => w.IsGroup ? MergeTokens(w.Items, _unk) : w[0]).ToList();
        foreach (var tk in final)
            if (!string.IsNullOrEmpty(tk.Phonemes)) tk.Phonemes = tk.Phonemes.Replace('ɾ', 'T').Replace('ʔ', 't');
        string result = string.Concat(final.Select(tk => (tk.Phonemes ?? _unk ?? "") + tk.Whitespace));
        return (result, final);
    }

    /// <summary>A word: a single token, or a group of sub-tokens with no whitespace between them.</summary>
    private sealed class Word(List<MToken> items, bool isGroup)
    {
        public List<MToken> Items { get; } = items;
        public bool IsGroup { get; } = isGroup;
        public int Count => Items.Count;
        public MToken this[int i] => Items[i];
    }

    private List<Word> Retokenize(List<MToken> tokens)
    {
        var words = new List<(List<MToken> Items, bool IsList)>();
        string? currency = null;
        for (int i = 0; i < tokens.Count; i++)
        {
            var token = tokens[i];
            List<MToken> tks;
            if (token.Alias == null && token.Phonemes == null)
            {
                tks = SubtokenRegex().Matches(token.Text).Select(m => new MToken
                {
                    Text = m.Value, Tag = token.Tag, Whitespace = "", IsHead = true, NumFlags = token.NumFlags, Stress = token.Stress,
                }).ToList();
                if (tks.Count == 0) tks = [token.Copy()];
            }
            else tks = [token];
            tks[^1].Whitespace = token.Whitespace;
            for (int j = 0; j < tks.Count; j++)
            {
                var tk = tks[j];
                if (tk.Alias != null || tk.Phonemes != null) { }
                else if (tk.Tag == "$" && CurrencySymbols.Contains(tk.Text))
                {
                    currency = tk.Text;
                    tk.Phonemes = "";
                    tk.Rating = 4;
                }
                else if (tk.Tag == ":" && tk.Text is "-" or "–")
                {
                    tk.Phonemes = "—";
                    tk.Rating = 3;
                }
                else if (tk.Tag != null && PunctTags.Contains(tk.Tag) && !tk.Text.All(c => char.ToLowerInvariant(c) is >= 'a' and <= 'z'))
                {
                    tk.Phonemes = PunctTagPhonemes.TryGetValue(tk.Tag, out var pp) ? pp : string.Concat(tk.Text.Where(Puncts.Contains));
                    tk.Rating = 4;
                }
                else if (currency != null)
                {
                    if (tk.Tag != "CD") currency = null;
                    else if (j + 1 == tks.Count && (i + 1 == tokens.Count || tokens[i + 1].Tag != "CD")) tk.Currency = currency;
                }
                else if (0 < j && j < tks.Count - 1 && tk.Text == "2" && Py.IsAlpha(tks[j - 1].Text[^1].ToString() + tks[j + 1].Text[0]))
                    tk.Alias = "to";

                if (tk.Alias != null || tk.Phonemes != null) words.Add(([tk], false));
                else if (words.Count > 0 && words[^1].IsList && words[^1].Items[^1].Whitespace.Length == 0)
                {
                    tk.IsHead = false;
                    words[^1].Items.Add(tk);
                }
                else words.Add(([tk], tk.Whitespace.Length == 0));
            }
        }
        return words.Select(w => new Word(w.Items, w.IsList && w.Items.Count > 1)).ToList();
    }

    internal static MToken MergeTokens(List<MToken> tokens, string? unk)
    {
        var stress = tokens.Where(t => t.Stress != null).Select(t => t.Stress).Distinct().ToList();
        var currency = tokens.Where(t => t.Currency != null).Select(t => t.Currency!).ToList();
        var ratings = tokens.Select(t => t.Rating).ToList();
        string? phonemes = null;
        if (unk != null)
        {
            phonemes = "";
            foreach (var tk in tokens)
            {
                if (tk.Prespace && phonemes.Length > 0 && !char.IsWhiteSpace(phonemes[^1]) && !string.IsNullOrEmpty(tk.Phonemes)) phonemes += " ";
                phonemes += tk.Phonemes ?? unk;
            }
        }
        MToken best = tokens[0];
        int bestScore = int.MinValue;
        foreach (var tk in tokens)
        {
            int score = tk.Text.Sum(c => c.ToString() == c.ToString().ToLowerInvariant() ? 1 : 2);
            if (score > bestScore) { bestScore = score; best = tk; }
        }
        return new MToken
        {
            Text = string.Concat(tokens.Take(tokens.Count - 1).Select(t => t.Text + t.Whitespace)) + tokens[^1].Text,
            Tag = best.Tag,
            Whitespace = tokens[^1].Whitespace,
            Phonemes = phonemes,
            IsHead = tokens[0].IsHead,
            Alias = null,
            Stress = stress.Count == 1 ? stress[0] : null,
            Currency = currency.Count > 0 ? currency.Max(StringComparer.Ordinal) : null,
            NumFlags = new string(tokens.SelectMany(t => t.NumFlags).Distinct().OrderBy(c => c).ToArray()),
            Prespace = tokens[0].Prespace,
            Rating = ratings.Any(r => r == null) ? null : ratings.Min(),
        };
    }

    private static TokenContext TokenContextAfter(TokenContext ctx, string? ps, MToken token)
    {
        bool? vowel = ctx.FutureVowel;
        if (!string.IsNullOrEmpty(ps))
        {
            foreach (char c in ps)
            {
                if (Lexicon.Vowels.Contains(c) || Lexicon.Consonants.Contains(c) || NonQuotePuncts.Contains(c))
                {
                    vowel = NonQuotePuncts.Contains(c) ? null : Lexicon.Vowels.Contains(c);
                    break;
                }
            }
        }
        bool futureTo = token.Text is "to" or "To" || (token.Text == "TO" && token.Tag is "TO" or "IN");
        return new TokenContext(vowel, futureTo);
    }

    private static void ResolveTokens(List<MToken> tokens)
    {
        string text = string.Concat(tokens.Take(tokens.Count - 1).Select(t => t.Text + t.Whitespace)) + tokens[^1].Text;
        var classes = new HashSet<int>();
        foreach (char c in text)
        {
            if (SubtokenJunks.Contains(c)) continue;
            classes.Add(char.IsLetter(c) ? 0 : c is >= '0' and <= '9' ? 1 : 2);
        }
        bool prespace = text.Contains(' ') || text.Contains('/') || classes.Count > 1;
        for (int i = 0; i < tokens.Count; i++)
        {
            var tk = tokens[i];
            if (tk.Phonemes == null)
            {
                if (i == tokens.Count - 1 && tk.Text.Length == 1 && NonQuotePuncts.Contains(tk.Text[0]))
                {
                    tk.Phonemes = tk.Text;
                    tk.Rating = 3;
                }
                else if (tk.Text.All(SubtokenJunks.Contains))
                {
                    tk.Phonemes = "";
                    tk.Rating = 3;
                }
            }
            else if (i > 0) tk.Prespace = prespace;
        }
        if (prespace) return;
        var indices = new List<(bool Primary, int Weight, int I)>();
        for (int i = 0; i < tokens.Count; i++)
            if (!string.IsNullOrEmpty(tokens[i].Phonemes))
                indices.Add((tokens[i].Phonemes!.Contains(Lexicon.Primary), Lexicon.StressWeight(tokens[i].Phonemes), i));
        if (indices.Count == 2 && tokens[indices[0].I].Text.Length == 1)
        {
            int i = indices[1].I;
            tokens[i].Phonemes = Lexicon.ApplyStress(tokens[i].Phonemes, -0.5f);
            return;
        }
        if (indices.Count < 2 || indices.Count(x => x.Primary) <= (indices.Count + 1) / 2) return;
        foreach (var (_, _, i) in indices.OrderBy(x => x.Primary).ThenBy(x => x.Weight).ThenBy(x => x.I).Take(indices.Count / 2))
            tokens[i].Phonemes = Lexicon.ApplyStress(tokens[i].Phonemes, -0.5f);
    }
}
