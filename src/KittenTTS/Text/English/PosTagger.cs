namespace KittenTTS.Text.English;

/// <summary>
/// Penn Treebank tags for misaki, by rules. The original front end tags with spaCy's statistical tagger
/// (<c>en_core_web_sm</c>), which is not ported; misaki only consults the tag in a few places, so rules cover them:
/// punctuation and currency tags (which decide pauses), <c>DT</c> / <c>PRP</c> / <c>TO</c> / <c>IN</c> for the function
/// words with special pronunciations (a, the, I, to, in, by, am, used), <c>CD</c> for numbers, <c>NNP</c> for
/// capitalized words inside a sentence, and NOUN / VERB / past-participle guesses from the previous word for the
/// ~800 heteronyms of the lexicon (record, present, live, read, ...). Where these rules guess differently from spaCy
/// the pronunciation of that word differs from the Python front end's.
/// </summary>
public static class PosTagger
{
    private static readonly HashSet<string> Determiners = new(StringComparer.OrdinalIgnoreCase)
        { "a", "an", "the", "this", "that", "these", "those", "every", "each", "some", "any", "no", "another", "all", "both" };
    private static readonly HashSet<string> Possessives = new(StringComparer.OrdinalIgnoreCase)
        { "my", "your", "his", "her", "its", "our", "their", "whose" };
    private static readonly HashSet<string> SubjectPronouns = new(StringComparer.OrdinalIgnoreCase)
        { "i", "you", "he", "she", "it", "we", "they" };
    private static readonly HashSet<string> ObjectPronouns = new(StringComparer.OrdinalIgnoreCase)
        { "me", "him", "us", "them", "myself", "yourself", "himself", "herself", "itself", "ourselves", "themselves" };
    private static readonly HashSet<string> Modals = new(StringComparer.OrdinalIgnoreCase)
        { "will", "would", "can", "could", "should", "shall", "may", "might", "must", "ca", "wo", "'ll", "'d" };
    private static readonly HashSet<string> BaseVerbTriggers = new(StringComparer.OrdinalIgnoreCase)
        { "to", "do", "does", "did", "n't", "not", "please", "let's", "lets" };
    private static readonly HashSet<string> PerfectTriggers = new(StringComparer.OrdinalIgnoreCase)
        { "have", "has", "had", "'ve", "having", "was", "were", "been", "be", "is", "are", "being", "'s", "'re", "get", "got", "gets" };
    private static readonly HashSet<string> Prepositions = new(StringComparer.OrdinalIgnoreCase)
    {
        "in", "on", "at", "by", "for", "with", "from", "of", "about", "into", "onto", "over", "under", "after", "before",
        "between", "through", "during", "without", "within", "against", "among", "around", "across", "behind", "beyond",
        "near", "since", "until", "upon", "toward", "towards", "via", "per", "than", "like", "as", "if", "because",
        "while", "although", "though", "whether", "unless", "vs", "vs.", "versus",
    };
    private static readonly HashSet<string> NumberWordSet = new(StringComparer.OrdinalIgnoreCase)
    {
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen",
        "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty", "sixty",
        "seventy", "eighty", "ninety", "hundred", "thousand", "million", "billion", "trillion",
    };
    private static readonly HashSet<string> ParticleFollowers = new(StringComparer.OrdinalIgnoreCase)
    {
        "when", "where", "while", "whenever", "and", "or", "but", "so", "then", "if", "because", "automatically", "again",
        "too", "now", "here", "there", "to", "first", "later", "already", "yet",
    };
    private static readonly HashSet<string> Conjunctions = new(StringComparer.OrdinalIgnoreCase) { "and", "or", "but", "nor", "yet", "so" };

    /// <summary>One tag per token.</summary>
    public static string[] Tag(IReadOnlyList<SpacyToken> tokens)
    {
        var tags = new string[tokens.Count];
        bool quoteOpen = false;
        bool sentenceStart = true;
        for (int i = 0; i < tokens.Count; i++)
        {
            string t = tokens[i].Text;
            string? prev = PreviousWord(tokens, i);
            string? next = NextWord(tokens, i);
            string tag = Punctuation(tokens, i, ref quoteOpen) ?? Word(t, prev, tags, i, next, sentenceStart, tokens);
            tags[i] = tag;
            if (t.Length > 0 && !char.IsWhiteSpace(t[0]))
                sentenceStart = tag == "." || (sentenceStart && (tag is "``" or "''" or "-LRB-" or ":"));
        }
        return tags;
    }

    private static string? PreviousWord(IReadOnlyList<SpacyToken> tokens, int i)
    {
        for (int k = i - 1; k >= 0; k--)
            if (tokens[k].Text.Any(char.IsLetterOrDigit) || tokens[k].Text.StartsWith('\'')) return tokens[k].Text;
            else if (!string.IsNullOrWhiteSpace(tokens[k].Text)) return null;   // punctuation breaks the context
        return null;
    }

    private static string? NextWord(IReadOnlyList<SpacyToken> tokens, int i) =>
        i + 1 < tokens.Count && tokens[i + 1].Text.Any(char.IsLetterOrDigit) ? tokens[i + 1].Text : null;

    private static string? Punctuation(IReadOnlyList<SpacyToken> tokens, int i, ref bool quoteOpen)
    {
        string t = tokens[i].Text;
        if (t.Length == 0) return null;
        if (t.All(char.IsWhiteSpace)) return "_SP";
        if (t.Any(char.IsLetterOrDigit)) return null;
        switch (t)
        {
            case "." or "!" or "?" or "?!" or "!!" or "!?" or "??": return ".";
            case ",": return ",";
            case ";" or ":" or "--" or "—" or "–" or "...": return ":";
            case "…": return ":";
            case "-":
            {
                // a hyphen inside a word (no space on either side) is HYPH; a free-standing dash is ':'
                bool glued = (i > 0 && tokens[i - 1].Whitespace.Length == 0) && tokens[i].Whitespace.Length == 0;
                if (glued) return "HYPH";
                // a free-standing hyphen after a number word or "equals" reads as part of the expression, not a pause
                string? before = i > 0 ? tokens[i - 1].Text.ToLowerInvariant() : null;
                return before != null && (NumberWordSet.Contains(before) || before == "equals") ? "HYPH" : ":";
            }
            case "(" or "[" or "{": return "-LRB-";
            case ")" or "]" or "}": return "-RRB-";
            case "“": quoteOpen = true; return "``";
            case "”": quoteOpen = false; return "''";
            case "\"" or "'":
            {
                // spaCy: a quote after a space (or at the start) and glued to the next token opens, anything else closes
                bool opens = (i == 0 || tokens[i - 1].Whitespace.Length > 0) && tokens[i].Whitespace.Length == 0 && i + 1 < tokens.Count;
                quoteOpen = opens;
                return opens ? "``" : "''";
            }
            case "‘": return "``";
            case "’": return "''";
            case "$" or "£" or "€" or "¥": return "$";
            case "#": return "#";
            case "%": return "NN";
            case "&": return "CC";
            case "+" or "=" or "*" or "/" or "<" or ">" or "@" or "~" or "^" or "|": return "SYM";
        }
        if (t.All(c => c == '.')) return ":";
        return "NFP";
    }

    private static bool IsNumber(string t) => t.Length > 0 && char.IsDigit(t[0]) || (t.Length > 1 && t[0] is '-' or '.' && char.IsDigit(t[1]));

    private static bool IsNounTag(string? tag) => tag is "NN" or "NNS" or "NNP" or "NNPS" or "CD";

    private static string Word(string t, string? prev, string[] tags, int i, string? next, bool sentenceStart, IReadOnlyList<SpacyToken> tokens)
    {
        string lower = t.ToLowerInvariant();
        string? prevTag = i > 0 ? tags[i - 1] : null;
        string? p = prev?.ToLowerInvariant();
        string? nx = next?.ToLowerInvariant();
        bool glued = tokens[i].Whitespace.Length == 0 && i + 1 < tokens.Count && tokens[i + 1].Text == "-";
        if (IsNumber(t)) return "CD";
        if (lower is "a" or "an" or "the") return "DT";
        if (t == "I") return "PRP";
        if (lower == "to") return "TO";
        if (lower is "n't" or "not") return "RB";
        if (lower is "'s" && p != null && SubjectPronouns.Contains(p)) return "VBZ";
        if (lower is "'s") return "POS";
        if (Possessives.Contains(lower)) return "PRP$";
        if (SubjectPronouns.Contains(lower) || ObjectPronouns.Contains(lower)) return "PRP";
        if (Modals.Contains(lower)) return "MD";
        if (lower == "that")
        {
            // relative "that" after a noun or pronoun, complementizer before a determiner or pronoun, else determiner
            if (IsNounTag(prevTag) || prevTag is "PRP" or "NNS") return "WDT";
            if (nx != null && (Determiners.Contains(nx) || SubjectPronouns.Contains(nx) || Possessives.Contains(nx))) return "IN";
            if (nx is "is" or "was" or "are" or "were" or "can" or "will" or "would" or "has" or "had" or "needs" or "makes") return "WDT";
            return "DT";
        }
        if (Determiners.Contains(lower)) return "DT";
        if (Conjunctions.Contains(lower)) return "CC";
        if (lower == "in" || lower == "by")
        {
            if (glued) return "IN";                                              // in-context
            if (next == null || ParticleFollowers.Contains(nx!) || nx!.EndsWith("ly")) return "RP";  // fills it in when ..., come in
            return "IN";
        }
        if (Prepositions.Contains(lower)) return "IN";
        if (lower is "am" or "is" or "are" or "was" or "were" or "be" or "been" or "being") return lower switch
        {
            "am" => "VBP", "is" => "VBZ", "are" => "VBP", "was" or "were" => "VBD", "been" => "VBN", "being" => "VBG", _ => "VB",
        };
        if (lower == "used") return nx == "to" ? "VBD" : "VBN";

        // all-caps words: acronyms are proper nouns, a plural-looking one (OPTIONS) a plural noun
        if (t.Length > 1 && t.All(c => !char.IsLetter(c) || char.IsUpper(c)) && t.Any(char.IsLetter))
            return t.Length > 4 && t.EndsWith('S') ? "NNS" : "NNP";

        // Verbs and nouns, guessed from the previous word. This only matters for the lexicon's heteronyms, whose
        // DEFAULT entry is misaki's most common reading: when no rule is confident they get the neutral tag "XX",
        // which selects that DEFAULT, rather than a guess.
        bool heteronym = Lexicon.Default.IsHeteronym(t);
        bool clauseStart = p == null && (sentenceStart || prevTag is "," or ":" or "-LRB-" or "``");
        if (clauseStart) return IsVerbish(lower) ? "VB" : heteronym ? "XX" : Noun(t, sentenceStart);
        if (p != null && prevTag == "RB" && i >= 2)
        {
            // skip one adverb: "can already use"
            string? pp = PreviousWord(tokens, i - 1)?.ToLowerInvariant();
            if (pp != null && (Modals.Contains(pp) || BaseVerbTriggers.Contains(pp))) return "VB";
        }
        if (p != null)
        {
            if (Modals.Contains(p) || BaseVerbTriggers.Contains(p)) return "VB";
            if (p is "have" or "has" or "had" or "'ve" or "having") return "VBN";
            if (p is "is" or "are" or "was" or "were" or "be" or "been" or "being" or "'re" or "'m")
                return lower == "read" || lower.EndsWith("ed") ? "VBN" : "JJ";
            if (Determiners.Contains(p) || Possessives.Contains(p) || prevTag is "JJ" or "POS" or "IN") return Noun(t, sentenceStart);
            if (SubjectPronouns.Contains(p)) return lower.EndsWith('s') && p is "he" or "she" or "it" ? "VBZ" : lower == "read" ? "VBD" : "VBP";
            if (IsNounTag(prevTag) && heteronym)
            {
                // "Smith lives", "Whisper uses", "the tag inserts": a verb after the subject noun
                if (lower.EndsWith('s') && !lower.EndsWith("ss")) return "VBZ";
                if (lower == "read") return "VBD";
                if (IsVerbish(lower)) return "VBP";
            }
            if (prevTag == "CC" && lower == "read") return "VBD";
        }
        if (heteronym) return "XX";
        if (lower.EndsWith("ly") && lower.Length > 4) return "RB";
        if (char.IsUpper(t[0]) && !sentenceStart) return "NNP";
        if (lower.EndsWith("ing") && lower.Length > 4) return "VBG";
        if (lower.EndsWith("ed") && lower.Length > 4) return "VBD";
        return Noun(t, sentenceStart);
    }

    private static readonly HashSet<string> CommonVerbs = new(StringComparer.OrdinalIgnoreCase)
    {
        "use", "read", "record", "present", "live", "lead", "close", "object", "project", "produce", "refuse", "reject",
        "permit", "conduct", "contract", "convert", "desert", "export", "import", "increase", "decrease", "insert",
        "progress", "protest", "rebel", "subject", "suspect", "transfer", "update", "upgrade", "estimate", "separate",
        "set", "run", "pass", "check", "select", "install", "reuse", "add", "save", "make", "try", "see", "note", "stop",
    };

    private static bool IsVerbish(string lower) => CommonVerbs.Contains(lower);

    private static string Noun(string t, bool sentenceStart) =>
        char.IsUpper(t[0]) && !sentenceStart ? "NNP" : t.Length > 2 && t.EndsWith('s') && !t.EndsWith("ss") ? "NNS" : "NN";
}
