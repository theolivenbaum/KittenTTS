using System.Text.Json;
using KittenTTS.Text;
using KittenTTS.Text.English;
using Xunit.Abstractions;

namespace KittenTTS.Tests;

/// <summary>
/// The C# English front end against the original Python one (testdata/english_reference.json, written by
/// scripts/english_reference.py on 600+ sentences): the normaliser and the tokenizer must agree exactly; the phoneme
/// chunks agree except where the rule-based POS tagger guesses a heteronym differently from spaCy's statistical one.
/// </summary>
public sealed class EnglishFrontendTests(ITestOutputHelper output)
{
    private sealed record RefCase(string Text, string Normalized, List<List<string>> Tokens, List<string> V1, List<string>? V2);

    private static List<RefCase> Cases()
    {
        var path = Path.Combine(AppContext.BaseDirectory, "testdata", "english_reference.json");
        using var doc = JsonDocument.Parse(File.ReadAllText(path));
        return doc.RootElement.GetProperty("cases").EnumerateArray().Select(c => new RefCase(
            c.GetProperty("text").GetString()!, c.GetProperty("normalized").GetString()!,
            c.GetProperty("tokens").EnumerateArray().Select(t => t.EnumerateArray().Select(x => x.GetString()!).ToList()).ToList(),
            c.GetProperty("v1").EnumerateArray().Select(x => x.GetString()!).ToList(),
            c.TryGetProperty("v2", out var v2) ? v2.EnumerateArray().Select(x => x.GetString()!).ToList() : null)).ToList();
    }

    [Fact]
    public void NormalizerMatchesPython()
    {
        int ok = 0, n = 0;
        foreach (var c in Cases())
        {
            n++;
            string got = TextNormalizer.Normalize(c.Text);
            if (got == c.Normalized) ok++;
            else output.WriteLine($"IN : {c.Text}\nPY : {c.Normalized}\nC# : {got}\n");
        }
        output.WriteLine($"normalizer: {ok}/{n} identical");
        Assert.Equal(n, ok);
    }

    [Fact]
    public void TokenizerMatchesSpacy()
    {
        int ok = 0, n = 0;
        foreach (var c in Cases())
        {
            n++;
            var got = SpacyTokenizer.Default.Tokenize(c.Normalized);
            var want = c.Tokens.Select(t => (t[0], t[1])).ToList();
            if (got.Select(t => (t.Text, t.Whitespace)).SequenceEqual(want)) ok++;
            else output.WriteLine($"IN : {c.Normalized}\nPY : {string.Join(" | ", want.Select(t => t.Item1))}\nC# : {string.Join(" | ", got.Select(t => t.Text))}\n");
        }
        output.WriteLine($"tokenizer: {ok}/{n} identical");
        Assert.Equal(n, ok);
    }

    [Fact]
    public void PhonemesMatchPython()
    {
        if (!Fixtures.Available(output, reference: false)) return;
        var repo = ModelRepository.Open(Fixtures.ModelDir!);
        var fe = Frontend.For(repo);
        var cases = Cases();
        int ok = 0, n = 0;
        foreach (var c in cases)
        {
            var want = repo.Phonemizer == "v2" ? c.V2 : c.V1;
            if (want == null) continue;
            n++;
            var (got, _) = fe.Phonemize(c.Text, "en");
            if (got.SequenceEqual(want)) ok++;
            else output.WriteLine($"IN : {c.Text}\nPY : {string.Join(" | ", want)}\nC# : {string.Join(" | ", got)}\n");
        }
        double rate = (double)ok / n;
        output.WriteLine($"phonemes ({repo.Phonemizer}): {ok}/{n} identical ({rate:P1})");
        Assert.True(rate >= 0.93, $"only {rate:P1} of the sentences phonemize identically");
    }
}
