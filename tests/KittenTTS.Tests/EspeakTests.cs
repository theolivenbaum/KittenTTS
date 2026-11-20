using System.Text.Json;
using KittenTTS.Text;
using KittenTTS.Text.Espeak;
using Xunit.Abstractions;

namespace KittenTTS.Tests;

/// <summary>
/// The non-English front end against the original Python one (testdata/espeak_reference.json, written by
/// scripts/frontend_reference.py with espeak-ng from espeakng_loader). The strings depend on the espeak-ng build, so
/// they are compared only when the loaded library has the reference's version; point KITTENTTS_ESPEAK_LIBRARY and
/// ESPEAK_DATA_PATH at espeakng_loader's library and data to run them. Skipped without espeak-ng.
/// </summary>
public sealed class EspeakTests(ITestOutputHelper output)
{
    private sealed record Case(string Version, string Language, string Text, string Phonemes);
    private sealed record Reference(string Espeak_version, List<Case> Cases);

    private static Reference Load() => JsonSerializer.Deserialize<Reference>(
        File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "testdata", "espeak_reference.json")),
        new JsonSerializerOptions { PropertyNameCaseInsensitive = true })!;

    private bool Available(out Reference reference)
    {
        reference = Load();
        if (!EspeakNative.IsAvailable)
        {
            output.WriteLine("SKIPPED: espeak-ng is not installed (set KITTENTTS_ESPEAK_LIBRARY).");
            return false;
        }
        if (EspeakNative.VersionString != reference.Espeak_version)
        {
            output.WriteLine($"SKIPPED: loaded espeak-ng {EspeakNative.VersionString} ({EspeakNative.LibraryPath}), the reference was " +
                             $"made with {reference.Espeak_version}; set KITTENTTS_ESPEAK_LIBRARY / ESPEAK_DATA_PATH to that build.");
            return false;
        }
        return true;
    }

    [Fact]
    public void MatchesPythonFrontEnd()
    {
        if (!Available(out var reference)) return;
        var failures = new List<string>();
        foreach (var c in reference.Cases)
        {
            var p = new LanguagePhonemizer(c.Language, c.Version);
            string got = p.Phonemize(c.Text);
            output.WriteLine($"{c.Version} {c.Language} {(got == c.Phonemes ? "ok  " : "DIFF")} {got}");
            if (got != c.Phonemes) failures.Add($"{c.Version} {c.Language} '{c.Text}': expected '{c.Phonemes}', got '{got}'");
        }
        Assert.True(failures.Count == 0, string.Join("\n", failures));
    }

    [Fact]
    public void ConcurrentCallsAreSerialized()
    {
        if (!Available(out var reference)) return;
        var cases = reference.Cases.Where(c => c.Version == "v2").ToList();
        var results = new string[cases.Count];
        Parallel.For(0, cases.Count, i => results[i] = new LanguagePhonemizer(cases[i].Language, "v2").Phonemize(cases[i].Text));
        for (int i = 0; i < cases.Count; i++) Assert.Equal(cases[i].Phonemes, results[i]);
    }

    [Fact]
    public void UnsupportedLanguagesFailClearly()
    {
        Assert.Throws<NotSupportedException>(() => new LanguagePhonemizer("ja", "v2"));
        Assert.Throws<NotSupportedException>(() => new LanguagePhonemizer("ko", "v1"));
        Assert.Throws<ArgumentException>(() => new LanguagePhonemizer("en", "v1"));
        Assert.Throws<ArgumentException>(() => new LanguagePhonemizer("xx", "v1"));
    }
}
