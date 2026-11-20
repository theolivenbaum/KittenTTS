using KittenTTS.IO;

namespace KittenTTS;

/// <summary>The released KittenTTS 2 small models.</summary>
public enum KittenModel
{
    /// <summary>KittenTTS 2 nano: 17 M parameters, one model per language (en de es it pt ru hi fr), int8 17 MB each.</summary>
    Nano,
    /// <summary>KittenTTS 2 pico: 7 M parameters, one model per language (en de es it pt ru hi fr), int8 8 MB each.</summary>
    Pico,
    /// <summary>KittenTTS 2 mini: 82 M parameters, one model for 16 languages, int8 83 MB.</summary>
    Mini,
}

/// <summary>
/// Where the model files come from. Files are looked up, in order, in the directories the KittenTTSSharp.Models.*
/// NuGet packages install next to the application (<c>&lt;app&gt;/kittentts/kitten-tts-2-&lt;family&gt;/</c>), then in the
/// download cache (<c>~/.cache/kittentts</c> or <c>$KITTENTTS_CACHE</c>), and are otherwise downloaded from the
/// Hugging Face repositories <c>KittenML/kitten-tts-2-{nano,pico,mini}</c> (only the files of the languages and the
/// weight format in use). <see cref="GetDownloadUrl"/> is the one place to point at a mirror.
/// </summary>
public static class KittenModels
{
    /// <summary>The Hugging Face repository of a model.</summary>
    public static string Repository(KittenModel model) => model switch
    {
        KittenModel.Nano => "KittenML/kitten-tts-2-nano",
        KittenModel.Pico => "KittenML/kitten-tts-2-pico",
        KittenModel.Mini => "KittenML/kitten-tts-2-mini",
        _ => throw new ArgumentOutOfRangeException(nameof(model)),
    };

    /// <summary>The directory name of a model (<c>kitten-tts-2-nano</c>), in the package layout and in the cache.</summary>
    public static string DirectoryName(KittenModel model) => Repository(model).Split('/')[1];

    /// <summary>Base URL the files are downloaded from; override with <c>$KITTENTTS_MODEL_BASE_URL</c>
    /// (<c>{base}/{repository}/resolve/main/{file}</c>).</summary>
    public static string BaseUrl => Environment.GetEnvironmentVariable("KITTENTTS_MODEL_BASE_URL") ?? "https://huggingface.co";

    public static string GetDownloadUrl(KittenModel model, string file) => $"{BaseUrl.TrimEnd('/')}/{Repository(model)}/resolve/main/{file}";

    /// <summary>The download cache root.</summary>
    public static string CacheDirectory =>
        Environment.GetEnvironmentVariable("KITTENTTS_CACHE")
        ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".cache", "kittentts");

    /// <summary>The directory a model package installs to (it may hold only some languages).</summary>
    public static string PackageDirectory(KittenModel model) => Path.Combine(AppContext.BaseDirectory, "kittentts", DirectoryName(model));

    /// <summary>The English G2P files every repository carries in <c>g2p/</c>.</summary>
    public static readonly string[] EnglishG2PFiles =
    [
        "g2p/cmu_misaki_us.json", "g2p/g2p_bart_en_us/config.json", "g2p/g2p_bart_en_us/generation_config.json",
        "g2p/g2p_bart_en_us/model.safetensors",
    ];

    /// <summary>
    /// A local directory holding every file <paramref name="languages"/> need in <paramref name="weights"/>: the
    /// package directory when it has them all, else the cache directory, downloading what is missing.
    /// </summary>
    public static async Task<string> EnsureFilesAsync(KittenModel model, IReadOnlyCollection<string>? languages, WeightFormat weights,
                                                      string? cacheDirectory = null, Action<DownloadProgress>? progress = null,
                                                      CancellationToken cancellationToken = default)
    {
        string package = PackageDirectory(model);
        if (File.Exists(Path.Combine(package, "config.json")))
        {
            // no languages asked for: whatever the installed packages hold
            if (languages is not { Count: > 0 })
            {
                var installed = InstalledLanguages(package, weights);
                if (installed.Count > 0) return package;
            }
            else if (HasAll(package, languages, weights)) return package;
        }

        string dir = Path.Combine(cacheDirectory ?? CacheDirectory, DirectoryName(model));
        foreach (var f in new[] { "config.json", "vocab.json" })
            await ModelDownloader.DownloadFileAsync(GetDownloadUrl(model, f), Path.Combine(dir, f), progress, null, cancellationToken).ConfigureAwait(false);
        var repo = ModelRepository.Open(dir);
        foreach (var f in Needed(repo, languages, weights))
            await ModelDownloader.DownloadFileAsync(GetDownloadUrl(model, f), Path.Combine(dir, f.Replace('/', Path.DirectorySeparatorChar)),
                                                    progress, null, cancellationToken).ConfigureAwait(false);
        return dir;
    }

    /// <summary>The languages whose files (in <paramref name="weights"/>) are present in a repository directory.</summary>
    public static List<string> InstalledLanguages(string dir, WeightFormat weights)
    {
        try
        {
            var repo = ModelRepository.Open(dir);
            return repo.Languages.Where(l => HasAll(dir, [l], weights)).ToList();
        }
        catch (Exception e) when (e is IOException or InvalidDataException or ArgumentException or System.Text.Json.JsonException)
        {
            return [];
        }
    }

    private static bool HasAll(string dir, IReadOnlyCollection<string>? languages, WeightFormat weights)
    {
        try
        {
            var repo = ModelRepository.Open(dir);
            return Needed(repo, languages, weights).All(f => File.Exists(Path.Combine(dir, f)));
        }
        catch (Exception e) when (e is IOException or InvalidDataException or ArgumentException or System.Text.Json.JsonException)
        {
            return false;
        }
    }

    private static IEnumerable<string> Needed(ModelRepository repo, IReadOnlyCollection<string>? languages, WeightFormat weights)
    {
        var langs = languages is { Count: > 0 } ? languages : [repo.DefaultLanguage];
        var files = new HashSet<string>(StringComparer.Ordinal);
        foreach (var lang in langs)
        {
            foreach (var f in repo.RequiredFiles(lang, weights)) files.Add(f);
            if (lang == "en") foreach (var f in EnglishG2PFiles) files.Add(f);
        }
        return files;
    }
}
