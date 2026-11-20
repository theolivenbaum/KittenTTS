using System.Diagnostics;
using System.Globalization;
using KittenTTS;
using KittenTTS.Audio;
using KittenTTS.IO;
using KittenTTS.Numerics;

// kitten-tts say       [options] "text"
// kitten-tts phonemize [options] "text"
// kitten-tts voices    [--model ...]
// kitten-tts download  [--model ...] [--language ...] [--weights ...]
// kitten-tts bench     [options] "text"

if (args.Length == 0 || args[0] is "-h" or "--help")
{
    Console.WriteLine("""
        usage: kitten-tts say       [options] <text>      synthesize to a WAV file
               kitten-tts phonemize [options] <text>      print the phoneme chunks
               kitten-tts voices    [--model ...]         list the voices
               kitten-tts download  [options]             only download (prints the directory)
               kitten-tts bench     [options] <text>      time repeated synthesis

        options:
          --model <name|dir>   nano (default), pico, mini, or a local repository directory
          --voice <name>       a voice name (any case); default: the language's first voice
          --language <code>    en (default), de, es, fr, ...; comma-separated for download
          --speed <x>          > 1 speaks faster (default 1)
          --weights <w>        int8 (default) or fp32
          --out <file.wav>     say: output file (default out.wav)
          --threads <n>        worker threads (default: all cores)
          --seed <n>           seed of the synthesis noise (default: random)
          --no-noise           no synthesis noise (deterministic)
          --runs <n>           bench: timed runs (default 5, after one warm-up)
        Models are downloaded from Hugging Face to ~/.cache/kittentts ($KITTENTTS_CACHE) unless a
        KittenTTSSharp.Models.* package installed them next to the application. Set KITTENTTS_PROFILE=1 for
        a per-stage time breakdown in bench.
        """);
    return 1;
}

string command = args[0];
var text = new List<string>();
string model = "nano", outPath = "out.wav";
string? voice = null, language = null;
float speed = 1f;
int threads = 0, runs = 5;
int? seed = null;
bool noise = true;
var weights = WeightFormat.Int8;
for (int i = 1; i < args.Length; i++)
{
    switch (args[i])
    {
        case "--model": model = args[++i]; break;
        case "--voice": voice = args[++i]; break;
        case "--language": language = args[++i]; break;
        case "--speed": speed = float.Parse(args[++i], CultureInfo.InvariantCulture); break;
        case "--out": outPath = args[++i]; break;
        case "--threads": threads = int.Parse(args[++i], CultureInfo.InvariantCulture); break;
        case "--seed": seed = int.Parse(args[++i], CultureInfo.InvariantCulture); break;
        case "--runs": runs = int.Parse(args[++i], CultureInfo.InvariantCulture); break;
        case "--no-noise": noise = false; break;
        case "--weights":
            weights = args[++i].ToLowerInvariant() switch
            {
                "int8" => WeightFormat.Int8,
                "fp32" or "float32" => WeightFormat.Float32,
                var w => throw new ArgumentException($"unknown weights '{w}' (int8 or fp32)"),
            };
            break;
        default: text.Add(args[i]); break;
    }
}

Action<DownloadProgress> progress = p =>
{
    string total = p.TotalBytes is long t ? $" of {t / 1e6:F1} MB ({p.Fraction:P0})" : "";
    Console.Error.Write($"\rDownloading {p.FileName}: {p.DownloadedBytes / 1e6:F1} MB{total}   ");
    if (p.TotalBytes is long all && p.DownloadedBytes >= all) Console.Error.WriteLine();
};

var options = new KittenTtsOptions { Threads = threads, Weights = weights, Seed = seed, Noise = noise };
string[]? languages = language?.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);

async Task<KittenTts> Open()
{
    if (Directory.Exists(model)) return KittenTts.Load(model, options);
    if (!Enum.TryParse<KittenModel>(model, ignoreCase: true, out var which))
        throw new ArgumentException($"unknown model '{model}' (nano, pico, mini or a directory)");
    // voice-only selection: fetch the files of the voice's language (the voice names are in config.json)
    var langs = languages;
    if (langs == null && voice != null)
    {
        using var probe = await KittenTts.LoadAsync(which, null, options, progress: progress);
        langs = [probe.ResolveVoice(voice).Language];
    }
    return await KittenTts.LoadAsync(which, langs, options, progress: progress);
}

try
{
    switch (command)
    {
        case "download":
        {
            if (!Enum.TryParse<KittenModel>(model, ignoreCase: true, out var which)) throw new ArgumentException($"unknown model '{model}'");
            Console.WriteLine(await KittenModels.EnsureFilesAsync(which, languages, weights, progress: progress));
            return 0;
        }
        case "voices":
        {
            using var tts = await Open();
            foreach (var v in tts.Voices()) Console.WriteLine($"{v.Language}  {v.Name,-12} {v.Description}");
            return 0;
        }
        case "phonemize":
        {
            using var tts = await Open();
            string lang = languages?.FirstOrDefault() ?? (voice != null ? tts.ResolveVoice(voice).Language : tts.Repository.DefaultLanguage);
            foreach (var chunk in tts.Phonemize(string.Join(' ', text), lang)) Console.WriteLine(chunk);
            return 0;
        }
        case "say":
        {
            using var tts = await Open();
            var sw = Stopwatch.StartNew();
            var audio = tts.Generate(string.Join(' ', text), voice, languages?.FirstOrDefault(), speed);
            sw.Stop();
            WaveFile.WriteMono16(outPath, audio, tts.SampleRate);
            double seconds = audio.Length / (double)tts.SampleRate;
            Console.Error.WriteLine($"{outPath}: {seconds:F2} s of audio in {sw.Elapsed.TotalSeconds:F2} s (RTF {sw.Elapsed.TotalSeconds / Math.Max(seconds, 1e-9):F3})");
            return 0;
        }
        case "bench":
        {
            using var tts = await Open();
            string t = text.Count > 0 ? string.Join(' ', text)
                : "The quick brown fox jumps over the lazy dog, then naps in the warm afternoon sun for an hour or two.";
            var audio = tts.Generate(t, voice, languages?.FirstOrDefault(), speed);   // warm-up
            double seconds = audio.Length / (double)tts.SampleRate;
            Profiler.Reset();
            var times = new List<double>();
            for (int r = 0; r < runs; r++)
            {
                var sw = Stopwatch.StartNew();
                tts.Generate(t, voice, languages?.FirstOrDefault(), speed);
                times.Add(sw.Elapsed.TotalSeconds);
            }
            times.Sort();
            double median = times[times.Count / 2];
            Console.WriteLine($"{tts.Repository.Name} ({weights}, {(threads == 0 ? Environment.ProcessorCount : threads)} threads): " +
                              $"{seconds:F2} s of audio, median {median * 1000:F0} ms, RTF {median / seconds:F3} ({seconds / median:F1}x realtime)");
            if (Profiler.Enabled) Console.WriteLine(Profiler.Report());
            return 0;
        }
        default:
            Console.Error.WriteLine($"unknown command '{command}'");
            return 1;
    }
}
catch (Exception e) when (e is ArgumentException or FileNotFoundException or DllNotFoundException or NotSupportedException or HttpRequestException)
{
    Console.Error.WriteLine($"error: {e.Message}");
    return 2;
}
