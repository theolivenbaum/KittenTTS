using System.Runtime.InteropServices;
using System.Text;

namespace KittenTTS.Text.Espeak;

/// <summary>
/// The few functions of the espeak-ng shared library the phonemizer needs (<c>espeak_Initialize</c>,
/// <c>espeak_ListVoices</c>, <c>espeak_SetVoiceByName</c>, <c>espeak_TextToPhonemes</c>), bound at run time with
/// <see cref="NativeLibrary"/> so that a missing library is an exception with install instructions instead of a crash,
/// and so the library can be chosen with <c>KITTENTTS_ESPEAK_LIBRARY</c>.
/// <para>
/// espeak-ng keeps its state in globals and is not thread-safe: every call goes through <see cref="Sync"/>. The data
/// directory is <c>ESPEAK_DATA_PATH</c> when set, else the library's compiled-in default.
/// </para>
/// </summary>
public static unsafe class EspeakNative
{
    /// <summary>Library names tried, in order, after <c>KITTENTTS_ESPEAK_LIBRARY</c>.</summary>
    public static readonly string[] LibraryNames =
    [
        "libespeak-ng.so.1", "libespeak-ng.so", "espeak-ng", "libespeak-ng.dylib", "libespeak-ng.1.dylib",
        "libespeak-ng.dll", "espeak-ng.dll",
    ];

    /// <summary>Held around every espeak call (the library is not reentrant).</summary>
    public static readonly object Sync = new();

    private const int AudioOutputSynchronous = 0x02;

    private static readonly Lazy<State> Loaded = new(Load, LazyThreadSafetyMode.ExecutionAndPublication);

    private sealed class State
    {
        public required string LibraryPath;
        public required string VersionString;
        public required (int, int, int) Version;
        public required delegate* unmanaged[Cdecl]<nint, nint> ListVoices;
        public required delegate* unmanaged[Cdecl]<byte*, int> SetVoiceByName;
        public required delegate* unmanaged[Cdecl]<byte**, int, int, byte*> TextToPhonemes;
        public string? CurrentVoice;
    }

    /// <summary>True when the espeak-ng library can be loaded and initialized.</summary>
    public static bool IsAvailable
    {
        get
        {
            try { _ = Loaded.Value; return true; }
            catch (DllNotFoundException) { return false; }
            catch (InvalidOperationException) { return false; }
        }
    }

    /// <summary>The loaded library (a name or a path, as found).</summary>
    public static string LibraryPath => Loaded.Value.LibraryPath;

    /// <summary>The espeak-ng version (major, minor, patch), from <c>espeak_Info</c>.</summary>
    public static (int Major, int Minor, int Patch) Version => Loaded.Value.Version;

    /// <summary>The version as a string, e.g. <c>1.52.0</c>.</summary>
    public static string VersionString => Loaded.Value.VersionString;

    private static State Load()
    {
        var candidates = new List<string>();
        var env = Environment.GetEnvironmentVariable("KITTENTTS_ESPEAK_LIBRARY");
        if (!string.IsNullOrEmpty(env)) candidates.Add(env);
        candidates.AddRange(LibraryNames);
        nint handle = 0;
        string? found = null;
        foreach (var name in candidates)
        {
            if (NativeLibrary.TryLoad(name, out handle)) { found = name; break; }
        }
        if (handle == 0)
            throw new DllNotFoundException(
                "The espeak-ng library was not found; it is needed to phonemize languages other than English. Install it " +
                "(Debian / Ubuntu: apt install espeak-ng; macOS: brew install espeak-ng; Windows: the espeak-ng .msi from " +
                "https://github.com/espeak-ng/espeak-ng/releases) or set KITTENTTS_ESPEAK_LIBRARY to the library's path. " +
                $"Tried: {string.Join(", ", candidates)}.");

        T Fn<T>(string name) where T : unmanaged
        {
            if (!NativeLibrary.TryGetExport(handle, name, out var p))
                throw new InvalidOperationException($"{found} does not export {name}; is it an espeak-ng library?");
            return *(T*)&p;
        }

        var info = Fn<nint>("espeak_Info");
        var initialize = Fn<nint>("espeak_Initialize");
        var state = new State
        {
            LibraryPath = found!,
            VersionString = "",
            Version = default,
            ListVoices = (delegate* unmanaged[Cdecl]<nint, nint>)Fn<nint>("espeak_ListVoices"),
            SetVoiceByName = (delegate* unmanaged[Cdecl]<byte*, int>)Fn<nint>("espeak_SetVoiceByName"),
            TextToPhonemes = (delegate* unmanaged[Cdecl]<byte**, int, int, byte*>)Fn<nint>("espeak_TextToPhonemes"),
        };

        lock (Sync)
        {
            // espeak_Info: "1.52.0  2024-..." (a "-dev" suffix on development builds)
            byte* v = ((delegate* unmanaged[Cdecl]<byte**, byte*>)info)(null);
            string vs = Marshal.PtrToStringUTF8((nint)v) ?? "";
            vs = vs.Trim().Split(' ')[0].Replace("-dev", "");
            var parts = vs.Split('.').Select(p => int.TryParse(p, out int n) ? n : 0).Concat([0, 0, 0]).ToArray();
            state.Version = (parts[0], parts[1], parts[2]);
            state.VersionString = $"{parts[0]}.{parts[1]}.{parts[2]}";

            string? dataPath = Environment.GetEnvironmentVariable("ESPEAK_DATA_PATH");
            byte[]? dp = string.IsNullOrEmpty(dataPath) ? null : NullTerminated(dataPath);
            int rate;
            fixed (byte* p = dp)
                rate = ((delegate* unmanaged[Cdecl]<int, int, byte*, int, int>)initialize)(AudioOutputSynchronous, 0, p, 0);
            if (rate <= 0)
                throw new InvalidOperationException($"espeak_Initialize failed (data path: {dataPath ?? "library default"}); " +
                                                    "set ESPEAK_DATA_PATH to the espeak-ng-data directory.");
        }
        return state;
    }

    private static byte[] NullTerminated(string s)
    {
        var b = new byte[Encoding.UTF8.GetByteCount(s) + 1];
        Encoding.UTF8.GetBytes(s, b);
        return b;
    }

    /// <summary>The installed voices as (language, identifier) pairs, in espeak's order (most relevant first);
    /// the language is the first one a voice lists.</summary>
    public static List<(string Language, string Identifier)> Voices()
    {
        var s = Loaded.Value;
        var result = new List<(string, string)>();
        lock (Sync)
        {
            // espeak_VOICE: const char *name, *languages, *identifier, ... ; languages = priority byte + name + '\0' ...
            nint* list = (nint*)s.ListVoices(0);
            for (int i = 0; list[i] != 0; i++)
            {
                nint* voice = (nint*)list[i];
                string languages = Marshal.PtrToStringUTF8(voice[1]) ?? "";
                string identifier = Marshal.PtrToStringUTF8(voice[2]) ?? "";
                result.Add((languages.Length > 0 ? languages[1..] : "", identifier));
            }
        }
        return result;
    }

    /// <summary>Selects the voice of a language code (e.g. <c>fr-fr</c>) as phonemizer does: the first installed voice
    /// listing that language, skipping the mbrola voices.</summary>
    public static void SetVoice(string languageCode)
    {
        var s = Loaded.Value;
        string? identifier = null;
        foreach (var (lang, id) in Voices())
        {
            if (id.Replace('\\', '/').StartsWith("mb/", StringComparison.Ordinal)) continue;
            if (lang == languageCode) { identifier = id; break; }
        }
        if (identifier == null) throw new ArgumentException($"espeak-ng has no voice for language code '{languageCode}'.");
        lock (Sync)
        {
            if (s.CurrentVoice == identifier) return;
            fixed (byte* p = NullTerminated(identifier))
                if (s.SetVoiceByName(p) != 0) throw new InvalidOperationException($"espeak-ng failed to load voice '{identifier}'.");
            s.CurrentVoice = identifier;
        }
    }

    /// <summary>
    /// <c>espeak_TextToPhonemes</c> over the whole text in the given voice: IPA, phonemes of one sound joined by the tie
    /// U+0361, stress marks kept; the clauses espeak returns one by one are joined with spaces (phonemizer's
    /// <c>EspeakWrapper.text_to_phonemes</c> with <c>tie=True</c>).
    /// </summary>
    public static string TextToPhonemes(string languageCode, string text)
    {
        var s = Loaded.Value;
        const int textModeUtf8 = 1;
        const int phonemeMode = 0x02 | 0x01 << 7 | 0x0361 << 8;   // IPA | tie | tie character
        var bytes = NullTerminated(text);
        var parts = new List<string>();
        lock (Sync)
        {
            SetVoice(languageCode);
            fixed (byte* p = bytes)
            {
                byte* ptr = p;
                while (ptr != null)
                {
                    byte* r = s.TextToPhonemes(&ptr, textModeUtf8, phonemeMode);
                    if (r != null)
                    {
                        string ph = Marshal.PtrToStringUTF8((nint)r) ?? "";
                        if (ph.Length > 0) parts.Add(ph);
                    }
                }
            }
        }
        return string.Join(" ", parts);
    }
}
