using System.Text;

namespace KittenTTS.Model;

/// <summary>Development aid: with <c>KITTENTTS_DUMP_DIR</c> set, intermediate activations are written there as .npy
/// (channels-last) for comparison with the PyTorch modules.</summary>
internal static class DebugDump
{
    private static readonly string? Dir = Environment.GetEnvironmentVariable("KITTENTTS_DUMP_DIR");

    public static bool Enabled => Dir != null;

    private static int _counter;

    /// <summary>A name with a per-process sequence number (for modules called several times).</summary>
    public static string Seq(string prefix) => $"{prefix}{Interlocked.Increment(ref _counter) - 1:D2}";

    public static void Write(string name, Act a) => Write(name, a.Data, [a.T, a.C]);

    public static void Write(string name, float[] data, int[] shape)
    {
        if (Dir == null) return;
        Directory.CreateDirectory(Dir);
        string header = $"{{'descr': '<f4', 'fortran_order': False, 'shape': ({string.Join(", ", shape)}{(shape.Length == 1 ? "," : "")}), }}";
        int total = 10 + header.Length + 1;
        header = header.PadRight(header.Length + (64 - total % 64) % 64) + "\n";
        using var f = File.Create(Path.Combine(Dir, name + ".npy"));
        f.Write([0x93, (byte)'N', (byte)'U', (byte)'M', (byte)'P', (byte)'Y', 1, 0]);
        f.Write(BitConverter.GetBytes((ushort)header.Length));
        f.Write(Encoding.ASCII.GetBytes(header));
        foreach (var v in data) f.Write(BitConverter.GetBytes(v));
    }
}
