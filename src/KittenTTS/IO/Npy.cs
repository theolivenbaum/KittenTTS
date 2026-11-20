using System.IO.Compression;
using System.Text;
using System.Text.RegularExpressions;

namespace KittenTTS.IO;

/// <summary>An array read from a NumPy <c>.npy</c> file, widened to float32 (C order).</summary>
public sealed record NpyArray(int[] Shape, float[] Data)
{
    public long Length => Data.LongLength;
}

/// <summary>Minimal readers for NumPy <c>.npy</c> and <c>.npz</c> files (the voice packs). Supports little-endian
/// <c>f2 / f4 / f8 / i4 / i8</c> in C order, stored or deflated archive members.</summary>
public static partial class Npy
{
    [GeneratedRegex(@"'descr':\s*'([^']+)'")] private static partial Regex DescrRe();
    [GeneratedRegex(@"'shape':\s*\(([^)]*)\)")] private static partial Regex ShapeRe();

    public static NpyArray Read(Stream s)
    {
        using var br = new BinaryReader(s, Encoding.ASCII, leaveOpen: true);
        var magic = br.ReadBytes(6);
        if (magic.Length < 6 || magic[0] != 0x93 || Encoding.ASCII.GetString(magic, 1, 5) != "NUMPY")
            throw new InvalidDataException("Not a .npy array.");
        byte major = br.ReadByte();
        br.ReadByte();
        int hlen = major == 1 ? br.ReadUInt16() : checked((int)br.ReadUInt32());
        string header = Encoding.ASCII.GetString(br.ReadBytes(hlen));
        string descr = DescrRe().Match(header).Groups[1].Value;
        if (header.Contains("'fortran_order': True")) throw new NotSupportedException(".npy arrays in Fortran order are not supported.");
        int[] shape = ShapeRe().Match(header).Groups[1].Value
            .Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).Select(int.Parse).ToArray();
        long count = shape.Aggregate(1L, (a, b) => a * b);
        var data = new float[count];
        int elem = descr switch { "<f2" => 2, "<f4" => 4, "<f8" => 8, "<i4" => 4, "<i8" => 8, _ => throw new NotSupportedException($".npy dtype {descr} is not supported.") };
        var raw = br.ReadBytes(checked((int)(count * elem)));
        if (raw.Length != count * elem) throw new InvalidDataException(".npy array is truncated.");
        var span = raw.AsSpan();
        for (long i = 0; i < count; i++)
        {
            var b = span.Slice((int)(i * elem), elem);
            data[i] = descr switch
            {
                "<f2" => (float)BitConverter.ToHalf(b),
                "<f4" => BitConverter.ToSingle(b),
                "<f8" => (float)BitConverter.ToDouble(b),
                "<i4" => BitConverter.ToInt32(b),
                _ => BitConverter.ToInt64(b),
            };
        }
        return new NpyArray(shape, data);
    }

    public static NpyArray Read(string path)
    {
        using var f = File.OpenRead(path);
        return Read(f);
    }

    /// <summary>Every array of an <c>.npz</c> archive, keyed by member name without the <c>.npy</c> suffix.</summary>
    public static Dictionary<string, NpyArray> ReadArchive(string path)
    {
        using var f = File.OpenRead(path);
        return ReadArchive(f);
    }

    public static Dictionary<string, NpyArray> ReadArchive(Stream stream)
    {
        var result = new Dictionary<string, NpyArray>(StringComparer.Ordinal);
        using var zip = new ZipArchive(stream, ZipArchiveMode.Read, leaveOpen: true);
        foreach (var entry in zip.Entries)
        {
            if (!entry.FullName.EndsWith(".npy", StringComparison.Ordinal)) continue;
            using var es = entry.Open();
            using var ms = new MemoryStream();
            es.CopyTo(ms);
            ms.Position = 0;
            result[entry.FullName[..^4]] = Read(ms);
        }
        return result;
    }
}
