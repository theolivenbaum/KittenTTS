using System.Buffers.Binary;
using System.Text.Json;

namespace KittenTTS.IO;

/// <summary>
/// A <c>.safetensors</c> checkpoint (the float32 PyTorch weights of a KittenTTS 2 repository, or the BART G2P),
/// memory-mapped. F32 tensors are views over the mapping; F16 / BF16 tensors are widened to float32 on access.
/// Names are the PyTorch state-dict names.
/// </summary>
public sealed unsafe class SafeTensors : ITensorSource
{
    private readonly MappedFile _file;
    private readonly Dictionary<string, (string Dtype, int[] Shape, long Begin, long End)> _entries = new(StringComparer.Ordinal);

    private SafeTensors(string path)
    {
        _file = new MappedFile(path);
        try
        {
            if (_file.Length < 8) throw new InvalidDataException($"{path}: not a safetensors file.");
            long headerLen = BinaryPrimitives.ReadInt64LittleEndian(_file.Span(0, 8));
            if (headerLen <= 0 || 8 + headerLen > _file.Length) throw new InvalidDataException($"{path}: bad safetensors header.");
            using var doc = JsonDocument.Parse(_file.Span(8, checked((int)headerLen)).ToArray());
            long dataStart = 8 + headerLen;
            foreach (var p in doc.RootElement.EnumerateObject())
            {
                if (p.Name == "__metadata__") continue;
                string dtype = p.Value.GetProperty("dtype").GetString()!;
                int[] shape = p.Value.GetProperty("shape").EnumerateArray().Select(e => e.GetInt32()).ToArray();
                var offs = p.Value.GetProperty("data_offsets");
                long b = offs[0].GetInt64(), e = offs[1].GetInt64();
                if (dataStart + e > _file.Length) throw new InvalidDataException($"{path}: tensor {p.Name} runs past the end of the file.");
                _entries[p.Name] = (dtype, shape, dataStart + b, dataStart + e);
            }
        }
        catch
        {
            _file.Dispose();
            throw;
        }
    }

    public static SafeTensors Open(string path) => new(path);

    public IEnumerable<string> Names => _entries.Keys;
    public bool Contains(string name) => _entries.ContainsKey(name);

    public Tensor Get(string name)
    {
        if (!_entries.TryGetValue(name, out var e)) throw new KeyNotFoundException($"{_file.Path} has no tensor '{name}'.");
        long count = Tensor.Count(e.Shape);
        byte* p = _file.Ptr + e.Begin;
        switch (e.Dtype)
        {
            case "F32":
                if (e.End - e.Begin != count * 4) throw new InvalidDataException($"{name}: size mismatch.");
                return new Tensor(e.Shape, (float*)p, _file);
            case "F16":
            {
                var data = new float[count];
                var src = (Half*)p;
                for (long i = 0; i < count; i++) data[i] = (float)src[i];
                return new Tensor(e.Shape, data);
            }
            case "BF16":
            {
                var data = new float[count];
                var src = (ushort*)p;
                for (long i = 0; i < count; i++) data[i] = BitConverter.Int32BitsToSingle(src[i] << 16);
                return new Tensor(e.Shape, data);
            }
            default:
                throw new NotSupportedException($"{name}: safetensors dtype {e.Dtype} is not supported.");
        }
    }

    public void Dispose() => _file.Dispose();
}
