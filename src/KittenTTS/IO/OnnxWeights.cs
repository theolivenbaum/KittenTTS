using System.Text;

namespace KittenTTS.IO;

/// <summary>
/// The weights of a KittenTTS 2 nano / pico / mini <b>int8</b> ONNX graph (<c>kitten_tts_2_*.onnx</c>, the default
/// weights of the Python package) presented under the PyTorch parameter names of the float32 checkpoint, read with a
/// minimal protobuf parser (no ONNX Runtime). Weights are dequantized once at load (<c>DequantizeLinear</c>: int8,
/// symmetric, per output channel), so inference runs the same float32 arithmetic as ONNX Runtime does on the graph.
/// <para>
/// Most initializers keep their module path (<c>m.decoder.encode.conv1.weight_q</c>; weight norm is already folded
/// into <c>weight</c>). The exporter renamed the rest, which are recovered from the graph:
/// MatMul weights (ALBERT, <c>bert_encoder</c>, <c>duration_proj</c>, <c>l_linear</c>) from the MatMul node's scope
/// (or, in a graph without node names, the order of first use), LSTM weights from the LSTM nodes in execution order
/// (gates reordered from ONNX <c>iofc</c> to PyTorch <c>ifgo</c>), and the affine parameters of the decoder's
/// instance norms (<c>[1, C, 1]</c> Mul / Add constants) in execution order, see <see cref="AffinePairs"/>.
/// </para>
/// The int4 graphs (<c>*_int4.onnx</c>) obfuscate every tensor and node name and are not supported.
/// </summary>
public sealed unsafe class OnnxWeights : ITensorSource
{
    private readonly MappedFile _file;
    private readonly Dictionary<string, Init> _inits = new(StringComparer.Ordinal);
    private readonly Dictionary<string, Node> _producers = new(StringComparer.Ordinal);
    private readonly Dictionary<string, Func<Tensor>> _tensors = new(StringComparer.Ordinal);
    private readonly List<(string Mul, string Add)> _affine = new();

    private sealed record Init(string Name, int[] Dims, int DataType, long Offset, long Length);
    private sealed record Node(string OpType, string Name, string[] Inputs, string[] Outputs, Dictionary<string, long> Ints);

    // ONNX TensorProto data types
    private const int Float = 1, UInt8 = 2, Int8 = 3, Float16 = 10, Int4 = 22, UInt4 = 21;

    /// <summary>The PyTorch modules whose weights the ONNX exporter gave numeric names, in order of first use.</summary>
    public static readonly string[] MatMulModules =
    [
        "bert.encoder.embedding_hidden_mapping_in",
        "bert.encoder.albert_layer_groups.0.albert_layers.0.attention.query",
        "bert.encoder.albert_layer_groups.0.albert_layers.0.attention.key",
        "bert.encoder.albert_layer_groups.0.albert_layers.0.attention.value",
        "bert.encoder.albert_layer_groups.0.albert_layers.0.attention.dense",
        "bert.encoder.albert_layer_groups.0.albert_layers.0.ffn",
        "bert.encoder.albert_layer_groups.0.albert_layers.0.ffn_output",
        "bert_encoder",
        "predictor.duration_proj.linear_layer",
        "decoder.generator.m_source.l_linear",
    ];

    /// <summary>The bidirectional LSTMs, in execution order.</summary>
    public static readonly string[] LstmModules =
    [
        "text_encoder.lstm",
        "predictor.text_encoder.lstms.0",
        "predictor.text_encoder.lstms.2",
        "predictor.text_encoder.lstms.4",
        "predictor.lstm",
        "predictor.shared",
    ];

    private OnnxWeights(string path)
    {
        _file = new MappedFile(path);
        try
        {
            Load(path);
        }
        catch
        {
            _file.Dispose();
            throw;
        }
    }

    public static OnnxWeights Open(string path) => new(path);

    public IEnumerable<string> Names => _tensors.Keys;
    public bool Contains(string name) => _tensors.ContainsKey(name);
    public Tensor Get(string name) => _tensors.TryGetValue(name, out var f) ? f() : throw new KeyNotFoundException($"{_file.Path} has no tensor '{name}'.");

    /// <summary>The unnamed per-channel affine parameters (<c>[1, C, 1]</c> Mul then Add constants) of the decoder's
    /// instance norms, in execution order. <see cref="AssignAffine"/> names them.</summary>
    public int AffinePairs => _affine.Count;

    /// <summary>Registers the unnamed instance-norm affine parameters as <c>{module}.weight</c> / <c>{module}.bias</c>
    /// for the given instance-norm modules in execution order (e.g. <c>decoder.encode.norm1.norm</c>).</summary>
    public void AssignAffine(IReadOnlyList<string> modules)
    {
        if (modules.Count != _affine.Count)
            throw new InvalidDataException($"{_file.Path}: found {_affine.Count} unnamed instance-norm affine pairs, expected {modules.Count}.");
        for (int i = 0; i < modules.Count; i++)
        {
            var (mul, add) = _affine[i];
            _tensors[modules[i] + ".weight"] = () => Squeeze(Resolve(mul));
            _tensors[modules[i] + ".bias"] = () => Squeeze(Resolve(add));
        }
    }

    private static Tensor Squeeze(Tensor t) => new([(int)t.Length], t.ToArray());

    private void Load(string path)
    {
        var model = new ProtoReader(_file, 0, _file.Length);
        long gOff = -1, gLen = 0;
        while (model.Next(out int field, out int wire))
        {
            if (field == 7 && wire == 2) (gOff, gLen) = model.Bytes();
            else model.Skip(wire);
        }
        if (gOff < 0) throw new InvalidDataException($"{path}: no graph in the ONNX model.");

        var nodes = new List<Node>();
        var g = new ProtoReader(_file, gOff, gLen);
        while (g.Next(out int field, out int wire))
        {
            if (field == 1 && wire == 2) { var (o, l) = g.Bytes(); nodes.Add(ReadNode(o, l)); }
            else if (field == 5 && wire == 2) { var (o, l) = g.Bytes(); var t = ReadTensor(o, l); _inits[t.Name] = t; }
            else g.Skip(wire);
        }
        foreach (var n in nodes)
            foreach (var o in n.Outputs) _producers[o] = n;

        if (!_inits.Keys.Any(k => k.StartsWith("m.", StringComparison.Ordinal)))
            throw new NotSupportedException($"{path}: this ONNX graph has no named weights (the int4 graphs obfuscate them); use the int8 graph or the safetensors weights.");

        // 1. Named weights: m.<module>.<param> (float) or m.<module>.<param>_q (+ _scale, _zp) behind a DequantizeLinear.
        foreach (var t in _inits.Values)
        {
            if (!t.Name.StartsWith("m.", StringComparison.Ordinal)) continue;
            string name = t.Name[2..];
            if (name.EndsWith("_scale", StringComparison.Ordinal) || name.EndsWith("_zp", StringComparison.Ordinal)) continue;
            if (name.EndsWith("_q", StringComparison.Ordinal))
            {
                string dq = t.Name[..^2] + "_dq";
                string key = name[..^2];
                if (_producers.ContainsKey(dq)) _tensors[key] = () => Resolve(dq);
                continue;
            }
            string n0 = t.Name;
            _tensors[name] = () => Resolve(n0);
        }

        // 2. MatMul weights: [in, out] constants, transposed back to the PyTorch [out, in].
        var seen = new HashSet<string>(StringComparer.Ordinal);
        var matmuls = new List<(Node Node, string Weight)>();
        foreach (var n in nodes)
        {
            if (n.OpType != "MatMul" || n.Inputs.Length != 2 || !IsConstant(n.Inputs[1])) continue;
            if (!IsUnnamedConstant(n.Inputs[1]) || !seen.Add(n.Inputs[1])) continue;
            matmuls.Add((n, n.Inputs[1]));
        }
        if (matmuls.Count != MatMulModules.Length)
            throw new InvalidDataException($"{path}: found {matmuls.Count} unnamed MatMul weights, expected {MatMulModules.Length}.");
        for (int i = 0; i < matmuls.Count; i++)
        {
            string module = ScopeModule(matmuls[i].Node.Name) ?? MatMulModules[i];
            if (!MatMulModules.Contains(module)) module = MatMulModules[i];
            string w = matmuls[i].Weight;
            _tensors[module + ".weight"] = () => Transpose2D(Resolve(w));
        }

        // 3. LSTMs in execution order.
        var lstms = nodes.Where(n => n.OpType == "LSTM").ToList();
        if (lstms.Count != LstmModules.Length)
            throw new InvalidDataException($"{path}: found {lstms.Count} LSTM nodes, expected {LstmModules.Length}.");
        for (int i = 0; i < lstms.Count; i++)
        {
            var n = lstms[i];
            string p = LstmModules[i] + ".";
            string W = n.Inputs[1], R = n.Inputs[2];
            string? B = n.Inputs.Length > 3 && n.Inputs[3].Length > 0 ? n.Inputs[3] : null;
            for (int dir = 0; dir < 2; dir++)
            {
                string sfx = dir == 0 ? "l0" : "l0_reverse";
                int d = dir;
                _tensors[$"{p}weight_ih_{sfx}"] = () => LstmGates(Resolve(W), d, 0, 1);
                _tensors[$"{p}weight_hh_{sfx}"] = () => LstmGates(Resolve(R), d, 0, 1);
                if (B != null)
                {
                    _tensors[$"{p}bias_ih_{sfx}"] = () => LstmGates(Resolve(B), d, 0, 2);
                    _tensors[$"{p}bias_hh_{sfx}"] = () => LstmGates(Resolve(B), d, 1, 2);
                }
            }
        }

        // 4. Unnamed [1, C, 1] Mul constants followed by an unnamed [1, C, 1] Add constant: instance-norm affine.
        var consumers = new Dictionary<string, List<Node>>(StringComparer.Ordinal);
        foreach (var n in nodes)
            foreach (var x in n.Inputs) (consumers.TryGetValue(x, out var l) ? l : consumers[x] = new()).Add(n);
        foreach (var n in nodes)
        {
            if (n.OpType != "Mul" || n.Inputs.Length != 2) continue;
            int ci = IsChannelConstant(n.Inputs[1]) ? 1 : IsChannelConstant(n.Inputs[0]) ? 0 : -1;
            if (ci < 0 || !consumers.TryGetValue(n.Outputs[0], out var next) || next.Count != 1 || next[0].OpType != "Add") continue;
            var add = next[0];
            string? addConst = add.Inputs.FirstOrDefault(x => x != n.Outputs[0] && IsChannelConstant(x));
            if (addConst != null) _affine.Add((n.Inputs[ci], addConst));
        }
    }

    /// <summary>"/bert/encoder/albert_layer_groups.0/albert_layers.0/attention/query_3/MatMul" → the query module.</summary>
    private static string? ScopeModule(string nodeName)
    {
        if (string.IsNullOrEmpty(nodeName) || !nodeName.StartsWith('/')) return null;
        var parts = nodeName.Trim('/').Split('/').ToList();
        if (parts.Count < 2) return null;
        parts.RemoveAt(parts.Count - 1);   // op name
        for (int i = 0; i < parts.Count; i++)
        {
            // repeated calls of a shared module get a numeric suffix: query_3 → query
            int us = parts[i].LastIndexOf('_');
            if (us > 0 && int.TryParse(parts[i].AsSpan(us + 1), out _)) parts[i] = parts[i][..us];
        }
        string joined = string.Join('.', parts);
        if (joined == "bert_encoder") return "bert_encoder";
        if (joined == "duration_proj.linear_layer") return "predictor.duration_proj.linear_layer";
        if (joined == "l_linear") return "decoder.generator.m_source.l_linear";
        return MatMulModules.FirstOrDefault(m => m == joined);
    }

    private bool IsConstant(string name)
    {
        if (_inits.ContainsKey(name)) return true;
        return _producers.TryGetValue(name, out var p) && p.OpType is "DequantizeLinear" or "Cast" or "Identity" && IsConstant(p.Inputs[0]);
    }

    private bool IsUnnamedConstant(string name)
    {
        var root = RootInit(name);
        return root != null && !root.Name.StartsWith("m.", StringComparison.Ordinal);
    }

    private Init? RootInit(string name)
    {
        if (_inits.TryGetValue(name, out var t)) return t;
        return _producers.TryGetValue(name, out var p) && p.OpType is "DequantizeLinear" or "Cast" or "Identity" ? RootInit(p.Inputs[0]) : null;
    }

    private bool IsChannelConstant(string name)
    {
        var root = RootInit(name);
        return root != null && !root.Name.StartsWith("m.", StringComparison.Ordinal) && root.Dims.Length == 3 && root.Dims[0] == 1 && root.Dims[2] == 1 && root.Dims[1] > 1;
    }

    /// <summary>Evaluates a constant: an initializer, or a DequantizeLinear / Cast / Identity chain over initializers.</summary>
    private Tensor Resolve(string name)
    {
        if (_inits.TryGetValue(name, out var t)) return ReadFloat(t);
        if (!_producers.TryGetValue(name, out var n)) throw new InvalidDataException($"{_file.Path}: '{name}' is not a constant.");
        switch (n.OpType)
        {
            case "Cast":
            case "Identity":
                return Resolve(n.Inputs[0]);
            case "DequantizeLinear":
            {
                var q = _inits[n.Inputs[0]];
                var scale = Resolve(n.Inputs[1]).ToArray();
                float[]? zp = n.Inputs.Length > 2 && n.Inputs[2].Length > 0 ? ReadIntsAsFloat(_inits[n.Inputs[2]]) : null;
                int axis = (int)n.Ints.GetValueOrDefault("axis", 1);
                int block = (int)n.Ints.GetValueOrDefault("block_size", 0);
                return Dequantize(q, scale, zp, axis, block);
            }
            default:
                throw new InvalidDataException($"{_file.Path}: '{name}' is produced by {n.OpType}, not a constant.");
        }
    }

    private Tensor ReadFloat(Init t)
    {
        long count = Tensor.Count(t.Dims);
        byte* p = _file.Ptr + t.Offset;
        switch (t.DataType)
        {
            case Float:
                if (t.Length != count * 4) throw new InvalidDataException($"{t.Name}: size mismatch.");
                return new Tensor(t.Dims, (float*)p, _file);
            case Float16:
            {
                var data = new float[count];
                for (long i = 0; i < count; i++) data[i] = (float)((Half*)p)[i];
                return new Tensor(t.Dims, data);
            }
            default:
                throw new NotSupportedException($"{t.Name}: ONNX data type {t.DataType} is not a float type.");
        }
    }

    private float[] ReadIntsAsFloat(Init t)
    {
        long count = Tensor.Count(t.Dims);
        var r = new float[count];
        byte* p = _file.Ptr + t.Offset;
        for (long i = 0; i < count; i++) r[i] = QuantValue(t.DataType, p, i);
        return r;
    }

    private static float QuantValue(int type, byte* p, long i) => type switch
    {
        Int8 => ((sbyte*)p)[i],
        UInt8 => p[i],
        Int4 => (sbyte)((((i & 1) == 0 ? p[i >> 1] & 0xF : p[i >> 1] >> 4) << 4)) >> 4,
        UInt4 => (i & 1) == 0 ? p[i >> 1] & 0xF : p[i >> 1] >> 4,
        _ => throw new NotSupportedException($"ONNX quantized data type {type} is not supported."),
    };

    /// <summary><c>(q - zp) · scale</c> per axis (block_size = 0) or per block of <paramref name="block"/> along the axis.</summary>
    private Tensor Dequantize(Init q, float[] scale, float[]? zp, int axis, int block)
    {
        int rank = q.Dims.Length;
        if (axis < 0) axis += rank;
        long count = Tensor.Count(q.Dims);
        long inner = 1;
        for (int d = axis + 1; d < rank; d++) inner *= q.Dims[d];
        int axisLen = q.Dims[axis];
        int scaleAxisLen = block > 0 ? (axisLen + block - 1) / block : axisLen;
        var data = new float[count];
        byte* p = _file.Ptr + q.Offset;
        for (long i = 0; i < count; i++)
        {
            long outer = i / (inner * axisLen);
            int a = (int)(i / inner % axisLen);
            long rest = i % inner;
            int sa = block > 0 ? a / block : a;
            // per-axis: 1-D scale of the axis length; blocked: scale of the input's rank, the axis divided by the block
            long si = scale.Length == 1 ? 0 : block == 0 ? a : (outer * scaleAxisLen + sa) * inner + rest;
            float z = zp == null ? 0 : zp.Length == 1 ? zp[0] : zp[si];
            data[i] = (QuantValue(q.DataType, p, i) - z) * scale[si];
        }
        return new Tensor(q.Dims, data);
    }

    private static Tensor Transpose2D(Tensor t)
    {
        if (t.Shape.Length != 2) throw new InvalidDataException($"MatMul weight of shape {t.ShapeString} is not 2-D.");
        int rows = t.Shape[0], cols = t.Shape[1];
        var src = t.Span;
        var data = new float[(long)rows * cols];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++) data[(long)c * rows + r] = src[r * cols + c];
        return new Tensor([cols, rows], data);
    }

    /// <summary>Direction <paramref name="dir"/>, part <paramref name="part"/> of <paramref name="parts"/> of an ONNX LSTM
    /// tensor (<c>[2, 4H, in]</c> or <c>[2, 8H]</c>), gates reordered from ONNX i, o, f, c to PyTorch i, f, g(c), o.</summary>
    private static Tensor LstmGates(Tensor t, int dir, int part, int parts)
    {
        var src = t.Span;
        long perDir = t.Length / t.Shape[0];
        long per = perDir / parts;
        long gate = per / 4;
        int[] order = [0, 2, 3, 1];
        var data = new float[per];
        for (int gi = 0; gi < 4; gi++)
            src.Slice((int)(dir * perDir + part * per + order[gi] * gate), (int)gate).CopyTo(data.AsSpan((int)(gi * gate)));
        int[] shape = t.Shape.Length == 3 ? [t.Shape[1], t.Shape[2]] : [(int)per];
        return new Tensor(shape, data);
    }

    private Node ReadNode(long off, long len)
    {
        var r = new ProtoReader(_file, off, len);
        var inputs = new List<string>();
        var outputs = new List<string>();
        var ints = new Dictionary<string, long>(StringComparer.Ordinal);
        string op = "", name = "";
        while (r.Next(out int field, out int wire))
        {
            if (field == 1 && wire == 2) inputs.Add(r.String());
            else if (field == 2 && wire == 2) outputs.Add(r.String());
            else if (field == 3 && wire == 2) name = r.String();
            else if (field == 4 && wire == 2) op = r.String();
            else if (field == 5 && wire == 2)
            {
                var (o, l) = r.Bytes();
                var a = new ProtoReader(_file, o, l);
                string an = "";
                long? iv = null;
                while (a.Next(out int f2, out int w2))
                {
                    if (f2 == 1 && w2 == 2) an = a.String();
                    else if (f2 == 3 && w2 == 0) iv = (long)a.Varint();
                    else a.Skip(w2);
                }
                if (iv.HasValue) ints[an] = iv.Value;
            }
            else r.Skip(wire);
        }
        return new Node(op, name, inputs.ToArray(), outputs.ToArray(), ints);
    }

    private Init ReadTensor(long off, long len)
    {
        var r = new ProtoReader(_file, off, len);
        var dims = new List<int>();
        int dataType = 0;
        string name = "";
        long rawOff = -1, rawLen = 0, floatOff = -1, floatLen = 0, i32Off = -1, i32Len = 0;
        while (r.Next(out int field, out int wire))
        {
            switch (field)
            {
                case 1 when wire == 0: dims.Add((int)r.Varint()); break;
                case 1 when wire == 2: { var (o, l) = r.Bytes(); var p = new ProtoReader(_file, o, l); while (!p.End) dims.Add((int)p.Varint()); break; }
                case 2: dataType = (int)r.Varint(); break;
                case 4 when wire == 2: (floatOff, floatLen) = r.Bytes(); break;
                case 5 when wire == 2: (i32Off, i32Len) = r.Bytes(); break;
                case 8: name = r.String(); break;
                case 9: (rawOff, rawLen) = r.Bytes(); break;
                case 14: throw new NotSupportedException($"{_file.Path}: external tensor data is not supported.");
                default: r.Skip(wire); break;
            }
        }
        if (rawOff >= 0) return new Init(name, dims.ToArray(), dataType, rawOff, rawLen);
        if (floatOff >= 0) return new Init(name, dims.ToArray(), dataType, floatOff, floatLen);
        if (i32Off >= 0 && dataType is Int8 or UInt8)
            throw new NotSupportedException($"{_file.Path}: tensor {name} stores int8 data as varints, which is not supported.");
        return new Init(name, dims.ToArray(), dataType, 0, 0);
    }

    public void Dispose() => _file.Dispose();

    /// <summary>Protobuf wire-format reader over a byte range of a memory-mapped file.</summary>
    private struct ProtoReader(MappedFile file, long offset, long length)
    {
        private long _pos = offset;
        private readonly long _end = offset + length;

        public readonly bool End => _pos >= _end;

        public bool Next(out int field, out int wire)
        {
            if (_pos >= _end) { field = wire = 0; return false; }
            ulong tag = Varint();
            field = (int)(tag >> 3);
            wire = (int)(tag & 7);
            return true;
        }

        public ulong Varint()
        {
            ulong v = 0;
            int shift = 0;
            byte* p = file.Ptr;
            while (true)
            {
                byte b = p[_pos++];
                v |= (ulong)(b & 0x7F) << shift;
                if (b < 0x80) return v;
                shift += 7;
            }
        }

        public (long Offset, long Length) Bytes()
        {
            long len = (long)Varint();
            long o = _pos;
            _pos += len;
            if (_pos > _end) throw new InvalidDataException($"{file.Path}: malformed protobuf.");
            return (o, len);
        }

        public string String()
        {
            var (o, l) = Bytes();
            return Encoding.UTF8.GetString(file.Span(o, (int)l));
        }

        public void Skip(int wire)
        {
            switch (wire)
            {
                case 0: Varint(); break;
                case 1: _pos += 8; break;
                case 2: Bytes(); break;
                case 5: _pos += 4; break;
                default: throw new InvalidDataException($"{file.Path}: unsupported protobuf wire type {wire}.");
            }
        }
    }
}
