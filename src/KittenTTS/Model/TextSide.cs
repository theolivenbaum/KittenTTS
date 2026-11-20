using System.Numerics.Tensors;
using KittenTTS.Numerics;

namespace KittenTTS.Model;

/// <summary><c>TextEncoder</c>: embedding → depth × (conv k5 → LayerNorm → LeakyReLU 0.2) → bidirectional LSTM.</summary>
internal sealed class TextEncoder : IDisposable
{
    private readonly float[] _embedding;
    private readonly int _channels;
    private readonly List<(Conv1d Conv, float[] Gamma, float[] Beta)> _cnn = new();
    private readonly BiLstm _lstm;

    public TextEncoder(WeightReader r, int kernelSize)
    {
        var shape = r.Shape("text_encoder.embedding.weight");
        _channels = shape[1];
        _embedding = r.Get("text_encoder.embedding.weight");
        for (int i = 0; r.Has($"text_encoder.cnn.{i}.0.weight"); i++)
            _cnn.Add((Conv1d.Load(r, $"text_encoder.cnn.{i}.0", padding: (kernelSize - 1) / 2),
                      r.Get($"text_encoder.cnn.{i}.1.gamma"), r.Get($"text_encoder.cnn.{i}.1.beta")));
        _lstm = new BiLstm(r, "text_encoder.lstm");
    }

    public Act Forward(int[] tokens, Parallelism par)
    {
        var x = Embed(_embedding, _channels, tokens);
        foreach (var (conv, gamma, beta) in _cnn)
        {
            x = conv.Forward(x, par);
            x = F.LayerNorm(x, gamma, beta, 1e-5f);
            F.LeakyRelu(x, 0.2f, inPlace: true);
        }
        return _lstm.Forward(x, par);
    }

    internal static Act Embed(float[] table, int dim, int[] tokens)
    {
        var x = new Act(tokens.Length, dim);
        int rows = table.Length / dim;
        for (int t = 0; t < tokens.Length; t++)
        {
            if ((uint)tokens[t] >= (uint)rows) throw new ArgumentOutOfRangeException(nameof(tokens), $"Token id {tokens[t]} is outside the vocabulary of {rows}.");
            table.AsSpan(tokens[t] * dim, dim).CopyTo(x.Row(t));
        }
        return x;
    }

    public void Dispose()
    {
        foreach (var c in _cnn) c.Conv.Dispose();
        _lstm.Dispose();
    }
}

/// <summary>
/// PL-BERT: a Hugging Face <c>AlbertModel</c> (one shared layer applied <c>num_hidden_layers</c> times, absolute
/// positions, token type 0, <c>gelu_new</c>), returning the last hidden state. A single unpadded sequence, so the
/// attention mask is all ones.
/// </summary>
internal sealed class Albert : IDisposable
{
    private const string L = "bert.encoder.albert_layer_groups.0.albert_layers.0.";
    private readonly float[] _word, _pos, _type, _embLnW, _embLnB;
    private readonly int _embDim, _hidden, _heads, _layers;
    private readonly float _lnEps;
    private readonly Linear _mapIn, _q, _k, _v, _dense, _ffn, _ffnOut;
    private readonly float[] _attLnW, _attLnB, _fullLnW, _fullLnB;

    public Albert(WeightReader r, int heads, int layers, float layerNormEps)
    {
        _embDim = r.Shape("bert.embeddings.word_embeddings.weight")[1];
        _word = r.Get("bert.embeddings.word_embeddings.weight");
        _pos = r.Get("bert.embeddings.position_embeddings.weight");
        _type = r.Get("bert.embeddings.token_type_embeddings.weight");
        _embLnW = r.Get("bert.embeddings.LayerNorm.weight");
        _embLnB = r.Get("bert.embeddings.LayerNorm.bias");
        _mapIn = Linear.Load(r, "bert.encoder.embedding_hidden_mapping_in");
        _q = Linear.Load(r, L + "attention.query");
        _k = Linear.Load(r, L + "attention.key");
        _v = Linear.Load(r, L + "attention.value");
        _dense = Linear.Load(r, L + "attention.dense");
        _ffn = Linear.Load(r, L + "ffn");
        _ffnOut = Linear.Load(r, L + "ffn_output");
        _attLnW = r.Get(L + "attention.LayerNorm.weight");
        _attLnB = r.Get(L + "attention.LayerNorm.bias");
        _fullLnW = r.Get(L + "full_layer_layer_norm.weight");
        _fullLnB = r.Get(L + "full_layer_layer_norm.bias");
        _hidden = _mapIn.Out;
        _heads = heads;
        _layers = layers;
        _lnEps = layerNormEps;
        MaxPositions = _pos.Length / _embDim;
    }

    public int MaxPositions { get; }

    public Act Forward(int[] tokens, Parallelism par)
    {
        int T = tokens.Length;
        if (T > MaxPositions) throw new ArgumentException($"PL-BERT takes at most {MaxPositions} tokens, got {T}.");
        var e = TextEncoder.Embed(_word, _embDim, tokens);
        for (int t = 0; t < T; t++)
        {
            var row = e.Row(t);
            // word + token_type + position, the order of BertEmbeddings / AlbertEmbeddings
            TensorPrimitives.Add(row, _type.AsSpan(0, _embDim), row);
            TensorPrimitives.Add(row, _pos.AsSpan(t * _embDim, _embDim), row);
        }
        e = F.LayerNorm(e, _embLnW, _embLnB, _lnEps);
        DebugDump.Write("bert_emb_ln", e);
        var h = _mapIn.Forward(e, par);
        DebugDump.Write("bert_map_in", h);
        for (int l = 0; l < _layers; l++)
        {
            h = Layer(h, par);
            DebugDump.Write($"bert_layer{l:D2}", h);
        }
        return h;
    }

    private Act Layer(Act h, Parallelism par)
    {
        int T = h.T, D = _hidden, hd = D / _heads;
        var q = _q.Forward(h, par);
        var k = _k.Forward(h, par);
        var v = _v.Forward(h, par);
        var ctx = new Act(T, D);
        float scale = 1f / MathF.Sqrt(hd);
        par.For(_heads, head =>
        {
            var scores = new float[T];
            var acc = new float[hd];
            int o = head * hd;
            for (int i = 0; i < T; i++)
            {
                var qi = q.Data.AsSpan(i * D + o, hd);
                for (int j = 0; j < T; j++) scores[j] = TensorPrimitives.Dot(qi, k.Data.AsSpan(j * D + o, hd)) * scale;
                float max = TensorPrimitives.Max(scores);
                TensorPrimitives.Subtract(scores, max, scores);
                TensorPrimitives.Exp(scores, scores);
                float inv = 1f / TensorPrimitives.Sum(scores);
                Array.Clear(acc);
                for (int j = 0; j < T; j++) TensorPrimitives.MultiplyAdd(v.Data.AsSpan(j * D + o, hd), scores[j] * inv, acc, acc);
                acc.CopyTo(ctx.Data.AsSpan(i * D + o, hd));
            }
        });
        var proj = _dense.Forward(ctx, par);
        F.AddInPlace(proj, h);
        var att = F.LayerNorm(proj, _attLnW, _attLnB, _lnEps);
        var f = _ffn.Forward(att, par);
        GeluNew(f.Data);
        var fo = _ffnOut.Forward(f, par);
        F.AddInPlace(fo, att);
        return F.LayerNorm(fo, _fullLnW, _fullLnB, _lnEps);
    }

    /// <summary><c>gelu_new</c>: 0.5 x (1 + tanh(√(2/π) (x + 0.044715 x³))).</summary>
    private static void GeluNew(float[] x)
    {
        const float c = 0.7978845608028654f;
        for (long i = 0; i < x.LongLength; i++)
        {
            float v = x[i];
            x[i] = 0.5f * v * (1f + MathF.Tanh(c * (v + 0.044715f * v * v * v)));
        }
    }

    public void Dispose()
    {
        _mapIn.Dispose(); _q.Dispose(); _k.Dispose(); _v.Dispose(); _dense.Dispose(); _ffn.Dispose(); _ffnOut.Dispose();
    }
}

/// <summary>
/// The prosody predictor: <c>DurationEncoder</c> (LSTM ↔ AdaLayerNorm stack over [text features, style]), the
/// duration LSTM and projection, and <c>F0Ntrain</c> (shared LSTM, then the F0 and energy (N) branches of
/// <c>AdainResBlk1d</c>, the second of each upsampling ×2).
/// </summary>
internal sealed class ProsodyPredictor : IDisposable
{
    private readonly List<(BiLstm Lstm, VectorLinear Fc)> _encoder = new();
    private readonly BiLstm _lstm, _shared;
    private readonly Linear _durationProj;
    private readonly AdainResBlk1d[] _f0, _n;
    private readonly Conv1d _f0Proj, _nProj;
    private readonly int _hidden;

    public ProsodyPredictor(WeightReader r)
    {
        for (int i = 0; r.Has($"predictor.text_encoder.lstms.{i}.weight_ih_l0"); i += 2)
            _encoder.Add((new BiLstm(r, $"predictor.text_encoder.lstms.{i}"), VectorLinear.Load(r, $"predictor.text_encoder.lstms.{i + 1}.fc")));
        _lstm = new BiLstm(r, "predictor.lstm");
        _shared = new BiLstm(r, "predictor.shared");
        _durationProj = Linear.Load(r, "predictor.duration_proj.linear_layer");
        _f0 = [new(r, "predictor.F0.0", false), new(r, "predictor.F0.1", true), new(r, "predictor.F0.2", false)];
        _n = [new(r, "predictor.N.0", false), new(r, "predictor.N.1", true), new(r, "predictor.N.2", false)];
        _f0Proj = Conv1d.Load(r, "predictor.F0_proj");
        _nProj = Conv1d.Load(r, "predictor.N_proj");
        _hidden = _lstm.H * 2;
    }

    public int MaxDuration => _durationProj.Out;

    /// <summary>DurationEncoder: [T, hidden] text features → [T, hidden + style].</summary>
    public Act EncodeText(Act d, ReadOnlySpan<float> s, Parallelism par)
    {
        var styleAct = Repeat(s, d.T);
        var x = Act.ConcatChannels(d, styleAct);
        foreach (var (lstm, fc) in _encoder)
        {
            x = lstm.Forward(x, par);
            // AdaLayerNorm: layer_norm without affine, then (1 + γ) x + β
            var gb = fc.Forward(s);
            x = F.LayerNorm(x, null, null, 1e-5f);
            int c = x.C;
            for (int t = 0; t < x.T; t++)
            {
                var row = x.Row(t);
                for (int i = 0; i < c; i++) row[i] = (1f + gb[i]) * row[i] + gb[c + i];
            }
            x = Act.ConcatChannels(x, styleAct);
        }
        return x;
    }

    /// <summary>Durations in frames: <c>round(Σ sigmoid(duration_proj(lstm(d))) / speed)</c>, at least 1.</summary>
    public int[] Durations(Act d, float speed, Parallelism par)
    {
        var x = _lstm.Forward(d, par);
        var logits = _durationProj.Forward(x, par);
        var dur = new int[d.T];
        for (int t = 0; t < d.T; t++)
        {
            var row = logits.Row(t);
            float s = 0;
            for (int i = 0; i < row.Length; i++) s += 1f / (1f + MathF.Exp(-row[i]));
            float v = s / speed;
            dur[t] = Math.Max(1, (int)MathF.Round(v, MidpointRounding.ToEven));
        }
        return dur;
    }

    /// <summary>F0 and energy curves at twice the frame rate, from the frame-aligned encoder output.</summary>
    public (float[] F0, float[] N) F0N(Act en, ReadOnlySpan<float> s, Parallelism par)
    {
        var x = _shared.Forward(en, par);
        var f0 = x;
        foreach (var b in _f0) f0 = b.Forward(f0, s, par);
        f0 = _f0Proj.Forward(f0, par);
        var n = x;
        foreach (var b in _n) n = b.Forward(n, s, par);
        n = _nProj.Forward(n, par);
        return (f0.Data, n.Data);
    }

    private static Act Repeat(ReadOnlySpan<float> s, int t)
    {
        var a = new Act(t, s.Length);
        for (int i = 0; i < t; i++) s.CopyTo(a.Row(i));
        return a;
    }

    public void Dispose()
    {
        foreach (var e in _encoder) e.Lstm.Dispose();
        _lstm.Dispose(); _shared.Dispose(); _durationProj.Dispose();
        foreach (var b in _f0) b.Dispose();
        foreach (var b in _n) b.Dispose();
        _f0Proj.Dispose(); _nProj.Dispose();
    }
}
