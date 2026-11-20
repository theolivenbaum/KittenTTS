using System.Numerics.Tensors;
using System.Text.Json;
using KittenTTS.IO;

namespace KittenTTS.Text.English;

/// <summary>
/// The character-level BART grapheme-to-phoneme model shipped in a model repository's <c>g2p/g2p_bart_en_us</c>
/// (PeterReid/graphemes_to_phonemes_en_us, 0.75 M parameters, Apache-2.0): one post-LN encoder layer and one decoder
/// layer of width 128 with a single head, learned positions (offset 2), GELU feed-forward, greedy decoding as
/// <c>BartForConditionalGeneration.generate(num_beams=1, do_sample=False, max_length=64)</c>. It writes misaki's
/// phoneme alphabet directly; the English front end uses it for words neither misaki nor CMUdict knows.
/// </summary>
public sealed class BartG2P
{
    private readonly int _d, _maxLen, _vocab;
    private readonly float[] _shared, _finalBias, _encPos, _decPos;
    private readonly float[] _encLnEmbW, _encLnEmbB, _decLnEmbW, _decLnEmbB;
    private readonly Layer _enc, _dec;
    private readonly Attention _cross;
    private readonly float[] _crossLnW, _crossLnB;
    private readonly Dictionary<char, int> _g2i = new();
    private readonly Dictionary<int, char> _i2p = new();

    private sealed record Linear(float[] W, float[] B, int Out, int In)
    {
        public float[] Apply(ReadOnlySpan<float> x)
        {
            var y = new float[Out];
            for (int o = 0; o < Out; o++) y[o] = TensorPrimitives.Dot(W.AsSpan(o * In, In), x) + B[o];
            return y;
        }
    }

    private sealed record Attention(Linear Q, Linear K, Linear V, Linear O);

    private sealed record Layer(Attention SelfAttn, float[] SelfLnW, float[] SelfLnB, Linear Fc1, Linear Fc2, float[] FinalLnW, float[] FinalLnB);

    public BartG2P(string directory)
    {
        using var cfgDoc = JsonDocument.Parse(File.ReadAllText(Path.Combine(directory, "config.json")));
        var cfg = cfgDoc.RootElement;
        _d = cfg.GetProperty("d_model").GetInt32();
        _maxLen = cfg.GetProperty("max_position_embeddings").GetInt32();
        _vocab = cfg.GetProperty("vocab_size").GetInt32();
        string graphemes = cfg.GetProperty("grapheme_chars").GetString()!;
        string phonemes = cfg.GetProperty("phoneme_chars").GetString()!;
        // both strings open with four placeholders for <pad> <s> </s> <unk>: a character's index is its token id
        for (int i = 4; i < graphemes.Length; i++) _g2i[graphemes[i]] = i;
        for (int i = 4; i < phonemes.Length; i++) _i2p[i] = phonemes[i];
        if (cfg.GetProperty("encoder_attention_heads").GetInt32() != 1 || cfg.GetProperty("encoder_layers").GetInt32() != 1
            || cfg.GetProperty("decoder_layers").GetInt32() != 1)
            throw new NotSupportedException("Only the one-layer, one-head G2P BART is supported.");

        using var st = SafeTensors.Open(Path.Combine(directory, "model.safetensors"));
        float[] T(string n) => st.Get(n).ToArray();
        Linear Lin(string p)
        {
            var shape = st.Get(p + ".weight").Shape;
            return new Linear(T(p + ".weight"), T(p + ".bias"), shape[0], shape[1]);
        }
        Attention Att(string p) => new(Lin(p + ".q_proj"), Lin(p + ".k_proj"), Lin(p + ".v_proj"), Lin(p + ".out_proj"));
        Layer Lay(string p) => new(Att(p + ".self_attn"), T(p + ".self_attn_layer_norm.weight"), T(p + ".self_attn_layer_norm.bias"),
                                   Lin(p + ".fc1"), Lin(p + ".fc2"), T(p + ".final_layer_norm.weight"), T(p + ".final_layer_norm.bias"));
        _shared = T("model.shared.weight");
        _finalBias = T("final_logits_bias");
        _encPos = T("model.encoder.embed_positions.weight");
        _decPos = T("model.decoder.embed_positions.weight");
        _encLnEmbW = T("model.encoder.layernorm_embedding.weight");
        _encLnEmbB = T("model.encoder.layernorm_embedding.bias");
        _decLnEmbW = T("model.decoder.layernorm_embedding.weight");
        _decLnEmbB = T("model.decoder.layernorm_embedding.bias");
        _enc = Lay("model.encoder.layers.0");
        _dec = Lay("model.decoder.layers.0");
        _cross = Att("model.decoder.layers.0.encoder_attn");
        _crossLnW = T("model.decoder.layers.0.encoder_attn_layer_norm.weight");
        _crossLnB = T("model.decoder.layers.0.encoder_attn_layer_norm.bias");
    }

    /// <summary>The characters the model can read (its grapheme alphabet).</summary>
    public IReadOnlyCollection<char> Graphemes => _g2i.Keys;

    /// <summary>Pronounces a word of grapheme characters; longer words are read in pieces of <c>max_length − 2</c>.</summary>
    public string Pronounce(string word)
    {
        var sb = new System.Text.StringBuilder();
        int step = _maxLen - 2;
        for (int i = 0; i < word.Length; i += step)
        {
            var piece = word.Substring(i, Math.Min(step, word.Length - i));
            var ids = new List<int> { 1 };
            foreach (char c in piece) ids.Add(_g2i[c]);
            ids.Add(2);
            foreach (int t in Generate(ids)) if (_i2p.TryGetValue(t, out char p)) sb.Append(p);
        }
        return sb.ToString();
    }

    private List<int> Generate(List<int> input)
    {
        int d = _d;
        // encoder
        var x = new float[input.Count][];
        for (int t = 0; t < input.Count; t++) x[t] = Embed(input[t], t, _encPos);
        for (int t = 0; t < x.Length; t++) x[t] = LayerNorm(x[t], _encLnEmbW, _encLnEmbB);
        x = EncoderLayer(x, _enc);
        var crossK = x.Select(r => _cross.K.Apply(r)).ToArray();
        var crossV = x.Select(r => _cross.V.Apply(r)).ToArray();

        // greedy decoding with a cache of the self-attention keys / values
        var output = new List<int> { 1 };
        var selfK = new List<float[]>();
        var selfV = new List<float[]>();
        float scale = 1f / MathF.Sqrt(d);
        while (output.Count < _maxLen)
        {
            int pos = output.Count - 1;
            var h = LayerNorm(Embed(output[^1], pos, _decPos), _decLnEmbW, _decLnEmbB);
            // self attention (causal through the cache)
            var a = _dec.SelfAttn;
            var q = a.Q.Apply(h);
            TensorPrimitives.Multiply(q, scale, q);
            selfK.Add(a.K.Apply(h));
            selfV.Add(a.V.Apply(h));
            var ctx = Attend(q, selfK, selfV);
            var o = a.O.Apply(ctx);
            TensorPrimitives.Add(o, h, o);
            h = LayerNorm(o, _dec.SelfLnW, _dec.SelfLnB);
            // cross attention
            var cq = _cross.Q.Apply(h);
            TensorPrimitives.Multiply(cq, scale, cq);
            var cctx = Attend(cq, crossK, crossV);
            var co = _cross.O.Apply(cctx);
            TensorPrimitives.Add(co, h, co);
            h = LayerNorm(co, _crossLnW, _crossLnB);
            h = FeedForward(h, _dec);
            // logits over the shared embedding
            int best = 0;
            float bestV = float.NegativeInfinity;
            bool forceEos = output.Count == _maxLen - 1;
            for (int v = 0; v < _vocab; v++)
            {
                float logit = TensorPrimitives.Dot(_shared.AsSpan(v * d, d), h) + _finalBias[v];
                if (logit > bestV) { bestV = logit; best = v; }
            }
            if (forceEos) best = 2;
            output.Add(best);
            if (best == 2) break;
        }
        return output;
    }

    private float[] Embed(int token, int position, float[] posTable)
    {
        var e = new float[_d];
        _shared.AsSpan(token * _d, _d).CopyTo(e);
        TensorPrimitives.Add(e, posTable.AsSpan((position + 2) * _d, _d), e);
        return e;
    }

    private float[][] EncoderLayer(float[][] x, Layer l)
    {
        float scale = 1f / MathF.Sqrt(_d);
        var q = x.Select(r => { var v = l.SelfAttn.Q.Apply(r); TensorPrimitives.Multiply(v, scale, v); return v; }).ToArray();
        var k = x.Select(r => l.SelfAttn.K.Apply(r)).ToList();
        var v = x.Select(r => l.SelfAttn.V.Apply(r)).ToList();
        var y = new float[x.Length][];
        for (int t = 0; t < x.Length; t++)
        {
            var o = l.SelfAttn.O.Apply(Attend(q[t], k, v));
            TensorPrimitives.Add(o, x[t], o);
            y[t] = FeedForward(LayerNorm(o, l.SelfLnW, l.SelfLnB), l);
        }
        return y;
    }

    private static float[] Attend(float[] q, IReadOnlyList<float[]> k, IReadOnlyList<float[]> v)
    {
        var s = new float[k.Count];
        for (int j = 0; j < k.Count; j++) s[j] = TensorPrimitives.Dot(q, k[j]);
        float max = TensorPrimitives.Max(s);
        TensorPrimitives.Subtract(s, max, s);
        TensorPrimitives.Exp(s, s);
        float inv = 1f / TensorPrimitives.Sum(s);
        var ctx = new float[q.Length];
        for (int j = 0; j < k.Count; j++) TensorPrimitives.MultiplyAdd(v[j], s[j] * inv, ctx, ctx);
        return ctx;
    }

    private static float[] FeedForward(float[] h, Layer l)
    {
        var f = l.Fc1.Apply(h);
        for (int i = 0; i < f.Length; i++) f[i] = (float)(0.5 * f[i] * (1 + Erf(f[i] / Math.Sqrt(2))));
        var o = l.Fc2.Apply(f);
        TensorPrimitives.Add(o, h, o);
        return LayerNorm(o, l.FinalLnW, l.FinalLnB);
    }

    private static float[] LayerNorm(float[] x, float[] w, float[] b)
    {
        double mean = 0;
        foreach (var v in x) mean += v;
        mean /= x.Length;
        double var = 0;
        foreach (var v in x) var += (v - mean) * (v - mean);
        float inv = (float)(1 / Math.Sqrt(var / x.Length + 1e-5));
        var y = new float[x.Length];
        for (int i = 0; i < x.Length; i++) y[i] = (float)(x[i] - mean) * inv * w[i] + b[i];
        return y;
    }

    /// <summary>erf for the exact GELU (.NET has no Math.Erf): Numerical Recipes' <c>erfcc</c> Chebyshev fit, fractional
    /// error below 1.2e-7 everywhere, i.e. float precision.</summary>
    private static double Erf(double x)
    {
        double z = Math.Abs(x);
        double t = 1.0 / (1.0 + 0.5 * z);
        double r = t * Math.Exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 +
                   t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
        return x >= 0 ? 1 - r : r - 1;
    }
}
