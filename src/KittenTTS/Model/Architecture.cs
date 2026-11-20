using System.Text.Json;

namespace KittenTTS.Model;

/// <summary>The decoder part of a model's <c>architecture</c> in config.json.</summary>
public sealed record DecoderArchitecture(
    string Type,
    int[] ResblockKernelSizes,
    int[] UpsampleRates,
    int UpsampleInitialChannel,
    int[][] ResblockDilationSizes,
    int[] UpsampleKernelSizes,
    int IstftNFft,
    int IstftHop);

/// <summary>A model's <c>architecture</c> entry in config.json (the shapes of kitten-tts <c>build_model</c>).</summary>
public sealed record Architecture(
    int HiddenDim,
    int Layers,
    int Tokens,
    int StyleDim,
    int MaxDuration,
    int TextEncoderKernelSize,
    DecoderArchitecture Decoder,
    int AlbertHeads,
    int AlbertLayers,
    float AlbertLayerNormEps)
{
    public static Architecture FromJson(JsonElement a)
    {
        var d = a.GetProperty("decoder");
        var b = a.GetProperty("albert");
        static int[] Ints(JsonElement e) => e.EnumerateArray().Select(x => x.GetInt32()).ToArray();
        return new Architecture(
            a.GetProperty("hidden_dim").GetInt32(),
            a.GetProperty("n_layer").GetInt32(),
            a.GetProperty("n_token").GetInt32(),
            a.GetProperty("style_dim").GetInt32(),
            a.GetProperty("max_dur").GetInt32(),
            a.TryGetProperty("text_encoder_kernel_size", out var k) ? k.GetInt32() : 5,
            new DecoderArchitecture(
                d.GetProperty("type").GetString()!,
                Ints(d.GetProperty("resblock_kernel_sizes")),
                Ints(d.GetProperty("upsample_rates")),
                d.GetProperty("upsample_initial_channel").GetInt32(),
                d.GetProperty("resblock_dilation_sizes").EnumerateArray().Select(Ints).ToArray(),
                Ints(d.GetProperty("upsample_kernel_sizes")),
                d.GetProperty("gen_istft_n_fft").GetInt32(),
                d.GetProperty("gen_istft_hop_size").GetInt32()),
            b.GetProperty("num_attention_heads").GetInt32(),
            b.GetProperty("num_hidden_layers").GetInt32(),
            b.TryGetProperty("layer_norm_eps", out var eps) ? eps.GetSingle() : 1e-12f);
    }
}
