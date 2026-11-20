namespace KittenTTS.Model;

/// <summary>An activation in channels-last layout: <c>T</c> time steps (rows) of <c>C</c> channels, row-major. The
/// PyTorch modules work on <c>[B, C, T]</c>; this layout makes every convolution a GEMM over contiguous rows.</summary>
internal sealed class Act
{
    public Act(int t, int c)
    {
        T = t;
        C = c;
        Data = new float[(long)t * c];
    }

    /// <summary>An activation whose every element the caller overwrites (skips zeroing the buffer).</summary>
    public static Act Uninit(int t, int c) => new(t, c, GC.AllocateUninitializedArray<float>(checked(t * c)));

    public Act(int t, int c, float[] data)
    {
        if (data.LongLength != (long)t * c) throw new ArgumentException("Data length does not match the shape.");
        T = t;
        C = c;
        Data = data;
    }

    public int T { get; }
    public int C { get; }
    public float[] Data { get; }

    public Span<float> Row(int t) => Data.AsSpan(t * C, C);

    public float this[int t, int c]
    {
        get => Data[(long)t * C + c];
        set => Data[(long)t * C + c] = value;
    }

    /// <summary>Concatenates activations of equal length along the channel axis (<c>torch.cat(dim=1)</c>).</summary>
    public static Act ConcatChannels(params Act[] parts)
    {
        int t = parts[0].T;
        int c = 0;
        foreach (var p in parts)
        {
            if (p.T != t) throw new ArgumentException($"Cannot concatenate activations of {p.T} and {t} steps.");
            c += p.C;
        }
        var r = Uninit(t, c);
        for (int i = 0; i < t; i++)
        {
            int o = 0;
            foreach (var p in parts)
            {
                p.Row(i).CopyTo(r.Data.AsSpan(i * c + o, p.C));
                o += p.C;
            }
        }
        return r;
    }

    /// <summary>Channels [<paramref name="c0"/>, <paramref name="c0"/> + <paramref name="count"/>).</summary>
    public Act SliceChannels(int c0, int count)
    {
        var r = new Act(T, count);
        for (int i = 0; i < T; i++) Data.AsSpan(i * C + c0, count).CopyTo(r.Row(i));
        return r;
    }

    public Act Clone() => new(T, C, (float[])Data.Clone());
}
