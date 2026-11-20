namespace KittenTTS.Model;

/// <summary>
/// The random noise the hn-NSF source adds to its sine harmonics (<c>noise_amp · randn</c>, the only random draw of a
/// forward pass that reaches the output). Standard normal samples from a seeded generator, or none at all
/// (<see cref="None"/>) for bit-reproducible output and parity tests against references run with zero noise.
/// </summary>
public sealed class SynthesisNoise
{
    private readonly Random? _random;
    private readonly float[]? _samples;

    private SynthesisNoise(Random? random, float[]? samples = null)
    {
        _random = random;
        _samples = samples;
    }

    /// <summary>No noise: the voiced parts are pure harmonics, the unvoiced parts silent in the source signal.</summary>
    public static SynthesisNoise None { get; } = new(null);

    /// <summary>Gaussian noise from a generator seeded with <paramref name="seed"/> (a random seed when null).</summary>
    public static SynthesisNoise Gaussian(int? seed = null) => new(seed is int s ? new Random(s) : new Random());

    /// <summary>Exactly these samples (row-major [samples, 9 harmonics]); the next forward pass must ask for this many.
    /// For parity tests against a reference that recorded its noise.</summary>
    public static SynthesisNoise FromSamples(float[] samples) => new(null, samples);

    public bool IsNone => _random == null && _samples == null;

    /// <summary><paramref name="count"/> standard normal samples (an empty array for <see cref="None"/>).</summary>
    internal float[] Normal(long count)
    {
        if (_samples != null)
        {
            if (_samples.LongLength != count) throw new ArgumentException($"The recorded noise has {_samples.LongLength} samples, the forward pass needs {count}.");
            return _samples;
        }
        if (_random == null) return [];
        var r = new float[count];
        lock (_random)
        {
            for (long i = 0; i < count; i += 2)
            {
                // Box-Muller
                double u1 = 1.0 - _random.NextDouble(), u2 = _random.NextDouble();
                double m = Math.Sqrt(-2.0 * Math.Log(u1));
                r[i] = (float)(m * Math.Cos(2 * Math.PI * u2));
                if (i + 1 < count) r[i + 1] = (float)(m * Math.Sin(2 * Math.PI * u2));
            }
        }
        return r;
    }
}
