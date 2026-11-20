namespace KittenTTS.Audio;

/// <summary>
/// <c>torch.stft</c> / <c>torch.istft</c> for the generator's tiny transform (n_fft 20, hop 5, periodic Hann window,
/// <c>center=True</c> with reflection padding, one-sided). The DFT is direct (11 bins × 20 taps, double precision):
/// at this size it is faster than an FFT and exact to float rounding.
/// </summary>
internal sealed class Stft
{
    private readonly int _n, _hop, _bins;
    private readonly double[] _window;
    private readonly double[] _cos, _sin;     // [bin, tap]

    public Stft(int nFft, int hop)
    {
        _n = nFft;
        _hop = hop;
        _bins = nFft / 2 + 1;
        _window = new double[nFft];
        for (int i = 0; i < nFft; i++) _window[i] = (float)(0.5 - 0.5 * Math.Cos(2 * Math.PI * i / nFft));   // float32 window, as torch
        _cos = new double[_bins * nFft];
        _sin = new double[_bins * nFft];
        for (int k = 0; k < _bins; k++)
            for (int t = 0; t < nFft; t++)
            {
                // exact angles mod n: k t (mod n) keeps cos / sin of identical arguments bit-identical
                double a = 2 * Math.PI * ((long)k * t % nFft) / nFft;
                _cos[k * nFft + t] = Math.Cos(a);
                _sin[k * nFft + t] = Math.Sin(a);
            }
    }

    /// <summary>Signal → (magnitude, phase) as [frames, bins] row-major, frames = len / hop + 1.</summary>
    public (float[] Magnitude, float[] Phase, int Frames) Forward(float[] x)
    {
        int pad = _n / 2;
        int len = x.Length;
        if (len <= pad) throw new ArgumentException($"STFT input of {len} samples is too short for reflection padding.");
        int frames = len / _hop + 1;
        var mag = new float[(long)frames * _bins];
        var ph = new float[(long)frames * _bins];
        var frame = new double[_n];
        for (int f = 0; f < frames; f++)
        {
            for (int t = 0; t < _n; t++)
            {
                int i = f * _hop + t - pad;
                if (i < 0) i = -i;
                else if (i >= len) i = 2 * (len - 1) - i;
                frame[t] = (float)(x[i] * (float)_window[t]);
            }
            for (int k = 0; k < _bins; k++)
            {
                double re = 0, im = 0;
                for (int t = 0; t < _n; t++)
                {
                    re += frame[t] * _cos[k * _n + t];
                    im -= frame[t] * _sin[k * _n + t];
                }
                // DC and Nyquist are real: a +0 imaginary part makes the angle of a negative value +π, as in torch
                if (k == 0 || 2 * k == _n) im = 0.0;
                float fr = (float)re, fi = (float)im;
                mag[f * _bins + k] = MathF.Sqrt(fr * fr + fi * fi);
                ph[f * _bins + k] = MathF.Atan2(fi, fr);
            }
        }
        return (mag, ph, frames);
    }

    /// <summary>(magnitude, phase) [frames, bins] → signal of hop · (frames − 1) samples: inverse real DFT per frame,
    /// window, overlap-add, division by the summed squared window, center trim.</summary>
    public float[] Inverse(float[] magnitude, float[] phase, int frames)
    {
        int pad = _n / 2;
        int full = _n + _hop * (frames - 1);
        var y = new double[full];
        var env = new double[full];
        var frame = new double[_n];
        var re = new double[_bins];
        var im = new double[_bins];
        for (int f = 0; f < frames; f++)
        {
            for (int k = 0; k < _bins; k++)
            {
                float m = magnitude[f * _bins + k], p = phase[f * _bins + k];
                re[k] = m * MathF.Cos(p);
                im[k] = m * MathF.Sin(p);
            }
            for (int t = 0; t < _n; t++)
            {
                double s = re[0];
                for (int k = 1; k < _bins; k++)
                {
                    double c = _cos[k * _n + t], sn = _sin[k * _n + t];
                    if (2 * k == _n) s += re[k] * c;          // Nyquist: real part only
                    else s += 2 * (re[k] * c - im[k] * sn);
                }
                frame[t] = s / _n;
            }
            int o = f * _hop;
            for (int t = 0; t < _n; t++)
            {
                y[o + t] += (float)frame[t] * _window[t];
                env[o + t] += _window[t] * _window[t];
            }
        }
        int outLen = _hop * (frames - 1);
        var result = new float[outLen];
        for (int i = 0; i < outLen; i++)
        {
            double e = env[i + pad];
            result[i] = e > 1e-11 ? (float)(y[i + pad] / e) : 0f;
        }
        return result;
    }
}
