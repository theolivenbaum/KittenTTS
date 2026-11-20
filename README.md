# KittenTTS (pure C#)

Pure C# (.NET 11) inference for the **[KittenTTS 2](https://github.com/KittenML/KittenTTS) nano / pico / mini**
text-to-speech models (7 M – 82 M parameters, 16 languages, 24 kHz), as a library, a command-line tool and NuGet
packages of the model weights.

- **No Python, no ONNX Runtime.** The PL-BERT text encoder, the prosody predictor, the iSTFTNet vocoder with its
  harmonic source, the checkpoint readers (float32 `.safetensors`, int8 ONNX graphs) and the English front end
  (text normalisation, spaCy's tokenizer, misaki's lexicon, CMUdict, a BART G2P) are all managed code, on the
  packed GEMM, worker pool and SIMD kernels of [curiosity-ai/parakeet](https://github.com/curiosity-ai/parakeet).
  Other languages use the espeak-ng library, as the original does.
- **Validated against the original.** Every stage up to F0 and energy matches PyTorch (float32 weights) and ONNX
  Runtime (int8 weights) to ~1e-6, durations are identical, and the English front end reproduces the Python one
  exactly on 95% of 600 reference sentences (100% for the normaliser and the tokenizer); the espeak front end matches
  on every test sentence of 14 languages. See [Validation](#validation).
- **Weights from NuGet or Hugging Face.** `KittenTTSSharp.Models.*` packages install a model next to your application;
  without one, the files are downloaded once from `KittenML/kitten-tts-2-{nano,pico,mini}`.

The original Python implementation (all three model families, including the 1.7B-parameter KittenTTS 2 language model
and the 0.8 ONNX models) is kept in [`.reference/`](.reference) as the reference for the port; its CI workflow is
disabled. **The port's goal** is to bring the whole KittenTTS family to C#; the small KittenTTS 2 models are done, the
rest is listed under [Roadmap](#roadmap).

## Quick start

```bash
# Synthesize (downloads kitten-tts-2-nano's English files to ~/.cache/kittentts on first use, ~30 MB)
dotnet run -c Release --project src/KittenTTS.Cli -- say --voice Eleanor --out hello.wav "Hello there! How are you?"

# Another model, language and voice; a voice name alone picks its language
dotnet run -c Release --project src/KittenTTS.Cli -- say --model mini --voice Paul --out de.wav "Guten Morgen."

# What the model will say, the voices, a benchmark
dotnet run -c Release --project src/KittenTTS.Cli -- phonemize "It costs \$3.50 on Jan. 5th."
dotnet run -c Release --project src/KittenTTS.Cli -- voices --model pico
dotnet run -c Release --project src/KittenTTS.Cli -- bench --model nano
```

| Option | Meaning |
|---|---|
| `--model <name\|dir>` | `nano` (default), `pico`, `mini`, or a local repository directory laid out like the Hugging Face one |
| `--voice <name>` | A voice name in any case (`Eleanor`, `paul`); default: the language's first voice |
| `--language <code>` | `en` (default), `de`, `es`, `it`, `pt`, `ru`, `hi`, `fr`; the mini adds `da fi nl sv sw tr ko` |
| `--speed <x>` | > 1 speaks faster |
| `--weights <w>` | `int8` (default, the Python package's default) or `fp32` (the PyTorch weights) |
| `--threads <n>` / `--seed <n>` / `--no-noise` | Worker threads; seed of the synthesis noise; no noise (deterministic) |

### Library

```bash
dotnet add package KittenTTSSharp                    # namespace: KittenTTS
dotnet add package KittenTTSSharp.Models.Nano.En     # optional: the weights, installed next to your app
```

```csharp
using KittenTTS;

using var tts = await KittenTts.LoadAsync(KittenModel.Nano);          // English; LoadAsync(KittenModel.Mini, ["de", "fr"]),
                                                                       // or KittenTts.Load("/path/to/kitten-tts-2-nano")
float[] audio = tts.Generate("Hello there.", voice: "Eleanor");       // float32, 24 kHz mono
tts.GenerateToFile("Guten Morgen.", "de.wav", voice: "Paul");          // a German voice: German
foreach (var chunk in tts.GenerateStream(longText, language: "en"))    // chunk by chunk (sentence-aligned)
    Play(chunk);

tts.Voices("en");                       // [{Language, Key, Name, Description}, ...]
tts.Phonemize("Hello there.");          // ["həlˈO ðˈɛɹ."]
```

`KittenTtsOptions` sets `Threads`, `Weights` (`Int8` / `Float32`), `Seed` and `Noise`. One `KittenTts` instance is
thread-safe. The stages are public too: `KittenNetwork` (one network, `tokens + style → waveform, durations`, with a
per-stage trace callback), `Frontend` / `EnglishG2P` / `LanguagePhonemizer` (text → phonemes), `ModelRepository`,
`SafeTensors` / `OnnxWeights` (checkpoint readers), `WaveFile`.

### Models and packages

| Model | Parameters | Languages | int8 per model | NuGet |
|---|---|---|---|---|
| kitten-tts-2-mini | 82 M | 16: en de es it pt ru hi fr da fi nl sv sw tr ko (+ ja, not supported here) | 83 MB | `KittenTTSSharp.Models.Mini` |
| kitten-tts-2-nano | 17 M | 8: en de es it pt ru hi fr (one model each) | 17 MB | `KittenTTSSharp.Models.Nano.{En,De,Es,It,Pt,Ru,Hi,Fr}` |
| kitten-tts-2-pico | 7 M | 8: en de es it pt ru hi fr (one model each) | 8 MB | `KittenTTSSharp.Models.Pico.{En,De,Es,It,Pt,Ru,Hi,Fr}` |

A model package carries the int8 ONNX graph, the voice packs and the config of its language(s) (English packages and
the mini also `g2p/`, the CMUdict lexicon and BART of the English front end) and copies them to
`<app>/kittentts/kitten-tts-2-<model>/` on build and publish. `KittenTts.LoadAsync` looks there first, then in
`~/.cache/kittentts` (`$KITTENTTS_CACHE`), and otherwise downloads only the files of the requested languages and
weight format from Hugging Face (`$KITTENTTS_MODEL_BASE_URL` points at a mirror). The weights are Apache-2.0
(KittenML / Stellon Labs).

**Non-English languages need espeak-ng** (`apt install espeak-ng`, `brew install espeak-ng`, or the Windows
installer), loaded with P/Invoke; `KITTENTTS_ESPEAK_LIBRARY` and `ESPEAK_DATA_PATH` select a specific build. English
needs nothing native. Japanese (misaki's own G2P in the original) is not ported.

## How it works

```
text ─► front end ─► phoneme chunks (≤ 350 symbols, cut at sentence ends) ─► token ids [0, …, 10, 0]
        English: tts_norm ─► spaCy tokenizer ─► misaki lexicon (+ stemming, numbers) ─► CMUdict ─► BART G2P
        others:  espeak-ng (phonemizer + misaki EspeakG2P semantics) ─► fold tables
tokens ─► TextEncoder (embedding, 3 × conv5 + LayerNorm, BiLSTM) ─────────────────────────┐ aligned
       ─► PL-BERT (ALBERT, 12 shared layers) ─► Linear ─► DurationEncoder(style₂) ─► LSTM ─► durations (frames)
                                                   └─aligned─► shared BiLSTM ─► F0 / energy AdaIN ResBlocks (×2 rate)
       ─► Decoder(style₁): F0/N convs, encode + 4 decode AdaIN ResBlocks (last ×2)
       ─► iSTFTNet generator: hn-NSF sine source (9 harmonics, its STFT injected per stage),
          ConvTranspose ×10, ×6 with AdaIN snake ResBlocks ─► magnitude + phase ─► iSTFT (n_fft 20, hop 5) ─► 24 kHz
```

Each voice is a `[510, 256]` style pack; a chunk uses the row of its token count (the first 128 values condition the
decoder, the rest the prosody predictor). Long text is synthesized chunk by chunk and joined as in the original: the
end token's frames and every chunk's leading pause but the first are cut, 5 ms fades, 0.15 s of silence.

| Source | Contents |
|---|---|
| `src/KittenTTS/KittenTts.cs` | Public API: load (packages, cache, download), voices, generate, stream |
| `src/KittenTTS/Model/` | The network: layers (Conv1d / ConvTranspose1d as GEMMs, BiLSTM), AdaIN blocks, PL-BERT, predictor, decoder, generator |
| `src/KittenTTS/Numerics/` | Packed FP32 GEMM (AVX-512 / AVX2), worker pool, vector math, profiler (from Parakeet) |
| `src/KittenTTS/IO/` | `.safetensors`, int8 ONNX graph (protobuf, names recovered from the graph), `.npz` voice packs, downloader |
| `src/KittenTTS/Audio/` | STFT / iSTFT, WAV |
| `src/KittenTTS/Text/English/` | Normaliser, num2words, spaCy tokenizer, POS rules, misaki lexicon and G2P, BART G2P |
| `src/KittenTTS/Text/Espeak/`, `Text/LanguagePhonemizer.cs` | espeak-ng P/Invoke, phonemizer / misaki espeak semantics, v1 / v2 folds |
| `src/KittenTTS.Cli/` | `kitten-tts` command-line tool |
| `models/` | The weight packages (`KittenTTSSharp.Models.csproj`, packed per model / language) |
| `tests/KittenTTS.Tests/` | Parity with PyTorch / ONNX Runtime, front-end parity, normalisation |
| `scripts/` | Reference dumps (`torch_reference.py`, `english_reference.py`, `frontend_reference.py`), data export, kernel generator |

### Numerics

- **Convolutions are GEMMs without im2col.** Activations are channels-last `[T, C]`, so a stride-`s` convolution's
  receptive field at step `t` is one contiguous run of `K · Cin` values in a zero-padded copy of the input: the GEMM's
  A operand is that buffer with overlapping rows (`lda = s · Cin`). Dilated convolutions run one accumulating GEMM per
  tap; transposed convolutions one GEMM for all taps, then an overlap-add.
- **LSTMs** project both directions' inputs in one GEMM, then run the two recurrences concurrently.
- **The sine source repeats PyTorch's arithmetic where it is ill-conditioned:** its phase reaches 10⁴ rad, so the
  cumulative sum accumulates in double (as `torch.cumsum` does on CPU) and the linear interpolation is
  `fma(p₀, w₀, p₁·w₁)` (as PyTorch's kernel compiles); for ONNX weights, the graph's separate Mul / Add instead.
- **int8 weights** are dequantized once at load (`(q − zp) · scale` per output channel), so inference is the float32
  arithmetic the graph specifies.

## Validation

```bash
dotnet test tests/KittenTTS.Tests -c Release
```

The kernel and front-end tests always run. The model tests need a local repository and a reference dump, and print
`SKIPPED` without them:

```bash
pip install torch safetensors transformers onnx onnxruntime numpy misaki==0.9.4 spacy phonemizer espeakng_loader num2words
python -m spacy download en_core_web_sm
python scripts/torch_reference.py --model <kitten-tts-2-nano dir> --language en --out ref/nano-en --onnx
KITTENTTS_MODEL_DIR=<kitten-tts-2-nano dir> KITTENTTS_REFERENCE_DIR=ref/nano-en dotnet test tests/KittenTTS.Tests -c Release
```

| | nano (en) | mini (en) |
|---|---|---|
| Text encoder, PL-BERT, duration encoder vs PyTorch (rel. L2) | ≤ 3e-6 | ≤ 3e-6 |
| F0 / energy vs PyTorch | ≤ 7e-7 | ≤ 4e-7 |
| Durations vs PyTorch and vs ONNX Runtime | identical | identical |
| Decoder alone, on PyTorch's F0 / features | 1.2e-5 | ≤ 3e-4 |
| Waveform vs PyTorch (float32 weights) | ≤ 2e-3 | ≤ 1e-2 |
| Waveform vs ONNX Runtime (int8 weights) | ≤ 3e-3 | ≤ 7e-3 |

The references are run with the same recorded synthesis noise as the C# side. The waveform bounds are looser than the
stage bounds for reasons that are properties of the model rather than of the port: the sine source integrates F0
into its phase, so a 1e-7 difference in F0 becomes ~1e-3 rad; and the harmonic STFT's phase is ill-conditioned where
a bin's magnitude is near zero (the first frame is exactly symmetric, so its imaginary parts are zero up to rounding
and the angle flips between ±π with the rounding of MKL's FFT, ONNX Runtime's convolution or the exact DFT here).
ONNX Runtime is compared with basic graph optimizations: its default (extended) level rewrites the int8
`DequantizeLinear → MatMul` pairs into kernels with dynamically quantized activations, which changes PL-BERT's output by
~4e-3 compared with the graph's own float32 semantics.

The English front end is compared on 604 sentences (`scripts/english_reference.py`: targeted cases plus every sentence
of the original documentation): the normaliser and the tokenizer agree on all of them, the phoneme strings on 95%.
The rest differ in heteronyms ("use" as noun or verb, "read" present or past, "that" stressed or not) and in whether a
free-standing dash is a pause, which misaki decides from spaCy's statistical part-of-speech tags; the C# port tags with
rules instead (see `PosTagger`).

### Performance

Same 4-vCPU machine (AVX-512), int8 weights, 4 threads, one ~6.5 s sentence, median of 5
(`kitten-tts bench --no-noise` against ONNX Runtime 1.32 running the same int8 graph in Python):

| | C# (this port) | ONNX Runtime | |
|---|---|---|---|
| nano | 647 ms, **10.1×** realtime | 522 ms, 12.8× realtime | 1.24× slower |
| mini | 2025 ms, **3.0×** realtime | 1558 ms, 4.0× realtime | 1.30× slower |

The iSTFTNet generator's convolutions at the audio-rate stages take 75–88% of the time (`KITTENTTS_PROFILE=1` prints a
per-stage breakdown). AdaIN and snake are fused into one parallel pass, dilated convolutions run as blocked im2col
GEMMs, and attention as per-head GEMMs; closing the remaining gap (fusing AdaIN into the conv epilogues, a faster
iSTFT and LSTM recurrence) is on the roadmap.

## Roadmap

- **KittenTTS 2 (1.7B)**: the Qwen3-based speech language model with ternary weights, the S3 tokenizer, the s3gen
  flow-matching vocoder, the speaker-embedding model and voice cloning (`.reference/kittenml/kittentts2/`).
- **KittenTTS 0.8 ONNX models** (`.reference/kittenml/kittentts_legacy/`): the same StyleTTS 2 family; needs their
  graphs' weights mapped like the int8 graphs here.
- **int4 graphs** (`*_int4.onnx`): their tensor and node names are obfuscated, so the weights must be matched to the
  model structurally.
- Japanese (misaki's Japanese G2P), a statistical POS tagger for the English heteronyms, faster generator kernels.

## Building and publishing

```bash
dotnet build KittenTTS.slnx -c Release
dotnet test tests/KittenTTS.Tests -c Release
dotnet pack src/KittenTTS -c Release -p:Version=26.11.1                                                  # KittenTTSSharp
dotnet pack models/KittenTTSSharp.Models.csproj -c Release -p:ModelFamily=nano -p:ModelLanguage=en -p:Version=26.11.1
dotnet pack models/KittenTTSSharp.Models.csproj -c Release -p:ModelFamily=mini -p:Version=26.11.1
```

The Azure DevOps pipeline (`.devops/azure-pipelines.yml`) builds, tests and packs `KittenTTSSharp` with a CalVer
version (`yy.M.<buildId>`) and pushes it from `master`; with the `publishModelPackages` parameter it also packs and
pushes the 17 weight packages (they depend on the library of the same build).

## License

The C# code is licensed under the [Apache License 2.0](LICENSE), like the original. The KittenTTS 2 nano / pico /
mini weights are Apache-2.0 (KittenML / Stellon Labs); the embedded data files are spaCy's tokenizer rules (MIT) and
misaki's lexicons (Apache-2.0), and the BART G2P and CMUdict lexicon come from the model repositories. See
[THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md).
