# Third-party notices

## Code

- `src/KittenTTS/Numerics` (`Gemm`, `PackedMatrix`, `NativeBuffer`, `Parallelism`, `VMath`, `Ops`, `Profiler`),
  `src/KittenTTS/IO` (`MappedFile`, `Tensor`, `ModelDownloader`, the protobuf reader of `OnnxWeights`),
  `src/KittenTTS/Audio/WaveFile.cs` and `scripts/gen_kernels.py` are copied from
  [curiosity-ai/parakeet](https://github.com/curiosity-ai/parakeet) (Apache-2.0). Parts of Parakeet's numerics
  (`NativeBuffer`, `Parallelism`, the structure of the packed-panel GEMM) are adapted from
  [Voxwerk](https://github.com/curiosity-ai/voxwerk), MIT License, Copyright (c) 2026 Voxwerk contributors:

  ```
  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
  documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
  rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
  permit persons to whom the Software is furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all copies or substantial portions of
  the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
  WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
  COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
  OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
  ```

- The model, front end and post-processing are ports of the KittenTTS Python implementation in `.reference/`
  (Apache-2.0, KittenML), which itself vendors parts of [misaki](https://github.com/hexgrad/misaki) (Apache-2.0).
  `Text/English/Lexicon.cs` and `MisakiG2P.cs` port misaki 0.9.4's `en.py`.
- `Text/Espeak/EspeakPhonemizer.cs` is a C# port of the logic of [phonemizer](https://github.com/bootphon/phonemizer)'s
  espeak backend and punctuation handling (phonemizer is GPL-3.0) and of misaki's `espeak.py`.
  `Text/English/NumberWords.cs` follows the algorithm of [num2words](https://github.com/savoirfairelinux/num2words)'
  English converter (num2words is LGPL-2.1). Whether these ports carry the original licenses' terms has not been
  reviewed; check before redistributing.

## Data embedded in the library (`src/KittenTTS/Text/English/Data`)

- `spacy_en_tokenizer.json`: the tokenizer rules of [spaCy](https://github.com/explosion/spaCy) 3.8 /
  `en_core_web_sm` (MIT License, Copyright (C) 2016-2024 ExplosionAI GmbH, 2016 spaCy GmbH, 2015 Matthew Honnibal).
- `us_gold.json.gz`, `us_silver.json.gz`: misaki's US English lexicons (Apache-2.0, hexgrad).
- `unicode_norm.json.gz`: Unicode normalization tables derived from the Unicode Character Database (Unicode License v3).

## Model files (downloaded, or shipped in the KittenTTSSharp.Models.* packages)

- KittenTTS 2 nano / pico / mini weights, voices and configs: Apache-2.0, KittenML / Stellon Labs
  (huggingface.co/KittenML/kitten-tts-2-nano, -pico, -mini).
- `g2p/g2p_bart_en_us`: PeterReid/graphemes_to_phonemes_en_us (Apache-2.0); `g2p/cmu_misaki_us.json`: CMUdict
  (BSD-2-Clause, Carnegie Mellon University) converted to misaki's alphabet.
