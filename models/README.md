# KittenTTSSharp model weights

Weights of one [KittenTTS 2](https://github.com/KittenML/KittenTTS) small model (nano, pico or mini) for
[KittenTTSSharp](https://www.nuget.org/packages/KittenTTSSharp), the pure C# (.NET 11) port: the int8 ONNX graph
(read by the C# engine, no ONNX Runtime), the voice packs and the config, plus for English the CMUdict lexicon and
BART G2P of the English front end. The files are copied to `kittentts/kitten-tts-2-<model>/` next to your
application on build and publish, where `KittenTts.LoadAsync` finds them without downloading anything.

```csharp
using KittenTTS;

using var tts = await KittenTts.LoadAsync(KittenModel.Nano, ["en"]);
tts.GenerateToFile("Hello there.", "hello.wav", voice: "Eleanor");
```

Packages: `KittenTTSSharp.Models.{Nano,Pico}.{En,De,Es,It,Pt,Ru,Hi,Fr}` (one model per language) and
`KittenTTSSharp.Models.Mini` (one model, 16 languages). Languages other than English need the espeak-ng library at
run time (see the KittenTTSSharp README).

The weights are © KittenML / Stellon Labs, released under the Apache License 2.0 on Hugging Face
(`KittenML/kitten-tts-2-nano`, `-pico`, `-mini`); each package carries the model card (`README.md` in its folder).
