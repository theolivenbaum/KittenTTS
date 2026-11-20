#!/usr/bin/env python3
"""Reference dumps for the C# parity tests, from the original Python implementation in .reference/.

For each test sentence this runs the Python front end (text -> phoneme chunks -> token ids), then the float32
PyTorch model (.reference/kittenml/kittentts2_tiny/torch_model.py) and saves every stage as .npy; with --onnx it also
runs the int8 ONNX graph in ONNX Runtime on the same inputs. The harmonic noise of the sine source (the only random
draw that reaches the output) comes from a seeded generator and is saved too (noise.npy, onnx_noise.npy), so the C#
side injects exactly the same samples. (With zero noise the comparison is ill-conditioned: in silence the source is
constant, every STFT bin but DC is rounding noise, and its phase is arbitrary.)

    pip install torch safetensors transformers onnx onnxruntime numpy misaki spacy phonemizer espeakng_loader num2words
    python scripts/torch_reference.py --model <repo dir, e.g. kitten-tts-2-nano> --language en --out <dir> [--onnx]

Output: <out>/manifest.json (text, language, voice, phoneme chunks, token ids per case) and <out>/<case>/<stage>.npy,
stages: tokens, style, noise, t_en, bert, d_en, d, durations, F0, N, waveform (+ onnx_noise, onnx_waveform,
onnx_durations), all
channels-last ([T, C]) like the C# activations. The C# tests read them through KITTENTTS_REFERENCE_DIR.
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, ".reference"))

SENTENCES = {
    "en": [
        "Hello there.",
        "The quick brown fox jumps over the lazy dog.",
        "Kevin paid $3.50 for two coffees on Jan. 5th, 2024, and read the receipt twice.",
        "I can't believe it's already 10:30 pm! Where did the time go?",
        "Long text is split at sentence ends into chunks; each chunk is synthesized on its own, "
        "then the chunks are joined with a short silence so that the seams are inaudible to the listener.",
    ],
    "de": ["Guten Morgen, wie geht es dir heute?", "Die Katze schläft auf dem warmen Sofa."],
    "fr": ["Bonjour, comment allez-vous aujourd'hui ?"],
    "es": ["Hola, ¿cómo estás? Hace un día precioso."],
}


class FixedNoise:
    """Replaces the random draws of SineGen / SourceModuleHnNSF: the harmonic noise (the [1, L, 9] randn_like, the
    only draw that reaches the output) comes from a seeded numpy generator and is recorded, so the C# side can inject
    exactly the same samples; every other draw is zero. With seed None all noise is zero."""

    def __init__(self, torch, seed):
        self.torch, self.seed, self.last = torch, seed, None

    def __enter__(self):
        torch = self.torch
        self.orig = (torch.randn_like, torch.rand)

        def randn_like(x, *a, **k):
            if self.seed is None or x.dim() != 3 or x.shape[-1] != 9:
                return torch.zeros_like(x)
            n = np.random.default_rng(self.seed).standard_normal(tuple(x.shape)).astype(np.float32)
            self.last = n.reshape(-1, 9)
            return torch.from_numpy(n).to(x.device)
        torch.randn_like = randn_like
        torch.rand = lambda *a, **k: torch.zeros(*[s for s in a if isinstance(s, int)], device=k.get("device"))
        return self

    def __exit__(self, *exc):
        self.torch.randn_like, self.torch.rand = self.orig


def zero_noise(torch):
    """Patch every random draw to zeros (returns an undo function)."""
    f = FixedNoise(torch, None).__enter__()
    return lambda: f.__exit__()


def torch_stages(model, tokens, style, speed):
    """TorchModel.__call__, keeping the intermediate tensors (channels-last)."""
    import torch
    from kittenml.kittentts2_tiny.torch_model import _length_to_mask
    m, dev = model.net, model.device
    out = {}
    with torch.no_grad():
        tok = torch.as_tensor([tokens], dtype=torch.long, device=dev)
        st = torch.as_tensor(np.asarray(style, np.float32).reshape(1, -1), device=dev)
        sa, sp = st[:, :128], st[:, 128:]
        n_tok = tok.shape[-1]
        il = torch.as_tensor([n_tok], dtype=torch.long, device=dev)
        tm = _length_to_mask(il)
        t_en = m.text_encoder(tok, il, tm)
        bd = m.bert(tok, attention_mask=(~tm).int())
        d_en = m.bert_encoder(bd).transpose(-1, -2)
        d = m.predictor.text_encoder(d_en, sp, il, tm)
        x, _ = m.predictor.lstm(d)
        dur = torch.sigmoid(m.predictor.duration_proj(x)).sum(-1) / float(speed)
        pr = torch.round(dur.reshape(-1)).clamp(min=1).long()
        aln = torch.repeat_interleave(torch.eye(n_tok, device=dev), pr, dim=1)
        en = d.transpose(-1, -2) @ aln.unsqueeze(0)
        F0, N = m.predictor.F0Ntrain(en, sp)
        wav = m.decoder(t_en @ aln.unsqueeze(0), F0, N, sa)
        out["t_en"] = t_en[0].T
        out["bert"] = bd[0]
        out["d_en"] = d_en[0].T
        out["d"] = d[0]
        out["durations"] = pr
        out["F0"] = F0.reshape(-1)
        out["N"] = N.reshape(-1)
        out["waveform"] = wav.reshape(-1)
    return {k: v.float().cpu().numpy() if v.dtype.is_floating_point else v.cpu().numpy() for k, v in out.items()}


def onnx_session(path, noise_input):
    """The ONNX graph with its RandomNormalLike node replaced by a graph input "noise" ([1, L, 9] float32), or by
    zeros when not noise_input."""
    import onnx
    import onnxruntime as ort
    from onnx import helper
    m = onnx.load(path)
    g = m.graph
    for i, n in enumerate(list(g.node)):
        if n.op_type == "RandomNormalLike":
            if noise_input:
                new = helper.make_node("Identity", ["noise"], [n.output[0]], name=n.name + "_input")
                g.input.append(helper.make_tensor_value_info("noise", onnx.TensorProto.FLOAT, [1, None, 9]))
            else:
                new = helper.make_node("Mul", [n.input[0], n.output[0] + "_zero"], [n.output[0]], name=n.name + "_zero")
                g.initializer.append(helper.make_tensor(n.output[0] + "_zero", onnx.TensorProto.FLOAT, [], [0.0]))
            g.node.remove(n)
            g.node.insert(i, new)
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    # ORT's extended optimizations rewrite DequantizeLinear -> MatMul into int8 kernels with dynamically quantized
    # activations (MatMulIntegerToFloat), which changes the numerics (~4e-3 relative at the first PL-BERT projection);
    # basic optimizations keep the graph's float32 semantics, which is what the C# int8 path computes.
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    return ort.InferenceSession(m.SerializeToString(), so, providers=["CPUExecutionProvider"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="local KittenTTS 2 nano / pico / mini repository directory")
    ap.add_argument("--language", default="en")
    ap.add_argument("--voice", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--onnx", action="store_true", help="also run the int8 ONNX graph in ONNX Runtime")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--noise-seed", type=int, default=1234,
                    help="seed of the injected harmonic noise (the same samples are saved for the C# side)")
    args = ap.parse_args()

    import torch
    torch.manual_seed(0)
    from kittenml.kittentts2_tiny.model import KittenTTS2Tiny

    with open(os.path.join(args.model, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    tts = KittenTTS2Tiny.__new__(KittenTTS2Tiny)
    tts._resolved = (args.model, None, cfg)
    tts.__init__(args.model, backend="torch", device="cpu")
    lang, key = tts.resolve_voice(args.voice, args.language)
    model = tts._torch_model(lang)
    onnx_path = os.path.join(args.model, tts._model_files(cfg["languages"][lang])["int8"])
    sess = onnx_session(onnx_path, True) if args.onnx else None
    sess0 = onnx_session(onnx_path, False) if args.onnx else None

    os.makedirs(args.out, exist_ok=True)
    manifest = {"model": os.path.basename(os.path.normpath(args.model)), "language": lang, "voice": key,
                "speed": args.speed, "cases": []}
    if True:
        case = 0
        for text in SENTENCES.get(lang, SENTENCES["en"]):
            chunks = tts.phonemize(text, lang)
            for ps in chunks:
                tokens = tts._tokens(ps)
                pack = tts._pack(lang, key)
                style = pack[min(len(tokens) - 1, pack.shape[0] - 1)]
                name = f"case{case:02d}"
                d = os.path.join(args.out, name)
                os.makedirs(d, exist_ok=True)
                with FixedNoise(torch, args.noise_seed + case) as noise:
                    stages = torch_stages(model, tokens, style, args.speed)
                stages["noise"] = noise.last
                stages["tokens"] = np.asarray(tokens, np.int64)
                stages["style"] = np.asarray(style, np.float32)
                if sess is not None:
                    feed = {"input_ids": np.array([tokens], np.int64), "style": style[None].astype(np.float32),
                            "speed": np.array([args.speed], np.float32)}
                    # the noise length depends on the durations: a first pass without noise finds them
                    _, dur = sess0.run(None, feed)
                    length = int(dur.sum()) * 600
                    onnx_noise = np.random.default_rng(args.noise_seed + 1000 + case).standard_normal((1, length, 9)).astype(np.float32)
                    feed["noise"] = onnx_noise
                    wav, dur = sess.run(None, feed)
                    stages["onnx_noise"] = onnx_noise.reshape(-1, 9)
                    stages["onnx_waveform"] = wav.reshape(-1).astype(np.float32)
                    stages["onnx_durations"] = dur.reshape(-1).astype(np.int64)
                for k, v in stages.items():
                    np.save(os.path.join(d, k + ".npy"), np.ascontiguousarray(v))
                manifest["cases"].append({"name": name, "text": text, "phonemes": ps, "tokens": tokens})
                print(name, len(tokens), "tokens", stages["waveform"].shape[0], "samples:", ps)
                case += 1
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
