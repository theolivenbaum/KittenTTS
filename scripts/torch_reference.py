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


class LayerRecorder:
    """Forward hooks on every module the C# port implements: records each call's input and output, channels-last
    ([T, C]), as <name>.in.npy / <name>.out.npy (repeated calls of a shared module, the ALBERT layer, get #k). The
    C# layer tests run each module on the recorded input and compare with the recorded output."""

    # modules whose tensors are [1, C, T] (channels first); everything else is [1, T, C] or 1-D
    CHANNELS_FIRST = ("Conv1d", "ConvTranspose1d", "AdainResBlk1d", "AdaINResBlock1", "Sequential")

    def __init__(self, torch, net):
        self.torch, self.net, self.records, self.handles, self.calls = torch, net, {}, [], {}

    def to2d(self, t, channels_first):
        torch = self.torch
        if isinstance(t, (tuple, list)):
            t = t[0]
        if isinstance(t, torch.nn.utils.rnn.PackedSequence):
            t = torch.nn.utils.rnn.pad_packed_sequence(t, batch_first=True)[0]
        if hasattr(t, "last_hidden_state"):
            t = t.last_hidden_state
        a = t.detach().float().cpu().numpy()
        if a.ndim == 3:
            a = a[0].T if channels_first else a[0]
        return np.ascontiguousarray(a)

    def add(self, name, module, in_cf=None, out_cf=None):
        kind = type(module).__name__
        cf = kind in self.CHANNELS_FIRST

        def hook(m, args, out):
            k = self.calls.get(name, 0)
            self.calls[name] = k + 1
            key = name if k == 0 else f"{name}#{k}"
            if args:
                self.records[key + ".in"] = self.to2d(args[0], cf if in_cf is None else in_cf)
            self.records[key + ".out"] = self.to2d(out, cf if out_cf is None else out_cf)
        self.handles.append(module.register_forward_hook(hook))

    def attach(self):
        n = self.net
        te = n.text_encoder
        for i, block in enumerate(te.cnn):
            self.add(f"text_encoder.cnn.{i}", block)
        self.add("text_encoder.lstm", te.lstm)
        self.add("bert.embeddings", n.bert.embeddings)
        self.add("bert.encoder.embedding_hidden_mapping_in", n.bert.encoder.embedding_hidden_mapping_in)
        self.add("bert.layer", n.bert.encoder.albert_layer_groups[0].albert_layers[0])
        self.add("bert_encoder", n.bert_encoder)
        p = n.predictor
        self.add("predictor.text_encoder", p.text_encoder, in_cf=True, out_cf=False)
        for i, block in enumerate(p.text_encoder.lstms):
            self.add(f"predictor.text_encoder.lstms.{i}", block)
        self.add("predictor.lstm", p.lstm)
        self.add("predictor.duration_proj", p.duration_proj)
        self.add("predictor.shared", p.shared)
        for i, b in enumerate(p.F0):
            self.add(f"predictor.F0.{i}", b)
        for i, b in enumerate(p.N):
            self.add(f"predictor.N.{i}", b)
        self.add("predictor.F0_proj", p.F0_proj)
        self.add("predictor.N_proj", p.N_proj)
        d = n.decoder
        for name in ("F0_conv", "N_conv", "encode"):
            self.add(f"decoder.{name}", getattr(d, name))
        self.add("decoder.asr_res.0", d.asr_res[0])
        for i, b in enumerate(d.decode):
            self.add(f"decoder.decode.{i}", b)
        g = d.generator
        for coll in ("noise_convs", "noise_res", "ups", "resblocks"):
            for i, b in enumerate(getattr(g, coll)):
                self.add(f"decoder.generator.{coll}.{i}", b)
        self.add("decoder.generator.conv_post", g.conv_post)
        self.add("decoder.generator.m_source", g.m_source)

        def gen_hook(m, args, out):     # the generator's inputs (x, s, F0 curve) and the waveform
            self.records["decoder.generator.in"] = self.to2d(args[0], True)
            self.records["decoder.generator.f0"] = np.ascontiguousarray(args[2].detach().float().numpy().reshape(-1))
            self.records["decoder.generator.out"] = np.ascontiguousarray(out.detach().float().numpy().reshape(-1))
        self.handles.append(g.register_forward_hook(gen_hook))
        orig = g.stft.transform

        def transform(x):               # the harmonic source's STFT: [frames, magnitudes | phases]
            mag, ph = orig(x)
            self.records["decoder.generator.har"] = self.to2d(self.torch.cat([mag, ph], 1), True)
            return mag, ph
        g.stft.transform = transform
        self._restore = lambda: setattr(g.stft, "transform", orig)
        return self

    def detach(self):
        for h in self.handles:
            h.remove()
        self._restore()


def int8_state_dict(onnx_path, net, torch):
    """The int8 ONNX graph's weights as a PyTorch state dict for the same modules (weight norm removed, since the
    graph stores the folded weights): named initializers dequantized ((q - zp) * scale per axis); MatMul weights by
    the MatMul node's scope, transposed; LSTMs by node name, gates reordered from ONNX iofc to PyTorch ifgo; the
    decoder's instance-norm affine parameters (unnamed [1, C, 1] Mul / Add constants) in execution order of the
    AdaIN modules, found by running the model once. Independent of the C# importer, which recovers names its own way."""
    import onnx
    from onnx import numpy_helper as nh
    g = onnx.load(onnx_path).graph
    init = {i.name: nh.to_array(i) for i in g.initializer}
    prod = {o: n for n in g.node for o in n.output}

    def const(name):
        if name in init:
            return init[name].astype(np.float32)
        n = prod[name]
        if n.op_type == "DequantizeLinear":
            q = init[n.input[0]].astype(np.float32)
            scale = const(n.input[1])
            zp = init[n.input[2]].astype(np.float32) if len(n.input) > 2 and n.input[2] else 0
            axis = next((a.i for a in n.attribute if a.name == "axis"), 1)
            shape = [1] * q.ndim
            shape[axis] = -1
            return (q - np.reshape(zp, shape)) * np.reshape(scale, shape)
        if n.op_type in ("Cast", "Identity"):
            return const(n.input[0])
        raise ValueError(f"{name} is not a constant")

    sd = {}
    for name in init:
        if name.startswith("m.") and not name.endswith(("_scale", "_zp")):
            key = name[2:-2] if name.endswith("_q") else name[2:]
            sd[key] = const(name[:-2] + "_dq" if name.endswith("_q") else name)
    lstm = {"/lstm/LSTM": "text_encoder.lstm", "/lstm_1/LSTM": "predictor.lstm", "/shared/LSTM": "predictor.shared"}
    seen_lstm = set()
    for n in g.node:
        if n.op_type == "MatMul" and n.input[1] not in init or n.op_type != "MatMul":
            pass
        if n.op_type == "MatMul" and (n.input[1] in init or n.input[1] in prod and prod[n.input[1]].op_type == "DequantizeLinear") \
                and not n.input[1].startswith("m."):
            scope = [p for p in n.name.strip("/").split("/")[:-1]]
            scope = [p.rsplit("_", 1)[0] if p.rsplit("_", 1)[-1].isdigit() else p for p in scope]
            key = ".".join(scope)
            key = {"duration_proj.linear_layer": "predictor.duration_proj.linear_layer",
                   "l_linear": "decoder.generator.m_source.l_linear"}.get(key, key)
            sd[key + ".weight"] = const(n.input[1]).T
        if n.op_type == "LSTM":
            mod = lstm.get(n.name)
            if mod is None and n.name.startswith("/lstms."):
                mod = "predictor.text_encoder" + n.name[:-len("/LSTM")].replace("/", ".")
            if mod in seen_lstm:          # the text encoder's and the predictor's LSTM share the scope "/lstm"
                raise ValueError(f"duplicate LSTM {n.name}")
            seen_lstm.add(mod)
            W, R, B = const(n.input[1]), const(n.input[2]), const(n.input[3])
            H = R.shape[-1]
            order = [0, 2, 3, 1]

            def gates(t):
                return np.concatenate([t[k * H:(k + 1) * H] for k in order], 0)
            for d, sfx in ((0, "l0"), (1, "l0_reverse")):
                sd[f"{mod}.weight_ih_{sfx}"] = gates(W[d])
                sd[f"{mod}.weight_hh_{sfx}"] = gates(R[d])
                sd[f"{mod}.bias_ih_{sfx}"] = gates(B[d][:4 * H])
                sd[f"{mod}.bias_hh_{sfx}"] = gates(B[d][4 * H:])
    # instance-norm affine constants, in node order, assigned to the AdaIN modules in execution order
    cons = {}
    for n in g.node:
        for x in n.input:
            cons.setdefault(x, []).append(n)

    def channel_const(x):
        try:
            return not x.startswith("m.") and const(x).shape[0] == 1 and const(x).ndim == 3 and const(x).shape[2] == 1 and const(x).shape[1] > 1
        except (KeyError, ValueError):
            return False
    pairs = []
    for n in g.node:
        if n.op_type == "Mul" and len(n.input) == 2 and channel_const(n.input[1]):
            nxt = cons.get(n.output[0], [])
            if len(nxt) == 1 and nxt[0].op_type == "Add":
                add = next(x for x in nxt[0].input if x != n.output[0])
                if channel_const(add):
                    pairs.append((const(n.input[1]).reshape(-1), const(add).reshape(-1)))
    adain = [name for name, m in net.named_modules() if isinstance(m, torch.nn.InstanceNorm1d) and m.affine]
    order, hooks = [], []
    for name, m in net.named_modules():
        if name in adain:
            hooks.append(m.register_forward_hook(lambda m, a, o, name=name: order.append(name) if name not in order else None))
    with torch.no_grad():
        dec = net.decoder
        T = 4
        hid = dec.asr_res[0].in_channels
        dec(torch.zeros(1, hid, T), torch.zeros(1, 2 * T), torch.zeros(1, 2 * T), torch.zeros(1, 128))
    for h in hooks:
        h.remove()
    if len(order) != len(pairs):
        raise ValueError(f"{len(pairs)} affine pairs in the graph, {len(order)} affine instance norms in the model")
    for name, (w, b) in zip(order, pairs):
        sd[name + ".weight"], sd[name + ".bias"] = w, b
    return {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in sd.items()}


def load_int8(model, onnx_path, torch):
    """Puts the int8 graph's (dequantized) weights into the PyTorch model, in place."""
    net = model.net
    for m in net.modules():
        if hasattr(m, "weight_g"):
            torch.nn.utils.remove_weight_norm(m)
    sd = int8_state_dict(onnx_path, net, torch)
    missing, unexpected = net.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not k.startswith("bert.pooler")]
    if missing or unexpected:
        raise ValueError(f"int8 weights do not cover the model: missing {missing[:8]}, unexpected {unexpected[:8]}")


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
    ap.add_argument("--weights", default="fp32", choices=["fp32", "int8"],
                    help="int8: run PyTorch with the int8 ONNX graph's dequantized weights (for the C# int8 layer tests)")
    ap.add_argument("--layers", action="store_true",
                    help="also record every layer's input and output (<case>/layers/), for the C# layer tests")
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
    if args.weights == "int8":
        load_int8(model, onnx_path, torch)
    sess = onnx_session(onnx_path, True) if args.onnx else None
    sess0 = onnx_session(onnx_path, False) if args.onnx else None

    os.makedirs(args.out, exist_ok=True)
    manifest = {"model": os.path.basename(os.path.normpath(args.model)), "language": lang, "voice": key, "weights": args.weights,
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
                rec = LayerRecorder(torch, model.net).attach() if args.layers else None
                with FixedNoise(torch, args.noise_seed + case) as noise:
                    stages = torch_stages(model, tokens, style, args.speed)
                if rec is not None:
                    rec.detach()
                    os.makedirs(os.path.join(d, "layers"), exist_ok=True)
                    for k, v in rec.records.items():
                        np.save(os.path.join(d, "layers", k + ".npy"), v)
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
