#!/usr/bin/env python3
"""Reference phoneme strings of the non-English front end, for the C# espeak tests.

Runs the original Python front end in .reference/ (misaki's EspeakG2P on phonemizer's espeak backend, then the v1 / v2
FOLD tables of phonemize_v1.py / phonemize_v2.py) on test sentences per language and writes
tests/KittenTTS.Tests/testdata/espeak_reference.json. The output depends on the espeak-ng build: the vendored
espeak.py loads the one bundled with espeakng_loader, which is what this script uses; its version is recorded in the
JSON, and the C# test only compares strings when it loaded the same version (point KITTENTTS_ESPEAK_LIBRARY and
ESPEAK_DATA_PATH at espeakng_loader's library and data to get it).

    pip install phonemizer espeakng_loader misaki
    python scripts/frontend_reference.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, ".reference"))

from kittenml.kittentts2_tiny.frontend import phonemize_v1, phonemize_v2  # noqa: E402
from phonemizer.backend.espeak.wrapper import EspeakWrapper  # noqa: E402

SENTENCES = {
    "de": ["Guten Morgen, wie geht es dir heute?", "Die Katze schläft auf dem warmen Sofa (seit 3 Stunden).",
           "Er sagte: „Das kostet 19,99 Euro!“ Wirklich?", "Übermorgen fahren wir nach München – endlich."],
    "es": ["Hola, ¿cómo estás? Hace un día precioso.", "Tengo 25 años y vivo en Madrid (España).",
           "¡Qué bien! «Vamos», dijo él."],
    "it": ["Buongiorno, come stai oggi?", "Il treno parte alle 7:45 dalla stazione centrale."],
    "pt": ["Olá, tudo bem? Hoje está um dia lindo.", "Eu tenho 3 irmãos e (às vezes) brigamos."],
    "ru": ["Привет, как дела?", "Щука плавала в реке; было 12 градусов."],
    "hi": ["नमस्ते, आप कैसे हैं? मैं ठीक हूँ।", "सड़क पर बहुत भीड़ थी। हम 67 लोग थे॥", "पढ़ाई ज़रूरी है।"],
    "fr": ["Bonjour, comment allez-vous aujourd'hui ?", "Il fait beau ; nous irons au parc (avec Marie).",
           "« C'est incroyable ! », s'exclama-t-il."],
    "da": ["Hej, hvordan har du det?", "Jeg bor i København (Danmark)."],
    "fi": ["Hyvää huomenta, mitä kuuluu?", "Minulla on 2 kissaa."],
    "nl": ["Goedemorgen, hoe gaat het met je?", "Het regent al 3 dagen."],
    "sv": ["Hej, hur mår du idag?", "Jag bor i Göteborg och älskar kaffe."],
    "sw": ["Habari za asubuhi? Nzuri sana.", "Ninaishi Nairobi, Kenya."],
    "tr": ["Merhaba, nasılsın?", "Bugün hava çok güzel (gerçekten)."],
    "ko": ["안녕하세요, 잘 지내세요?", "오늘 날씨가 정말 좋네요."],
}
V1 = ["de", "es", "it", "pt", "ru", "hi", "fr"]
V2 = V1 + ["da", "fi", "nl", "sv", "sw", "tr", "ko"]


def main():
    cases = []
    for version, mod, langs in (("v1", phonemize_v1, V1), ("v2", phonemize_v2, V2)):
        for lang in langs:
            g2p = mod.get_g2p(lang)
            for text in SENTENCES[lang]:
                cases.append({"version": version, "language": lang, "text": text,
                              "phonemes": mod.phonemize_lang(g2p, text, lang)})
    out = {"espeak_version": ".".join(map(str, EspeakWrapper().version)), "espeak_build": "espeakng_loader",
           "cases": cases}
    path = os.path.join(ROOT, "tests", "KittenTTS.Tests", "testdata", "espeak_reference.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("espeak", out["espeak_version"])
    for c in cases:
        print(c["version"], c["language"], c["phonemes"])


if __name__ == "__main__":
    main()
