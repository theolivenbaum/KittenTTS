#!/usr/bin/env python3
"""Reference output of the English front end, for the C# tests.

Runs the original Python front end in .reference/ (tts_norm -> spaCy + misaki -> CMUdict -> BART -> folds -> vocab)
on a corpus: the targeted sentences below plus every sentence of the original documentation (.reference/README.md
and .reference/docs/*.md, real prose with numbers, units and names). For each it records the normalised text, the
spaCy tokens and tags, and the phoneme chunks of a v1 (nano / pico) and a v2 (mini) repository, in
tests/KittenTTS.Tests/testdata/english_reference.json.

    pip install misaki==0.9.4 spacy num2words torch transformers && python -m spacy download en_core_web_sm
    python scripts/english_reference.py --v1 <kitten-tts-2-nano dir> --v2 <kitten-tts-2-mini dir>
"""
import argparse
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, ".reference"))

TARGETED = [
    "Hello there.",
    "The quick brown fox jumps over the lazy dog.",
    "Kevin paid $3.50 for two coffees on Jan. 5th, 2024, and read the receipt twice.",
    "I can't believe it's already 10:30 pm! Where did the time go?",
    "It cost $5.99 at the store.",
    "Dr. Smith lives on St. Paul St.",
    "The meeting is at 3:45pm on 2026-10-13.",
    "NASA's budget rose 12% to $24.5B.",
    "Chapter IV, section 3b.",
    "Call 555-0123 ext. 42.",
    "He ran 5km in 22min.",
    "The temperature is -3°C today.",
    "e.g. apples, i.e. fruit, etc.",
    "She owns 1,234 books & 7 cats.",
    "Flight AA1234 departs from JFK at 07:05.",
    "Version 2.0.1 fixes 3/4 of the bugs.",
    "Revenue grew 150% to $1.2B in Q3 2025.",
    "Mix 250ml water with 1.5kg flour at 180°C.",
    "Visit https://example.com or email a@b.com.",
    "The 3rd of May, 1999 — see pages 10-15.",
    "We'll record a new record; they present the present to the presenter.",
    "Don't you think the well-known author's 21st novel is better than her first?",
    "I used to live in London, but now I live near the old live music venue.",
    "Søren and Zoë met at the café in Zürich, then flew to São Paulo.",
    "\"Stop!\" she shouted. (Nobody listened.) Then -- silence...",
    "In 1066, William won; in the 1990s, everybody had a pager.",
    "The U.S. economy grew 2.5 percent, the fastest since the mid-'80s.",
    "Read the manual: it explains how to read files in 2 minutes.",
]


def doc_sentences():
    out = []
    for path in [os.path.join(ROOT, ".reference", "README.md"), *sorted(glob.glob(os.path.join(ROOT, ".reference", "docs", "*.md")))]:
        text = open(path, encoding="utf-8").read()
        text = re.sub(r"```.*?```", " ", text, flags=re.S)           # code blocks
        text = re.sub(r"<[^>]+>", " ", text)                           # html
        text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text)         # links -> text
        text = re.sub(r"`([^`]*)`", r"\1", text)
        for line in text.splitlines():
            line = line.strip().lstrip("#>*-| ").strip()
            if len(line) < 20 or line.count("|") > 1 or not re.search(r"[a-z]{3}", line):
                continue
            for s in re.split(r"(?<=[.!?])\s+", line):
                if 20 <= len(s) <= 300:
                    out.append(s)
    seen, uniq = set(), []
    for s in out:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq


def frontend(model_dir):
    from kittenml.kittentts2_tiny.model import KittenTTS2Tiny
    with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    t = KittenTTS2Tiny.__new__(KittenTTS2Tiny)
    t._resolved = (model_dir, None, cfg)
    t.__init__(model_dir, backend="torch", device="cpu")
    return t.frontend


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", required=True, help="a v1 repository (kitten-tts-2-nano or -pico) with g2p/")
    ap.add_argument("--v2", help="a v2 repository (kitten-tts-2-mini) with g2p/")
    ap.add_argument("--out", default=os.path.join(ROOT, "tests", "KittenTTS.Tests", "testdata", "english_reference.json"))
    args = ap.parse_args()

    from kittenml.kittentts2_tiny.frontend.tts_norm import normalize
    import spacy
    nlp = spacy.load("en_core_web_sm", enable=["tok2vec", "tagger"])
    f1 = frontend(args.v1)
    f2 = frontend(args.v2) if args.v2 else None
    cases = []
    for text in TARGETED + doc_sentences():
        norm = normalize(text)
        doc = nlp(norm)
        c = {"text": text, "normalized": norm,
             "tokens": [[tk.text, tk.whitespace_, tk.tag_] for tk in doc],
             "v1": f1(text, "en")[0]}
        if f2 is not None:
            c["v2"] = f2(text, "en")[0]
        cases.append(c)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"cases": cases}, f, ensure_ascii=False, indent=0)
    print(len(cases), "cases ->", args.out)


if __name__ == "__main__":
    main()
