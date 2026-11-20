#!/usr/bin/env python3
"""Exports the data files of the C# English front end (src/KittenTTS/Text/English/Data), from the Python packages
the original front end uses:

  spacy_en_tokenizer.json   spaCy's English tokenizer (en_core_web_sm, spaCy 3.8): the special cases (string ->
                            token texts), the prefix / suffix / infix patterns and the URL pattern. spaCy is MIT.
  us_gold.json.gz,          misaki's US English lexicons (misaki 0.9.4, Apache-2.0), gzipped.
  us_silver.json.gz
  unicode_norm.json.gz      Unicode normalization tables from Python's unicodedata (canonical and compatibility
                            decompositions, canonical combining classes, composition pairs), for runtimes where
                            string.Normalize is unavailable (.NET with InvariantGlobalization).

    pip install spacy==3.8.* misaki==0.9.4 && python -m spacy download en_core_web_sm
    python scripts/export_english_data.py
"""
import gzip
import importlib.resources
import json
import os
import shutil

import spacy
from misaki import data as misaki_data

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "KittenTTS", "Text", "English", "Data")


def main():
    os.makedirs(OUT, exist_ok=True)
    nlp = spacy.load("en_core_web_sm", enable=["tok2vec", "tagger"])
    t = nlp.tokenizer
    rules = {k: [tok[65] for tok in v] for k, v in t.rules.items()}   # 65 = ORTH
    out = {
        "spacy_version": spacy.__version__,
        "model": nlp.meta["name"] + "-" + nlp.meta["version"],
        "prefix": t.prefix_search.__self__.pattern,
        "suffix": t.suffix_search.__self__.pattern,
        "infix": t.infix_finditer.__self__.pattern,
        "url": t.url_match.__self__.pattern if t.url_match else None,
        "token_match": t.token_match.__self__.pattern if t.token_match else None,
        "special_cases": rules,
    }
    with open(os.path.join(OUT, "spacy_en_tokenizer.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
    for name in ("us_gold.json", "us_silver.json"):
        with importlib.resources.open_binary(misaki_data, name) as src, gzip.open(os.path.join(OUT, name + ".gz"), "wb", 9) as dst:
            shutil.copyfileobj(src, dst)
    export_unicode(os.path.join(OUT, "unicode_norm.json.gz"))
    print("written to", OUT)


def export_unicode(path):
    import unicodedata
    nfd, nfkd, ccc, pairs = {}, {}, {}, []
    for cp in range(0x110000):
        if 0xD800 <= cp <= 0xDFFF or 0xAC00 <= cp <= 0xD7A3:     # surrogates; Hangul syllables are algorithmic
            continue
        ch = chr(cp)
        d = unicodedata.normalize("NFD", ch)
        k = unicodedata.normalize("NFKD", ch)
        if d != ch:
            nfd[cp] = [ord(c) for c in d]
        if k != ch:
            nfkd[cp] = [ord(c) for c in k]
        c = unicodedata.combining(ch)
        if c:
            ccc[cp] = c
        # primary composites: the canonical decomposition is a pair and NFC recomposes it
        raw = unicodedata.decomposition(ch)
        if raw and not raw.startswith("<"):
            parts = [int(x, 16) for x in raw.split()]
            if len(parts) == 2 and unicodedata.normalize("NFC", chr(parts[0]) + chr(parts[1])) == ch:
                pairs.append([parts[0], parts[1], cp])
    data = {"unicode_version": unicodedata.unidata_version, "nfd": nfd, "nfkd": nfkd, "ccc": ccc, "compose": pairs}
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=9) as f:
        json.dump(data, f, separators=(",", ":"))


if __name__ == "__main__":
    main()
