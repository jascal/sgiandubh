#!/usr/bin/env python
"""build_grounding.py — corpus → package/knowledge.tsv (section <TAB> passage).

The grounding layer: at serve time sgiandubh attaches the best-matching passage from THIS file (verbatim + section)
to every answer, so each answer is grounded in the owner's own content. Built from the corpus text alone (no model).
Input lines may be section-tagged (`[§1.2 ...] passage`) or plain (section left blank). Lines are split into
sentences so grounding is at sentence granularity.
"""
import argparse, os, re

SEC = re.compile(r'^\s*\[\s*\xa7?\s*([^\]]+?)\s*\]\s*(.*)$')  # [§1.2 Overview] passage
SENT = re.compile(r'(?<=[.!?])\s+')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-len", type=int, default=20, help="drop passages shorter than this (chars)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    n = 0
    with open(os.path.join(a.out, "knowledge.tsv"), "w", encoding="utf-8") as w:
        for line in open(a.corpus, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            m = SEC.match(line)
            sec, text = (m.group(1).strip(), m.group(2).strip()) if m else ("", line)
            for sent in SENT.split(text):
                sent = sent.strip()
                if len(sent) >= a.min_len:
                    w.write(f"{sec}\t{sent}\n")
                    n += 1
    print(f"knowledge.tsv: {n} passages -> {a.out}/knowledge.tsv")


if __name__ == "__main__":
    main()
