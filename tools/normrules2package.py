#!/usr/bin/env python
"""normrules2package.py — RISC-V norm-rules.json -> sgiandubh corpus + (empty) package index.

A *structured-source* adapter (like dl2package, but the source is a spec's machine-readable normative rules, not
fieldrun decisions). Each normative rule becomes one retrievable, citable passage. No model involved. Emits:
  <out>/rules.txt        "[norm:<name> · <chapter>] <text>"   (for build_grounding --no-split: one passage per rule)
  <out>/rules_plain.txt  "<text>"                              (for build_gram)
  <out>/index.json       {model, items: []}                    (no distilled items — served by retrieval-answer)

Then:
  build_grounding.py --corpus <out>/rules.txt --out <out> --no-split
  build_gram.py      --corpus <out>/rules_plain.txt --out <out>/gram
  sgiandubh <out> 8080 --answer-from-corpus
"""
import argparse, json, os, re


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("norm_rules", help="path to RISC-V norm-rules.json")
    ap.add_argument("out", help="output package dir")
    ap.add_argument("--model", default="sgiandubh-riscv-spec")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    d = json.load(open(a.norm_rules))
    rules = d["normative_rules"]
    n = 0
    with open(os.path.join(a.out, "rules.txt"), "w", encoding="utf-8") as ft, \
         open(os.path.join(a.out, "rules_plain.txt"), "w", encoding="utf-8") as fp:
        for r in rules:
            name = r.get("name", "")
            chap = r.get("chapter_name", "")
            text = " ".join(t.get("text", "") for t in r.get("tags", []))
            text = re.sub(r"\s+", " ", text).strip()
            if not text or not name:
                continue
            ft.write(f"[norm:{name} · {chap}] {text}\n")
            fp.write(text + "\n")
            n += 1

    json.dump({"model": a.model, "items": []},
              open(os.path.join(a.out, "index.json"), "w"), indent=1, ensure_ascii=False)
    print(f"{n} normative rules -> {a.out}/rules.txt (+ rules_plain.txt, index.json)")


if __name__ == "__main__":
    main()
