#!/usr/bin/env python
"""dl2package — turn fieldrun's emitted Datalog decisions into a sgiandubh expert package.

The seam of the generic workflow: fieldrun extracts ANY corpus into per-decision `.dl` programs; this converter
assembles them into the package sgiandubh serves (`index.json` + per-item `candidate.facts`/`contrib.facts`).
Nothing here is domain-specific.

Two modes:
  --corpus FILE --dl DIR     auto: one item per prompt, query = the corpus line, decisions = that prompt's `.dl`
                             (the `p{NNNNN}_{SS}.dl` naming from `--export-logic-corpus --out`). Push-button.
  --manifest FILE            explicit: a JSON list of {id, query, citation, dl(glob)} (for ad-hoc / mixed sources).

Each `.dl` carries `candidate(id).`, `contrib("block", id, w).`, and a `// model predicts: "<text>" [id]` comment.
An item's ANSWER is the concatenated per-step predicted text (faithful — straight from the extraction); its FACTS
are the first step's candidate/contrib (the decision the compiled engine re-derives as the certificate).
"""
import argparse, glob, json, os, re

CAND = re.compile(r'candidate\((\d+)\)')
CONTRIB = re.compile(r'contrib\("([^"]+)",\s*(\d+),\s*(-?[\d.eE+]+)\)')
PRED = re.compile(r'model predicts:\s*"([^"]*)"\s*\[(\d+)\]')
STEP = re.compile(r'(\d+)\.dl$')           # trailing step number in a filename
PROMPT = re.compile(r'p(\d+)_\d+\.dl$')    # --export-logic-corpus --out naming: p{prompt}_{step}.dl


# fieldrun writes token text Rust-debug-escaped ({:?}) in the .dl comment, so a newline token reads as a literal
# "\n". Undo those escapes (real Unicode like § or 📖 is left untouched — {:?} only escapes \n \t \r " \).
_ESC = {'n': '\n', 't': '\t', 'r': '\r', '"': '"', '\\': '\\', '0': '\0', "'": "'"}
def _unescape(s):
    return re.sub(r'\\(.)', lambda m: _ESC.get(m.group(1), m.group(1)), s)


def parse_dl(path):
    txt = open(path, encoding="utf-8").read()
    cands = [int(x) for x in CAND.findall(txt)]
    contribs = [(b, int(t), float(w)) for b, t, w in CONTRIB.findall(txt)]
    m = PRED.search(txt)
    return cands, contribs, (_unescape(m.group(1)) if m else "")


def build_item(out, item_id, query, citation, dl_files):
    steps = sorted(dl_files, key=lambda p: int(STEP.search(p).group(1)) if STEP.search(p) else 0)
    if not steps:
        return None
    answer = ""
    first_cands, first_contribs = [], []
    for i, s in enumerate(steps):
        cands, contribs, ptext = parse_dl(s)
        answer += ptext
        if i == 0:
            first_cands, first_contribs = cands, contribs
    fdir = os.path.join(out, f"facts_{item_id}")
    os.makedirs(fdir, exist_ok=True)
    with open(os.path.join(fdir, "candidate.facts"), "w") as f:
        for c in first_cands:
            f.write(f"{c}\n")
    with open(os.path.join(fdir, "contrib.facts"), "w") as f:
        for b, t, w in first_contribs:
            f.write(f"{b}\t{t}\t{w}\n")
    return {"id": item_id, "query": query, "answer": answer.strip(), "citation": citation, "facts": os.path.basename(fdir)}


def cite_for(cite, idx):
    if cite is None:
        return ""
    e = cite[idx] if isinstance(cite, list) and idx < len(cite) else (cite.get(str(idx), "") if isinstance(cite, dict) else "")
    if isinstance(e, dict):
        return e.get("citation") or e.get("section") or ""
    return e or ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest")
    ap.add_argument("--corpus")
    ap.add_argument("--dl")
    ap.add_argument("--cite", help="JSON list/dict of per-prompt citations (e.g. fieldrun's qmap.json)")
    ap.add_argument("--citation", default="", help="constant citation applied to every item (corpus mode)")
    ap.add_argument("--model", default="sgiandubh")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    items = []

    if args.manifest:
        man = json.load(open(args.manifest))
        args.model = man.get("model", args.model)
        for it in man["items"]:
            item = build_item(args.out, it["id"], it["query"], it.get("citation", ""), glob.glob(it["dl"]))
            if item:
                items.append(item)
                print(f"  {item['id']}: answer {item['answer']!r}")
            else:
                print(f"  skip {it['id']}: no .dl matched {it['dl']}")
    elif args.corpus and args.dl:
        lines = [l.strip() for l in open(args.corpus, encoding="utf-8") if l.strip()]  # matches fieldrun's non-empty filter
        cite = json.load(open(args.cite)) if args.cite else None
        groups = {}
        for f in glob.glob(os.path.join(args.dl, "*.dl")):
            m = PROMPT.search(os.path.basename(f))
            if m:
                groups.setdefault(int(m.group(1)), []).append(f)
        for idx in sorted(groups):
            query = lines[idx] if idx < len(lines) else f"[prompt {idx}]"
            citation = cite_for(cite, idx) or args.citation
            item = build_item(args.out, f"p{idx:05}", query, citation, groups[idx])
            if item:
                items.append(item)
        print(f"  corpus mode: {len(groups)} prompts -> {len(items)} items "
              f"({sum(1 for i in items if i['answer'])} with non-empty answers)")
    else:
        ap.error("give either --manifest, or --corpus + --dl")

    json.dump({"model": args.model, "items": items},
              open(os.path.join(args.out, "index.json"), "w"), indent=1, ensure_ascii=False)
    print(f"package: {len(items)} items -> {args.out}/index.json")


if __name__ == "__main__":
    main()
