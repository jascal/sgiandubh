#!/usr/bin/env python
"""dl2package — turn fieldrun's emitted Datalog decisions into a sgiandubh expert package.

This is the seam of the generic workflow: fieldrun extracts ANY corpus into per-decision `.dl` programs; this
converter assembles them into the package sgiandubh serves. Nothing here is domain-specific — give it a manifest
mapping each query to its decision `.dl` (one program for a single-token decision, or a numbered trace for a
generated answer) and it emits `index.json` + per-item `candidate.facts`/`contrib.facts`.

Manifest (JSON):
  {"model": "<name>",
   "items": [{"id": "...", "query": "<the student-facing question/context>",
              "citation": "<source>", "dl": "<glob to this item's .dl step(s)>"}]}

Each `.dl` carries `candidate(id).`, `contrib("block", id, w).`, and a `// model predicts: "<text>" [id]` comment.
The item's ANSWER is the concatenation of the per-step predicted text (faithful — straight from the extraction);
the item's FACTS are the first step's candidate/contrib (the decision the compiled engine re-derives as the
certificate). Multi-file globs are ordered by the trailing step number in the filename.
"""
import argparse, glob, json, os, re

CAND = re.compile(r'candidate\((\d+)\)')
CONTRIB = re.compile(r'contrib\("([^"]+)",\s*(\d+),\s*(-?[\d.eE+]+)\)')
PRED = re.compile(r'model predicts:\s*"([^"]*)"\s*\[(\d+)\]')
PRED_NOTXT = re.compile(r'model predicts:\s*\[(\d+)\]')
STEP = re.compile(r'(\d+)\.dl$')


def parse_dl(path):
    txt = open(path).read()
    cands = [int(x) for x in CAND.findall(txt)]
    contribs = [(b, int(t), float(w)) for b, t, w in CONTRIB.findall(txt)]
    m = PRED.search(txt)
    if m:
        return cands, contribs, m.group(1), int(m.group(2))
    m2 = PRED_NOTXT.search(txt)
    return cands, contribs, "", (int(m2.group(1)) if m2 else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    man = json.load(open(args.manifest))
    os.makedirs(args.out, exist_ok=True)
    items = []
    for it in man["items"]:
        steps = sorted(glob.glob(it["dl"]), key=lambda p: int(STEP.search(p).group(1)) if STEP.search(p) else 0)
        if not steps:
            print(f"  skip {it['id']}: no .dl matched {it['dl']}"); continue
        answer = ""
        first_cands, first_contribs = None, None
        for i, s in enumerate(steps):
            cands, contribs, ptext, _pid = parse_dl(s)
            answer += ptext
            if i == 0:
                first_cands, first_contribs = cands, contribs
        fdir = os.path.join(args.out, f"facts_{it['id']}")
        os.makedirs(fdir, exist_ok=True)
        with open(os.path.join(fdir, "candidate.facts"), "w") as f:
            for c in first_cands:
                f.write(f"{c}\n")
        with open(os.path.join(fdir, "contrib.facts"), "w") as f:
            for b, t, w in first_contribs:
                f.write(f"{b}\t{t}\t{w}\n")
        items.append({
            "id": it["id"], "query": it["query"], "answer": answer.strip(),
            "citation": it.get("citation", ""), "facts": os.path.basename(fdir),
        })
        print(f"  {it['id']}: {len(steps)} step(s) -> answer {answer.strip()!r}")
    json.dump({"model": man.get("model", "sgiandubh"), "items": items},
              open(os.path.join(args.out, "index.json"), "w"), indent=1, ensure_ascii=False)
    print(f"package: {len(items)} items -> {args.out}/index.json")


if __name__ == "__main__":
    main()
