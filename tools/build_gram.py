#!/usr/bin/env python
"""build_gram — build the gram kernel (corpus n-gram + vocab) for a sgiandubh expert.

The gram kernel is the generative/generalizing half of the expert: where the distilled FACTS give the model's exact
decisions on seen contexts, the gram kernel lets the expert *continue in-domain text it wasn't distilled on* — at
n-gram fidelity, and BOUNDED to the corpus (it can only emit continuations the corpus supports; an out-of-corpus
context has no continuation → the expert abstains). Built from the corpus text alone — no model, no fieldrun.

Emits (into <out>/):
  grams.tsv  — `prev <TAB> next <TAB> count` for orders 1..N (prev = the joined previous words; "" for unigram).
  vocab.txt  — the corpus vocabulary (the in-domain bound).
"""
import argparse, collections, os, re

def tokenize(text):
    # words + sentence punctuation, case-preserved (so generation reads naturally); unicode-aware
    return re.findall(r"\w+|[.,;:!?]", text, re.UNICODE)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--order", type=int, default=3)
    ap.add_argument("--demo", help="seed text to generate a sample continuation (validation)")
    args = ap.parse_args()

    words = tokenize(open(args.corpus, encoding="utf-8").read())
    counts = collections.defaultdict(collections.Counter)  # joined-prev -> Counter(next)
    for i, w in enumerate(words):
        for o in range(1, args.order + 1):
            if i >= o - 1:
                key = " ".join(words[i - (o - 1):i])
                counts[key][w] += 1

    os.makedirs(args.out, exist_ok=True)
    rows = 0
    with open(os.path.join(args.out, "grams.tsv"), "w", encoding="utf-8") as f:
        for key, c in counts.items():
            for nxt, n in c.items():
                f.write(f"{key}\t{nxt}\t{n}\n"); rows += 1
    vocab = sorted(set(w.lower() for w in words if w.isalnum() or w.isalpha() or any(ch.isalnum() for ch in w)))
    open(os.path.join(args.out, "vocab.txt"), "w", encoding="utf-8").write("\n".join(vocab))
    print(f"gram kernel: {len(words)} tokens · {rows} n-gram rows (order {args.order}) · {len(vocab)} vocab -> {args.out}")

    if args.demo:
        # same backoff-argmax decode the C++ runtime will use (validation)
        def gen(seed, n=20):
            ctx = tokenize(seed); out = []
            for _ in range(n):
                nxt = None
                for o in range(min(len(ctx), args.order - 1), -1, -1):
                    key = " ".join(ctx[len(ctx) - o:]) if o else ""
                    if key in counts and counts[key]:
                        nxt = counts[key].most_common(1)[0][0]; break
                if nxt is None:
                    break
                out.append(nxt); ctx.append(nxt)
                if nxt in ".!?":
                    break
            return out
        cont = gen(args.demo)
        print(f"  seed:    {args.demo!r}")
        print(f"  gram →   {' '.join(cont)!r}")

if __name__ == "__main__":
    main()
