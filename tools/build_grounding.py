#!/usr/bin/env python
"""build_grounding.py — corpus → package/knowledge.tsv (+ optional package/wordvec.txt).

Grounding data for sgiandubh (no runtime model — embeddings are *data*, not a model):
  knowledge.tsv : section <TAB> passage  — the owner's content, attached verbatim to answers.
  wordvec.txt   : `word v0 .. vD`        — corpus-derived word embeddings (PPMI + truncated SVD over THIS corpus),
                  so the server can ground by *meaning* (cosine) instead of word-overlap. Built from the corpus
                  alone; the server forms passage/query vectors by averaging this table at load/serve time, and
                  falls back to lexical (Jaccard) grounding when wordvec.txt is absent.

Quality scales with corpus size: a whole textbook gives useful domain vectors; a few sentences will be noisy.
Drop-in upgrade path: replace the PPMI+SVD vectors with pretrained ones (restricted to the corpus vocab) — same
file format, no server change.
"""
import argparse, os, re
from collections import Counter
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds

SEC = re.compile(r'^\s*\[\s*\xa7?\s*([^\]]+?)\s*\]\s*(.*)$')
SENT = re.compile(r'(?<=[.!?])\s+')
# matches the server's tokenizer: lowercase alnum runs, len>1, minus a small stoplist
STOP = set("the is are was were be been a an of to in on for and or but what which who how why when where do does "
           "did you your it its that this these those with as at by from about can could would should i we".split())


def toks(s):
    return [w for w in re.findall(r'[a-z0-9]+', s.lower()) if len(w) > 1 and w not in STOP]


def build_wordvec(sentences, dim, cap):
    """PPMI + truncated SVD word vectors over the corpus. Returns [(word, vec)], actual_dim."""
    tokd = [toks(s) for s in sentences]
    freq = Counter()
    for t in tokd:
        freq.update(set(t))
    vocab = [w for w, _ in freq.most_common(cap)]
    idx = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)
    if V < 4:
        return [], 0
    # within-sentence co-occurrence (presence per sentence)
    rows, cols = [], []
    for t in tokd:
        present = sorted(set(idx[w] for w in t if w in idx))
        for a in present:
            for b in present:
                if a != b:
                    rows.append(a)
                    cols.append(b)
    if not rows:
        return [], 0
    C = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(V, V)).tocsr().tocoo()
    total = C.data.sum()
    rowsum = np.asarray(coo_matrix((C.data, (C.row, C.col)), shape=(V, V)).sum(axis=1)).ravel()
    pmi = np.log((C.data * total) / (rowsum[C.row] * rowsum[C.col] + 1e-12) + 1e-12)
    P = coo_matrix((np.maximum(0.0, pmi), (C.row, C.col)), shape=(V, V)).tocsr()
    k = min(dim, V - 1)
    u, s, _ = svds(P, k=k)
    W = u * np.sqrt(np.maximum(s, 0.0))                       # V x k
    nrm = np.linalg.norm(W, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    W = W / nrm                                               # L2-normalized rows
    return list(zip(vocab, W)), k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-len", type=int, default=20, help="drop passages shorter than this (chars)")
    ap.add_argument("--dim", type=int, default=64, help="embedding dim (0 = lexical only, no wordvec.txt)")
    ap.add_argument("--vocab-cap", type=int, default=4000)
    ap.add_argument("--no-split", action="store_true", help="treat each line as one passage (don't split sentences)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    passages = []
    with open(os.path.join(a.out, "knowledge.tsv"), "w", encoding="utf-8") as w:
        for line in open(a.corpus, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            m = SEC.match(line)
            sec, text = (m.group(1).strip(), m.group(2).strip()) if m else ("", line)
            for sent in ([text] if a.no_split else SENT.split(text)):
                sent = sent.strip()
                if len(sent) >= a.min_len:
                    w.write(f"{sec}\t{sent}\n")
                    passages.append(sent)
    print(f"knowledge.tsv: {len(passages)} passages -> {a.out}/knowledge.tsv")

    if a.dim > 0:
        vecs, k = build_wordvec(passages, a.dim, a.vocab_cap)
        if vecs:
            with open(os.path.join(a.out, "wordvec.txt"), "w", encoding="utf-8") as f:
                for word, vec in vecs:
                    f.write(word + " " + " ".join(f"{x:.5f}" for x in vec) + "\n")
            print(f"wordvec.txt: {len(vecs)} words x {k}d (PPMI+SVD) -> {a.out}/wordvec.txt")
        else:
            print("wordvec: corpus too small for vectors; lexical grounding only")


if __name__ == "__main__":
    main()
