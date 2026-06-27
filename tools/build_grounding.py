#!/usr/bin/env python
"""build_grounding.py — corpus → package/knowledge.tsv (+ optional package/wordvec.txt).

Grounding data for sgiandubh (no runtime model — embeddings are *data*, not a model):
  knowledge.tsv : section <TAB> passage  — the owner's content, attached verbatim to answers.
  wordvec.txt   : `word v0 .. vD`        — word embeddings restricted to the corpus vocab, so the server grounds by
                  *meaning* (cosine) rather than word-overlap. The server averages this table to form passage/query
                  vectors at serve time and falls back to lexical (Jaccard) grounding when wordvec.txt is absent — so
                  the runtime stays model-free either way.

DEFAULT: pretrained GloVe (auto-fetched once into ~/.cache/sgiandubh), restricted to the corpus vocab. Pretrained
vectors live in one shared, well-calibrated semantic space, so cosines actually mean something — that is what makes
off-domain queries abstain reliably and in-domain citations land on the right passage. This is the recommended path.
  --corpus-vectors : the older PPMI+SVD-over-THIS-corpus vectors (no download, but noisy on small corpora, and each
                     spoke gets its own incompatible space — which is why off-domain queries leak).
  --pretrained P   : use a specific vectors file (GloVe/fastText text format) instead of auto-fetching GloVe.
  --dim 0          : lexical only (no wordvec.txt; needs no numpy/scipy).
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


def ensure_glove(dim, cache=os.path.expanduser("~/.cache/sgiandubh")):
    """Path to glove.6B.<dim>d.txt, extracting from (and downloading once if needed) glove.6B.zip in the cache."""
    import zipfile, urllib.request
    txt = os.path.join(cache, f"glove.6B.{dim}d.txt")
    if os.path.exists(txt):
        return txt
    os.makedirs(cache, exist_ok=True)
    zp = os.path.join(cache, "glove.6B.zip")
    if not os.path.exists(zp):
        url = "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip"
        print(f"downloading GloVe (~822MB, one-time) -> {zp}")
        urllib.request.urlretrieve(url, zp)
    print(f"extracting glove.6B.{dim}d.txt -> {cache}")
    with zipfile.ZipFile(zp) as z:
        z.extract(f"glove.6B.{dim}d.txt", cache)
    return txt


def build_pretrained(sentences, vec_path, cap):
    """Restrict pretrained vectors (GloVe/fastText text format) to the corpus vocab, L2-normalize each row.
    Returns [(word, vec)], dim. Off-domain words simply aren't in the vocab, so they don't contribute → abstain."""
    freq = Counter()
    for s in sentences:
        freq.update(set(toks(s)))
    vocab = set(w for w, _ in freq.most_common(cap))
    out, dim = [], 0
    with open(vec_path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            if len(parts) < 3 or parts[0] not in vocab:
                continue                                       # first line of a fastText file is "count dim" → skipped
            vec = np.asarray(parts[1:], dtype=np.float32)
            if dim == 0:
                dim = len(vec)
            elif len(vec) != dim:
                continue
            n = float(np.linalg.norm(vec))
            out.append((parts[0], vec / n if n > 0 else vec))
    return out, dim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-len", type=int, default=20, help="drop passages shorter than this (chars)")
    ap.add_argument("--dim", type=int, default=64, help="0 = lexical only (no wordvec.txt); >0 = build vectors "
                    "(SVD dim for --corpus-vectors; ignored for pretrained, which uses --glove-dim)")
    ap.add_argument("--vocab-cap", type=int, default=4000)
    ap.add_argument("--corpus-vectors", action="store_true",
                    help="use corpus-derived PPMI+SVD vectors instead of the default pretrained GloVe")
    ap.add_argument("--pretrained", help="pretrained word vectors file (GloVe/fastText text format); "
                    "default: auto-fetch GloVe")
    ap.add_argument("--glove-dim", type=int, default=300, choices=[50, 100, 200, 300],
                    help="GloVe dimension to auto-fetch when no --pretrained given (default 300)")
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
        if a.corpus_vectors:
            vecs, k = build_wordvec(passages, a.dim, a.vocab_cap)
            method = "PPMI+SVD over corpus"
        else:                                                  # DEFAULT: pretrained GloVe, restricted to corpus vocab
            vpath = a.pretrained or ensure_glove(a.glove_dim)
            vecs, k = build_pretrained(passages, vpath, a.vocab_cap)
            method = f"pretrained {os.path.basename(vpath)}"
        if vecs:
            with open(os.path.join(a.out, "wordvec.txt"), "w", encoding="utf-8") as f:
                for word, vec in vecs:
                    f.write(word + " " + " ".join(f"{x:.5f}" for x in vec) + "\n")
            print(f"wordvec.txt: {len(vecs)} words x {k}d ({method}) -> {a.out}/wordvec.txt")
        else:
            print("wordvec: no vectors built (corpus too small / no vocab overlap); lexical grounding only")


if __name__ == "__main__":
    main()
