# Building an expert — the generic workflow

sgiandubh + [fieldrun](../fieldrun) form a generic factory for **small, targeted, bounded experts**: give it a
corpus for a domain, get back a tiny standalone OpenAI server that answers within that material and abstains outside
it. Nothing in the pipeline is domain-specific — the corpus *is* the configuration.

```
   YOUR CORPUS              fieldrun (extractor, heavy, once)         sgiandubh (runtime, tiny, embedded)
   ───────────              ─────────────────────────────────        ──────────────────────────────────
   textbook / Q&A /   ──►   convert model → bundle                    dl2package: .dl  →  expert package
   manual / dialogues       --export-logic-corpus → per-decision .dl  build.sh: souffle -o engine + g++ server
   (+ citations)            (candidate / contrib / predicted)    ──►  ./sgiandubh  →  OpenAI /v1/chat/completions
```

## Stage 1 — extract (fieldrun, with a GPU-free CPU box; run once)
```bash
fieldrun convert --model <hf-id|dir> --arch <arch> --dtype int4 -o bundles/m
fieldrun --bundle bundles/m --export-logic-corpus domain.txt --out export/ [--steps N] [--all-positions]
```
Produces one `.dl` per decision: `candidate(id)`, `contrib("block", id, w)`, and a `// model predicts: "<text>" [id]`.
With `--steps N` each prompt yields a short *generated* answer (a numbered trace); the model is used **only here**.

## Stage 2 — package (sgiandubh `tools/dl2package.py`)
Map each query to its decision `.dl` in a manifest, then assemble:
```bash
python tools/dl2package.py --manifest manifest.json --out package
```
This is the seam — it turns fieldrun's raw decisions into the package the server consumes. Generic; no domain logic.

## Stage 3 — build (sgiandubh)
```bash
./build.sh        # souffle -o engine  +  g++ server  →  build/{engine, sgiandubh}  (~1.4 MB)
```

## Stage 4 — deploy
```bash
./build/sgiandubh package 8080      # OpenAI-compatible, offline, no model, no GPU
```
Point any OpenAI client at it. In-scope → cited answer; out-of-scope → abstain (the bound, by construction).

## The package contract (`package/`)
`index.json`:
```json
{ "model": "<name>",
  "items": [ { "id": "...", "query": "<student-facing question/context>",
               "answer": "<generated answer text>",      // OR "options": ["a","b",...] for live MC decisions
               "citation": "<source/section>",
               "facts": "<facts dir>" } ] }
```
`<facts dir>/candidate.facts` (one candidate id per line) and `<facts dir>/contrib.facts`
(`block <TAB> id <TAB> weight`) — the decision the compiled Soufflé engine re-derives as the certificate.

## Properties (all inherent, not bolted on)
- **Bounded** — answers its material, abstains elsewhere; off-domain is structurally unanswerable, not filtered.
- **Cited** — every answer carries its source (from the manifest / fieldrun's citations).
- **Auditable** — the answer is (or is certified by) a Datalog decision; `Σ block == logit`.
- **Small & fast** — a few-MB native binary, microsecond decisions, no model/GPU at runtime.

## Roadmap (makes the generic feature richer)
- `--steps` traces → multi-token generated answers (already assembled by the converter).
- in-process libsouffle (drop the per-request spawn); log-semiring functor → `logprobs` (confidence + distractors).
- **gram kernel** (n-gram + induction) → generalize past exact-match queries; **vocab-pruned tokenizer** → smaller.
- fieldrun emitting the package + manifest directly (so Stage 2 disappears).
