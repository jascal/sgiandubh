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

## Two ways to build (the model is optional)
- **Model-free** (fastest — specs, manuals, structured references): *skip fieldrun.* Turn the source into passages,
  then `build_grounding` + `build_gram`, and serve with `--answer-from-corpus` (return the best-matching passage
  verbatim, cited) → a bounded, grounded, cited expert with **no model, no GPU**. Example structured-source adapter:
  `tools/normrules2package.py` (RISC-V `norm-rules.json` → package; each normative rule becomes a citable passage).
- **Model-distilled** (Stages 1–4 below): run fieldrun to add *model-reasoned* answers with `logprobs` + a Datalog
  certificate — the high-stakes/auditable tier, layered on top of (or instead of) retrieval.

## Stage 1 — extract (fieldrun, with a GPU-free CPU box; run once)
```bash
fieldrun convert --model <hf-id|dir> --arch <arch> --dtype int4 -o bundles/m
fieldrun --bundle bundles/m --export-logic-corpus domain.txt --out export/ [--steps N] [--all-positions]
```
Produces one `.dl` per decision: `candidate(id)`, `contrib("block", id, w)`, and a `// model predicts: "<text>" [id]`.
With `--steps N` each prompt yields a short *generated* answer (a numbered trace); the model is used **only here**.

## Stage 2 — package (sgiandubh `tools/dl2package.py`)
The seam — turns fieldrun's raw decisions into the package the server consumes. Generic; no domain logic. Two modes:
```bash
# push-button: one item per prompt, query = the corpus line, decisions = that prompt's p{NNNNN}_*.dl
python tools/dl2package.py --corpus domain.txt --dl export/ [--cite qmap.json] --out package
# or explicit (ad-hoc / mixed sources): a manifest mapping each query to its .dl
python tools/dl2package.py --manifest manifest.json --out package
```
Corpus mode needs no hand-written manifest — give it the same corpus you extracted and the `--out` `.dl` directory.

## Stage 2.5 — gram kernel (optional, generative tail)
```bash
python tools/build_gram.py --corpus domain.txt --out package/gram   # n-gram KB + vocab from the corpus
```
Adds generative coverage: in-domain queries with no distilled answer get a corpus-bounded continuation instead of an
abstain. Built from the corpus text alone (no model). Routing at serve time: **faithful** (distilled item) →
**gram** (in-domain continuation) → **abstain** (out of domain).

## Stage 2.6 — grounding (optional, the publishing win)
```bash
python tools/build_grounding.py --corpus knowledge.txt --out package           # knowledge.tsv + wordvec.txt
python tools/build_grounding.py --corpus knowledge.txt --out package --dim 0    # lexical only (no numpy/scipy)
```
Attaches the best-matching passage from the owner's **own content** (verbatim + section) to EVERY answer — faithful
and generative alike — so each answer is grounded in their material, not just labeled. Built from the corpus text
alone (no model): `wordvec.txt` holds corpus-derived word embeddings (PPMI + SVD over the corpus), so the server
grounds by **meaning** (cosine) and falls back to lexical word-overlap when no vectors are shipped. The embeddings
are *data*, not a model — the runtime stays model-free. Vectors need numpy+scipy at build time (`--dim 0` skips
them); quality scales with corpus size, and pretrained vectors (same file format) are a drop-in upgrade.

## Stage 3 — build (sgiandubh)
```bash
./build.sh        # souffle -g engine + g++  →  ONE binary (engine embedded), ~1.2 MB
```

## Stage 4 — deploy
```bash
./build/sgiandubh package 8080 [--answer-from-corpus] [--require-citation]   # OpenAI-compatible, offline, no GPU
```
`--answer-from-corpus`: when no distilled item matches, return the best corpus passage **verbatim** (cited) instead
of generating — the safest mode for specs/manuals (the answer is literally the source text). Falls through to the
gram kernel, then abstain.
Point any OpenAI client at it. In-scope → grounded/cited answer; out-of-scope → abstain (the bound, by construction).
`--require-citation` refuses any answer it can't ground in a passage or attach a citation to — the strong mode for
regulated/high-stakes domains: every served answer carries provenance, or it abstains.

**APIs** (verified against the official `openai` and `anthropic` SDKs):
- OpenAI: `POST /v1/chat/completions`, `POST /v1/completions`, `GET /v1/models` — non-streaming + SSE `stream:true`,
  with `logprobs` and the standard `created`/`usage` fields.
- Anthropic: `POST /v1/messages` — content-block response + Anthropic SSE events; accepts string or block content.
- **Structured output** for an embedded app: `response_format:{type:"json_object"}` → `content` is a JSON string of
  the answer's components `{answer, kind, citation, source, confidence}`, so the app renders its own UI.
- `--repl` — interactive stdin loop for local testing (no server): prints the answer + kind + confidence.
- `GET /health` (alias `/healthz`) — readiness/liveness: `{model, engine, items, gram, grounding, knowledge_passages,
  status}` (200 ok / 503 if the embedded engine failed to load). Matches claymore's `/health` so a hub can probe it.

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
`knowledge.tsv` (optional): `section <TAB> passage` — the owner's content, for grounding.
`wordvec.txt` (optional): `word v0 .. vD` — corpus word embeddings; enables cosine (semantic) grounding.
`gram/` (optional): the n-gram KB for the generative tail.

## Properties (all inherent, not bolted on)
- **Bounded** — answers its material, abstains elsewhere; off-domain is structurally unanswerable, not filtered.
- **Grounded** — every answer carries the verbatim passage from the owner's content it's supported by (+ section),
  matched semantically (corpus-derived embeddings, cosine) or lexically; `--require-citation` refuses anything it
  can't ground or cite.
- **Auditable** — the answer is (or is certified by) a Datalog decision; `Σ block == logit`.
- **Small & fast** — a few-MB native binary, microsecond decisions, no model/GPU at runtime.

## Roadmap (makes the generic feature richer)
- `--steps` traces → multi-token generated answers (already assembled by the converter).
- in-process libsouffle (drop the per-request spawn); log-semiring functor → `logprobs` (confidence + distractors).
- **gram kernel** (n-gram + induction) → generalize past exact-match queries; **vocab-pruned tokenizer** → smaller.
- fieldrun emitting the package + manifest directly (so Stage 2 disappears).
