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

## One command (the two demo experts)
The whole staged pipeline below is wrapped for the riscv + logic demo experts — distilled, end to end, on a CPU box:
```bash
# prereqs: souffle, g++, rust+cargo (builds the tokenizer FFI staticlib), python3+numpy+scipy, a built fieldrun, a model bundle. Logic corpus ships in corpora/;
# RISC-V needs norm-rules.json from a riscv-isa-manual release.
BUNDLE=~/.cache/fieldrun/bundles/Qwen2.5-7B-Instruct/Qwen2.5-7B-Instruct \
NORM_RULES=/path/to/norm-rules.json ./tools/build_experts.sh        # ONLY=logic|riscv to build one; STEPS=N for depth
```
`tools/build_expert.sh` is the reusable one-expert builder it calls (`--out … --bundle … --questions … --grounding …`);
use it directly for your own corpus. The stages below are what those scripts run.

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
python tools/build_grounding.py --corpus knowledge.txt --out package                   # GloVe (default) + knowledge.tsv
python tools/build_grounding.py --corpus knowledge.txt --out package --corpus-vectors  # PPMI+SVD over this corpus
python tools/build_grounding.py --corpus knowledge.txt --out package --dim 0           # lexical only (no numpy/scipy)
```
Attaches the best-matching passage from the owner's **own content** (verbatim + section) to EVERY answer — faithful
and generative alike — so each answer is grounded in their material, not just labeled. The runtime is model-free
(the embeddings are *data*): the server averages `wordvec.txt` for passage/query vectors, grounds by **meaning**
(cosine), and falls back to lexical word-overlap when no vectors are shipped.

**Default: pretrained GloVe** (auto-fetched once to `~/.cache/sgiandubh`, restricted to the corpus vocab). Pretrained
vectors share one well-calibrated semantic space, so cosines mean something — that's what makes off-domain queries
**abstain** and in-domain citations land on the **right** passage. `--corpus-vectors` keeps the older PPMI+SVD path
(no download, but noisy on small corpora and per-spoke-incompatible — off-domain leaks). `--pretrained <file>` uses a
specific vectors file; `--dim 0` skips vectors (no numpy/scipy).
> Changing the vector source changes the cosine scale, so the abstain gates (`--answer-cos` / `--answer-margin`,
> Stage 4) should be **recalibrated on a representative test set** when you switch — don't tune them to a small corpus.
> Ceiling: mean-pooled word vectors have a similarity floor; a sentence-embedding model would separate better but would
> need a model at query time (breaking model-free runtime), so GloVe is the deliberate sweet spot.

## Stage 3 — build (sgiandubh)
```bash
./build.sh                                   # souffle -g engine + g++  →  ONE binary (engine embedded), ~1.2 MB
./build.sh --with-core                       # also union the ergo core reasoning KB into the engine (derive, don't just recall)
CORE_DL=/path/to/ergo/core.dl ./build.sh     # ...from an explicit path (default: ../ergo/core.dl)
```
`--with-core` is opt-in: it reads ergo's *single* `core.dl` at build time into an ephemeral combined `.dl` under
`build/` (gitignored, regenerated each build) — no committed copy. Off by default → the plain decode engine. (The
shared reasoning rules are compiled in once; the per-expert structured-shadow *facts* are loaded like the decode facts.
See `claymore/docs/core-reasoning-kb.md` and the `ergo` repo.)

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
`--no-gram` disables the generative tail entirely (faithful → retrieval → abstain only) — the strongest-trust config:
no n-gram continuations, so every served answer is a distilled item or a verbatim corpus passage, else it abstains.

**Tunable matching thresholds** (defaults are conservative; *calibrate against a representative test set, not a toy
corpus*). All are flags, printed at startup: `--tau` (faithful lexical-Jaccard match, default 0.25 — below this the
query falls through to retrieval/gram/abstain, so a single shared common word can't trigger a wrong faithful answer),
`--answer-cos` / `--answer-lex` / `--answer-margin` (the retrieval-as-answer gate: min cosine / lexical overlap, and
how far the top passage must beat the mean — the off-domain reject), `--ground-cos` / `--ground-lex` (the bar to
attach a supporting passage). Raise them to abstain more (precision), lower them to answer more (recall).

**APIs** (verified against the official `openai` and `anthropic` SDKs):
- OpenAI: `POST /v1/chat/completions`, `POST /v1/completions`, `GET /v1/models` — non-streaming + SSE `stream:true`,
  with `logprobs` and the standard `created`/`usage` fields.
- Anthropic: `POST /v1/messages` — content-block response + Anthropic SSE events; accepts string or block content.
- **Structured output** for an embedded app: `response_format:{type:"json_object"}` → `content` is a JSON string of
  the answer's components `{answer, kind, citation, source, confidence}`, so the app renders its own UI.
- `--repl` — interactive stdin loop for local testing (no server): prints the answer + kind + confidence.
- `GET /health` (alias `/healthz`) — readiness/liveness: `{model, engine, items, gram, grounding, knowledge_passages,
  status}` (200 ok / 503 if the embedded engine failed to load). Matches claymore's `/health` so a hub can probe it.

**Structured retrieval — NON-STANDARD extension** (deliberately *not* under `/v1`, which is reserved for the
OpenAI-compatible surface above). There is no OpenAI endpoint for "return a set of matching passages", so this is a
small documented extension for "list / table / all X" queries the single-best chat path can't serve. Aggregation lives
in the spoke (it owns the corpus + index), and the hub exposes it to LLMs as an ordinary tool — so the `/v1` surface
stays a pristine drop-in and only callers that want aggregation touch the extension.
- `POST /retrieve` — body `{query?, section?, k?, min_score?}` → `{count, matches:[{section, passage, score}]}`: a
  *set* of cited passages ranked by similarity, not the single best. An empty `query` with a `section` lists that whole
  section (pure faceted enumeration); a `query` ranks matches; `section` filters by facet (substring). Bounded:
  `count` 0 when nothing clears `min_score`.
- `GET /sections` — the distinct facets (the `· Facet` tail of each citation, deduped) you can filter `/retrieve` by.
  Note: facets are limited to what the package carries (currently the citation's section). Richer faceting (e.g. an
  instruction's *type*) needs the source adapter to preserve that structure instead of flattening it into passages.

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
