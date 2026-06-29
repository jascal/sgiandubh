# sgiandubh

A *sgian-dubh* is the small concealed blade. This is the small, concealed expert: a **standalone, OpenAI-compatible
server** for a **bounded domain expert** distilled out of a large model — a native C++ binary whose per-decision decode
is a ~15-line **semiring combine** (`logit = Σ contrib`, `decide = argmax`; the formal spec is `src/engine.dl`, ported
to C++ in `src/rosetta_package.h`). **No model, no GPU, no [fieldrun](../fieldrun), no Soufflé at runtime.**

It answers everything in its material and **abstains** on everything else — by construction, not by a filter. Every
answer is cited and reproducible from an auditable decode. Tens of milliseconds, a few MB, runs anywhere.

## The two-stage pipeline
```
fieldrun / rosetta  (heavy, once)               sgiandubh  (tiny, embedded)
────────────────────────────────               ─────────────────────────
model → corpus → distil → EXPERT PACKAGE   →    g++  +  OpenAI server
        (Datalog facts + n-gram/induction              ↓
         + citations + decode rules)            one small native binary, serves /v1/chat/completions
```
fieldrun (the extractor) and sgiandubh (the runtime) are coupled **only by the package** — never by code. fieldrun is
used once, offline, to manufacture the expert; sgiandubh ships without it.

## Expert package (what fieldrun emits / `package/`)
- `index.json` — items: `{id, question, options, citation, facts}` (the retrieval keys + citations + scope).
- `<facts>/candidate.facts` — the candidate set for a decision.
- `<facts>/contrib.facts` — `block <TAB> candidate <TAB> weight`: each residual block's contribution (`Σ = logit`).
- (planned) the **gram kernel** (n-gram KB + induction) for generalizing past exact matches, and per-step traces.

## Runtime loop
1. **match** the query to the item set (lexical now; embeddings later). Below threshold → **abstain** (the bound).
2. **decide** by the C++ semiring decode of the item's facts (`Σ contrib`, `argmax` — the tropical decode).
3. reply in OpenAI shape, with the **citation**. (Planned: the log-sum-exp ⊕ as a functor → calibrated confidence +
   distractor mass on the standard `logprobs` field.)

## Build & run
```bash
./build.sh                              # cargo (tokenizer FFI) + g++ → ONE binary, no spawn
./build/sgiandubh package 8080          # OpenAI server; no model, no GPU, no Soufflé
curl localhost:8080/v1/chat/completions -d '{"messages":[{"role":"user","content":"smallest unit of an organism?"}]}'
```
Answers in-scope come back with a citation and `logprobs` (the candidate distribution = confidence + distractor
mass); off-scope abstains.
Needs `g++` (C++17) and `rust`/`cargo` (builds the tokenizer FFI staticlib). Header deps (`cpp-httplib`,
`nlohmann/json`) vendored in `third_party/`.

## Why small & fast
The model cost was paid once, at extraction. Serving is a lexical match + a **native semiring combine** over a handful
of candidates — microseconds, MBs of RAM, no neural forward. Size lives in the *facts* (bounded by the material) plus a
**vocabulary-pruned tokenizer** (a bounded expert emits only its material's tokens) — a few MB total, vs the GBs of the
model it came from.

## Routing (per request)
1. **faithful** — query matches a distilled item → replay its answer (the model's exact decision) + citation + `logprobs`.
2. **gram** — in-domain but no distilled answer → the **gram kernel** continues the corpus text (n-gram + induction),
   bounded (only continuations the corpus supports), marked generative.
3. **abstain** — out of domain (vocab-overlap gate fails) → refuses, by construction.

## Status
Working: embedded in-process engine (one ~1.3 MB binary, no spawn; 1-ULP faithful); cite + abstain; **`logprobs`**
(confidence + distractor mass); **SSE streaming**; the **gram kernel** generative fallback (`tools/build_gram.py`);
generic `tools/dl2package.py` (.dl → package). The full OpenAI surface — `/v1/models`, `/v1/chat/completions`
(streaming + non-streaming), logprobs — is in place, and the runtime is tokenizer-free (it works at the text level).
See [WORKFLOW.md](./WORKFLOW.md). Next: consuming fieldrun's real emitted package (the end-to-end showcase) and the
push-button workflow (fieldrun token-text labels + `dl2package` corpus mode, to drop the hand-written manifest).

## Licence note
Code: see `LICENSE`. The `package/` demo facts are hand-built from OpenStax Anatomy & Physiology 2e (CC BY-NC-SA) — a
placeholder; a shipped expert runs on the content owner's own material.
