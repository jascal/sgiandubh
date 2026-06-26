# sgiandubh

A *sgian-dubh* is the small concealed blade. This is the small, concealed expert: a **standalone, OpenAI-compatible
server** for a **bounded domain expert** distilled out of a large model — compiled to native C++ over a
[Soufflé](https://souffle-lang.github.io/) decision engine. **No model, no GPU, no [fieldrun](../fieldrun) at runtime.**

It answers everything in its material and **abstains** on everything else — by construction, not by a filter. Every
answer is cited and reproducible from an auditable Datalog program. Tens of milliseconds, a few MB, runs anywhere.

## The two-stage pipeline
```
fieldrun  (heavy, once)                         sgiandubh  (tiny, embedded)
──────────────────────                          ─────────────────────────
model → corpus → distil → EXPERT PACKAGE   →    souffle -o engine  +  OpenAI server
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
2. **decide** by running the compiled Datalog engine on the item's facts (`Σ contrib`, `argmax` — the tropical decode).
3. reply in OpenAI shape, with the **citation**. (Planned: the log-sum-exp ⊕ as a functor → calibrated confidence +
   distractor mass on the standard `logprobs` field.)

## Build & run
```bash
./build.sh                              # souffle -g engine + g++ → ONE binary (engine embedded, ~1.2 MB)
./build/sgiandubh package 8080          # OpenAI server; no model, no GPU, no spawn
curl localhost:8080/v1/chat/completions -d '{"messages":[{"role":"user","content":"smallest unit of an organism?"}]}'
```
Answers in-scope come back with a citation and `logprobs` (the candidate distribution = confidence + distractor
mass); off-scope abstains.
Needs `souffle` (with compiled-mode headers — see fieldrun `SOUFFLE.md` §1.1) and `g++` (C++17). Header deps
(`cpp-httplib`, `nlohmann/json`) vendored in `third_party/`.

## Why small & fast
The model cost was paid once, at extraction. Serving is a lexical match + a **native semiring combine** over a handful
of candidates — microseconds, MBs of RAM, no neural forward. Size lives in the *facts* (bounded by the material) plus a
**vocabulary-pruned tokenizer** (a bounded expert emits only its material's tokens) — a few MB total, vs the GBs of the
model it came from.

## Status
Working: the engine is **embedded in-process** (one binary, no spawn; 1-ULP faithful to the interpreter); the server
matches → decides → cites → abstains, and emits **`logprobs`** (host-side softmax over the candidate logits =
confidence + distractor mass). Generic `tools/dl2package.py` converts fieldrun's `.dl` into a package (see
[WORKFLOW.md](./WORKFLOW.md)). Next: the **gram kernel** (generalize past lexical match), a **vocab-pruned tokenizer**
(smaller), **streaming**, and consuming fieldrun's real emitted package.

## Licence note
Code: see `LICENSE`. The `package/` demo facts are hand-built from OpenStax Anatomy & Physiology 2e (CC BY-NC-SA) — a
placeholder; a shipped expert runs on the content owner's own material.
