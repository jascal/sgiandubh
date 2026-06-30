// sgiandubh — a small, standalone, OpenAI-compatible server over a bounded expert. No model, no GPU, no fieldrun, no
// Soufflé at runtime.
//
// Per request it: (1) lexically matches the query against the extracted decision set, (2) derives the decision +
// per-candidate logits via a pure-C++ semiring decode of the per-item facts (rosetta::decode_facts: logit = Σ contrib,
// decide = argmax — engine.dl in C++), (3) replies in OpenAI /v1/chat/completions shape — grounded in the owner's
// content (the verbatim supporting passage + section), with `logprobs` (confidence + distractor mass) — or abstains
// when out of scope. With --require-citation it abstains on any answer it can't ground or cite (the regulated mode).
//
// The decode is the only "engine": a ~15-line semiring combine, verified identical to the former embedded Soufflé
// engine (max |Δlogprob| ~3e-6, float32↔float64). The Datalog reasoning path (ergo core) is retired — that role is
// rosetta's now. src/engine.dl remains as the reference spec decode_facts implements, but is not compiled.
#include "httplib.h"
#include "json.hpp"
#include "gram.h"
#include "rosetta_package.h"        // consume a rosetta expert package + the C++ semiring decode (the convergence runtime)
#include "../tok_ffi/tok_ffi.h"     // HF tokenizers via FFI — BPE tokenize for the rosetta-package path
#include <algorithm>
#include <cctype>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>
using json = nlohmann::json;

struct Item {
    std::string id, question, citation, facts, answer, route;  // route: RETRIEVED|SELECTED|COMPOSED (provenance tier)
    double margin = 1e9;                                        // thinnest-decision margin in the answer (1e9 = absent)
    std::vector<std::string> options;
};
struct Decision {
    int decide = -1;
    std::vector<std::pair<int, double>> logits; // (candidate id, logit)
};

static std::vector<Item> g_items;
static std::string g_pkg, g_model;
static double g_tau = 0.25; // faithful-match threshold (lexical Jaccard) — below this → fall through to retrieval/gram/abstain.
                            // Default kept conservative so a single shared common word (~0.20, e.g. "law") can't trigger a
                            // faithful answer (the old 0.12 let "De Morgan's law" wrongly match "excluded middle"). This and
                            // the gates below are DEFAULTS, overridable per-deployment via flags (--tau, --answer-cos, …) —
                            // tune them against a representative test set, not this small corpus.
static Gram g_gram; // generative fallback (n-gram + induction), loaded if package/gram/ exists

// Grounding: the owner's corpus passages, so every answer can carry the verbatim source it's supported by.
struct Passage { std::string id, section, text; std::set<std::string> w; std::vector<float> vec; };
static std::vector<Passage> g_knowledge;  // loaded from package/knowledge.tsv (optional)
static std::unordered_map<std::string, const Passage*> g_by_id;  // id -> passage (O(1) lookup at ~100-document scale)
// Strategy tables (package/strategy.tsv, built by ergo/strategy.dl): query-handling intents materialized at BUILD, so
// the runtime routes "how many / list" to the right aggregate DECLARATIVELY — no query heuristics in the server.
// These tables (like g_knowledge / g_wordvec) are populated ONCE at startup and only read while serving — no locking.
static std::unordered_map<std::string, std::string> g_cues;            // cue word -> intent (the i18n-able lexicon, DATA)
// The uniform answer table: intent -> [(entity, passage-id)]. A query of <intent> that NAMES <entity> is answered by
// <passage-id>. Count/list/define are just different intents — one lookup, no per-kind code (REASONING.md).
static std::unordered_map<std::string, std::vector<std::pair<std::string, std::string>>> g_answers;
static std::unordered_map<std::string, std::vector<float>> g_wordvec; // corpus word embeddings (package/wordvec.txt)
static int g_dim = 0;                      // embedding dim (>0 once wordvec.txt is loaded → cosine grounding)
static double g_ground_tau = 0.10;        // min lexical overlap to ground (lexical fallback)
static double g_cos_tau = 0.35;           // min cosine to ground (attach a supporting passage)
static double g_answer_cos_tau = 0.70;    // stricter bar to RETURN a passage AS the answer (retrieval-answer)
static double g_answer_lex_tau = 0.18;    //   "  (lexical fallback)
static double g_answer_margin = 0.30;     // top match must beat the mean cosine by this (off-domain = flat → reject)
// cos/margin calibrated against rosetta examples/riscv/testset.jsonl (pack.score_retrieval): rules-only wanted
// 0.60/0.25, but the richer prose corpus reintroduced a cosine near-collision ("boiling point" ↔ "floating point"),
// so 0.70/0.30 → 0% off-domain leak at 100% in-domain recall + precision (28 q). Small set — directional + safe-side
// (stricter = more abstaining, the bounded-expert bias); --answer-cos / --answer-margin stay exposed for tuning.
static double g_answer_cov_tau = 0.60;    // OR accept on lexical term-COVERAGE: a passage holding this fraction of the
                                          // query's content-words is a strong match even if mean-pooled cosine is low
static bool g_require_citation = false;   // --require-citation: refuse any answer that can't be grounded/cited
static bool g_answer_from_corpus = false; // --answer-from-corpus: return the best passage verbatim (retrieval-as-answer)
static bool g_no_gram = false;            // --no-gram: disable the generative tail (faithful → retrieval → abstain only) — the strongest-trust config
static bool g_repl = false;               // --repl: interactive stdin loop for local testing (no server)
static bool g_rosetta_pkg = false;        // --rosetta-package: serve a rosetta expert package (manifest.json + tokenizer), host-side

static bool stop(const std::string& w) {
    static const std::set<std::string> S = {
        "the","is","are","was","were","be","been","a","an","of","to","in","on","for","and","or","but",
        "what","which","who","how","why","when","where","do","does","did","you","your","it","its","that",
        "this","these","those","with","as","at","by","from","about","can","could","would","should","i","we",
        // common function words (so the domain gate isn't satisfied by ubiquitous corpus words like "there")
        "there","here","into","than","then","they","them","their","such","also","not","only","other","will",
        "may","must","has","have","had","being","over","under","out","up","each","both","some","any","all","more","most"};
    return S.count(w) > 0;
}
static std::set<std::string> words(const std::string& s) {
    std::set<std::string> w;
    std::string cur;
    auto flush = [&] { if (cur.size() > 1 && !stop(cur)) w.insert(cur); cur.clear(); };
    for (char c : s) {
        if (std::isalnum((unsigned char)c)) cur += (char)std::tolower((unsigned char)c);
        else flush();
    }
    flush();
    return w;
}
static double jaccard(const std::set<std::string>& a, const std::set<std::string>& b) {
    if (a.empty() || b.empty()) return 0.0;
    size_t inter = 0;
    for (const auto& x : a) if (b.count(x)) inter++;
    return (double)inter / (double)(a.size() + b.size() - inter);
}
// Fraction of the QUERY's content-words present in a passage — the lexical signal mean-pooled cosine misses on short /
// term-heavy queries ("hart", "aq rl bits"). Denominator is the query (not the union), so a focused query whose terms
// all appear scores 1.0 regardless of passage length. Pairs with cosine in a hybrid re-rank.
static double coverage(const std::set<std::string>& q, const std::set<std::string>& p) {
    if (q.empty()) return 0.0;
    size_t inter = 0;
    for (const auto& x : q) if (p.count(x)) inter++;
    return (double)inter / (double)q.size();
}

// The machine handle of a citation: a section is "id · Facet" (e.g. "norm:fence_i_op · Zifencei"); the id is the part
// before the middle dot — a stable key an LLM/agent can pass to /lookup to fetch this exact passage. No dot → whole string.
static std::string cite_id(const std::string& section) {
    auto dot = section.find("\xC2\xB7");                       // UTF-8 middle dot (·)
    std::string s = (dot == std::string::npos) ? section : section.substr(0, dot);
    size_t e = s.find_last_not_of(' ');
    return (e == std::string::npos) ? s : s.substr(0, e + 1);
}

// Corpus word embeddings (package/wordvec.txt: `word v0 .. vD`). Optional — enables cosine grounding.
static void load_wordvec(const std::string& path) {
    std::ifstream f(path);
    if (!f) return;
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream ss(line);
        std::string w;
        if (!(ss >> w)) continue;
        std::vector<float> v;
        float x;
        while (ss >> x) v.push_back(x);
        if (v.empty()) continue;
        if (g_dim == 0) g_dim = (int)v.size();
        if ((int)v.size() == g_dim) g_wordvec.emplace(std::move(w), std::move(v));
    }
}
// Mean of the in-vocabulary words' vectors, L2-normalized (empty if no word is known).
static std::vector<float> embed(const std::set<std::string>& ws) {
    if (g_dim <= 0) return {};
    std::vector<float> v(g_dim, 0.0f);
    int n = 0;
    for (const auto& w : ws) {
        auto it = g_wordvec.find(w);
        if (it == g_wordvec.end()) continue;
        for (int i = 0; i < g_dim; i++) v[i] += it->second[i];
        n++;
    }
    if (n == 0) return {};
    double nrm = 0;
    for (float x : v) nrm += (double)x * x;
    nrm = std::sqrt(nrm);
    if (nrm > 0) for (float& x : v) x = (float)(x / nrm);
    return v;
}
// Grounding passages (package/knowledge.tsv: `section <TAB> passage`). The owner's content, verbatim.
static void load_knowledge(const std::string& path) {
    std::ifstream f(path);
    if (!f) return;
    std::string line;
    while (std::getline(f, line)) {
        auto tab = line.find('\t');
        if (tab == std::string::npos) continue;
        Passage p;
        p.section = line.substr(0, tab);
        p.id = cite_id(p.section);                              // the callable handle (the slug before "·")
        p.text = line.substr(tab + 1);
        p.w = words(p.text);
        if (p.w.empty()) continue;
        p.vec = embed(p.w);  // precompute the passage embedding (empty if no vectors loaded)
        g_knowledge.push_back(std::move(p));
    }
    g_by_id.reserve(g_knowledge.size());                          // index by id AFTER loading (pointers now stable —
    for (const auto& p : g_knowledge) g_by_id.emplace(p.id, &p);  // the vector won't grow/reallocate again). O(1) lookup.
}
// Strategy tables (package/strategy.tsv): "cue <word> <intent>" + "route <intent> <kind> <arg>". Optional — built by
// the ergo strategy tier (rosetta). Lets the runtime route a query's intent to the right aggregate without any
// natural-language logic of its own (the cue lexicon IS the language, and it lives here as data).
static std::string lower(const std::string& s) {
    std::string o(s.size(), '\0');
    std::transform(s.begin(), s.end(), o.begin(), [](unsigned char c) { return (char)std::tolower(c); });
    return o;
}
// Truncate to ~n bytes WITHOUT splitting a UTF-8 sequence (else nlohmann::json::dump throws on the invalid bytes).
static std::string clip(const std::string& s, size_t n) {
    if (s.size() <= n) return s;
    size_t e = n;
    while (e > 0 && ((unsigned char)s[e] & 0xC0) == 0x80) e--;   // back off continuation bytes to a char boundary
    return s.substr(0, e);
}
// Does `entity` occur in `ql` as whole word(s) — bounded by non-alphanumerics, not inside a larger word? A trailing
// plural "s" is tolerated (so the entity "instruction" matches "instructions" but NOT "instructional"); "m extension"
// matches "…the m extension." but not "harm extension".
static bool phrase_in(const std::string& ql, const std::string& entity) {
    if (entity.empty()) return false;
    for (size_t p = ql.find(entity); p != std::string::npos; p = ql.find(entity, p + 1)) {
        bool lok = (p == 0) || !std::isalnum((unsigned char)ql[p - 1]);
        size_t e = p + entity.size();
        if (e < ql.size() && (ql[e] == 's' || ql[e] == 'S')) e++;     // tolerate an English plural suffix
        bool rok = (e >= ql.size()) || !std::isalnum((unsigned char)ql[e]);
        if (lok && rok) return true;
    }
    return false;
}
// Load package/strategy.tsv (built by rosetta strategy_tables + ergo/strategy.dl). Two row shapes:
//   cue    <word>   <intent>             intent lexicon  (word/intent lowercased to match the lowercased query)
//   answer <intent> <entity> <passage>   the uniform table; <entity> lowercased (matched as a substring of the query)
static void load_strategy(const std::string& path) {
    std::ifstream f(path);
    if (!f) return;
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream ss(line);
        std::string tag, a, b, c;
        if (!std::getline(ss, tag, '\t')) continue;
        if (tag == "cue" && std::getline(ss, a, '\t') && std::getline(ss, b)) g_cues[lower(a)] = b;
        else if (tag == "answer" && std::getline(ss, a, '\t') && std::getline(ss, b, '\t') && std::getline(ss, c))
            g_answers[a].push_back({lower(b), c});                // intent -> (entity, passage-id); entity lowercased
    }
}
// Best supporting passage for a cue (query + answer words): cosine over embeddings if loaded, else lexical Jaccard.
static const Passage* ground(const std::set<std::string>& cue, double cos_tau = -1, double lex_tau = -1) {
    if (cos_tau < 0) cos_tau = g_cos_tau;
    if (lex_tau < 0) lex_tau = g_ground_tau;
    if (g_dim > 0) {
        std::vector<float> qv = embed(cue);
        if (!qv.empty()) {
            const Passage* best = nullptr;
            double bs = cos_tau;
            for (const auto& p : g_knowledge) {
                if ((int)p.vec.size() != g_dim) continue;
                double s = 0;
                for (int i = 0; i < g_dim; i++) s += (double)qv[i] * p.vec[i];
                if (s > bs) { bs = s; best = &p; }
            }
            return best;
        }
    }
    const Passage* best = nullptr;
    double bs = lex_tau;
    for (const auto& p : g_knowledge) {
        double s = jaccard(cue, p.w);
        if (s > bs) { bs = s; best = &p; }
    }
    return best;
}

// Retrieve the best passage to return AS the answer, with a MARGIN gate: the top match must stand out from the field.
// An off-domain query produces a flat, uniformly-mediocre cosine profile (no clear winner) → nullptr → abstain.
// Per-shard partial of the retrieval scan (best-by-rank + cosine mean accumulator) — reduced across threads.
struct RetrievePart {
    const Passage* best = nullptr;
    double brank = -2.0, bcos = -2.0, bcov = 0.0, sum = 0.0;
    long n = 0;
};

static const Passage* retrieve_answer(const std::set<std::string>& qw) {
    if (g_knowledge.empty()) return nullptr;
    if (g_dim > 0) {
        std::vector<float> qv = embed(qw);                     // may be empty (no in-vocab query words) → lexical only
        const size_t N = g_knowledge.size();
        // Scan a contiguous shard [s,e) → its RetrievePart. The cosine scan is the per-query cost; at ~100-document
        // scale (tens of thousands of passages) it is split across threads. Shards are processed in index order and the
        // reduction uses strict '>', so the result is IDENTICAL to the serial scan regardless of thread count.
        // The scan body calls only pure, read-only helpers (the dot product over p.vec/qv, and coverage() over the
        // word sets) — no shared mutable state — so each shard runs lock-free; pt is written by exactly one thread.
        auto scan = [&](size_t s, size_t e, RetrievePart& pt) {
            for (size_t i = s; i < e; i++) {
                const Passage& p = g_knowledge[i];
                double cos = -2.0;
                if (!qv.empty() && (int)p.vec.size() == g_dim) {
                    cos = 0;
                    const float* pv = p.vec.data();
                    for (int j = 0; j < g_dim; j++) cos += (double)qv[j] * pv[j];
                    pt.sum += cos; pt.n++;
                }
                double cov = coverage(qw, p.w);                // coverage dominates; cosine breaks ties (define vs mention)
                double rank = cov + 0.4 * (cos > 0 ? cos : 0.0);
                if (rank > pt.brank) { pt.brank = rank; pt.best = &p; pt.bcos = cos; pt.bcov = cov; }
            }
        };
        unsigned T = std::thread::hardware_concurrency();
        // ~4000 = a conservative break-even heuristic (not hardware-profiled): below it the thread spawn/join cost
        // outweighs a ~4k×300d scan; the cap of 8 avoids oversubscription on many-core hosts. Tune if profiling warrants.
        if (T < 2 || N < 4000) T = 1; else T = std::min<unsigned>(T, 8);
        std::vector<RetrievePart> parts(T);
        if (T == 1) {
            scan(0, N, parts[0]);
        } else {
            std::vector<std::thread> ts;
            size_t chunk = (N + T - 1) / T;
            for (unsigned t = 0; t < T; t++) {
                size_t s = (size_t)t * chunk, e = std::min(N, s + chunk);
                if (s >= e) break;
                ts.emplace_back([&scan, s, e, &parts, t] { scan(s, e, parts[t]); });
            }
            for (auto& th : ts) th.join();
        }
        RetrievePart g;                                        // reduce in shard order (== serial tie-breaking)
        for (const auto& pt : parts) {
            g.sum += pt.sum; g.n += pt.n;
            if (pt.brank > g.brank) { g.brank = pt.brank; g.best = pt.best; g.bcos = pt.bcos; g.bcov = pt.bcov; }
        }
        double mean = g.n ? g.sum / g.n : 0.0;
        // Accept on a strong SEMANTIC match (cosine clears its bar AND stands out from the field) OR a strong LEXICAL one
        // (most query terms present) — the lexical path rescues short/term-heavy in-domain queries the cosine floor drops,
        // while off-domain (low cosine AND low coverage) is still rejected.
        bool ok = g.best && ((g.bcos >= g_answer_cos_tau && (g.bcos - mean) >= g_answer_margin) || g.bcov >= g_answer_cov_tau);
        return ok ? g.best : nullptr;
    }
    return ground(qw, g_answer_cos_tau, g_answer_lex_tau);  // lexical fallback (jaccard is already discriminative)
}

// Retrieve-MANY: the structured-retrieval counterpart to retrieve_answer. Returns up to k passages scoring above a
// threshold (cosine if vectors are loaded, else lexical Jaccard), best-first, optionally restricted to a `section`
// (the one facet the package carries). This is the spoke-side primitive for "list / table / all X" queries the
// single-best retriever can't serve — aggregation belongs where the data + index live, not at the hub.
// An EMPTY query with a section lists everything in that section (pure faceted enumeration).
static std::vector<std::pair<const Passage*, double>>
retrieve_many(const std::set<std::string>& qw, int k, double min_score, const std::string& section) {
    std::vector<std::pair<const Passage*, double>> hits;
    bool vec = g_dim > 0;
    std::vector<float> qv;
    if (vec) { qv = embed(qw); if (qv.empty()) vec = false; }
    bool list_all = qw.empty();                       // no query → the section filter alone selects
    for (const auto& p : g_knowledge) {
        if (!section.empty() && p.section.find(section) == std::string::npos) continue;
        double s;
        if (list_all) s = 1.0;
        else if (vec) {
            if ((int)p.vec.size() != g_dim) continue;
            double cos = 0; for (int i = 0; i < g_dim; i++) cos += (double)qv[i] * p.vec[i];
            s = std::max(cos, coverage(qw, p.w));         // hybrid: semantic OR lexical term-coverage
        } else s = jaccard(qw, p.w);
        if (s >= min_score) hits.push_back({&p, s});
    }
    std::sort(hits.begin(), hits.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
    if (k > 0 && (int)hits.size() > k) hits.resize(k);
    return hits;
}

// Decode a distilled item's per-decision facts → (decide, logits): the pure-C++ semiring decode (rosetta::decode_facts,
// = engine.dl). logit = Σ contrib per candidate; decide = argmax. Per-candidate logits feed the host-side softmax below.
static Decision decode_item(const std::string& facts_dir) {
    rosetta::FactsDecode fd = rosetta::decode_facts(facts_dir);
    Decision r;
    r.decide = fd.decide;
    r.logits = std::move(fd.logits);
    return r;
}

// OpenAI logprobs (chat schema): the decided token + top_logprobs over the candidates (softmax of the logits).
static json logprobs_of(const Decision& d, const Item& it) {
    if (d.logits.empty()) return nullptr;
    double m = -1e300;
    for (auto& [id, s] : d.logits) m = std::max(m, s);
    double Z = 0;
    for (auto& [id, s] : d.logits) Z += std::exp(s - m);
    double lnZ = std::log(Z);
    auto label = [&](int id) -> std::string {
        if (!it.options.empty() && id >= 0 && id < (int)it.options.size()) return it.options[id];
        return "tok:" + std::to_string(id);
    };
    json top = json::array();
    double declp = 0;
    for (auto& [id, s] : d.logits) {
        double lp = (s - m) - lnZ;
        json e; e["token"] = label(id); e["logprob"] = lp;
        top.push_back(e);
        if (id == d.decide) declp = lp;
    }
    json entry; entry["token"] = label(d.decide); entry["logprob"] = declp; entry["top_logprobs"] = top;
    json lp; lp["content"] = json::array({entry});
    return lp;
}

// One OpenAI streaming chunk (object "chat.completion.chunk").
static json stream_chunk(const json& delta, const char* finish, const json& lp) {
    json ch;
    ch["index"] = 0;
    ch["delta"] = delta;
    ch["finish_reason"] = finish ? json(finish) : json(nullptr);
    if (!lp.is_null()) ch["logprobs"] = lp;
    json c;
    c["id"] = "chatcmpl-sgiandubh";
    c["object"] = "chat.completion.chunk";
    c["created"] = (long)std::time(nullptr);
    c["model"] = g_model;
    c["choices"] = json::array({ch});
    return c;
}

static json completion(const std::string& content, const json& logprobs, int ptoks) {
    json choice;
    choice["index"] = 0;
    choice["message"] = {{"role", "assistant"}, {"content", content}};
    choice["finish_reason"] = "stop";
    choice["logprobs"] = logprobs; // null when not applicable
    json r;
    r["id"] = "chatcmpl-sgiandubh";
    r["object"] = "chat.completion";
    r["created"] = (long)std::time(nullptr);
    r["model"] = g_model;
    r["choices"] = json::array({choice});
    r["usage"] = {{"prompt_tokens", ptoks}, {"completion_tokens", 0}, {"total_tokens", ptoks}};
    return r;
}

// ---- shared answer core: a query → rendered string + structured components, used by every API and the REPL ----
struct Answer {
    std::string content;   // rendered markdown (default format)
    std::string body;      // the answer text alone (no provenance)
    std::string citation;  // section / rule id / source label ("" if none)
    std::string citation_id; // the machine handle (slug before "·") — an agent passes it to /lookup to refetch this source
    std::string passage;   // the supporting passage it's grounded in ("" if none / the answer IS the passage)
    std::string kind;      // distilled | retrieved | generated | abstain
    std::string route;     // provenance tier (RETRIEVED|SELECTED|COMPOSED) for a distilled answer ("" if n/a)
    double margin = 1e9;   // thinnest-decision margin (1e9 = absent) — the answer's fragile-link signal
    double confidence = -1; // exp(decided logprob) for a model decision, else -1
    json lp;               // OpenAI logprobs object (null unless a model decision backed it)
    int ptoks = 0;
};

// UNIFORM, table-driven strategy dispatch (package/strategy.tsv) — the de-special-cased redesign. The query's INTENT
// is supplied by the caller (the orchestrating LLM's job) or inferred from the package's cue lexicon (the only natural
// language in play, and it's data). Then ONE rule, identical for count / list / define / anything: among the answer
// rows for that intent, return the passage whose ENTITY the query NAMES (longest match wins). The entity-must-appear
// check IS the domain gate — "how many planets" names no known entity, so it falls through to retrieval → abstain.
static const Passage* strategy_answer(const std::string& raw, const std::string& caller_intent) {
    if (g_answers.empty()) return nullptr;
    std::string ql;                                              // raw query, lowercased (entities are matched against it)
    for (char ch : raw) ql += (char)std::tolower((unsigned char)ch);
    std::set<std::string> intents;                              // the candidate intents to try
    if (!caller_intent.empty()) {
        intents.insert(caller_intent);                          // a caller-supplied intent is authoritative
    } else {                                                    // else: EVERY intent whose cue word appears — not just the
        std::string tok;                                        // first. "what is the total number of X" signals both
        auto flush = [&] {                                      // define and count; the entity match below disambiguates.
            if (!tok.empty()) { auto c = g_cues.find(tok); if (c != g_cues.end()) intents.insert(c->second); }
            tok.clear();
        };
        for (char ch : ql) { if (std::isalnum((unsigned char)ch)) tok += ch; else flush(); }
        flush();
    }
    const std::string* best_id = nullptr;                        // most specific named entity across all candidate intents
    size_t blen = 0;
    for (const auto& intent : intents) {
        auto a = g_answers.find(intent);
        if (a == g_answers.end()) continue;
        for (const auto& row : a->second)                       // row = (entity, passage-id); longest named entity wins
            if (row.first.size() > blen && phrase_in(ql, row.first)) { blen = row.first.size(); best_id = &row.second; }
    }
    if (!best_id) return nullptr;                                // query named no known entity → fall through (abstain)
    auto it = g_by_id.find(*best_id);                            // O(1) — scales to ~100 documents
    return it != g_by_id.end() ? it->second : nullptr;
}

static Answer answer(const std::string& user, const std::string& caller_intent = "") {
    static const std::string ABSTAIN = "That isn't covered in this material. Try rephrasing your question.";
    auto qw = words(user);
    for (auto it = qw.begin(); it != qw.end();)                  // cue words are intent signals, not CONTENT — drop them
        it = g_cues.count(*it) ? qw.erase(it) : std::next(it);   // from the content terms (inference uses raw tokens)
    double best = 0.0;
    const Item* hit = nullptr;
    for (const auto& it : g_items) {
        double s = jaccard(qw, words(it.question));
        if (s > best) { best = s; hit = &it; }
    }
    std::string body, item_cite;
    bool is_answer = false, is_generated = false, is_retrieval = false;
    json lp = nullptr;
    Answer a;
    a.ptoks = (int)qw.size();
    a.kind = "abstain";
    if (hit && best >= g_tau) {
        item_cite = hit->citation;
        Decision d = hit->facts.empty() ? Decision{} : decode_item(g_pkg + "/" + hit->facts);
        lp = logprobs_of(d, *hit);
        if (!hit->answer.empty()) body = hit->answer;                                  // distilled answer
        else if (!hit->options.empty() && d.decide >= 0 && d.decide < (int)hit->options.size())
            body = "The answer is " + hit->options[d.decide] + ".";                     // MC: engine decides live
        else body = (d.decide >= 0 ? "decide=" + std::to_string(d.decide) : "(engine error)");
        is_answer = true; a.kind = "distilled";
        a.route = hit->route; a.margin = hit->margin;   // carry the provenance tier + fragile-link margin
    }
    if (!is_answer && g_answer_from_corpus && !g_knowledge.empty()) {
        const Passage* sp = strategy_answer(user, caller_intent);                       // uniform intent·entity → passage
        if (sp) { body = sp->text; item_cite = sp->section; is_answer = true; is_retrieval = true; a.kind = "retrieved"; }
    }
    if (!is_answer && g_answer_from_corpus && !g_knowledge.empty()) {
        const Passage* gp = retrieve_answer(qw);                                        // retrieval-as-answer (strict + margin)
        if (gp) { body = gp->text; item_cite = gp->section; is_answer = true; is_retrieval = true; a.kind = "retrieved"; }
    }
    if (!is_answer && !g_no_gram && g_gram.loaded) {
        auto toks = Gram::tokenize(user);
        std::vector<std::string> cont = g_gram.in_domain(toks) ? g_gram.generate(toks, 24) : std::vector<std::string>{};
        if (!cont.empty()) { body = detok(cont); is_answer = true; is_generated = true; a.kind = "generated"; }
    }
    if (!is_answer) { a.content = a.body = ABSTAIN; a.kind = "abstain"; return a; }

    const Passage* gp = nullptr;
    if (!is_retrieval && !g_knowledge.empty()) {                                        // ground faithful/gram answers
        std::set<std::string> cue = qw;
        auto aw = words(body);
        cue.insert(aw.begin(), aw.end());
        gp = ground(cue);
    }
    // Footnote-numbered citation: an inline [n] marker on the answer + a "Sources" block mapping each number to its
    // provenance. One source today (retrieval/faithful); the [n] scheme is what scales to multi-source / reasoning answers.
    std::string src;
    if (gp)
        src = (gp->section.empty() ? std::string("From the material") : "\xC2\xA7" + gp->section) + ": \"" + gp->text + "\"";
    else if (!item_cite.empty())
        src = item_cite;
    bool cited = !src.empty();
    if (g_require_citation && !cited) { a.content = a.body = ABSTAIN; a.kind = "abstain"; return a; }

    a.body = body;
    // Prefer the grounding passage's section, but only if it HAS one — otherwise keep the item's own citation.
    // (A section-less grounding passage must not clobber a real item citation, e.g. logic's "Open Logic Project".)
    a.citation = (gp && !gp->section.empty()) ? gp->section : item_cite;
    a.citation_id = cite_id(a.citation);                  // the callable handle for the cited source
    a.passage = gp ? gp->text : std::string();
    std::string prov_tier;   // per-answer provenance/confidence marker (debug + calibration): tier + fragile-link margin
    if (!a.route.empty() || a.margin < 1e8) {
        char mb[64];
        if (!a.route.empty() && a.margin < 1e8) snprintf(mb, sizeof mb, "%s · margin %+.2f", a.route.c_str(), a.margin);
        else if (!a.route.empty()) snprintf(mb, sizeof mb, "%s", a.route.c_str());
        else snprintf(mb, sizeof mb, "margin %+.2f", a.margin);
        prov_tier = "\n\n[provenance: " + std::string(mb) + "]";
    }
    std::string sources = cited ? "\n\n\xF0\x9F\x93\x96 Sources:\n[1] " + src : std::string();
    a.content = (cited ? body + " [1]" : body) + sources + prov_tier +
                (is_generated && !gp ? "\n\n(generated from the material)" : "");
    a.lp = lp;
    if (!lp.is_null() && lp.contains("content") && !lp["content"].empty())
        a.confidence = std::exp(lp["content"][0].value("logprob", 0.0));
    return a;
}

// Latest user-message text from an OpenAI- or Anthropic-shaped body (content may be a string or Anthropic text
// blocks), or the legacy `prompt` field. System/history are ignored — this expert answers from the query alone.
static std::string user_text(const json& body) {
    if (body.contains("prompt") && body["prompt"].is_string()) return body["prompt"].get<std::string>();
    std::string user;
    if (body.contains("messages"))
        for (const auto& m : body["messages"]) {
            if (m.value("role", "") != "user") continue;
            const auto& c = m["content"];
            if (c.is_string()) user = c.get<std::string>();
            else if (c.is_array()) { user.clear(); for (const auto& b : c) if (b.value("type", "") == "text") user += b.value("text", ""); }
        }
    return user;
}

// Did the caller ask for structured JSON? (OpenAI response_format, or a plain "format":"json".)
static bool wants_json(const json& body) {
    if (body.contains("response_format") && body["response_format"].is_object()) {
        std::string t = body["response_format"].value("type", "");
        if (t == "json_object" || t == "json_schema" || t == "json") return true;
    }
    return body.value("format", "") == "json";
}

// Structured form for an embedded app mentor: the answer's components, so the app renders its own UI.
static std::string structured_json(const Answer& a) {
    json j;
    j["answer"] = a.body;
    j["kind"] = a.kind;
    if (!a.citation.empty()) j["citation"] = a.citation;
    if (!a.citation_id.empty()) j["citation_id"] = a.citation_id;   // the handle: GET /lookup?id=<citation_id> refetches it
    if (!a.passage.empty()) j["source"] = a.passage;
    if (a.confidence >= 0) j["confidence"] = a.confidence;
    if (!a.route.empty()) j["route"] = a.route;          // provenance tier: RETRIEVED|SELECTED|COMPOSED
    if (a.margin < 1e8) j["margin"] = a.margin;          // thinnest-decision margin (fragile-link signal)
    return j.dump();
}

static std::string render(const Answer& a, bool json_fmt) { return json_fmt ? structured_json(a) : a.content; }

static json openai_text_completion(const std::string& content, const json& lp, int ptoks) { // /v1/completions
    json choice; choice["index"] = 0; choice["text"] = content; choice["finish_reason"] = "stop"; choice["logprobs"] = lp;
    json r; r["id"] = "cmpl-sgiandubh"; r["object"] = "text_completion"; r["created"] = (long)std::time(nullptr); r["model"] = g_model;
    r["choices"] = json::array({choice});
    r["usage"] = {{"prompt_tokens", ptoks}, {"completion_tokens", 0}, {"total_tokens", ptoks}};
    return r;
}

static json anthropic_message(const std::string& content, int ptoks) { // /v1/messages
    json block; block["type"] = "text"; block["text"] = content;
    json r; r["id"] = "msg_sgiandubh"; r["type"] = "message"; r["role"] = "assistant"; r["model"] = g_model;
    r["content"] = json::array({block}); r["stop_reason"] = "end_turn"; r["stop_sequence"] = nullptr;
    r["usage"] = {{"input_tokens", ptoks}, {"output_tokens", 0}};
    return r;
}

// Route-aware SSE streaming. route: "chat" (OpenAI chat) | "text" (OpenAI completions) | "anthropic" (messages).
static void stream_answer(httplib::Response& res, std::string route, std::string content, json lp, int ptoks) {
    res.set_chunked_content_provider("text/event-stream",
        [route, content, lp, ptoks](size_t, httplib::DataSink& sink) {
            auto raw = [&](const std::string& s) { sink.write(s.data(), s.size()); };
            if (route == "anthropic") {
                json msg; msg["id"] = "msg_sgiandubh"; msg["type"] = "message"; msg["role"] = "assistant";
                msg["model"] = g_model; msg["content"] = json::array(); msg["stop_reason"] = nullptr;
                msg["usage"] = {{"input_tokens", ptoks}, {"output_tokens", 0}};
                json ms; ms["type"] = "message_start"; ms["message"] = msg;
                raw("event: message_start\ndata: " + ms.dump() + "\n\n");
                json cbs; cbs["type"] = "content_block_start"; cbs["index"] = 0;
                cbs["content_block"] = {{"type", "text"}, {"text", ""}};
                raw("event: content_block_start\ndata: " + cbs.dump() + "\n\n");
            } else if (route == "chat") {
                raw("data: " + stream_chunk(json{{"role", "assistant"}}, nullptr, json(nullptr)).dump() + "\n\n");
            }
            size_t i = 0;
            while (i < content.size()) {
                size_t sp = content.find(' ', i);
                std::string piece = (sp == std::string::npos) ? content.substr(i) : content.substr(i, sp - i + 1);
                if (route == "anthropic") {
                    json d; d["type"] = "content_block_delta"; d["index"] = 0;
                    d["delta"] = {{"type", "text_delta"}, {"text", piece}};
                    raw("event: content_block_delta\ndata: " + d.dump() + "\n\n");
                } else if (route == "text") {
                    json ch; ch["text"] = piece; ch["index"] = 0; ch["finish_reason"] = nullptr;
                    json c; c["id"] = "cmpl-sgiandubh"; c["object"] = "text_completion"; c["created"] = (long)std::time(nullptr); c["model"] = g_model;
                    c["choices"] = json::array({ch});
                    raw("data: " + c.dump() + "\n\n");
                } else {
                    raw("data: " + stream_chunk(json{{"content", piece}}, nullptr, json(nullptr)).dump() + "\n\n");
                }
                i = (sp == std::string::npos) ? content.size() : sp + 1;
            }
            if (route == "anthropic") {
                raw("event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n");
                json md; md["type"] = "message_delta"; md["delta"] = {{"stop_reason", "end_turn"}};
                md["usage"] = {{"output_tokens", 0}};
                raw("event: message_delta\ndata: " + md.dump() + "\n\n");
                raw("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n");
            } else if (route == "chat") {
                raw("data: " + stream_chunk(json::object(), "stop", lp).dump() + "\n\n");
                raw("data: [DONE]\n\n");
            } else {
                json ch; ch["text"] = ""; ch["index"] = 0; ch["finish_reason"] = "stop";
                json c; c["id"] = "cmpl-sgiandubh"; c["object"] = "text_completion"; c["created"] = (long)std::time(nullptr); c["model"] = g_model;
                c["choices"] = json::array({ch});
                raw("data: " + c.dump() + "\n\n");
                raw("data: [DONE]\n\n");
            }
            sink.done();
            return true;
        });
}

// One handler for all three POST routes (OpenAI chat/completions, Anthropic messages).
static void handle(const httplib::Request& req, httplib::Response& res, const std::string& route) {
    json body;
    try { body = json::parse(req.body); }
    catch (...) { res.status = 400; res.set_content("{\"error\":\"invalid json\"}", "application/json"); return; }
    Answer a = answer(user_text(body), body.value("intent", ""));   // caller may supply intent (LLM); else cue lexicon
    std::string content = render(a, wants_json(body));
    if (body.value("stream", false)) { stream_answer(res, route, content, a.lp, a.ptoks); return; }
    if (route == "anthropic") res.set_content(anthropic_message(content, a.ptoks).dump(), "application/json");
    else if (route == "text") res.set_content(openai_text_completion(content, a.lp, a.ptoks).dump(), "application/json");
    else res.set_content(completion(content, a.lp, a.ptoks).dump(), "application/json");
}

int main(int argc, char** argv) {
    std::vector<std::string> pos;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto fval = [&](double& dst) { if (i + 1 < argc) dst = std::atof(argv[++i]); };
        if (a == "--require-citation") g_require_citation = true;
        else if (a == "--answer-from-corpus") g_answer_from_corpus = true;
        else if (a == "--no-gram") g_no_gram = true;
        else if (a == "--repl") g_repl = true;
        else if (a == "--rosetta-package") g_rosetta_pkg = true;
        // Tunable matching thresholds (defaults above are conservative; tune on a representative test set):
        else if (a == "--tau") fval(g_tau);                       // faithful lexical-Jaccard match
        else if (a == "--ground-cos") fval(g_cos_tau);            // min cosine to attach a supporting passage
        else if (a == "--ground-lex") fval(g_ground_tau);         //   "  (lexical fallback)
        else if (a == "--answer-cos") fval(g_answer_cos_tau);     // min cosine to RETURN a passage as the answer
        else if (a == "--answer-lex") fval(g_answer_lex_tau);     //   "  (lexical fallback)
        else if (a == "--answer-cov") fval(g_answer_cov_tau);     // term-coverage bar for the hybrid lexical accept
        else if (a == "--answer-margin") fval(g_answer_margin);   // top match must beat the mean by this (off-domain reject)
        else pos.push_back(a);
    }
    g_pkg = pos.size() > 0 ? pos[0] : "package";
    int port = pos.size() > 1 ? std::stoi(pos[1]) : 8080;

    if (g_rosetta_pkg) {  // --- the rosetta→sgiandubh convergence runtime: serve a rosetta expert package, host-side ---
        // No souffle engine, no per-item index.json: load manifest.json (the tiered cover) + the model's BPE tokenizer,
        // then serve queries: tokenize → trusted idioms → gated n-grams → ABSTAIN (port of rosetta/py/serve_package.py).
        rosetta::Package pk;
        try { pk = rosetta::Package::load(g_pkg + "/manifest.json"); }
        catch (const std::exception& e) { fprintf(stderr, "sgiandubh: %s\n", e.what()); return 1; }
        Tokenizer* tok = tk_new((g_pkg + "/bundle.tokenizer.json").c_str());
        if (!tok) { fprintf(stderr, "sgiandubh: --rosetta-package needs %s/bundle.tokenizer.json\n", g_pkg.c_str()); return 1; }
        fprintf(stderr, "sgiandubh rosetta-package: %d rules (%zu trusted idioms, W=%d) — type a query; blank/Ctrl-D to exit.\n",
                pk.n_rules, pk.idioms.size(), pk.W);
        char buf[512];
        std::string line;
        while (std::getline(std::cin, line)) {
            if (line.empty()) break;
            unsigned ids[256];
            int n = tk_encode(tok, line.c_str(), ids, 256);
            if (n < 0) { printf("(tokenize error)\n"); continue; }
            std::vector<int> ctx(ids, ids + n);
            auto d = pk.serve(ctx);
            if (d) {
                int m = tk_decode(tok, (unsigned)d->answer, buf, sizeof buf);
                std::string ans = m > 0 ? std::string(buf, m) : ("#" + std::to_string(d->answer));
                printf("%s   [%s/%s]  cite: %s\n", ans.c_str(), d->tier.c_str(), d->basis.c_str(),
                       d->citation.empty() ? "(none)" : d->citation.c_str());
            } else {
                printf("(abstain — not in this expert's material)\n");
            }
        }
        tk_free(tok);
        return 0;
    }

    std::ifstream f(g_pkg + "/index.json");
    if (!f) { fprintf(stderr, "sgiandubh: cannot open %s/index.json\n", g_pkg.c_str()); return 1; }
    json idx; f >> idx;
    g_model = idx.value("model", "sgiandubh");
    for (const auto& it : idx["items"]) {
        Item x;
        x.id = it.value("id", "");
        x.question = it.contains("query") ? it.value("query", "") : it.value("question", "");
        x.citation = it.value("citation", "");
        x.facts = it.value("facts", "");
        x.answer = it.value("answer", "");
        x.route = it.value("route", "");
        x.margin = it.value("margin", 1e9);
        if (it.contains("options")) for (const auto& o : it["options"]) x.options.push_back(o.get<std::string>());
        g_items.push_back(x);
    }
    g_gram.load(g_pkg + "/gram"); // optional generative fallback
    load_wordvec(g_pkg + "/wordvec.txt");      // optional corpus embeddings (enables cosine grounding)
    load_knowledge(g_pkg + "/knowledge.tsv");  // optional grounding passages (vectors computed here)
    load_strategy(g_pkg + "/strategy.tsv");    // optional intent→aggregate routing (ergo strategy tier)
    const char* gmode = g_knowledge.empty() ? "off" : (g_dim > 0 ? "vector" : "lexical");
    const char* gram_state = g_no_gram ? "off(--no-gram)" : (g_gram.loaded ? "on" : "off");
    fprintf(stderr, "sgiandubh: %zu items · model=%s · embedded engine · gram-kernel=%s · grounding=%s%s%s%s · listening :%d\n",
            g_items.size(), g_model.c_str(), gram_state, gmode,
            g_dim > 0 ? ("/" + std::to_string(g_dim) + "d").c_str() : "",
            g_answer_from_corpus ? " · retrieval-answer" : "",
            g_require_citation ? " · require-citation" : "", port);
    if (!g_answers.empty()) {
        size_t nans = 0, dangling = 0;
        for (const auto& kv : g_answers)
            for (const auto& row : kv.second) { nans++; if (!g_by_id.count(row.second)) dangling++; }
        fprintf(stderr, "sgiandubh: strategy table — %zu intent cues, %zu answer rows (uniform intent·entity → passage)\n",
                g_cues.size(), nans);
        if (dangling)                                          // a strategy row pointing at a missing passage = build/runtime drift
            fprintf(stderr, "sgiandubh: WARNING — %zu strategy rows reference passage ids absent from knowledge.tsv "
                            "(stale strategy.tsv? rebuild the package)\n", dangling);
    }
    fprintf(stderr, "sgiandubh: thresholds  faithful-tau=%.2f  answer-cos=%.2f  answer-lex=%.2f  answer-margin=%.2f  ground-cos=%.2f  ground-lex=%.2f  (override via flags)\n",
            g_tau, g_answer_cos_tau, g_answer_lex_tau, g_answer_margin, g_cos_tau, g_ground_tau);

    if (g_repl) {  // local testing: read queries from stdin, print answers (no server)
        fprintf(stderr, "sgiandubh REPL — type a query; blank line or Ctrl-D to exit.\n");
        std::string line;
        while (true) {
            fprintf(stderr, "\n> "); fflush(stderr);
            if (!std::getline(std::cin, line) || line.empty()) break;
            Answer a = answer(line);
            printf("%s\n", a.content.c_str());
            if (a.confidence >= 0) printf("  [kind=%s · confidence p≈%.2f]\n", a.kind.c_str(), a.confidence);
            else printf("  [kind=%s]\n", a.kind.c_str());
            fflush(stdout);
        }
        return 0;
    }

    httplib::Server svr;

    svr.Get("/v1/models", [](const httplib::Request&, httplib::Response& res) {
        json e; e["id"] = g_model; e["object"] = "model"; e["owned_by"] = "sgiandubh";
        json m; m["object"] = "list"; m["data"] = json::array({e});
        res.set_content(m.dump(), "application/json");
    });
    // readiness/liveness — the expert's own state (it's a leaf; no spokes). Mirrors claymore's /health.
    auto health = [](const httplib::Request&, httplib::Response& res) {
        json m;
        m["object"] = "health";
        m["status"] = "ok";
        m["model"] = g_model;
        m["engine"] = "semiring-c++";
        m["items"] = (int)g_items.size();
        m["gram"] = g_gram.loaded;
        m["grounding"] = g_knowledge.empty() ? "off" : (g_dim > 0 ? "vector" : "lexical");
        m["knowledge_passages"] = (int)g_knowledge.size();
        res.status = 200;
        res.set_content(m.dump(), "application/json");
    };
    svr.Get("/health", health);
    svr.Get("/healthz", health);

    // Structured retrieval — a NON-STANDARD extension (deliberately NOT under /v1, which is reserved for the
    // OpenAI-compatible surface). There is no OpenAI endpoint for "return a set of matching passages"; this is the
    // spoke-side primitive for aggregation, exposed to LLMs as a normal tool by the hub. The /v1 surface stays pristine.
    // POST {query?, section?, k?, min_score?} → a SET of cited passages (not the single best). For "list/table/all"
    // queries: empty query + a section lists that whole section; a query ranks the matches. Bounded: count 0 if none.
    svr.Post("/retrieve", [](const httplib::Request& q, httplib::Response& res) {
        json body; try { body = json::parse(q.body); } catch (...) { body = json::object(); }
        std::string query = body.value("query", "");
        std::string section = body.value("section", "");
        int k = body.value("k", 20);
        double min_score = body.value("min_score", g_dim > 0 ? g_cos_tau : g_ground_tau);
        auto hits = retrieve_many(words(query), k, min_score, section);
        json arr = json::array();
        for (const auto& h : hits)
            arr.push_back({{"section", h.first->section}, {"passage", h.first->text}, {"score", h.second}});
        json out;
        out["object"] = "retrieval";
        out["query"] = query;
        out["section"] = section;
        out["count"] = (int)hits.size();
        out["matches"] = arr;
        res.set_content(out.dump(), "application/json");
    });
    // Citation-as-handle (extension; not /v1): refetch the EXACT passage a citation points to, by its id. The expert
    // answers with a citation_id (the slug, e.g. "norm:fence_i_op"); an LLM/agent reading the result calls this to
    // verify / expand / quote the source verbatim. Exact-match (not the substring /retrieve), and bounded — only ids
    // that exist resolve (404 otherwise), so it can't be used to fish outside the material.
    auto lookup = [](const httplib::Request& q, httplib::Response& res) {
        std::string id = q.has_param("id") ? q.get_param_value("id") : "";
        if (id.empty() && !q.body.empty()) { try { id = json::parse(q.body).value("id", ""); } catch (...) {} }
        auto it = g_by_id.find(id);                              // O(1) exact-match by handle
        const Passage* hit = it != g_by_id.end() ? it->second : nullptr;
        json out;
        out["object"] = "lookup";
        out["id"] = id;
        out["found"] = (hit != nullptr);
        if (hit) { out["section"] = hit->section; out["passage"] = hit->text; }
        res.status = hit ? 200 : 404;
        res.set_content(out.dump(), "application/json");
    };
    svr.Get("/lookup", lookup);                               // GET /lookup?id=norm:fence_i_op   (curl/agent-friendly)
    svr.Post("/lookup", lookup);                              // POST {"id":"norm:fence_i_op"}

    // Facet discovery (extension; not /v1) — the distinct facets you can filter /retrieve by. The section field is
    // "id · Facet" (e.g. "norm:lw_op · RV32I Base ISA"); the useful facet is the part after the middle dot, so we
    // surface those (deduped) rather than the thousands of per-item ids.
    svr.Get("/sections", [](const httplib::Request&, httplib::Response& res) {
        std::set<std::string> secs;
        for (const auto& p : g_knowledge) {
            std::string s = p.section;
            auto dot = s.find("\xC2\xB7");                       // UTF-8 middle dot (·)
            if (dot != std::string::npos) {
                s = s.substr(dot + 2);
                size_t a = s.find_first_not_of(' ');
                s = (a == std::string::npos) ? std::string() : s.substr(a);
            }
            if (!s.empty()) secs.insert(s);
        }
        json out; out["object"] = "sections"; out["count"] = (int)secs.size(); out["sections"] = json(secs);
        res.set_content(out.dump(), "application/json");
    });
    // DEGENERATE LIBRARIAN: every expert describes ITSELF as a catalog card — so a hub can aggregate one self-card per
    // spoke into a federated catalog (and a librarian-package's per-document cards, which are its lib:* passages, ride
    // along). The catalog capability is universal, not a special node; claymore federates these up the hierarchy.
    svr.Get("/catalog", [](const httplib::Request&, httplib::Response& res) {
        std::set<std::string> facets;
        json doc_cards = json::array();
        for (const auto& p : g_knowledge) {
            if (p.id.rfind("lib:", 0) == 0) {                    // a librarian package: its passages ARE document cards
                doc_cards.push_back(json{{"handle", p.id}, {"title", p.section}, {"summary", clip(p.text, 280)}});
                continue;
            }
            auto dot = p.section.find("\xC2\xB7");
            std::string f = (dot != std::string::npos) ? p.section.substr(dot + 2) : "";
            size_t a = f.find_first_not_of(' ');
            if (a != std::string::npos) facets.insert(f.substr(a));
        }
        std::vector<std::string> top(facets.begin(), facets.end());
        if (top.size() > 24) top.resize(24);
        json card;                                               // the self-card (degenerate librarian)
        card["id"] = g_model;
        card["kind"] = doc_cards.empty() ? "expert" : "catalog";
        card["passages"] = (int)g_knowledge.size();
        card["facets"] = top;
        card["summary"] = "Expert '" + g_model + "' — " + std::to_string(g_knowledge.size()) + " passages" +
                          (top.empty() ? "" : (" over: " + [&] { std::string s; for (size_t i = 0; i < top.size() && i < 8; i++) s += (i ? ", " : "") + top[i]; return s; }())) + ".";
        json cards = json::array({card});
        for (auto& dc : doc_cards) cards.push_back(dc);          // librarian package: include the per-document cards too
        json out; out["object"] = "catalog"; out["cards"] = cards;
        res.set_content(out.dump(-1, ' ', false, json::error_handler_t::replace), "application/json");  // tolerate stray bytes
    });

    svr.Post("/v1/chat/completions", [](const httplib::Request& q, httplib::Response& r) { handle(q, r, "chat"); });
    svr.Post("/v1/completions", [](const httplib::Request& q, httplib::Response& r) { handle(q, r, "text"); });       // OpenAI legacy
    svr.Post("/v1/messages", [](const httplib::Request& q, httplib::Response& r) { handle(q, r, "anthropic"); });     // Anthropic

    // httplib defaults to 5 requests/connection — far too low for OpenAI clients that pool connections (the cap
    // forces a reconnect every 5 calls). Raise it so a pooled connection serves many requests; TCP_NODELAY drops
    // Nagle latency on these small request/response round-trips.
    svr.set_keep_alive_max_count(1000);
    svr.set_keep_alive_timeout(30);
    svr.set_tcp_nodelay(true);
    svr.listen("0.0.0.0", port);
    return 0;
}
