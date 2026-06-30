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
struct Passage { std::string id, section, text, text_lc; std::set<std::string> w; std::vector<float> vec; };
static std::vector<Passage> g_knowledge;  // loaded from package/knowledge.tsv (optional)
static std::unordered_map<std::string, std::vector<float>> g_wordvec; // corpus word embeddings (package/wordvec.txt)
static int g_dim = 0;                      // embedding dim (>0 once wordvec.txt is loaded → cosine grounding)
static double g_ground_tau = 0.10;        // min lexical overlap to ground (lexical fallback)
static double g_cos_tau = 0.35;           // min cosine to ground (attach a supporting passage)
static double g_answer_cos_tau = 0.60;    // stricter bar to RETURN a passage AS the answer (retrieval-answer)
static double g_answer_lex_tau = 0.18;    //   "  (lexical fallback)
static double g_answer_margin = 0.25;     // top match must beat the mean cosine by this (off-domain = flat → reject)
// cos/margin calibrated against rosetta examples/riscv/testset.jsonl (pack.score_retrieval): 0.50/0.20 leaked 33%
// off-domain; 0.60/0.25 → 0% leak at 100% in-domain recall (coverage carries recall at the stricter cosine bar).
// Small testset (18 q) — directional + safe-side (stricter = more abstaining, the bounded-expert bias); --answer-cos
// / --answer-margin remain exposed for per-deployment tuning against a larger set.
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
        "this","these","those","with","as","at","by","from","about","can","could","would","should","i","we"};
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
        p.text_lc.resize(p.text.size());
        std::transform(p.text.begin(), p.text.end(), p.text_lc.begin(),
                       [](unsigned char c) { return (char)std::tolower(c); });    // for the definitional re-rank
        p.w = words(p.text);
        if (p.w.empty()) continue;
        p.vec = embed(p.w);  // precompute the passage embedding (empty if no vectors loaded)
        g_knowledge.push_back(std::move(p));
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

// Is this a definitional query — "what is X", "define X", "describe/explain X", "what does X mean"? Such a query wants
// the passage that DEFINES its head term, not just any passage mentioning it. Used to enable the defining-passage boost.
static bool is_definitional(const std::string& raw) {
    std::string q;
    for (char c : raw) { if (c == '\t' || c == '\n') c = ' '; q += (char)std::tolower((unsigned char)c); }
    while (!q.empty() && q[0] == ' ') q.erase(0, 1);
    auto starts = [&](const char* s) { return q.rfind(s, 0) == 0; };
    return starts("what is") || starts("what are") || starts("what's") || starts("whats ") ||
           starts("define ") || starts("definition of") || starts("describe ") || starts("explain ") ||
           (starts("what does") && q.find("mean") != std::string::npos);
}

// Whole-word position of `term` in `text` (bounded by non-alnum), or npos.
static size_t word_pos(const std::string& text, const std::string& term) {
    for (size_t from = 0;;) {
        size_t p = text.find(term, from);
        if (p == std::string::npos) return std::string::npos;
        bool lok = (p == 0) || !std::isalnum((unsigned char)text[p - 1]);
        size_t e = p + term.size();
        bool rok = (e >= text.size()) || !std::isalnum((unsigned char)text[e]);
        if (lok && rok) return p;
        from = p + 1;
    }
}

// Does `text_lc` DEFINE `term`? True when the term is in subject position with a copula/defining cue — "<term> is / are
// / refers to / denotes / means …" just after it, or "termed / called / known as <term>" just before. A bare mention
// ("…the hart has retired") is not a definition. Cheap, order-aware (a word SET can't see "X is"); intentionally loose.
static bool defines_term(const std::string& text_lc, const std::string& term) {
    size_t p = word_pos(text_lc, term);
    if (p == std::string::npos) return false;
    std::string after = text_lc.substr(p + term.size(), 24);
    // A definition has the copula DIRECTLY after the subject ("hart is a", "satp CSR is an") — allow a short noun head
    // in between but NOT a clause boundary. So require a copula+noun-phrase cue ("is a/an/the", "are", "refers to", …)
    // with no punctuation between the term and the cue: that admits "satp CSR is an…" but rejects "…a hart h, s is a…".
    for (const char* cue : {" is a", " is an", " is the", " is one", " is defined", " is used to",
                            " are ", " refers to", " denotes", " means "}) {
        size_t k = after.find(cue);
        if (k != std::string::npos && after.substr(0, k).find_first_of(",.;:()") == std::string::npos)
            return true;
    }
    std::string before = text_lc.substr(p >= 18 ? p - 18 : 0, p >= 18 ? 18 : p);
    for (const char* cue : {"termed", "called", "known as", "term "})
        if (before.find(cue) != std::string::npos) return true;
    return false;
}

// Retrieve the best passage to return AS the answer, with a MARGIN gate: the top match must stand out from the field.
// An off-domain query produces a flat, uniformly-mediocre cosine profile (no clear winner) → nullptr → abstain.
static const Passage* retrieve_answer(const std::set<std::string>& qw, const std::string& raw = "") {
    if (g_knowledge.empty()) return nullptr;
    bool defq = is_definitional(raw);                          // "what is X" → prefer the passage that DEFINES X
    if (g_dim > 0) {
        std::vector<float> qv = embed(qw);                     // may be empty (no in-vocab query words) → lexical only
        const Passage* best = nullptr;
        double brank = -2.0, bcos = -2.0, bcov = 0.0, sum = 0.0;
        int n = 0;
        for (const auto& p : g_knowledge) {
            double cos = -2.0;
            if (!qv.empty() && (int)p.vec.size() == g_dim) {
                cos = 0;
                for (int i = 0; i < g_dim; i++) cos += (double)qv[i] * p.vec[i];
                sum += cos; n++;
            }
            double cov = coverage(qw, p.w);
            // HYBRID re-rank: coverage dominates (recall on term-heavy queries), cosine breaks ties toward the most
            // semantically-relevant of the equally-covered passages (so a ubiquitous term picks the DEFINING rule, not
            // an arbitrary mentioning one).
            double rank = cov + 0.4 * (cos > 0 ? cos : 0.0);
            // DEFINITIONAL boost: for "what is X", prefer the passage that DEFINES X over one that merely mentions it,
            // so the sourced prose ("a hart is …") beats a terse rule that happens to contain "hart". Two stacking
            // signals, gated to cov>=0.5 (an off-domain query sharing one stray word is never boosted → no leak):
            //   +0.6  prose namespace — "manual:" ids are explanatory definitions; "norm:" ids are requirements (a
            //         package-id convention; a builder-marked passage-kind column would be the cleaner long-term form).
            //   +0.3  a literal "<term> is a/are/refers to…" defining pattern (works for prose AND rules).
            if (defq && cov >= 0.5) {
                if (p.id.rfind("manual:", 0) == 0) rank += 0.6;          // prose namespace = definitions, not rules
                bool def = false;
                size_t fp = std::string::npos;
                for (const auto& t : qw) {
                    if (defines_term(p.text_lc, t)) def = true;          // a literal "<term> is a…" definition
                    size_t pos = word_pos(p.text_lc, t);
                    if (pos < fp) fp = pos;
                }
                if (def) rank += 0.3;
                // centrality tie-break: a passage that's ABOUT the term introduces it early (topic sentence), vs one
                // that mentions it in passing — a uniform signal that resolves the cov-1.0 ties cosine breaks unreliably.
                if (fp != std::string::npos)
                    rank += 0.3 * (1.0 - (double)std::min(fp, (size_t)400) / 400.0);
            }
            if (rank > brank) { brank = rank; best = &p; bcos = cos; bcov = cov; }
        }
        double mean = n ? sum / n : 0.0;
        // Accept on a strong SEMANTIC match (cosine clears its bar AND stands out from the field) OR a strong LEXICAL one
        // (most query terms present) — the lexical path rescues short/term-heavy in-domain queries the cosine floor drops,
        // while off-domain (low cosine AND low coverage) is still rejected.
        bool ok = best && ((bcos >= g_answer_cos_tau && (bcos - mean) >= g_answer_margin) || bcov >= g_answer_cov_tau);
        return ok ? best : nullptr;
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

static Answer answer(const std::string& user) {
    static const std::string ABSTAIN = "That isn't covered in this material. Try rephrasing your question.";
    auto qw = words(user);
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
        const Passage* gp = retrieve_answer(qw, user);                                  // retrieval-as-answer (strict + margin)
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
    Answer a = answer(user_text(body));
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
    const char* gmode = g_knowledge.empty() ? "off" : (g_dim > 0 ? "vector" : "lexical");
    const char* gram_state = g_no_gram ? "off(--no-gram)" : (g_gram.loaded ? "on" : "off");
    fprintf(stderr, "sgiandubh: %zu items · model=%s · embedded engine · gram-kernel=%s · grounding=%s%s%s%s · listening :%d\n",
            g_items.size(), g_model.c_str(), gram_state, gmode,
            g_dim > 0 ? ("/" + std::to_string(g_dim) + "d").c_str() : "",
            g_answer_from_corpus ? " · retrieval-answer" : "",
            g_require_citation ? " · require-citation" : "", port);
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
        const Passage* hit = nullptr;
        for (const auto& p : g_knowledge) if (p.id == id) { hit = &p; break; }
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
