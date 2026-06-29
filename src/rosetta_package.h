// rosetta_package.h — sgiandubh's consumer of a rosetta expert package (the convergence: rosetta builds, sgiandubh serves).
// Loads manifest.json (idiom_learn --package, or the flat abstain_emit manifest) and serves host-side, NO souffle, NO model:
//   tokenize(query) → TRUSTED idioms (frame-match) → GATED n-grams (longest suffix) → ABSTAIN.
// This is the C++ port of rosetta/py/serve_package.py (load_package + serve); the schema is rosetta/PACKAGE.md.
#ifndef ROSETTA_PACKAGE_H
#define ROSETTA_PACKAGE_H
#include "json.hpp"
#include <algorithm>
#include <fstream>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace rosetta {
using json = nlohmann::json;
using Ctx = std::vector<int>;                                    // token ids (model vocab < 2^31)

struct Idiom {                                                   // TRUSTED tier (causal): gate or compose
    std::string kind, cite;                                      // "gate" | "compose"
    int id = -1;
    std::map<int, int> frame;                                    // offset (1 = last token) -> token
    int slot = 0;                                                // gate: content slot
    std::map<int, int> table;                                    // gate: content token -> out
    int k1 = 0, k2 = 0;                                          // compose: operand offsets
    std::map<int, int> valmap, sum;                             // compose: token->value, value-sum->out
};
struct NGram { int out; std::string basis, cite; };             // GATED tier (observational)
struct Decision { int answer; std::string tier, basis, citation; int rule = -1; };

struct Package {
    std::vector<Idiom> idioms;                                   // TRUSTED, in priority order
    std::vector<std::map<Ctx, NGram>> ngrams;                    // index = suffix length; GATED
    int W = 0, n_rules = 0;

    static std::string cite_of(const json& r) {
        if (r.contains("citation") && r["citation"].is_array() && !r["citation"].empty()) {
            std::string s;
            for (auto& c : r["citation"]) { if (!s.empty()) s += ", "; s += c.get<std::string>(); }
            return s;
        }
        if (r.contains("cite") && r["cite"].is_array() && !r["cite"].empty())
            return "corpus@" + std::to_string(r["cite"][0].get<long>());
        return "";
    }
    static std::map<int, int> imap(const json& o) {              // JSON object with string int-keys -> int->int map
        std::map<int, int> m;
        for (auto it = o.begin(); it != o.end(); ++it) m[std::stoi(it.key())] = it.value().get<int>();
        return m;
    }

    static Package load(const std::string& manifest_path) {
        std::ifstream f(manifest_path);
        if (!f) throw std::runtime_error("rosetta package: cannot open " + manifest_path);
        json m;
        try { f >> m; }
        catch (const json::exception& e) {
            throw std::runtime_error("rosetta package: malformed JSON in " + manifest_path + " — " + e.what());
        }
        if (!m.contains("rules") || !m["rules"].is_array())
            throw std::runtime_error("rosetta package: " + manifest_path + " is missing a \"rules\" array");
        Package p;
        int maxlen = 0;
        for (auto& r : m["rules"])
            if (r.value("kind", std::string("ngram")) == "ngram")
                maxlen = std::max(maxlen, (int)r["ctx"].size());
        p.W = maxlen;
        p.ngrams.assign(p.W + 1, {});
        int ri = -1;
        for (auto& r : m["rules"]) {
            ++ri;
            std::string kind = r.value("kind", std::string("ngram"));
            std::string cite = cite_of(r);
            try {                                               // a malformed/short-of-fields rule names itself, not a raw json throw
                if (kind == "ngram") {
                    Ctx ctx = r.at("ctx").get<Ctx>();
                    p.ngrams[ctx.size()][ctx] = {r.at("out").get<int>(), r.value("basis", std::string("observational")), cite};
                } else if (kind == "gate") {
                    Idiom id; id.kind = "gate"; id.id = r.value("id", -1); id.cite = cite;
                    id.frame = imap(r.at("frame")); id.slot = r.at("slot").get<int>(); id.table = imap(r.at("table"));
                    p.idioms.push_back(std::move(id));
                } else if (kind == "compose") {
                    Idiom id; id.kind = "compose"; id.id = r.value("id", -1); id.cite = cite;
                    id.frame = imap(r.at("frame")); id.k1 = r.at("operands").at(0); id.k2 = r.at("operands").at(1);
                    id.valmap = imap(r.at("valmap")); id.sum = imap(r.at("sum"));
                    p.idioms.push_back(std::move(id));
                }                                               // unknown kinds are skipped (forward-compat with newer packages)
            } catch (const json::exception& e) {
                throw std::runtime_error("rosetta package: rule #" + std::to_string(ri) + " (" + kind + ") in "
                                         + manifest_path + " is malformed — " + e.what());
            }
        }
        p.n_rules = (int)(p.idioms.size());
        for (auto& tab : p.ngrams) p.n_rules += (int)tab.size();
        return p;
    }

    // Decide the next token for `ctx`. Priority encodes TRUST, not just specificity:
    //   1) TRUSTED idioms (causal): gate/compose rules fieldrun causally CONFIRMED (perturb→output follows), so they
    //      generalize out-of-distribution. They win even over a longer matching n-gram suffix — a matched idiom is
    //      *vouched*, an n-gram is only *observed*.
    //   2) GATED n-grams (observational): longest matching suffix wins (more context = more specific). Correlations that
    //      passed a build-time support/determinism gate — reliable in-distribution, not vouched.
    //   3) ABSTAIN (nullopt): no trusted idiom and no confident suffix → honest refusal (sgiandubh has no exact backstop).
    // Decision.tier ∈ {"trusted","gated"} = which fired; Decision.basis ∈ {"causal", n-gram basis e.g. "observational"}
    // = *why* it's believed — surfaced to the caller next to the provenance citation. All offset checks guard against
    // ctx shorter than a rule's frame/operand reach, so a short context abstains rather than reading out of bounds.
    std::optional<Decision> serve(const Ctx& ctx) const {
        int n = (int)ctx.size();
        for (const auto& r : idioms) {                          // TRUSTED tier (causal) first
            bool fr = true;
            for (auto& kv : r.frame) { if (kv.first > n || ctx[n - kv.first] != kv.second) { fr = false; break; } }
            if (!fr) continue;
            if (r.kind == "gate") {
                if (r.slot <= n) {
                    auto it = r.table.find(ctx[n - r.slot]);
                    if (it != r.table.end()) return Decision{it->second, "trusted", "causal", r.cite, r.id};
                }
            } else {                                            // compose
                if (r.k1 <= n && r.k2 <= n) {
                    auto a = r.valmap.find(ctx[n - r.k1]), b = r.valmap.find(ctx[n - r.k2]);
                    if (a != r.valmap.end() && b != r.valmap.end()) {
                        auto s = r.sum.find(a->second + b->second);
                        if (s != r.sum.end()) return Decision{s->second, "trusted", "causal", r.cite, r.id};
                    }
                }
            }
        }
        for (int k = std::min(n, W); k >= 1; --k) {             // GATED n-gram tier (longest suffix wins)
            Ctx suf(ctx.end() - k, ctx.end());
            auto it = ngrams[k].find(suf);
            if (it != ngrams[k].end()) return Decision{it->second.out, "gated", it->second.basis, it->second.cite};
        }
        return std::nullopt;                                    // ABSTAIN
    }
};

// ---- the per-decision semiring decode: the C++ port of engine.dl, replacing the embedded Soufflé engine ----
// A fieldrun-distilled expert ships, per decision, candidate.facts (one candidate token id per line) + contrib.facts
// (block<ws>id<ws>weight). engine.dl is exactly: logit(T) = Σ contrib(_,T,w)  [⊗ = +, block contributions sum];
// decide = argmax_T logit(T)  [⊕ = max, the T=0 tropical decode]. The per-candidate logits feed a host-side softmax
// for OpenAI logprobs. This is that, in ~15 lines of pure arithmetic — no Soufflé, no model. Verified identical to the
// former embedded Soufflé engine (max |Δlogprob| ~3e-6 = the float32→float64 upgrade; src/engine.dl is the formal spec).
//
// Robustness contract (graceful by design — a thin runtime degrades to ABSTAIN, it does not abort):
//   * missing / empty candidate.facts → no candidates → decide=-1, empty logits → the distilled tier yields null
//     logprobs and the caller falls through to retrieval/abstain (the correct bounded behavior, not an error).
//   * a malformed contrib line (not "block<ws>int<ws>float") fails the stream extract and is skipped.
//   * a contrib for a non-candidate id is ignored (logit is defined only over candidates, per engine.dl).
struct FactsDecode {
    int decide = -1;                                 // argmax candidate (-1 if no candidates)
    std::vector<std::pair<int, double>> logits;      // (candidate id, Σ contrib) for every candidate
};

inline FactsDecode decode_facts(const std::string& facts_dir) {
    FactsDecode r;
    std::set<int> cand;                              // logit is defined only for candidates (engine.dl: logit(T):-candidate(T)…)
    {
        std::ifstream f(facts_dir + "/candidate.facts");
        std::string line;
        while (std::getline(f, line)) {
            std::istringstream ss(line);
            int id;
            if (ss >> id) cand.insert(id);
        }
    }
    std::map<int, double> sum;                       // candidate id -> Σ w
    {
        std::ifstream f(facts_dir + "/contrib.facts");
        std::string line;
        while (std::getline(f, line)) {              // "block\tid\tw" (souffle .facts) — ws-tolerant
            std::istringstream ss(line);
            std::string block;
            int id;
            double w;
            if (ss >> block >> id >> w && cand.count(id)) sum[id] += w;
        }
    }
    double best = -1e300;
    for (int id : cand) {                            // every candidate gets a logit (0 if it had no contrib)
        double s = sum.count(id) ? sum[id] : 0.0;
        r.logits.emplace_back(id, s);
        if (s > best) { best = s; r.decide = id; }
    }
    return r;
}
}  // namespace rosetta
#endif
