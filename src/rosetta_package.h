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
        json m; f >> m;
        Package p;
        int maxlen = 0;
        for (auto& r : m["rules"])
            if (r.value("kind", std::string("ngram")) == "ngram")
                maxlen = std::max(maxlen, (int)r["ctx"].size());
        p.W = maxlen;
        p.ngrams.assign(p.W + 1, {});
        for (auto& r : m["rules"]) {
            std::string kind = r.value("kind", std::string("ngram"));
            std::string cite = cite_of(r);
            if (kind == "ngram") {
                Ctx ctx = r["ctx"].get<Ctx>();
                p.ngrams[ctx.size()][ctx] = {r["out"].get<int>(), r.value("basis", std::string("observational")), cite};
            } else if (kind == "gate") {
                Idiom id; id.kind = "gate"; id.id = r.value("id", -1); id.cite = cite;
                id.frame = imap(r["frame"]); id.slot = r["slot"].get<int>(); id.table = imap(r["table"]);
                p.idioms.push_back(std::move(id));
            } else if (kind == "compose") {
                Idiom id; id.kind = "compose"; id.id = r.value("id", -1); id.cite = cite;
                id.frame = imap(r["frame"]); id.k1 = r["operands"][0]; id.k2 = r["operands"][1];
                id.valmap = imap(r["valmap"]); id.sum = imap(r["sum"]);
                p.idioms.push_back(std::move(id));
            }
        }
        p.n_rules = (int)(p.idioms.size());
        for (auto& tab : p.ngrams) p.n_rules += (int)tab.size();
        return p;
    }

    // The runtime decision: trusted idioms (frame-matched, in order) -> gated n-grams (longest suffix) -> ABSTAIN.
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
}  // namespace rosetta
#endif
