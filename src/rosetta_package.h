// rosetta_package.h — sgiandubh's consumer of a rosetta expert package (the convergence: rosetta builds, sgiandubh serves).
// Loads manifest.json (idiom_learn --package, or the flat abstain_emit manifest) and serves host-side, NO souffle, NO model:
//   tokenize(query) → TRUSTED idioms (frame-match) → GATED n-grams (longest suffix) → relation → induction → ABSTAIN.
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

struct Idiom {                                                   // TRUSTED tier (causal): gate, compose or induction
    std::string kind, cite;                                      // "gate" | "compose" | "induction"
    int id = -1;
    std::map<int, int> frame;                                    // offset (1 = last token) -> token
    int slot = 0;                                                // gate: content slot
    std::map<int, int> table;                                    // gate: content token -> out
    int k1 = 0, k2 = 0;                                          // compose: operand offsets
    std::map<int, int> valmap, sum;                             // compose: token->value, value-sum->out
    int L = 0;                                                   // induction: suffix match length
    std::vector<std::pair<int, int>> eq;                         // relation: eq-guard offset pairs
    int copy_off = 0;                                            // relation: copy offset
    double conf = -1.0;                                          // optional confidence (trusted kinds)
    int stratum = 1;                                             // labeled trust pool (fall-through)
    std::string origin;                                          // document | teacher | feedback
    std::map<int, double> confs;                                 // gate: per-key confidence (sw cover)
    std::map<int, std::pair<long, long>> counts;                 // gate: raw (cnt, tot), optional
    std::string feature, featureB;                               // dgate: derived-feature id(s)
    int lmax = 6;                                                // pointer: max match depth
    std::map<std::pair<int, int>, double> cells;                 // pointer: (l, lc) -> confidence
    std::map<std::pair<int, int>, int> dtable;                   // dgate: (feature, last) -> out
    std::map<std::pair<int, int>, double> dconfs;                // dgate: per-key confidence
    std::map<std::pair<int, int>, std::pair<long, long>> dcounts;// dgate: raw (cnt, tot), optional
};
struct Derived { std::string id, kind, of; std::set<int> openers, closers, members, quote_members, avoid, entity_members, value_members;
        int within = 7;
        std::vector<int> slot;
        std::set<int> entities, locations, objects, move_verbs, take_verbs, drop_verbs;
        std::string mode = "is"; int cap = 8, succ = 0, of_shift = 0; };
struct NGram { int out; std::string basis, cite; double det = -1.0, conf = -1.0; int stratum = 1; std::string origin; long cnt = -1, tot = -1; };
struct Decision { int answer; std::string tier, basis, citation; int rule = -1; double conf = -1.0; };

struct Package {
    std::vector<Idiom> idioms;                                   // TRUSTED, in priority order
    std::vector<std::map<Ctx, NGram>> ngrams;                    // index = suffix length; GATED
    int W = 0, n_rules = 0;
    std::string cover;                                           // "" = tiered; "support-weighted" = argmax-confidence
    int schema_version = -1;                                     // optional machine schema version
    double alpha = -1.0;                                         // optional confidence shrinkage
    struct CanonTemplate { std::string prefix; std::set<std::string> keywords; };
    std::vector<CanonTemplate> canon_templates;                  // mined template inventory
    std::map<std::string, std::string> canon_entities;           // lowercase -> proper form
    double strata_tau = 0.35;                                    // stratum-1 confidence floor
    std::string origin, origin_model, grounding_path;            // unified-spec provenance
    std::vector<Derived> derived;                                // two-layer: feature extractors
    std::map<int, int> cmap;                                     // concepts: member -> representative

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
        p.cover = m.value("cover", std::string(""));
        p.schema_version = m.value("schema_version", -1);
        p.alpha = m.value("alpha", -1.0);
        p.strata_tau = m.value("strata_tau", 0.35);
        p.origin = m.value("origin", std::string("teacher"));
        p.origin_model = m.value("origin_model", std::string(""));
        p.grounding_path = m.value("grounding", std::string(""));
        if (m.contains("canon")) {
            for (auto& e : m["canon"]["entities"]) {
                std::string en = e.get<std::string>(), lo = en;
                for (auto& c : lo) c = tolower(c);
                p.canon_entities[lo] = en;
            }
            for (auto& t : m["canon"]["templates"]) {
                CanonTemplate ct; ct.prefix = t["prefix"].get<std::string>();
                for (auto& k : t["keywords"]) {
                    std::string kw = k.get<std::string>();
                    while (!kw.empty() && kw.back() == 's') kw.pop_back();
                    if (!kw.empty()) ct.keywords.insert(kw);
                }
                p.canon_templates.push_back(ct);
            }
        }

        if (m.contains("concepts"))
            for (auto it = m["concepts"].begin(); it != m["concepts"].end(); ++it)
                for (auto& mm : it.value()) p.cmap[mm.get<int>()] = std::stoi(it.key());
        if (m.contains("derived"))
            for (auto& d : m["derived"]) {
                Derived dv; dv.id = d.value("id", std::string("")); dv.kind = d.value("kind", std::string(""));
                if (d.contains("openers")) for (auto& t : d["openers"]) dv.openers.insert(t.get<int>());
                if (d.contains("closers")) for (auto& t : d["closers"]) dv.closers.insert(t.get<int>());
                if (d.contains("members")) for (auto& t : d["members"]) dv.members.insert(t.get<int>());
                if (d.contains("quote_members")) for (auto& t : d["quote_members"]) dv.quote_members.insert(t.get<int>());
                dv.cap = d.value("cap", 8); dv.succ = d.value("succ", 0);
                if (d.contains("avoid")) for (auto& t : d["avoid"]) dv.avoid.insert(t.get<int>());
                if (d.contains("entity_members")) for (auto& t : d["entity_members"]) dv.entity_members.insert(t.get<int>());
                if (d.contains("value_members")) for (auto& t : d["value_members"]) dv.value_members.insert(t.get<int>());
                dv.within = d.value("within", 7);
                if (d.contains("slot")) for (auto& t : d["slot"]) dv.slot.push_back(t.get<int>());
                dv.mode = d.value("mode", std::string("is"));
                for (auto& [key, tgt] : std::map<std::string, std::set<int>*>{
                        {"entities", &dv.entities}, {"locations", &dv.locations},
                        {"objects", &dv.objects}, {"move_verbs", &dv.move_verbs},
                        {"take_verbs", &dv.take_verbs}, {"drop_verbs", &dv.drop_verbs}})
                    if (d.contains(key)) for (auto& t : d[key]) tgt->insert(t.get<int>());
                dv.of = d.value("of", std::string("")); dv.of_shift = d.value("of_shift", 0);
                p.derived.push_back(std::move(dv));
            }
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
                    p.ngrams[ctx.size()][ctx] = {r.at("out").get<int>(), r.value("basis", std::string("observational")),
                                                 cite, r.value("determinism", -1.0), r.value("confidence", -1.0),
                                                 r.value("stratum", 1), r.value("origin", p.origin), -1, -1};
                    if (r.contains("counts") && r["counts"].is_array()
                        && r["counts"].size() == 2) {
                        p.ngrams[ctx.size()][ctx].cnt = r["counts"][0].get<long>();
                        p.ngrams[ctx.size()][ctx].tot = r["counts"][1].get<long>();
                    }
                } else if (kind == "gate") {
                    Idiom id; id.kind = "gate"; id.id = r.value("id", -1); id.cite = cite;
                    id.frame = imap(r.at("frame")); id.slot = r.at("slot").get<int>(); id.table = imap(r.at("table"));
                    if (r.contains("confs"))
                        for (auto it = r["confs"].begin(); it != r["confs"].end(); ++it)
                            id.confs[std::stoi(it.key())] = it.value().get<double>();
                    if (r.contains("counts"))
                        for (auto it = r["counts"].begin(); it != r["counts"].end(); ++it)
                            id.counts[std::stoi(it.key())] = {it.value()[0].get<long>(), it.value()[1].get<long>()};
                    id.stratum = r.value("stratum", 1); id.origin = r.value("origin", p.origin); p.idioms.push_back(std::move(id));
                } else if (kind == "compose") {
                    Idiom id; id.kind = "compose"; id.id = r.value("id", -1); id.cite = cite;
                    id.frame = imap(r.at("frame")); id.k1 = r.at("operands").at(0); id.k2 = r.at("operands").at(1);
                    id.valmap = imap(r.at("valmap")); id.sum = imap(r.at("sum"));
                    id.stratum = r.value("stratum", 1); id.origin = r.value("origin", p.origin); p.idioms.push_back(std::move(id));
                } else if (kind == "induction") {               // causal COPY circuit, routed OOD after n-grams
                    Idiom id; id.kind = "induction"; id.id = r.value("id", -1); id.cite = cite;
                    id.conf = r.value("confidence", -1.0);
                    id.L = r.at("L").get<int>();
                    id.stratum = r.value("stratum", 1); id.origin = r.value("origin", p.origin); p.idioms.push_back(std::move(id));
                } else if (kind == "pointer") {                 // generalized copy: (l, lc)-cell scorer
                    Idiom id; id.kind = "pointer"; id.id = r.value("id", -1); id.cite = cite;
                    id.lmax = r.value("lmax", 6);
                    for (auto it = r["cells"].begin(); it != r["cells"].end(); ++it) {
                        auto k = it.key(); auto c = k.find(':');
                        id.cells[{std::stoi(k.substr(0, c)), std::stoi(k.substr(c + 1))}] = it.value().get<double>();
                    }
                    id.stratum = r.value("stratum", 1); id.origin = r.value("origin", p.origin); p.idioms.push_back(std::move(id));
                } else if (kind == "dgate2") {                  // PAIR gate: two features jointly
                    Idiom id; id.kind = "dgate2"; id.id = r.value("id", -1); id.cite = cite;
                    id.feature = r.at("featureA").get<std::string>();
                    id.featureB = r.at("featureB").get<std::string>();
                    for (auto it = r["table"].begin(); it != r["table"].end(); ++it) {
                        auto k = it.key(); auto c = k.find(':');
                        id.dtable[{std::stoi(k.substr(0, c)), std::stoi(k.substr(c + 1))}] = it.value().get<int>();
                    }
                    if (r.contains("confs"))
                        for (auto it = r["confs"].begin(); it != r["confs"].end(); ++it) {
                            auto k = it.key(); auto c = k.find(':');
                            id.dconfs[{std::stoi(k.substr(0, c)), std::stoi(k.substr(c + 1))}] = it.value().get<double>();
                        }
                    if (r.contains("counts"))
                        for (auto it = r["counts"].begin(); it != r["counts"].end(); ++it) {
                            auto k = it.key(); auto c = k.find(':');
                            id.dcounts[{std::stoi(k.substr(0, c)), std::stoi(k.substr(c + 1))}] =
                                {it.value()[0].get<long>(), it.value()[1].get<long>()};
                        }
                    id.stratum = r.value("stratum", 1); id.origin = r.value("origin", p.origin); p.idioms.push_back(std::move(id));
                } else if (kind == "dgate") {                   // TWO-LAYER: gate over a derived predicate
                    Idiom id; id.kind = "dgate"; id.id = r.value("id", -1); id.cite = cite;
                    id.feature = r.at("feature").get<std::string>();
                    for (auto it = r["table"].begin(); it != r["table"].end(); ++it) {
                        auto k = it.key(); auto c = k.find(':');
                        id.dtable[{std::stoi(k.substr(0, c)), std::stoi(k.substr(c + 1))}] = it.value().get<int>();
                    }
                    if (r.contains("confs"))
                        for (auto it = r["confs"].begin(); it != r["confs"].end(); ++it) {
                            auto k = it.key(); auto c = k.find(':');
                            id.dconfs[{std::stoi(k.substr(0, c)), std::stoi(k.substr(c + 1))}] = it.value().get<double>();
                        }
                    if (r.contains("counts"))
                        for (auto it = r["counts"].begin(); it != r["counts"].end(); ++it) {
                            auto k = it.key(); auto c = k.find(':');
                            id.dcounts[{std::stoi(k.substr(0, c)), std::stoi(k.substr(c + 1))}] =
                                {it.value()[0].get<long>(), it.value()[1].get<long>()};
                        }
                    id.stratum = r.value("stratum", 1); id.origin = r.value("origin", p.origin); p.idioms.push_back(std::move(id));
                } else if (kind == "relation") {                // causal EQ-GUARD + COPY, routed OOD
                    Idiom id; id.kind = "relation"; id.id = r.value("id", -1); id.cite = cite;
                    id.conf = r.value("confidence", -1.0);
                    for (auto& ij : r.at("eq")) id.eq.emplace_back(ij.at(0).get<int>(), ij.at(1).get<int>());
                    id.copy_off = r.at("copy").get<int>();
                    id.stratum = r.value("stratum", 1); id.origin = r.value("origin", p.origin); p.idioms.push_back(std::move(id));
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
            if (r.kind == "induction" || r.kind == "relation" || r.kind == "dgate"
                || r.kind == "pointer" || r.kind == "dgate2") continue;   // routed below
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
            if (it != ngrams[k].end()) return Decision{it->second.out, "gated", it->second.basis, it->second.cite,
                                                       -1, it->second.det};
        }
        // relation (causal EQ-GUARD + COPY), OOD fallback ABOVE succession/induction — the most specific routed
        // circuit: fires iff ctx[-i] == ctx[-j] for every eq pair (1-based from the end), then copies ctx[-copy].
        for (const auto& r : idioms) {
            if (r.kind != "relation") continue;
            int need = r.copy_off;
            bool ok = true;
            for (auto& ij : r.eq) { need = std::max({need, ij.first, ij.second}); }
            if (need > n) continue;
            for (auto& ij : r.eq) if (ctx[n - ij.first] != ctx[n - ij.second]) { ok = false; break; }
            if (ok) return Decision{ctx[n - r.copy_off], "trusted", "causal", r.cite, r.id, r.conf};
        }
        // induction (causal COPY), OOD fallback — reached ONLY after an n-gram miss (port of
        // serve_package.py): longest L first; find the previous occurrence of the current L-token
        // suffix; among several, copy the MOST RECENT occurrence's successor ([… A B … A] → B).
        std::vector<const Idiom*> inds;
        for (const auto& r : idioms) if (r.kind == "induction") inds.push_back(&r);
        std::sort(inds.begin(), inds.end(), [](const Idiom* a, const Idiom* b) { return a->L > b->L; });
        for (const Idiom* r : inds) {
            if (n <= r->L) continue;
            int bestj = -1;
            for (int j = 0; j + r->L < n; ++j)
                if (std::equal(ctx.end() - r->L, ctx.end(), ctx.begin() + j)) bestj = j;
            if (bestj >= 0)
                return Decision{ctx[bestj + r->L], "trusted", "causal", r->cite, r->id, r->conf};
        }
        return std::nullopt;                                    // ABSTAIN
    }

    // SUPPORT-WEIGHTED cover (manifest cover: "support-weighted"): every applicable rule fires and the
    // answer with the highest confidence wins -- the argmax policy whose dominance over every fixed
    // priority is kernel-checked (i-orca Arbitration.thy, argmax_policy_optimal), with calibration as
    // the stated premise. Ties keep the FIRST candidate: idioms in manifest order (the learner's
    // admitted order), then n-grams longest-first. Port of serve_package.py serve_sw.
    // map a free-phrased query onto the mined template inventory; empty = no parse (ABSTAIN).
    // returns {canonical_query, entity} -- the TRANSPARENCY fields the caller must surface.
    std::pair<std::string, std::string> canonicalize(const std::string& query) const {
        static const std::set<std::string> STOP = {"the", "of", "is", "a", "an", "in", "for",
            "as", "with", "what", "which", "does", "do", "where", "how", "many", "s", "its",
            "on", "table", "periodic"};
        std::vector<std::string> words;
        std::string cur;
        for (char c : query + " ") {
            if (isalpha((unsigned char)c)) cur += (char)tolower(c);
            else if (c == '\'') { words.push_back(cur); cur.clear(); }   // possessives split
            else if (!cur.empty()) { words.push_back(cur); cur.clear(); }
        }
        std::string ent, entlo;
        for (auto& w : words) if (canon_entities.count(w)) { entlo = w; ent = canon_entities.at(w); break; }
        if (ent.empty() || canon_templates.empty()) return {};
        {   // ALREADY canonical -> pass through unchanged
            std::string qn2 = query;
            while (!qn2.empty() && (qn2.back() == ' ' || qn2.back() == '.')) qn2.pop_back();
            for (auto& c : qn2) c = tolower(c);
            for (auto& t : canon_templates) {
                std::string pf = t.prefix;
                auto pos0 = pf.find("{E}");
                if (pos0 != std::string::npos) pf.replace(pos0, 3, ent);
                for (auto& c : pf) c = tolower(c);
                if (qn2 == pf) {
                    std::string orig = query;
                    while (!orig.empty() && (orig.back() == ' ' || orig.back() == '.')) orig.pop_back();
                    return {orig, ent};
                }
            }
        }
        std::set<std::string> qkw;
        for (auto w : words) {
            while (!w.empty() && w.back() == 's') w.pop_back();
            if (!w.empty() && !STOP.count(w) && w != entlo) qkw.insert(w);
        }
        const CanonTemplate* best = nullptr; double bestscore = 0;
        for (auto& t : canon_templates) {
            int hit = 0;
            for (auto& w : qkw) if (t.keywords.count(w)) hit++;
            double score = qkw.empty() ? 0 : (double)hit / (double)qkw.size();
            if (hit && score > bestscore) { best = &t; bestscore = score; }
        }
        if (!best || bestscore < 0.4) return {};
        std::string out = best->prefix;
        auto pos = out.find("{E}");
        if (pos != std::string::npos) out.replace(pos, 3, ent);
        return {out, ent};
    }

    std::optional<Decision> serve_sw(const Ctx& ctx) const {
        int n = (int)ctx.size();
        std::optional<Decision> best, best2;
        double bestc = -1e18, bestc2 = -1e18;
        std::map<std::string, int> feats, fpos;                 // derived predicates (PROVED extractors)
        for (const auto& d : derived) {
            if (d.kind == "bracket-mate") {
                std::vector<int> stack;                        // positions; "succ" composes the role
                for (int i = 0; i < n; i++) {
                    if (d.openers.count(ctx[i])) stack.push_back(i);
                    else if (d.closers.count(ctx[i]) && !stack.empty()) stack.pop_back();
                }
                fpos[d.id] = stack.empty() ? -1 : stack.back();
                int p2 = stack.empty() ? -1 : stack.back() + d.succ;
                feats[d.id] = (!stack.empty() && p2 >= 0 && p2 < n) ? ctx[p2] : -1;
            } else if (d.kind == "recent-member" || d.kind == "recent-unique") {
                std::map<int, int> cnt;
                if (d.kind == "recent-unique") for (int t : ctx) cnt[t]++;
                int pp = -1;
                for (int i = 0; i < n; i++)
                    if (d.members.count(ctx[i]) && (d.kind == "recent-member" || cnt[ctx[i]] == 1)) pp = i;
                fpos[d.id] = pp;
                int p2 = pp < 0 ? -1 : pp + d.succ;
                feats[d.id] = (pp >= 0 && p2 >= 0 && p2 < n) ? ctx[p2] : -1;
            } else if (d.kind == "bracket-depth") {
                int depth = 0;
                for (int t : ctx) depth += (int)d.openers.count(t) - (int)d.closers.count(t);
                feats[d.id] = std::min(std::max(depth, 0), d.cap);
            } else if (d.kind == "since-member") {             // DISCOURSE: tokens since last member
                int pp = -1;
                for (int i = 0; i < n; i++) if (d.members.count(ctx[i])) pp = i;
                feats[d.id] = pp >= 0 ? std::min(n - 1 - pp, d.cap) : d.cap + 1;
            } else if (d.kind == "dstate") {                   // RHETORICAL STATE (bucket x parity)
                int pp = -1, qc = 0;
                for (int i = 0; i < n; i++) {
                    if (d.members.count(ctx[i])) pp = i;
                    if (d.quote_members.count(ctx[i])) qc++;
                }
                int since = pp >= 0 ? std::min(n - 1 - pp, 33) : 34;
                int bucket = (since > 2) + (since > 6) + (since > 14) + (since > 32);
                feats[d.id] = bucket * 2 + qc % 2;
            } else if (d.kind == "member-parity") {            // DISCOURSE: quotation scope
                int c = 0;
                for (int t : ctx) c += (int)d.members.count(t);
                feats[d.id] = c % 2;
            } else if (d.kind == "estate2") {                // WORLD-STATE FOLD
                if (!d.slot.empty() && (n < 2 || ctx[n - 2] != d.slot[0] || ctx[n - 1] != d.slot[1])) {
                    feats[d.id] = -1; fpos[d.id] = -1; continue;
                }
                std::map<int, int> loc2, holder2, oloc2;
                std::map<int, std::vector<int>> ohist2;
                for (int i = 0; i + 1 < n; i++) {
                    int w = ctx[i], v = ctx[i + 1];
                    if (!d.entities.count(w)) continue;
                    if (d.move_verbs.count(v)) {
                        int dest = -1;
                        for (int j = i + 2; j < std::min(i + 7, n); j++)
                            if (d.locations.count(ctx[j])) { dest = ctx[j]; break; }
                        if (dest < 0) continue;
                        loc2[w] = dest;
                        for (auto& [o, h] : holder2) if (h == w && (!oloc2.count(o) || oloc2[o] != dest)) {
                            ohist2[o].push_back(oloc2.count(o) ? oloc2[o] : -1);
                            oloc2[o] = dest;
                        }
                    } else if (d.take_verbs.count(v) || d.drop_verbs.count(v)) {
                        int ob = -1;
                        for (int j = i + 2; j < std::min(i + 6, n); j++)
                            if (d.objects.count(ctx[j])) { ob = ctx[j]; break; }
                        if (ob < 0) continue;
                        if (d.take_verbs.count(v)) {
                            holder2[ob] = w;
                            if (loc2.count(w) && (!oloc2.count(ob) || oloc2[ob] != loc2[w])) {
                                ohist2[ob].push_back(oloc2.count(ob) ? oloc2[ob] : -1);
                                oloc2[ob] = loc2[w];
                            }
                        } else holder2.erase(ob);
                    }
                }
                int qo = -1, qi = -1;
                for (int i = n - 1; i >= 0; i--) if (d.objects.count(ctx[i])) { qo = ctx[i]; qi = i; break; }
                int feat = -1;
                if (qo >= 0) {
                    if (d.mode == "is") feat = oloc2.count(qo) ? oloc2[qo] : -1;
                    else {
                        int ref = -1;
                        for (int i = n - 1; i > qi; i--) if (d.locations.count(ctx[i])) { ref = ctx[i]; break; }
                        std::vector<int> hist = ohist2.count(qo) ? ohist2[qo] : std::vector<int>{};
                        hist.push_back(oloc2.count(qo) ? oloc2[qo] : -1);
                        for (int k = (int)hist.size() - 1; k > 0; k--)
                            if (hist[k] == ref && ref >= 0) { feat = hist[k - 1]; break; }
                    }
                }
                feats[d.id] = feat; fpos[d.id] = n - 1;
            } else if (d.kind == "estate") {                 // ENTITY-STATE REGISTER: last-
                if (!d.slot.empty() && (n < 2 || ctx[n - 2] != d.slot[0] || ctx[n - 1] != d.slot[1])) {
                    feats[d.id] = -1; fpos[d.id] = -1; continue;
                }
                int qi = -1;                                 // writer-wins (entity,value) fold
                for (int i = n - 1; i >= 0; i--) if (d.entity_members.count(ctx[i])) { qi = i; break; }
                int feat = -1;
                if (qi >= 0) {
                    for (int i = 0; i < qi; i++) {
                        if (ctx[i] != ctx[qi]) continue;
                        bool clean = true;
                        for (int j = i + 1; j <= i + 3 && j < n; j++)
                            if (d.avoid.count(ctx[j])) { clean = false; break; }
                        if (!clean) continue;
                        for (int j = i + 1; j <= i + d.within && j < n; j++)
                            if (d.value_members.count(ctx[j])) { feat = ctx[j]; break; }
                    }
                }
                feats[d.id] = feat; fpos[d.id] = qi;
            } else if (d.kind == "prev-occ") {                 // CHAINED role (entity echo with succ)
                auto bi = fpos.find(d.of);
                int bp = bi == fpos.end() ? -1 : bi->second, q = -1;
                if (bp >= 0 && d.of_shift) { bp += d.of_shift; if (bp < 0 || bp >= n) bp = -1; }
                if (bp >= 0) for (int i = 0; i < bp; i++) {
                    if (ctx[i] != ctx[bp]) continue;
                    bool ok2 = true;
                    for (int j = i + 1; j <= i + 3 && j < n; j++)
                        if (d.avoid.count(ctx[j])) { ok2 = false; break; }
                    if (ok2) q = i;
                }
                fpos[d.id] = q;
                int p2 = q < 0 ? -1 : q + d.succ;
                feats[d.id] = (q >= 0 && p2 >= 0 && p2 < n) ? ctx[p2] : -1;
            }
        }
        auto consider = [&](int ans, double c, Decision d, int stratum = 1) {
            if (stratum <= 1) { if (c > bestc) { d.answer = ans; d.conf = c; best = d; bestc = c; } }
            else if (c > bestc2) { d.answer = ans; d.conf = c; best2 = d; bestc2 = c; }
        };
        auto crep = [&](int t) { auto it = cmap.find(t); return it == cmap.end() ? t : it->second; };
        for (const auto& r : idioms) {
            if (r.kind == "pointer") {
                int bl = -1, blc = -1, bp = -1;
                for (int pp = 1; pp < n; pp++) {
                    int l = 0, lc = 0;
                    for (int j = 1; j <= r.lmax && pp - j >= 0; j++) {
                        int a = ctx[pp - j], b = ctx[n - j];
                        if (l == j - 1 && a == b) l = j;
                        if (lc == j - 1 && crep(a) == crep(b)) lc = j;
                        if (lc < j) break;
                    }
                    if ((l >= 1 || lc >= 2)
                        && std::make_tuple(l, lc, pp) >= std::make_tuple(bl, blc, bp)) {
                        bl = l; blc = lc; bp = pp;
                    }
                }
                if (bp >= 0) {
                    auto it = r.cells.find({bl, blc});
                    if (it != r.cells.end())
                        consider(ctx[bp], it->second, Decision{0, "trusted", "causal", r.cite, r.id}, r.stratum);
                }
            } else if (r.kind == "dgate") {
                auto fi = feats.find(r.feature);
                if (fi == feats.end() || fi->second < 0 || n < 1) continue;
                auto it = r.dtable.find({fi->second, ctx[n - 1]});
                if (it == r.dtable.end()) continue;
                auto ci = r.dconfs.find({fi->second, ctx[n - 1]});
                consider(it->second, ci == r.dconfs.end() ? 0.0 : ci->second,
                         Decision{0, "gated", "observational", r.cite, r.id}, r.stratum);
            } else if (r.kind == "gate") {
                bool ok = true;
                for (auto& [o, t] : r.frame) if (o > n || ctx[n - o] != t) { ok = false; break; }
                if (!ok || r.slot > n) continue;
                auto it = r.table.find(ctx[n - r.slot]);
                if (it == r.table.end()) continue;
                auto ci = r.confs.find(ctx[n - r.slot]);
                consider(it->second, ci == r.confs.end() ? 0.0 : ci->second,
                         Decision{0, "gated", "observational", r.cite, r.id}, r.stratum);
            } else if (r.kind == "relation") {
                int need = r.copy_off;
                bool ok = true;
                for (auto& ij : r.eq) need = std::max({need, ij.first, ij.second});
                if (need > n) continue;
                for (auto& ij : r.eq) if (ctx[n - ij.first] != ctx[n - ij.second]) { ok = false; break; }
                if (ok) consider(ctx[n - r.copy_off], r.conf < 0 ? 0.0 : r.conf,
                                 Decision{0, "trusted", "causal", r.cite, r.id}, r.stratum);
            } else if (r.kind == "induction") {
                int L = r.L;
                if (n <= L) continue;
                int bestj = -1;
                for (int j = 0; j + L < n; j++) {
                    bool m2 = true;
                    for (int q = 0; q < L; q++) if (ctx[j + q] != ctx[n - L + q]) { m2 = false; break; }
                    if (m2) bestj = j;
                }
                if (bestj >= 0)
                    consider(ctx[bestj + L], r.conf < 0 ? 0.0 : r.conf,
                             Decision{0, "trusted", "causal", r.cite, r.id}, r.stratum);
            }
        }
        for (int k = std::min(n, W); k >= 1; k--) {
            Ctx suffix(ctx.end() - k, ctx.end());
            auto it = ngrams[k].find(suffix);
            if (it != ngrams[k].end())
                consider(it->second.out, it->second.conf < 0 ? 0.0 : it->second.conf,
                         Decision{0, "gated", it->second.basis, it->second.cite, -1},
                         it->second.stratum);
        }
        if (best && bestc >= strata_tau) return best;
        if (best2 && bestc2 > bestc) return best2;
        return best;
    }

    // dispatch on the manifest's declared cover semantics
    std::optional<Decision> decide(const Ctx& ctx) const {
        return cover == "support-weighted" ? serve_sw(ctx) : serve(ctx);
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
