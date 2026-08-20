// UD parse -> component tree, driven by a glossa GRAMMAR PACK rather than by
// hardcoded German rules.
//
// component_transform.h is the German transform this generalizes; it stays as it
// is, because it is what production serves. This header is the multilingual
// path: the language arrives as data (packs/<lang>.json — labels, lexicons and
// the ordered leaf/group/clause/predicate rule tables), so one implementation
// serves German, Spanish and English.
//
// The normative spec is glossa's dl/<lang>.dl; this port and glossa's Python
// engine are implementations of it. test/pack_transform_gate.cpp holds this one
// to glossa's frozen fixtures — the same trees the Python engine is gated on.
#pragma once
#include <algorithm>
#include <cctype>
#include <fstream>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "json.hpp"

namespace packtrans {

using json = nlohmann::json;

struct Tok {
    std::string word, lower, pos, deprel, base, cas, gnn, leaf;
    int head = 0;
    bool relclause = false;
    std::string case_ov;
};

// ---------------------------------------------------------------- the pack
struct Pack {
    json j;
    std::map<std::string, std::set<std::string>> cache;   // resolved rule operands

    bool load(const std::string& path) {
        std::ifstream f(path);
        if (!f) return false;
        f >> j;
        return true;
    }

    const std::string& label(const std::string& role) const {
        static const std::string unknown = "?";
        auto it = j["labels"].find(role);
        return it == j["labels"].end() ? unknown : it->get_ref<const std::string&>();
    }

    // A rule operand is either an inline list or "@lexicon"; both become a set.
    const std::set<std::string>& values(const json& spec) {
        std::string key = spec.dump();
        auto it = cache.find(key);
        if (it != cache.end()) return it->second;
        std::set<std::string> out;
        if (spec.is_string()) {
            std::string name = spec.get<std::string>().substr(1);       // strip '@'
            const json& lex = j["lexicons"][name];
            if (lex.is_object())
                for (auto& [k, _v] : lex.items()) out.insert(k);
            else
                for (auto& w : lex) out.insert(w.get<std::string>());
        } else {
            for (auto& w : spec) out.insert(w.get<std::string>());
        }
        return cache.emplace(key, std::move(out)).first->second;
    }

    const json& section(const char* name) const {
        static const json empty = json::object();
        auto it = j.find(name);
        return it == j.end() ? empty : *it;
    }

    std::set<std::string> feature_set(const char* name) {
        std::set<std::string> out;
        const json& f = j["features"];
        if (f.contains(name))
            for (auto& w : f[name]) out.insert(w.get<std::string>());
        return out;
    }
};

// ---------------------------------------------------------------- matching
struct Ctx {
    Pack* P;
    std::vector<Tok>* toks;
    std::map<int, std::vector<int>>* by_head;
};

// Lexicon lookups are on the lowercased form, so lowercasing must match Python's
// str.lower() for the letters these languages use: ASCII plus Latin-1 Supplement
// (ÄÖÜ, ÁÉÍÓÚÑ) and Latin Extended-A. A byte-wise tolower() would leave every
// accented word unmatched by its register entry.
inline std::string utf8_lower(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size();) {
        unsigned char c = (unsigned char)s[i];
        if (c < 0x80) {
            out += (char)std::tolower(c);
            i += 1;
        } else if ((c == 0xC3) && i + 1 < s.size()) {           // U+00C0..U+00FF
            unsigned char d = (unsigned char)s[i + 1];
            if (d >= 0x80 && d <= 0x9E && d != 0x97) d += 0x20;  // × (U+00D7) is not a letter
            out += (char)c;
            out += (char)d;
            i += 2;
        } else if ((c == 0xC4 || c == 0xC5) && i + 1 < s.size()) {   // Latin Extended-A
            unsigned char d = (unsigned char)s[i + 1];
            unsigned int cp = ((c & 0x1Fu) << 6) | (d & 0x3Fu);
            if (cp >= 0x100 && cp <= 0x17F && (cp % 2 == 0)) cp += 1;   // even = upper
            out += (char)(0xC0 | (cp >> 6));
            out += (char)(0x80 | (cp & 0x3F));
            i += 2;
        } else {
            out += s[i];
            i += 1;
        }
    }
    return out;
}

inline bool ends_with(const std::string& s, const std::string& suf) {
    return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

bool tok_match(json& cond, int i, Ctx& C);

inline bool child_match(json& cond, int i, Ctx& C) {
    auto it = C.by_head->find(i);
    if (it == C.by_head->end()) return false;
    for (int k : it->second)
        if (tok_match(cond, k, C)) return true;
    return false;
}

inline bool tok_match(json& cond, int i, Ctx& C) {
    const Tok& t = (*C.toks)[i - 1];
    for (auto& [atom, spec] : cond.items()) {
        if (atom == "has_child") {
            if (!child_match(spec, i, C)) return false;
        } else if (atom == "no_child") {
            if (child_match(spec, i, C)) return false;
        } else if (atom == "pos_in") {
            if (!C.P->values(spec).count(t.pos)) return false;
        } else if (atom == "pos_not_in") {
            if (C.P->values(spec).count(t.pos)) return false;
        } else if (atom == "base_in") {
            if (!C.P->values(spec).count(t.base)) return false;
        } else if (atom == "base_not_in") {
            if (C.P->values(spec).count(t.base)) return false;
        } else if (atom == "deprel_in") {
            if (!C.P->values(spec).count(t.deprel)) return false;
        } else if (atom == "deprel_not_in") {
            if (C.P->values(spec).count(t.deprel)) return false;
        } else if (atom == "lower_in") {
            if (!C.P->values(spec).count(t.lower)) return false;
        } else if (atom == "lower_not_in") {
            if (C.P->values(spec).count(t.lower)) return false;
        } else if (atom == "lower_prefix_in") {
            bool hit = false;
            for (const auto& p : C.P->values(spec))
                if (t.lower.rfind(p, 0) == 0) { hit = true; break; }
            if (!hit) return false;
        } else if (atom == "lower_suffix_in") {
            bool hit = false;
            for (const auto& p : C.P->values(spec))
                if (ends_with(t.lower, p)) { hit = true; break; }
            if (!hit) return false;
        } else if (atom == "lower_suffix_not_in") {
            for (const auto& p : C.P->values(spec))
                if (ends_with(t.lower, p)) return false;
        } else if (atom == "flag_in") {
            if (!t.relclause) return false;                 // the only flag the engine sets
        } else if (atom == "flag_not_in") {
            if (t.relclause) return false;
        } else if (atom == "head_pos_in") {
            if (!t.head || !C.P->values(spec).count((*C.toks)[t.head - 1].pos)) return false;
        } else if (atom == "head_pos_not_in") {
            if (t.head && C.P->values(spec).count((*C.toks)[t.head - 1].pos)) return false;
        } else if (atom == "head_lower_in") {
            if (!t.head || !C.P->values(spec).count((*C.toks)[t.head - 1].lower)) return false;
        } else if (atom == "head_leaf_in") {
            if (!t.head || !C.P->values(spec).count((*C.toks)[t.head - 1].leaf)) return false;
        } else if (atom == "leaf_in") {
            if (!C.P->values(spec).count(t.leaf)) return false;
        } else if (atom == "leaf_not_in") {
            if (C.P->values(spec).count(t.leaf)) return false;
        } else {
            return false;                                   // unknown atom: never matches
        }
    }
    return true;
}

// ---------------------------------------------------------------- transform
class Transform {
  public:
    Transform(Pack& pack) : P(pack) {}

    json run(const std::vector<Tok>& input) {
        toks = input;
        by_head.clear();
        n = (int)toks.size();
        for (int i = 1; i <= n; i++) by_head[toks[i - 1].head].push_back(i);
        C = Ctx{&P, &toks, &by_head};

        const json& cl = P.section("clause");
        intro_bases = as_set(cl, "intro_bases");
        head_pos = as_set(cl, "head_pos");
        clausal_kid = as_set(cl, "clausal_kid_bases");
        conj_free = as_set(cl, "conj_unconditional_pos");
        rel_base = cl.value("rel_base", "acl");
        forbid_rel_head = P.feature_set("rel_clause_forbidden_head_pos");
        sent_final = P.feature_set("sentence_final_punct");
        case_display.clear();
        if (P.j["features"].contains("case_display"))
            for (auto& [k, v] : P.j["features"]["case_display"].items())
                case_display[k] = v.get<std::string>();
        // a clitic that is part of the verb form carries no case FUNCTION
        case_exclude = P.feature_set("case_display_exclude_deprels");

        mark_relative_clauses();
        apply_contracted_case();
        for (int i = 1; i <= n; i++) toks[i - 1].leaf = leaf_role(i);
        segment();
        return assemble();
    }

  private:
    Pack& P;
    Ctx C{};
    std::vector<Tok> toks;
    std::map<int, std::vector<int>> by_head;
    int n = 0;
    std::set<std::string> intro_bases, head_pos, clausal_kid, conj_free,
        forbid_rel_head, sent_final;
    std::string rel_base;
    std::map<std::string, std::string> case_display;
    std::set<std::string> case_exclude;      // deprels whose leaves show no case
    std::map<int, int> clause_of;
    std::map<int, std::vector<int>> clauses;
    std::set<int> sentence_punct;
    struct Group { int first; std::string role; std::vector<int> members; int anchor = 0; };
    std::map<int, std::vector<Group>> groups;      // per clause head
    std::map<int, std::vector<int>> preds;
    std::set<int> lone;
    std::map<int, json*> group_nodes;              // anchor -> emitted node (nesting target)

    static std::set<std::string> as_set(const json& obj, const char* key) {
        std::set<std::string> out;
        if (obj.contains(key))
            for (auto& w : obj[key]) out.insert(w.get<std::string>());
        return out;
    }

    bool clausal_conj(int i) const {
        if (conj_free.count(toks[i - 1].pos)) return true;
        auto it = by_head.find(i);
        if (it == by_head.end()) return false;
        for (int k : it->second)
            if (clausal_kid.count(toks[k - 1].base)) return true;
        return false;
    }

    bool introduces_clause(int i) const {
        const Tok& t = toks[i - 1];
        return t.head != 0 && intro_bases.count(t.base) && head_pos.count(t.pos) &&
               (t.base != "conj" || clausal_conj(i));
    }

    std::vector<int> subtree(int i) const {
        std::vector<int> out{i};
        auto it = by_head.find(i);
        if (it != by_head.end())
            for (int c : it->second) {
                auto s = subtree(c);
                out.insert(out.end(), s.begin(), s.end());
            }
        std::sort(out.begin(), out.end());
        return out;
    }

    void mark_relative_clauses() {
        if (!P.j["features"].value("rel_pronoun_marking", false)) return;
        std::set<std::string> rel_pos = P.feature_set("rel_pronoun_pos");
        if (rel_pos.empty()) rel_pos.insert("PRON");
        std::set<int> rel_heads, bound;
        for (int i = 1; i <= n; i++) {
            const Tok& t = toks[i - 1];
            if (t.base == rel_base &&
                !(t.head && forbid_rel_head.count(toks[t.head - 1].pos)))
                rel_heads.insert(i);
            if (t.base != rel_base && introduces_clause(i)) bound.insert(i);
        }
        for (int i = 1; i <= n; i++) {
            if (!rel_pos.count(toks[i - 1].pos)) continue;
            int h = i;
            std::set<int> seen;
            while (toks[h - 1].head != 0 && !seen.count(h)) {
                seen.insert(h);
                if (rel_heads.count(h)) { toks[i - 1].relclause = true; break; }
                if (bound.count(h) && h != i) break;        // bounded to its own clause
                h = toks[h - 1].head;
            }
        }
    }

    void apply_contracted_case() {
        if (!P.j["features"].value("contracted_case_override", false) || case_display.empty())
            return;
        const json& tab = P.j["lexicons"]["contracted"];
        for (int i = 1; i <= n; i++) {
            Tok& t = toks[i - 1];
            if (!case_display.count(t.cas)) continue;
            auto it = by_head.find(i);
            if (it == by_head.end()) continue;
            for (int k : it->second) {
                const Tok& c = toks[k - 1];
                if (c.pos == "ADP" && c.base == "case" && tab.contains(c.lower)) {
                    t.case_ov = tab[c.lower].get<std::string>();
                    break;
                }
            }
        }
    }

    std::string leaf_role(int i) {
        for (auto& rule : P.j["leaf_rules"]) {
            json cond = rule.value("when", json::object());
            if (tok_match(cond, i, C)) return rule["then"].get<std::string>();
        }
        return "WORD";
    }

    json leaf_node(int i) {
        const Tok& t = toks[i - 1];
        json node = {{"word", t.word}, {"component", P.label(t.leaf)}, {"i", i}};
        if (case_display.count(t.cas) && !case_exclude.count(t.deprel))
            node["case"] = case_display.at(t.case_ov.empty() ? t.cas : t.case_ov);
        return node;
    }

    std::string span(const std::vector<int>& idx) {
        std::string out;
        for (size_t k = 0; k < idx.size(); k++) {
            if (k) out += " ";
            out += toks[idx[k] - 1].word;
        }
        return out;
    }

    void assign_clause(int i, int own) {
        clause_of[i] = own;
        auto it = by_head.find(i);
        if (it == by_head.end()) return;
        for (int c : it->second) assign_clause(c, introduces_clause(c) ? c : own);
    }

    void segment() {
        std::vector<int> roots;
        for (int i = 1; i <= n; i++)
            if (toks[i - 1].head == 0) roots.push_back(i);
        if (roots.empty()) roots.push_back(1);
        assign_clause(roots[0], roots[0]);
        for (size_t k = 1; k < roots.size(); k++) assign_clause(roots[k], roots[k]);

        std::set<int> heads;
        for (int i = 1; i <= n; i++) heads.insert(clause_of[i]);
        if (heads.size() > 1 && n) {
            const Tok& last = toks[n - 1];
            if (last.pos == "PUNCT" && sent_final.count(last.word) && clause_of[n] != n) {
                sentence_punct.insert(n);
                clause_of[n] = 0;
            }
        }
        for (int i = 1; i <= n; i++)
            if (clause_of[i] != 0) clauses[clause_of[i]].push_back(i);
    }

    // ---- per clause -----------------------------------------------------
    void build_clause(int ch) {
        const json& PR = P.section("predicate");
        std::set<std::string> phead = as_set(PR, "head_pos"), chain_b = as_set(PR, "chain_bases"),
                              chain_p = as_set(PR, "chain_pos"), mpos = as_set(PR, "member_pos"),
                              mdep = as_set(PR, "member_deprel");
        std::set<int> members(clauses[ch].begin(), clauses[ch].end());
        bool head_verbal = phead.count(toks[ch - 1].pos) > 0;
        std::set<int> used;

        std::vector<int> chain{ch}, frontier{ch};
        while (!frontier.empty()) {
            int v = frontier.back();
            frontier.pop_back();
            auto it = by_head.find(v);
            if (it == by_head.end()) continue;
            for (int c : it->second)
                if (members.count(c) && chain_b.count(toks[c - 1].base) &&
                    chain_p.count(toks[c - 1].pos)) {
                    chain.push_back(c);
                    frontier.push_back(c);
                }
        }
        std::vector<int> kids;
        for (int v : chain) {
            auto it = by_head.find(v);
            if (it == by_head.end()) continue;
            for (int c : it->second)
                if (members.count(c) &&
                    std::find(chain.begin(), chain.end(), c) == chain.end())
                    kids.push_back(c);
        }

        std::set<int> pred;
        if (head_verbal) pred.insert(ch);
        for (int k : kids) {
            if (!head_verbal)
                for (auto& r : PR["nonverbal_head_rules"]) {
                    json cond = r;
                    if (tok_match(cond, k, C)) { pred.insert(k); break; }
                }
            for (auto& r : PR["member_rules"]) {
                json cond = r;
                if (tok_match(cond, k, C)) { pred.insert(k); break; }
            }
        }
        for (int v : chain)
            if (v != ch || head_verbal) pred.insert(v);
        std::vector<int> pred_members;
        for (int p : pred)
            if (mpos.count(toks[p - 1].pos) || mdep.count(toks[p - 1].deprel))
                pred_members.push_back(p);
        std::sort(pred_members.begin(), pred_members.end());
        preds[ch] = pred_members;

        auto take = [&](std::vector<int> idx, const std::string& role, int anchor) {
            std::vector<int> keep;
            for (int j : idx)
                if (members.count(j) && !used.count(j)) keep.push_back(j);
            if (keep.empty()) return;
            std::sort(keep.begin(), keep.end());
            used.insert(keep.begin(), keep.end());
            groups[ch].push_back({keep.front(), role, keep, anchor});
        };

        for (int k : kids) {
            std::vector<int> st;
            for (int j : subtree(k))
                if (clause_of[j] == ch) st.push_back(j);
            for (auto& rule : P.j["groups"]) {
                json cond = rule.value("when", json::object());
                if (!group_match(cond, k, st)) continue;
                const json& act = rule["then"];
                if (act.value("skip", false)) break;
                if (act.contains("group")) {
                    take(st, act["group"].get<std::string>(), k);
                    break;
                }
                int j = st.front();                          // leaf action: singleton collapse
                if (act.contains("leaf_in")) {
                    json in = act["leaf_in"];
                    if (P.values(in).count(toks[j - 1].leaf))
                        toks[j - 1].leaf = act["leaf"].get<std::string>();
                }
                used.insert(j);
                lone.insert(j);
                groups[ch].push_back({j, "", {j}, 0});       // empty role = a flat leaf
                break;
            }
        }
        take(pred_members, "PREDICATE", 0);

        const json& PV = P.section("predicative");
        if (PV.value("enabled", false) &&
            !as_set(PV, "exclude_head_pos").count(toks[ch - 1].pos)) {
            std::set<std::string> bb = as_set(PV, "blocklist_bases"), bp = as_set(PV, "blocklist_pos");
            std::set<int> own;
            for (int j : subtree(ch)) own.insert(j);
            std::set<int> cand;
            for (int j : members)
                if (!used.count(j) && own.count(j)) cand.insert(j);
            std::set<int> run{ch};
            for (int j = ch + 1; cand.count(j); j++) run.insert(j);
            for (int j = ch - 1; cand.count(j); j--) run.insert(j);
            std::vector<int> kept;
            for (int j : run)
                if (!bb.count(toks[j - 1].base) && !bp.count(toks[j - 1].pos)) kept.push_back(j);
            std::sort(kept.begin(), kept.end());
            if (!kept.empty()) {
                std::vector<int> take_idx;
                for (int j : run)
                    if (j >= kept.front() && j <= kept.back()) take_idx.push_back(j);
                std::sort(take_idx.begin(), take_idx.end());
                take(take_idx, PV.value("label", std::string("PREDICATIVE")), 0);
            }
        }
        for (int j : members)
            if (!used.count(j)) groups[ch].push_back({j, "", {j}, 0});
        std::sort(groups[ch].begin(), groups[ch].end(),
                  [](const Group& a, const Group& b) { return a.first < b.first; });
    }

    bool group_match(json& cond, int k, const std::vector<int>& st) {
        json tokenish = json::object();
        for (auto& [atom, spec] : cond.items())
            if (atom != "singleton" && atom != "subtree_has_pos" && atom != "subtree_lacks_pos" &&
                atom != "subtree_has_case" && atom != "subtree_has_lower" &&
                atom != "token_case_in" && atom != "governed_pp")
                tokenish[atom] = spec;
        if (!tok_match(tokenish, k, C)) return false;
        if (cond.contains("singleton") && (st.size() == 1) != cond["singleton"].get<bool>())
            return false;
        if (cond.contains("subtree_has_pos")) {
            const auto& want = P.values(cond["subtree_has_pos"]);
            bool hit = false;
            for (int j : st) if (want.count(toks[j - 1].pos)) { hit = true; break; }
            if (!hit) return false;
        }
        if (cond.contains("subtree_lacks_pos")) {
            const auto& bad = P.values(cond["subtree_lacks_pos"]);
            for (int j : st) if (bad.count(toks[j - 1].pos)) return false;
        }
        if (cond.contains("subtree_has_case")) {
            const auto& want = P.values(cond["subtree_has_case"]);
            bool hit = false;
            for (int j : st) if (want.count(toks[j - 1].cas)) { hit = true; break; }
            if (!hit) return false;
        }
        if (cond.contains("subtree_has_lower")) {
            const auto& want = P.values(cond["subtree_has_lower"]);
            bool hit = false;
            for (int j : st) if (want.count(toks[j - 1].lower)) { hit = true; break; }
            if (!hit) return false;
        }
        if (cond.contains("token_case_in") && !P.values(cond["token_case_in"]).count(toks[k - 1].cas))
            return false;
        if (cond.contains("governed_pp")) {
            std::string prep;
            for (int j : st)
                if (toks[j - 1].pos == "ADP") { prep = toks[j - 1].lower; break; }
            std::string lemma;
            const json& f2l = P.j["lexicons"][P.j["features"].value("lemma_lexicon",
                                                                   std::string("form2lemma"))];
            if (toks[k - 1].head) {
                const std::string& hw = toks[toks[k - 1].head - 1].lower;
                if (f2l.contains(hw)) lemma = f2l[hw].get<std::string>();
            }
            bool hit = false;
            if (!prep.empty() && !lemma.empty()) {
                const json& gov = P.j["lexicons"][P.j["features"].value("governed_lexicon",
                                                                       std::string("governed"))];
                std::string key = lemma + "|" + prep;
                for (auto& g : gov)
                    if (g.get<std::string>() == key) { hit = true; break; }
            }
            if (hit != cond["governed_pp"].get<bool>()) return false;
        }
        return true;
    }

    std::string clause_role(int ch) {
        const json& cl = P.section("clause");
        const Tok& t = toks[ch - 1];
        const json& labels = cl["labels"];
        if (t.head == 0)
            return labels.value("root", std::string("MAIN_CLAUSE"));
        if (t.base == rel_base) {
            if (forbid_rel_head.count(toks[t.head - 1].pos))
                return cl.value("acl_fallback_label", std::string("MAIN_CLAUSE"));
            return cl.value("acl_label", std::string("REL_CLAUSE"));
        }
        if (labels.contains(t.base)) return labels[t.base].get<std::string>();
        return cl.value("default_label", std::string("SUB_CLAUSE"));
    }

    // ---- assembly (ordering, spans, nesting) ----------------------------
    std::vector<int> order_and_span(json& node) {
        if (!node.contains("children")) return {node["i"].get<int>()};
        std::vector<std::pair<std::vector<int>, json>> keyed;
        for (auto& c : node["children"]) {
            json child = c;
            auto idx = order_and_span(child);
            if (!idx.empty()) keyed.emplace_back(idx, child);
        }
        std::stable_sort(keyed.begin(), keyed.end(),
                         [](const auto& a, const auto& b) { return a.first[0] < b.first[0]; });
        json kids = json::array();
        std::vector<int> all;
        for (auto& [idx, child] : keyed) {
            kids.push_back(child);
            all.insert(all.end(), idx.begin(), idx.end());
        }
        std::sort(all.begin(), all.end());
        node["children"] = kids;
        node["word"] = span(all);
        return all;
    }

    static bool drop_leaf(json& node, int j) {
        if (!node.contains("children")) return false;
        auto& kids = node["children"];
        for (size_t k = 0; k < kids.size(); k++) {
            if (!kids[k].contains("children")) {
                if (kids[k].value("i", -1) == j) {
                    kids.erase(kids.begin() + k);
                    return true;
                }
            } else if (drop_leaf(kids[k], j)) {
                return true;
            }
        }
        return false;
    }

    json assemble() {
        std::vector<int> order;
        for (auto& [ch, mem] : clauses) order.push_back(ch);
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            return *std::min_element(clauses[a].begin(), clauses[a].end()) <
                   *std::min_element(clauses[b].begin(), clauses[b].end());
        });
        for (int ch : order) build_clause(ch);

        std::map<int, json> nodes;
        std::map<int, std::pair<int, int>> anchor_of;    // anchor -> (clause, group index)
        for (int ch : order) {
            json node = {{"word", ""}, {"component", nullptr}, {"children", json::array()}};
            int gi = 0;
            for (auto& g : groups[ch]) {
                if (g.role.empty()) {
                    node["children"].push_back(leaf_node(g.members.front()));
                } else {
                    json gnode = {{"word", span(g.members)}, {"component", P.label(g.role)},
                                  {"children", json::array()}};
                    for (int j : g.members) gnode["children"].push_back(leaf_node(j));
                    node["children"].push_back(gnode);
                    if (g.anchor) anchor_of[g.anchor] = {ch, gi};
                }
                gi++;
            }
            nodes[ch] = node;
        }

        if (order.size() == 1) {
            json root = nodes[order[0]];
            root["component"] = P.label("SENTENCE");
            order_and_span(root);
            return root;
        }
        std::map<int, std::string> role;
        for (int ch : order) {
            role[ch] = clause_role(ch);
            nodes[ch]["component"] = P.label(role[ch]);
        }

        // nest exactly the clauses that INTERRUPT their matrix
        const json& labels = P.section("clause")["labels"];
        std::string main_role = labels.value("root", std::string("MAIN_CLAUSE"));
        std::map<int, std::pair<int, int>> nest_group;    // clause -> (matrix, group index)
        std::map<int, int> nest_clause, adopt;
        for (int ch : order) {
            if (role[ch] == main_role) continue;
            int h = toks[ch - 1].head;
            int m = h ? clause_of[h] : 0;
            if (!m || m == ch || !clauses.count(m)) continue;
            int lo = *std::min_element(clauses[ch].begin(), clauses[ch].end());
            int hi = *std::max_element(clauses[ch].begin(), clauses[ch].end());
            bool left = false, right = false;
            for (int x : clauses[m]) {
                if (x < lo) left = true;
                if (x > hi) right = true;
            }
            if (!left || !right) continue;
            bool placed = false;
            if (toks[ch - 1].base == rel_base) {
                for (auto& g : groups[m])
                    if (g.anchor && anchor_of.count(g.anchor) &&
                        std::find(g.members.begin(), g.members.end(), h) != g.members.end()) {
                        nest_group[ch] = anchor_of[g.anchor];
                        placed = true;
                        break;
                    }
            }
            if (!placed) nest_clause[ch] = m;
            int j = hi + 1;
            if (j <= n && toks[j - 1].word == "," && toks[j - 1].pos == "PUNCT" &&
                clause_of.count(j) && clause_of[j] == m)
                adopt[ch] = j;
        }
        for (auto& [ch, j] : adopt)
            if (drop_leaf(nodes[clause_of[j]], j)) nodes[ch]["children"].push_back(leaf_node(j));

        // Nesting chains (a relative clause inside a clause that is itself nested)
        // must be applied INNERMOST FIRST: a node is copied into its parent here,
        // so anything placed into it afterwards would be dropped. Python holds the
        // nodes by reference and does not have to care; this does.
        std::map<int, std::vector<int>> kids_of;
        for (auto& [ch, target] : nest_group) kids_of[target.first].push_back(ch);
        for (auto& [ch, m] : nest_clause)
            if (!nest_group.count(ch)) kids_of[m].push_back(ch);
        std::set<int> nested;
        std::function<void(int)> place_into = [&](int parent) {
            for (int ch : kids_of[parent]) {
                place_into(ch);                     // assemble the child completely first
                if (nest_group.count(ch)) {
                    auto [m, gi] = nest_group[ch];
                    nodes[m]["children"][gi]["children"].push_back(nodes[ch]);
                } else {
                    nodes[nest_clause[ch]]["children"].push_back(nodes[ch]);
                }
                nested.insert(ch);
            }
        };
        for (int ch : order)
            if (!nest_group.count(ch) && !nest_clause.count(ch)) place_into(ch);

        json root = {{"word", ""}, {"component", P.label("SENTENCE")}, {"children", json::array()}};
        for (int ch : order)
            if (!nested.count(ch)) root["children"].push_back(nodes[ch]);
        for (int j : sentence_punct) root["children"].push_back(leaf_node(j));
        order_and_span(root);
        return root;
    }
};

inline std::vector<Tok> tokens_from_json(const json& arr) {
    std::vector<Tok> out;
    for (const auto& t : arr) {
        Tok tk;
        tk.word = t.value("word", "");
        tk.lower = utf8_lower(tk.word);
        tk.pos = t.value("pos", "X");
        tk.deprel = t.value("deprel", "dep");
        tk.base = tk.deprel.substr(0, tk.deprel.find(':'));
        tk.head = t.value("head", 0);
        tk.cas = t.value("case", "-");
        tk.gnn = t.value("gnn", "-|-");
        out.push_back(tk);
    }
    return out;
}

}  // namespace packtrans
