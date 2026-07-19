// UD parse -> satzklar component tree + register-backed error flags, in C++.
// Faithful port of germandata's certified layer: transform/dl/transform.dl + errors.dl are the
// normative specs; this port is differentially gated against them via the python twin
// (germandata transform/cpp_gate.py — trees must be identical, error sets identical).
// All grammar data (word lists, paradigms, mined lexicons) comes from the package's grammar.json.
#pragma once
#include <algorithm>
#include <fstream>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "json.hpp"

namespace ctrans {

using json = nlohmann::json;

struct Tok {  // 1-based index i is implicit (position in vector + 1)
    std::string word, lower, pos, deprel, base, cas, gnn, leaf;
    int head = 0;
};

struct Grammar {
    std::set<std::string> modal, modalpart, interrog, pronword, relpron, governed, subord;
    std::map<std::string, std::string> form2lemma;
    std::vector<std::array<std::string, 3>> art_form;             // form, case, gender
    std::map<std::string, std::set<std::string>> prep_gov;        // prep -> governed cases
    std::map<std::string, std::set<std::string>> noun_read;       // noun form -> gender readings
    bool loaded = false;

    bool load(const std::string& path) {
        std::ifstream f(path);
        if (!f) return false;
        json g; f >> g;
        auto to_set = [&](const char* k, std::set<std::string>& s) {
            for (auto& w : g[k]) s.insert(w.get<std::string>());
        };
        to_set("modal", modal); to_set("modalpart", modalpart); to_set("interrog", interrog);
        to_set("pronword", pronword); to_set("relpron", relpron); to_set("governed", governed);
        to_set("subord", subord);
        for (auto& [k, v] : g["form2lemma"].items()) form2lemma[k] = v.get<std::string>();
        for (auto& row : g["art_form"])
            art_form.push_back({row[0].get<std::string>(), row[1].get<std::string>(), row[2].get<std::string>()});
        for (auto& [k, v] : g["prep_gov"].items())
            for (auto& c : v) prep_gov[k].insert(c.get<std::string>());
        for (auto& [k, v] : g["noun_read"].items())
            for (auto& c : v) noun_read[k].insert(c.get<std::string>());
        loaded = true;
        return true;
    }
};

static const std::map<std::string, std::string> CASE_MAP = {
    {"Nom", "nominative"}, {"Acc", "accusative"}, {"Dat", "dative"}, {"Gen", "genitive"}};
static const std::set<std::string> CLAUSE_DEPS = {"advcl", "ccomp", "csubj", "acl", "parataxis", "conj"};

struct Analysis {                        // shared machinery for transform + error rules
    std::vector<Tok> toks;               // toks[i-1]
    std::map<int, std::vector<int>> by_head;
    std::vector<int> clause_of;           // [i] -> clause head (1-based; index 0 unused)
    std::vector<int> roots;
    std::map<int, std::vector<int>> clauses;
    std::map<int, std::vector<int>> pred_chain, pred_members;   // per clause head
    // per clause: groups in python claim order: (anchor_or_min, label, members)
    struct Group { int first; std::string label; std::vector<int> members; };
    std::map<int, std::vector<Group>> groups;
    std::map<int, std::vector<int>> lone;       // singleton function leaves per clause
    std::map<int, std::string> clause_label;    // multi-clause labels (Hauptsatz/...)

    const Grammar* G = nullptr;

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

    void leaf_types() {
        int n = (int)toks.size();
        std::set<int> acl_heads, clause_bound;
        for (int i = 1; i <= n; i++) {
            if (toks[i - 1].deprel.rfind("acl", 0) == 0) acl_heads.insert(i);
            if (CLAUSE_DEPS.count(toks[i - 1].base) && toks[i - 1].deprel.rfind("acl", 0) != 0)
                clause_bound.insert(i);
        }
        std::set<int> relclause;
        for (int i = 1; i <= n; i++) {
            if (toks[i - 1].pos != "PRON") continue;
            int h = i;
            std::set<int> seen;
            while (toks[h - 1].head != 0 && !seen.count(h)) {
                seen.insert(h);
                if (acl_heads.count(h)) { relclause.insert(i); break; }
                if (clause_bound.count(h) && h != i) break;
                h = toks[h - 1].head;
            }
        }
        for (int i = 1; i <= n; i++) {
            Tok& t = toks[i - 1];
            const std::string &w = t.lower, &pos = t.pos, &dep = t.deprel, &base = t.base;
            auto kids_of = by_head.count(i) ? by_head.at(i) : std::vector<int>{};
            auto res = [&]() -> std::string {
                if (pos == "PUNCT") return "Interpunktion";
                if ((w == "als" || w == "wie") && (base == "case" || base == "mark" || base == "cc"))
                    return "Konjunktion";
                if (pos == "ADP") return "Präposition";
                if (G->interrog.count(w)) return "Interrogativpronomen";
                if (G->pronword.count(w)) return "Pronomen";
                if (w.rfind("ander", 0) == 0 && (pos == "DET" || pos == "ADJ" || pos == "PRON"))
                    return "Adjektiv";
                if ((pos == "ADV" || pos == "PART" || pos == "INTJ") && G->modalpart.count(w) &&
                    (base == "advmod" || base == "discourse" || base == "intj"))
                    return "Modalpartikel";
                if (pos == "DET") return "Artikel";
                if (pos == "NOUN" || pos == "PROPN") return "Substantiv";
                if (pos == "NUM") return "Numerale";
                if (pos == "ADV") return "Adverb";
                if (pos == "CCONJ" || pos == "SCONJ") return "Konjunktion";
                if (pos == "PRON") {
                    if ((w == "sich" || w == "mich" || w == "dich" || w == "uns" || w == "euch") &&
                        base != "nsubj")
                        return "Reflexivpronomen";
                    if (G->relpron.count(w) && relclause.count(i)) return "Relativpronomen";
                    return "Pronomen";
                }
                if (pos == "AUX") {
                    if (G->modal.count(w)) return "Modalverb";
                    if (base == "cop" || dep == "root" || base == "root") return "Verb";
                    return "Hilfsverb";
                }
                if (pos == "VERB") {
                    if (base == "compound" || dep == "compound:prt") return "Verbzusatz";
                    for (int k : kids_of) {
                        const Tok& kt = toks[k - 1];
                        if (kt.pos == "AUX" && !G->modal.count(kt.lower)) return "Partizip";
                    }
                    for (int k : kids_of) {
                        const Tok& kt = toks[k - 1];
                        if (kt.lower == "zu" && kt.pos == "PART") return "Infinitiv";
                    }
                    return "Verb";
                }
                if (pos == "PART") return dep == "compound:prt" ? "Verbzusatz" : "Partikel";
                if (pos == "ADJ") return "Adjektiv";
                return "Wort";
            }();
            t.leaf = res;
        }
    }

    void segment() {
        int n = (int)toks.size();
        for (int i = 1; i <= n; i++)
            if (toks[i - 1].head == 0) roots.push_back(i);
        if (roots.empty()) roots.push_back(1);
        clause_of.assign(n + 1, 0);
        std::function<void(int, int)> walk = [&](int h, int own) {
            clause_of[h] = own;
            auto it = by_head.find(h);
            if (it == by_head.end()) return;
            for (int c : it->second) {
                const Tok& t = toks[c - 1];
                bool intro = CLAUSE_DEPS.count(t.base) &&
                    (t.pos == "VERB" || t.pos == "AUX" || t.pos == "NOUN" || t.pos == "ADJ" ||
                     t.pos == "PROPN" || t.pos == "ADV");
                walk(c, intro ? c : own);
            }
        };
        walk(roots[0], roots[0]);
        for (size_t r = 1; r < roots.size(); r++) walk(roots[r], roots[r]);
        for (int i = 1; i <= n; i++) clauses[clause_of[i]].push_back(i);
    }

    void predicates() {
        for (auto& [ch, members] : clauses) {
            std::vector<int> chain{ch}, frontier{ch};
            std::set<int> mem(members.begin(), members.end());
            while (!frontier.empty()) {
                int v = frontier.back(); frontier.pop_back();
                auto it = by_head.find(v);
                if (it == by_head.end()) continue;
                for (int c : it->second)
                    if (mem.count(c) && toks[c - 1].base == "xcomp" &&
                        (toks[c - 1].pos == "VERB" || toks[c - 1].pos == "AUX")) {
                        chain.push_back(c);
                        frontier.push_back(c);
                    }
            }
            pred_chain[ch] = chain;
        }
    }

    std::vector<int> kids_of_clause(int ch) const {
        std::set<int> chain(pred_chain.at(ch).begin(), pred_chain.at(ch).end());
        std::set<int> mem(clauses.at(ch).begin(), clauses.at(ch).end());
        std::vector<int> kids;
        for (int v : pred_chain.at(ch)) {
            auto it = by_head.find(v);
            if (it == by_head.end()) continue;
            for (int c : it->second)
                if (mem.count(c) && !chain.count(c)) kids.push_back(c);
        }
        return kids;
    }

    void build_groups(int ch) {
        const Tok& head_tok = toks[ch - 1];
        std::set<int> mem(clauses.at(ch).begin(), clauses.at(ch).end());
        std::set<int> used;
        auto kids = kids_of_clause(ch);
        bool head_verbal = head_tok.pos == "VERB" || head_tok.pos == "AUX";

        auto clause_subtree = [&](int t) {
            std::vector<int> st;
            for (int j : subtree(t))
                if (clause_of[j] == ch) st.push_back(j);
            return st;
        };
        auto take = [&](const std::vector<int>& idxs, const std::string& label) {
            std::vector<int> take_idx;
            for (int j : idxs)
                if (mem.count(j) && !used.count(j)) take_idx.push_back(j);
            if (take_idx.empty()) return;
            std::sort(take_idx.begin(), take_idx.end());
            for (int j : take_idx) used.insert(j);
            groups[ch].push_back({take_idx.front(), label, take_idx});
        };

        std::vector<int> pred;
        if (head_verbal) pred.push_back(ch);
        else {
            for (int k : kids) {
                const Tok& t = toks[k - 1];
                if (t.base == "cop" || (t.pos == "AUX" && t.base == "aux")) pred.push_back(k);
            }
        }
        for (int k : kids) {
            const Tok& t = toks[k - 1];
            if (t.base == "aux" && !std::count(pred.begin(), pred.end(), k)) pred.push_back(k);
            if (t.deprel == "compound:prt") pred.push_back(k);
            if (t.base == "mark" && t.pos == "PART") pred.push_back(k);
        }
        for (int v : pred_chain.at(ch))
            if (v != ch && !std::count(pred.begin(), pred.end(), v)) pred.push_back(v);
        {
            std::set<int> ps;
            for (int p : pred) {
                const Tok& t = toks[p - 1];
                if (t.pos == "VERB" || t.pos == "AUX" || t.pos == "PART") ps.insert(p);
            }
            pred.assign(ps.begin(), ps.end());
        }

        for (int k : kids) {
            const Tok& t = toks[k - 1];
            auto st = clause_subtree(k);
            const std::string& base = t.base;
            if (base == "nsubj") take(st, "Subjekt");
            else if (base == "obj") take(st, "Akkusativobjekt");
            else if (base == "iobj") take(st, "Dativobjekt");
            else if (base == "obl") {
                std::string prep;
                for (int j : st)
                    if (toks[j - 1].pos == "ADP") { prep = toks[j - 1].lower; break; }
                auto lem = G->form2lemma.find(toks[t.head - 1].lower);
                bool gov = !prep.empty() && lem != G->form2lemma.end() &&
                    G->governed.count(lem->second + "|" + prep);
                bool any_dat = false;
                for (int j : st) any_dat |= toks[j - 1].cas == "Dat";
                if (gov) take(st, "Präpositionalobjekt");
                else if (prep.empty() && t.deprel == "obl:arg" && any_dat) take(st, "Dativobjekt");
                else if (prep.empty() && t.deprel == "obl:arg" && t.cas == "Acc") take(st, "Akkusativobjekt");
                else if (st.size() == 1 && prep.empty()) {
                    int j = st[0];
                    std::string& lf = toks[j - 1].leaf;
                    if (lf == "Adverb" || lf == "Adjektiv" || lf == "Numerale" || lf == "Substantiv")
                        lf = "Adverbiale";
                    used.insert(j);
                    lone[ch].push_back(j);
                } else take(st, "Adverbiale");
            } else if (base == "xcomp" && (t.pos == "ADJ" || t.pos == "NOUN" || t.pos == "PROPN")) {
                take(st, "Prädikativ");
            } else if (base == "advmod") {
                if (st.size() == 1) {
                    int j = st[0];
                    std::string& lf = toks[j - 1].leaf;
                    if (lf == "Adverb" || lf == "Adjektiv" || lf == "Numerale") lf = "Adverbiale";
                    used.insert(j);
                    lone[ch].push_back(j);
                } else take(st, "Adverbiale");
            }
        }
        take(pred, "Prädikat");
        pred_members[ch] = pred;
        if (!head_verbal) {
            std::set<int> own;
            for (int j : subtree(ch)) own.insert(j);
            static const std::set<std::string> XB = {"punct", "cc", "mark", "advmod", "obl",
                                                     "vocative", "discourse"};
            static const std::set<std::string> XP = {"PUNCT", "CCONJ", "SCONJ", "ADV"};
            std::vector<int> leftover;
            for (int j : clauses.at(ch)) {
                if (used.count(j) || !own.count(j)) continue;
                const Tok& t = toks[j - 1];
                if (XB.count(t.base) || XP.count(t.pos)) continue;
                leftover.push_back(j);
            }
            take(leftover, "Prädikativ");
        }
        // remaining tokens become flat leaves at assembly time (not stored — derived from used)
        used_by_clause[ch] = used;
    }
    std::map<int, std::set<int>> used_by_clause;

    void labels() {
        for (auto& [ch, _] : clauses) {
            const Tok& t = toks[ch - 1];
            if (t.head == 0 || t.base == "conj" || t.base == "parataxis") clause_label[ch] = "Hauptsatz";
            else if (t.deprel.rfind("acl", 0) == 0) clause_label[ch] = "Relativsatz";
            else clause_label[ch] = "Nebensatz";
        }
    }

    void run(const Grammar& g) {
        G = &g;
        for (int i = 1; i <= (int)toks.size(); i++)
            if (toks[i - 1].head != 0) by_head[toks[i - 1].head].push_back(i);
        segment();
        leaf_types();
        predicates();
        for (auto& [ch, _] : clauses) build_groups(ch);
        labels();
    }

    // ---- tree assembly (matches python build_clause ordering exactly) ------------------------
    json leaf_node(int j) const {
        const Tok& t = toks[j - 1];
        json n = {{"word", t.word}, {"component", t.leaf}};
        auto it = CASE_MAP.find(t.cas);
        if (it != CASE_MAP.end()) n["case"] = it->second;
        return n;
    }
    std::string span(const std::vector<int>& idxs) const {
        std::string out;
        for (size_t k = 0; k < idxs.size(); k++) {
            if (k) out += " ";
            out += toks[idxs[k] - 1].word;
        }
        return out;
    }
    json build_clause(int ch) const {
        std::vector<std::pair<int, json>> items;
        auto git = groups.find(ch);
        if (git != groups.end())
            for (auto& gr : git->second) {
                json node = {{"word", span(gr.members)}, {"component", gr.label},
                             {"children", json::array()}};
                for (int j : gr.members) node["children"].push_back(leaf_node(j));
                items.push_back({gr.first, node});
            }
        auto lit = lone.find(ch);
        if (lit != lone.end())
            for (int j : lit->second) items.push_back({j, leaf_node(j)});
        const auto& used = used_by_clause.at(ch);   // contains lone leaves too
        for (int j : clauses.at(ch))
            if (!used.count(j)) items.push_back({j, leaf_node(j)});
        std::sort(items.begin(), items.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
        std::vector<int> mem = clauses.at(ch);
        std::sort(mem.begin(), mem.end());
        json node = {{"word", span(mem)}, {"component", nullptr}, {"children", json::array()}};
        for (auto& it2 : items) node["children"].push_back(it2.second);
        return node;
    }
    json tree() const {
        std::vector<int> order;
        for (auto& [ch, _] : clauses) order.push_back(ch);
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            return *std::min_element(clauses.at(a).begin(), clauses.at(a).end()) <
                   *std::min_element(clauses.at(b).begin(), clauses.at(b).end());
        });
        if (order.size() == 1) {
            json root = build_clause(order[0]);
            root["component"] = "Satz";
            return root;
        }
        std::vector<int> all;
        for (int i = 1; i <= (int)toks.size(); i++) all.push_back(i);
        json root = {{"word", span(all)}, {"component", "Satz"}, {"children", json::array()}};
        for (int ch : order) {
            json n = build_clause(ch);
            n["component"] = clause_label.at(ch);
            root["children"].push_back(n);
        }
        return root;
    }

    // ---- error rules (port of errors.dl v2) --------------------------------------------------
    json errors() const {
        json out = json::array();
        auto flag = [&](int i, const char* kind) { out.push_back({{"index", i}, {"kind", kind}}); };
        int n = (int)toks.size();
        // rule 1 + 2 helpers
        std::set<int> det_flagged_np;
        for (int d = 1; d <= n; d++) {
            const Tok& t = toks[d - 1];
            if (t.pos != "DET" || t.base != "det" || t.head == 0) continue;
            const Tok& h = toks[t.head - 1];
            if (h.pos != "NOUN" && h.pos != "PROPN") continue;
            bool known_art = false, ok = false;
            auto nr = G->noun_read.find(h.word);
            for (auto& row : G->art_form) {
                if (row[0] != t.lower) continue;
                known_art = true;
                if (nr != G->noun_read.end() && nr->second.count(row[2])) ok = true;
            }
            if (known_art && nr != G->noun_read.end() && !ok) {
                flag(d, "det_noun_agree");
                det_flagged_np.insert(t.head);
            }
        }
        for (int p = 1; p <= n; p++) {
            const Tok& t = toks[p - 1];
            if (t.pos != "ADP" || t.base != "case" || t.head == 0) continue;
            auto pg = G->prep_gov.find(t.lower);
            if (pg == G->prep_gov.end()) continue;
            int h = t.head;
            if (det_flagged_np.count(h)) continue;
            const Tok& ht = toks[h - 1];
            auto nr = G->noun_read.find(ht.word);
            bool has_cases = false, sat = false;
            auto bh = by_head.find(h);
            if (bh != by_head.end() && nr != G->noun_read.end()) {
                for (int d : bh->second) {
                    const Tok& dt = toks[d - 1];
                    if (dt.pos != "DET" || dt.base != "det") continue;
                    for (auto& row : G->art_form) {
                        if (row[0] != dt.lower || !nr->second.count(row[2])) continue;
                        has_cases = true;
                        if (pg->second.count(row[1])) sat = true;
                    }
                }
            }
            if (has_cases && !sat) flag(p, "prep_case");
        }
        // rules 3 + 4
        for (auto& [ch, members] : clauses) {
            const std::string& lab = clause_label.count(ch) ? clause_label.at(ch)
                                                            : std::string("Hauptsatz");
            bool head_verbal = toks[ch - 1].pos == "VERB" || toks[ch - 1].pos == "AUX";
            // finite verb: last AUX in pred, else verbal head
            int fin = 0;
            for (int v : pred_members.count(ch) ? pred_members.at(ch) : std::vector<int>{})
                if (toks[v - 1].pos == "AUX") fin = std::max(fin, v);
            if (fin == 0 && head_verbal) fin = ch;
            if (fin == 0) continue;
            if (lab != "Hauptsatz") {
                bool is_sub = false;
                for (int m : members)
                    if (toks[m - 1].base == "mark" && toks[m - 1].head == ch &&
                        G->subord.count(toks[m - 1].lower))
                        is_sub = true;
                if (is_sub) {
                    std::set<int> pm(pred_members.at(ch).begin(), pred_members.at(ch).end());
                    for (int j : members)
                        if (j > fin && toks[j - 1].pos != "PUNCT" && !pm.count(j)) {
                            flag(fin, "sub_verb_final");
                            break;
                        }
                }
            } else if (head_verbal) {
                // v2: subject-first + >=2 pre-verb constituents + no punct before fin
                std::vector<std::pair<int, std::string>> consts;   // (first, label)
                if (groups.count(ch))
                    for (auto& gr : groups.at(ch))
                        if (gr.label != "Prädikat") consts.push_back({gr.first, gr.label});
                if (lone.count(ch))
                    for (int j : lone.at(ch)) consts.push_back({j, "lone"});
                if (consts.empty()) continue;
                auto mn = *std::min_element(consts.begin(), consts.end());
                bool subj_first = mn.second == "Subjekt";
                int before = 0;
                for (auto& c : consts) before += c.first < fin;
                bool punct_before = false;
                for (int j : members)
                    punct_before |= j < fin && toks[j - 1].pos == "PUNCT";
                if (subj_first && before >= 2 && !punct_before) flag(fin, "v2_position");
            }
        }
        return out;
    }
};

inline void analyze(std::vector<Tok>& toks, const Grammar& g, json& tree_out, json& errors_out) {
    Analysis a;
    a.toks = std::move(toks);
    a.run(g);
    tree_out = a.tree();
    errors_out = a.errors();
    toks = std::move(a.toks);
}

}  // namespace ctrans
