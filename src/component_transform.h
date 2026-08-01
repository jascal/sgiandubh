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
    std::map<std::string, std::string> contracted;                // fused prep form -> governed case
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
        if (g.contains("contracted"))   // absent in packages exported before germandata#15
            for (auto& [k, v] : g["contracted"].items()) contracted[k] = v.get<std::string>();
        loaded = true;
        return true;
    }
};

static const std::map<std::string, std::string> CASE_MAP = {
    {"Nom", "nominative"}, {"Acc", "accusative"}, {"Dat", "dative"}, {"Gen", "genitive"}};
static const std::set<std::string> CLAUSE_DEPS = {"advcl", "ccomp", "csubj", "acl", "parataxis", "conj"};
static const std::set<std::string> CLAUSE_POS = {"VERB", "AUX", "NOUN", "ADJ", "PROPN", "ADV"};
static const std::set<std::string> CLAUSAL_KID = {"nsubj", "csubj", "cop", "aux"};

struct Analysis {                        // shared machinery for transform + error rules
    std::vector<Tok> toks;               // toks[i-1]
    std::map<int, std::vector<int>> by_head;
    std::vector<int> clause_of;           // [i] -> clause head (1-based; index 0 unused)
    std::vector<int> roots;
    std::map<int, std::vector<int>> clauses;
    std::vector<int> sentence_punct;      // Satz-level tokens, in no clause (see segment())
    std::map<int, std::vector<int>> pred_chain, pred_members;   // per clause head
    // per clause: groups in python claim order: (anchor_or_min, label, members)
    // `anchor` is the garg kid token that anchors the group, 0 for the Prädikat, the promoted-head
    // Prädikativ leftover and lone leaves — precisely the groups a nested clause may not enter (#6).
    struct Group { int first; std::string label; std::vector<int> members; int anchor = 0; };
    std::map<int, std::vector<Group>> groups;
    std::map<int, std::vector<int>> lone;       // singleton function leaves per clause
    std::map<int, std::string> clause_label;    // multi-clause labels (Hauptsatz/...)
    std::map<int, std::string> case_ov;         // contracted-prep case override per token (#15)

    const Grammar* G = nullptr;

    // Is this `conj` conjunct a clause, or just a coordinated phrase? POS cannot make the call:
    // `Lehrerin` in `Er ist Arzt und sie Lehrerin.` (gapping — a real clause, the case the POS
    // gate was there for) and `Peter` in `Anna und Peter kommen morgen.` (a noun phrase) are both
    // conj + NOUN. Gating on POS alone promoted EVERY phrase-level conjunct to a second clause —
    // `Anna kommen morgen` + a bogus `Hauptsatz` `und Peter`. The discriminator is clause-level
    // material of the conjunct's own. See germandata#7.
    bool clausal_conj(int i) const {
        if (toks[i - 1].pos == "VERB" || toks[i - 1].pos == "AUX") return true;
        auto it = by_head.find(i);
        if (it == by_head.end()) return false;
        for (int k : it->second)
            if (CLAUSAL_KID.count(toks[k - 1].base)) return true;
        return false;
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

    void leaf_types() {
        int n = (int)toks.size();
        std::set<int> acl_heads, clause_bound;
        for (int i = 1; i <= n; i++) {
            // An acl whose modified head is an interjection is a mis-parsed comma splice
            // ("Danke, das hilft mir sehr."): relative clauses modify nominals, never
            // interjections. Labelled Hauptsatz in labels(); marks no relative pronouns here.
            if (toks[i - 1].deprel.rfind("acl", 0) == 0 &&
                !(toks[i - 1].head > 0 && toks[toks[i - 1].head - 1].pos == "INTJ"))
                acl_heads.insert(i);
            // A phrase conjunct is no longer a clause head, so it is no longer a clause boundary
            // either — this set has to track segment()'s walk or the two disagree about where a
            // clause ends (the .dl bounds this on clause_head itself, via intervening_clause).
            if (CLAUSE_DEPS.count(toks[i - 1].base) && toks[i - 1].deprel.rfind("acl", 0) != 0 &&
                (toks[i - 1].base != "conj" || clausal_conj(i)))
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
                // A separable-verb prefix is identified by its DEPENDENCY, not its POS: "kam … an"
                // makes the sentence's verb ankommen, and calling the prefix a preposition hides
                // that. The two Verbzusatz branches below are gated on POS = VERB / PART, which
                // between them match NONE of the 17963 compound:prt tokens in train_union.conllu
                // (ADP 17279, ADV 521, ADJ 162, NOUN 1, PART/VERB zero) — so every separable prefix
                // reached the ADP branch. See germandata#3. Tested on the FULL deprel, never base:
                // base == "compound" would also catch noun compounds, which are not verb prefixes.
                if (dep == "compound:prt") return "Verbzusatz";
                if (pos == "ADP") return "Präposition";
                if (G->interrog.count(w)) return "Interrogativpronomen";
                if (G->pronword.count(w)) return "Pronomen";
                if (w.rfind("ander", 0) == 0 && (pos == "DET" || pos == "ADJ" || pos == "PRON"))
                    return "Adjektiv";
                // The tag layer distinguishes the readings now (satzklar-model#7/#9): INTJ from
                // the tagger is the answer-word / greeting reading (Ja., Nein., Hallo!), kept
                // apart from the unstressed mid-field Modalpartikel (Das ist ja toll — ADV/PART).
                // Sits ABOVE Modalpartikel so a discourse "Ja," keeps its interjection identity;
                // the INTJ alternative Modalpartikel carried was unreachable in production until
                // the tag layer could emit INTJ at all.
                if (pos == "INTJ") return "Interjektion";
                if ((pos == "ADV" || pos == "PART") && G->modalpart.count(w) &&
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
                bool intro = CLAUSE_DEPS.count(t.base) && CLAUSE_POS.count(t.pos) &&
                    (t.base != "conj" || clausal_conj(c));
                walk(c, intro ? c : own);
            }
        };
        walk(roots[0], roots[0]);
        for (size_t r = 1; r < roots.size(); r++) walk(roots[r], roots[r]);
        // The sentence-final terminator punctuates the WHOLE sentence, so it belongs to Satz, not
        // to a clause: UD attaches it to the root and the root heads the FIRST clause, so without
        // this it lands inside clause 1 even when the sentence runs on (germandata#1). Guarded on
        // multi-clause to keep single-clause trees byte-identical — there the clause node IS Satz.
        // Narrow by design: clause-internal commas and quotes stay in the clause they punctuate.
        static const std::set<std::string> SENT_FINAL = {".", "?", "!", "…", "?!", "!?"};
        std::set<int> distinct;
        for (int i = 1; i <= n; i++) distinct.insert(clause_of[i]);
        // clause_of[n] != n guards the degenerate parse where the final PUNCT is itself the root,
        // hence a clause head: lifting it would leave its own clause headless.
        if (distinct.size() > 1 && toks[n - 1].pos == "PUNCT" &&
            SENT_FINAL.count(toks[n - 1].word) && clause_of[n] != n) {
            sentence_punct.push_back(n);
            clause_of[n] = 0;
        }
        for (int i = 1; i <= n; i++)
            if (clause_of[i] != 0) clauses[clause_of[i]].push_back(i);
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
        auto take = [&](const std::vector<int>& idxs, const std::string& label, int anchor = 0) {
            std::vector<int> take_idx;
            for (int j : idxs)
                if (mem.count(j) && !used.count(j)) take_idx.push_back(j);
            if (take_idx.empty()) return;
            std::sort(take_idx.begin(), take_idx.end());
            for (int j : take_idx) used.insert(j);
            groups[ch].push_back({take_idx.front(), label, take_idx, anchor});
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
            // The POS filter keeps non-verbal material out of the predicate complex, but a
            // separable prefix is verbal by DEPENDENCY, not by POS (compound:prt is ADP 17279,
            // ADV 521, ADJ 162, NOUN 1, PART/VERB zero) — so it undid the compound:prt append
            // above for every separable verb. Same defect as germandata#3 one layer up: there a
            // POS gate cost the prefix its label, here it cost the predicate its prefix.
            // `kam … an` is ONE Prädikat, exactly as `hat … gelesen` already is. See germandata#4.
            std::set<int> ps;
            for (int p : pred) {
                const Tok& t = toks[p - 1];
                if (t.pos == "VERB" || t.pos == "AUX" || t.pos == "PART" ||
                    t.deprel == "compound:prt")
                    ps.insert(p);
            }
            pred.assign(ps.begin(), ps.end());
        }

        for (int k : kids) {
            const Tok& t = toks[k - 1];
            auto st = clause_subtree(k);
            const std::string& base = t.base;
            if (base == "nsubj") take(st, "Subjekt", k);
            else if (base == "obj") take(st, "Akkusativobjekt", k);
            else if (base == "iobj") take(st, "Dativobjekt", k);
            else if ((base == "obl" || base == "advmod") && t.pos == "INTJ") {
                // A discourse interjection the parser labelled advmod/obl ("Ja, das stimmt." —
                // gold says discourse) is not a Satzglied. No Adverbiale wrapper — the tokens
                // fall through to flat leaves, exactly as when the parser says discourse.
                // Subjekt/objects stay grouped: in "Ja ist eine Antwort" the Ja IS the subject.
            }
            else if (base == "obl") {
                std::string prep;
                for (int j : st)
                    if (toks[j - 1].pos == "ADP") { prep = toks[j - 1].lower; break; }
                auto lem = G->form2lemma.find(toks[t.head - 1].lower);
                bool gov = !prep.empty() && lem != G->form2lemma.end() &&
                    G->governed.count(lem->second + "|" + prep);
                bool any_dat = false;
                for (int j : st) any_dat |= toks[j - 1].cas == "Dat";
                if (gov) take(st, "Präpositionalobjekt", k);
                else if (prep.empty() && t.deprel == "obl:arg" && any_dat) take(st, "Dativobjekt", k);
                else if (prep.empty() && t.deprel == "obl:arg" && t.cas == "Acc") take(st, "Akkusativobjekt", k);
                else if (st.size() == 1 && prep.empty()) {
                    int j = st[0];
                    std::string& lf = toks[j - 1].leaf;
                    if (lf == "Adverb" || lf == "Adjektiv" || lf == "Numerale" || lf == "Substantiv")
                        lf = "Adverbiale";
                    used.insert(j);
                    lone[ch].push_back(j);
                } else take(st, "Adverbiale", k);
            } else if (base == "xcomp" && (t.pos == "ADJ" || t.pos == "NOUN" || t.pos == "PROPN")) {
                take(st, "Prädikativ", k);
            } else if (base == "advmod") {
                if (st.size() == 1) {
                    int j = st[0];
                    std::string& lf = toks[j - 1].leaf;
                    if (lf == "Adverb" || lf == "Adjektiv" || lf == "Numerale") lf = "Adverbiale";
                    used.insert(j);
                    lone[ch].push_back(j);
                } else take(st, "Adverbiale", k);
            }
        }
        take(pred, "Prädikat");
        pred_members[ch] = pred;
        // INTJ is excluded: the promoted-head Prädikativ encodes "a verbless clause is an
        // elliptical copular predication", which holds for nominal heads (Schönes Wetter heute!)
        // but not for an interjection utterance — there is no elided "ist" in "Oh!". An
        // INTJ-headed clause emits no Prädikativ; its leaves stay flat under the clause.
        if (!head_verbal && head_tok.pos != "INTJ") {
            // The promoted head's Prädikativ used to be "its subtree MINUS a blocklist", and the
            // complement of a blocklist is not a constituent: a blocklisted token sitting INSIDE
            // the kept material (`erst`, the `und` of an NP coordination, an apposition comma) was
            // expelled and rendered after the whole group. So take the maximal contiguous RUN
            // through the head instead, trimmed back to its outermost kept tokens — interior
            // blocklisted tokens are absorbed because their position is evidence they belong, edge
            // ones are trimmed because adjacency is not. Every group emitted here is a contiguous
            // interval by construction. Detached kept-chunks fall through to the flat-leaf path,
            // which is position-sorted — ordered, not yet grouped (stage B). See germandata#8.
            std::set<int> own;
            for (int j : subtree(ch)) own.insert(j);
            static const std::set<std::string> XB = {"punct", "cc", "mark", "advmod", "obl",
                                                     "vocative", "discourse"};
            static const std::set<std::string> XP = {"PUNCT", "CCONJ", "SCONJ", "ADV"};
            std::set<int> cand, keep;
            for (int j : clauses.at(ch)) {
                if (used.count(j) || !own.count(j)) continue;
                cand.insert(j);
                const Tok& t = toks[j - 1];
                if (!XB.count(t.base) && !XP.count(t.pos)) keep.insert(j);
            }
            std::set<int> run{ch};
            for (int j = ch + 1; cand.count(j); j++) run.insert(j);
            for (int j = ch - 1; cand.count(j); j--) run.insert(j);
            int lo = 0, hi = 0;
            for (int j : run)
                if (keep.count(j)) { if (!lo) lo = j; hi = j; }   // run is ascending
            if (lo) {
                std::vector<int> praedikativ;
                for (int j : run)
                    if (j >= lo && j <= hi) praedikativ.push_back(j);
                take(praedikativ, "Prädikativ");
            }
        }
        // remaining tokens become flat leaves at assembly time (not stored — derived from used)
        used_by_clause[ch] = used;
    }
    std::map<int, std::set<int>> used_by_clause;

    void labels() {
        for (auto& [ch, _] : clauses) {
            const Tok& t = toks[ch - 1];
            if (t.head == 0 || t.base == "conj" || t.base == "parataxis") clause_label[ch] = "Hauptsatz";
            else if (t.deprel.rfind("acl", 0) == 0)
                // off an interjection the acl is a mis-parsed comma splice — the clause is the
                // Hauptsatz it actually is (see acl_heads in leaf_types)
                clause_label[ch] = toks[t.head - 1].pos == "INTJ" ? "Hauptsatz" : "Relativsatz";
            else clause_label[ch] = "Nebensatz";
        }
    }

    // ---- placement: nest exactly the clauses that INTERRUPT their matrix (germandata#6) --------
    // Flat siblings ordered by first token cannot express an embedded clause: a relative clause
    // cutting into its matrix renders after the whole of it, in a position the sentence never had.
    // Scoped to interruption — an extraposed clause already reads in surface order and stays flat.
    // dl/transform.dl carries the normative rules; this computes the same predicate natively.
    std::set<int> nested;                              // clause heads that moved
    std::set<int> adopted;                             // tokens moved out of their matrix
    std::map<int, int> adopt_of;                       // clause head -> comma it adopted
    std::map<int, std::vector<int>> nested_in_group;   // group anchor -> clause heads
    std::map<int, std::vector<int>> nested_in_clause;  // matrix clause head -> clause heads

    void placement() {
        for (auto& [ch, members] : clauses) {
            if (clause_label.count(ch) && clause_label.at(ch) == "Hauptsatz") continue;
            int h = toks[ch - 1].head;
            if (h == 0) continue;
            int m = clause_of[h];
            if (m == 0 || m == ch || !clauses.count(m)) continue;
            int lo = *std::min_element(members.begin(), members.end());
            int hi = *std::max_element(members.begin(), members.end());
            bool left = false, right = false;
            for (int x : clauses.at(m)) { left |= x < lo; right |= x > hi; }
            if (!(left && right)) continue;            // does not interrupt — leave it flat
            int anchor = 0;
            if (toks[ch - 1].deprel.rfind("acl", 0) == 0 && groups.count(m))
                for (auto& gr : groups.at(m))
                    if (gr.anchor && std::count(gr.members.begin(), gr.members.end(), h)) {
                        anchor = gr.anchor;            // into the Satzglied the clause modifies
                        break;
                    }
            if (anchor) nested_in_group[anchor].push_back(ch);
            else nested_in_clause[m].push_back(ch);    // a pred anchor sits in no group
            nested.insert(ch);
            int j = hi + 1;                            // the closing comma of the pair belongs to it
            if (j <= (int)toks.size() && toks[j - 1].word == "," && toks[j - 1].pos == "PUNCT" &&
                clause_of[j] == m) {
                adopt_of[ch] = j;
                adopted.insert(j);
            }
        }
    }

    void run(const Grammar& g) {
        G = &g;
        for (int i = 1; i <= (int)toks.size(); i++)
            if (toks[i - 1].head != 0) by_head[toks[i - 1].head].push_back(i);
        // A contracted preposition determines its noun's case (übers = über + das → Acc): the
        // fused article is deterministic where the tagger's morph tag is not — trained on
        // MWT-split corpora, it rarely sees fused forms (germandata#15). Display-scope only:
        // grouping keeps the tagger case, and a leaf that emits no case gains none. The lowest
        // case-child wins, matching the .dl's min-aggregate and the python twin's first-child.
        for (int i = 1; i <= (int)toks.size(); i++) {
            if (!CASE_MAP.count(toks[i - 1].cas) || !by_head.count(i)) continue;
            for (int k : by_head.at(i)) {
                const Tok& c = toks[k - 1];
                if (c.pos == "ADP" && c.base == "case" && G->contracted.count(c.lower)) {
                    case_ov[i] = G->contracted.at(c.lower);
                    break;
                }
            }
        }
        segment();
        leaf_types();
        predicates();
        for (auto& [ch, _] : clauses) build_groups(ch);
        labels();
        placement();
    }

    // ---- tree assembly (matches python build_clause ordering exactly) ------------------------
    // `i` is the 1-based token index. Additive, and it makes every position-aware consumer exact
    // rather than heuristic: without it, anything needing to know WHERE a constituent sits (the
    // Satzklammer display, germanapp#268) has to align tokens back to the sentence by matching
    // text, which is what produced a real bug on `das das` and repeated punctuation
    // (germanapp#259). Groups carry no index — their extent is the indices of their children.
    json leaf_node(int j) const {
        const Tok& t = toks[j - 1];
        json n = {{"word", t.word}, {"component", t.leaf}, {"i", j}};
        auto it = CASE_MAP.find(t.cas);
        if (it != CASE_MAP.end()) {
            auto ov = case_ov.find(j);
            n["case"] = ov == case_ov.end() ? it->second : CASE_MAP.at(ov->second);
        }
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
    // Sort children by first token and derive every span from the leaves beneath it. Span text is
    // derived data, which is what lets nesting chains fall out of plain node placement with no
    // special casing. On a tree with nothing nested it is a no-op: children were already emitted
    // in first-token order and a span was already the join of its sorted members.
    std::vector<int> order_and_span(json& node) const {
        if (!node.contains("children")) return {node.at("i").get<int>()};
        std::vector<std::pair<std::vector<int>, json>> keyed;
        for (auto& c : node["children"]) {
            auto idxs = order_and_span(c);
            if (!idxs.empty()) keyed.push_back({idxs, c});   // a group emptied by adoption vanishes
        }
        std::stable_sort(keyed.begin(), keyed.end(),
                         [](const auto& a, const auto& b) { return a.first.front() < b.first.front(); });
        json arr = json::array();
        std::vector<int> all;
        for (auto& k : keyed) {
            arr.push_back(k.second);
            all.insert(all.end(), k.first.begin(), k.first.end());
        }
        std::sort(all.begin(), all.end());
        node["children"] = arr;
        node["word"] = span(all);
        return all;
    }
    json clause_node(int ch) const {
        json n = build_clause(ch);
        n["component"] = clause_label.count(ch) ? clause_label.at(ch) : std::string("Nebensatz");
        return n;
    }
    json build_clause(int ch) const {
        json node = {{"word", ""}, {"component", nullptr}, {"children", json::array()}};
        auto git = groups.find(ch);
        if (git != groups.end())
            for (auto& gr : git->second) {
                json g = {{"word", ""}, {"component", gr.label}, {"children", json::array()}};
                for (int j : gr.members)
                    if (!adopted.count(j)) g["children"].push_back(leaf_node(j));
                if (gr.anchor) {
                    auto nit = nested_in_group.find(gr.anchor);
                    if (nit != nested_in_group.end())
                        for (int c : nit->second) g["children"].push_back(clause_node(c));
                }
                if (!g["children"].empty()) node["children"].push_back(g);
            }
        auto lit = lone.find(ch);
        if (lit != lone.end())
            for (int j : lit->second)
                if (!adopted.count(j)) node["children"].push_back(leaf_node(j));
        const auto& used = used_by_clause.at(ch);   // contains lone leaves too
        for (int j : clauses.at(ch))
            if (!used.count(j) && !adopted.count(j)) node["children"].push_back(leaf_node(j));
        auto nit = nested_in_clause.find(ch);
        if (nit != nested_in_clause.end())
            for (int c : nit->second) node["children"].push_back(clause_node(c));
        auto ait = adopt_of.find(ch);
        if (ait != adopt_of.end()) node["children"].push_back(leaf_node(ait->second));
        return node;
    }
    json tree() const {
        json root;
        if (clauses.size() == 1) {
            root = build_clause(clauses.begin()->first);
            root["component"] = "Satz";
        } else {
            root = {{"word", ""}, {"component", "Satz"}, {"children", json::array()}};
            for (auto& [ch, _] : clauses)
                if (!nested.count(ch)) root["children"].push_back(clause_node(ch));
            for (int j : sentence_punct) root["children"].push_back(leaf_node(j));
        }
        order_and_span(root);   // sorts every level by first token and fills in every span
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
