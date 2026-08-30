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

// ---------------------------------------------------------------- the pack
struct Pack {
    json j;
    std::map<std::string, std::set<std::string>> cache;   // resolved rule operands
    std::map<std::string, std::map<std::string, std::vector<std::pair<std::string, std::string>>>>
        read_cache;                                       // resolved reading tables

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

    // A register is looked up on the written form or the lowercase one — a
    // property of the REGISTER, not the language (German nouns are capitalised;
    // the article paradigm is written lowercase and must match a leading "Der").
    bool case_sensitive(const std::string& reg) {
        const json& f = j["features"];
        if (!f.contains("case_sensitive_registers")) return false;
        for (auto& r : f["case_sensitive_registers"])
            if (r.get<std::string>() == reg) return true;
        return false;
    }

    using Reading = std::pair<std::string, std::string>;   // ("Masc", "Sing"); "-" = unspecified

    const std::map<std::string, std::vector<Reading>>& readings(const std::string& name) {
        auto it = read_cache.find(name);
        if (it != read_cache.end()) return it->second;
        std::map<std::string, std::vector<Reading>> tab;
        bool cs = case_sensitive(name);
        if (j["lexicons"].contains(name) && j["lexicons"][name].is_object())
            for (auto& [form, vals] : j["lexicons"][name].items()) {
                std::string key = cs ? form : utf8_lower(form);
                for (auto& v : vals) {
                    std::string r = v.get<std::string>();
                    size_t bar = r.find('|');
                    Reading rd = bar == std::string::npos ? Reading{r, "-"}
                                                          : Reading{r.substr(0, bar), r.substr(bar + 1)};
                    auto& into = tab[key];
                    if (std::find(into.begin(), into.end(), rd) == into.end()) into.push_back(rd);
                }
            }
        return read_cache.emplace(name, std::move(tab)).first->second;
    }

    std::vector<Reading> read_of(const std::string& reg, const std::string& word) {
        const auto& tab = readings(reg);
        auto it = tab.find(case_sensitive(reg) ? word : utf8_lower(word));
        return it == tab.end() ? std::vector<Reading>{} : it->second;
    }

    /** Everything the PACK can prove about one surface form.
     *
     * The certified half of a dictionary: lemma, morphological readings, closed-class
     * membership and government, all read straight out of the lexicons this expert
     * already holds in memory. No model, no network, and the same shape in every
     * language, because the packs share a schema.
     *
     * What it does NOT do is invent. A form the registers do not know comes back
     * `known: false` with whatever little was found, so a caller can tell "the
     * lexicon says this lemma" from "a model guessed it" — the registers are mined
     * from treebanks and their coverage is corpus-bound (English form2lemma has
     * ~5.5k entries against German's ~14k), which makes that distinction the whole
     * point rather than a detail.
     */
    json lexical(const std::string& word) {
        // scope=form is load-bearing, not decoration. This answers about the FORM,
        // not about an occurrence of it: German `der` is in the relative-pronoun
        // register AND the article paradigm, and no amount of lexicon lookup can
        // say which one "Der Mann" is. The parse can, so a caller that has a
        // sentence should disambiguate with the token's deprel/POS and treat these
        // as the candidate readings, not the answer.
        json out = {{"word", word}, {"known", false}, {"scope", "form"}};
        const json& lex = j["lexicons"];
        const std::string low = utf8_lower(word);

        // lemma
        if (lex.contains("form2lemma") && lex["form2lemma"].is_object()) {
            auto it = lex["form2lemma"].find(low);
            if (it == lex["form2lemma"].end()) it = lex["form2lemma"].find(word);
            if (it != lex["form2lemma"].end() && it->is_string()) {
                out["lemma"] = it->get<std::string>();
                out["known"] = true;
            }
        }

        // morphological readings, whichever tables this pack carries
        json readings_out = json::object();
        for (const char* reg : {"noun_gn", "noun_gnum", "verb_pn", "adj_gn", "det_gn",
                                "pron_pn", "art_gender", "art_case"}) {
            if (!lex.contains(reg) || !lex[reg].is_object()) continue;
            auto rs = read_of(reg, word);
            if (rs.empty()) continue;
            json arr = json::array();
            for (auto& r : rs) arr.push_back(r.second == "-" ? json(r.first)
                                                             : json(r.first + "|" + r.second));
            readings_out[reg] = arr;
            out["known"] = true;
        }
        if (!readings_out.empty()) out["readings"] = readings_out;

        // closed-class membership: every lexicon that is a plain list of forms
        json classes = json::array();
        for (auto& [name, entries] : lex.items()) {
            if (!entries.is_array()) continue;
            for (auto& e : entries)
                if (e.is_string() && (e.get<std::string>() == word ||
                                      utf8_lower(e.get<std::string>()) == low)) {
                    classes.push_back(name);
                    out["known"] = true;
                    break;
                }
        }
        if (!classes.empty()) out["classes"] = classes;

        // government: prepositions this lemma takes ("abbauen|in"), and for a
        // preposition the case it governs
        const std::string lemma = out.value("lemma", low);
        const std::string gov_name = j["features"].value("governed_lexicon", std::string("governed"));
        if (lex.contains(gov_name)) {
            json takes = json::array();
            for (auto& e : lex[gov_name]) {
                if (!e.is_string()) continue;
                const std::string v = e.get<std::string>();
                size_t bar = v.find('|');
                if (bar != std::string::npos && v.substr(0, bar) == lemma)
                    takes.push_back(v.substr(bar + 1));
            }
            if (!takes.empty()) { out["governs"] = takes; out["known"] = true; }
        }
        if (lex.contains("prep_gov") && lex["prep_gov"].is_object()) {
            auto it = lex["prep_gov"].find(low);
            if (it != lex["prep_gov"].end()) { out["governsCase"] = *it; out["known"] = true; }
        }
        if (lex.contains("contracted") && lex["contracted"].is_object()) {
            auto it = lex["contracted"].find(low);
            if (it != lex["contracted"].end()) { out["contraction"] = *it; out["known"] = true; }
        }

        out["provenance"] = {{"lang", j.value("lang", "?")},
                             {"pack", j.value("name", "")},
                             {"version", j.value("version", "")},
                             {"lexicons", j.value("provenance", json::object()).value("lexicons", "")}};
        return out;
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
        } else if (atom == "head_has_child") {
            // a sibling test: does the token's HEAD have another dependent like
            // this? (the ditransitive test behind Spanish's indirect object)
            if (!t.head) return false;
            auto it = C.by_head->find(t.head);
            bool hit = false;
            if (it != C.by_head->end())
                for (int k : it->second)
                    if (k != i && tok_match(spec, k, C)) { hit = true; break; }
            if (!hit) return false;
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
// --- character offsets -------------------------------------------------------
// Offsets are CODE POINT indices, not bytes. Python's str indexes code points and
// a JS string indexes UTF-16 code units, which agree with code points for
// everything below U+10000 — every umlaut, accent and quote these languages use.
// Byte offsets agree with neither the moment the text stops being ASCII, so
// counting them here would put this port out of step with the spec it is gated
// against AND with the browser that consumes it.

/** Decode UTF-8 to code points; malformed bytes pass through as themselves. */
inline std::vector<char32_t> utf8_cps(const std::string& s) {
    std::vector<char32_t> out;
    for (size_t i = 0; i < s.size();) {
        unsigned char c = (unsigned char)s[i];
        int len = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : (c >> 3) == 0x1E ? 4 : 1;
        char32_t cp = len == 1 ? c : (c & (0xFF >> (len + 1)));
        for (int k = 1; k < len && i + k < s.size(); k++)
            cp = (cp << 6) | ((unsigned char)s[i + k] & 0x3F);
        out.push_back(cp);
        i += len;
    }
    return out;
}

/** The fold table of glossa/engine.py, code point for code point. 1:1 so offsets survive. */
inline char32_t fold_cp(char32_t c) {
    switch (c) {
        case 0x201C: case 0x201D: case 0x201E: case 0x201F: return U'"';
        case 0x2018: case 0x2019: case 0x201A: case 0x201B: return U'\'';
        case 0x2013: case 0x2014: case 0x2212: return U'-';
        case 0x00A0: return U' ';
        case 0x00E1: return U'a'; case 0x00E9: return U'e'; case 0x00ED: return U'i';
        case 0x00F3: return U'o'; case 0x00FA: return U'u';
        case 0x00C1: return U'A'; case 0x00C9: return U'E'; case 0x00CD: return U'I';
        case 0x00D3: return U'O'; case 0x00DA: return U'U';
        case 0x00FC: return U'u'; case 0x00DC: return U'U';
        case 0x00F1: return U'n'; case 0x00D1: return U'N';
        default: return c;
    }
}

inline bool cp_space(char32_t c) {
    if (c < 0x80) return c == U' ' || c == U'\t' || c == U'\n' || c == U'\r' || c == U'\f' || c == U'\v';
    return c == 0x00A0 || (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 ||
           c == 0x202F || c == 0x205F || c == 0x3000;
}

inline std::vector<char32_t> folded_cps(const std::string& s) {
    auto v = utf8_cps(s);
    for (auto& c : v) c = fold_cp(c);
    return v;
}

class Transform {
  public:
    Transform(Pack& pack) : P(pack) {}

    // `with_roles` adds the canonical role beside every display label. Off by
    // default because the German served tree is gated byte-identical against
    // germandata's; a caller comparing structures ACROSS languages asks for it,
    // since labels differ per language and roles do not.
    // `with_spans` adds the extent of every node: iStart/iEnd (1-based inclusive
    // token indices) always, and start/end CODE POINT offsets into `text` when a
    // text is supplied. Off by default for the same reason roles are.
    json run(const std::vector<Tok>& input, bool with_roles = false,
             bool with_spans = false, const std::string& text = "") {
        roles = with_roles;
        spans = with_spans;
        cspans.clear();
        if (spans && !text.empty()) cspans = char_spans(input, text);
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
    bool roles = false;                      // emit the canonical role beside the label
    bool spans = false;                      // emit iStart/iEnd (+ start/end with a text)
    std::vector<std::pair<int, int>> cspans; // per token, {-1,-1} where unplaceable
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

    /** The display label, and the canonical role beside it when asked. */
    void name(json& node, const std::string& role) {
        node["component"] = P.label(role);
        if (roles) node["role"] = role;
    }

    /** Locate each token in `text`; {-1,-1} where it cannot be placed.
     *
     * The port of glossa/engine.py's char_spans, and it must stay the port: the
     * differential gate compares this against that. Alignment is against the
     * WRITTEN WORDS of the text rather than a search, because a free find matches
     * a later occurrence of the same string and drags the cursor past everything
     * between — AnCora expands the name particle *da* into `de` + `a`, and `de`
     * matched "defensa de Brasil" thirty characters downstream. Those offsets were
     * wrong rather than missing, which no equality check on the substring can see.
     */
    static std::vector<std::pair<int, int>> char_spans(const std::vector<Tok>& toks,
                                                      const std::string& text) {
        std::vector<char32_t> f = folded_cps(text);
        const int n = (int)f.size();
        std::vector<std::pair<int, int>> words;
        for (int i = 0; i < n;) {
            while (i < n && cp_space(f[i])) i++;
            if (i >= n) break;
            int j = i;
            while (j < n && !cp_space(f[j])) j++;
            words.push_back({i, j});
            i = j;
        }
        std::vector<std::pair<int, int>> out;
        size_t wi = 0;
        int pos = words.empty() ? 0 : words[0].first;
        int misses = 0;
        for (const Tok& t : toks) {
            std::vector<char32_t> w = folded_cps(t.word);
            if (wi >= words.size()) { out.push_back({-1, -1}); continue; }
            const int end_w = words[wi].second;
            bool fits = !w.empty() && pos + (int)w.size() <= end_w;
            if (fits)
                for (size_t k = 0; k < w.size(); k++)
                    if (f[pos + k] != w[k]) { fits = false; break; }
            if (fits) {
                out.push_back({pos, pos + (int)w.size()});
                pos += (int)w.size();
                misses = 0;
                if (pos >= end_w) {
                    wi++;
                    pos = wi < words.size() ? words[wi].first : n;
                }
                continue;
            }
            out.push_back({-1, -1});
            if (++misses >= 2) {            // a fused form the treebank expanded
                wi++;
                pos = wi < words.size() ? words[wi].first : n;
                misses = 0;
            }
        }
        return out;
    }

    json leaf_node(int i) {
        const Tok& t = toks[i - 1];
        json node = {{"word", t.word}, {"i", i}};
        name(node, t.leaf);
        if (case_display.count(t.cas) && !case_exclude.count(t.deprel))
            node["case"] = case_display.at(t.case_ov.empty() ? t.cas : t.case_ov);
        if (spans) {
            node["iStart"] = i;
            node["iEnd"] = i;
            if (!cspans.empty() && cspans[i - 1].first >= 0) {
                node["start"] = cspans[i - 1].first;
                node["end"] = cspans[i - 1].second;
            }
        }
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

    // ---- error rules (glossa/errors.py, same five templates) -------------
  public:
    json errors() {
        json out = json::array();
        if (!P.j.contains("errors")) return out;
        std::set<std::pair<int, std::string>> flags, dropped;
        for (auto& rule : P.j["errors"]) {
            const std::string t = rule["template"].get<std::string>();
            const std::string id = rule["id"].get<std::string>();
            std::vector<int> hits;
            if (t == "agreement") hits = err_agreement(rule);
            else if (t == "subject_verb") hits = err_subject_verb(rule);
            else if (t == "case_government") hits = err_case_government(rule);
            else if (t == "verb_position") hits = err_verb_position(rule);
            else if (t == "form_choice") hits = err_form_choice(rule);
            for (int i : hits) flags.insert({i, id});
        }
        // one flag per NP: a rule that explains a phrase suppresses the named others
        for (auto& rule : P.j["errors"]) {
            if (!rule.contains("suppresses")) continue;
            const std::string id = rule["id"].get<std::string>();
            for (auto& o : rule["suppresses"]) {
                std::string other = o.get<std::string>();
                for (auto& [i, kind] : flags) {
                    if (kind != id) continue;
                    int head = toks[i - 1].head;
                    for (auto& [j, k2] : flags)
                        if (k2 == other && (toks[j - 1].head == head || toks[j - 1].head == i ||
                                            j == head))
                            dropped.insert({j, k2});
                }
            }
        }
        for (auto& f : flags)
            if (!dropped.count(f)) out.push_back({{"index", f.first}, {"kind", f.second}});
        return out;
    }

  private:
    // `-` is UNSPECIFIED, not a value: UD omits Gender on invariant forms, so a
    // reading with `-` agrees with anything in that slot.
    static bool compatible(const std::vector<Pack::Reading>& a,
                           const std::vector<Pack::Reading>& b) {
        for (auto& x : a)
            for (auto& y : b)
                if ((x.first == y.first || x.first == "-" || y.first == "-") &&
                    (x.second == y.second || x.second == "-" || y.second == "-"))
                    return true;
        return false;
    }

    std::vector<int> kids_of(int i) const {
        auto it = by_head.find(i);
        return it == by_head.end() ? std::vector<int>{} : it->second;
    }

    static std::set<std::string> json_set(const json& arr) {
        std::set<std::string> out;
        for (auto& x : arr) out.insert(x.get<std::string>());
        return out;
    }

    std::vector<int> err_agreement(const json& rule) {
        std::vector<int> out;
        auto deprels = json_set(rule["deprel"]);
        auto head_pos = json_set(rule["head_pos"]);
        const std::string dep_reg = rule["dep_register"], head_reg = rule["head_register"];
        std::set<std::string> exempt;
        if (rule.contains("exempt_head_lexicon"))
            exempt = P.values(json(std::string("@") + rule["exempt_head_lexicon"].get<std::string>()));
        for (int i = 1; i <= n; i++) {
            const Tok& t = toks[i - 1];
            if (!deprels.count(t.base) || !t.head) continue;
            const Tok& h = toks[t.head - 1];
            if (!head_pos.count(h.pos)) continue;
            if (!exempt.empty() && exempt.count(h.lower)) continue;
            auto dr = P.read_of(dep_reg, t.word), hr = P.read_of(head_reg, h.word);
            if (dr.empty() || hr.empty()) continue;     // unknown to the registers = silence
            if (!compatible(dr, hr)) out.push_back(i);
        }
        return out;
    }

    std::vector<int> err_subject_verb(const json& rule) {
        std::vector<int> out;
        const std::string vr = rule["verb_register"], pr = rule["pron_register"],
                          nr = rule["noun_register"];
        for (int i = 1; i <= n; i++) {
            const Tok& t = toks[i - 1];
            if (t.pos != "VERB" && t.pos != "AUX") continue;
            auto verb_r = P.read_of(vr, t.word);
            if (verb_r.empty()) continue;
            bool auxed = false;
            for (int k : kids_of(i))
                if (toks[k - 1].base == "aux" || toks[k - 1].base == "cop") { auxed = true; break; }
            if (auxed) continue;                        // agreement lives on the auxiliary
            int subj = 0;
            for (int k : kids_of(i))
                if (toks[k - 1].base == "nsubj") { subj = k; break; }
            if (!subj) continue;
            const Tok& s = toks[subj - 1];
            std::vector<Pack::Reading> subj_r;
            if (s.pos == "PRON") {
                subj_r = P.read_of(pr, s.word);
            } else if (s.pos == "NOUN" || s.pos == "PROPN") {
                for (auto& r : P.read_of(nr, s.word)) {
                    Pack::Reading rd{"3", r.second};
                    if (std::find(subj_r.begin(), subj_r.end(), rd) == subj_r.end())
                        subj_r.push_back(rd);
                }
            } else {
                continue;
            }
            if (subj_r.empty()) continue;
            if (!compatible(subj_r, verb_r)) out.push_back(i);
        }
        return out;
    }

    std::vector<int> err_case_government(const json& rule) {
        std::vector<int> out;
        const std::string det_reg = rule["det_register"], prep_reg = rule["prep_register"],
                          noun_reg = rule["noun_register"];
        for (int i = 1; i <= n; i++) {
            const Tok& t = toks[i - 1];
            if (t.pos != "ADP" || t.base != "case" || !t.head) continue;
            auto gov = P.read_of(prep_reg, t.word);
            if (gov.empty()) continue;
            const Tok& h = toks[t.head - 1];
            auto noun_r = P.read_of(noun_reg, h.word);
            std::set<std::string> cases;
            bool known = false;
            for (int d : kids_of(t.head)) {
                if (toks[d - 1].base != "det") continue;
                for (auto& r : P.read_of(det_reg, toks[d - 1].word)) {   // ("Case", "Gender")
                    for (auto& nrd : noun_r)
                        if (nrd.first == r.second) { known = true; cases.insert(r.first); }
                }
            }
            if (!known) continue;
            bool sat = false;
            for (auto& g : gov)
                if (cases.count(g.first)) { sat = true; break; }
            if (!sat) out.push_back(i);
        }
        return out;
    }

    std::vector<int> err_verb_position(const json& rule) {
        std::vector<int> out;
        const std::string mode = rule["mode"].get<std::string>();
        const std::string main_role = rule.value("main_role", std::string("MAIN_CLAUSE"));
        std::set<std::string> subord;
        if (rule.contains("subord_lexicon"))
            subord = P.values(json(std::string("@") + rule["subord_lexicon"].get<std::string>()));
        for (auto& [ch, members] : clauses) {
            const std::string role = clause_role(ch);
            const auto& pred = preds.count(ch) ? preds.at(ch) : std::vector<int>{};
            if (pred.empty()) continue;
            std::vector<int> auxes;
            for (int j : pred)
                if (toks[j - 1].pos == "AUX") auxes.push_back(j);
            bool head_verbal = toks[ch - 1].pos == "VERB" || toks[ch - 1].pos == "AUX";
            int v = !auxes.empty() ? auxes.back() : (head_verbal ? ch : 0);
            if (!v) continue;
            if (mode == "final") {
                if (role == main_role) continue;
                bool introduced = false;
                for (int j : members)
                    if (toks[j - 1].base == "mark" && subord.count(toks[j - 1].lower) &&
                        toks[j - 1].head == ch) { introduced = true; break; }
                if (!introduced) continue;
                for (int j : members)
                    if (j > v && toks[j - 1].pos != "PUNCT" &&
                        std::find(pred.begin(), pred.end(), j) == pred.end()) {
                        out.push_back(v);
                        break;
                    }
            } else if (mode == "v2") {
                if (role != main_role || !head_verbal) continue;
                std::vector<int> starts;
                std::set<int> subj_starts;
                for (auto& g : groups[ch]) {
                    if (!g.anchor || g.members.empty()) continue;
                    int first = *std::min_element(g.members.begin(), g.members.end());
                    starts.push_back(first);
                    if (toks[g.anchor - 1].base == "nsubj") subj_starts.insert(first);
                }
                for (int j : lone)
                    if (clause_of.count(j) && clause_of.at(j) == ch) starts.push_back(j);
                if (starts.empty()) continue;
                int first = *std::min_element(starts.begin(), starts.end());
                if (!subj_starts.count(first)) continue;
                bool punct_before = false;
                for (int j : members)
                    if (j < v && toks[j - 1].pos == "PUNCT") { punct_before = true; break; }
                if (punct_before) continue;
                int before = 0;
                for (int f : starts)
                    if (f < v) before++;
                if (before >= 2) out.push_back(v);
            }
        }
        return out;
    }

    std::vector<int> err_form_choice(const json& rule) {
        std::vector<int> out;
        const std::string before_v = utf8_lower(rule["before_vowel"].get<std::string>()),
                          before_c = utf8_lower(rule["before_consonant"].get<std::string>());
        std::set<std::string> exc_v, exc_c;
        if (rule.contains("vowel_exceptions"))
            exc_v = P.values(json(std::string("@") + rule["vowel_exceptions"].get<std::string>()));
        if (rule.contains("consonant_exceptions"))
            exc_c = P.values(json(std::string("@") + rule["consonant_exceptions"].get<std::string>()));
        const std::string vowels = "aeiou";
        for (int i = 1; i < n; i++) {                    // needs a following token
            const std::string& w = toks[i - 1].lower;
            if (w != before_v && w != before_c) continue;
            const std::string& nx = toks[i].lower;
            if (nx.empty() || !std::isalpha((unsigned char)nx[0])) continue;
            bool vowelish = vowels.find(nx[0]) != std::string::npos;
            if (exc_v.count(nx)) vowelish = true;         // hour: vowel sound, consonant letter
            else if (exc_c.count(nx)) vowelish = false;   // university: the other way
            if ((vowelish && w == before_c) || (!vowelish && w == before_v)) out.push_back(i);
        }
        return out;
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
        if (spans && !all.empty()) {
            node["iStart"] = all.front();
            node["iEnd"] = all.back();
            if (!cspans.empty()) {
                // min/max over the tokens that were placed: the ENCLOSING extent,
                // so a node an interrupting clause was lifted out of still covers
                // the interruption, which is what a highlight should paint.
                int lo = -1, hi = -1;
                for (int j : all) {
                    if (cspans[j - 1].first < 0) continue;
                    if (lo < 0 || cspans[j - 1].first < lo) lo = cspans[j - 1].first;
                    if (cspans[j - 1].second > hi) hi = cspans[j - 1].second;
                }
                if (lo >= 0) { node["start"] = lo; node["end"] = hi; }
            }
        }
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
                    json gnode = {{"word", span(g.members)}, {"children", json::array()}};
                    name(gnode, g.role);
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
            name(root, "SENTENCE");
            order_and_span(root);
            return root;
        }
        std::map<int, std::string> role;
        for (int ch : order) {
            role[ch] = clause_role(ch);
            name(nodes[ch], role[ch]);
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

        json root = {{"word", ""}, {"children", json::array()}};
        name(root, "SENTENCE");
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
