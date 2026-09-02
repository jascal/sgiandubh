// neural-expert package: German UD parse + tags served from the fieldrun BERT encoder (bert_ffi) plus
// flat post-encoder head weights (heads.json/.bin) — a faithful C++ port of the supar 1.1.4
// BiaffineDependencyParser inference path (scalar-mix last-4, mean subword pooling, 768x768 projection,
// shared MLPs, biaffine arc/rel scoring, greedy decode with single-root Chu-Liu/Edmonds fallback) and the
// satzklar-model frozen-encoder tagging heads (PREREG_TAGGING_HEADS.md). No torch at runtime.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include "json.hpp"
#include "../bert_ffi/bert_ffi.h"
#include "../tok_ffi/tok_ffi.h"

namespace nexp {

using json = nlohmann::json;
static constexpr float MIN_SCORE = -1e32f;

struct Tensor {
    std::vector<size_t> shape;
    const float* data = nullptr;
    size_t numel() const { size_t n = 1; for (auto s : shape) n *= s; return n; }
};

// Tokenizer configuration read from the grammar pack. Absent (max_enclitics 0 / empty vocab)
// means "behave exactly as before", which is what keeps deployed German byte-identical: its
// package ships grammar.json with no `tokenizer` block, so nothing here ever engages.
//
// Ported from langs.py::_split_enclitics, which is the reference this must match token for token
// (satzklar-model#25, gated by scripts/tok_parity.py).
// The five vowels Spanish writes with an acute, upper and lower. The accent on an enclitic host
// exists only WHILE the clitic is attached (`da` + `me` + `lo` -> `dámelo`), so splitting has to
// take it back off — but only when the accented form is not itself a word.
inline std::string strip_acute(const std::string& s) {
    static const std::pair<const char*, char> MAP[] = {
        {"\xC3\xA1", 'a'}, {"\xC3\xA9", 'e'}, {"\xC3\xAD", 'i'},
        {"\xC3\xB3", 'o'}, {"\xC3\xBA", 'u'},
        {"\xC3\x81", 'A'}, {"\xC3\x89", 'E'}, {"\xC3\x8D", 'I'},
        {"\xC3\x93", 'O'}, {"\xC3\x9A", 'U'}};
    std::string out;
    for (size_t i = 0; i < s.size();) {
        bool hit = false;
        for (auto& m : MAP)
            if (s.compare(i, 2, m.first, 2) == 0) { out += m.second; i += 2; hit = true; break; }
        if (!hit) out += s[i++];
    }
    return out;
}

// Python's str.lower() folds Á -> á; a byte-wise tolower does not, and the reference lowercases
// before every vocabulary lookup, so the two must agree on accented capitals (`Ándale`).
inline std::string lower_utf8(const std::string& s) {
    static const std::pair<const char*, const char*> UP[] = {
        {"\xC3\x81", "\xC3\xA1"}, {"\xC3\x89", "\xC3\xA9"}, {"\xC3\x8D", "\xC3\xAD"},
        {"\xC3\x93", "\xC3\xB3"}, {"\xC3\x9A", "\xC3\xBA"}};
    std::string out;
    for (size_t i = 0; i < s.size();) {
        bool hit = false;
        for (auto& u : UP)
            if (s.compare(i, 2, u.first, 2) == 0) { out += u.second; i += 2; hit = true; break; }
        if (hit) continue;
        unsigned char c = (unsigned char)s[i];
        out += (c & 0x80) ? s[i] : (char)std::tolower(c);
        i++;
    }
    return out;
}

// The reference slices by CHARACTER (`word[:len(stem)]`); bytes would cut an accent in half.
inline std::string take_chars(const std::string& s, size_t n) {
    size_t i = 0, seen = 0;
    while (i < s.size() && seen < n) {
        unsigned char c = (unsigned char)s[i];
        i += (c < 0x80) ? 1 : (c < 0xE0) ? 2 : (c < 0xF0) ? 3 : 4;
        seen++;
    }
    return s.substr(0, i);
}

inline size_t count_chars(const std::string& s) {
    size_t i = 0, n = 0;
    while (i < s.size()) {
        unsigned char c = (unsigned char)s[i];
        i += (c < 0x80) ? 1 : (c < 0xE0) ? 2 : (c < 0xF0) ? 3 : 4;
        n++;
    }
    return n;
}

struct TokCfg {
    std::vector<std::string> enclitics;   // ORDER MATTERS: the reference takes the FIRST match in
                                          // this order, not the longest, so `los` must precede `os`
    int max_enclitics = 0;
    int min_stem = 1;
    std::set<std::string> vocab;          // lowercased surface forms of the language

    // Every table below is EMPTY unless a pack supplied it, and empty means "use the built-in
    // German behaviour". That is what keeps deployed German byte-identical: its package ships no
    // tokenizer block, so each fallback below is the code that ran before this existed.
    std::set<std::string> punct;          // note: langs.py curates this — `_` and `@` are NOT
                                          // punctuation there, while std::ispunct says they are
    std::set<std::string> runnable;
    std::set<std::string> abbrev;
    std::string apostrophes;
    std::set<std::string> clitics;
    bool has_punct = false, has_runnable = false, has_abbrev = false, has_clitics = false;
    bool have_flags = false;              // ordinal_period / initial_period were supplied
    bool ordinal_period = false;
    bool initial_period = false;
    std::vector<std::string> split_suffixes;
    std::vector<std::pair<std::string, std::vector<std::string>>> split_words;
    std::vector<std::string> split_internal;
    std::set<std::string> internal_keep_prefixes;
    int internal_keep_max_left = 0;

    bool enclitics_on() const { return max_enclitics > 0 && !vocab.empty(); }
    bool active() const { return enclitics_on(); }
};

// langs.py::_URLISH — an address is never split on its internal punctuation.
inline bool urlish(const std::string& w) {
    std::string lo;
    for (char c : w) lo += (char)std::tolower((unsigned char)c);
    if (lo.find("http://") != std::string::npos || lo.find("https://") != std::string::npos ||
        lo.find("www.") != std::string::npos || lo.find('@') != std::string::npos)
        return true;
    for (const char* t : {".com", ".org", ".net", ".edu", ".gov", ".htm", ".html"}) {
        size_t k = lo.find(t);
        if (k == std::string::npos) continue;
        size_t end = k + std::strlen(t);
        if (end == lo.size() || !std::isalnum((unsigned char)lo[end])) return true;  // \b
    }
    return false;
}

// langs.py::_split_internal — UD gives many internal hyphens and slashes their own token, except
// after a productive prefix (anti-American) or a short left part (e-mail).
inline std::vector<std::string> split_internal(const std::string& w, const TokCfg& cfg) {
    if (cfg.split_internal.empty() || urlish(w)) return {w};
    auto is_sep = [&](char c) {
        for (auto& s : cfg.split_internal) if (s.size() == 1 && s[0] == c) return true;
        return false;
    };
    bool inner = false;                                   // word[1:-1], so not at either edge
    for (size_t i = 1; i + 1 < w.size(); i++) if (is_sep(w[i])) inner = true;
    if (!inner) return {w};
    std::string seps;
    for (auto& s2 : cfg.split_internal) seps += s2;
    std::string left = w.substr(0, w.find_first_of(seps));
    std::string llo;
    for (char c : left) llo += (char)std::tolower((unsigned char)c);
    if (cfg.internal_keep_prefixes.count(llo) ||
        (int)count_chars(left) <= cfg.internal_keep_max_left)
        return {w};
    std::vector<std::string> out;
    std::string cur;
    for (char c : w) {
        if (is_sep(c)) {
            if (!cur.empty()) { out.push_back(cur); cur.clear(); }
            out.push_back(std::string(1, c));
        } else {
            cur += c;
        }
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

// langs.py::_split_contraction — English written contractions become the words UD annotates.
inline std::vector<std::string> split_contraction(const std::string& w, const TokCfg& cfg) {
    std::string lo = lower_utf8(w);
    for (auto& sw : cfg.split_words) {
        if (lo != sw.first) continue;
        std::vector<std::string> out = sw.second;
        if (!w.empty() && std::isupper((unsigned char)w[0]) && !out[0].empty())
            out[0][0] = (char)std::toupper((unsigned char)out[0][0]);
        return out;
    }
    std::vector<std::string> sufs = cfg.split_suffixes;
    std::sort(sufs.begin(), sufs.end(),
              [](const std::string& a, const std::string& b) { return a.size() > b.size(); });
    for (auto& suf : sufs)
        if (lo.size() > suf.size() && lo.compare(lo.size() - suf.size(), suf.size(), suf) == 0 &&
            (int)count_chars(w) > (int)count_chars(suf) + cfg.min_stem - 1)
            return {w.substr(0, w.size() - suf.size()), w.substr(w.size() - suf.size())};
    return {w};
}

// dárselo -> [dar, se, lo]. Exact port of langs.py::_split_enclitics.
inline std::vector<std::string> split_enclitics(const std::string& word, const TokCfg& cfg) {
    if (!cfg.active()) return {word};
    std::string lo = lower_utf8(word);
    if (cfg.vocab.count(lo)) return {word};                       // already a word: leave it whole
    for (int n = cfg.max_enclitics; n >= 1; n--) {
        std::string stem = lo;
        std::vector<std::string> tail;
        bool ok = true;
        for (int k = 0; k < n; k++) {
            const std::string* hit = nullptr;
            for (auto& c : cfg.enclitics)
                if (stem.size() >= c.size() &&
                    stem.compare(stem.size() - c.size(), c.size(), c) == 0 &&
                    (int)count_chars(stem.substr(0, stem.size() - c.size())) >= cfg.min_stem) {
                    hit = &c;
                    break;                                        // FIRST match, per the reference
                }
            if (!hit) { ok = false; break; }
            tail.push_back(*hit);
            stem.erase(stem.size() - hit->size());
        }
        if (!ok) continue;
        bool plain = cfg.vocab.count(stem) > 0;
        if (!plain && !cfg.vocab.count(strip_acute(stem))) continue;
        std::string base = take_chars(word, count_chars(stem));
        if (!plain) base = strip_acute(base);
        std::vector<std::string> out{base};
        for (auto it = tail.rbegin(); it != tail.rend(); ++it) out.push_back(*it);
        return out;
    }
    return {word};
}

struct Package {
    json meta;
    TokCfg tokcfg;          // empty unless load_tokcfg() was given a pack with a tokenizer block
    std::vector<char> blob;
    std::map<std::string, Tensor> t;
    BertHandle* enc = nullptr;
    Tokenizer* tok = nullptr;
    int d = 0;

    // Read the tokenizer half of the grammar pack. Called separately from load() because the
    // pack is the transform's artifact and is loaded by the server, not by the neural package.
    // A pack without a `tokenizer` block, or one naming no vocab_lexicon, leaves tokcfg inert —
    // which is how deployed German keeps its exact current behaviour.
    void load_tokcfg(const json& pack) {
        if (!pack.contains("tokenizer")) return;
        const json& t = pack["tokenizer"];
        tokcfg.max_enclitics = t.value("max_enclitics", 0);
        tokcfg.min_stem = t.value("min_stem", 1);
        for (auto& e : t.value("enclitics", json::array()))
            tokcfg.enclitics.push_back(e.get<std::string>());
        auto fill = [&](const char* key, std::set<std::string>& dst) {
            for (auto& v : t.value(key, json::array())) dst.insert(v.get<std::string>());
        };
        fill("punct", tokcfg.punct);
        fill("runnable", tokcfg.runnable);
        fill("abbrev", tokcfg.abbrev);
        fill("clitics", tokcfg.clitics);
        // Presence, not emptiness: `clitics: []` means this language HAS no clitics.
        tokcfg.has_punct = t.contains("punct");
        tokcfg.has_runnable = t.contains("runnable");
        tokcfg.has_abbrev = t.contains("abbrev");
        tokcfg.has_clitics = t.contains("clitics");
        fill("internal_keep_prefixes", tokcfg.internal_keep_prefixes);
        tokcfg.apostrophes = t.value("apostrophes", std::string());
        tokcfg.have_flags = t.contains("ordinal_period") || t.contains("initial_period");
        tokcfg.ordinal_period = t.value("ordinal_period", false);
        tokcfg.initial_period = t.value("initial_period", false);
        tokcfg.internal_keep_max_left = t.value("internal_keep_max_left", 0);
        for (auto& v : t.value("split_suffixes", json::array()))
            tokcfg.split_suffixes.push_back(v.get<std::string>());
        for (auto& v : t.value("split_internal", json::array()))
            tokcfg.split_internal.push_back(v.get<std::string>());
        // NOT `t.value(...).items()` in the range expression: .items() returns a proxy over a
        // temporary that dies before the loop body, and the dangling read surfaced as
        // "type must be array, but is null". Bind the object first.
        if (t.contains("split_words") && t["split_words"].is_object()) {
            const json& sw = t["split_words"];
            for (auto it = sw.begin(); it != sw.end(); ++it)
                tokcfg.split_words.emplace_back(it.key(),
                                                it.value().get<std::vector<std::string>>());
        }
        const std::string vl = t.value("vocab_lexicon", std::string());
        if (!vl.empty() && pack.contains("lexicons") && pack["lexicons"].contains(vl)) {
            // A lexicon is either a list of forms (`forms`) or a map keyed by form
            // (`form2lemma`, `noun_gnum`). Either way the FORM is what the guard needs, so an
            // object contributes its keys — iterating it would otherwise collect the values.
            const json& lex = pack["lexicons"][vl];
            if (lex.is_object())
                for (auto it = lex.begin(); it != lex.end(); ++it)
                    tokcfg.vocab.insert(lower_utf8(it.key()));
            else
                for (auto& w : lex)
                    if (w.is_string()) tokcfg.vocab.insert(lower_utf8(w.get<std::string>()));
        }
    }

    bool load(const std::string& dir) {
        std::ifstream mf(dir + "/meta.json");
        if (!mf) return false;
        mf >> meta;
        std::ifstream hj(dir + "/heads.json");
        if (!hj) return false;
        json hidx; hj >> hidx;
        std::ifstream hb(dir + "/heads.bin", std::ios::binary);
        if (!hb) return false;
        blob.assign(std::istreambuf_iterator<char>(hb), {});
        for (auto& e : hidx["tensors"]) {
            Tensor tt;
            for (auto& s : e["shape"]) tt.shape.push_back(s.get<size_t>());
            tt.data = reinterpret_cast<const float*>(blob.data() + e["offset"].get<size_t>());
            t[e["name"].get<std::string>()] = tt;
        }
        enc = be_load((dir + "/gbert").c_str());
        tok = tk_new((dir + "/bundle.tokenizer.json").c_str());
        d = enc ? be_dim(enc) : -1;
        return enc && tok && d > 0;
    }

    // ---- small dense math (all row-major f32) ----------------------------------------------------
    // y[out] = W[out,in] x[in] + b   (torch Linear convention)
    void linear(const Tensor& W, const float* b, const float* x, float* y) const {
        size_t out = W.shape[0], in = W.shape[1];
        for (size_t o = 0; o < out; o++) {
            float acc = b ? b[o] : 0.0f;
            const float* w = W.data + o * in;
            for (size_t i = 0; i < in; i++) acc += w[i] * x[i];
            y[o] = acc;
        }
    }

    // ---- word splitting: edge peeling, mirroring the python spoke exactly ------------------------
    // Rules chosen by measurement against gold UD tokenization, not taste — see sgiandubh#37 and
    // satzklar-model/scripts/tok_eval.py, which scores any candidate over treebank sentences whose
    // raw text is reconstructible (SpaceAfter=No). On 3000 such sentences, exact-sentence accuracy:
    //
    //     peel-anywhere (what this used to do)   86.1%   token F1 0.9759
    //     peel-at-edges (the python spoke)       88.1%   token F1 0.9797
    //     this                                   95.9%   token F1 0.9946
    //
    // Peeling only at token EDGES is what makes 4:20, 1,24, Nr.1 and c't survive as single tokens
    // without any special case, and it is why internal hyphens (Wort-Spiel) need no exception.
    // On top of that, three classes both older tokenizers got wrong:
    //   R1 runs        gold keeps -- (199), ... (244), `` (180), '' (171) whole. Only for the
    //                  marks in RUNNABLE: gold SPLITS repeated ! and ? (406 single vs 5 merged).
    //   R2 abbrev      Dr. Inc. St. ca. bzw. … keep their period (Dr. alone 183x). No sentence-final
    //                  carve-out — A/B'd at 2877 vs 2878 of 3000, so it is not justified.
    //   R3 ordinals    digits + '.' is one token mid-sentence (27x) and NEVER sentence-final
    //                  (0 against 255 split), so the carve-out here IS justified.
    //   R4 clitics     geht's -> geht + 's, c't stays whole (see the apostrophe notes below).
    static std::vector<std::string> split_words(const std::string& text,
                                               const TokCfg* cfg = nullptr) {
        // Punctuation. '-' IS peelable: gold has 1779 bare '-' tokens and only '--' ever ends one.
        // Multi-byte marks are listed explicitly because std::ispunct is byte-wise and ASCII-only.
        static constexpr std::string_view MB_PUNCT[] = {
            "\xC2\xAB", "\xC2\xBB",                          // «  »   (U+00AB / U+00BB)
            "\xE2\x80\x9E", "\xE2\x80\x9C", "\xE2\x80\x9D",  // „  “  ” (U+201E / U+201C / U+201D)
            "\xE2\x80\x9A", "\xE2\x80\x98", "\xE2\x80\x99",  // ‚  ‘  ’ (U+201A / U+2018 / U+2019)
            "\xE2\x80\x93", "\xE2\x80\x94", "\xC2\xA7",      // –  —  § (U+2013 / U+2014 / U+00A7)
            // Spanish opening marks. This table is otherwise German, but these two are safe to
            // list unconditionally: they are punctuation in every language that writes them, they
            // occur 0x in the German treebanks (hdt+gsd), and Spanish gold has them ONLY as
            // standalone tokens (¿ 403x, ¡ 80x in AnCora+GSD, never glued to a word). Without
            // them the leading-peel loop stopped on byte 0 of "¿Qué" — std::ispunct is ASCII-only —
            // and the served expert got a token the parser has never seen (glossa#es-inverted).
            "\xC2\xBF", "\xC2\xA1",                          // ¿  ¡   (U+00BF / U+00A1)
        };
        static const std::set<std::string> RUNNABLE = {
            ".", "`", "'", "-", "\xE2\x80\x93", "\xE2\x80\x94", "\xC2\xA7"};
        static const std::set<std::string> ABBREV = {
            "dr.", "prof.", "st.", "nr.", "inc.", "ca.", "bzw.", "etc.", "usw.", "vgl.", "ggf.",
            "evtl.", "u.a.", "z.b.", "d.h.", "sog.", "inkl.", "exkl.", "max.", "min.", "mio.",
            "mrd.", "jh.", "jhd.", "abb.", "tab.", "s.", "b.", "w.", "v.", "a.", "m.", "hr.",
            "fr.", "co.", "ltd."};
        static const std::set<std::string> CLITICS = {"s", "ne", "nen", "nem", "n"};

        auto is_ascii_punct = [](unsigned char c) { return c < 0x80 && std::ispunct(c); };
        auto lower = [](std::string s) {
            for (char& ch : s)
                if (!((unsigned char)ch & 0x80)) ch = (char)std::tolower((unsigned char)ch);
            return s;
        };
        // Byte length of the punctuation mark starting at i (0 if none).
        auto punct_at = [&](const std::string& s, size_t i) -> size_t {
            if (cfg && cfg->has_punct) {                  // longest UTF-8 mark first
                for (size_t n : {3u, 2u, 1u})
                    if (i + n <= s.size() && cfg->punct.count(s.substr(i, n))) return n;
                return 0;
            }
            for (std::string_view g : MB_PUNCT)
                if (s.compare(i, g.size(), g.data(), g.size()) == 0) return g.size();
            return is_ascii_punct((unsigned char)s[i]) ? 1 : 0;
        };
        // Byte length of the punctuation mark ENDING the string (0 if none).
        auto punct_end = [&](const std::string& s) -> size_t {
            if (cfg && cfg->has_punct) {
                for (size_t n : {3u, 2u, 1u})
                    if (s.size() >= n && cfg->punct.count(s.substr(s.size() - n, n))) return n;
                return 0;
            }
            for (std::string_view g : MB_PUNCT)
                if (s.size() >= g.size() &&
                    s.compare(s.size() - g.size(), g.size(), g.data(), g.size()) == 0)
                    return g.size();
            return (!s.empty() && is_ascii_punct((unsigned char)s.back())) ? 1 : 0;
        };
        auto all_digits_dot = [&](const std::string& s) {         // R3: "31." (langs _ORDINAL)
            if (cfg && cfg->have_flags && !cfg->ordinal_period) return false;
            if (s.size() < 2 || s.back() != '.') return false;
            for (size_t k = 0; k + 1 < s.size(); k++)
                if (!std::isdigit((unsigned char)s[k])) return false;
            return true;
        };
        // langs _INITIAL: a single letter plus a period is an initial (`J. P. Morgan`) and keeps
        // its period. Off unless the pack asks for it, so German is unaffected.
        auto is_initial = [&](const std::string& s) {
            return cfg && cfg->initial_period && s.size() == 2 && s[1] == '.' &&
                   std::isalpha((unsigned char)s[0]);
        };
        // R4: does an apostrophe at k introduce a bare clitic that ends the token?
        auto clitic_at = [&](const std::string& s, size_t k) -> size_t {
            // Which characters count as an apostrophe is per language: German writes ' and ’,
            // English also writes a backtick (`driver`s`). The pack's `apostrophes` string is
            // the authority when present.
            size_t alen = 0;
            const std::string* ap = (cfg && !cfg->apostrophes.empty()) ? &cfg->apostrophes : nullptr;
            if (ap) {
                // ASCII only for the single-byte test: `apostrophes` holds ’ as three UTF-8
                // bytes, and a byte-wise find would match one of its CONTINUATION bytes and cut
                // the character in half — which showed up as invalid UTF-8 on stdout.
                if ((unsigned char)s[k] < 0x80 && ap->find(s[k]) != std::string::npos) alen = 1;
                else if (s.compare(k, 3, "\xE2\x80\x99", 3) == 0 &&
                         ap->find("\xE2\x80\x99") != std::string::npos) alen = 3;
            } else {
                if (s[k] == '\'') alen = 1;
                else if (s.compare(k, 3, "\xE2\x80\x99", 3) == 0) alen = 3;
            }
            if (!alen) return 0;
            std::string rest = lower(s.substr(k + alen));
            return (cfg && cfg->has_clitics ? cfg->clitics.count(rest)
                                            : CLITICS.count(rest)) ? alen : 0;
        };

        std::vector<std::string> words;
        std::vector<std::string> chunks;                          // whitespace split
        for (size_t i = 0; i < text.size();) {
            if (std::isspace((unsigned char)text[i])) { i++; continue; }
            size_t j = i;
            while (j < text.size() && !std::isspace((unsigned char)text[j])) j++;
            chunks.push_back(text.substr(i, j - i));
            i = j;
        }
        for (size_t ci = 0; ci < chunks.size(); ci++) {
            std::string chunk = chunks[ci];
            bool last_chunk = ci + 1 == chunks.size();
            std::vector<std::string> leading, trailing;
            while (!chunk.empty()) {
                size_t plen = punct_at(chunk, 0);
                if (!plen) break;
                if (clitic_at(chunk, 0)) break;                   // 'ne is a token, not a quote
                std::string mark = chunk.substr(0, plen);
                size_t run = plen;
                if (cfg && cfg->has_runnable ? cfg->runnable.count(mark)
                                                 : RUNNABLE.count(mark))          // R1
                    while (run < chunk.size() &&
                           chunk.compare(run, plen, mark) == 0) run += plen;
                if (run == chunk.size()) break;                   // all punctuation: leave whole
                leading.push_back(chunk.substr(0, run));
                chunk.erase(0, run);
            }
            while (!chunk.empty()) {
                size_t plen = punct_end(chunk);
                if (!plen) break;
                if ((cfg && cfg->has_abbrev ? cfg->abbrev.count(lower_utf8(chunk))
                                                 : ABBREV.count(lower(chunk)))) break;   // R2
                if (is_initial(chunk)) break;                     // J. keeps its period
                if (all_digits_dot(chunk) && !(last_chunk && trailing.empty())) break;  // R3
                std::string mark = chunk.substr(chunk.size() - plen);
                size_t run = plen;
                if (cfg && cfg->has_runnable ? cfg->runnable.count(mark)
                                                 : RUNNABLE.count(mark))          // R1
                    while (run + plen <= chunk.size() &&
                           chunk.compare(chunk.size() - run - plen, plen, mark) == 0) run += plen;
                if (run == chunk.size()) break;
                trailing.push_back(chunk.substr(chunk.size() - run));
                chunk.erase(chunk.size() - run);
            }
            for (auto& t : leading) words.push_back(t);
            if (!chunk.empty()) {                                 // R4: geht's -> geht + 's
                size_t cut = 0, alen = 0;
                for (size_t k = 1; k < chunk.size(); k++)
                    if ((alen = clitic_at(chunk, k))) { cut = k; break; }
                // The reference nests these: enclitic splitting runs last, over each piece
                // the earlier splits produced (langs.tokenize's innermost loop).
                // langs.tokenize's nesting, in its order:
                //   _split_internal -> _split_clitic -> _split_contraction -> _split_enclitics
                // The clitic split is the R4 block above; the rest run here.
                auto emit = [&](const std::string& piece) {
                    if (!cfg) { words.push_back(piece); return; }
                    for (auto& a : split_internal(piece, *cfg))
                        for (auto& b : split_contraction(a, *cfg))
                            for (auto& c : split_enclitics(b, *cfg)) words.push_back(c);
                };
                if (cut) {
                    emit(chunk.substr(0, cut));
                    emit(chunk.substr(cut));
                } else {
                    emit(chunk);
                }
            }
            for (auto it = trailing.rbegin(); it != trailing.rend(); ++it) words.push_back(*it);
        }
        return words;
    }

    struct Parse {
        bool abstain = false;
        std::string reason;
        std::vector<std::string> words, deprel, pos, case_, gnn;
        std::vector<int> head;  // 1-based word index, 0 = root
    };

    Parse parse(const std::string& text, const std::vector<std::string>& pretok = {}) const {
        Parse r;
        auto& ab = meta["abstain"];
        r.words = pretok.empty() ? split_words(text, &tokcfg) : pretok;
        size_t n_words = r.words.size();
        if (n_words < ab["min_words"].get<size_t>() || n_words > ab["max_words"].get<size_t>()) {
            r.abstain = true; r.reason = "length"; return r;
        }
        // language gate (the bound IS the router): abstain when the input reads more like the
        // out-of-scope stoplist than the in-scope one. Lists live in the package (meta.json), not here.
        if (meta.contains("lang_gate")) {
            auto hits = [&](const json& lst) {
                size_t h = 0;
                for (auto& w : r.words) {
                    std::string lw = w;
                    for (auto& c : lw) c = (char)std::tolower((unsigned char)c);
                    for (auto& sw : lst)
                        if (lw == sw.get<std::string>()) { h++; break; }
                }
                return h;
            };
            // in-scope vs out-of-scope lists. A German package predates this and
            // names them by language ("de" in, "en" out); a package for any other
            // language says in/out, because an English expert reading the German
            // package's gate would abstain on all of its own input.
            const json& lg = meta["lang_gate"];
            const json* in_ = nullptr;
            const json* out_ = nullptr;
            if (lg.contains("in") && lg.contains("out")) { in_ = &lg["in"]; out_ = &lg["out"]; }
            else if (lg.contains("de") && lg.contains("en")) { in_ = &lg["de"]; out_ = &lg["en"]; }
            if (in_ && out_ && hits(*out_) > hits(*in_)) {
                r.abstain = true; r.reason = "language"; return r;
            }
        }
        // per-word WordPiece (supar SubwordField: no specials, empty -> [UNK], cap fix_len pieces)
        int fix_len = meta["pipeline"]["fix_len"].get<int>();
        uint32_t cls = meta["pipeline"]["cls_id"].get<uint32_t>();
        // [UNK] is not the same id in every vocabulary (gbert 101, bert-base-cased
        // 100): reading it from the package keeps the unknown-fraction abstention
        // honest instead of counting a real subword as unknown.
        uint32_t unk_id = meta["pipeline"].value("unk_id", 101u);
        std::vector<uint32_t> ids = {cls};
        std::vector<std::pair<size_t, size_t>> span;  // [start,end) into ids per word
        size_t unk = 0;
        for (auto& w : r.words) {
            uint32_t buf[64];
            int k = tk_encode(tok, w.c_str(), buf, 64);
            if (k <= 0) { buf[0] = unk_id; k = 1; }
            if (k > fix_len) k = fix_len;
            size_t s = ids.size();
            for (int i = 0; i < k; i++) ids.push_back(buf[i]);
            span.push_back({s, ids.size()});
            for (int i = 0; i < k; i++) if (buf[i] == unk_id) unk++;
        }
        if ((double)unk / (double)(ids.size() - 1) > ab["max_unk_frac"].get<double>()) {
            r.abstain = true; r.reason = "vocabulary"; return r;
        }
        size_t seq = ids.size();

        // encoder: last 4 snapshots -> scalar mix -> mean pool per word (root = CLS) -> projection
        std::vector<float> hs(4 * seq * d);
        if (be_encode(enc, ids.data(), (int)seq, hs.data(), (int)hs.size()) < 0) {
            r.abstain = true; r.reason = "encoder"; return r;
        }
        const Tensor& smw = t.at("scalar_mix_weights");
        float g = t.at("scalar_mix_gamma").data[0];
        float sm[4], mx = *std::max_element(smw.data, smw.data + 4), z = 0;
        for (int k = 0; k < 4; k++) { sm[k] = std::exp(smw.data[k] - mx); z += sm[k]; }
        for (int k = 0; k < 4; k++) sm[k] = g * sm[k] / z;
        std::vector<float> mixed(seq * (size_t)d, 0.0f);
        for (int k = 0; k < 4; k++)
            for (size_t i = 0; i < seq * (size_t)d; i++) mixed[i] += sm[k] * hs[k * seq * d + i];
        size_t n = n_words + 1;  // + root
        std::vector<float> pooled(n * (size_t)d, 0.0f);
        std::copy(mixed.begin(), mixed.begin() + d, pooled.begin());  // root = CLS vector
        for (size_t w = 0; w < n_words; w++) {
            auto [s, e] = span[w];
            for (size_t p = s; p < e; p++)
                for (int i = 0; i < d; i++) pooled[(w + 1) * d + i] += mixed[p * d + i];
            for (int i = 0; i < d; i++) pooled[(w + 1) * d + i] /= (float)(e - s);
        }
        std::vector<float> x(n * (size_t)d);
        for (size_t i = 0; i < n; i++) linear(t.at("projection"), nullptr, &pooled[i * d], &x[i * d]);

        // MLPs (LeakyReLU 0.1) + biaffines
        auto mlp = [&](const char* w, const char* b, size_t dim, std::vector<float>& out) {
            out.resize(n * dim);
            for (size_t i = 0; i < n; i++) {
                linear(t.at(w), t.at(b).data, &x[i * d], &out[i * dim]);
                for (size_t j = 0; j < dim; j++)
                    if (out[i * dim + j] < 0) out[i * dim + j] *= 0.1f;
            }
        };
        std::vector<float> ad, ah, rd, rh;
        mlp("arc_mlp_d_w", "arc_mlp_d_b", 500, ad);
        mlp("arc_mlp_h_w", "arc_mlp_h_b", 500, ah);
        mlp("rel_mlp_d_w", "rel_mlp_d_b", 100, rd);
        mlp("rel_mlp_h_w", "rel_mlp_h_b", 100, rh);

        // s_arc[dep, head] = [ad(dep);1] . A . ah(head)   A = (1,501,500)
        const float* A = t.at("arc_attn").data;
        std::vector<float> M(n * 500);  // [ad;1] @ A
        for (size_t i = 0; i < n; i++)
            for (size_t j = 0; j < 500; j++) {
                float acc = A[500 * 500 + j];  // bias row (the appended 1)
                for (size_t k = 0; k < 500; k++) acc += ad[i * 500 + k] * A[k * 500 + j];
                M[i * 500 + j] = acc;
            }
        std::vector<float> s_arc(n * n);
        for (size_t i = 0; i < n; i++)
            for (size_t j = 0; j < n; j++) {
                float acc = 0;
                for (size_t k = 0; k < 500; k++) acc += M[i * 500 + k] * ah[j * 500 + k];
                s_arc[i * n + j] = acc;
            }

        // greedy heads for deps 1..n-1; single-root CLE fallback when not a valid tree
        std::vector<int> head(n, 0);
        for (size_t i = 1; i < n; i++) {
            size_t best = 0;
            for (size_t j = 1; j < n; j++)
                if (s_arc[i * n + j] > s_arc[i * n + best]) best = j;
            head[i] = (int)best;
        }
        if (!is_tree(head)) head = cle_single_root(s_arc, n);

        // labels: rel biaffine at the chosen head. B = (49,101,101), both sides bias-augmented.
        const auto& rels = meta["vocabs"]["rel"];
        const float* B = t.at("rel_attn").data;
        size_t R = t.at("rel_attn").shape[0];
        r.head.resize(n_words);
        r.deprel.resize(n_words);
        for (size_t i = 1; i < n; i++) {
            size_t h = (size_t)head[i];
            float bestv = -1e30f; size_t bestr = 1;
            std::vector<float> xd(rd.begin() + i * 100, rd.begin() + (i + 1) * 100), xh(rh.begin() + h * 100, rh.begin() + (h + 1) * 100);
            xd.push_back(1.0f); xh.push_back(1.0f);
            for (size_t rr = 0; rr < R; rr++) {
                const float* Br = B + rr * 101 * 101;
                float acc = 0;
                for (size_t a = 0; a < 101; a++) {
                    float row = 0;
                    for (size_t b2 = 0; b2 < 101; b2++) row += Br[a * 101 + b2] * xh[b2];
                    acc += xd[a] * row;
                }
                if (acc > bestv) { bestv = acc; bestr = rr; }
            }
            r.head[i - 1] = head[i];
            r.deprel[i - 1] = rels[bestr].get<std::string>();
        }

        // tagging heads on the same word vectors (positions 1..n-1)
        auto tag = [&](const char* w, const char* b, const json& vocab, std::vector<std::string>& out) {
            size_t C = t.at(w).shape[0];
            std::vector<float> lg(C);
            out.resize(n_words);
            for (size_t i = 1; i < n; i++) {
                linear(t.at(w), t.at(b).data, &x[i * d], lg.data());
                out[i - 1] = vocab[(size_t)(std::max_element(lg.begin(), lg.end()) - lg.begin())].get<std::string>();
            }
        };
        tag("head_pos_w", "head_pos_b", meta["vocabs"]["pos"], r.pos);
        tag("head_morph_case_w", "head_morph_case_b", meta["vocabs"]["morph_case"], r.case_);
        tag("head_morph_gnn_w", "head_morph_gnn_b", meta["vocabs"]["morph_gnn"], r.gnn);
        return r;
    }

    // valid single-root tree over head[1..n-1] (0 = root)?
    static bool is_tree(const std::vector<int>& head) {
        size_t n = head.size();
        int roots = 0;
        for (size_t i = 1; i < n; i++) {
            if (head[i] == (int)i) return false;
            if (head[i] == 0) roots++;
        }
        if (roots != 1) return false;
        for (size_t i = 1; i < n; i++) {  // acyclic: walk to root with a step bound
            size_t cur = i, steps = 0;
            while (cur != 0 && steps++ <= n) cur = (size_t)head[cur];
            if (cur != 0) return false;
        }
        return true;
    }

    // Chu-Liu/Edmonds max spanning arborescence rooted at 0 over s[dep*n+head], with supar's
    // single-root enforcement: if >1 root child, retry per candidate root keeping only its root arc.
    static std::vector<int> cle_single_root(const std::vector<float>& s_arc, size_t n) {
        std::vector<std::vector<float>> s(n, std::vector<float>(n));
        for (size_t i = 0; i < n; i++)
            for (size_t j = 0; j < n; j++) s[i][j] = s_arc[i * n + j];
        for (size_t j = 1; j < n; j++) s[0][j] = MIN_SCORE;  // root has no head
        for (size_t i = 1; i < n; i++) s[i][i] = MIN_SCORE;  // no self loops
        auto h = cle(s, n);
        int roots = 0;
        for (size_t i = 1; i < n; i++) roots += h[i] == 0;
        if (roots <= 1) return h;
        float best = -std::numeric_limits<float>::infinity();
        std::vector<int> bh = h;
        for (size_t rt = 1; rt < n; rt++) {
            if (h[rt] != 0) continue;
            auto s2 = s;
            for (size_t i = 1; i < n; i++)
                if (i != rt) s2[i][0] = MIN_SCORE;
            auto h2 = cle(s2, n);
            float sc = 0;
            for (size_t i = 1; i < n; i++) sc += s[i][(size_t)h2[i]] > MIN_SCORE / 2 ? s[i][(size_t)h2[i]] : 0;
            if (sc > best) { best = sc; bh = h2; }
        }
        return bh;
    }

    // classic greedy-then-contract CLE on score[dep][head]; nodes 0..n-1, root 0.
    static std::vector<int> cle(std::vector<std::vector<float>> s, size_t n) {
        std::vector<int> head(n, 0);
        while (true) {
            for (size_t i = 1; i < n; i++) {
                size_t best = 0;
                for (size_t j = 0; j < n; j++)
                    if (s[i][j] > s[i][best]) best = j;
                head[i] = (int)best;
            }
            // find a cycle
            std::vector<int> color(n, 0);
            std::vector<size_t> cyc;
            for (size_t i = 1; i < n && cyc.empty(); i++) {
                if (color[i]) continue;
                std::vector<size_t> path;
                size_t cur = i;
                while (cur != 0 && color[cur] == 0) { color[cur] = 1; path.push_back(cur); cur = (size_t)head[cur]; }
                if (cur != 0 && color[cur] == 1) {  // found a fresh cycle; extract it
                    auto it = std::find(path.begin(), path.end(), cur);
                    if (it != path.end()) cyc.assign(it, path.end());
                }
                for (auto p : path) color[p] = 2;
            }
            if (cyc.empty()) return head;
            // contract: adjust scores for entering/leaving the cycle, re-run on modified graph
            std::vector<char> inc(n, 0);
            for (auto c : cyc) inc[c] = 1;
            float cyc_sum = 0;
            for (auto c : cyc) cyc_sum += s[c][(size_t)head[c]];
            // representative node = cyc[0]; redirect entering arcs with the standard CLE adjustment
            size_t rep = cyc[0];
            std::vector<size_t> bestin(n, rep);
            for (size_t j = 0; j < n; j++) {
                if (inc[j]) continue;
                float bv = -std::numeric_limits<float>::infinity();
                for (auto c : cyc) {
                    float v = s[c][j] + cyc_sum - s[c][(size_t)head[c]];
                    if (v > bv) { bv = v; bestin[j] = c; }
                }
                s[rep][j] = bv;
            }
            for (size_t j = 0; j < n; j++) {  // arcs leaving the cycle: best member as source
                if (inc[j] || j == 0) continue;
                float bv = -std::numeric_limits<float>::infinity();
                size_t bs = rep;
                for (auto c : cyc) if (s[j][c] > bv) { bv = s[j][c]; bs = c; }
                s[j][rep] = bv;
                s[j][bs] = bv;  // keep original best for reconstruction
            }
            for (auto c : cyc)
                if (c != rep) { for (size_t j = 0; j < n; j++) { s[c][j] = MIN_SCORE; s[j][c] = j == 0 ? s[j][c] : MIN_SCORE; } }
            // re-run; break the cycle at the member whose head-arc is replaced by the chosen entering arc
            std::vector<int> h2(n, 0);
            for (size_t i = 1; i < n; i++) {
                size_t best = 0;
                for (size_t j = 0; j < n; j++)
                    if (s[i][j] > s[i][best]) best = j;
                h2[i] = (int)best;
            }
            size_t entry = bestin[(size_t)h2[rep]];
            head[entry] = h2[rep] == (int)rep ? head[entry] : h2[rep];
            // simple fallback: recompute greedily with the cycle broken at `entry`
            s[entry][(size_t)head[cyc[0] == entry ? cyc.back() : cyc[0]]] = MIN_SCORE;
        }
    }
};

}  // namespace nexp
