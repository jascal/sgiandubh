// The gram kernel — the generative/generalizing half of a sgiandubh expert. Where the distilled FACTS give the
// model's exact decisions on seen contexts, the gram kernel continues *in-domain* text the expert wasn't distilled
// on, at n-gram fidelity, BOUNDED to the corpus: it only emits continuations the corpus n-grams support, and the
// vocab-overlap gate refuses out-of-domain queries. Built offline by tools/build_gram.py; no model, no fieldrun.
#pragma once
#include <cctype>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <set>

struct Gram {
    std::unordered_map<std::string, std::vector<std::pair<std::string, int>>> tab; // joined-prev -> [(next, count)]
    std::set<std::string> vocab;
    int order = 3;
    bool loaded = false;

    // Tokenizer matching tools/build_gram.py: maximal alnum/UTF-8 runs are words; .,;:!? are their own tokens;
    // everything else (space, parens, quotes, hyphens) is a separator.
    static std::vector<std::string> tokenize(const std::string& s) {
        std::vector<std::string> out;
        std::string cur;
        auto flush = [&] { if (!cur.empty()) { out.push_back(cur); cur.clear(); } };
        for (unsigned char b : s) {
            if (std::isalnum(b) || b >= 0x80) cur += (char)b;
            else if (b == '.' || b == ',' || b == ';' || b == ':' || b == '!' || b == '?') { flush(); out.emplace_back(1, (char)b); }
            else flush();
        }
        flush();
        return out;
    }
    static std::string lower(const std::string& w) {
        std::string o = w;
        for (auto& c : o) c = (char)std::tolower((unsigned char)c); // ASCII-only; UTF-8 letters pass through
        return o;
    }

    bool load(const std::string& dir) {
        std::ifstream g(dir + "/grams.tsv");
        if (!g) return false;
        for (std::string line; std::getline(g, line);) {
            auto t1 = line.find('\t'), t2 = line.rfind('\t');
            if (t1 == std::string::npos || t2 == t1) continue;
            tab[line.substr(0, t1)].push_back({line.substr(t1 + 1, t2 - t1 - 1), std::atoi(line.c_str() + t2 + 1)});
        }
        std::ifstream v(dir + "/vocab.txt");
        for (std::string w; std::getline(v, w);) if (!w.empty()) vocab.insert(w);
        loaded = !tab.empty();
        return loaded;
    }

    // The bound: a query is in-domain only if a third of its content words are corpus vocabulary; else → abstain.
    bool in_domain(const std::vector<std::string>& q) const {
        if (vocab.empty()) return false;
        int known = 0, n = 0;
        for (const auto& w : q) { if (w.size() < 2) continue; n++; if (vocab.count(lower(w))) known++; }
        return n > 0 && known * 3 >= n;
    }

    // Backoff-argmax decode: greedy, deterministic; stops on sentence-end or when the corpus has no continuation.
    std::vector<std::string> generate(std::vector<std::string> ctx, int n) const {
        std::vector<std::string> out;
        for (int step = 0; step < n; step++) {
            const std::vector<std::pair<std::string, int>>* best = nullptr;
            for (int o = std::min((int)ctx.size(), order - 1); o >= 0; o--) {
                std::string key;
                for (int k = (int)ctx.size() - o; k < (int)ctx.size(); k++) { if (!key.empty()) key += " "; key += ctx[k]; }
                auto it = tab.find(key);
                if (it != tab.end() && !it->second.empty()) { best = &it->second; break; }
            }
            if (!best) break; // the bound: no corpus continuation
            const std::string* nxt = nullptr; int bc = -1;
            for (const auto& [w, c] : *best) if (c > bc) { bc = c; nxt = &w; }
            out.push_back(*nxt); ctx.push_back(*nxt);
            if (*nxt == "." || *nxt == "!" || *nxt == "?") break;
        }
        return out;
    }
};

// Re-join tokens to text (no space before punctuation).
inline std::string detok(const std::vector<std::string>& ws) {
    std::string s;
    for (const auto& w : ws) {
        bool punct = w.size() == 1 && std::string(".,;:!?").find(w) != std::string::npos;
        if (!s.empty() && !punct) s += " ";
        s += w;
    }
    return s;
}
