// sgiandubh — a small, standalone, OpenAI-compatible server over a Soufflé-compiled bounded expert.
//
// No model, no GPU, no fieldrun at runtime. Per request it: (1) lexically matches the query against the extracted
// decision set, (2) runs the compiled Datalog engine to pick the answer, (3) replies in OpenAI /v1/chat/completions
// shape — or abstains when nothing is in scope. The model's competence was distilled offline (by fieldrun) into the
// `package/` facts; serving is match + a native semiring combine.
//
// SCAFFOLD NOTE: this shells out to the compiled `engine` binary per request (process spawn). Correct + simple, but
// the spawn is the one non-"fast" bit — the perf upgrade is to embed the generated SouffleProgram in-process (one
// binary, no spawn). Tracked as the next increment.
#include "httplib.h"
#include "json.hpp"
#include <array>
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <fstream>
#include <set>
#include <string>
#include <vector>
using json = nlohmann::json;

struct Item {
    std::string id, question, citation, facts;
    std::vector<std::string> options;
};
static std::vector<Item> g_items;
static std::string g_pkg, g_engine, g_model;
static double g_tau = 0.12; // abstain threshold (lexical Jaccard) — the bound: below this, off-material → abstain

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

// Run the compiled Datalog engine on a facts dir; return the decided candidate id (-1 on failure).
static int run_engine(const std::string& facts_dir) {
    std::string cmd = g_engine + " -F " + facts_dir + " -D - 2>/dev/null";
    FILE* p = popen(cmd.c_str(), "r");
    if (!p) return -1;
    std::array<char, 256> buf;
    int decided = -1;
    while (fgets(buf.data(), buf.size(), p)) {
        std::string line(buf.data()), t;
        for (char c : line) if (!std::isspace((unsigned char)c)) t += c;
        if (t.empty()) continue;
        try { decided = std::stoi(t); break; } catch (...) { /* header/divider line */ }
    }
    pclose(p);
    return decided;
}

static json completion(const std::string& content, const std::string& finish, int ptoks) {
    json choice;
    choice["index"] = 0;
    choice["message"] = {{"role", "assistant"}, {"content", content}};
    choice["finish_reason"] = finish;
    json r;
    r["id"] = "chatcmpl-sgiandubh";
    r["object"] = "chat.completion";
    r["model"] = g_model;
    r["choices"] = json::array({choice});
    r["usage"] = {{"prompt_tokens", ptoks}, {"completion_tokens", 0}, {"total_tokens", ptoks}};
    return r;
}

int main(int argc, char** argv) {
    g_pkg = argc > 1 ? argv[1] : "package";
    g_engine = argc > 2 ? argv[2] : "build/engine";
    int port = argc > 3 ? std::stoi(argv[3]) : 8080;

    std::ifstream f(g_pkg + "/index.json");
    if (!f) { fprintf(stderr, "sgiandubh: cannot open %s/index.json\n", g_pkg.c_str()); return 1; }
    json idx; f >> idx;
    g_model = idx.value("model", "sgiandubh");
    for (const auto& it : idx["items"]) {
        Item x;
        x.id = it.value("id", "");
        x.question = it.value("question", "");
        x.citation = it.value("citation", "");
        x.facts = it.value("facts", "");
        if (it.contains("options")) for (const auto& o : it["options"]) x.options.push_back(o.get<std::string>());
        g_items.push_back(x);
    }
    fprintf(stderr, "sgiandubh: %zu items · model=%s · engine=%s · listening :%d\n",
            g_items.size(), g_model.c_str(), g_engine.c_str(), port);

    httplib::Server svr;

    svr.Get("/v1/models", [](const httplib::Request&, httplib::Response& res) {
        json e; e["id"] = g_model; e["object"] = "model"; e["owned_by"] = "sgiandubh";
        json m; m["object"] = "list"; m["data"] = json::array({e});
        res.set_content(m.dump(), "application/json");
    });

    svr.Post("/v1/chat/completions", [](const httplib::Request& req, httplib::Response& res) {
        json body;
        try { body = json::parse(req.body); }
        catch (...) { res.status = 400; res.set_content("{\"error\":\"invalid json\"}", "application/json"); return; }
        std::string user;
        if (body.contains("messages"))
            for (const auto& m : body["messages"]) if (m.value("role", "") == "user") user = m.value("content", "");

        auto qw = words(user);
        double best = 0.0;
        const Item* hit = nullptr;
        for (const auto& it : g_items) {
            double s = jaccard(qw, words(it.question));
            if (s > best) { best = s; hit = &it; }
        }

        std::string content;
        if (hit && best >= g_tau) {
            int d = run_engine(g_pkg + "/" + hit->facts);
            if (d >= 0 && d < (int)hit->options.size())
                content = "The answer is " + hit->options[d] + ".\n\n\xF0\x9F\x93\x96 Source: " + hit->citation;
            else
                content = "(engine error)";
        } else {
            // the bound: nothing in scope → abstain rather than confabulate
            content = "That isn't covered in this material. Try rephrasing, or ask your teacher.";
        }
        res.set_content(completion(content, "stop", (int)qw.size()).dump(), "application/json");
    });

    svr.listen("0.0.0.0", port);
    return 0;
}
