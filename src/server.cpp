// sgiandubh — a small, standalone, OpenAI-compatible server over a Soufflé-compiled bounded expert.
//
// No model, no GPU, no fieldrun at runtime. Per request it: (1) lexically matches the query against the extracted
// decision set, (2) runs the EMBEDDED Datalog engine (in-process, no spawn) to derive the decision + per-candidate
// logits, (3) replies in OpenAI /v1/chat/completions shape — with a citation, and `logprobs` (calibrated confidence
// + distractor mass, from the semiring's per-candidate logits) — or abstains when nothing is in scope.
//
// The engine is the Soufflé-generated class for src/engine.dl, linked in (-D__EMBEDDED_SOUFFLE__). Built by build.sh.
#include "httplib.h"
#include "json.hpp"
#include "souffle/SouffleInterface.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <set>
#include <string>
#include <vector>
using json = nlohmann::json;

struct Item {
    std::string id, question, citation, facts, answer;
    std::vector<std::string> options;
};
struct Decision {
    int decide = -1;
    std::vector<std::pair<int, double>> logits; // (candidate id, logit)
};

static std::vector<Item> g_items;
static std::string g_pkg, g_model;
static double g_tau = 0.12; // abstain threshold (lexical Jaccard) — the bound: below this → abstain
static souffle::SouffleProgram* g_prog = nullptr;

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

// Run the EMBEDDED Datalog engine on a facts dir, in-process (no spawn). Returns the decided id + per-candidate logits.
static Decision run_engine(const std::string& facts_dir) {
    Decision r;
    if (!g_prog) return r;
    g_prog->purgeInputRelations();
    g_prog->purgeInternalRelations();
    g_prog->purgeOutputRelations();
    g_prog->loadAll(facts_dir);
    g_prog->run();
    for (auto& t : *g_prog->getRelation("decide")) { souffle::RamSigned v; t >> v; r.decide = (int)v; }
    for (auto& t : *g_prog->getRelation("logit")) {
        souffle::RamSigned id; souffle::RamFloat s; t >> id >> s;
        r.logits.emplace_back((int)id, (double)s);
    }
    return r;
}

// OpenAI logprobs (chat schema): the decided token + top_logprobs over the candidates (softmax of the logits).
static json logprobs_of(const Decision& d, const Item& it) {
    if (d.logits.empty()) return nullptr;
    double m = -1e300;
    for (auto& [id, s] : d.logits) m = std::max(m, s);
    double Z = 0;
    for (auto& [id, s] : d.logits) Z += std::exp(s - m);
    double lnZ = std::log(Z);
    auto label = [&](int id) -> std::string {
        if (!it.options.empty() && id >= 0 && id < (int)it.options.size()) return it.options[id];
        return "tok:" + std::to_string(id);
    };
    json top = json::array();
    double declp = 0;
    for (auto& [id, s] : d.logits) {
        double lp = (s - m) - lnZ;
        json e; e["token"] = label(id); e["logprob"] = lp;
        top.push_back(e);
        if (id == d.decide) declp = lp;
    }
    json entry; entry["token"] = label(d.decide); entry["logprob"] = declp; entry["top_logprobs"] = top;
    json lp; lp["content"] = json::array({entry});
    return lp;
}

static json completion(const std::string& content, const json& logprobs, int ptoks) {
    json choice;
    choice["index"] = 0;
    choice["message"] = {{"role", "assistant"}, {"content", content}};
    choice["finish_reason"] = "stop";
    choice["logprobs"] = logprobs; // null when not applicable
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
    int port = argc > 2 ? std::stoi(argv[2]) : 8080;

    g_prog = souffle::ProgramFactory::newInstance("engine");
    if (!g_prog) { fprintf(stderr, "sgiandubh: embedded engine 'engine' not found\n"); return 1; }

    std::ifstream f(g_pkg + "/index.json");
    if (!f) { fprintf(stderr, "sgiandubh: cannot open %s/index.json\n", g_pkg.c_str()); return 1; }
    json idx; f >> idx;
    g_model = idx.value("model", "sgiandubh");
    for (const auto& it : idx["items"]) {
        Item x;
        x.id = it.value("id", "");
        x.question = it.contains("query") ? it.value("query", "") : it.value("question", "");
        x.citation = it.value("citation", "");
        x.facts = it.value("facts", "");
        x.answer = it.value("answer", "");
        if (it.contains("options")) for (const auto& o : it["options"]) x.options.push_back(o.get<std::string>());
        g_items.push_back(x);
    }
    fprintf(stderr, "sgiandubh: %zu items · model=%s · embedded engine · listening :%d\n",
            g_items.size(), g_model.c_str(), port);

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
        json lp = nullptr;
        if (hit && best >= g_tau) {
            const std::string cite = hit->citation.empty() ? "" : ("\n\n\xF0\x9F\x93\x96 Source: " + hit->citation);
            Decision d = hit->facts.empty() ? Decision{} : run_engine(g_pkg + "/" + hit->facts);
            lp = logprobs_of(d, *hit);
            if (!hit->answer.empty())
                content = hit->answer + cite;                                   // generated answer (faithful)
            else if (!hit->options.empty() && d.decide >= 0 && d.decide < (int)hit->options.size())
                content = "The answer is " + hit->options[d.decide] + "." + cite; // MC: engine decides live
            else
                content = (d.decide >= 0 ? "decide=" + std::to_string(d.decide) : "(engine error)") + cite;
        } else {
            content = "That isn't covered in this material. Try rephrasing, or ask your teacher."; // the bound
        }
        res.set_content(completion(content, lp, (int)qw.size()).dump(), "application/json");
    });

    svr.listen("0.0.0.0", port);
    return 0;
}
