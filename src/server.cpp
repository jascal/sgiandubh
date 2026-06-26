// sgiandubh — a small, standalone, OpenAI-compatible server over a Soufflé-compiled bounded expert.
//
// No model, no GPU, no fieldrun at runtime. Per request it: (1) lexically matches the query against the extracted
// decision set, (2) runs the EMBEDDED Datalog engine (in-process, no spawn) to derive the decision + per-candidate
// logits, (3) replies in OpenAI /v1/chat/completions shape — grounded in the owner's content (the verbatim
// supporting passage + section), with `logprobs` (confidence + distractor mass) — or abstains when out of scope.
// With --require-citation it abstains on any answer it can't ground or cite (the strong regulated/high-stakes mode).
//
// The engine is the Soufflé-generated class for src/engine.dl, linked in (-D__EMBEDDED_SOUFFLE__). Built by build.sh.
#include "httplib.h"
#include "json.hpp"
#include "gram.h"
#include "souffle/SouffleInterface.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
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
static std::mutex g_engine_mu; // the embedded engine is a single stateful instance (purge/loadAll/run) — serialize it
static Gram g_gram; // generative fallback (n-gram + induction), loaded if package/gram/ exists

// Grounding: the owner's corpus passages, so every answer can carry the verbatim source it's supported by.
struct Passage { std::string section, text; std::set<std::string> w; std::vector<float> vec; };
static std::vector<Passage> g_knowledge;  // loaded from package/knowledge.tsv (optional)
static std::unordered_map<std::string, std::vector<float>> g_wordvec; // corpus word embeddings (package/wordvec.txt)
static int g_dim = 0;                      // embedding dim (>0 once wordvec.txt is loaded → cosine grounding)
static double g_ground_tau = 0.10;        // min lexical overlap to ground (lexical fallback)
static double g_cos_tau = 0.35;           // min cosine to ground (vector path)
static bool g_require_citation = false;   // --require-citation: refuse any answer that can't be grounded/cited

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

// Corpus word embeddings (package/wordvec.txt: `word v0 .. vD`). Optional — enables cosine grounding.
static void load_wordvec(const std::string& path) {
    std::ifstream f(path);
    if (!f) return;
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream ss(line);
        std::string w;
        if (!(ss >> w)) continue;
        std::vector<float> v;
        float x;
        while (ss >> x) v.push_back(x);
        if (v.empty()) continue;
        if (g_dim == 0) g_dim = (int)v.size();
        if ((int)v.size() == g_dim) g_wordvec.emplace(std::move(w), std::move(v));
    }
}
// Mean of the in-vocabulary words' vectors, L2-normalized (empty if no word is known).
static std::vector<float> embed(const std::set<std::string>& ws) {
    if (g_dim <= 0) return {};
    std::vector<float> v(g_dim, 0.0f);
    int n = 0;
    for (const auto& w : ws) {
        auto it = g_wordvec.find(w);
        if (it == g_wordvec.end()) continue;
        for (int i = 0; i < g_dim; i++) v[i] += it->second[i];
        n++;
    }
    if (n == 0) return {};
    double nrm = 0;
    for (float x : v) nrm += (double)x * x;
    nrm = std::sqrt(nrm);
    if (nrm > 0) for (float& x : v) x = (float)(x / nrm);
    return v;
}
// Grounding passages (package/knowledge.tsv: `section <TAB> passage`). The owner's content, verbatim.
static void load_knowledge(const std::string& path) {
    std::ifstream f(path);
    if (!f) return;
    std::string line;
    while (std::getline(f, line)) {
        auto tab = line.find('\t');
        if (tab == std::string::npos) continue;
        Passage p;
        p.section = line.substr(0, tab);
        p.text = line.substr(tab + 1);
        p.w = words(p.text);
        if (p.w.empty()) continue;
        p.vec = embed(p.w);  // precompute the passage embedding (empty if no vectors loaded)
        g_knowledge.push_back(std::move(p));
    }
}
// Best supporting passage for a cue (query + answer words): cosine over embeddings if loaded, else lexical Jaccard.
static const Passage* ground(const std::set<std::string>& cue) {
    if (g_dim > 0) {
        std::vector<float> qv = embed(cue);
        if (!qv.empty()) {
            const Passage* best = nullptr;
            double bs = g_cos_tau;
            for (const auto& p : g_knowledge) {
                if ((int)p.vec.size() != g_dim) continue;
                double s = 0;
                for (int i = 0; i < g_dim; i++) s += (double)qv[i] * p.vec[i];
                if (s > bs) { bs = s; best = &p; }
            }
            return best;
        }
    }
    const Passage* best = nullptr;
    double bs = g_ground_tau;
    for (const auto& p : g_knowledge) {
        double s = jaccard(cue, p.w);
        if (s > bs) { bs = s; best = &p; }
    }
    return best;
}

// Run the EMBEDDED Datalog engine on a facts dir, in-process (no spawn). Returns the decided id + per-candidate logits.
static Decision run_engine(const std::string& facts_dir) {
    Decision r;
    if (!g_prog) return r;
    std::lock_guard<std::mutex> lk(g_engine_mu); // shared stateful engine → one decode at a time (microseconds)
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

// One OpenAI streaming chunk (object "chat.completion.chunk").
static json stream_chunk(const json& delta, const char* finish, const json& lp) {
    json ch;
    ch["index"] = 0;
    ch["delta"] = delta;
    ch["finish_reason"] = finish ? json(finish) : json(nullptr);
    if (!lp.is_null()) ch["logprobs"] = lp;
    json c;
    c["id"] = "chatcmpl-sgiandubh";
    c["object"] = "chat.completion.chunk";
    c["model"] = g_model;
    c["choices"] = json::array({ch});
    return c;
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
    std::vector<std::string> pos;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--require-citation") g_require_citation = true;
        else pos.push_back(a);
    }
    g_pkg = pos.size() > 0 ? pos[0] : "package";
    int port = pos.size() > 1 ? std::stoi(pos[1]) : 8080;

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
    g_gram.load(g_pkg + "/gram"); // optional generative fallback
    load_wordvec(g_pkg + "/wordvec.txt");      // optional corpus embeddings (enables cosine grounding)
    load_knowledge(g_pkg + "/knowledge.tsv");  // optional grounding passages (vectors computed here)
    const char* gmode = g_knowledge.empty() ? "off" : (g_dim > 0 ? "vector" : "lexical");
    fprintf(stderr, "sgiandubh: %zu items · model=%s · embedded engine · gram-kernel=%s · grounding=%s%s%s · listening :%d\n",
            g_items.size(), g_model.c_str(), g_gram.loaded ? "on" : "off", gmode,
            g_dim > 0 ? ("/" + std::to_string(g_dim) + "d").c_str() : "",
            g_require_citation ? " · require-citation" : "", port);

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

        static const std::string ABSTAIN = "That isn't covered in this material. Try rephrasing, or ask your teacher.";
        std::string answer_body, item_cite;
        bool is_answer = false, is_generated = false;
        json lp = nullptr;
        if (hit && best >= g_tau) {
            item_cite = hit->citation;
            Decision d = hit->facts.empty() ? Decision{} : run_engine(g_pkg + "/" + hit->facts);
            lp = logprobs_of(d, *hit);
            if (!hit->answer.empty())
                answer_body = hit->answer;                                          // generated answer (faithful)
            else if (!hit->options.empty() && d.decide >= 0 && d.decide < (int)hit->options.size())
                answer_body = "The answer is " + hit->options[d.decide] + ".";       // MC: engine decides live
            else
                answer_body = (d.decide >= 0 ? "decide=" + std::to_string(d.decide) : "(engine error)");
            is_answer = true;
        } else if (g_gram.loaded) {
            // no distilled answer → generative fallback, bounded to the corpus (gram kernel); abstain if off-domain
            auto toks = Gram::tokenize(user);
            std::vector<std::string> cont = g_gram.in_domain(toks) ? g_gram.generate(toks, 24) : std::vector<std::string>{};
            if (!cont.empty()) { answer_body = detok(cont); is_answer = true; is_generated = true; }
        }

        std::string content;
        if (is_answer) {
            // Ground the answer in the owner's content: attach the best supporting passage verbatim (+ section).
            std::set<std::string> cue = qw;
            auto aw = words(answer_body);
            cue.insert(aw.begin(), aw.end());
            const Passage* gp = g_knowledge.empty() ? nullptr : ground(cue);
            std::string prov;
            if (gp)
                prov = "\n\n\xF0\x9F\x93\x96 From the material" +
                       (gp->section.empty() ? std::string() : " (\xC2\xA7" + gp->section + ")") +
                       ": \"" + gp->text + "\"";
            else if (!item_cite.empty())
                prov = "\n\n\xF0\x9F\x93\x96 Source: " + item_cite;
            bool cited = (gp != nullptr) || !item_cite.empty();
            if (g_require_citation && !cited) {
                content = ABSTAIN;                                                   // no provenance → refuse
                lp = nullptr;
            } else {
                content = answer_body + prov + (is_generated && !gp ? "\n\n(generated from the material)" : "");
            }
        } else {
            content = ABSTAIN;                                                       // the bound
        }
        if (!body.value("stream", false)) {
            res.set_content(completion(content, lp, (int)qw.size()).dump(), "application/json");
            return;
        }
        // OpenAI SSE streaming: role chunk → content word-pieces → final (finish + logprobs) → [DONE]
        res.set_chunked_content_provider(
            "text/event-stream",
            [content, lp](size_t, httplib::DataSink& sink) {
                auto sse = [&](const json& j) { std::string s = "data: " + j.dump() + "\n\n"; sink.write(s.data(), s.size()); };
                sse(stream_chunk(json{{"role", "assistant"}}, nullptr, json(nullptr)));
                size_t i = 0;
                while (i < content.size()) {
                    size_t sp = content.find(' ', i);
                    std::string piece = (sp == std::string::npos) ? content.substr(i) : content.substr(i, sp - i + 1);
                    sse(stream_chunk(json{{"content", piece}}, nullptr, json(nullptr)));
                    i = (sp == std::string::npos) ? content.size() : sp + 1;
                }
                sse(stream_chunk(json::object(), "stop", lp));
                std::string done = "data: [DONE]\n\n";
                sink.write(done.data(), done.size());
                sink.done();
                return true;
            });
    });

    svr.listen("0.0.0.0", port);
    return 0;
}
