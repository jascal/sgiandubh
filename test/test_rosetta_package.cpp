// test_rosetta_package.cpp — unit test for the rosetta expert-package consumer (src/rosetta_package.h).
// No model, no tokenizer, no souffle: writes a synthetic manifest.json (one of each tier), loads it, and asserts
// the serve() decision on hand-built contexts — trusted idioms (gate, compose) beat gated n-grams (longest suffix),
// uncovered contexts ABSTAIN. Build+run:  ./test/run.sh
#include "rosetta_package.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>

using rosetta::Ctx;
using rosetta::Package;

// A manifest exercising every branch of Package::load / serve.
//  gate    : frame {2-back == 100}, content slot = last token; 7->42, 8->43
//  compose : frame {3-back == 200}, operands (2-back, 1-back) valmap 5->1 6->2, sum 3->70
//  ngram   : [11,12]->99  and  [12]->88   (longest-suffix wins)
static const char* MANIFEST = R"JSON({
  "schema_version": 3,
  "alpha": 2.0,
  "n_rules": 4,
  "rules": [
    {"kind":"gate","id":1,"frame":{"2":100},"slot":1,"table":{"7":42,"8":43},"counts":{"7":[5,6]},"citation":["corpus@5"]},
    {"kind":"compose","id":2,"frame":{"3":200},"operands":[2,1],"valmap":{"5":1,"6":2},"sum":{"3":70},"citation":["corpus@9"]},
    {"kind":"ngram","ctx":[11,12],"out":99,"basis":"observational","counts":[7,9],"cite":[22]},
    {"kind":"ngram","ctx":[12],"out":88,"basis":"observational","cite":[23]}
  ]
})JSON";

static int served(const Package& p, const Ctx& ctx, const char* tier, const char* basis) {
    auto d = p.serve(ctx);
    assert(d.has_value() && "expected an answer, got ABSTAIN");
    assert(d->tier == tier);
    assert(d->basis == basis);
    return d->answer;
}

int main() {
    const std::string path = "/tmp/rosetta_test_manifest.json";
    { std::ofstream f(path); f << MANIFEST; }

    Package p = Package::load(path);
    assert(p.n_rules == 4);
    assert(p.idioms.size() == 2);
    assert(p.W == 2);
    assert(p.schema_version == 3);
    assert(std::abs(p.alpha - 2.0) < 1e-9);
    assert(p.idioms[0].counts.at(7) == std::make_pair(5L, 6L));
    assert(p.idioms[0].counts.count(8) == 0);                  // absent is not the pair (0, 0)
    assert(p.ngrams[2].at(Ctx{11, 12}).cnt == 7);
    assert(p.ngrams[2].at(Ctx{11, 12}).tot == 9);
    assert(p.ngrams[1].at(Ctx{12}).cnt == -1);                // old package/rule default
    assert(p.ngrams[1].at(Ctx{12}).tot == -1);

    // gate: ...,100,7 -> 42 ; ...,100,8 -> 43  (trusted/causal beats any n-gram)
    assert(served(p, {1, 100, 7}, "trusted", "causal") == 42);
    assert(served(p, {1, 100, 8}, "trusted", "causal") == 43);
    // gate frame must match: 2-back != 100 -> gate skipped
    assert(!p.serve(Ctx{1, 99, 7}).has_value());

    // compose: ...,200,5,6 -> sum[valmap[5]+valmap[6]] = sum[3] = 70
    assert(served(p, {200, 5, 6}, "trusted", "causal") == 70);
    // compose with an operand not in valmap -> skipped -> abstain
    assert(!p.serve(Ctx{200, 5, 99}).has_value());
    // compose: ctx too short for the frame reach -> abstain, no out-of-bounds read
    assert(!p.serve(Ctx{5, 6}).has_value());

    // n-gram longest-suffix: [...,11,12] -> 99 ; [...,x,12] (x!=11) -> 88
    assert(served(p, {0, 11, 12}, "gated", "observational") == 99);
    assert(served(p, {0, 9, 12}, "gated", "observational") == 88);

    // citations propagate
    assert(p.serve(Ctx{1, 100, 7})->citation == "corpus@5");
    assert(p.serve(Ctx{0, 11, 12})->citation == "corpus@22");

    // uncovered -> ABSTAIN
    assert(!p.serve(Ctx{500, 501}).has_value());
    assert(!p.serve(Ctx{}).has_value());

    // --- decode_facts: the C++ semiring decode (engine.dl in C++) = logit(T)=Σ contrib; decide=argmax ---
    {
        namespace fs = std::filesystem;
        const std::string dir = "/tmp/rosetta_decode_test";
        fs::create_directories(dir);
        { std::ofstream f(dir + "/candidate.facts"); f << "10\n20\n30\n"; }          // 30 has no contrib → logit 0
        { std::ofstream f(dir + "/contrib.facts");                                   // 10 → 0.5+0.5=1.0, 20 → 2.0
          f << "L0.attn\t10\t0.5\nL1.ffn\t10\t0.5\nL0.attn\t20\t2.0\n"; }
        auto fd = rosetta::decode_facts(dir);
        assert(fd.decide == 20);                                                     // argmax (2.0)
        assert(fd.logits.size() == 3);                                               // a logit per candidate incl. contrib-less 30
        std::map<int, double> got;
        for (auto& kv : fd.logits) got[kv.first] = kv.second;
        assert(std::abs(got[10] - 1.0) < 1e-9);
        assert(std::abs(got[20] - 2.0) < 1e-9);
        assert(std::abs(got[30] - 0.0) < 1e-9);
        // missing dir / no candidates -> decide=-1, empty logits (the distilled tier then yields null logprobs)
        auto none = rosetta::decode_facts("/tmp/rosetta_decode_does_not_exist");
        assert(none.decide == -1 && none.logits.empty());
        fs::remove_all(dir);
    }

    std::remove(path.c_str());
    std::printf("test_rosetta_package: OK (gate, compose, n-gram longest-suffix, abstain, citations, decode_facts)\n");
    return 0;
}
