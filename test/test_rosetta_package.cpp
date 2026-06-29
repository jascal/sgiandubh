// test_rosetta_package.cpp — unit test for the rosetta expert-package consumer (src/rosetta_package.h).
// No model, no tokenizer, no souffle: writes a synthetic manifest.json (one of each tier), loads it, and asserts
// the serve() decision on hand-built contexts — trusted idioms (gate, compose) beat gated n-grams (longest suffix),
// uncovered contexts ABSTAIN. Build+run:  ./test/run.sh
#include "rosetta_package.h"
#include <cassert>
#include <cstdio>
#include <fstream>

using rosetta::Ctx;
using rosetta::Package;

// A manifest exercising every branch of Package::load / serve.
//  gate    : frame {2-back == 100}, content slot = last token; 7->42, 8->43
//  compose : frame {3-back == 200}, operands (2-back, 1-back) valmap 5->1 6->2, sum 3->70
//  ngram   : [11,12]->99  and  [12]->88   (longest-suffix wins)
static const char* MANIFEST = R"JSON({
  "n_rules": 4,
  "rules": [
    {"kind":"gate","id":1,"frame":{"2":100},"slot":1,"table":{"7":42,"8":43},"citation":["corpus@5"]},
    {"kind":"compose","id":2,"frame":{"3":200},"operands":[2,1],"valmap":{"5":1,"6":2},"sum":{"3":70},"citation":["corpus@9"]},
    {"kind":"ngram","ctx":[11,12],"out":99,"basis":"observational","cite":[22]},
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

    // gate: ...,100,7 -> 42 ; ...,100,8 -> 43  (trusted/causal beats any n-gram)
    assert(served(p, {1, 100, 7}, "trusted", "causal") == 42);
    assert(served(p, {1, 100, 8}, "trusted", "causal") == 43);
    // gate frame must match: 2-back != 100 -> gate skipped
    assert(!p.serve(Ctx{1, 99, 7}).has_value());

    // compose: ...,200,5,6 -> sum[valmap[5]+valmap[6]] = sum[3] = 70
    assert(served(p, {200, 5, 6}, "trusted", "causal") == 70);
    // compose with an operand not in valmap -> skipped -> abstain
    assert(!p.serve(Ctx{200, 5, 99}).has_value());

    // n-gram longest-suffix: [...,11,12] -> 99 ; [...,x,12] (x!=11) -> 88
    assert(served(p, {0, 11, 12}, "gated", "observational") == 99);
    assert(served(p, {0, 9, 12}, "gated", "observational") == 88);

    // citations propagate
    assert(p.serve(Ctx{1, 100, 7})->citation == "corpus@5");
    assert(p.serve(Ctx{0, 11, 12})->citation == "corpus@22");

    // uncovered -> ABSTAIN
    assert(!p.serve(Ctx{500, 501}).has_value());
    assert(!p.serve(Ctx{}).has_value());

    std::remove(path.c_str());
    std::printf("test_rosetta_package: OK (gate, compose, n-gram longest-suffix, abstain, citations)\n");
    return 0;
}
