// test_split_words.cpp — unit test for the neural-expert word tokenizer (src/neural_expert.h).
// No model, no FFI calls: split_words is a pure static. Asserts it peels German/Unicode
// quotation marks (satzklar-model#1) into their own tokens while keeping umlauts/ß and
// in-word hyphens intact. Build+run:  ./test/run.sh
#include "neural_expert.h"
#include <cassert>
#include <cstdio>
#include <string>
#include <vector>

using nexp::Package;

static std::string join(const std::vector<std::string>& v) {
    std::string s;
    for (size_t i = 0; i < v.size(); i++) { if (i) s += ' '; s += v[i]; }
    return s;
}

static void check(const char* in, const char* expect) {
    std::string got = join(Package::split_words(in));
    if (got != expect) {
        fprintf(stderr, "FAIL split_words(\"%s\")\n  got: [%s]\n  exp: [%s]\n", in, got.c_str(), expect);
        assert(false);
    }
}

int main() {
    // German curly quotes „ … “ peeled → the quoted letter is its own token. Before the fix
    // „e“ survived as one fused token, was tagged PUNCT, and became the parse root.
    check("Ist das ein „e“ oder ein „a“?", "Ist das ein „ e “ oder ein „ a “ ?");
    check("Er sagte: „Hallo Welt“.",       "Er sagte : „ Hallo Welt “ .");
    // Guillemets » … « (also German quotation marks) peeled.
    check("»Guten Tag«", "» Guten Tag «");
    // Umlauts and ß are multi-byte UTF-8 too, but must stay inside words — only the
    // specific quote glyphs are peeled, not any high byte.
    check("Schöne Grüße.", "Schöne Grüße .");
    // In-word hyphen kept (UD German convention); ASCII quotes still peel as before.
    check("Das Wort-Spiel bleibt.", "Das Wort-Spiel bleibt .");
    check("ist das ein \"e\"?",     "ist das ein \" e \" ?");
    printf("test_split_words: all passed\n");
    return 0;
}
