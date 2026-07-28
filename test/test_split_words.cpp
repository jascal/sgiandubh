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

    // ---- apostrophes (sgiandubh#37) ---------------------------------------------------------
    // Flanked by word characters → stays inside the token. Corpus: c't 891x PROPN.
    check("Die c't-Redaktion testet.", "Die c't-Redaktion testet .");
    check("Er liest die c't.",         "Er liest die c't .");
    // ...EXCEPT before a bare clitic, where the split goes BEFORE the apostrophe and it travels
    // with the clitic. Corpus: 's 105x PRON. Both the ASCII and the typographic apostrophe.
    check("Geht's dir gut?",  "Geht 's dir gut ?");
    check("Geht’s dir gut?",  "Geht ’s dir gut ?");
    check("Wie geht's?",      "Wie geht 's ?");
    // Chunk-initial clitic apostrophe is part of the token, not an opening quote.
    check("Das ist 'ne gute Idee.", "Das ist 'ne gute Idee .");
    check("Das ist 'n Buch.",       "Das ist 'n Buch .");
    // An ASCII apostrophe used as a QUOTE still peels — the remainder is not a clitic, so the
    // clitic rule does not fire and nothing here loosens quote handling.
    check("Er sagte 'Hallo' laut.", "Er sagte ' Hallo ' laut .");
    // A clitic-looking suffix that does not end the word is not a clitic (c't-Redaktion above
    // covers the hyphen case; here the run after ' continues into more letters).
    check("Die c'ts sind da.", "Die c'ts sind da .");
    // Trailing apostrophe (genitive Gates') is deliberately NOT special-cased: it is ambiguous
    // with a closing quote, and the corpus has 59 standalone ' against 18 Gates'. Documented in
    // sgiandubh#37 as out of scope; this pins the current behavior so a change is deliberate.
    check("Das ist Gates' Buch.", "Das ist Gates ' Buch .");
    printf("test_split_words: all passed\n");
    return 0;
}
