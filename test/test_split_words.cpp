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

    // ---- edge peeling + measured rules (sgiandubh#37) -----------------------------------------
    // Peeling only at token EDGES is what keeps these whole with no special case. Peel-anywhere
    // split every one of them; against gold UD tokenization this took 86.1% -> 95.9% exact.
    check("Bin 4:20 Minuten vor dem Termin.", "Bin 4:20 Minuten vor dem Termin .");
    check("Er kostet 1,24 Euro.",             "Er kostet 1,24 Euro .");
    check("Die Nr.1 im Land.",                "Die Nr.1 im Land .");
    check("Wieso nicht 1/2h früher?",         "Wieso nicht 1/2h früher ?");
    // R1: runs of these marks are ONE token (gold: -- 199x, ... 244x, `` 180x, '' 171x)...
    check("Dr. Meier kam -- und ging...", "Dr. Meier kam -- und ging ...");
    check("Es war ``gut'' gesagt.",       "Es war `` gut '' gesagt .");
    // ...but NOT ! or ?, which gold splits (406 single against 5 merged).
    check("Was!!! Wirklich???", "Was ! ! ! Wirklich ? ? ?");
    // R2: abbreviations keep their period; no sentence-final carve-out (A/B'd, not justified).
    check("Es kostet ca. 5 Euro etc.", "Es kostet ca. 5 Euro etc.");
    // R3: an ordinal keeps its period mid-sentence, but never at the sentence end — gold has
    // 0 sentence-final "NN." against 255 split, so both halves of this are measured.
    check("Am 31. März 2001.",  "Am 31. März 2001 .");
    check("Es waren nur 2001.", "Es waren nur 2001 .");
    // '-' peels at edges (gold: 1779 bare '-'), which leaves internal hyphens alone for free.
    check("Sonn- und Feiertage.",   "Sonn - und Feiertage .");
    check("Das Wort-Spiel bleibt.", "Das Wort-Spiel bleibt .");

    // ---- Spanish inverted marks (glossa#es-inverted) ------------------------------------------
    // ¿/¡ are NOT ascii, so std::ispunct never saw them and the leading peel stopped on byte 0:
    // "¿Qué" was served as ONE token to a parser whose gold has 403 standalone ¿ and 0 glued.
    // On the es test treebanks this alone was 44 of 70 disagreements with the python reference.
    check("¿Qué hora es?",   "¿ Qué hora es ?");
    check("¡Hola mundo!",    "¡ Hola mundo !");
    check("¿Cómo estás?",    "¿ Cómo estás ?");
    check("Dime, ¿vienes?",  "Dime , ¿ vienes ?");
    // They are not RUNNABLE: gold never merges a run of them, matching langs.py's es tokenizer.
    check("¿¡Qué!?", "¿ ¡ Qué ! ?");
    // A mark on its own chunk is still one token, and a closing mark peels from the right edge.
    check("¿ Y ahora ?", "¿ Y ahora ?");
    // Accented letters are multi-byte too and must stay inside the word — only these glyphs peel.
    check("Corrí más rápido.", "Corrí más rápido .");

    printf("test_split_words: all passed\n");
    return 0;
}
