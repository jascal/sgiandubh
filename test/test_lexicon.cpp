// test_lexicon.cpp — the certified lexical lookup (src/pack_transform.h Pack::lexical).
//
// No model, no server: a pack is data and this is a lookup over it. What is asserted is
// the CONTRACT a dictionary depends on — that a known form comes back known with the
// register's own facts, that an unknown one comes back known:false rather than guessed
// at, and that the answer is scoped to the form rather than to an occurrence of it.
//
// Build+run: ./test/run.sh   (needs ../glossa/packs/*.json)
#include "pack_transform.h"
#include <cassert>
#include <cstdio>
#include <string>

using packtrans::Pack;
using packtrans::json;

static int failures = 0;

static void ok(bool cond, const std::string& what) {
    if (!cond) { fprintf(stderr, "FAIL %s\n", what.c_str()); failures++; }
}

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : "../glossa/packs";
    Pack de, es, en;
    if (!de.load(dir + "/de.json") || !es.load(dir + "/es.json") || !en.load(dir + "/en.json")) {
        fprintf(stderr, "skipping: no packs under %s\n", dir.c_str());
        return 0;                       // not a failure where glossa is not checked out
    }

    // Every answer says what it is scoped to, in every language.
    for (auto* p : {&de, &es, &en}) ok(p->lexical("x").value("scope", "") == "form", "scope=form");

    // A form the registers do not know is NOT guessed at. This is the property the
    // whole design rests on: a caller must be able to tell a register fact from a
    // model's invention, and coverage is corpus-bound (en form2lemma ~5.5k entries).
    json miss = de.lexical("xyzzyplugh");
    ok(miss["known"] == false, "unknown form is known:false");
    ok(!miss.contains("lemma"), "unknown form invents no lemma");
    ok(!miss.contains("readings"), "unknown form invents no readings");

    // German: a noun carries its gender; a preposition carries the case it governs;
    // a fused form carries the case it was contracted with.
    ok(de.lexical("Hund")["readings"].contains("noun_gn"), "de Hund has a gender reading");
    ok(de.lexical("mit").contains("governsCase"), "de mit governs a case");
    ok(de.lexical("im").contains("contraction"), "de im is a contraction");

    // Verb government, which is the same field in all three packs.
    ok(en.lexical("gave").value("lemma", "") == "give", "en gave -> give");
    json gave = en.lexical("gave");
    ok(gave.contains("governs"), "en give records the prepositions it takes");

    // Spanish nouns carry gender AND number, unlike German's gender-only table —
    // the shape differs per language and the caller reads whichever key is present.
    ok(es.lexical("libro")["readings"].contains("noun_gnum"), "es libro has gender|number");

    // Provenance travels with every answer: a consumer showing this to a learner can
    // cite where it came from, and can tell two packs apart.
    ok(de.lexical("Hund")["provenance"].value("lang", "") == "de", "provenance names the language");

    // The disambiguation caveat, pinned: `der` is in more than one register at once,
    // which is exactly why scope=form is not a formality.
    json der = de.lexical("der");
    ok(der["known"] == true, "de der is known");
    ok(der.contains("classes") || der.contains("readings"), "de der reports what it belongs to");

    if (failures) { fprintf(stderr, "%d lexicon check(s) failed\n", failures); return 1; }
    printf("lexicon: all checks passed\n");
    return 0;
}
