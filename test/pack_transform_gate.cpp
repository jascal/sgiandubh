// Gate: the pack-driven C++ transform must reproduce glossa's frozen trees.
//
// Same fixtures the Python engine and the Soufflé spec are gated on, so the
// three implementations are held to one behaviour per language.
//
//   g++ -std=c++17 -O2 -isystem third_party -o build/pack_transform_gate
//       test/pack_transform_gate.cpp
//   build/pack_transform_gate ../glossa/packs/de.json ../glossa/fixtures/de_golden.jsonl
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

#include "../src/pack_transform.h"

using json = nlohmann::json;

/** Roles are additive: strip them and the tree must be the fixture again. */
static void strip_roles(json& node) {
    node.erase("role");
    if (node.contains("children"))
        for (auto& c : node["children"]) strip_roles(c);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: pack_transform_gate <pack.json> <fixtures.jsonl> [limit]\n";
        return 2;
    }
    packtrans::Pack pack;
    if (!pack.load(argv[1])) {
        std::cerr << "cannot read pack " << argv[1] << "\n";
        return 2;
    }
    long limit = 0;
    bool roles = false;
    std::string dump_errors, dump_trees;
    for (int k = 3; k < argc; k++) {
        std::string a = argv[k];
        if (a == "--dump-errors" && k + 1 < argc) dump_errors = argv[++k];
        else if (a == "--roles") roles = true;
        else if (a == "--dump-trees" && k + 1 < argc) dump_trees = argv[++k];
        else limit = std::stol(a);
    }
    std::ofstream ef, tf;
    if (!dump_errors.empty()) ef.open(dump_errors);
    if (!dump_trees.empty()) tf.open(dump_trees);
    std::ifstream f(argv[2]);
    if (!f) {
        std::cerr << "cannot read fixtures " << argv[2] << "\n";
        return 2;
    }
    long same = 0, diff = 0, shown = 0;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        if (limit && same + diff >= limit) break;
        json rec = json::parse(line);
        packtrans::Transform tr(pack);
        json got = tr.run(packtrans::tokens_from_json(rec["tokens"]), roles);
        if (tf.is_open())
            tf << json{{"sent_id", rec.value("sent_id", "?")}, {"tree", got}}.dump() << "\n";
        if (ef.is_open())
            ef << json{{"sent_id", rec.value("sent_id", "?")}, {"errors", tr.errors()}}.dump()
               << "\n";
        json cmp = got;
        if (roles) strip_roles(cmp);
        if (cmp == rec["tree"]) {
            same++;
        } else {
            diff++;
            if (shown++ < 2) {
                size_t cut = std::getenv("DUMP") ? std::string::npos : 400;
                std::cerr << "--- DIFF " << rec.value("sent_id", "?") << "\n"
                          << "  want: " << rec["tree"].dump().substr(0, cut) << "\n"
                          << "  got : " << got.dump().substr(0, cut) << "\n";
            }
        }
    }
    std::cout << argv[1] << ": identical " << same << " / different " << diff << "\n";
    return diff ? 1 : 0;
}
