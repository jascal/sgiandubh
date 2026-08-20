// Gate: the pack-driven C++ transform must reproduce glossa's frozen trees.
//
// Same fixtures the Python engine and the Soufflé spec are gated on, so the
// three implementations are held to one behaviour per language.
//
//   g++ -std=c++17 -O2 -isystem third_party -o build/pack_transform_gate \
//       test/pack_transform_gate.cpp
//   build/pack_transform_gate ../glossa/packs/de.json ../glossa/fixtures/de_golden.jsonl
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

#include "../src/pack_transform.h"

using json = nlohmann::json;

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
    long limit = argc > 3 ? std::stol(argv[3]) : 0;
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
        json got = tr.run(packtrans::tokens_from_json(rec["tokens"]));
        if (got == rec["tree"]) {
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
