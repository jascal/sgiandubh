// tok_dump.cpp — read lines on stdin, print split_words() output space-joined, one line each.
// Exists so the deployed C++ tokenizer can be diffed against the python reference spoke without
// standing up a server or loading a model (split_words is a pure static). Consumed by
// satzklar-model/scripts/tok_parity.py, the gate for sgiandubh#37.
//   ./build/tok_dump < sentences.txt
#include "neural_expert.h"
#include <fstream>
#include <iostream>
#include <string>

// Optional argv[1]: a grammar pack whose `tokenizer` block drives the split. Without it the
// German-hardcoded behaviour is used, which is what deployed German still gets. The parity gate
// passes the pack so it measures the SERVED configuration rather than the default one.
int main(int argc, char** argv) {
    nexp::Package pkg;
    if (argc > 1) {
        std::ifstream pf(argv[1]);
        if (!pf) { std::cerr << "tok_dump: cannot open " << argv[1] << "\n"; return 2; }
        nlohmann::json pack;
        pf >> pack;
        pkg.load_tokcfg(pack);
    }
    std::string line;
    while (std::getline(std::cin, line)) {
        auto w = nexp::Package::split_words(line, &pkg.tokcfg);
        for (size_t i = 0; i < w.size(); i++) {
            if (i) std::cout << ' ';
            std::cout << w[i];
        }
        std::cout << '\n';
    }
    return 0;
}
