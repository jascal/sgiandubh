// tok_dump.cpp — read lines on stdin, print split_words() output space-joined, one line each.
// Exists so the deployed C++ tokenizer can be diffed against the python reference spoke without
// standing up a server or loading a model (split_words is a pure static). Consumed by
// satzklar-model/scripts/tok_parity.py, the gate for sgiandubh#37.
//   ./build/tok_dump < sentences.txt
#include "neural_expert.h"
#include <iostream>
#include <string>

int main() {
    std::string line;
    while (std::getline(std::cin, line)) {
        auto w = nexp::Package::split_words(line);
        for (size_t i = 0; i < w.size(); i++) {
            if (i) std::cout << ' ';
            std::cout << w[i];
        }
        std::cout << '\n';
    }
    return 0;
}
