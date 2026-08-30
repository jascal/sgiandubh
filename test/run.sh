#!/usr/bin/env bash
# Build + run sgiandubh's C++ unit tests. No model, no souffle — pure host-side logic over src/*.h.
set -euo pipefail
cd "$(dirname "$0")/.."
CXX="${CXX:-g++}"
echo "[test] tok_ffi (Rust FFI null-safety contract)"
( cd tok_ffi && cargo test --release 2>&1 | grep -E "running|test result|null_inputs" )
echo "[test] rosetta_package.h"
"$CXX" -std=c++17 -O2 -Wall -Wextra -isystem third_party -isystem src test/test_rosetta_package.cpp -o build/test_rosetta_package
./build/test_rosetta_package
echo "[test] neural_expert.h split_words"
"$CXX" -std=c++17 -O2 -Wall -Wextra -isystem third_party -isystem src test/test_split_words.cpp -o build/test_split_words
./build/test_split_words
echo "[test] pack_transform.h lexical lookup"
"$CXX" -std=c++17 -O2 -Wall -Wextra -isystem third_party -isystem src test/test_lexicon.cpp -o build/test_lexicon
./build/test_lexicon ../glossa/packs
# tok_dump is a helper, not a test: it lets satzklar-model/scripts/tok_parity.py diff this
# tokenizer against the python reference spoke without standing up a server (sgiandubh#37).
"$CXX" -std=c++17 -O2 -Wall -Wextra -isystem third_party -isystem src test/tok_dump.cpp -o build/tok_dump
echo "[test] all passed"
