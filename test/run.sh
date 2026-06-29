#!/usr/bin/env bash
# Build + run sgiandubh's C++ unit tests. No model, no souffle — pure host-side logic over src/*.h.
set -euo pipefail
cd "$(dirname "$0")/.."
CXX="${CXX:-g++}"
echo "[test] rosetta_package.h"
"$CXX" -std=c++17 -O2 -Wall -Wextra -isystem third_party -isystem src test/test_rosetta_package.cpp -o build/test_rosetta_package
./build/test_rosetta_package
echo "[test] all passed"
