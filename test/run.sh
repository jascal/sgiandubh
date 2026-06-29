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
echo "[test] all passed"
