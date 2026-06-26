#!/usr/bin/env bash
# Build sgiandubh into ONE native binary: the OpenAI server with the Datalog decode engine embedded in-process.
# No fieldrun, no model. Needs: souffle (with compiled-mode headers) + g++ (C++17).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build

# compiled-mode toolchain paths (Soufflé headers/libs under ~/.local; see fieldrun SOUFFLE.md §1.1)
export CPATH="$HOME/.local/include:${CPATH:-}"
export LIBRARY_PATH="$HOME/.local/lib:${LIBRARY_PATH:-}"

echo "[1/2] souffle -g -> engine C++ class"
souffle -g build/engine.cpp src/engine.dl

echo "[2/2] g++ -> one binary (server + embedded engine)"
g++ -O2 -std=c++17 -D__EMBEDDED_SOUFFLE__ -Wno-deprecated-declarations \
    -Ithird_party -I"$HOME/.local/include" \
    src/server.cpp build/engine.cpp -o build/sgiandubh -lpthread -lsqlite3 -lz

echo "built: build/sgiandubh ($(du -h build/sgiandubh | cut -f1)) — server + embedded Datalog engine, no spawn"
echo "run:   ./build/sgiandubh package 8080"
