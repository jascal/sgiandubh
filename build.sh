#!/usr/bin/env bash
# Build sgiandubh: (1) compile the Datalog decode engine to a native binary, (2) compile the OpenAI server.
# No fieldrun, no model. Needs: souffle (with compiled-mode headers) + g++ (C++17).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build

# compiled-mode toolchain paths (Soufflé's headers/libs installed under ~/.local; see fieldrun SOUFFLE.md §1.1)
export CPATH="$HOME/.local/include:${CPATH:-}"
export LIBRARY_PATH="$HOME/.local/lib:${LIBRARY_PATH:-}"

echo "[1/2] souffle -> native engine"
souffle -o build/engine src/engine.dl

echo "[2/2] g++ -> OpenAI server"
g++ -O2 -std=c++17 -Ithird_party src/server.cpp -o build/sgiandubh -lpthread

echo "built: build/engine (datalog decode) + build/sgiandubh (openai server)"
echo "run:   ./build/sgiandubh package build/engine 8080"
