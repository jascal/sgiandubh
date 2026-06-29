#!/usr/bin/env bash
# Build sgiandubh into ONE native binary: the OpenAI server with the semiring decode embedded in-process.
# No model, no fieldrun, no Soufflé. Needs: g++ (C++17) + rust/cargo (the tokenizer FFI staticlib).
#
# DECODE: the per-decision decode is engine.dl ported to C++ (rosetta::decode_facts in src/rosetta_package.h) — a plain
# semiring combine (logit = Σ contrib; decide = argmax). It is verified identical to the former Soufflé engine. The
# Datalog reasoning path (ergo core) is retired — that role is now rosetta's. src/engine.dl is kept as the reference
# spec decode_facts implements, but is no longer compiled.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build

DEBUG=0
for a in "$@"; do case "$a" in --debug) DEBUG=1 ;; esac; done
if [ "$DEBUG" = 1 ]; then MODE="-g -O1 -fno-omit-frame-pointer -fsanitize=address,undefined"; echo "[*] debug build: ASan + UBSan + warnings";
else MODE="-O2"; fi

# -isystem (not -I) for vendored headers so their warnings don't drown ours; -Wall/-Wextra only on OUR code.
CXXFLAGS="-std=c++17 $MODE -Wno-deprecated-declarations -isystem third_party"

TOKLIB=tok_ffi/target/release/libtok_ffi.a
echo "[1/3] cargo -> tokenizer FFI staticlib (HF tokenizers; powers the --rosetta-package runtime's BPE tokenize)"
( cd tok_ffi && cargo build --release )
echo "[2/3] g++ -> server object (-Wall -Wextra; pure-C++ semiring decode)"
g++ $CXXFLAGS -Wall -Wextra -c src/server.cpp -o build/server.o
echo "[3/3] g++ -> link (+ the tokenizer FFI staticlib)"
g++ $CXXFLAGS build/server.o "$TOKLIB" -o build/sgiandubh -lpthread -ldl -lm

# compile_commands.json for clang-tidy / LSP (server.cpp is the unit tooling cares about). Absolute paths → gitignored.
printf '[{"directory": "%s", "file": "src/server.cpp", "command": "g++ %s -Wall -Wextra -c src/server.cpp"}]\n' \
    "$PWD" "$CXXFLAGS" > compile_commands.json

echo "built: build/sgiandubh ($(du -h build/sgiandubh | cut -f1))$([ "$DEBUG" = 1 ] && echo ' [debug: ASan+UBSan]') — server + pure-C++ semiring decode, no spawn"
echo "run:   ./build/sgiandubh package 8080"
