#!/usr/bin/env bash
# Build sgiandubh into ONE native binary: the OpenAI server with the Datalog decode engine embedded in-process.
# No fieldrun, no model. Needs: souffle (with compiled-mode headers) + g++ (C++17).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build

# OPTIONAL: union the ergo core reasoning KB into the engine.  --with-core  (or  CORE_DL=/path/to/ergo/core.dl).
# Single source of truth = ergo/core.dl; read at build time into an EPHEMERAL combined .dl under build/ (regenerated
# every build, gitignored) — never a committed copy. Off by default → the plain decode engine.
ENGINE=src/engine.dl
WITH_CORE=0; DEBUG=0
for a in "$@"; do case "$a" in --with-core) WITH_CORE=1 ;; --debug) DEBUG=1 ;; esac; done
if [ "$DEBUG" = 1 ]; then MODE="-g -O1 -fno-omit-frame-pointer -fsanitize=address,undefined"; echo "[*] debug build: ASan + UBSan + warnings";
else MODE="-O2"; fi
if [ "$WITH_CORE" = 1 ] || [ -n "${CORE_DL:-}" ]; then
    CORE_DL="${CORE_DL:-../ergo/core.dl}"
    [ -f "$CORE_DL" ] || { echo "build: core.dl not found at '$CORE_DL' — set CORE_DL=/path/to/ergo/core.dl" >&2; exit 1; }
    ENGINE=build/engine_with_core.dl
    { cat src/engine.dl; printf '\n// ---- unioned core reasoning KB (%s) ----\n' "$CORE_DL"; cat "$CORE_DL"; } > "$ENGINE"
    echo "[*] core reasoning KB unioned from $CORE_DL  (ephemeral $ENGINE)"
fi

# compiled-mode toolchain paths (Soufflé headers/libs under ~/.local; see fieldrun SOUFFLE.md §1.1)
export CPATH="$HOME/.local/include:${CPATH:-}"
export LIBRARY_PATH="$HOME/.local/lib:${LIBRARY_PATH:-}"

# -isystem (not -I) for vendored + souffle headers so their warnings don't drown ours; -Wall/-Wextra only on OUR code.
CXXFLAGS="-std=c++17 $MODE -D__EMBEDDED_SOUFFLE__ -Wno-deprecated-declarations -isystem third_party -isystem $HOME/.local/include"

echo "[1/4] cargo -> tokenizer FFI staticlib (HF tokenizers; powers the --rosetta-package runtime's BPE tokenize)"
( cd tok_ffi && cargo build --release )
echo "[2/4] souffle -g -> engine C++ class"
souffle -g build/engine.cpp "$ENGINE"
echo "[3/4] g++ -> objects (engine: generated, warnings off; server: -Wall -Wextra)"
g++ $CXXFLAGS -w -c build/engine.cpp -o build/engine.o
g++ $CXXFLAGS -Wall -Wextra -c src/server.cpp -o build/server.o
echo "[4/4] g++ -> link (+ the tokenizer FFI staticlib)"
g++ $CXXFLAGS build/engine.o build/server.o tok_ffi/target/release/libtok_ffi.a -o build/sgiandubh -lpthread -lsqlite3 -lz -ldl -lm

# compile_commands.json for clang-tidy / LSP (server.cpp is the unit tooling cares about). Absolute paths → gitignored.
printf '[{"directory": "%s", "file": "src/server.cpp", "command": "g++ %s -Wall -Wextra -c src/server.cpp"}]\n' \
    "$PWD" "$CXXFLAGS" > compile_commands.json

echo "built: build/sgiandubh ($(du -h build/sgiandubh | cut -f1))$([ "$DEBUG" = 1 ] && echo ' [debug: ASan+UBSan]') — server + embedded Datalog engine, no spawn"
echo "run:   ./build/sgiandubh package 8080"
