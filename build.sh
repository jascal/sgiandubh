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
WITH_CORE=0
for a in "$@"; do [ "$a" = "--with-core" ] && WITH_CORE=1; done
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

echo "[1/2] souffle -g -> engine C++ class"
souffle -g build/engine.cpp "$ENGINE"

echo "[2/2] g++ -> one binary (server + embedded engine)"
g++ -O2 -std=c++17 -D__EMBEDDED_SOUFFLE__ -Wno-deprecated-declarations \
    -Ithird_party -I"$HOME/.local/include" \
    src/server.cpp build/engine.cpp -o build/sgiandubh -lpthread -lsqlite3 -lz

echo "built: build/sgiandubh ($(du -h build/sgiandubh | cut -f1)) — server + embedded Datalog engine, no spawn"
echo "run:   ./build/sgiandubh package 8080"
