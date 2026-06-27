#!/usr/bin/env bash
# build_experts.sh — build BOTH demo experts (riscv + logic), distilled, end to end. One command on a CPU box.
#
#   BUNDLE=~/.cache/fieldrun/bundles/Qwen2.5-7B-Instruct/Qwen2.5-7B-Instruct ./tools/build_experts.sh
#   NORM_RULES=/path/to/norm-rules.json BUNDLE=... ./tools/build_experts.sh   # if norm-rules.json isn't in repo root
#   ONLY=logic BUNDLE=... ./tools/build_experts.sh                            # build just one (logic|riscv)
#   STEPS=128 ...                                                             # distill depth (default 128)
#
# Prereqs: souffle, g++, python3+numpy+scipy, a built fieldrun (FIELDRUN=... or on PATH), a model bundle (BUNDLE=).
# RISC-V source: norm-rules.json from a riscv-isa-manual release (https://github.com/riscv/riscv-isa-manual/releases).
# Logic source: corpora/logic_kb.txt + corpora/logic_questions.txt (shipped, Open Logic Project, CC BY 4.0).
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-python3}"
BUNDLE="${BUNDLE:-}"; STEPS="${STEPS:-128}"; ONLY="${ONLY:-both}"
[ -n "$BUNDLE" ] || { echo "set BUNDLE=<fieldrun model bundle stem>, e.g. ~/.cache/fieldrun/bundles/Qwen2.5-7B-Instruct/Qwen2.5-7B-Instruct" >&2; exit 1; }

build_riscv() {
  local NORM="${NORM_RULES:-$HERE/norm-rules.json}"
  if [ ! -f "$NORM" ]; then
    echo "RISC-V: norm-rules.json not found at $NORM" >&2
    echo "  grab it from a riscv-isa-manual release: https://github.com/riscv/riscv-isa-manual/releases" >&2
    echo "  then re-run with NORM_RULES=/path/to/norm-rules.json" >&2
    return 1
  fi
  echo "=== RISC-V expert ==="
  $PY "$HERE/tools/normrules2package.py" "$NORM" "$HERE/package_riscv"
  $PY "$HERE/tools/riscv_questions.py" --rules "$HERE/package_riscv/rules.txt" --out "$HERE/package_riscv/questions.txt"
  "$HERE/tools/build_expert.sh" --out "$HERE/package_riscv" --bundle "$BUNDLE" --steps "$STEPS" \
    --questions "$HERE/package_riscv/questions.txt" --grounding "$HERE/package_riscv/rules.txt" \
    --gram "$HERE/package_riscv/rules_plain.txt" --no-split --citation "RISC-V ISA Manual (CC BY 4.0)"
}

build_logic() {
  echo "=== Logic expert ==="
  "$HERE/tools/build_expert.sh" --out "$HERE/package_logic" --bundle "$BUNDLE" --steps "$STEPS" \
    --questions "$HERE/corpora/logic_questions.txt" --grounding "$HERE/corpora/logic_kb.txt" \
    --citation "Open Logic Project (CC BY 4.0)"
}

case "$ONLY" in
  riscv) build_riscv ;;
  logic) build_logic ;;
  both)  build_riscv; build_logic ;;
  *) echo "ONLY must be riscv|logic|both" >&2; exit 1 ;;
esac

echo "=== done — serve the spokes ==="
echo "  $HERE/build/sgiandubh package_riscv 8081 --answer-from-corpus --require-citation"
echo "  $HERE/build/sgiandubh package_logic 8082 --answer-from-corpus"
echo "  (or bring up the whole hub stack: claymore/examples/run-local.sh)"
