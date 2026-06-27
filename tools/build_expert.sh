#!/usr/bin/env bash
# build_expert.sh — build ONE sgiandubh expert end to end:
#   [distill with a model] → items → grounding (GloVe) → gram → binary.
# Distilled by default (needs fieldrun + a model bundle); pass --model-free for retrieval-only.
#
#   tools/build_expert.sh --out package_x --bundle <model-stem> \
#       --questions q.txt --grounding corpus.txt [--gram g.txt] [--steps 128] [--citation "..."] [--no-split]
#
# Env: PYTHON (default python3), FIELDRUN (default: `fieldrun` on PATH, else ~/code/fieldrun/target/release/fieldrun).
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-python3}"
FIELDRUN="${FIELDRUN:-$(command -v fieldrun 2>/dev/null || echo "$HOME/code/fieldrun/target/release/fieldrun")}"

OUT="" BUNDLE="" QUESTIONS="" GROUNDING="" GRAM="" STEPS=128 CITATION="" NOSPLIT="" MODELFREE=0
while [ $# -gt 0 ]; do case "$1" in
  --out)        OUT="$2"; shift 2 ;;
  --bundle)     BUNDLE="$2"; shift 2 ;;
  --questions)  QUESTIONS="$2"; shift 2 ;;
  --grounding)  GROUNDING="$2"; shift 2 ;;
  --gram)       GRAM="$2"; shift 2 ;;
  --steps)      STEPS="$2"; shift 2 ;;
  --citation)   CITATION="$2"; shift 2 ;;
  --no-split)   NOSPLIT="--no-split"; shift ;;
  --model-free) MODELFREE=1; shift ;;
  *) echo "build_expert: unknown arg $1" >&2; exit 1 ;;
esac; done
[ -n "$OUT" ] && [ -n "$GROUNDING" ] || {
  echo "usage: build_expert.sh --out DIR --grounding CORPUS [--bundle B --questions Q | --model-free] [--gram G --steps N --citation C --no-split]" >&2
  exit 1; }
GRAM="${GRAM:-$GROUNDING}"
command -v souffle >/dev/null || { echo "build_expert: need souffle on PATH" >&2; exit 1; }
$PY -c "import numpy, scipy" 2>/dev/null || { echo "build_expert: need python3 + numpy + scipy" >&2; exit 1; }
mkdir -p "$OUT"

if [ "$MODELFREE" = 0 ]; then
  [ -n "$BUNDLE" ] && [ -n "$QUESTIONS" ] || { echo "build_expert: distilled build needs --bundle and --questions (or --model-free)" >&2; exit 1; }
  [ -x "$FIELDRUN" ] || { echo "build_expert: fieldrun not found at '$FIELDRUN' — build it or set FIELDRUN=/path/to/fieldrun" >&2; exit 1; }
  echo "[1/5] distill $(wc -l < "$QUESTIONS") prompts (--steps $STEPS) — the heavy step, hours on CPU for a big set…"
  "$FIELDRUN" --bundle "$BUNDLE" --export-logic-corpus "$QUESTIONS" --steps "$STEPS" --out "$OUT/_export"
  echo "[2/5] package distilled items"
  $PY "$HERE/tools/dl2package.py" --corpus "$QUESTIONS" --dl "$OUT/_export" --out "$OUT" ${CITATION:+--citation "$CITATION"}
else
  echo "[1-2/5] model-free: skipping distill (items come from the source adapter, if any)"
fi
echo "[3/5] grounding (pretrained GloVe — default)"
$PY "$HERE/tools/build_grounding.py" --corpus "$GROUNDING" --out "$OUT" $NOSPLIT
echo "[4/5] gram kernel"
$PY "$HERE/tools/build_gram.py" --corpus "$GRAM" --out "$OUT/gram"
echo "[5/5] binary"
[ -x "$HERE/build/sgiandubh" ] || ( cd "$HERE" && ./build.sh >/dev/null )
echo "✓ expert built: $OUT"
echo "  serve: $HERE/build/sgiandubh $OUT 8080 --answer-from-corpus [--require-citation]"
