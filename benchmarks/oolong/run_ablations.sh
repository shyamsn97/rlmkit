#!/usr/bin/env bash
# Run the OOLONG ablation matrix — port of Prime Intellect's
# `run_ablations.sh` from
# https://github.com/PrimeIntellect-ai/verifiers/tree/sebastian/experiment/rlm/environments/oolong
# adapted to the rlmflow runner.
#
# Sweeps:
# - 3 modes: standard, rlm, rlm_tips
# - 3 subsets: synth, synth_with_labels, real
#
# Usage:
#   ./benchmarks/oolong/run_ablations.sh
#   N=20 MODELS_FULL="gpt-4.1-mini" ./benchmarks/oolong/run_ablations.sh
#   WORKERS=4 SPLIT=test ./benchmarks/oolong/run_ablations.sh
#
# Per-config outputs go under benchmarks/oolong/outputs/evals/ with one
# directory per (mode, subset, model). After the sweep, run
#   python benchmarks/oolong/aggregate.py
# for a comparison table.

set -euo pipefail

# Models to run all (mode × subset) combinations on.
MODELS_FULL="${MODELS_FULL:-gpt-4.1-mini}"
# Models to run all modes on, but only on a single subset (broader
# coverage at lower cost). Set to "" to skip this group.
MODELS_STANDARD="${MODELS_STANDARD:-}"

N="${N:-50}"
SEED="${SEED:-42}"
SPLIT="${SPLIT:-validation}"
WORKERS="${WORKERS:-1}"
MAX_ITERATIONS="${MAX_ITERATIONS:-30}"
MAX_DEPTH="${MAX_DEPTH:-3}"
DEFAULT_SUBSET="${DEFAULT_SUBSET:-real}"
DOCKER_IMAGE="${DOCKER_IMAGE:-}"

MODES=("rlm" "rlm_tips" "standard")
SUBSETS=("synth" "synth_with_labels" "real")

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_PY="$ROOT_DIR/benchmarks/oolong/run.py"
OUT_BASE="$ROOT_DIR/benchmarks/oolong/outputs/evals"

run_one() {
    local model="$1" mode="$2" subset="$3"
    local safe_model
    safe_model="$(printf '%s' "$model" | tr -c 'a-zA-Z0-9' '-')"
    local outdir="$OUT_BASE/${mode}_${subset}_${safe_model}"

    echo ""
    echo "========================================"
    echo "model=$model  mode=$mode  subset=$subset"
    echo "out=$outdir"
    echo "========================================"

    local docker_arg=()
    if [[ -n "$DOCKER_IMAGE" ]]; then
        docker_arg=(--docker-image "$DOCKER_IMAGE")
    fi

    python "$RUN_PY" \
        --mode "$mode" \
        --subset "$subset" \
        --split "$SPLIT" \
        --shuffle --seed "$SEED" \
        --limit "$N" \
        --workers "$WORKERS" \
        --max-iterations "$MAX_ITERATIONS" \
        --max-depth "$MAX_DEPTH" \
        --model "$model" \
        --out "$outdir" \
        "${docker_arg[@]}"
}

echo "=== OOLONG ablations ==="
echo "MODELS_FULL: $MODELS_FULL"
[[ -n "$MODELS_STANDARD" ]] && echo "MODELS_STANDARD: $MODELS_STANDARD (subset=$DEFAULT_SUBSET only)"
echo "N=$N  split=$SPLIT  workers=$WORKERS  max_iterations=$MAX_ITERATIONS"

# Part 1 — full (mode × subset) sweep for MODELS_FULL.
for model in $MODELS_FULL; do
    for mode in "${MODES[@]}"; do
        for subset in "${SUBSETS[@]}"; do
            run_one "$model" "$mode" "$subset"
        done
    done
done

# Part 2 — every mode but only $DEFAULT_SUBSET for MODELS_STANDARD.
if [[ -n "$MODELS_STANDARD" ]]; then
    for model in $MODELS_STANDARD; do
        for mode in "${MODES[@]}"; do
            run_one "$model" "$mode" "$DEFAULT_SUBSET"
        done
    done
fi

echo ""
echo "=== ablations complete ==="
echo "Aggregate with: python benchmarks/oolong/aggregate.py"
