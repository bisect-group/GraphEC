#!/usr/bin/env bash
# GraphEC paper runs: 4 split groups, test and val evaluation, TF32, no --amp. train also evaluates.
# rescore scores the test rows with non-canonical residues (X-mapped); the paper tables use it.
# Usage: emulator_bench/run.sh [--dry-run] [cache|train|rescore|aggregate|all]
# Env: PY GPUS SEEDS RUNS_PER_GPU DATA_ROOT OUT_ROOT EMULATOR_ROOT DRY_RUN=1
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
EMULATOR_ROOT=${EMULATOR_ROOT:-$(realpath ../..)}
DATA_ROOT=${DATA_ROOT:-$EMULATOR_ROOT/data/processed/datasets/enzyme_classification_dataset}
OUT_ROOT=${OUT_ROOT:-$EMULATOR_ROOT/outputs/baselines/GraphEC}
RUNS_PER_GPU=${RUNS_PER_GPU:-1}
read -ra py <<<"${PY:-python}"
read -ra gpus <<<"${GPUS:-0}"
visible=$(IFS=,; echo "${gpus[*]}")
read -ra seeds <<<"${SEEDS:-666 777 888}"
seed_args=()
for s in "${seeds[@]}"; do seed_args+=(--seed "$s"); done
STEP=all
for a in "$@"; do
  case $a in
    --dry-run) DRY_RUN=1 ;;
    cache|train|rescore|aggregate|all) STEP=$a ;;
    *) echo "usage: $0 [--dry-run] [cache|train|rescore|aggregate|all]" >&2; exit 2 ;;
  esac
done
run() { echo "+ $*"; [[ -n ${DRY_RUN:-} ]] || "$@"; }
want() { [[ $STEP == all || $STEP == "$1" ]]; }

# cache shards over the first --num-gpus visible GPUs, so show it all of GPUS.
want cache && run env CUDA_VISIBLE_DEVICES="$visible" "${py[@]}" -m emulator_bench.cache_features \
  --dataset-root "$DATA_ROOT" --out-root "$OUT_ROOT" --num-gpus "${#gpus[@]}"
# queue_pipeline caches missing features itself; --env-name current runs the children with $PY.
want train && run "${py[@]}" -m emulator_bench.queue_pipeline --dataset-root "$DATA_ROOT" --out-root "$OUT_ROOT" \
  --env-name current --gpus "${gpus[@]}" --runs-per-gpu "$RUNS_PER_GPU" "${seed_args[@]}" --eval-split both
want rescore && run env CUDA_VISIBLE_DEVICES="${gpus[0]}" "${py[@]}" -m emulator_bench.rescore \
  --dataset-root "$DATA_ROOT" --out-root "$OUT_ROOT" "${seed_args[@]}"
want aggregate && run "${py[@]}" -m emulator_bench.aggregate_results --out-root "$OUT_ROOT"
exit 0
