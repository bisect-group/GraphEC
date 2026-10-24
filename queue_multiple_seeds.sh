#!/bin/bash
# Queue GraphEC feature caching, training and evaluation for seeds 666 777 888
# on every split folder found under the enzyme classification dataset.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
PYTHON_BIN="${PYTHON_BIN:-/home/adhil/.conda/envs/mldb/bin/python}"

"$PYTHON_BIN" -m emulator_bench.queue_pipeline \
    --seed 666 --seed 777 --seed 888 \
    ${PROTTRANS_MODEL_PATH:+--prottrans-model-path "$PROTTRANS_MODEL_PATH"} \
    "$@"
