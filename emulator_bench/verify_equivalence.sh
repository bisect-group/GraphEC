#!/usr/bin/env bash
# Checks that the GraphEC wrapper matches the original code. Usage: verify_equivalence.sh [--gpu | --negative-control]
# Expected env: mldb (torch, torch_geometric, pytest). The script does not create envs.
set -euo pipefail
cd "$(dirname "$0")/.."
echo "Expected conda env: mldb"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
files=(tests/test_original_equivalence.py tests/test_edge_mlp.py tests/test_active_site_batching.py
    tests/test_cache_and_diffusion.py tests/test_precision_gpu.py)
marker=(-m "not gpu")
extra=()
allow_skip=
if [[ ${1:-} == --gpu ]]; then
    marker=()
    extra=(-s)  # shows the printed gap and floor
    if ! python -c "import torch,sys; sys.exit(not torch.cuda.is_available())"; then
        echo "No CUDA device visible: gpu tests will be skipped."
        allow_skip=1
    fi
elif [[ ${1:-} == --negative-control ]]; then
    # Scales the production run's initial weights by 1.05. The GPU test must FAIL, else its bound is vacuous.
    out=$(EQUIV_NEGATIVE_CONTROL=1 python -m pytest tests/test_precision_gpu.py -q -p no:cacheprovider --color=no -s -m gpu 2>&1) || true
    echo "$out"
    if grep -Eq "[0-9]+ passed" <<<"$out" || ! grep -Eq "[0-9]+ failed" <<<"$out"; then
        echo "FAIL: a GPU test did not fail under the negative control"
        exit 1
    fi
    echo "OK: every GPU test fails under the negative control"
    exit 0
else
    export CUDA_VISIBLE_DEVICES=""
fi
out=$(python -m pytest "${files[@]}" -q -rs -p no:cacheprovider --color=no "${marker[@]}" "${extra[@]}" 2>&1) || { echo "$out"; exit 1; }
echo "$out"
if [[ -z $allow_skip ]] && grep -Eq "[0-9]+ skipped" <<<"$out"; then
    echo "FAIL: tests were skipped"
    exit 1
fi
