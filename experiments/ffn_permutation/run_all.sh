#!/usr/bin/env bash
# Staged, resumable entry point for the FFN permutation experiments.
# Stage A must pass before B; B before C.
set -euo pipefail
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
mkdir -p results logs

echo "=== Stage A: synthetic algebra + directionality unit tests ==="
conda run --no-capture-output -n qwen3 python probe_synthetic.py 2>&1 | tee logs/stage_a.log

echo "=== Stage B: single real MLP isolation (layers 0/17/35) ==="
conda run --no-capture-output -n qwen3 python probe_single_mlp.py --resume 2>&1 | tee logs/stage_b.log

echo "=== Stage C: full-model propagation ==="
conda run --no-capture-output -n qwen3 python probe_full_model.py --resume 2>&1 | tee logs/stage_c.log

echo "All stages complete."
