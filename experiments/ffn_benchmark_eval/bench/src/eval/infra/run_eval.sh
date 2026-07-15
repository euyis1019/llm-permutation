#!/usr/bin/env bash
# Unified eval driver (single Hope job, in-process vLLM).
#
# Replaces the deploy+client (HTTP server) path of run_suite_src.sh. Both bench
# families now run vLLM in-process — no separate server, no endpoint-ready wait,
# no BENCH_CONCURRENCY knob (vLLM continuous batching saturates the GPU on its
# own). One job evaluates one bench for N_RUNS greedy runs and writes the unified
# layout: OUTPUT_BASE/<MODEL_TAG>/<BENCH_ID>/{run_NN/result.json, summary.json}.
#
# Dispatch by RUNNER:
#   RUNNER=protocol  → src.eval.benchmark.inproc_runner  (gsm8k/ceval/cmmlu/cruxeval/mmlu)
#   RUNNER=external  → src/eval/external/run_code_eval.sh (evalplus: humaneval/mbpp)
#
# Required env vars (injected by submit_eval.py [user_args]):
#   MODEL_PATH    absolute NFS path to model directory
#   BENCH_ID      bench id (unified schema)
#   OUTPUT_BASE   absolute NFS root for all outputs
#   RUNNER        protocol | external
#
# Optional env vars:
#   MODEL_TAG     output tag (default: basename(MODEL_PATH) lowercased)
#   N_RUNS        number of greedy runs (default: 3)
#   RUN_OFFSET    starting run index (default: 0)
#   TP_SIZE       tensor-parallel (default: protocol=2, external=4)
#   BATCH_CAP     protocol only: max prompt-list copies per generate (default: 3)
#   NROWS         protocol only: truncate to first N rows (default: full)
#   DATASET       external only: evalplus dataset key (humaneval | mbpp)
#   METRIC        external only: pass@1 metric (default: base_pass_at_1)

set -euo pipefail
export no_proxy="localhost,127.0.0.1,0.0.0.0"
export NO_PROXY="localhost,127.0.0.1,0.0.0.0"

# REPO = repo root, derived from this script's location (src/eval/infra/ → repo root,
# three levels up). No hardcoded absolute path — portable across platforms.
# Execution-layer assets (compute_summary.py, reference_evalplus, make_patched_model.sh,
# patch_longcat_model.py) live alongside this driver under src/eval/infra/.
WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${WORK_DIR}/../../.." && pwd)"
CODE_EVAL_SH="${REPO}/src/eval/external/run_code_eval.sh"

MODEL_PATH="${MODEL_PATH:?MODEL_PATH must be set}"
BENCH_ID="${BENCH_ID:?BENCH_ID must be set}"
OUTPUT_BASE="${OUTPUT_BASE:?OUTPUT_BASE must be set}"
RUNNER="${RUNNER:?RUNNER must be set (protocol | external)}"
N_RUNS="${N_RUNS:-3}"
RUN_OFFSET="${RUN_OFFSET:-0}"

MODEL_TAG="${MODEL_TAG:-$(basename "${MODEL_PATH}" | tr '[:upper:]' '[:lower:]')}"
BENCH_DIR="${OUTPUT_BASE}/${MODEL_TAG}/${BENCH_ID}"

# ── Logging (NFS; platform may drop stderr) ───────────────────────────────────
LOG_DIR="${OUTPUT_BASE}/logs"
mkdir -p "${LOG_DIR}" "${BENCH_DIR}"
THIS_LOG="${LOG_DIR}/run_eval_${BENCH_ID}_$(hostname)_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${THIS_LOG}") 2>&1

echo "[$(date)] ── run_eval.sh ──────────────────────────────────────────────"
echo "[$(date)] RUNNER      = ${RUNNER}"
echo "[$(date)] BENCH_ID    = ${BENCH_ID}"
echo "[$(date)] MODEL_PATH  = ${MODEL_PATH}"
echo "[$(date)] MODEL_TAG   = ${MODEL_TAG}"
echo "[$(date)] OUTPUT_BASE = ${OUTPUT_BASE}"
echo "[$(date)] BENCH_DIR   = ${BENCH_DIR}"
echo "[$(date)] N_RUNS      = ${N_RUNS}"
echo "[$(date)] Hostname    = $(hostname)"
nvidia-smi -L 2>/dev/null || true

export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HF_HOME=/tmp/hf_home

# ── Patch LongCat remote-code for transformers 5.8.x + vLLM 0.21 MoE backend ──
# The raw checkpoints crash vLLM (rope API removed, create_causal_mask signature
# change, MoE field names vLLM can't find). make_patched_model.sh builds a shadow
# dir (weights symlinked, *.py/*.json copied + patched) on NFS, shared by all 7
# bench jobs of this tag. flock serializes the concurrent first-builders; the
# marker file makes it a no-op afterwards. Then point MODEL_PATH at the shadow so
# both protocol (inproc_runner) and external (run_code_eval.sh, via MODEL_PATH env)
# load the patched model.
PATCH_SH="$(dirname "${BASH_SOURCE[0]}")/make_patched_model.sh"
SHADOW_DIR="${OUTPUT_BASE}/../_patched_models/${MODEL_TAG}"
mkdir -p "$(dirname "${SHADOW_DIR}")"
echo "[$(date)] building/reusing patched shadow model (flock-guarded) ..."
PATCHED_PATH="$(
  flock "$(dirname "${SHADOW_DIR}")/.lock_${MODEL_TAG}" \
    bash "${PATCH_SH}" "${MODEL_PATH}" "${SHADOW_DIR}"
)"
if [ -z "${PATCHED_PATH}" ] || [ ! -f "${PATCHED_PATH}/.patched_ok" ]; then
    echo "[$(date)] ERROR: patched model build failed (got '${PATCHED_PATH}')" >&2
    exit 1
fi
echo "[$(date)] MODEL_PATH (patched) = ${PATCHED_PATH}"
MODEL_PATH="${PATCHED_PATH}"

GPU_COUNT=$(nvidia-smi --list-gpus 2>/dev/null | wc -l || echo 1)

case "${RUNNER}" in
  protocol)
    TP_SIZE="${TP_SIZE:-2}"
    BATCH_CAP="${BATCH_CAP:-3}"
    META_PATH="${REPO}/datasets/benchmark/normalized/${BENCH_ID}/benchmark_meta.json"
    if [ ! -f "${META_PATH}" ]; then
        echo "[$(date)] ERROR: benchmark_meta.json not found: ${META_PATH}" >&2
        exit 1
    fi
    NROWS_ARG=()
    if [ -n "${NROWS:-}" ]; then NROWS_ARG=(--nrows "${NROWS}"); fi
    echo "[$(date)] protocol bench → inproc_runner (TP=${TP_SIZE}, batch_cap=${BATCH_CAP})"
    EXIT=0
    python3 -m src.eval.benchmark.inproc_runner \
        --meta-path "${META_PATH}" \
        --bench-id "${BENCH_ID}" \
        --model-path "${MODEL_PATH}" \
        --output-base "${OUTPUT_BASE}" \
        --model-tag "${MODEL_TAG}" \
        --n-runs "${N_RUNS}" \
        --run-offset "${RUN_OFFSET}" \
        --tp-size "${TP_SIZE}" \
        --batch-cap "${BATCH_CAP}" \
        "${NROWS_ARG[@]}" || EXIT=$?
    ;;

  external)
    export TP_SIZE="${TP_SIZE:-4}"
    export EVALPLUS_SRC="${WORK_DIR}/reference_evalplus"
    DATASET="${DATASET:?DATASET must be set for external bench (humaneval | mbpp)}"
    METRIC="${METRIC:-base_pass_at_1}"
    if [ ! -f "${CODE_EVAL_SH}" ]; then
        echo "[$(date)] ERROR: run_code_eval.sh not found: ${CODE_EVAL_SH}" >&2
        exit 1
    fi
    echo "[$(date)] external bench → run_code_eval.sh (evalplus, TP=${TP_SIZE}, dataset=${DATASET})"
    OVERALL_EXIT=0
    for i in $(seq "${RUN_OFFSET}" $((RUN_OFFSET + N_RUNS - 1))); do
        RUN_DIR="${BENCH_DIR}/$(printf "run_%02d" "${i}")"
        mkdir -p "${RUN_DIR}"
        echo "[$(date)] ══ run_$(printf "%02d" "${i}") ($((i - RUN_OFFSET + 1))/${N_RUNS}) ══"
        MODEL_PATH="${MODEL_PATH}" \
        DATASET="${DATASET}" \
        OUTPUT_DIR="${RUN_DIR}" \
        RESULT_PATH="${RUN_DIR}/result.json" \
        BENCH_ID="${BENCH_ID}" \
        METRIC="${METRIC}" \
        bash "${CODE_EVAL_SH}" || OVERALL_EXIT=$?
    done
    echo "[$(date)] computing summary over ${N_RUNS} runs ..."
    python3 "${WORK_DIR}/compute_summary.py" --bench-dir "${BENCH_DIR}" || OVERALL_EXIT=$?
    EXIT=${OVERALL_EXIT}
    ;;

  *)
    echo "[$(date)] ERROR: unknown RUNNER='${RUNNER}' (expected protocol | external)" >&2
    exit 1
    ;;
esac

echo "[$(date)] ── run_eval.sh DONE (exit=${EXIT}) ──"
exit "${EXIT}"
