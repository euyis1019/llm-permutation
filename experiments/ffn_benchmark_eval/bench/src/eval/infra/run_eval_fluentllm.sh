#!/usr/bin/env bash
# FluentLLM eval driver (收口版) — ONE benchmark per job.
#
# The FluentLLM engine counterpart of run_eval.sh. Both drivers live under
# src/eval/infra/ and are dispatched by submit_eval.py via the --engine flag:
#   --engine vllm       → run_eval.sh            (in-process vLLM)
#   --engine fluentllm  → run_eval_fluentllm.sh  (SGLang server + HTTP client, THIS)
#
# Architecture: each Hope job runs ONE (checkpoint, bench) pair.
#   1. FluentLLM (SGLang) server starts (4-GPU TP).
#   2. Only the bench specified by BENCH_ID is evaluated against the endpoint.
#   3. Results written to OUTPUT_BASE/<MODEL_TAG>/<BENCH_ID>/.
#
# Execution-layer assets (compute_summary.py, reference_evalplus) live alongside
# this driver under src/eval/infra/ (WORK_DIR). The code-completion generator
# lives under src/eval/external/. This driver no longer depends on the old
# experiments/12_.../eval_infra directory.
#
# Required env vars (injected by submit_eval.py):
#   MODEL_PATH   absolute NFS path to HF-format model
#   MODEL_TAG    stable tag (e.g. flash3b_orig)
#   OUTPUT_BASE  absolute NFS output root
#   BENCH_ID     one of: gsm8k ceval cmmlu cruxeval mmlu humaneval_plus mbpp_plus
#
# Optional:
#   N_RUNS       protocol bench runs  (default: 3)
#   N_CODE_RUNS  code bench runs      (default: 1)
#   PORT         FluentLLM port       (default: 8080)

set -euo pipefail

export no_proxy="localhost,127.0.0.1,0.0.0.0"
export NO_PROXY="localhost,127.0.0.1,0.0.0.0"

# REPO = repo root, derived from this script's location (src/eval/infra/ → repo root,
# three levels up). No hardcoded absolute path — portable across platforms.
# compute_summary.py and reference_evalplus live alongside this driver.
WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${WORK_DIR}/../../.." && pwd)"
GEN_CODE_PY="${REPO}/src/eval/external/gen_code_completions.py"

MODEL_PATH="${MODEL_PATH:?MODEL_PATH must be set}"
MODEL_TAG="${MODEL_TAG:?MODEL_TAG must be set}"
OUTPUT_BASE="${OUTPUT_BASE:?OUTPUT_BASE must be set}"
BENCH_ID="${BENCH_ID:?BENCH_ID must be set}"
N_RUNS="${N_RUNS:-3}"
N_CODE_RUNS="${N_CODE_RUNS:-1}"
PORT="${PORT:-8080}"
ENDPOINT="http://localhost:${PORT}"

PROTOCOL_BENCHES=(gsm8k ceval cmmlu cruxeval mmlu)
CODE_BENCHES=(humaneval_plus mbpp_plus)

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_DIR="${OUTPUT_BASE}/${MODEL_TAG}/logs"
mkdir -p "${LOG_DIR}"
THIS_LOG="${LOG_DIR}/eval_${BENCH_ID}_$(hostname)_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${THIS_LOG}") 2>&1

echo "============================================================"
echo "  FluentLLM Eval — ${MODEL_TAG}  bench=${BENCH_ID}"
echo "  $(date)  host=$(hostname)"
echo "============================================================"
echo "  MODEL_PATH  = ${MODEL_PATH}"
echo "  OUTPUT_BASE = ${OUTPUT_BASE}"
echo "  WORK_DIR    = ${WORK_DIR}"
nvidia-smi -L 2>/dev/null || true

# ── Activate FluentLLM env ────────────────────────────────────────────────────
source /home/fluentllmenv/bin/activate
export EPS_HOME=/home/fluentllm/3rdparty/eps/
export PYTHONPATH="${EPS_HOME}/python/:${REPO}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HF_HOME=/tmp/hf_home

# 500 concurrent in-flight requests: FluentLLM continuous batching keeps GPU
# saturated. Serial (=1) leaves GPU idle between requests.
export BENCH_CONCURRENCY=500

# ── Start FluentLLM server ────────────────────────────────────────────────────
SERVER_LOG="${LOG_DIR}/server_${BENCH_ID}_$(hostname)_$(date +%Y%m%d_%H%M%S).log"
echo "[$(date)] Starting FluentLLM server on port ${PORT} ..."
python3 -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --trust-remote-code \
    --mem-fraction-static 0.8 \
    --port "${PORT}" \
    --host 0.0.0.0 \
    --attention-backend triton \
    --chunked-prefill-size 4096 \
    --context-length 8192 \
    --low-latency-max-num-tokens-per-gpu 4096 \
    --max-running-requests 256 \
    --moe-parallel-strategy tp \
    --dist-init-addr 127.0.0.1:3000 \
    --nnodes 1 --node-rank 0 \
    --nprocs-per-node 4 --attn-tp-size 4 \
    --log-level info \
    --disable-overlap-schedule \
    --cuda-graph-max-bs 96 \
    --capture-sample-graph \
    --chunker-backend flashinfer \
    >> "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
echo "[$(date)] Server PID=${SERVER_PID}"

# ── Wait for server ready ─────────────────────────────────────────────────────
echo "[$(date)] Waiting for server startup (up to 600s)..."
READY=0
for i in $(seq 1 120); do
    sleep 5
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "[$(date)] ERROR: server process died"
        tail -30 "${SERVER_LOG}" || true
        exit 1
    fi
    if grep -q -E "Application startup complete|Uvicorn running on" \
               "${SERVER_LOG}" 2>/dev/null; then
        READY=1; break
    fi
    echo "[$(date)] ... waiting (${i}/120, $((i*5))s elapsed)"
done

if [ "${READY}" -eq 0 ]; then
    echo "[$(date)] ERROR: server did not start within 600s"
    kill "${SERVER_PID}" 2>/dev/null || true
    exit 1
fi
echo "[$(date)] Server ready. 15s grace period..."
sleep 15

OVERALL_EXIT=0

# ── Protocol benchmarks ───────────────────────────────────────────────────────
is_protocol=false
for b in "${PROTOCOL_BENCHES[@]}"; do
    [ "$b" = "$BENCH_ID" ] && is_protocol=true && break
done

if $is_protocol; then
    META_PATH="${REPO}/datasets/benchmark/normalized/${BENCH_ID}/benchmark_meta.json"
    if [ ! -f "${META_PATH}" ]; then
        echo "[$(date)] ERROR: benchmark_meta.json not found: ${META_PATH}"
        exit 1
    fi
    BENCH_DIR="${OUTPUT_BASE}/${MODEL_TAG}/${BENCH_ID}"
    mkdir -p "${BENCH_DIR}"

    echo ""
    echo "[$(date)] ══ ${BENCH_ID} (${N_RUNS} runs, concurrency=${BENCH_CONCURRENCY}) ══"
    for i in $(seq 0 $((N_RUNS - 1))); do
        RUN_TAG=$(printf "run_%02d" "${i}")
        RUN_DIR="${BENCH_DIR}/${RUN_TAG}"
        RESULT="${RUN_DIR}/result.json"
        RAW="${RUN_DIR}/raw.json"

        if [ -f "${RESULT}" ]; then
            echo "[$(date)] ${BENCH_ID}/${RUN_TAG}: already exists, skipping"
            continue
        fi
        mkdir -p "${RUN_DIR}"
        echo "[$(date)] ${BENCH_ID}/${RUN_TAG}: running..."

        python3 - <<PYEOF
import sys, traceback
sys.path.insert(0, "${REPO}")
try:
    from src.eval.benchmark.client_runner import run_bench_from_meta
    summary = run_bench_from_meta(
        meta_path="${META_PATH}",
        bench_id="${BENCH_ID}",
        mode="remote_vllm_service",
        endpoint="${ENDPOINT}",
        result_path="${RESULT}",
        raw_path="${RAW}",
        timeout=360,
    )
    print(f"  [${RUN_TAG}] accuracy={summary['accuracy']:.4f} "
          f"correct={summary['correct']}/{summary['total']} "
          f"errors={summary['errors']}")
except Exception:
    traceback.print_exc()
    sys.stdout.flush()
    raise
PYEOF
        RUN_EXIT=$?
        if [ "${RUN_EXIT}" -ne 0 ]; then
            echo "[$(date)] ERROR: ${BENCH_ID}/${RUN_TAG} failed"
            OVERALL_EXIT=${RUN_EXIT}
        fi
    done

    echo "[$(date)] ${BENCH_ID}: computing summary..."
    python3 "${WORK_DIR}/compute_summary.py" --bench-dir "${BENCH_DIR}" || OVERALL_EXIT=1
fi

# ── Code benchmarks ───────────────────────────────────────────────────────────
is_code=false
for b in "${CODE_BENCHES[@]}"; do
    [ "$b" = "$BENCH_ID" ] && is_code=true && break
done

if $is_code; then
    EVALPLUS_SRC="${WORK_DIR}/reference_evalplus"
    EVALPLUS_DATASET_DIR="${REPO}/datasets/benchmark/evalplus"

    echo ""
    echo "[$(date)] Setting up evalplus via PYTHONPATH ..."
    EXTRAS_DIR="/tmp/pip_evalplus_extras"
    mkdir -p "${EXTRAS_DIR}"
    pip install fire tree_sitter_python \
        -i http://pypi.sankuai.com/simple --trusted-host pypi.sankuai.com \
        --quiet --target "${EXTRAS_DIR}" 2>/dev/null || true
    export PYTHONPATH="${EVALPLUS_SRC}:${EXTRAS_DIR}:${PYTHONPATH:-}"

    export XDG_CACHE_HOME="/tmp/xdg_cache"
    mkdir -p "${XDG_CACHE_HOME}/evalplus"
    cp -n "${EVALPLUS_DATASET_DIR}/HumanEval.jsonl" \
          "${EVALPLUS_DATASET_DIR}/sanitized-mbpp.json" \
          "${XDG_CACHE_HOME}/evalplus/" 2>/dev/null || true
    export HUMANEVAL_OVERRIDE_PATH="${EVALPLUS_DATASET_DIR}/HumanEvalPlus-v0.1.10.jsonl"
    export MBPP_OVERRIDE_PATH="${EVALPLUS_DATASET_DIR}/MbppPlus-v0.2.0.jsonl"

    python3 -c "
from evalplus.data import get_human_eval_plus, get_mbpp_plus
print('evalplus data import OK')
"

    case "${BENCH_ID}" in
        humaneval_plus) DATASET=humaneval ;;
        mbpp_plus)      DATASET=mbpp      ;;
    esac
    METRIC="base_pass_at_1"

    BENCH_DIR="${OUTPUT_BASE}/${MODEL_TAG}/${BENCH_ID}"
    mkdir -p "${BENCH_DIR}"

    echo ""
    echo "[$(date)] ══ ${BENCH_ID} (${N_CODE_RUNS} runs) ══"
    for i in $(seq 0 $((N_CODE_RUNS - 1))); do
        RUN_TAG=$(printf "run_%02d" "${i}")
        RUN_DIR="${BENCH_DIR}/${RUN_TAG}"
        RESULT="${RUN_DIR}/result.json"
        SAMPLES_JSONL="${RUN_DIR}/${DATASET}/default_vllm_temp_0.0.jsonl"

        if [ -f "${RESULT}" ]; then
            echo "[$(date)] ${BENCH_ID}/${RUN_TAG}: already exists, skipping"
            continue
        fi
        mkdir -p "${RUN_DIR}/${DATASET}"

        echo "[$(date)] ${BENCH_ID}/${RUN_TAG}: code generation..."
        python3 "${GEN_CODE_PY}" \
            --dataset "${DATASET}" \
            --endpoint "${ENDPOINT}" \
            --output-jsonl "${SAMPLES_JSONL}" \
            --n-samples 1 \
            --max-tokens 768 \
            --workers 64 \
            --timeout 180

        echo "[$(date)] ${BENCH_ID}/${RUN_TAG}: evaluation (sandbox)..."
        find "${RUN_DIR}/${DATASET}" -name "*.eval_results.json" -delete 2>/dev/null || true
        python3 -m evalplus.evaluate \
            --dataset "${DATASET}" \
            --samples "${SAMPLES_JSONL}" \
            --parallel 64 \
            --i-just-wanna-run

        echo "[$(date)] ${BENCH_ID}/${RUN_TAG}: normalizing..."
        ( cd "${REPO}" && \
          DATASET="${DATASET}" OUTPUT_DIR="${RUN_DIR}" MODEL_PATH="${MODEL_PATH}" \
          RESULT_PATH="${RESULT}" BENCH_ID="${BENCH_ID}" METRIC="${METRIC}" \
          python3 -m src.eval.external.code_eval normalize ) || OVERALL_EXIT=1

        if [ -f "${RESULT}" ]; then
            ACC=$(python3 -c "import json; d=json.load(open('${RESULT}')); print(d['accuracy'])" 2>/dev/null || echo "?")
            echo "[$(date)] ${BENCH_ID}/${RUN_TAG}: accuracy=${ACC}"
        fi
    done

    echo "[$(date)] ${BENCH_ID}: computing summary..."
    python3 "${WORK_DIR}/compute_summary.py" --bench-dir "${BENCH_DIR}" || OVERALL_EXIT=1
fi

# ── Stop server ───────────────────────────────────────────────────────────────
echo ""
echo "[$(date)] Stopping server (PID=${SERVER_PID})..."
kill "${SERVER_PID}" 2>/dev/null || true
sleep 3

echo "[$(date)] ══ DONE  bench=${BENCH_ID}  exit=${OVERALL_EXIT} ══"
exit "${OVERALL_EXIT}"
