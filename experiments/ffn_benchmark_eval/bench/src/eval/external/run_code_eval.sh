#!/usr/bin/env bash
# Code-benchmark eval worker (evalplus, in-process vLLM) — runs on a GPU node.
#
# All-in-one single GPU job:
#   1. Install evalplus from local source (offline node has no DNS / public PyPI).
#   2. Point evalplus at pre-cached datasets (override env + XDG cache redirect).
#   3. evalplus.codegen  (vLLM in-process; continuous batching saturates the GPU).
#   4. evalplus.evaluate (sandbox execution → pass@1).
#   5. Normalize the result to the unified benchmark summary schema.
#
# This worker is shared by all experiments. Submit it via src/scripts/suite/
# submit_benchmark.py (external runner) — NOT by hand — so paths/env are injected
# consistently and it is launched from a minimal cwd (small staging, see §8.1 of
# 经验总结-MLP-Hope-踩坑.md).
#
# Required env vars (injected by the submitter):
#   MODEL_PATH       path to model directory
#   DATASET          evalplus dataset key: humaneval | mbpp
#   OUTPUT_DIR       NFS dir for raw evalplus artifacts + logs
#   RESULT_PATH      NFS path for the normalized unified-schema result JSON
#   BENCH_ID         bench id used in the unified result (e.g. humaneval_plus)
#   METRIC           which pass@1 maps to unified "accuracy" (base_pass_at_1 | plus_pass_at_1)
#
# Optional env vars (have repo-relative defaults; override to relocate):
#   REPO                 repo root (default: derived from this script's location)
#   EVALPLUS_SRC         evalplus source checkout (default: reference/evalplus under suite dir)
#   EVALPLUS_DATASET_DIR pre-cached dataset dir (default: datasets/benchmark/evalplus)
#   TP_SIZE              tensor-parallel override (default: GPU count, i.e. 8 on 8×A100)
#   EVAL_PARALLEL        sandbox eval parallelism (default: 64)

set -euo pipefail

# ── Proxy / network ───────────────────────────────────────────────────────────
# Local-address requests must bypass the platform-injected HTTP proxy (踩坑 §3.3).
export no_proxy="localhost,127.0.0.1,0.0.0.0"
export NO_PROXY="localhost,127.0.0.1,0.0.0.0"

# ── Resolve repo + framework + dataset paths ──────────────────────────────────
# Default REPO = three levels up from this script (src/eval/external/ → repo root).
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "${_SCRIPT_DIR}/../../.." && pwd)}"
EVALPLUS_SRC="${EVALPLUS_SRC:?EVALPLUS_SRC must be set (path to evalplus source checkout)}"
EVALPLUS_DATASET_DIR="${EVALPLUS_DATASET_DIR:-${REPO}/datasets/benchmark/evalplus}"

MODEL_PATH="${MODEL_PATH:?MODEL_PATH must be set}"
DATASET="${DATASET:?DATASET must be set (humaneval or mbpp)}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR must be set}"
RESULT_PATH="${RESULT_PATH:?RESULT_PATH must be set}"
BENCH_ID="${BENCH_ID:?BENCH_ID must be set}"
METRIC="${METRIC:-base_pass_at_1}"

# ── Logging (NFS; platform may drop stderr — see 踩坑 §6.5) ────────────────────
mkdir -p "${OUTPUT_DIR}/logs"
THIS_LOG="${OUTPUT_DIR}/logs/code_eval_${DATASET}_$(hostname)_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${THIS_LOG}") 2>&1

echo "[$(date)] ── run_code_eval.sh ───────────────────────────────────────────"
echo "[$(date)] REPO        = ${REPO}"
echo "[$(date)] MODEL_PATH  = ${MODEL_PATH}"
echo "[$(date)] DATASET     = ${DATASET}"
echo "[$(date)] BENCH_ID    = ${BENCH_ID}"
echo "[$(date)] METRIC      = ${METRIC}"
echo "[$(date)] OUTPUT_DIR  = ${OUTPUT_DIR}"
echo "[$(date)] RESULT_PATH = ${RESULT_PATH}"
echo "[$(date)] Hostname    = $(hostname)"
nvidia-smi -L 2>/dev/null || true

# ── Python env ────────────────────────────────────────────────────────────────
export PYTHONUNBUFFERED=1
export HF_HOME=/tmp/hf_home
export TOKENIZERS_PARALLELISM=false

# ── Detect tensor parallel ────────────────────────────────────────────────────
GPU_COUNT=$(nvidia-smi --list-gpus 2>/dev/null | wc -l || echo 1)
TENSOR_PARALLEL="${TP_SIZE:-${GPU_COUNT}}"
echo "[$(date)] GPU_COUNT=${GPU_COUNT}  TENSOR_PARALLEL=${TENSOR_PARALLEL}"

# ── Install evalplus from source ──────────────────────────────────────────────
# GPU nodes have NO internet/DNS. The published pypi evalplus (0.3.1) computes a
# different dataset cache_path and ALWAYS tries to download the *Plus datasets at
# runtime -> URLError. The local source checkout supports the *_OVERRIDE_PATH env
# vars (set below) that point at pre-cached jsonl and skip all network access.
echo "[$(date)] Installing evalplus from source: ${EVALPLUS_SRC}"
if [ ! -d "${EVALPLUS_SRC}" ]; then
    echo "[$(date)] ERROR: EVALPLUS_SRC not found: ${EVALPLUS_SRC}" >&2
    exit 1
fi
pip install -e "${EVALPLUS_SRC}" \
    -i http://pypi.sankuai.com/simple --trusted-host pypi.sankuai.com \
    --quiet 2>/dev/null \
  || pip install -e "${EVALPLUS_SRC}" \
       -i http://pypi.sankuai.com/simple --trusted-host pypi.sankuai.com

# Verify we got the source checkout (the version that honors override env vars).
python3 -c "
import evalplus.data.mbpp as m, evalplus.data.humaneval as h
assert hasattr(m, 'MBPP_OVERRIDE_PATH'), 'installed evalplus lacks MBPP_OVERRIDE_PATH (wrong version)'
assert hasattr(h, 'HUMANEVAL_OVERRIDE_PATH'), 'installed evalplus lacks HUMANEVAL_OVERRIDE_PATH (wrong version)'
print('evalplus (source) installed OK; override env vars supported')
" || { echo "[$(date)] ERROR: evalplus install/verify failed" >&2; exit 1; }
echo "[$(date)] evalplus ready."

# ── Point evalplus at pre-cached datasets (no network) ────────────────────────
# Belt-and-suspenders (see 踩坑 §8.3):
#  1. *_OVERRIDE_PATH -> the *Plus* jsonl directly.
#  2. XDG_CACHE_HOME -> a writable cache dir pre-seeded with the *base* datasets
#     (HumanEval.jsonl, sanitized-mbpp.json) which have no override env var.
#     evalplus CACHE_DIR = user_cache_dir('evalplus') honors XDG_CACHE_HOME.
#  NOTE: datasets live on the NFS project dir (确定挂载), NOT ~/.cache which is
#        container-local and not mounted inside the job (踩坑 §8.3 坑 B).
export XDG_CACHE_HOME="/tmp/xdg_cache"
SEEDED_CACHE="${XDG_CACHE_HOME}/evalplus"
mkdir -p "${SEEDED_CACHE}"
cp -n "${EVALPLUS_DATASET_DIR}/HumanEvalPlus-v0.1.10.jsonl" \
      "${EVALPLUS_DATASET_DIR}/MbppPlus-v0.2.0.jsonl" \
      "${EVALPLUS_DATASET_DIR}/HumanEval.jsonl" \
      "${EVALPLUS_DATASET_DIR}/sanitized-mbpp.json" \
      "${SEEDED_CACHE}/" 2>/dev/null || true

export HUMANEVAL_OVERRIDE_PATH="${EVALPLUS_DATASET_DIR}/HumanEvalPlus-v0.1.10.jsonl"
export MBPP_OVERRIDE_PATH="${EVALPLUS_DATASET_DIR}/MbppPlus-v0.2.0.jsonl"
echo "[$(date)] EVALPLUS_DATASET_DIR    = ${EVALPLUS_DATASET_DIR}"
echo "[$(date)] XDG_CACHE_HOME          = ${XDG_CACHE_HOME}"
echo "[$(date)] HUMANEVAL_OVERRIDE_PATH = ${HUMANEVAL_OVERRIDE_PATH}"
echo "[$(date)] MBPP_OVERRIDE_PATH      = ${MBPP_OVERRIDE_PATH}"

# fail-fast: required dataset files readable from inside the container.
for f in "${HUMANEVAL_OVERRIDE_PATH}" "${MBPP_OVERRIDE_PATH}" \
         "${EVALPLUS_DATASET_DIR}/HumanEval.jsonl" "${EVALPLUS_DATASET_DIR}/sanitized-mbpp.json"; do
    if [ ! -s "${f}" ]; then
        echo "[$(date)] ERROR: required cached dataset missing/unreadable: ${f}" >&2
        ls -la "${EVALPLUS_DATASET_DIR}/" || true
        exit 1
    fi
done

# fail-fast: confirm evalplus resolves CACHE_DIR to our seeded dir.
python3 -c "
from evalplus.data.utils import CACHE_DIR
import os
print(f'[verify] evalplus CACHE_DIR = {CACHE_DIR}')
assert os.path.exists(os.path.join(CACHE_DIR,'HumanEval.jsonl')), 'base HumanEval.jsonl not in CACHE_DIR'
assert os.path.exists(os.path.join(CACHE_DIR,'sanitized-mbpp.json')), 'sanitized-mbpp.json not in CACHE_DIR'
print('[verify] base datasets present in CACHE_DIR')
" || { echo "[$(date)] ERROR: evalplus CACHE_DIR missing seeded base datasets" >&2; exit 1; }
echo "[$(date)] All cached datasets present and resolvable."

# ── Step 1: Code Generation ───────────────────────────────────────────────────
SAMPLES_DIR="${OUTPUT_DIR}/${DATASET}"
mkdir -p "${SAMPLES_DIR}"
echo "[$(date)] ═══ Step 1: Code Generation (${DATASET}) ═══"
echo "[$(date)] Backend: vllm in-process, TP=${TENSOR_PARALLEL}, greedy(pass@1), force-base-prompt"

python3 -m evalplus.codegen \
    --model "${MODEL_PATH}" \
    --dataset "${DATASET}" \
    --backend vllm \
    --greedy \
    --force-base-prompt \
    --tp "${TENSOR_PARALLEL}" \
    --trust-remote-code \
    --root "${SAMPLES_DIR}"
echo "[$(date)] Code generation complete."

SAMPLES_FILE=$(find "${SAMPLES_DIR}" -name "*.jsonl" -not -name "*.raw.jsonl" | head -1)
if [ -z "${SAMPLES_FILE}" ]; then
    echo "[$(date)] ERROR: No samples.jsonl found in ${SAMPLES_DIR}" >&2
    ls -la "${SAMPLES_DIR}/"
    exit 1
fi
echo "[$(date)] Samples file: ${SAMPLES_FILE}  ($(wc -l < "${SAMPLES_FILE}") rows)"

# ── Step 2: Evaluation (sandbox execution) ────────────────────────────────────
echo "[$(date)] ═══ Step 2: Evaluation (sandbox execution) ═══"
# Remove stale eval_results to avoid interactive overwrite prompt from evalplus
find "${SAMPLES_DIR}" -name "*.eval_results.json" -delete 2>/dev/null || true
python3 -m evalplus.evaluate \
    --dataset "${DATASET}" \
    --samples "${SAMPLES_FILE}" \
    --parallel "${EVAL_PARALLEL:-64}" \
    --i-just-wanna-run
echo "[$(date)] Evaluation complete."

# ── Step 3: Normalize to unified benchmark summary schema ─────────────────────
# Run from REPO so `python3 -m src.eval.external.code_eval` resolves the package
# regardless of the (minimal) job launch cwd.
echo "[$(date)] ═══ Step 3: Normalize result → unified schema ═══"
( cd "${REPO}" && \
  DATASET="${DATASET}" OUTPUT_DIR="${OUTPUT_DIR}" MODEL_PATH="${MODEL_PATH}" \
  RESULT_PATH="${RESULT_PATH}" BENCH_ID="${BENCH_ID}" METRIC="${METRIC}" \
  python3 -m src.eval.external.code_eval normalize ) \
  || { echo "[$(date)] ERROR: result normalization failed" >&2; exit 1; }

echo "[$(date)] ── run_code_eval.sh DONE ──────────────────────────────────────"
