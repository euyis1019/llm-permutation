#!/usr/bin/env bash
# Build a "shadow" model dir that is safe to patch without touching the original
# (zhangsiyuan22 / liupei09 checkpoints are read-only shared artifacts).
#
# WHY: LongCat custom remote-code (modeling_longcat_clean.py + config.json) is
# incompatible with the image's transformers 5.8.x + vLLM 0.21 MoE backend in
# three ways (rope API removed, create_causal_mask signature change, MoE field
# names vLLM can't find). patch_longcat_model.py fixes all three IN PLACE, so we
# must run it on a COPY, not the source.
#
# Strategy: symlink the big weight/tokenizer files, hard-copy only the small files
# that the patch rewrites (*.py + *.json), then patch the copy. Idempotent: a
# completed shadow (marker file) is reused.
#
# Usage: make_patched_model.sh <SRC_MODEL_DIR> <SHADOW_DIR>
#   echoes the shadow dir path on success (so caller can capture it).
set -euo pipefail

SRC="${1:?SRC model dir required}"
SHADOW="${2:?SHADOW dir required}"
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_PY="${SELF_DIR}/patch_longcat_model.py"

MARKER="${SHADOW}/.patched_ok"
if [ -f "${MARKER}" ]; then
    echo "[make_patched_model] reuse existing patched shadow: ${SHADOW}" >&2
    echo "${SHADOW}"
    exit 0
fi

rm -rf "${SHADOW}"
mkdir -p "${SHADOW}"

# Files the patch rewrites must be real copies; everything else is symlinked.
for f in "${SRC}"/* "${SRC}"/.[!.]*; do
    [ -e "${f}" ] || continue
    base="$(basename "${f}")"
    case "${base}" in
        *.py|*.json)
            cp "${f}" "${SHADOW}/${base}"
            ;;
        *)
            ln -s "${f}" "${SHADOW}/${base}"
            ;;
    esac
done

echo "[make_patched_model] patching shadow ${SHADOW}" >&2
python3 "${PATCH_PY}" "${SHADOW}" >&2

touch "${MARKER}"
echo "[make_patched_model] done: ${SHADOW}" >&2
echo "${SHADOW}"
