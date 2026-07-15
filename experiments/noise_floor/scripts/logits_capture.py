"""Non-mutating vLLM V1 logits processor that persists raw logits.

The capture destination is supplied per request in
``SamplingParams.extra_args["noise_floor_capture"]``.  The processor returns
the input tensor unchanged and is therefore argmax invariant.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
from vllm.v1.sample.logits_processor.interface import (
    BatchUpdate,
    LogitsProcessor,
    MoveDirectionality,
)


def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _atomic_json(path: Path, obj: dict) -> None:
    payload = json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")
    _atomic_bytes(path, payload)


class RawLogitsCapture(LogitsProcessor):
    """Save each request's pre-sampling logits without changing them."""

    def __init__(self, vllm_config, device: torch.device, is_pin_memory: bool):
        self._destinations: dict[int, str] = {}

    @classmethod
    def validate_params(cls, sampling_params):
        return None

    def is_argmax_invariant(self) -> bool:
        # vLLM's all-greedy fast path skips processors declared invariant.
        # Returning False places this read-only identity processor on the
        # mandatory pre-greedy path; apply() still returns logits unchanged.
        return False

    def update_state(self, batch_update: BatchUpdate | None) -> None:
        if batch_update is None:
            return
        for idx in batch_update.removed:
            self._destinations.pop(idx, None)
        for idx, params, _prompt_ids, _output_ids in batch_update.added:
            extra = params.extra_args or {}
            dest = extra.get("noise_floor_capture")
            if dest:
                self._destinations[idx] = str(dest)
            else:
                self._destinations.pop(idx, None)
        for src, dst, direction in batch_update.moved:
            src_value = self._destinations.get(src)
            dst_value = self._destinations.get(dst)
            if src_value is None:
                self._destinations.pop(dst, None)
            else:
                self._destinations[dst] = src_value
            if direction == MoveDirectionality.SWAP:
                if dst_value is None:
                    self._destinations.pop(src, None)
                else:
                    self._destinations[src] = dst_value
            else:
                self._destinations.pop(src, None)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        for idx, stem_text in tuple(self._destinations.items()):
            if idx >= logits.shape[0]:
                continue
            stem = Path(stem_text)
            meta_path = stem.with_suffix(".meta.json")
            if meta_path.exists():
                continue
            row = logits[idx].detach().contiguous().cpu()
            raw = row.view(torch.uint8).numpy().tobytes(order="C")
            raw_path = stem.with_suffix(".raw.bin")
            _atomic_bytes(raw_path, raw)

            f32_path = stem.with_suffix(".float32.npy")
            fd, tmp = tempfile.mkstemp(dir=stem.parent, suffix=".tmp")
            try:
                with os.fdopen(fd, "wb") as f:
                    np.save(f, row.float().numpy(), allow_pickle=False)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp, f32_path)
            except BaseException:
                if os.path.exists(tmp):
                    os.unlink(tmp)
                raise

            _atomic_json(
                meta_path,
                {
                    "shape": list(row.shape),
                    "torch_dtype": str(row.dtype),
                    "numel": row.numel(),
                    "element_size": row.element_size(),
                    "raw_file": raw_path.name,
                    "float32_file": f32_path.name,
                    "raw_sha256": hashlib.sha256(raw).hexdigest(),
                },
            )
        return logits
