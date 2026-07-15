"""Single dispatch point for the pre-registered matrix multiplication backends."""

from __future__ import annotations

import torch


def linear(x: torch.Tensor, weight: torch.Tensor, backend: str = "torch_bf16") -> torch.Tensor:
    """Evaluate ``F.linear(x, weight)`` with the selected frozen backend."""
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise TypeError(f"{backend} requires BF16 operands, got {x.dtype} and {weight.dtype}")
    if backend == "torch_bf16":
        return torch.nn.functional.linear(x, weight)
    if backend == "vllm_bi":
        from vllm.model_executor.layers.batch_invariant import linear_batch_invariant
        return linear_batch_invariant(x, weight)
    raise ValueError(f"unsupported backend {backend!r}")


def matmul(a: torch.Tensor, b: torch.Tensor, backend: str = "torch_bf16") -> torch.Tensor:
    """Legacy Stage-1 orientation: return ``a @ b``.

    Amendment v1.1 uses :func:`linear`; the original Stage 1 remains exactly
    reproducible through this wrapper.
    """
    if backend != "torch_bf16":
        raise ValueError("legacy matmul is frozen to torch_bf16")
    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
        raise TypeError(f"torch_bf16 requires BF16 operands, got {a.dtype} and {b.dtype}")
    return torch.matmul(a, b)
