"""In-memory vLLM weight mutations for noise_floor Part 1a."""

from __future__ import annotations

import hashlib

import torch


M = 9728


def _generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(int(seed))


def _nonidentity_shuffle(n: int, g: torch.Generator) -> torch.Tensor:
    base = torch.arange(n)
    while True:
        q = torch.randperm(n, generator=g)
        if not torch.equal(q, base):
            return q


def make_perm(variant: str, layer_idx: int, m: int = M) -> torch.Tensor:
    if variant == "f9_k100":
        g = _generator(401 + layer_idx)
        p = torch.arange(m, dtype=torch.int64)
        for start in range(0, m, 8):
            p[start : start + 8] = start + _nonidentity_shuffle(8, g)
        return p
    if variant == "f10_k100":
        p = torch.arange(m, dtype=torch.int64)
        for left in range(1, m - 1, 2):
            p[left], p[left + 1] = left + 1, left
        return p
    if variant == "f7":
        return torch.randperm(m, generator=_generator(402 + layer_idx))
    if variant == "identity":
        return torch.arange(m, dtype=torch.int64)
    raise ValueError(f"unknown variant: {variant}")


@torch.no_grad()
def apply_vllm_variant(model, variant: str) -> dict:
    """Apply the joint FFN permutation to vLLM's fused gate/up layout."""
    if variant == "identity":
        return {"variant": variant, "layers": 0, "perm_sha256": []}
    layers = model.model.layers
    hashes = []
    for layer_idx, layer in enumerate(layers):
        p_cpu = make_perm(variant, layer_idx)
        hashes.append(hashlib.sha256(p_cpu.numpy().tobytes()).hexdigest())
        mlp = layer.mlp
        gate_up = mlp.gate_up_proj.weight
        down = mlp.down_proj.weight
        m = p_cpu.numel()
        if gate_up.shape[0] != 2 * m or down.shape[1] != m:
            raise AssertionError(
                f"unexpected fused MLP shapes L{layer_idx}: "
                f"gate_up={tuple(gate_up.shape)} down={tuple(down.shape)}"
            )
        p = p_cpu.to(gate_up.device)
        gate_up[:m].copy_(gate_up[:m].index_select(0, p))
        gate_up[m : 2 * m].copy_(gate_up[m : 2 * m].index_select(0, p))
        down.copy_(down.index_select(1, p.to(down.device)))
    return {"variant": variant, "layers": len(layers), "perm_sha256": hashes}

