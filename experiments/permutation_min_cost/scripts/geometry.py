"""Pre-registered geometry metrics for a permutation."""

from __future__ import annotations

import hashlib

import numpy as np
import torch

from permutation import check_bijection


def _inversion_fraction_mc(perm: torch.Tensor, n_pairs: int = 1_000_000) -> float:
    # The PRNG seed is derived from P, making the estimate deterministic and a
    # function of the permutation rather than execution order.
    raw = perm.contiguous().numpy().tobytes()
    seed = int.from_bytes(hashlib.sha256(raw).digest()[:8], "little") % (2**63 - 1)
    rng = np.random.default_rng(seed)
    m = int(perm.numel())
    a = rng.integers(0, m, size=n_pairs, dtype=np.int64)
    b = rng.integers(0, m, size=n_pairs, dtype=np.int64)
    equal = a == b
    while bool(equal.any()):
        b[equal] = rng.integers(0, m, size=int(equal.sum()), dtype=np.int64)
        equal = a == b
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    pn = perm.numpy()
    return float(np.mean(pn[lo] > pn[hi]))


def geometry_metrics(perm: torch.Tensor, n_inversion_pairs: int = 1_000_000) -> dict:
    perm = perm.detach().cpu().to(torch.int64).contiguous()
    m = int(perm.numel())
    if not check_bijection(perm, m):
        raise ValueError("perm is not a bijection")
    idx = torch.arange(m)
    disp = (perm - idx).abs()
    moved = disp != 0
    n_moved = int(moved.sum().item())
    out = {
        "frac_moved": n_moved / m,
        "mean_disp": float(disp[moved].double().mean().item()) if n_moved else 0.0,
        "max_disp": int(disp.max().item()),
        "total_disp": float(disp.double().sum().item() / m),
    }
    for block in (16, 64, 256):
        out[f"cross_{block}"] = float(((idx // block) != (perm // block)).double().mean().item())
    out["inversions"] = _inversion_fraction_mc(perm, n_inversion_pairs)
    return out
