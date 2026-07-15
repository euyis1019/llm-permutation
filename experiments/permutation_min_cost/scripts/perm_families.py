"""Pre-registered permutation families for permutation_min_cost."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from permutation import check_bijection

M = 9728
SEEDS = (201, 202, 203, 204, 205)
V11_SEEDS = (301, 302, 303, 304, 305)


@dataclass(frozen=True)
class PermSpec:
    family: str
    parameter: Any
    seed: int | None

    @property
    def key(self) -> str:
        p = "none" if self.parameter is None else str(self.parameter)
        s = "none" if self.seed is None else str(self.seed)
        return f"{self.family}:p={p}:s={s}"


def all_specs() -> list[PermSpec]:
    out: list[PermSpec] = []
    seeded = {
        "F1_window_shuffle": (2, 8, 32, 128, 512, 2048, 9728),
        "F2_block_local": (0.05, 0.10, 0.30, 0.50),
        "F3_scattered_global": (0.05, 0.10, 0.30, 0.50),
        "F4_adjacent_pairs": (0.10, 0.30, 0.50, 1.00),
        "F5_strided_pairs": (1, 4, 16, 64, 256, 1024, 4096),
        "F6_block_swap": (0.10, 0.30),
        "F7_global_random": (None,),
    }
    for family, params in seeded.items():
        for parameter in params:
            for seed in SEEDS:
                out.append(PermSpec(family, parameter, seed))
    out.extend([
        PermSpec("F8_reverse", None, None),
        PermSpec("F8_identity", None, None),
    ])
    return out


def _generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(int(seed))


def _random_derangement(n: int, g: torch.Generator) -> torch.Tensor:
    if n < 2:
        raise ValueError("a derangement requires at least two items")
    base = torch.arange(n)
    while True:
        q = torch.randperm(n, generator=g)
        if not bool((q == base).any()):
            return q


def _strided_matching(m: int, distance: int) -> list[tuple[int, int]]:
    """One maximum matching of edges (i, i+D) in disjoint residue paths."""
    edges: list[tuple[int, int]] = []
    for residue in range(distance):
        path = list(range(residue, m, distance))
        edges.extend((path[j], path[j + 1]) for j in range(0, len(path) - 1, 2))
    return edges


def make_family_perm(
    family: str,
    m: int = M,
    parameter: Any = None,
    seed: int | None = None,
) -> torch.Tensor:
    p = torch.arange(m, dtype=torch.int64)
    if family == "F8_identity":
        pass
    elif family == "F8_reverse":
        p = torch.arange(m - 1, -1, -1, dtype=torch.int64)
    else:
        if seed is None:
            raise ValueError(f"{family} requires a seed")
        g = _generator(seed)
        if family == "F1_window_shuffle":
            width = int(parameter)
            if width <= 0:
                raise ValueError("window width must be positive")
            for start in range(0, m, width):
                stop = min(start + width, m)
                p[start:stop] = start + torch.randperm(stop - start, generator=g)
        elif family == "F2_block_local":
            length = int(float(parameter) * m)
            start = int(torch.randint(0, m - length + 1, (1,), generator=g).item())
            p[start:start + length] = start + torch.randperm(length, generator=g)
        elif family == "F3_scattered_global":
            count = int(float(parameter) * m)
            chosen = torch.randperm(m, generator=g)[:count]
            q = _random_derangement(count, g)
            p[chosen] = chosen[q]
        elif family == "F4_adjacent_pairs":
            count = int(float(parameter) * m / 2)
            pair_ids = torch.randperm(m // 2, generator=g)[:count]
            left = pair_ids * 2
            p[left], p[left + 1] = left + 1, left
        elif family == "F5_strided_pairs":
            distance = int(parameter)
            count = int(0.30 * m / 2)
            matching = _strided_matching(m, distance)
            if len(matching) < count:
                raise ValueError(f"not enough disjoint D={distance} pairs")
            order = torch.randperm(len(matching), generator=g)[:count].tolist()
            for j in order:
                a, b = matching[j]
                p[a], p[b] = b, a
        elif family == "F6_block_swap":
            length = int(float(parameter) * m / 2)
            half = m // 2
            start = int(torch.randint(0, half - length + 1, (1,), generator=g).item())
            a = torch.arange(start, start + length)
            b = a + half
            p[a], p[b] = b, a
        elif family == "F7_global_random":
            p = torch.randperm(m, generator=g)
        else:
            raise ValueError(f"unknown family {family!r}")
    if not check_bijection(p, m):
        raise AssertionError(f"{family} did not produce a bijection")
    return p


def make_from_spec(spec: PermSpec, m: int = M, layer_idx: int | None = None) -> torch.Tensor:
    seed = spec.seed
    if seed is not None and layer_idx is not None:
        seed = seed * 1000 + layer_idx
    return make_family_perm(spec.family, m, spec.parameter, seed)


def all_specs_v11() -> list[PermSpec]:
    """The 52 fresh Amendment-v1.1 permutation instances, in frozen order."""
    out: list[PermSpec] = []
    for parameter in (0.05, 0.30, 1.00):
        out.extend(PermSpec("F9_inblock_shuffle", parameter, s) for s in V11_SEEDS)
    out.extend(PermSpec("F10_odd_pairs", 0.30, s) for s in V11_SEEDS)
    out.append(PermSpec("F10_odd_pairs", 1.00, None))
    for offset in (0, 4):
        out.extend(PermSpec("F11_window_offset", offset, s) for s in V11_SEEDS)
    out.extend(PermSpec("F12_win16_aligned", None, s) for s in V11_SEEDS)
    for parameter in (0.05, 0.30):
        out.extend(PermSpec("F3_scattered_global", parameter, s) for s in V11_SEEDS)
    out.extend(PermSpec("F7_global_random", None, s) for s in V11_SEEDS)
    out.append(PermSpec("F8_identity", None, None))
    assert len(out) == 52
    return out


def _nonidentity_shuffle(n: int, g: torch.Generator) -> torch.Tensor:
    base = torch.arange(n)
    while True:
        q = torch.randperm(n, generator=g)
        if not torch.equal(q, base):
            return q


def make_family_perm_v11(spec: PermSpec, m: int = M) -> torch.Tensor:
    """Construct one frozen Amendment-v1.1 permutation."""
    family, parameter, seed = spec.family, spec.parameter, spec.seed
    if family in {"F3_scattered_global", "F7_global_random", "F8_identity"}:
        # Reuse the original implementation verbatim with the fresh v1.1 seed.
        return make_family_perm(family, m, parameter, seed)

    p = torch.arange(m, dtype=torch.int64)
    if family == "F10_odd_pairs" and float(parameter) == 1.0:
        # All valid odd-aligned candidates; endpoints 0 and M-1 remain fixed.
        for left in range(1, m - 1, 2):
            p[left], p[left + 1] = left + 1, left
    else:
        if seed is None:
            raise ValueError(f"{family} requires a seed")
        g = _generator(seed)
        if family == "F9_inblock_shuffle":
            n_blocks = min(m // 8, __import__("math").ceil(float(parameter) * m / 8))
            chosen = torch.randperm(m // 8, generator=g)[:n_blocks]
            for block in chosen.tolist():
                start = block * 8
                p[start:start + 8] = start + _nonidentity_shuffle(8, g)
        elif family == "F10_odd_pairs":
            candidates = torch.arange(1, m - 1, 2, dtype=torch.int64)
            count = min(len(candidates), __import__("math").ceil(float(parameter) * m / 2))
            chosen = candidates[torch.randperm(len(candidates), generator=g)[:count]]
            p[chosen] = chosen + 1
            p[chosen + 1] = chosen
        elif family == "F11_window_offset":
            offset = int(parameter)
            for start in range(offset, m - 8 + 1, 8):
                p[start:start + 8] = start + torch.randperm(8, generator=g)
        elif family == "F12_win16_aligned":
            for start in range(0, m, 16):
                p[start:start + 16] = start + torch.randperm(16, generator=g)
        else:
            raise ValueError(f"unknown v1.1 family {family!r}")
    if not check_bijection(p, m):
        raise AssertionError(f"{family} did not produce a bijection")
    return p


def predicted_tier_v11(perm: torch.Tensor) -> str:
    """Apply the pre-registered three-tier label rule pointwise."""
    idx = torch.arange(perm.numel(), dtype=torch.int64)
    if bool(torch.equal(idx // 8, perm.cpu() // 8)):
        return "zero"
    if bool(torch.equal(idx // 16, perm.cpu() // 16)):
        return "sub"
    return "ceil"
