"""Shared permutation / restore / checksum / metric utilities for the
Qwen3-4B FFN permutation experiments.

Convention (pre-registered in ffn_permutation_experiment_plan.md §3):

  A permutation is a 1-D LongTensor `perm` that is a bijection of [0, m).
  Vector permutation is defined as   z_perm = z[..., perm]      (z_perm = P z)
  Matching weight application:
      gate.weight <- gate.weight[perm, :]     (P W_g)
      up.weight   <- up.weight[perm, :]       (P W_u)
      down.weight <- down.weight[:, perm]     (W_d P^T)
  Inverse:
      inv_perm = argsort(perm)
      gate.weight <- gate.weight[inv_perm, :]
      down.weight <- down.weight[:, inv_perm]
"""

import hashlib
import json
import os
import tempfile

import torch


# ---------------------------------------------------------------------------
# permutation construction
# ---------------------------------------------------------------------------

def make_perm(kind: str, m: int, seed: int | None = None) -> torch.Tensor:
    """Build a permutation of [0, m) of the given kind."""
    if kind == "identity":
        return torch.arange(m)
    if kind == "adjacent_swap":
        p = torch.arange(m)
        even = m - (m % 2)
        p[0:even:2], p[1:even:2] = p[1:even:2].clone(), p[0:even:2].clone()
        return p
    if kind == "reverse":
        return torch.arange(m - 1, -1, -1)
    if kind == "random":
        assert seed is not None, "random perm requires a seed"
        g = torch.Generator().manual_seed(seed)
        return torch.randperm(m, generator=g)
    raise ValueError(f"unknown perm kind: {kind}")


def check_bijection(perm: torch.Tensor, m: int) -> bool:
    return (
        perm.dtype == torch.int64
        and perm.shape == (m,)
        and torch.equal(torch.sort(perm).values, torch.arange(m))
    )


def inverse_perm(perm: torch.Tensor) -> torch.Tensor:
    return torch.argsort(perm)


# ---------------------------------------------------------------------------
# weight application on a SwiGLU triplet (gate, up, down)
# ---------------------------------------------------------------------------

@torch.no_grad()
def permute_rows(w: torch.Tensor, perm: torch.Tensor) -> None:
    """In-place  w <- w[perm, :]  (for gate/up weights, shape [m, d])."""
    w.copy_(w[perm.to(w.device), :])


@torch.no_grad()
def permute_cols(w: torch.Tensor, perm: torch.Tensor) -> None:
    """In-place  w <- w[:, perm]  (for down weight, shape [d, m])."""
    w.copy_(w[:, perm.to(w.device)])


@torch.no_grad()
def apply_case(mlp, case: dict) -> None:
    """Apply a permutation case to an MLP module in place.

    `case` maps each of 'gate' / 'up' / 'down' to a perm tensor or None.
    gate/up perms permute rows; the down perm permutes columns.
    """
    if case.get("gate") is not None:
        permute_rows(mlp.gate_proj.weight, case["gate"])
    if case.get("up") is not None:
        permute_rows(mlp.up_proj.weight, case["up"])
    if case.get("down") is not None:
        permute_cols(mlp.down_proj.weight, case["down"])


@torch.no_grad()
def invert_case(mlp, case: dict) -> None:
    """Undo `apply_case` exactly (index-based inverse, no data copies kept)."""
    if case.get("gate") is not None:
        permute_rows(mlp.gate_proj.weight, inverse_perm(case["gate"]))
    if case.get("up") is not None:
        permute_rows(mlp.up_proj.weight, inverse_perm(case["up"]))
    if case.get("down") is not None:
        permute_cols(mlp.down_proj.weight, inverse_perm(case["down"]))


# ---------------------------------------------------------------------------
# checksums
# ---------------------------------------------------------------------------

def tensor_sha256(t: torch.Tensor) -> str:
    return hashlib.sha256(
        t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def mlp_checksums(mlp) -> dict:
    return {
        "gate": tensor_sha256(mlp.gate_proj.weight),
        "up": tensor_sha256(mlp.up_proj.weight),
        "down": tensor_sha256(mlp.down_proj.weight),
    }


# ---------------------------------------------------------------------------
# error metrics
# ---------------------------------------------------------------------------

def diff_metrics(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> dict:
    """Pre-registered numeric drift metrics between two same-shape tensors.

    Differences are computed in float32 to avoid BF16 rounding inside the
    metric itself; equality is checked on the raw tensors.
    """
    assert a.shape == b.shape, (a.shape, b.shape)
    af = a.detach().float()
    bf = b.detach().float()
    d = af - bf
    a_l2 = af.norm().item()
    a_linf = af.abs().max().item()
    cos = torch.nn.functional.cosine_similarity(
        af.flatten(), bf.flatten(), dim=0
    ).item()
    return {
        "bitwise_equal": bool(torch.equal(a, b)),
        "max_abs": d.abs().max().item(),
        "mean_abs": d.abs().mean().item(),
        "rel_l2": d.norm().item() / max(a_l2, eps),
        "rel_linf": d.abs().max().item() / max(a_linf, eps),
        "cosine": cos,
        "n_diff": int((a != b).sum().item()),
        "n_total": int(a.numel()),
    }


# ---------------------------------------------------------------------------
# atomic JSON output
# ---------------------------------------------------------------------------

def atomic_write_json(path: str, obj) -> None:
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def append_jsonl(path: str, obj) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def env_info() -> dict:
    import transformers

    return {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
        "device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        ),
        "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
