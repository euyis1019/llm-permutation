"""Reviewer probe (post-hoc): alignment vs displacement as the free-class criterion.

Findings (2026-07-12, RTX 4090, torch bf16 GEMM + vLLM 0.24 batch-invariant):
- even-aligned pair swaps (2k,2k+1) and aligned 8-window shuffles: bitwise-free
  under BOTH backends, L0/L17/L35, T in {1,124,512,2048,8192}.
- odd-aligned pair swaps (2k+1,2k+2), displacement 1: torch drift saturates at
  the BF16 output-ulp ceiling (~2.7e-3). Displacement is NOT the criterion.
- aligned 16/32-window shuffles: sub-ulp tier (~3e-5..1e-4).
Formalized confirmation on fresh seeds: AMENDMENT_v1.1.md Stage 1b.
"""
import sys
import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "scripts"))
from stage1_singlelayer import capture_inputs_and_weights, DEFAULT_MODEL  # noqa: E402
from permutation import diff_metrics  # noqa: E402
from vllm.model_executor.layers.batch_invariant import linear_batch_invariant  # noqa: E402

M = 9728


def perm_windows(width, offset, g):
    p = torch.arange(M)
    for s in range(offset, M - width + 1, width):
        p[s:s + width] = s + torch.randperm(width, generator=g)
    return p


def perm_pairs(start):
    p = torch.arange(M)
    for i in range(start, M - 1, 2):
        p[i], p[i + 1] = p[i + 1].clone(), p[i].clone()
    return p


def main():
    real, weights, _ = capture_inputs_and_weights(DEFAULT_MODEL)
    dev = torch.device("cuda:0")
    g = torch.Generator().manual_seed(301)
    probes = {
        "pairs_even": perm_pairs(0),
        "pairs_odd": perm_pairs(1),
        "win8_aligned": perm_windows(8, 0, g),
        "win8_offset4": perm_windows(8, 4, g),
        "win16_aligned": perm_windows(16, 0, g),
        "win32_aligned": perm_windows(32, 0, g),
    }
    for li in (0, 17, 35):
        W = weights[li].to(dev)
        x = real[li].to(dev).T.contiguous()
        for name, fn in (("torch", torch.nn.functional.linear), ("vllm_bi", linear_batch_invariant)):
            base = fn(x, W)
            out = []
            for pname, perm in probes.items():
                pc = perm.to(dev)
                m = diff_metrics(fn(x[:, pc].contiguous(), W[:, pc].contiguous()), base)
                out.append(f'{pname}={m["rel_l2"]:.1e}{"*" if m["bitwise_equal"] else ""}')
            print(f"L{li:<3}{name:8s} " + "  ".join(out))
    print("(* = bitwise equal)")


if __name__ == "__main__":
    main()
