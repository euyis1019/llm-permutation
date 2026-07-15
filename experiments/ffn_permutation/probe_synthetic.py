"""Stage A — small-matrix algebra + directionality unit tests (no model).

Validates, before touching Qwen3-4B:
  1. perm construction is a bijection; inv_perm = argsort(perm) restores weights
     bitwise (torch.equal + sha256).
  2. coordinate alignment: g' == g[..., perm], u' == u[..., perm],
     h' == h[..., perm]  (bitwise, BF16).
  3. down path split: native  y' = W_d[:, perm] @ h'  vs
     canonical  y_c = W_d @ (h'[..., inv_perm]).  canonical must be bitwise
     equal to baseline; any native drift is then attributable to GEMM
     reduction order only.
  4. directionality: with a non-involutive perm, indexing down with inv_perm
     instead of perm must break equivalence (catches perm/argsort confusion).
  5. the full 10-case control table evaluated in float64, where valid-triplet
     must match baseline to ~1e-12 and every negative control must not.

Runs on CPU and CUDA. All main checks in BF16 per plan; float64 is a
supplementary exact-math check of the criteria themselves.
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from permutation import (
    apply_case,
    atomic_write_json,
    check_bijection,
    diff_metrics,
    env_info,
    inverse_perm,
    make_perm,
)

D, M = 7, 11
WEIGHT_SEEDS = [1000, 1001, 1002, 1003, 1004]
PERM_SPECS = (
    [("identity", None), ("adjacent_swap", None), ("reverse", None)]
    + [("random", s) for s in [42, 43, 44, 45, 46]]
)
DEFAULT_RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "results"
)


def swiglu_parts(wg, wu, wd, x):
    g = torch.nn.functional.silu(x @ wg.T)
    u = x @ wu.T
    h = g * u
    y = h @ wd.T
    return g, u, h, y


def control_table(perm, perm_g, perm_u, perm_d, m):
    """The 10 pre-registered cases as (name, gate_perm, up_perm, down_perm)."""
    inv = inverse_perm(perm)
    return [
        ("baseline-repeat", None, None, None),
        ("valid-triplet", perm, perm, perm),
        ("gate-only", perm, None, None),
        ("up-only", None, perm, None),
        ("down-only", None, None, perm),
        ("gate+up", perm, perm, None),
        ("gate+down", perm, None, perm),
        ("up+down", None, perm, perm),
        ("independent-triplet", perm_g, perm_u, perm_d),
        ("wrong-direction", perm, perm, inv),
    ]


def run_device(device: str) -> dict:
    out = {"device": device, "perm_checks": [], "float64_controls": []}

    for kind, seed in PERM_SPECS:
        perm = make_perm(kind, M, seed)
        entry = {
            "perm": f"{kind}" + (f"-s{seed}" if seed is not None else ""),
            "bijection": check_bijection(perm, M),
            "involution": bool(torch.equal(inverse_perm(perm), perm)),
            "seeds": [],
        }
        inv = inverse_perm(perm)
        # sanity of inverse definition itself
        entry["inv_roundtrip"] = bool(
            torch.equal(perm[inv], torch.arange(M))
            and torch.equal(inv[perm], torch.arange(M))
        )

        for ws in WEIGHT_SEEDS:
            g = torch.Generator().manual_seed(ws)
            wg = torch.randn(M, D, generator=g).bfloat16().to(device)
            wu = torch.randn(M, D, generator=g).bfloat16().to(device)
            wd = torch.randn(D, M, generator=g).bfloat16().to(device)
            x = torch.randn(4, D, generator=g).bfloat16().to(device)

            g0, u0, h0, y0 = swiglu_parts(wg, wu, wd, x)

            wg_p, wu_p = wg[perm, :], wu[perm, :]
            wd_p = wd[:, perm]
            g1, u1, h1, _ = swiglu_parts(wg_p, wu_p, wd_p, x)

            # (2) coordinate alignment, bitwise in BF16
            coord = {
                "gate_coordinate_equal": bool(torch.equal(g1, g0[..., perm.to(device)])),
                "up_coordinate_equal": bool(torch.equal(u1, u0[..., perm.to(device)])),
                "product_coordinate_equal": bool(torch.equal(h1, h0[..., perm.to(device)])),
            }

            # (3) native vs canonical down
            y_native = h1 @ wd_p.T
            y_canonical = h1[..., inv.to(device)] @ wd.T
            down = {
                "canonical_bitwise_equal_baseline": bool(torch.equal(y_canonical, y0)),
                "native_vs_baseline": diff_metrics(y_native, y0),
            }

            # (1) restore via in-place apply/invert on real nn.Linear modules
            mlp = torch.nn.Module()
            mlp.gate_proj = torch.nn.Linear(D, M, bias=False)
            mlp.up_proj = torch.nn.Linear(D, M, bias=False)
            mlp.down_proj = torch.nn.Linear(M, D, bias=False)
            with torch.no_grad():
                mlp.gate_proj.weight.copy_(wg.float())
                mlp.up_proj.weight.copy_(wu.float())
                mlp.down_proj.weight.copy_(wd.float())
            mlp = mlp.to(device).bfloat16()
            before = {
                "gate": mlp.gate_proj.weight.clone(),
                "up": mlp.up_proj.weight.clone(),
                "down": mlp.down_proj.weight.clone(),
            }
            case = {"gate": perm, "up": perm, "down": perm}
            apply_case(mlp, case)
            changed = {
                k: not torch.equal(getattr(mlp, f"{k}_proj").weight, before[k])
                for k in before
            }
            from permutation import invert_case

            invert_case(mlp, case)
            restored = {
                k: bool(torch.equal(getattr(mlp, f"{k}_proj").weight, before[k]))
                for k in before
            }

            # (4) directionality trap (only meaningful for non-involutions)
            wrong_dir_differs = None
            if not entry["involution"]:
                y_wrong = h1 @ wd[:, inv].T
                wrong_dir_differs = not torch.allclose(
                    y_wrong.float(), y0.float(), atol=1e-2, rtol=1e-2
                )

            entry["seeds"].append(
                {
                    "weight_seed": ws,
                    **coord,
                    **down,
                    "weights_changed_by_apply": changed
                    if kind != "identity"
                    else "identity-noop-expected",
                    "restore_equal": restored,
                    "wrong_direction_differs": wrong_dir_differs,
                }
            )
        out["perm_checks"].append(entry)

    # (5) full control table in float64 — exact-math criterion check
    for seed in [42, 43, 44, 45, 46]:
        perm = make_perm("random", M, seed)
        perm_g = make_perm("random", M, seed + 100)
        perm_u = make_perm("random", M, seed + 200)
        perm_d = make_perm("random", M, seed + 300)
        g = torch.Generator().manual_seed(seed)
        wg = torch.randn(M, D, generator=g, dtype=torch.float64).to(device)
        wu = torch.randn(M, D, generator=g, dtype=torch.float64).to(device)
        wd = torch.randn(D, M, generator=g, dtype=torch.float64).to(device)
        x = torch.randn(8, D, generator=g, dtype=torch.float64).to(device)
        _, _, _, y0 = swiglu_parts(wg, wu, wd, x)

        rows = {}
        for name, pg, pu, pd in control_table(perm, perm_g, perm_u, perm_d, M):
            wg2 = wg[pg, :] if pg is not None else wg
            wu2 = wu[pu, :] if pu is not None else wu
            wd2 = wd[:, pd] if pd is not None else wd
            _, _, _, y2 = swiglu_parts(wg2, wu2, wd2, x)
            rows[name] = {
                "max_abs": (y2 - y0).abs().max().item(),
                "rel_l2": ((y2 - y0).norm() / y0.norm()).item(),
            }
        out["float64_controls"].append({"perm_seed": seed, "cases": rows})

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    args = ap.parse_args()
    results_dir = os.path.abspath(args.results_dir)
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, "synthetic.json")

    torch.manual_seed(0)
    report = {"env": env_info(), "d": D, "m": M, "runs": []}
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for dev in devices:
        report["runs"].append(run_device(dev))

    # ---- verdicts -------------------------------------------------------
    problems = []
    for run in report["runs"]:
        for pc in run["perm_checks"]:
            if not (pc["bijection"] and pc["inv_roundtrip"]):
                problems.append(f"{run['device']}:{pc['perm']}: bijection/inverse failed")
            for s in pc["seeds"]:
                for k in (
                    "gate_coordinate_equal",
                    "up_coordinate_equal",
                    "product_coordinate_equal",
                    "canonical_bitwise_equal_baseline",
                ):
                    if not s[k]:
                        problems.append(
                            f"{run['device']}:{pc['perm']}:ws{s['weight_seed']}: {k}=False"
                        )
                if not all(s["restore_equal"].values()):
                    problems.append(
                        f"{run['device']}:{pc['perm']}:ws{s['weight_seed']}: restore failed"
                    )
                if s["wrong_direction_differs"] is False:
                    problems.append(
                        f"{run['device']}:{pc['perm']}:ws{s['weight_seed']}: wrong-direction NOT caught"
                    )
        for fc in run["float64_controls"]:
            c = fc["cases"]
            if c["valid-triplet"]["max_abs"] > 1e-9:
                problems.append(
                    f"{run['device']}:fp64 seed{fc['perm_seed']}: valid-triplet max_abs="
                    f"{c['valid-triplet']['max_abs']:.3e} (expected ~0)"
                )
            if c["baseline-repeat"]["max_abs"] != 0.0:
                problems.append(f"{run['device']}:fp64 seed{fc['perm_seed']}: baseline not deterministic")
            for name, v in c.items():
                if name in ("baseline-repeat", "valid-triplet"):
                    continue
                if v["rel_l2"] < 1e-3:
                    problems.append(
                        f"{run['device']}:fp64 seed{fc['perm_seed']}: negative control "
                        f"{name} unexpectedly close (rel_l2={v['rel_l2']:.3e})"
                    )

    report["problems"] = problems
    report["passed"] = len(problems) == 0
    atomic_write_json(results_path, report)
    print(f"Stage A passed={report['passed']}  problems={len(problems)}")
    for p in problems:
        print("  PROBLEM:", p)
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
