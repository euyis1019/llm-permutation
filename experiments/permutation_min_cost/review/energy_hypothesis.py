"""Reviewer post-hoc analysis (NOT pre-registered).

Hypothesis: within the saturated zone, single-layer BF16 drift magnitude is set by
the *activation-energy share* of coordinates displaced beyond the kernel reduction
threshold, not by pure geometry. Gate: displacement >= d0 (d0 scanned).

    E_share(pi, d0) = sum_{i: |pi(i)-i| >= d0} e_i / sum_i e_i
    e_i = mean_t h[i,t]^2           (variant: * ||W[:,i]||^2)

Fit log(rel_l2) ~ log(E_share) per real layer on non-bitwise points.
CPU-only; reuses the executor's frozen capture path.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "scripts"))

from perm_families import M, all_specs, make_from_spec  # noqa: E402
from stage1_singlelayer import capture_inputs_and_weights, DEFAULT_MODEL, LAYERS  # noqa: E402

GATES = (1, 9, 16, 64)


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    d = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / d) if d else float("nan")


def loglog_fit(x, y):
    lx, ly = np.log10(x), np.log10(y)
    A = np.vstack([lx, np.ones_like(lx)]).T
    coef, *_ = np.linalg.lstsq(A, ly, rcond=None)
    pred = A @ coef
    ss_res = ((ly - pred) ** 2).sum()
    ss_tot = ((ly - ly.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
    return float(r2), float(coef[0])


def main():
    print("[review] capturing real activations on CPU ...", flush=True)
    real, weights, meta = capture_inputs_and_weights(DEFAULT_MODEL)
    energy = {}
    for li in LAYERS:
        h = real[li].to(torch.float32)          # [M, T]
        w = weights[li].to(torch.float32)       # [hidden, M]
        e_h = (h * h).mean(dim=1)               # [M]
        e_hw = e_h * (w * w).sum(dim=0)
        energy[li] = {"h": e_h.numpy(), "hw": e_hw.numpy()}

    recs = [json.loads(l) for l in (ROOT / "results/stage1_singlelayer.jsonl").open()]
    drift = {(r["perm_key"], r["layer"]): r["drift"]["rel_l2"]
             for r in recs if r["input"] == "real"}

    print("[review] rebuilding 147 permutations, computing gated energy shares ...", flush=True)
    rows = []
    for spec in all_specs():
        p = make_from_spec(spec, M).numpy()
        disp = np.abs(p - np.arange(M))
        for li in LAYERS:
            row = {"perm_key": spec.key, "family": spec.family,
                   "parameter": spec.parameter, "seed": spec.seed, "layer": li,
                   "rel_l2": drift[(spec.key, li)]}
            for wkind in ("h", "hw"):
                e = energy[li][wkind]
                tot = e.sum()
                for d0 in GATES:
                    row[f"E_{wkind}_ge{d0}"] = float(e[disp >= d0].sum() / tot)
            rows.append(row)

    out = HERE / "energy_hypothesis_points.jsonl"
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    summary = {"capture_tokens": meta["rendered_n_tokens"], "fits": {}}
    for li in LAYERS:
        pts = [r for r in rows if r["layer"] == li and r["rel_l2"] > 0]
        y = np.array([r["rel_l2"] for r in pts])
        layer_fit = {"n_nonzero": len(pts)}
        for wkind in ("h", "hw"):
            for d0 in GATES:
                key = f"E_{wkind}_ge{d0}"
                x = np.array([r[key] for r in pts])
                keep = x > 0
                if keep.sum() < 10:
                    continue
                r2, slope = loglog_fit(x[keep], y[keep])
                layer_fit[key] = {
                    "n": int(keep.sum()),
                    "n_dropped_zero_metric": int((~keep).sum()),
                    "r2": round(r2, 4), "slope": round(slope, 3),
                    "spearman": round(spearman(x[keep], y[keep]), 4),
                }
        summary["fits"][f"real_L{li}"] = layer_fit

    (HERE / "energy_hypothesis_summary.json").write_text(json.dumps(summary, indent=1))
    for lk, lf in summary["fits"].items():
        print(f"\n{lk} (n_nonzero={lf['n_nonzero']})")
        for k, v in lf.items():
            if isinstance(v, dict):
                print(f"  {k:12s} R2={v['r2']:.4f} rho={v['spearman']:.4f} "
                      f"slope={v['slope']:+.3f} n={v['n']} (dropped {v['n_dropped_zero_metric']})")


if __name__ == "__main__":
    main()
