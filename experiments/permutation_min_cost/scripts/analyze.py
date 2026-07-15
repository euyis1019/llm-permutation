"""Pre-registered analyses and acceptance checks."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import linregress, spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from permutation import atomic_write_json

GEOMETRY = (
    "frac_moved", "mean_disp", "max_disp", "total_disp",
    "cross_16", "cross_64", "cross_256", "inversions",
    "frac_moved_x_mean_disp",
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def gvalue(record: dict, metric: str) -> float:
    if metric == "frac_moved_x_mean_disp":
        return record["geometry"]["frac_moved"] * record["geometry"]["mean_disp"]
    return float(record["geometry"][metric])


def regress(records: list[dict], metric: str) -> dict:
    points = [r for r in records if not r["drift"]["bitwise_equal"]
              and r["drift"]["rel_l2"] > 0 and gvalue(r, metric) > 0]
    if len(points) < 3:
        return {"n": len(points), "slope": None, "intercept": None, "r2": None, "spearman_rho": None}
    x = np.log([gvalue(r, metric) for r in points])
    y = np.log([r["drift"]["rel_l2"] for r in points])
    fit = linregress(x, y)
    rho = spearmanr(x, y).statistic
    return {
        "n": len(points),
        "slope": float(fit.slope),
        "intercept": float(fit.intercept),
        "coefficient": float(math.exp(fit.intercept)),
        "r2": float(fit.rvalue ** 2),
        "spearman_rho": float(rho),
        "law": f"drift ~= {math.exp(fit.intercept):.6g} * {metric}^{fit.slope:.6g}",
    }


def stage1(results_dir: Path) -> tuple[dict, dict]:
    records = read_jsonl(results_dir / "stage1_singlelayer.jsonl")
    groups = {"all": records}
    for li in (0, 17, 35):
        groups[f"real_L{li}"] = [r for r in records if r["input"] == "real" and r["layer"] == li]
        for inp in ("randn_s7_x1", "randn_s7_x10"):
            groups[f"{inp}_L{li}"] = [r for r in records if r["input"] == inp and r["layer"] == li]
    regression = {
        "metric_definitions": list(GEOMETRY),
        "cross_64_pre_registered_standalone": True,
        "groups": {name: {metric: regress(rows, metric) for metric in GEOMETRY}
                   for name, rows in groups.items()},
    }

    nonzero = sum(not r["drift"]["bitwise_equal"] for r in records)
    s1a_layers = {}
    for li in (0, 17, 35):
        fits = regression["groups"][f"real_L{li}"]
        qualifying = {
            k: v for k, v in fits.items()
            if v["r2"] is not None and (v["r2"] >= 0.8 or abs(v["spearman_rho"]) >= 0.9)
        }
        s1a_layers[str(li)] = {
            "pass": bool(qualifying),
            "qualifying_metrics": qualifying,
            "best_r2": max((v["r2"] for v in fits.values() if v["r2"] is not None), default=None),
            "best_abs_rho": max((abs(v["spearman_rho"]) for v in fits.values() if v["spearman_rho"] is not None), default=None),
        }
    s1a_pass = nonzero >= 200 and all(x["pass"] for x in s1a_layers.values())

    f5 = [r for r in records if r["family"] == "F5_strided_pairs" and r["input"] == "real"]
    s1b_layers = {}
    for li in (0, 17, 35):
        rows = [r for r in f5 if r["layer"] == li]
        ds = sorted({int(r["parameter"]) for r in rows})
        medians = [float(np.median([r["drift"]["rel_l2"] for r in rows if int(r["parameter"]) == d])) for d in ds]
        rho = float(spearmanr(ds, medians).statistic)
        s1b_layers[str(li)] = {"D": ds, "median_rel_l2": medians, "rho": rho, "pass": rho >= 0.9}
    s1b_pass = all(x["pass"] for x in s1b_layers.values())

    s1c_layers = {}
    for li in (0, 17, 35):
        def med(family):
            vals = [r["drift"]["rel_l2"] for r in records
                    if r["layer"] == li and r["input"] == "real" and r["family"] == family
                    and float(r["parameter"]) == 0.30]
            if len(vals) != 5:
                raise RuntimeError(f"S1c expected 5 values, got {len(vals)} for L{li} {family}")
            return float(np.median(vals))
        f4, f3 = med("F4_adjacent_pairs"), med("F3_scattered_global")
        ratio = f4 / f3 if f3 else (0.0 if f4 == 0 else math.inf)
        s1c_layers[str(li)] = {"F4_K30_median": f4, "F3_K30_median": f3, "ratio": ratio, "pass": ratio < 1/3}
    s1c_pass = all(x["pass"] for x in s1c_layers.values())
    acceptance = {
        "S1a": {"pass": s1a_pass, "threshold": "non-bitwise >=200 and each real L0/L17/L35 has R2>=0.8 or |rho|>=0.9", "non_bitwise_points": nonzero, "layers": s1a_layers},
        "S1b": {"pass": s1b_pass, "threshold": "rho(D, median drift)>=0.9 in each real layer", "layers": s1b_layers},
        "S1c": {"pass": s1c_pass, "threshold": "F4-K30 median < F3-K30 median / 3 in each real layer", "layers": s1c_layers},
    }
    acceptance["stage1_pass"] = all(acceptance[k]["pass"] for k in ("S1a", "S1b", "S1c"))
    atomic_write_json(str(results_dir / "stage1_regression.json"), regression)
    make_stage1_figures(records, regression, results_dir / "figures")
    return regression, acceptance


def make_stage1_figures(records: list[dict], regression: dict, figures: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures.mkdir(parents=True, exist_ok=True)
    real = [r for r in records if r["input"] == "real"]
    candidates = regression["groups"]["all"]
    best = max((k for k, v in candidates.items() if v["r2"] is not None), key=lambda k: candidates[k]["r2"])
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    for li, marker in ((0, "o"), (17, "s"), (35, "^")):
        rs = [r for r in real if r["layer"] == li and gvalue(r, best) > 0 and r["drift"]["rel_l2"] > 0]
        ax.scatter([gvalue(r, best) for r in rs], [r["drift"]["rel_l2"] for r in rs], s=13, alpha=.55, marker=marker, label=f"L{li}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(best); ax.set_ylabel("BF16 rel_l2"); ax.legend()
    ax.set_title("Stage 1 real-activation geometry law")
    fig.tight_layout(); fig.savefig(figures / "stage1_geometry_law.png", dpi=170); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    for li in (0, 17, 35):
        rs = [r for r in real if r["layer"] == li and r["family"] == "F5_strided_pairs"]
        ds = sorted({int(r["parameter"]) for r in rs})
        ys = [np.median([r["drift"]["rel_l2"] for r in rs if int(r["parameter"]) == d]) for d in ds]
        ax.plot(ds, ys, marker="o", label=f"L{li}")
    ax.set_xscale("log", base=2); ax.set_yscale("symlog", linthresh=1e-12)
    ax.set_xlabel("F5 displacement D"); ax.set_ylabel("5-seed median rel_l2")
    ax.legend(); ax.set_title("Fixed K=30% distance intervention")
    fig.tight_layout(); fig.savefig(figures / "stage1_f5_distance.png", dpi=170); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=ROOT / "results")
    ap.add_argument("--stage", choices=["1"], default="1")
    args = ap.parse_args()
    _, acceptance = stage1(args.results_dir)
    path = args.results_dir / "acceptance.json"
    atomic_write_json(str(path), acceptance)
    print(json.dumps(acceptance, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
