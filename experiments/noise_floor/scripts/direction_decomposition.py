"""Reviewer analysis: decompose the Base(+)/Instruct(-) direction by benchmark.

For each model family and each of the 6 benchmarks:
  - baseline accuracy (original layout, run1) and the same-weight rerun range
  - mean/std/range of the 20 permutation seeds (s1000..s1019)
  - offset = seed_mean - baseline, its contribution to the suite-macro offset,
    per-benchmark sign consistency and z-score of the baseline inside the seed cloud
  - per-question systematics: questions the baseline got right but most seeds lose
    (and the reverse), to separate question-specific luck from diffuse noise.

CPU-only, reads ffn_benchmark_eval/results/raw.
Writes noise_floor/reviewer_analysis/direction_decomposition.json.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

RAW = Path("/nvme0/if/permutation/experiments/ffn_benchmark_eval/results/raw")
OUT = Path("/nvme0/if/permutation/experiments/noise_floor/reviewer_analysis/direction_decomposition.json")

BENCHMARKS = ["mmlu", "gsm8k", "ceval", "cmmlu", "humaneval_plus", "mbpp_plus"]
SEEDS = list(range(1000, 1020))
FAMILIES = {
    "base": {"prefix": "qwen3_4b_base", "baseline": "qwen3_4b_base__baseline_original_run1"},
    "instruct": {"prefix": "qwen3_4b", "baseline": "qwen3_4b__baseline_original_run1"},
}
RERUN_TAGS = ["baseline_original_run2", "baseline_copy"] + [f"baseline_rep{r:02d}" for r in range(2, 10)]


def load(tag: str, bench: str) -> dict:
    d = json.load(open(RAW / tag / f"{bench}.raw.json"))
    recs = {s["sample_id"]: bool(s["correct"]) for s in d["samples"]}
    return {"accuracy": float(d["accuracy"]), "recs": recs}


def main() -> None:
    out = {}
    for fam, cfg in FAMILIES.items():
        fam_out = {"benchmarks": {}, "suite_macro": {}}
        macro_base, macro_seeds = [], np.zeros(len(SEEDS))
        for bench in BENCHMARKS:
            base = load(cfg["baseline"], bench)
            reruns = []
            for rt in RERUN_TAGS:
                p = RAW / f"{cfg['prefix']}__{rt}" / f"{bench}.raw.json"
                if p.exists():
                    reruns.append(float(json.load(open(p))["accuracy"]))
            seeds = [load(f"{cfg['prefix']}__perm_all36_s{s}", bench) for s in SEEDS]
            accs = np.array([s["accuracy"] for s in seeds])
            deltas = (accs - base["accuracy"]) * 100.0

            ids = sorted(base["recs"])
            base_vec = np.array([base["recs"][i] for i in ids])
            seed_mat = np.array([[s["recs"][i] for i in ids] for s in seeds])
            n_seeds_correct = seed_mat.sum(axis=0)
            sys_lost = [ids[j] for j in range(len(ids)) if base_vec[j] and n_seeds_correct[j] <= 5]
            sys_gained = [ids[j] for j in range(len(ids)) if not base_vec[j] and n_seeds_correct[j] >= 15]
            flip_frac = float(np.mean(seed_mat != base_vec[None, :]))

            std = float(accs.std(ddof=0))
            fam_out["benchmarks"][bench] = {
                "baseline_acc": base["accuracy"] * 100.0,
                "rerun_accs": [a * 100.0 for a in reruns],
                "rerun_range_pp": (max(reruns) - min(reruns)) * 100.0 if reruns else None,
                "seed_mean": float(accs.mean()) * 100.0,
                "seed_std_pp": std * 100.0,
                "seed_min": float(accs.min()) * 100.0,
                "seed_max": float(accs.max()) * 100.0,
                "offset_pp": float(deltas.mean()),
                "offset_contrib_to_macro_pp": float(deltas.mean()) / len(BENCHMARKS),
                "n_seeds_above_baseline": int((deltas > 0).sum()),
                "n_seeds_below_baseline": int((deltas < 0).sum()),
                "baseline_z_in_seed_cloud": float((base["accuracy"] - accs.mean()) / std) if std > 0 else None,
                "n_questions": len(ids),
                "mean_per_question_flip_rate": flip_frac,
                "systematically_lost_questions": {"n": len(sys_lost), "ids": sys_lost[:20]},
                "systematically_gained_questions": {"n": len(sys_gained), "ids": sys_gained[:20]},
            }
            macro_base.append(base["accuracy"])
            macro_seeds += accs / len(BENCHMARKS)

        macro_baseline = float(np.mean(macro_base)) * 100.0
        macro_seeds_pp = macro_seeds * 100.0
        fam_out["suite_macro"] = {
            "baseline": macro_baseline,
            "seed_mean": float(macro_seeds_pp.mean()),
            "seed_std_pp": float(macro_seeds_pp.std(ddof=0)),
            "offset_pp": float(macro_seeds_pp.mean() - macro_baseline),
        }
        out[fam] = fam_out

    OUT.write_text(json.dumps(out, indent=1))
    for fam in out:
        print(f"\n== {fam} ==  suite-macro offset {out[fam]['suite_macro']['offset_pp']:+.3f} pp")
        print(f"{'bench':16s} {'base':>7s} {'rerunRg':>8s} {'seedMean':>9s} {'offset':>8s} {'contrib':>8s} {'sign':>7s} {'z':>6s} {'sysL':>5s} {'sysG':>5s}")
        for b in BENCHMARKS:
            r = out[fam]["benchmarks"][b]
            rr = f"{r['rerun_range_pp']:.2f}" if r["rerun_range_pp"] is not None else "n/a"
            print(f"{b:16s} {r['baseline_acc']:7.2f} {rr:>8s} {r['seed_mean']:9.2f} {r['offset_pp']:+8.2f} "
                  f"{r['offset_contrib_to_macro_pp']:+8.3f} {r['n_seeds_above_baseline']:>2d}+/{r['n_seeds_below_baseline']:<2d}- "
                  f"{r['baseline_z_in_seed_cloud']:6.2f} {r['systematically_lost_questions']['n']:5d} {r['systematically_gained_questions']['n']:5d}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
