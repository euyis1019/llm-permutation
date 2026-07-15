"""Figures for the FFN benchmark equivalence experiment.

- stage2_seed_delta_{family}.png : per-benchmark box/scatter of the 20-seed
  accuracy deltas (pp), with the ±1pp equivalence band and the baseline-rerun
  null range overlaid.
- stage2_macro_hist_{family}.png : histogram of suite-macro deltas vs ±0.5pp band.
- ablation_scope_magnitude.png : suite-macro delta + mean disagreement across the
  permutation-scope and magnitude arms.
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common

RES = common.EXP_ROOT / "results"
FIG = RES / "figures"
FIG.mkdir(parents=True, exist_ok=True)
BENCHES = ["mmlu", "gsm8k", "ceval", "cmmlu", "humaneval_plus", "mbpp_plus"]
FAMILIES = ["qwen3_4b", "qwen3_4b_base"]


def load(name):
    p = RES / name
    return json.loads(p.read_text()) if p.is_file() else None


def stage2_figs():
    import analyze
    cfg = common.load_config()
    null = load("null_distribution.json") or {}
    for fam in FAMILIES:
        base_tag = f"{fam}__baseline_original_run1"
        seeds = cfg["stage2_seeds"]
        per_bench_deltas = {b: [] for b in BENCHES}
        macro = []
        for s in seeds:
            res = analyze.analyze_pair(fam, f"{fam}__perm_all36_s{s}", base_tag)
            if res is None:
                continue
            for b in BENCHES:
                per_bench_deltas[b].append(res["per_bench"][b]["accuracy_delta_pp"])
            macro.append(res["suite_macro"]["macro_delta_pp"])
        if not macro:
            continue
        # box plot per benchmark
        fig, ax = plt.subplots(figsize=(9, 5))
        data = [per_bench_deltas[b] for b in BENCHES]
        ax.axhspan(-1, 1, color="green", alpha=0.08, label="±1pp equivalence band")
        bp = ax.boxplot(data, tick_labels=BENCHES, showmeans=True)
        # overlay null range
        nb = (null.get(fam) or {}).get("per_bench", {})
        for i, b in enumerate(BENCHES, 1):
            if b in nb:
                lo = nb[b]["delta_pp"]["min"]; hi = nb[b]["delta_pp"]["max"]
                ax.plot([i, i], [lo, hi], color="red", lw=6, alpha=0.25,
                        solid_capstyle="butt", label="baseline-rerun null range" if i == 1 else None)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_ylabel("accuracy delta vs baseline (pp)")
        ax.set_title(f"{fam}: 20-seed permutation accuracy deltas per benchmark")
        ax.legend(fontsize=8)
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.savefig(FIG / f"stage2_seed_delta_{fam}.png", dpi=120)
        plt.close()

        # macro histogram
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.axvspan(-0.5, 0.5, color="green", alpha=0.1, label="±0.5pp macro band")
        ax.hist(macro, bins=12, color="steelblue", edgecolor="k")
        ax.axvline(0, color="k", lw=0.6)
        ax.set_xlabel("suite-macro accuracy delta (pp)")
        ax.set_ylabel("# seeds")
        ax.set_title(f"{fam}: suite-macro delta over 20 permutation seeds")
        ax.legend()
        plt.tight_layout()
        plt.savefig(FIG / f"stage2_macro_hist_{fam}.png", dpi=120)
        plt.close()
    print(f"stage2 figures -> {FIG}")


def ablation_fig():
    abl = load("ablation_summary.json")
    if not abl:
        return
    order = ["scope_single_L0", "scope_single_L17", "scope_single_L35",
             "scope_prefix6", "scope_prefix18", "scope_all36_random",
             "mag_adjacent_swap_all36", "mag_reverse_all36"]
    order = [a for a in order if a in abl]
    macro = [abl[a]["suite_macro_delta_pp"] for a in order]
    disag = [abl[a]["mean_behavior_disagreement"] * 100 for a in order]
    fig, ax1 = plt.subplots(figsize=(10, 5))
    x = range(len(order))
    ax1.bar([i - 0.2 for i in x], macro, width=0.4, color="steelblue", label="suite-macro Δ (pp)")
    ax1.set_ylabel("suite-macro delta (pp)", color="steelblue")
    ax1.axhline(0, color="k", lw=0.6)
    ax2 = ax1.twinx()
    ax2.bar([i + 0.2 for i in x], disag, width=0.4, color="darkorange", label="mean disagreement (%)")
    ax2.set_ylabel("mean behavior disagreement (%)", color="darkorange")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(order, rotation=30, ha="right", fontsize=8)
    ax1.set_title("qwen3_4b: permutation scope & magnitude ablation")
    plt.tight_layout()
    plt.savefig(FIG / "ablation_scope_magnitude.png", dpi=120)
    plt.close()
    print(f"ablation figure -> {FIG}")


if __name__ == "__main__":
    stage2_figs()
    ablation_fig()
