"""Reviewer analysis: supplementary behavior-facing metrics from saved fp32 logits.

For every unit (variant x seed) with saved last-token logits, compare against its
model-family baseline and compute per prompt:
  - top1_flip: greedy argmax changed
  - base_margin / case_margin: top1 minus top2 logit (fp32 values, fp64 math)
  - kl_nats: KL(softmax(base) || softmax(case)) over the full 151936 vocab, fp64
  - top10_overlap: |top10(base) & top10(case)| / 10
  - rank_of_base_top1: rank (1-indexed) of the baseline argmax token in the case ranking
  - rel_l2_check: recomputed rel_l2, cross-checked against the value stored during the run

CPU-only, reads results/units, results/supp_units, results/part0_run_a.
Writes reviewer_analysis/metrics_supplement.json.
"""
from __future__ import annotations
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path("/nvme0/if/permutation/experiments/noise_floor")
UNITS = ROOT / "results" / "units"
SUPP_UNITS = ROOT / "results" / "supp_units"
BASE_BASELINE = ROOT / "results" / "part0_run_a" / "logits"
INSTRUCT_BASELINE = UNITS / "part2_identity" / "logits"
OUT = ROOT / "reviewer_analysis" / "metrics_supplement.json"

N_PROMPTS = 32
SIGMA_GRID = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]
S1_GRID = [1e-8, 1e-7, 3e-7]
S2_GRID = [1e-6, 1e-5, 1e-4]


def load_logits(d: Path, pid: int) -> np.ndarray:
    return np.load(d / f"prompt_{pid:02d}.float32.npy").astype(np.float64)


def log_softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    return z - np.log(np.exp(z).sum())


def compare_prompt(zb: np.ndarray, zc: np.ndarray) -> dict:
    ob, oc = np.argsort(zb)[::-1], np.argsort(zc)[::-1]
    b1, b2 = int(ob[0]), int(ob[1])
    c1, c2 = int(oc[0]), int(oc[1])
    lb, lc = log_softmax(zb), log_softmax(zc)
    kl = float(np.sum(np.exp(lb) * (lb - lc)))
    denom = np.linalg.norm(zb)
    rel_l2 = float(np.linalg.norm(zc - zb) / denom) if denom > 0 else 0.0
    return {
        "top1_flip": b1 != c1,
        "base_top1": b1,
        "case_top1": c1,
        "base_margin": float(zb[b1] - zb[b2]),
        "case_margin": float(zc[c1] - zc[c2]),
        "kl_nats": kl,
        "top10_overlap": len(set(ob[:10].tolist()) & set(oc[:10].tolist())) / 10.0,
        "rank_of_base_top1": int(np.where(oc == b1)[0][0]) + 1,
        "rel_l2_check": rel_l2,
    }


def stored_rel_l2(unit_dir: Path) -> dict[int, float]:
    """Per-prompt rel_l2 recorded during the run, for cross-checking."""
    summ = unit_dir / "summary.json"
    if not summ.exists():
        return {}
    d = json.load(open(summ))
    rows = d.get("per_prompt") or d.get("prompts") or []
    return {int(r["prompt_id"]): float(r["rel_l2"]) for r in rows if "rel_l2" in r}


def analyze_unit(unit_dir: Path, baseline_dir: Path) -> dict:
    stored = stored_rel_l2(unit_dir)
    per_prompt, mismatches = [], 0
    for pid in range(N_PROMPTS):
        zb = load_logits(baseline_dir, pid)
        zc = load_logits(unit_dir / "logits", pid)
        r = compare_prompt(zb, zc)
        r["prompt_id"] = pid
        if pid in stored and not np.isclose(r["rel_l2_check"], stored[pid], rtol=1e-3, atol=1e-9):
            mismatches += 1
        per_prompt.append(r)
    kls = np.array([r["kl_nats"] for r in per_prompt])
    flips = [r for r in per_prompt if r["top1_flip"]]
    return {
        "n_prompts": N_PROMPTS,
        "top1_flips": len(flips),
        "flipped_prompts": [
            {"prompt_id": r["prompt_id"], "base_margin": r["base_margin"],
             "rank_of_base_top1": r["rank_of_base_top1"]} for r in flips
        ],
        "median_kl_nats": float(np.median(kls)),
        "max_kl_nats": float(kls.max()),
        "median_top10_overlap": float(np.median([r["top10_overlap"] for r in per_prompt])),
        "min_top10_overlap": float(min(r["top10_overlap"] for r in per_prompt)),
        "median_rel_l2_check": float(np.median([r["rel_l2_check"] for r in per_prompt])),
        "rel_l2_crosscheck_mismatches": mismatches,
        "per_prompt": per_prompt,
    }


def pool(unit_results: list[dict]) -> dict:
    pp = [r for u in unit_results for r in u["per_prompt"]]
    n = len(pp)
    flips = [r for r in pp if r["top1_flip"]]
    return {
        "n_measurements": n,
        "top1_flips": len(flips),
        "flip_rate": len(flips) / n if n else None,
        "median_kl_nats": float(np.median([r["kl_nats"] for r in pp])),
        "max_kl_nats": float(max(r["kl_nats"] for r in pp)),
        "median_top10_overlap": float(np.median([r["top10_overlap"] for r in pp])),
        "median_rel_l2": float(np.median([r["rel_l2_check"] for r in pp])),
        "flipped_base_margins": sorted(round(r["base_margin"], 4) for r in flips),
        "worst_rank_of_base_top1": max((r["rank_of_base_top1"] for r in pp), default=1),
    }


def main() -> None:
    # Baseline margin landscape (Base model): the margins that any flip has to beat.
    base_margins = []
    for pid in range(N_PROMPTS):
        zb = load_logits(BASE_BASELINE, pid)
        ob = np.argsort(zb)[::-1]
        base_margins.append(float(zb[ob[0]] - zb[ob[1]]))
    instr_margins = []
    for pid in range(N_PROMPTS):
        zb = load_logits(INSTRUCT_BASELINE, pid)
        ob = np.argsort(zb)[::-1]
        instr_margins.append(float(zb[ob[0]] - zb[ob[1]]))

    units: dict[str, dict] = {}

    def run(name: str, unit_dir: Path, baseline: Path) -> dict:
        res = analyze_unit(unit_dir, baseline)
        units[name] = res
        print(f"{name:32s} flips {res['top1_flips']:2d}/32  medKL {res['median_kl_nats']:.3e}  "
              f"medOv10 {res['median_top10_overlap']:.2f}  xchk_miss {res['rel_l2_crosscheck_mismatches']}")
        return res

    for tag in ["part1a_f9_k100", "part1a_f10_k100", "part1a_f7"]:
        run(tag, UNITS / tag, BASE_BASELINE)
    for si in range(10):
        for rep in range(3):
            tag = f"sigma_{si:02d}_rep{rep}"
            run(tag, UNITS / tag, BASE_BASELINE)
    for si in range(3):
        for rep in range(3):
            for arm, prefix in [("s1", "supp_s1_all"), ("s2", "supp_s2_ffn")]:
                tag = f"{prefix}_sigma{si}_rep{rep}"
                run(tag, SUPP_UNITS / tag, BASE_BASELINE)
    part2_tags = [d.name for d in sorted(UNITS.glob("part2_*")) if d.name != "part2_identity"]
    for tag in part2_tags:
        run(tag, UNITS / tag, INSTRUCT_BASELINE)

    groups = {
        "perm_inblock_f9": pool([units["part1a_f9_k100"]]),
        "perm_adjswap_f10": pool([units["part1a_f10_k100"]]),
        "perm_random_f7": pool([units["part1a_f7"]]),
    }
    for si, sigma in enumerate(SIGMA_GRID):
        groups[f"gauss_sigma_{sigma:g}"] = pool([units[f"sigma_{si:02d}_rep{r}"] for r in range(3)])
    for si, sigma in enumerate(S1_GRID):
        groups[f"gauss_s1_sigma_{sigma:g}"] = pool([units[f"supp_s1_all_sigma{si}_rep{r}"] for r in range(3)])
    for si, sigma in enumerate(S2_GRID):
        groups[f"gauss_s2ffn_sigma_{sigma:g}"] = pool([units[f"supp_s2_ffn_sigma{si}_rep{r}"] for r in range(3)])
    for tag in part2_tags:
        groups[f"instruct_{tag}"] = pool([units[tag]])

    # Margin-vs-flip check across all Base-model nonzero-drift units.
    base_pp = [r for name, u in units.items() if name not in part2_tags
               for r in u["per_prompt"] if r["rel_l2_check"] > 0]
    flipped = [r["base_margin"] for r in base_pp if r["top1_flip"]]
    kept = [r["base_margin"] for r in base_pp if not r["top1_flip"]]
    margin_split = {
        "n_flipped": len(flipped),
        "n_kept": len(kept),
        "median_base_margin_flipped": float(np.median(flipped)) if flipped else None,
        "median_base_margin_kept": float(np.median(kept)) if kept else None,
        "max_base_margin_flipped": float(max(flipped)) if flipped else None,
    }

    out = {
        "baseline_margins": {
            "base_per_prompt": [round(m, 4) for m in base_margins],
            "base_median": float(np.median(base_margins)),
            "base_min": float(min(base_margins)),
            "instruct_per_prompt": [round(m, 4) for m in instr_margins],
            "instruct_median": float(np.median(instr_margins)),
            "instruct_min": float(min(instr_margins)),
        },
        "margin_vs_flip": margin_split,
        "groups": groups,
        "units": units,
        "notes": [
            "kl_nats = KL(softmax(baseline_logits) || softmax(case_logits)), fp64, full 151936 vocab, last token of each of the 32 frozen prompts",
            "margins are raw logit gaps (top1 - top2) of fp32 saved logits",
            "baselines: part0_run_a (Base family incl. sigma and supp arms), part2_identity (Instruct anchors)",
            "rel_l2_crosscheck_mismatches counts prompts where recomputed rel_l2 differs from the value stored during the GPU run (rtol 1e-3)",
        ],
    }
    OUT.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {OUT}")
    print(json.dumps({"margin_vs_flip": margin_split, "base_median_margin": out['baseline_margins']['base_median'],
                      "instruct_median_margin": out['baseline_margins']['instruct_median']}, indent=1))


if __name__ == "__main__":
    main()
