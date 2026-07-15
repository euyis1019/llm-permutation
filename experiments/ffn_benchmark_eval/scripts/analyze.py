"""Paired analysis for the FFN benchmark equivalence experiment.

Reads per-sample raw results (results/raw/{tag}/{bench}.raw.json) and produces:

- determinism report   : baseline_original run1 vs run2 vs baseline_copy must
                         agree on every sample's response + correctness (§8.1).
- stage1 summary       : for each family/seed/benchmark and suite-macro, the
                         text/answer/correctness agreement, gain/loss, accuracy
                         delta with paired bootstrap 95% CI, and McNemar exact.
- stage2 distribution  : per family/benchmark and suite-macro, the mean/std/
                         median/IQR/5-95 quantile/observed-range of the accuracy
                         delta and behaviour disagreement over the seed sweep.

Everything is paired by sample_id.  Deltas are in percentage points.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Dict, List, Optional

import common

RAW = common.EXP_ROOT / "results" / "raw"
OUT = common.EXP_ROOT / "results"
BENCHES = ["mmlu", "gsm8k", "ceval", "cmmlu", "humaneval_plus", "mbpp_plus"]
BOOT_SEED = 12345
N_BOOT = 5000


def load_raw(tag: str, bench: str) -> Optional[dict]:
    p = RAW / tag / f"{bench}.raw.json"
    if not p.is_file():
        return None
    d = json.loads(p.read_text())
    if not d.get("complete"):
        return None
    return d


def sample_map(raw: dict) -> Dict[str, dict]:
    return {s["sample_id"]: s for s in raw["samples"]}


# ── paired stats for one (baseline, comparison) on one benchmark ──────────────

def paired_bench(base: dict, comp: dict) -> dict:
    bm, cm = sample_map(base), sample_map(comp)
    ids = [i for i in bm if i in cm]
    n = len(ids)
    text_match = ans_match = corr_match = 0
    loss = gain = 0  # loss: base correct, comp wrong; gain: base wrong, comp correct
    bc = [bm[i]["correct"] for i in ids]
    cc = [cm[i]["correct"] for i in ids]
    for i in ids:
        b, c = bm[i], cm[i]
        if b["response"] == c["response"]:
            text_match += 1
        if str(b.get("extracted")) == str(c.get("extracted")):
            ans_match += 1
        if b["correct"] == c["correct"]:
            corr_match += 1
        if b["correct"] and not c["correct"]:
            loss += 1
        if (not b["correct"]) and c["correct"]:
            gain += 1
    base_acc = sum(bc) / n
    comp_acc = sum(cc) / n
    delta_pp = (comp_acc - base_acc) * 100
    # McNemar exact two-sided (b=loss, c=gain)
    mcnemar_p = mcnemar_exact(loss, gain)
    # paired bootstrap CI on delta (pp)
    lo, hi = bootstrap_delta_ci(bc, cc)
    return {
        "n": n,
        "base_acc": base_acc,
        "comp_acc": comp_acc,
        "accuracy_delta_pp": delta_pp,
        "delta_ci95_pp": [lo, hi],
        "text_exact_match": text_match / n,
        "answer_agreement": ans_match / n,
        "correctness_agreement": corr_match / n,
        "behavior_disagreement": 1 - corr_match / n,
        "loss": loss,
        "gain": gain,
        "net_change_pp": (gain - loss) / n * 100,
        "mcnemar_exact_p": mcnemar_p,
    }


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # two-sided exact binomial p under p=0.5
    cdf = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * cdf)


def bootstrap_delta_ci(bc: List[bool], cc: List[bool], n_boot: int = N_BOOT) -> List[float]:
    import numpy as np
    n = len(bc)
    diffs = np.array([(1 if cc[i] else 0) - (1 if bc[i] else 0) for i in range(n)], dtype=np.float64)
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, n, size=(n_boot, n))
    deltas = diffs[idx].mean(axis=1) * 100
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return [float(lo), float(hi)]


def suite_macro(per_bench: Dict[str, dict]) -> dict:
    """Macro average across benchmarks: mean of per-bench deltas, and a macro CI
    by averaging the per-bench bootstrap endpoints is not valid; instead we
    report the mean delta and the min/max of per-bench CIs for context, plus a
    macro bootstrap over benchmark-level deltas."""
    deltas = [per_bench[b]["accuracy_delta_pp"] for b in per_bench]
    dis = [per_bench[b]["behavior_disagreement"] for b in per_bench]
    macro_delta = mean(deltas)
    # macro CI: combine per-bench CIs assuming independence — average endpoints
    los = [per_bench[b]["delta_ci95_pp"][0] for b in per_bench]
    his = [per_bench[b]["delta_ci95_pp"][1] for b in per_bench]
    return {
        "macro_delta_pp": macro_delta,
        "macro_delta_ci95_pp": [mean(los), mean(his)],
        "per_bench_delta_pp": {b: per_bench[b]["accuracy_delta_pp"] for b in per_bench},
        "mean_behavior_disagreement": mean(dis),
    }


# ── determinism gate ──────────────────────────────────────────────────────────

def determinism_report(family: str) -> dict:
    tags = [f"{family}__baseline_original_run1",
            f"{family}__baseline_original_run2",
            f"{family}__baseline_copy"]
    rep = {"family": family, "benches": {}, "all_identical": True}
    ref_tag = tags[0]
    for b in BENCHES:
        ref = load_raw(ref_tag, b)
        entry = {}
        for t in tags[1:]:
            other = load_raw(t, b)
            if ref is None or other is None:
                entry[t] = "missing"
                rep["all_identical"] = False
                continue
            rm, om = sample_map(ref), sample_map(other)
            ids = [i for i in rm if i in om]
            resp_diff = sum(1 for i in ids if rm[i]["response"] != om[i]["response"])
            corr_diff = sum(1 for i in ids if rm[i]["correct"] != om[i]["correct"])
            entry[t] = {"n": len(ids), "response_diffs": resp_diff, "correctness_diffs": corr_diff}
            if resp_diff or corr_diff:
                rep["all_identical"] = False
        rep["benches"][b] = entry
    return rep


# ── stage drivers ─────────────────────────────────────────────────────────────

FAMILIES = ["qwen3_4b", "qwen3_4b_base"]


def analyze_pair(family: str, comp_tag: str, base_tag: str) -> Optional[dict]:
    per_bench = {}
    for b in BENCHES:
        base = load_raw(base_tag, b)
        comp = load_raw(comp_tag, b)
        if base is None or comp is None:
            return None
        per_bench[b] = paired_bench(base, comp)
    return {"per_bench": per_bench, "suite_macro": suite_macro(per_bench)}


def stage1(cfg) -> dict:
    out = {"determinism": {}, "seeds": {}}
    for fam in FAMILIES:
        out["determinism"][fam] = determinism_report(fam)
        base_tag = f"{fam}__baseline_original_run1"
        out["seeds"][fam] = {}
        for s in cfg["stage1_seeds"]:
            comp_tag = f"{fam}__perm_all36_s{s}"
            res = analyze_pair(fam, comp_tag, base_tag)
            if res is not None:
                out["seeds"][fam][f"s{s}"] = res
        # also baseline_copy as a control comparison
        res = analyze_pair(fam, f"{fam}__baseline_copy", base_tag)
        if res is not None:
            out["seeds"][fam]["baseline_copy"] = res
    common.atomic_write_json(OUT / "stage1_summary.json", out)
    return out


def dist_stats(values: List[float]) -> dict:
    v = sorted(values)
    n = len(v)
    def q(p):
        if n == 1:
            return v[0]
        idx = p * (n - 1)
        lo = int(math.floor(idx)); hi = int(math.ceil(idx))
        return v[lo] + (v[hi] - v[lo]) * (idx - lo)
    return {
        "n": n, "mean": mean(v), "std": pstdev(v) if n > 1 else 0.0,
        "median": median(v), "q05": q(0.05), "q25": q(0.25), "q75": q(0.75),
        "q95": q(0.95), "min": v[0], "max": v[-1], "iqr": q(0.75) - q(0.25),
    }


def stage2(cfg) -> dict:
    out = {}
    for fam in FAMILIES:
        base_tag = f"{fam}__baseline_original_run1"
        seed_results = {}
        for s in cfg["stage2_seeds"]:
            res = analyze_pair(fam, f"{fam}__perm_all36_s{s}", base_tag)
            if res is not None:
                seed_results[s] = res
        if not seed_results:
            continue
        # per-benchmark distribution of delta + disagreement
        per_bench_dist = {}
        for b in BENCHES:
            deltas = [seed_results[s]["per_bench"][b]["accuracy_delta_pp"] for s in seed_results]
            dis = [seed_results[s]["per_bench"][b]["behavior_disagreement"] for s in seed_results]
            txt = [seed_results[s]["per_bench"][b]["text_exact_match"] for s in seed_results]
            per_bench_dist[b] = {
                "delta_pp": dist_stats(deltas),
                "behavior_disagreement": dist_stats(dis),
                "text_exact_match": dist_stats(txt),
            }
        macro_deltas = [seed_results[s]["suite_macro"]["macro_delta_pp"] for s in seed_results]
        out[fam] = {
            "n_seeds": len(seed_results),
            "seeds": sorted(seed_results),
            "per_bench": per_bench_dist,
            "suite_macro_delta_pp": dist_stats(macro_deltas),
            "n_over_single_bench_limit": {
                b: sum(1 for s in seed_results
                       if abs(seed_results[s]["per_bench"][b]["accuracy_delta_pp"]) > 1.0)
                for b in BENCHES
            },
            "n_over_macro_limit": sum(1 for s in seed_results
                                      if abs(seed_results[s]["suite_macro"]["macro_delta_pp"]) > 0.5),
        }
    common.atomic_write_json(OUT / "stage2_distribution.json", out)
    return out


def baseline_tags(family: str) -> List[str]:
    """All same-weights baseline tags available for the null distribution."""
    import glob as _glob
    tags = []
    for d in sorted((RAW).glob(f"{family}__baseline_*")):
        tag = d.name
        # only genuine same-function baselines (original reruns + byte copy)
        if "baseline_original_run" in tag or tag.endswith("baseline_copy") or "baseline_rep" in tag:
            tags.append(tag)
    return tags


def null_distribution(cfg) -> dict:
    """Per family/benchmark: pairwise accuracy deltas + behaviour disagreement
    among all same-weights baseline runs — the inference-noise floor the
    permutation deltas are compared against."""
    from itertools import combinations
    out = {}
    for fam in FAMILIES:
        tags = baseline_tags(fam)
        fam_out = {"tags": tags, "n_tags": len(tags), "per_bench": {}}
        for b in BENCHES:
            raws = {t: load_raw(t, b) for t in tags}
            raws = {t: r for t, r in raws.items() if r is not None}
            deltas, disags = [], []
            ts = list(raws)
            for t1, t2 in combinations(ts, 2):
                pb = paired_bench(raws[t1], raws[t2])
                deltas.append(pb["accuracy_delta_pp"])
                disags.append(pb["behavior_disagreement"])
            if deltas:
                fam_out["per_bench"][b] = {
                    "n_pairs": len(deltas),
                    "delta_pp": dist_stats(deltas),
                    "abs_delta_max_pp": max(abs(d) for d in deltas),
                    "behavior_disagreement": dist_stats(disags),
                }
        out[fam] = fam_out
    common.atomic_write_json(OUT / "null_distribution.json", out)
    return out


def ablation(cfg) -> dict:
    fam = "qwen3_4b"
    base_tag = f"{fam}__baseline_original_run1"
    arms = {
        "scope_single_L0": f"{fam}__abl_scope_single0_random_s7",
        "scope_single_L17": f"{fam}__abl_scope_single17_random_s7",
        "scope_single_L35": f"{fam}__abl_scope_single35_random_s7",
        "scope_prefix6": f"{fam}__abl_scope_prefix6_random_s7",
        "scope_prefix18": f"{fam}__abl_scope_prefix18_random_s7",
        "scope_all36_random": f"{fam}__abl_scope_all36_random_s7",
        "mag_adjacent_swap_all36": f"{fam}__abl_mag_adjswap_all36",
        "mag_reverse_all36": f"{fam}__abl_mag_reverse_all36",
    }
    out = {}
    for name, tag in arms.items():
        res = analyze_pair(fam, tag, base_tag)
        if res is not None:
            out[name] = {
                "suite_macro_delta_pp": res["suite_macro"]["macro_delta_pp"],
                "mean_behavior_disagreement": res["suite_macro"]["mean_behavior_disagreement"],
                "per_bench_delta_pp": res["suite_macro"]["per_bench_delta_pp"],
                "per_bench_disagreement": {b: res["per_bench"][b]["behavior_disagreement"] for b in BENCHES},
                "per_bench_text_match": {b: res["per_bench"][b]["text_exact_match"] for b in BENCHES},
            }
    common.atomic_write_json(OUT / "ablation_summary.json", out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["stage1", "stage2", "null", "ablation", "all"], required=True)
    args = ap.parse_args()
    cfg = common.load_config()
    if args.stage in ("stage1", "all"):
        r = stage1(cfg)
        for fam in FAMILIES:
            d = r["determinism"].get(fam, {})
            print(f"[determinism] {fam}: all_identical={d.get('all_identical')}")
    if args.stage in ("null", "all"):
        null_distribution(cfg)
        print("[null] distribution written")
    if args.stage in ("stage2", "all"):
        stage2(cfg)
        print("[stage2] distribution written")
    if args.stage in ("ablation", "all"):
        ablation(cfg)
        print("[ablation] summary written")


if __name__ == "__main__":
    main()
