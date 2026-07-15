"""Deterministic analyses and machine-readable acceptance for noise_floor."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import tempfile
from pathlib import Path

import numpy as np


EXP_ROOT = Path(__file__).resolve().parents[1]
RESULTS = EXP_ROOT / "results"
FFN_EVAL = EXP_ROOT.parent / "ffn_benchmark_eval"
SIGMAS = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]


def atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def load_summary(path: Path) -> dict:
    d = json.loads((path / "summary.json").read_text())
    if not d.get("complete") or d.get("n_prompts") != 32:
        raise RuntimeError(f"incomplete logits unit: {path}")
    return d


def record_map(summary: dict) -> dict[int, dict]:
    return {int(r["prompt_id"]): r for r in summary["records"]}


def raw_bytes(unit: Path, rec: dict) -> bytes:
    return (unit / rec["logits"]["raw_path"]).read_bytes()


def f32(unit: Path, rec: dict) -> np.ndarray:
    return np.load(unit / rec["logits"]["float32_path"], allow_pickle=False)


def compare_prompt(base_unit: Path, base_rec: dict, case_unit: Path, case_rec: dict) -> dict:
    br = raw_bytes(base_unit, base_rec)
    cr = raw_bytes(case_unit, case_rec)
    b = f32(base_unit, base_rec).astype(np.float64)
    c = f32(case_unit, case_rec).astype(np.float64)
    diff = c - b
    return {
        "bitwise_equal": br == cr,
        "max_abs": float(np.max(np.abs(diff))),
        "rel_l2": float(np.linalg.norm(diff) / max(np.linalg.norm(b), 1e-300)),
        "top1_same": bool(np.argmax(b) == np.argmax(c)),
        "baseline_top1": int(np.argmax(b)),
        "case_top1": int(np.argmax(c)),
        "n_diff_float32": int(np.count_nonzero(b != c)),
    }


def acceptance() -> dict:
    path = RESULTS / "acceptance_noise_floor.json"
    if path.is_file():
        return json.loads(path.read_text())
    return {
        "experiment": "noise_floor",
        "preregistration": str(EXP_ROOT / "EXPERIMENT_PLAN.md"),
        "criteria": {
            k: {"status": "pending"}
            for k in ["S0-1", "S1-1", "S1-2", "S1-3", "S1-4", "S2-1", "P6-1", "P6-2"]
        },
    }


def save_acceptance(acc: dict) -> None:
    atomic_json(RESULTS / "acceptance_noise_floor.json", acc)


def part0() -> bool:
    a_dir, b_dir = RESULTS / "part0_run_a", RESULTS / "part0_run_b"
    a, b = load_summary(a_dir), load_summary(b_dir)
    am, bm = record_map(a), record_map(b)
    rows = []
    for pid in sorted(am):
        rows.append({"prompt_id": pid, **compare_prompt(a_dir, am[pid], b_dir, bm[pid])})
    out = {
        "run_a": str(a_dir),
        "run_b": str(b_dir),
        "n_prompts": len(rows),
        "n_bitwise_equal": sum(r["bitwise_equal"] for r in rows),
        "all_bitwise_equal": all(r["bitwise_equal"] for r in rows),
        "records": rows,
    }
    atomic_json(RESULTS / "part0_compare.json", out)
    passed = out["n_bitwise_equal"] == 32
    acc = acceptance()
    acc["criteria"]["S0-1"] = {
        "status": "pass" if passed else "fail",
        "hard": True,
        "observed": f"{out['n_bitwise_equal']}/32 bitwise equal",
        "required": "32/32 bitwise equal",
        "evidence": "results/part0_compare.json",
    }
    save_acceptance(acc)
    print(f"S0-1 {'PASS' if passed else 'FAIL'}: {out['n_bitwise_equal']}/32")
    return passed


def part1a() -> bool:
    base_dir = RESULTS / "part0_run_a"
    base = load_summary(base_dir)
    bm = record_map(base)
    variants = {
        "identity": base_dir,
        "F9-K100-all36": RESULTS / "units" / "part1a_f9_k100",
        "F10-K100-all36": RESULTS / "units" / "part1a_f10_k100",
        "F7-all36": RESULTS / "units" / "part1a_f7",
    }
    rows = []
    medians = {}
    for variant, unit in variants.items():
        s = base if variant == "identity" else load_summary(unit)
        sm = record_map(s)
        rels = []
        for pid in sorted(bm):
            cmp = compare_prompt(base_dir, bm[pid], unit, sm[pid])
            rels.append(cmp["rel_l2"])
            rows.append(
                {
                    "variant": variant,
                    "prompt_id": pid,
                    **cmp,
                    "mean_nll": sm[pid]["mean_nll"],
                    "baseline_mean_nll": bm[pid]["mean_nll"],
                }
            )
        medians[variant] = float(statistics.median(rels))
    write_jsonl(RESULTS / "part1a_logits.jsonl", rows)

    f9 = [r for r in rows if r["variant"] == "F9-K100-all36"]
    f10 = [r for r in rows if r["variant"] == "F10-K100-all36"]
    f7 = [r for r in rows if r["variant"] == "F7-all36"]
    s11 = len(f9) == 32 and all(r["bitwise_equal"] for r in f9)
    ratio = medians["F10-K100-all36"] / max(medians["F7-all36"], 1e-300)
    s12 = (
        all(not r["bitwise_equal"] for r in f10)
        and all(not r["bitwise_equal"] for r in f7)
        and 0.2 <= ratio <= 5.0
    )
    summary = {
        "medians_rel_l2": medians,
        "f9_bitwise_equal": sum(r["bitwise_equal"] for r in f9),
        "f10_bitwise_equal": sum(r["bitwise_equal"] for r in f10),
        "f7_bitwise_equal": sum(r["bitwise_equal"] for r in f7),
        "f10_to_f7_median_ratio": ratio,
    }
    atomic_json(RESULTS / "part1a_summary.json", summary)
    acc = acceptance()
    acc["criteria"]["S1-1"] = {
        "status": "pass" if s11 else "fail", "hard": True,
        "observed": f"{summary['f9_bitwise_equal']}/32 F9 bitwise equal",
        "required": "32/32", "evidence": "results/part1a_summary.json",
    }
    acc["criteria"]["S1-2"] = {
        "status": "pass" if s12 else "fail", "hard": False,
        "observed": {
            "f10_all_nonbitwise": all(not r["bitwise_equal"] for r in f10),
            "f7_all_nonbitwise": all(not r["bitwise_equal"] for r in f7),
            "median_ratio_f10_over_f7": ratio,
        },
        "required": "both non-bitwise and median ratio in [0.2,5]",
        "evidence": "results/part1a_summary.json",
    }
    save_acceptance(acc)
    print(f"S1-1 {'PASS' if s11 else 'FAIL'}; S1-2 {'PASS' if s12 else 'FAIL'} ratio={ratio:.6g}")
    return s11


def compare_benchmark_arm(tag: str) -> dict:
    base_root = FFN_EVAL / "results" / "raw" / "qwen3_4b_base__baseline_original_run1"
    arm_root = RESULTS / "part1b" / tag
    per = {}
    all_mismatch = []
    for bench in ["mmlu", "gsm8k", "ceval", "cmmlu", "humaneval_plus", "mbpp_plus"]:
        base = json.loads((base_root / f"{bench}.raw.json").read_text())
        case = json.loads((arm_root / f"{bench}.raw.json").read_text())
        bm = {str(x["sample_id"]): x for x in base["samples"]}
        cm = {str(x["sample_id"]): x for x in case["samples"]}
        if bm.keys() != cm.keys():
            raise AssertionError(f"sample ids differ: {tag}/{bench}")
        corr_same = 0
        resp_same = 0
        mismatch = []
        for sid in bm:
            cs = bool(bm[sid]["correct"]) == bool(cm[sid]["correct"])
            rs = bm[sid]["response"].encode("utf-8") == cm[sid]["response"].encode("utf-8")
            corr_same += int(cs)
            resp_same += int(rs)
            if not cs or not rs:
                item = {"benchmark": bench, "sample_id": sid, "correctness_same": cs, "response_bytes_same": rs}
                mismatch.append(item)
                all_mismatch.append(item)
        total = len(bm)
        per[bench] = {
            "total": total,
            "correctness_same": corr_same,
            "correctness_disagreement_rate": (total - corr_same) / total,
            "response_bytes_same": resp_same,
            "response_match_rate": resp_same / total,
            "baseline_correct": base["correct"],
            "case_correct": case["correct"],
            "delta_pp": 100.0 * (case["accuracy"] - base["accuracy"]),
            "mismatches": mismatch,
        }
    return {
        "tag": tag,
        "per_benchmark": per,
        "mean_correctness_disagreement": statistics.mean(x["correctness_disagreement_rate"] for x in per.values()),
        "suite_macro_delta_pp": statistics.mean(x["delta_pp"] for x in per.values()),
        "all_mismatches": all_mismatch,
    }


def part1b(tags: list[str], final: bool) -> bool:
    arms = {tag: compare_benchmark_arm(tag) for tag in tags}
    atomic_json(RESULTS / "part1b_compare.json", {"arms": arms})
    f9tag = "f9_k100_all36"
    f9 = arms[f9tag]
    s13 = all(x["correctness_disagreement_rate"] == 0 for x in f9["per_benchmark"].values())
    acc = acceptance()
    acc["criteria"]["S1-3"] = {
        "status": "pass" if s13 else "fail", "hard": True,
        "observed": {b: x["correctness_disagreement_rate"] for b, x in f9["per_benchmark"].items()},
        "required": "zero correctness differences on all six benchmarks",
        "evidence": "results/part1b_compare.json",
    }
    if final:
        f10 = arms["f10_k100_all36"]["mean_correctness_disagreement"]
        f3 = arms["f3_k30_all36_s301"]["mean_correctness_disagreement"]
        ratio = f10 / max(f3, 1e-300)
        s14 = 0.2 <= ratio <= 5.0
        acc["criteria"]["S1-4"] = {
            "status": "pass" if s14 else "fail", "hard": False,
            "observed": {"f10_rate": f10, "f3_rate": f3, "ratio": ratio},
            "required": "ratio in [0.2,5]", "evidence": "results/part1b_compare.json",
        }
    save_acceptance(acc)
    print(f"S1-3 {'PASS' if s13 else 'FAIL'}")
    return s13


def part2() -> bool:
    base_dir = RESULTS / "units" / "part2_identity"
    base = load_summary(base_dir)
    bm = record_map(base)
    mapping = {
        "scope_single_L0": "part2_scope_single0",
        "scope_single_L17": "part2_scope_single17",
        "scope_single_L35": "part2_scope_single35",
        "scope_prefix6": "part2_scope_prefix6",
        "scope_prefix18": "part2_scope_prefix18",
        "scope_all36_random": "part2_scope_all36",
        "mag_adjacent_swap_all36": "part2_mag_adjswap",
        "mag_reverse_all36": "part2_mag_reverse",
    }
    behavior = json.loads((FFN_EVAL / "results" / "ablation_summary.json").read_text())
    rows = []
    for anchor, unit_name in mapping.items():
        unit = RESULTS / "units" / unit_name
        s = load_summary(unit)
        sm = record_map(s)
        comps = [compare_prompt(base_dir, bm[pid], unit, sm[pid]) for pid in sorted(bm)]
        b_all = np.concatenate([f32(base_dir, bm[pid]).astype(np.float64) for pid in sorted(bm)])
        c_all = np.concatenate([f32(unit, sm[pid]).astype(np.float64) for pid in sorted(bm)])
        rel = float(np.linalg.norm(c_all - b_all) / max(np.linalg.norm(b_all), 1e-300))
        rows.append({
            "anchor": anchor,
            "unit": unit_name,
            "rel_l2": rel,
            "n_bitwise_equal": sum(x["bitwise_equal"] for x in comps),
            "top1_flips": sum(not x["top1_same"] for x in comps),
            "mean_nll": s["mean_nll"],
            "baseline_mean_nll": base["mean_nll"],
            "existing_mean_behavior_disagreement": behavior[anchor]["mean_behavior_disagreement"],
            "per_prompt": comps,
        })
    write_jsonl(RESULTS / "part2_anchor_drift.jsonl", rows)
    from scipy.stats import spearmanr
    rho_result = spearmanr(
        [x["rel_l2"] for x in rows],
        [x["existing_mean_behavior_disagreement"] for x in rows],
    )
    rho = float(rho_result.statistic)
    pvalue = float(rho_result.pvalue)
    passed = rho >= 0.9
    summary = {"spearman_rho": rho, "pvalue": pvalue, "n_anchors": 8}
    atomic_json(RESULTS / "part2_summary.json", summary)
    acc = acceptance()
    acc["criteria"]["S2-1"] = {
        "status": "pass" if passed else "fail", "hard": False,
        "observed": rho, "required": ">=0.9", "evidence": "results/part2_summary.json",
    }
    save_acceptance(acc)
    print(f"S2-1 {'PASS' if passed else 'FAIL'} rho={rho:.6f}")
    return passed


def _sigma_tag(idx: int, rep: int) -> str:
    return f"sigma_{idx:02d}_rep{rep}"


def part6() -> dict:
    base_dir = RESULTS / "part0_run_a"
    base = load_summary(base_dir)
    bm = record_map(base)
    p1rows = [json.loads(x) for x in (RESULTS / "part1a_logits.jsonl").read_text().splitlines() if x.strip()]
    f7_reference = statistics.median(x["rel_l2"] for x in p1rows if x["variant"] == "F7-all36")
    f10_reference = statistics.median(x["rel_l2"] for x in p1rows if x["variant"] == "F10-K100-all36")
    rows = []
    curves = []
    weight_units = []
    for idx, sigma in enumerate(SIGMAS):
        tier_rels = []
        tier_bitwise = 0
        for rep in range(3):
            tag = _sigma_tag(idx, rep)
            unit = RESULTS / "units" / tag
            s = load_summary(unit)
            sm = record_map(s)
            comps = []
            for pid in sorted(bm):
                cmp = compare_prompt(base_dir, bm[pid], unit, sm[pid])
                tier_rels.append(cmp["rel_l2"])
                tier_bitwise += int(cmp["bitwise_equal"])
                comps.append({"prompt_id": pid, **cmp})
            seed = 1000 + 10 * idx + rep
            rows.append({
                "sigma_index": idx, "sigma": sigma, "rep": rep, "seed": seed,
                "unit_tag": tag,
                "median_rel_l2": float(statistics.median(x["rel_l2"] for x in comps)),
                "n_bitwise_equal": sum(x["bitwise_equal"] for x in comps),
                "top1_flips": sum(not x["top1_same"] for x in comps),
                "mean_nll": s["mean_nll"],
                "baseline_mean_nll": base["mean_nll"],
                "per_prompt": comps,
            })
            weight_path = RESULTS / "part6_weight_stats_units" / f"{tag}.json"
            wm = json.loads(weight_path.read_text())
            if wm.get("stats_collected"):
                weight_units.append(wm)
        median = float(statistics.median(tier_rels))
        curves.append({
            "sigma_index": idx, "sigma": sigma,
            "median_rel_l2": median,
            "f7_ratio": median / max(f7_reference, 1e-300),
            "n_bitwise_equal": tier_bitwise,
            "n_measurements": len(tier_rels),
        })
    write_jsonl(RESULTS / "part6_sigma_sweep.jsonl", rows)

    sigma_star = None
    sigma_star_method = "no_crossing"
    for i, point in enumerate(curves):
        if point["median_rel_l2"] >= f7_reference:
            if i == 0:
                sigma_star = point["sigma"]
                sigma_star_method = "at_or_below_lowest_grid"
            else:
                lo, hi = curves[i - 1], point
                if lo["median_rel_l2"] > 0 and hi["median_rel_l2"] > lo["median_rel_l2"]:
                    x0, x1 = math.log(lo["sigma"]), math.log(hi["sigma"])
                    y0, y1 = math.log(lo["median_rel_l2"]), math.log(hi["median_rel_l2"])
                    sigma_star = math.exp(x0 + (math.log(f7_reference) - y0) * (x1 - x0) / (y1 - y0))
                    sigma_star_method = "log_log_interpolation"
                else:
                    sigma_star = hi["sigma"]
                    sigma_star_method = "upper_grid_bound_due_zero_or_nonmonotonic_lower"
            break

    plateau_pairs = []
    for i in range(1, len(curves) - 1):
        a, b = curves[i], curves[i + 1]
        adjacent_ratio = b["median_rel_l2"] / max(a["median_rel_l2"], 1e-300)
        smaller_all_bitwise = all(x["n_bitwise_equal"] == x["n_measurements"] for x in curves[:i])
        if (
            a["median_rel_l2"] > 0 and b["median_rel_l2"] > 0
            and 1 / 3 <= adjacent_ratio <= 3
            and 1 / 3 <= a["f7_ratio"] <= 3
            and 1 / 3 <= b["f7_ratio"] <= 3
            and smaller_all_bitwise
        ):
            plateau_pairs.append({"lower_sigma": a["sigma"], "upper_sigma": b["sigma"], "adjacent_ratio": adjacent_ratio})
    p61 = bool(plateau_pairs)
    p62 = sigma_star is not None and sigma_star >= 1e-4
    part6c_eligible = sigma_star is not None and 1e-4 <= sigma_star <= 1e-2

    weight_summary = []
    for idx, sigma in enumerate(SIGMAS):
        reps = [x for x in weight_units if int(x["seed"]) // 10 == (1000 + 10 * idx) // 10]
        # Exact selection avoids assumptions about decimal formatting.
        reps = [x for x in weight_units if float(x["sigma"]) == sigma]
        weight_summary.append({
            "sigma_index": idx, "sigma": sigma,
            "measurement_stats": [
                {"seed": x["seed"], **x["stats"]["all"]} for x in reps
            ],
            "layers_rep0": reps[0]["stats"] if reps else None,
        })
    atomic_json(RESULTS / "part6_weight_quant.json", {"scope": "all unique floating named_parameters", "tiers": weight_summary})
    summary = {
        "f7_reference_median_rel_l2": f7_reference,
        "f10_reference_median_rel_l2": f10_reference,
        "curve": curves,
        "sigma_star": sigma_star,
        "sigma_star_method": sigma_star_method,
        "plateau_pairs": plateau_pairs,
        "P6-1": p61,
        "P6-2": p62,
        "part6c": {
            "eligible": part6c_eligible,
            "status": "pending" if part6c_eligible else "skipped",
            "reason": None if part6c_eligible else "sigma_star outside [1e-4,1e-2]",
        },
    }
    atomic_json(RESULTS / "part6_summary.json", summary)
    acc = acceptance()
    acc["criteria"]["P6-1"] = {
        "status": "pass" if p61 else "fail", "hard": False, "prediction": True,
        "observed": plateau_pairs,
        "required": "a plateau segment at F7 ratio [1/3,3] with all smaller sigma tiers bitwise",
        "evidence": "results/part6_summary.json",
    }
    acc["criteria"]["P6-2"] = {
        "status": "pass" if p62 else "fail", "hard": False, "prediction": True,
        "observed": {"estimate": sigma_star, "method": sigma_star_method},
        "required": "sigma_star >= 1e-4",
        "evidence": "results/part6_summary.json",
    }
    acc["part6c"] = summary["part6c"]
    acc["overall"] = {
        "status": "complete_with_soft_failures",
        "all_hard_criteria_passed": all(
            acc["criteria"][key]["status"] == "pass"
            for key in ("S0-1", "S1-1", "S1-3")
        ),
        "hard_failures": [
            key for key in ("S0-1", "S1-1", "S1-3")
            if acc["criteria"][key]["status"] == "fail"
        ],
        "soft_or_prediction_failures": [
            key for key in ("S1-2", "S1-4", "S2-1", "P6-1", "P6-2")
            if acc["criteria"][key]["status"] == "fail"
        ],
    }
    save_acceptance(acc)
    print(f"P6-1 {'PASS' if p61 else 'FAIL'}; P6-2 {'PASS' if p62 else 'FAIL'} sigma*={sigma_star}")
    return summary


def manifest() -> None:
    files = []
    candidates = []
    for root in [RESULTS, EXP_ROOT / "scripts", EXP_ROOT / "logs", EXP_ROOT / "checkpoints"]:
        candidates.extend(p for p in root.rglob("*") if p.is_file())
    candidates.extend(
        p for p in [
            EXP_ROOT / "EXPERIMENT_PLAN.md", EXP_ROOT / "DECISIONS.md",
            EXP_ROOT / "PROGRESS.md", EXP_ROOT / "EXECUTION_REPORT.md",
            EXP_ROOT / "FAILURE_REPORT_noise_floor.md",
        ] if p.is_file()
    )
    for p in sorted(set(candidates)):
        if p == RESULTS / "manifest.json" or "__pycache__" in p.parts:
            continue
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(8 << 20), b""):
                h.update(chunk)
        files.append({"path": str(p.relative_to(EXP_ROOT)), "bytes": p.stat().st_size, "sha256": h.hexdigest()})
    import torch, transformers, vllm
    atomic_json(RESULTS / "manifest.json", {
        "files": files,
        "environment": {
            "python": __import__("sys").version,
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "transformers": transformers.__version__, "vllm": vllm.__version__,
        },
    })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["part0", "part1a", "part1b_f9", "part1b_all", "part2", "part6", "manifest"])
    args = ap.parse_args()
    if args.stage == "part0": part0()
    elif args.stage == "part1a": part1a()
    elif args.stage == "part1b_f9": part1b(["f9_k100_all36"], False)
    elif args.stage == "part1b_all": part1b(["f9_k100_all36", "f10_k100_all36", "f3_k30_all36_s301"], True)
    elif args.stage == "part2": part2()
    elif args.stage == "part6": part6()
    elif args.stage == "manifest": manifest()


if __name__ == "__main__":
    main()
