"""Analyze the post-hoc supplementary arms S1/S2/S3 against the frozen baselines."""
from __future__ import annotations
import json, statistics
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parents[1]
RES = EXP / "results"
BASE_UNIT = RES / "part0_run_a"          # identity Base logits (bitwise-verified)
RAW20 = EXP.parent / "ffn_benchmark_eval" / "results" / "raw"

F7_MEDIAN_REL_L2 = json.load(open(RES / "part6_summary.json"))["f7_reference_median_rel_l2"]


def load_logits(unit_dir: Path):
    out = {}
    for meta_path in sorted((unit_dir / "logits").glob("prompt_*.meta.json")):
        pid = int(meta_path.stem.split("_")[1].split(".")[0])
        raw = (unit_dir / "logits" / meta_path.stem.replace(".meta", "")).with_suffix(".raw.bin")
        f32 = np.load(meta_path.parent / json.loads(meta_path.read_text())["float32_file"])
        out[pid] = {"raw": raw.read_bytes(), "f32": f32.astype(np.float64)}
    return out


def compare(unit_dir: Path, base):
    case = load_logits(unit_dir)
    summ = json.loads((unit_dir / "summary.json").read_text())
    nll = {r["prompt_id"]: r["mean_nll"] for r in summ["records"]}
    per, rels = [], []
    n_bitwise, flips = 0, 0
    for pid in sorted(base):
        b, c = base[pid], case[pid]
        bit = b["raw"] == c["raw"]
        rel = float(np.linalg.norm(c["f32"] - b["f32"]) / np.linalg.norm(b["f32"]))
        same = int(np.argmax(c["f32"])) == int(np.argmax(b["f32"]))
        n_bitwise += bit
        flips += not same
        rels.append(rel)
        per.append({"prompt_id": pid, "bitwise_equal": bool(bit), "rel_l2": rel,
                    "top1_same": bool(same)})
    return {"unit": unit_dir.name, "median_rel_l2": statistics.median(rels),
            "f7_ratio": statistics.median(rels) / F7_MEDIAN_REL_L2,
            "n_bitwise_equal": n_bitwise, "top1_flips": flips,
            "mean_nll": summ["mean_nll"], "per_prompt": per}


def main():
    base = load_logits(BASE_UNIT)
    base_nll = json.loads((BASE_UNIT / "summary.json").read_text())["mean_nll"]

    arms = {"s1": [], "s2": []}
    for unit_dir in sorted((RES / "supp_units").iterdir()):
        if not (unit_dir / "summary.json").is_file():
            continue
        arm = unit_dir.name.split("_")[1]
        stats_file = RES / "supp_weight_stats" / f"{unit_dir.name}.json"
        wstats = json.loads(stats_file.read_text()) if stats_file.is_file() else {}
        r = compare(unit_dir, base)
        r["sigma"] = wstats.get("sigma")
        r["scope"] = wstats.get("scope")
        r["dnll"] = r["mean_nll"] - base_nll
        r["weight_changed_fraction"] = wstats.get("stats", {}).get("changed_fraction")
        r["weight_rel_l2"] = wstats.get("stats", {}).get("weight_rel_l2")
        arms[arm].append(r)

    # S3: GSM8K accs vs the 20-seed permutation null
    null_accs = []
    for s in range(1000, 1020):
        d = json.load(open(RAW20 / f"qwen3_4b_base__perm_all36_s{s}" / "gsm8k.raw.json"))
        null_accs.append(d["accuracy"])
    bl = json.load(open(RAW20 / "qwen3_4b_base__baseline_original_run1" / "gsm8k.raw.json"))["accuracy"]
    s3 = []
    for d3 in sorted((RES / "supp_behavior").iterdir()):
        f = d3 / "gsm8k.raw.json"
        if not f.is_file():
            continue
        acc = json.load(open(f))["accuracy"]
        wstats = json.loads((RES / "supp_weight_stats" / f"{d3.name}.json").read_text())
        s3.append({"unit": d3.name, "sigma": wstats["sigma"], "seed": wstats["seed"],
                   "gsm8k_acc": acc,
                   "inside_null_range": min(null_accs) <= acc <= max(null_accs),
                   "delta_vs_baseline_pp": 100 * (acc - bl),
                   "delta_vs_null_mean_pp": 100 * (acc - statistics.mean(null_accs))})
    out = {
        "f7_median_rel_l2": F7_MEDIAN_REL_L2,
        "s1_left_edge": sorted(arms["s1"], key=lambda r: (r["sigma"], r["unit"])),
        "s2_ffn_scope": sorted(arms["s2"], key=lambda r: (r["sigma"], r["unit"])),
        "s3_null": {"n": 20, "mean": statistics.mean(null_accs),
                    "sd": statistics.pstdev(null_accs),
                    "min": min(null_accs), "max": max(null_accs), "baseline": bl},
        "s3_behavior": s3,
    }
    outp = EXP / "reviewer_analysis" / "supp_summary.json"
    json.dump(out, open(outp, "w"), indent=2)
    for arm in ("s1_left_edge", "s2_ffn_scope"):
        for r in out[arm]:
            print(f"[{arm}] {r['unit']} sigma={r['sigma']:.0e} scope={r['scope']} "
                  f"changed={r['weight_changed_fraction']:.3e} "
                  f"med_rel_l2={r['median_rel_l2']:.6f} f7_ratio={r['f7_ratio']:.3f} "
                  f"bitwise={r['n_bitwise_equal']}/32 flips={r['top1_flips']}")
    print(f"[s3] null: mean={out['s3_null']['mean']:.4f} sd={out['s3_null']['sd']:.4f} "
          f"range=[{out['s3_null']['min']:.4f},{out['s3_null']['max']:.4f}] baseline={bl:.4f}")
    for r in s3:
        print(f"[s3] {r['unit']} sigma={r['sigma']:.0e} acc={r['gsm8k_acc']:.4f} "
              f"in_null_range={r['inside_null_range']} dBaseline={r['delta_vs_baseline_pp']:+.2f}pp")
    print("wrote", outp)


if __name__ == "__main__":
    main()
