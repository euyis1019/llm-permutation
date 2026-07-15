"""Aggregate A/B/C results into machine-readable summary tables (stdout JSON +
markdown fragments) used to author RESULT.md. Read-only over results/."""

import argparse
import json
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")


def R(*p):
    return os.path.join(RESULTS_DIR, *p)

CONTROL_ORDER = [
    "baseline-repeat", "valid-triplet", "gate-only", "up-only", "down-only",
    "gate+up", "gate+down", "up+down", "independent-triplet", "wrong-direction",
]


def stage_b():
    recs = [json.loads(l) for l in open(R("single_mlp.jsonl"))]
    out = {"n_cases": len(recs)}
    out["all_restores_ok"] = all(
        r["restore_equal"] and r["restore_sha_match"] for r in recs
    )
    det, coord, canon = [], [], []
    groups, groups_by_layer = {}, {}
    canon_metrics = []
    for r in recs:
        for iname, e in r["inputs"].items():
            det.append(e["baseline_deterministic"])
            groups.setdefault(r["control"], []).append(e["vs_baseline"]["rel_l2"])
            groups_by_layer.setdefault((r["layer"], r["control"]), []).append(
                e["vs_baseline"]["rel_l2"]
            )
            if r["control"] == "valid-triplet":
                coord.append(
                    e["gate_coordinate_equal"]
                    and e["up_coordinate_equal"]
                    and e["product_coordinate_equal"]
                )
                canon.append(e["canonical_down_bitwise_equal"])
                canon_metrics.append(e["canonical_vs_baseline"]["rel_l2"])
    out["baseline_deterministic_all"] = all(det)
    out["coord_align_all"] = all(coord) and len(coord) > 0
    out["canonical_down_bitwise_all"] = all(canon)
    out["canonical_down_rel_l2_max"] = max(canon_metrics)
    out["groups"] = {
        k: {
            "n": len(v),
            "median_rel_l2": st.median(v),
            "min": min(v),
            "max": max(v),
        }
        for k, v in groups.items()
    }
    neg = [
        x for k, v in groups.items()
        if k not in ("baseline-repeat", "valid-triplet") for x in v
    ]
    out["separation_ratio_median"] = st.median(neg) / st.median(
        groups["valid-triplet"]
    )
    out["separation_ratio_worstcase"] = min(neg) / max(groups["valid-triplet"])
    out["by_layer_valid"] = {
        str(k[0]): {
            "median": st.median(v), "max": max(v)
        }
        for k, v in groups_by_layer.items() if k[1] == "valid-triplet"
    }
    return out


def stage_c():
    recs = [json.loads(l) for l in open(R("full_model.jsonl"))]
    out = {"cases": {}}
    for r in recs:
        key = r["case_key"]
        if key == "baseline-repeat":
            out["baseline_repeat"] = {
                "n_prompts": len(r["prompts"]),
                "logits_bitwise_all": all(p["logits_bitwise"] for p in r["prompts"]),
                "streams_bitwise_all": all(p["streams_bitwise"] for p in r["prompts"]),
            }
            continue
        ps = r["prompts"]
        rel = [p["logits"]["full"]["rel_l2"] for p in ps]
        lt_rel = [p["logits"]["last_token"]["rel_l2"] for p in ps]
        n_tok = sum(p["logits"]["n_tokens"] for p in ps)
        n_flip = sum(p["logits"]["n_top1_flips"] for p in ps)
        flips_m = [m for p in ps for m in p["logits"]["flip_baseline_margins"]]
        overall_m = [p["logits"]["median_baseline_margin"] for p in ps]
        first = [p.get("first_diff") for p in ps if p.get("first_diff")]
        out["cases"][key] = {
            "control": r["control"],
            "seed": r["seed"],
            "n_layers_permuted": len(r["layers"]),
            "restore_equal": r.get("restore_equal"),
            "logits_rel_l2_median": st.median(rel),
            "logits_rel_l2_max": max(rel),
            "last_token_rel_l2_median": st.median(lt_rel),
            "cosine_min": min(p["logits"]["full"]["cosine"] for p in ps),
            "top1_agreement": 1 - n_flip / n_tok,
            "n_top1_flips": n_flip,
            "n_tokens": n_tok,
            "flip_margin_median": st.median(flips_m) if flips_m else None,
            "overall_margin_median": st.median(overall_m),
            "top5_jaccard_mean": st.mean(
                p["logits"]["top5_jaccard_mean"] for p in ps
            ),
            "last_token_top1_same_frac": st.mean(
                float(p["logits"]["last_token"]["top1_same"]) for p in ps
            ),
            "first_diff_layers": sorted({f["layer"] for f in first}),
            "first_diff_streams": sorted({f["stream"] for f in first}),
        }
    # generation
    if os.path.exists(R("full_model_generation.jsonl")):
        out["generation"] = {}
        for line in open(R("full_model_generation.jsonl")):
            g = json.loads(line)
            if g["case_key"] == "gen-baseline":
                continue
            ps = g["prompts"]
            out["generation"][g["case_key"]] = {
                "exact_match": sum(p["exact_match"] for p in ps),
                "n": len(ps),
                "diverged_ids": [p["id"] for p in ps if not p["exact_match"]],
            }
    return out


def layerwise_growth():
    """Per-layer error growth for all-36 cases (median across prompts)."""
    recs = [json.loads(l) for l in open(R("full_model.jsonl"))]
    out = {}
    for r in recs:
        if not r["case_key"].startswith(("all-36", "half-18", "prefix-6", "one-layer")):
            continue
        arr = []
        for li in range(36):
            vals = [p["per_layer_rel_l2"]["block_out"][li] for p in r["prompts"]]
            arr.append(st.median(vals))
        out[r["case_key"]] = arr
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=RESULTS_DIR)
    args = ap.parse_args()
    RESULTS_DIR = os.path.abspath(args.results_dir)
    summary = {}
    a = json.load(open(R("synthetic.json")))
    summary["stage_a"] = {"passed": a["passed"], "problems": a["problems"]}
    summary["stage_b"] = stage_b()
    if os.path.exists(R("full_model.jsonl")):
        summary["stage_c"] = stage_c()
        summary["layerwise"] = layerwise_growth()
    print(json.dumps(summary, indent=2, ensure_ascii=False))
