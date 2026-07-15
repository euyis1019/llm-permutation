"""Analyze Amendment-v1.1 outputs without executing any additional measurements."""

from __future__ import annotations

import json
import math
import os
import statistics
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
RAW = RESULTS / "stage1b_singlelayer.jsonl"


def atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_jsonl(path: Path, rows: list[dict]) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def pct(v: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(v, dtype=np.float64), q))


def main() -> None:
    rows = []
    with RAW.open() as f:
        for line_no, line in enumerate(f, 1):
            row = json.loads(line)
            row["raw_line"] = line_no
            rows.append(row)
    keys = {(r["perm_key"], r["layer"], r["prompt_id"], r["backend"], r["shape"])
            for r in rows}
    if len(rows) != 1248 or len(keys) != 1248:
        raise RuntimeError(f"Stage 1b raw data incomplete: rows={len(rows)} keys={len(keys)}")

    # Operationalize the amendment's "corresponding backend median ceiling":
    # the median of the pre-registered predicted-ceil arm for that backend.
    ceiling_estimate = {
        b: statistics.median(r["drift"]["rel_l2"] for r in rows
                             if r["backend"] == b and r["predicted_tier"] == "ceil")
        for b in ("torch_bf16", "vllm_bi")
    }
    sub_threshold = {
        "torch_bf16": 3e-4,
        "vllm_bi": ceiling_estimate["vllm_bi"] / 3,
    }
    classified = []
    for r in rows:
        if r["drift"]["bitwise_equal"]:
            measured = "zero"
        elif r["drift"]["rel_l2"] < sub_threshold[r["backend"]]:
            measured = "sub"
        else:
            measured = "ceil"
        q = dict(r)
        q["measured_tier"] = measured
        q["classification_correct"] = measured == r["predicted_tier"]
        q["sub_threshold"] = sub_threshold[r["backend"]]
        classified.append(q)
    atomic_jsonl(RESULTS / "stage1b_classified.jsonl", classified)

    free = [r for r in classified if r["predicted_tier"] == "zero"]
    free_failures = [r for r in free if not r["drift"]["bitwise_equal"]]
    atomic_jsonl(RESULTS / "stage1b_free_failures.jsonl", free_failures)

    free_breakdown = []
    groups = defaultdict(list)
    for r in free:
        groups[(r["family"], r["backend"], r["shape"])].append(r)
    for (family, backend, shape), vals in sorted(groups.items()):
        passed = sum(v["drift"]["bitwise_equal"] for v in vals)
        free_breakdown.append({
            "family": family, "backend": backend, "shape": shape,
            "n": len(vals), "bitwise_equal": passed, "failures": len(vals) - passed,
            "max_rel_l2": max(v["drift"]["rel_l2"] for v in vals),
            "max_n_diff": max(v["drift"]["n_diff"] for v in vals),
        })

    saturation = [r for r in classified if r["predicted_tier"] == "ceil"]
    sat_success = [r for r in saturation
                   if not r["drift"]["bitwise_equal"] and r["measured_tier"] == "ceil"]
    confusion = {
        pred: {actual: sum(r["predicted_tier"] == pred and r["measured_tier"] == actual
                          for r in classified)
               for actual in ("zero", "sub", "ceil")}
        for pred in ("zero", "sub", "ceil")
    }

    spread = []
    spread_pass = True
    for backend in ("torch_bf16", "vllm_bi"):
        for shape in ("full", "decode1"):
            for layer in (0, 17, 35):
                vals = [r["drift"]["rel_l2"] for r in classified
                        if r["backend"] == backend and r["shape"] == shape
                        and r["layer"] == layer and r["measured_tier"] == "ceil"]
                if not vals:
                    item = {"backend": backend, "shape": shape, "layer": layer,
                            "n": 0, "p5": None, "p95": None, "p95_over_p5": None,
                            "pass": False, "reason": "no measured-ceil records"}
                else:
                    p5, p95 = pct(vals, 5), pct(vals, 95)
                    ratio = p95 / p5 if p5 > 0 else math.inf
                    item = {"backend": backend, "shape": shape, "layer": layer,
                            "n": len(vals), "p5": p5, "p95": p95,
                            "p95_over_p5": ratio, "pass": ratio <= 3}
                spread.append(item)
                spread_pass &= item["pass"]

    correct = sum(r["classification_correct"] for r in classified)
    actual_ceiling_median = {}
    for backend in ("torch_bf16", "vllm_bi"):
        vals = [r["drift"]["rel_l2"] for r in classified
                if r["backend"] == backend and r["measured_tier"] == "ceil"]
        actual_ceiling_median[backend] = statistics.median(vals) if vals else None
    ratio = actual_ceiling_median["torch_bf16"] / actual_ceiling_median["vllm_bi"]

    acceptance = {
        "amendment": "v1.1", "stage": "1b", "stage1b_complete": True,
        "raw_records": len(rows), "unique_measurement_keys": len(keys),
        "classification": {
            "ceiling_estimate_rule": "median rel_l2 of predicted-ceil arm within backend",
            "ceiling_estimate": ceiling_estimate, "sub_threshold": sub_threshold,
            "confusion_predicted_by_measured": confusion,
        },
        "S1b-1": {
            "name": "free tier hard criterion", "hard": True, "threshold": "100% bitwise_equal",
            "n": len(free), "bitwise_equal": len(free) - len(free_failures),
            "failures": len(free_failures), "rate": (len(free)-len(free_failures))/len(free),
            "pass": len(free_failures) == 0, "breakdown": free_breakdown,
            "failure_records": "results/stage1b_free_failures.jsonl",
        },
        "S1b-2": {
            "name": "predicted saturation", "threshold": ">=95% non-bitwise and measured ceil",
            "n": len(saturation), "success": len(sat_success),
            "rate": len(sat_success) / len(saturation),
            "non_bitwise_rate": sum(not r["drift"]["bitwise_equal"] for r in saturation)/len(saturation),
            "pass": len(sat_success) / len(saturation) >= .95,
        },
        "S1b-3": {"name": "ceiling universality", "threshold": "p95/p5 <= 3 in every backend x shape x layer",
                  "groups": spread, "pass": spread_pass},
        "S1b-4": {"name": "three-tier classification accuracy", "threshold": ">=85%",
                  "correct": correct, "n": len(classified), "accuracy": correct/len(classified),
                  "pass": correct/len(classified) >= .85},
        "S1b-5": {"name": "kernel ceiling difference (record only)",
                  "measured_ceiling_median": actual_ceiling_median,
                  "torch_over_vllm_ratio": ratio, "pass": None},
        "hard_stop_triggered": len(free_failures) > 0,
        "next_stage_authorized": False,
        "stage2b": "not run due S1b-1 hard failure",
        "stage3b": "not run due S1b-1 hard failure",
        "artifacts": {
            "raw": "results/stage1b_singlelayer.jsonl",
            "classified": "results/stage1b_classified.jsonl",
            "free_failures": "results/stage1b_free_failures.jsonl",
            "manifest": "results/stage1b_manifest.json",
        },
    }
    atomic_json(RESULTS / "acceptance_v11.json", acceptance)
    print(json.dumps({k: acceptance[k] for k in ("S1b-1", "S1b-2", "S1b-3", "S1b-4", "S1b-5")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
