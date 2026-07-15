#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compute mean±std summary for one benchmark across N run results.

Reads  BENCH_DIR/run_00/result.json … run_{N-1:02d}/result.json
Writes BENCH_DIR/summary.json

Called at the end of run_suite_src.sh / run_suite_external.sh after all
N_RUNS have completed.  Safe to re-run: overwrites summary.json.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--bench-dir",
        required=True,
        help="Directory containing run_00/, run_01/, … subdirs with result.json",
    )
    args = ap.parse_args()

    bench_dir = Path(args.bench_dir)
    if not bench_dir.is_dir():
        print(f"[compute_summary] ERROR: bench-dir not found: {bench_dir}", file=sys.stderr)
        sys.exit(1)

    # Collect run_XX directories in sorted order
    run_dirs = sorted(
        [d for d in bench_dir.iterdir() if d.is_dir() and re.match(r"run_\d+$", d.name)],
        key=lambda d: d.name,
    )
    if not run_dirs:
        print(f"[compute_summary] ERROR: no run_XX dirs found in {bench_dir}", file=sys.stderr)
        sys.exit(1)

    runs: list[dict] = []
    missing: list[str] = []
    for run_dir in run_dirs:
        result_path = run_dir / "result.json"
        if not result_path.exists():
            missing.append(str(result_path))
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        runs.append(
            {
                "run": run_dir.name,
                "accuracy": float(result["accuracy"]),
                "correct": result.get("correct"),
                "total": result.get("total"),
                "errors": result.get("errors", 0),
                "metrics": result.get("metrics"),
            }
        )

    if missing:
        print(f"[compute_summary] WARNING: missing result.json in: {missing}")

    if not runs:
        print("[compute_summary] ERROR: no result.json files found", file=sys.stderr)
        sys.exit(1)

    accuracies = [r["accuracy"] for r in runs]
    n = len(accuracies)
    mean = sum(accuracies) / n
    variance = sum((a - mean) ** 2 for a in accuracies) / max(n - 1, 1)
    std = math.sqrt(variance)

    first_result = json.loads(
        (run_dirs[0] / "result.json").read_text(encoding="utf-8")
    )
    bench_id = first_result.get("bench_id", bench_dir.name)

    summary = {
        "bench_id": bench_id,
        "model": bench_dir.parent.name,
        "n_runs": n,
        "runs": runs,
        "accuracy_mean": round(mean, 6),
        "accuracy_std": round(std, 6),
        "accuracy_min": round(min(accuracies), 6),
        "accuracy_max": round(max(accuracies), 6),
        "accuracy_std_pct": round(std * 100, 4),
    }

    out = bench_dir / "summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"[compute_summary] {bench_id:15s}  "
        f"mean={mean:.4f}  std={std:.6f}  ({n} runs)  "
        f"range=[{min(accuracies):.4f}, {max(accuracies):.4f}]"
        f"  → {out}"
    )


if __name__ == "__main__":
    main()
