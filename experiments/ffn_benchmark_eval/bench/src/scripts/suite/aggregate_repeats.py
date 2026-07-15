#!/usr/bin/env python3
"""Aggregate repeated benchmark runs into mean±std summary.

Usage
-----
  python src/scripts/suite/aggregate_repeats.py \
      --output-dir experiments/.../output \
      --suite code \
      --model-tag qwen3-14b-base \
      --repeat 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))


def aggregate_bench(tasks_dir: Path, bench: str, repeat: int) -> dict | None:
    """Read run_01..run_N result.json files and compute statistics."""
    results: List[dict] = []
    missing: List[int] = []

    for i in range(1, repeat + 1):
        result_path = tasks_dir / bench / f"run_{i:02d}" / "result.json"
        if not result_path.exists():
            missing.append(i)
            continue
        results.append(json.loads(result_path.read_text()))

    if not results:
        print(f"  [SKIP] {bench}: no result.json found (checked run_01..run_{repeat:02d})")
        return None

    if missing:
        print(f"  [WARN] {bench}: missing runs {missing}")

    metric = results[0]["metric"]
    accuracies = [r["accuracy"] for r in results]
    base_scores = [r["metrics"]["base_pass_at_1"] for r in results]
    plus_scores = [r["metrics"]["plus_pass_at_1"] for r in results]

    def stats(values):
        n = len(values)
        mean = sum(values) / n
        var = sum((x - mean) ** 2 for x in values) / n
        std = var ** 0.5
        return {"mean": mean, "std": std, "min": min(values), "max": max(values)}

    summary = {
        "bench": bench,
        "repeat": len(results),
        "expected_repeat": repeat,
        "metric": metric,
        "accuracy": stats(accuracies),
        "base_pass_at_1": stats(base_scores),
        "plus_pass_at_1": stats(plus_scores),
        "raw_accuracies": accuracies,
        "all_identical": len(set(accuracies)) == 1,
    }

    if missing:
        summary["missing_runs"] = missing

    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", required=True, help="Root output directory.")
    ap.add_argument("--suite", required=True, help="Suite name (e.g. code).")
    ap.add_argument("--model-tag", required=True, help="Model tag (e.g. qwen3-14b-base).")
    ap.add_argument("--repeat", type=int, required=True, help="Number of repeats.")
    args = ap.parse_args()

    tasks_dir = Path(args.output_dir) / "eval" / "benchmark" / args.suite / args.model_tag / "tasks"
    if not tasks_dir.exists():
        print(f"ERROR: tasks directory not found: {tasks_dir}", file=sys.stderr)
        sys.exit(1)

    bench_dirs = sorted(d.name for d in tasks_dir.iterdir() if d.is_dir())
    print(f"[aggregate] suite={args.suite}  model_tag={args.model_tag}  repeat={args.repeat}")
    print(f"[aggregate] tasks_dir={tasks_dir}")
    print(f"[aggregate] benches found: {bench_dirs}\n")

    all_summaries = []
    for bench in bench_dirs:
        summary = aggregate_bench(tasks_dir, bench, args.repeat)
        if summary is None:
            continue
        all_summaries.append(summary)

        out_path = tasks_dir / bench / "repeat_summary.json"
        out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"  [OK] {bench}: wrote {out_path}")

    # Print human-readable table
    if all_summaries:
        print(f"\n{'='*72}")
        print(f"  {'Benchmark':<18} {'Metric':<16} {'Mean':>8} {'±Std':>8} {'Min':>8} {'Max':>8}  Identical?")
        print(f"  {'-'*18} {'-'*16} {'-'*8} {'-'*8} {'-'*8} {'-'*8}  {'-'*10}")
        for s in all_summaries:
            acc = s["accuracy"]
            print(
                f"  {s['bench']:<18} {s['metric']:<16} "
                f"{acc['mean']:>8.4f} {acc['std']:>8.4f} {acc['min']:>8.4f} {acc['max']:>8.4f}  "
                f"{'YES' if s['all_identical'] else 'NO'}"
            )
        print(f"{'='*72}")


if __name__ == "__main__":
    main()
