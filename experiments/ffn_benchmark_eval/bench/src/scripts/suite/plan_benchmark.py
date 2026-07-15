#!/usr/bin/env python3
"""Generate a benchmark execution plan (no GPU, no job submission).

This script drives BenchmarkPlanRunner to produce:
  - execution_plan.json   (all deploy + client job descriptors)
  - resolved_run.json     (model + suite snapshot)

It does NOT submit Hope jobs.  Use submit_benchmark.py for that.

Usage
-----
  # From repo root:
  python src/scripts/suite/plan_benchmark.py \\
      --suite smoke \\
      --model-path /path/to/Qwen3-14B-Base \\
      --model-tag baseline \\
      --output-dir /path/to/output/qwen3-14b

  # With explicit backend:
  python src/scripts/suite/plan_benchmark.py \\
      --suite smoke \\
      --model-path /path/to/model \\
      --model-tag baseline \\
      --output-dir /path/to/output \\
      --backend remote_vllm_service

  # Filter to a single benchmark:
  python src/scripts/suite/plan_benchmark.py \\
      --suite smoke \\
      --model-path /path/to/model \\
      --model-tag baseline \\
      --output-dir /path/to/output \\
      --benchmark mmlu
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from src.eval.registry import EvalRun, ModelConfig, ExecutionConfig
from src.eval.suites import load_suite_by_name, BenchmarkSuiteSpec


def plan(
    suite_name: str,
    model_path: str,
    model_tag: str,
    output_dir: str,
    original_path: str | None = None,
    benchmark_filter: list | None = None,
    backend_mode: str = "remote_vllm_service",
    created_at: str | None = None,
) -> dict:
    """Build and write an execution plan.

    Args:
        suite_name:       Suite name from configs/eval_suites/benchmark/.
        model_path:       Path to the model directory.
        model_tag:        Stable tag for artifact naming (e.g. "baseline", "r025").
        output_dir:       Root output directory.
        original_path:    Path to original model (required for pruned models).
        benchmark_filter: If set, only include these benchmark_ids in the plan.
        backend_mode:     Execution mode ("remote_vllm_service"/"remote_hf_service").
        created_at:       Timestamp string.

    Returns:
        The execution plan dict.
    """
    suite = load_suite_by_name(
        kind="benchmark",
        name=suite_name,
        suites_root=str(REPO / "configs" / "eval_suites"),
    )

    # Apply benchmark filter if specified (filters on benchmark_id).
    if benchmark_filter:
        filtered = [b for b in suite.spec.benchmarks if b.id in benchmark_filter]
        if not filtered:
            raise ValueError(
                f"No benchmarks match filter={benchmark_filter!r}. "
                f"Available: {suite.spec.benchmark_ids}"
            )
        suite = type(suite)(
            kind=suite.kind,
            name=suite.name,
            spec=BenchmarkSuiteSpec(benchmarks=filtered),
            description=suite.description,
            source_path=suite.source_path,
        )

    model = ModelConfig(
        path=model_path,
        model_tag=model_tag,
        original_path=original_path,
    )
    execution = ExecutionConfig(mode=backend_mode)
    run = EvalRun(suite=suite, model=model, output_dir=output_dir, execution=execution)

    from src.eval.benchmark.compiler import BenchmarkPlanRunner
    runner = BenchmarkPlanRunner(
        run, created_at=created_at or datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    )
    return runner.execute()


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--suite", required=True, help="Suite name (e.g. smoke).")
    ap.add_argument("--model-path", required=True, help="Path to the model directory.")
    ap.add_argument("--model-tag", required=True, help="Stable tag (e.g. baseline, r025).")
    ap.add_argument("--output-dir", required=True, help="Root output directory.")
    ap.add_argument(
        "--original-path", default=None,
        help="Path to original model (required for pruned models).",
    )
    ap.add_argument(
        "--benchmark", nargs="+", default=None,
        help="Filter: only plan these benchmark IDs.",
    )
    ap.add_argument(
        "--backend",
        default="remote_vllm_service",
        choices=["remote_vllm_service", "remote_hf_service"],
        help="Execution backend mode.",
    )
    ap.add_argument(
        "--created-at",
        default=datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        help="Timestamp to embed in plan files.",
    )
    args = ap.parse_args()

    print(f"[plan] suite={args.suite}  model_tag={args.model_tag}")
    print(f"[plan] model_path={args.model_path}")
    print(f"[plan] output_dir={args.output_dir}")
    if args.benchmark:
        print(f"[plan] benchmark_filter={args.benchmark}")

    plan_dict = plan(
        suite_name=args.suite,
        model_path=args.model_path,
        model_tag=args.model_tag,
        output_dir=args.output_dir,
        original_path=args.original_path,
        benchmark_filter=args.benchmark,
        backend_mode=args.backend,
        created_at=args.created_at,
    )

    external = plan_dict.get("external_jobs", [])
    print()
    print(f"[plan] Protocol benches: {len(plan_dict['jobs'])} (deploy + client each)")
    print(f"[plan] External benches: {len(external)} (single GPU job each)")
    print(f"[plan] Total jobs:       {plan_dict['total_jobs']}")
    for job_pair in plan_dict["jobs"]:
        print(
            f"  [protocol] bench={job_pair['bench']!r}  "
            f"deploy={job_pair['deploy']['job_type']}  "
            f"client={job_pair['client']['job_type']}"
        )
    for ext in external:
        print(
            f"  [external] bench={ext['bench']!r}  "
            f"framework={ext['framework']!r}  dataset={ext['dataset']!r}  "
            f"metric={ext['metric']!r}"
        )


if __name__ == "__main__":
    main()
