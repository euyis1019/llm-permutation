"""Eval Suite CLI entry point.

Usage
-----
Run a ppl suite (requires GPU — use inside a Hope job)::

    python -m src.eval.cli run \\
        --suite configs/eval_suites/ppl/smoke.yaml \\
        --model-path /path/to/model \\
        --model-tag baseline \\
        --output-dir output/olmo3-32b/d1_math \\
        --created-at 2026-06-10T14:00:00

    # Or pass a run YAML:
    python -m src.eval.cli run --run-config path/to/run.yaml

Generate a benchmark execution plan (no GPU required)::

    python -m src.eval.cli plan \\
        --suite configs/eval_suites/benchmark/smoke.yaml \\
        --model-path /path/to/model \\
        --model-tag baseline \\
        --output-dir output/olmo3-32b/d1_math \\
        [--original-path /path/to/original]  # required for pruned models

    # Or pass a run YAML:
    python -m src.eval.cli plan --run-config path/to/run.yaml

Validate a suite YAML without running::

    python -m src.eval.cli validate --suite configs/eval_suites/ppl/smoke.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add model + output args shared by 'run' and 'plan' subcommands."""
    parser.add_argument(
        "--run-config",
        metavar="YAML",
        help="Run config YAML file (alternative to individual flags).",
    )
    parser.add_argument(
        "--suite",
        metavar="YAML",
        help="Suite YAML file path (used when --run-config is not given).",
    )
    parser.add_argument(
        "--model-path",
        metavar="PATH",
        help="Path to the model directory.",
    )
    parser.add_argument(
        "--model-tag",
        metavar="TAG",
        help="Stable model tag for artifact naming (e.g. 'baseline', 'r025').",
    )
    parser.add_argument(
        "--original-path",
        metavar="PATH",
        default=None,
        help="Path to the original (non-pruned) model. Required for pruned benchmark models.",
    )
    parser.add_argument(
        "--output-dir",
        metavar="DIR",
        help="Root output directory (e.g. output/olmo3-32b/d1_math).",
    )
    parser.add_argument(
        "--created-at",
        metavar="TIMESTAMP",
        default=None,
        help="Timestamp string for manifest/result files. src/ does not call time.time().",
    )


def _build_run_from_args(args: argparse.Namespace):
    """Build an EvalRun from CLI args (either --run-config or individual flags)."""
    from .registry import EvalRun

    if args.run_config:
        return EvalRun.from_yaml(args.run_config)

    # Build from individual flags
    if not args.suite:
        print("ERROR: --suite is required when --run-config is not given", file=sys.stderr)
        sys.exit(1)
    if not args.model_path:
        print("ERROR: --model-path is required", file=sys.stderr)
        sys.exit(1)
    if not args.model_tag:
        print("ERROR: --model-tag is required", file=sys.stderr)
        sys.exit(1)
    if not args.output_dir:
        print("ERROR: --output-dir is required", file=sys.stderr)
        sys.exit(1)

    from .registry import build_run
    return build_run(
        suite_path=args.suite,
        model_path=args.model_path,
        model_tag=args.model_tag,
        output_dir=args.output_dir,
        original_path=args.original_path,
    )


# ── Subcommand: run ───────────────────────────────────────────────────────────

def cmd_run(args: argparse.Namespace) -> int:
    """Execute a ppl run (requires GPU)."""
    run = _build_run_from_args(args)

    if run.suite.kind != "ppl":
        print(
            f"ERROR: 'run' subcommand only supports ppl suites. "
            f"Got kind={run.suite.kind!r}. Use 'plan' for benchmark suites.",
            file=sys.stderr,
        )
        return 1

    from .registry import create_runner
    from .ppl.runner import PPLRunner

    runner = create_runner(run)
    assert isinstance(runner, PPLRunner)

    # Inject CLI overrides
    if args.created_at:
        runner.created_at = args.created_at
    if args.device:
        runner.device = args.device
    if args.batch_size:
        runner.batch_size = args.batch_size

    print(f"[eval run] suite={run.suite.name!r} kind=ppl model_tag={run.model.model_tag!r}")
    print(f"[eval run] output_dir={run.output_dir!r}")

    summary = runner.execute()

    print("\n[eval run] Done.")
    ppl_by_ds = summary.get("ppl_by_dataset", {})
    for ds_id, ppl in ppl_by_ds.items():
        print(f"  {ds_id}: ppl={ppl:.4f}")

    failed = summary.get("failed_datasets", [])
    if failed:
        print(f"\nWARNING: {len(failed)} failed dataset(s): {failed}")
        return 1
    return 0


# ── Subcommand: plan ──────────────────────────────────────────────────────────

def cmd_plan(args: argparse.Namespace) -> int:
    """Compile a benchmark execution plan (no GPU required)."""
    run = _build_run_from_args(args)

    if run.suite.kind != "benchmark":
        print(
            f"ERROR: 'plan' subcommand only supports benchmark suites. "
            f"Got kind={run.suite.kind!r}. Use 'run' for ppl suites.",
            file=sys.stderr,
        )
        return 1

    from .registry import create_runner
    from .benchmark.compiler import BenchmarkPlanRunner

    runner = create_runner(run)
    assert isinstance(runner, BenchmarkPlanRunner)

    if args.created_at:
        runner.created_at = args.created_at

    print(f"[eval plan] suite={run.suite.name!r} kind=benchmark model_tag={run.model.model_tag!r}")
    print(f"[eval plan] output_dir={run.output_dir!r}")

    plan = runner.execute()

    print("\n[eval plan] Execution plan compiled.")
    print(f"  benches: {len(plan['jobs'])}")
    print(f"  total jobs: {plan['total_jobs']} (deploy + client per bench)")
    for job in plan["jobs"]:
        bench = job["bench"]
        deploy_path = job["deploy"]["artifact_path"]
        client_path = job["client"]["artifact_path"]
        print(f"  {bench}:")
        print(f"    deploy → {deploy_path}")
        print(f"    client → {client_path}")

    return 0


# ── Subcommand: validate ──────────────────────────────────────────────────────

def cmd_validate(args: argparse.Namespace) -> int:
    """Validate a suite YAML file."""
    if not args.suite:
        print("ERROR: --suite is required for validate", file=sys.stderr)
        return 1

    from .suites import load_suite
    try:
        suite = load_suite(args.suite)
        print(f"✓ Suite valid: kind={suite.kind!r} name={suite.name!r}")
        bench_ids = suite.bench_ids
        print(f"  {len(bench_ids)} bench(es): {bench_ids}")
        return 0
    except Exception as exc:
        print(f"✗ Suite invalid: {exc}", file=sys.stderr)
        return 1


# ── Main ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.eval.cli",
        description="Eval Suite CLI: run ppl evals or compile benchmark plans.",
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    # run
    p_run = sub.add_parser("run", help="Execute a ppl eval run (requires GPU).")
    _add_common_args(p_run)
    p_run.add_argument("--device", default=None, help="Torch device (e.g. 'cuda:0').")
    p_run.add_argument("--dtype", default="bfloat16", help="Model dtype.")
    p_run.add_argument("--batch-size", type=int, default=8, help="PPL eval batch size.")

    # plan
    p_plan = sub.add_parser("plan", help="Compile a benchmark execution plan (no GPU).")
    _add_common_args(p_plan)

    # validate
    p_val = sub.add_parser("validate", help="Validate a suite YAML file.")
    p_val.add_argument("--suite", metavar="YAML", required=True, help="Suite YAML file path.")

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.subcommand == "run":
        return cmd_run(args)
    elif args.subcommand == "plan":
        return cmd_plan(args)
    elif args.subcommand == "validate":
        return cmd_validate(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
