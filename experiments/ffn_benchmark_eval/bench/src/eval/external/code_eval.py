"""evalplus code-benchmark adapter: job descriptor + result normalization.

Two responsibilities, kept in one small module because there is only one
external framework today (see DESIGN_external_eval_integration.md — we
deliberately do NOT build a framework registry yet):

1. ``build_external_job_descriptor`` (plan side, no GPU):
   Turn an external ``BenchmarkTaskSpec`` (runner=external, framework=evalplus)
   into a self-contained job descriptor that ``submit_benchmark.py`` renders
   into a single GPU Hope job.

2. ``normalize`` (worker side, on the GPU node, invoked as
   ``python -m src.eval.external.code_eval normalize`` by run_code_eval.sh):
   Read evalplus's ``*.eval_results.json`` and write a unified result JSON that
   matches the schema protocol benches produce (``accuracy`` + identifying
   fields), so the H1 report aggregates code and non-code benches uniformly.

Decision: pass@1 IS accuracy (user-confirmed; semantically equivalent here).
The chosen METRIC (default ``base_pass_at_1``) becomes ``accuracy``; both
base/plus pass@1 are kept under ``metrics`` so nothing is lost.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional


# evalplus dataset key (--dataset) per supported bench. Only these two exist.
SUPPORTED_DATASETS = {"humaneval", "mbpp"}
VALID_METRICS = {"base_pass_at_1", "plus_pass_at_1"}

# Worker script shipped with this package (absolute, so staging stays minimal).
WORKER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_code_eval.sh")


# ── Plan side: external job descriptor ────────────────────────────────────────

@dataclass
class ExternalJobDescriptor:
    """Descriptor for one external (single-GPU) benchmark job.

    Unlike protocol benches (deploy + client pair), an external bench is a
    single in-process GPU job: the framework loads vLLM itself and relies on
    vLLM continuous batching to saturate the GPU — no client, no concurrency.
    """

    bench: str              # unified bench id, e.g. "humaneval_plus"
    framework: str          # "evalplus"
    dataset: str            # evalplus --dataset key: "humaneval" | "mbpp"
    metric: str             # which pass@1 → unified accuracy
    parent_benchmark: str   # for report grouping, e.g. "humaneval"
    worker_script: str      # absolute path to run_code_eval.sh
    model_load_args: Dict[str, Any]
    resources: Dict[str, Any]
    result_path: str        # unified-schema result JSON (== protocol task result path)
    output_dir: str         # raw evalplus artifacts + logs
    job_type: str = "external"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_type": self.job_type,
            "bench": self.bench,
            "framework": self.framework,
            "dataset": self.dataset,
            "metric": self.metric,
            "parent_benchmark": self.parent_benchmark,
            "worker_script": self.worker_script,
            "model_load_args": self.model_load_args,
            "resources": self.resources,
            "result_path": self.result_path,
            "output_dir": self.output_dir,
        }


# Default resources for an external bench: one 8-GPU job (8×A100 with big KV
# cache so vLLM continuous batching keeps the GPU saturated — see design §0).
DEFAULT_EXTERNAL_RESOURCES = {"replicas": 1, "gpus_per_job": 2}


def validate_external_task(task: Any) -> None:
    """Fail-fast validation of an external BenchmarkTaskSpec (plan time)."""
    if task.framework != "evalplus":
        raise ValueError(
            f"bench {task.id!r}: unsupported external framework {task.framework!r} "
            f"(only 'evalplus' is supported)"
        )
    if task.dataset not in SUPPORTED_DATASETS:
        raise ValueError(
            f"bench {task.id!r}: evalplus dataset must be one of "
            f"{sorted(SUPPORTED_DATASETS)}, got {task.dataset!r}"
        )
    if task.metric not in VALID_METRICS:
        raise ValueError(
            f"bench {task.id!r}: metric must be one of {sorted(VALID_METRICS)}, "
            f"got {task.metric!r}"
        )


def build_external_job_descriptor(
    task: Any,                      # suites.BenchmarkTaskSpec with runner=external
    model_load_args: Dict[str, Any],
    result_path: str,
    output_dir: str,
    resources: Optional[Dict[str, Any]] = None,
) -> ExternalJobDescriptor:
    """Build an ExternalJobDescriptor from an external benchmark task spec.

    ``parent_benchmark`` is derived from the dataset (humaneval / mbpp) so the
    report can group HumanEval+/MBPP+ under their logical benchmark.
    """
    validate_external_task(task)
    return ExternalJobDescriptor(
        bench=task.id,
        framework=task.framework,
        dataset=task.dataset,
        metric=task.metric,
        parent_benchmark=task.dataset,  # "humaneval" / "mbpp"
        worker_script=WORKER_SCRIPT,
        model_load_args=model_load_args,
        resources={**DEFAULT_EXTERNAL_RESOURCES, **(resources or {})},
        result_path=result_path,
        output_dir=output_dir,
    )


# ── Worker side: result normalization ─────────────────────────────────────────

def _find_eval_results(samples_dir: str) -> str:
    """Locate evalplus's *.eval_results.json under samples_dir."""
    hits = glob.glob(os.path.join(samples_dir, "**", "*eval_results.json"), recursive=True)
    if not hits:
        hits = glob.glob(os.path.join(samples_dir, "*.eval_results.json"))
    if not hits:
        listing = "\n".join(f"  {p}" for p in glob.glob(os.path.join(samples_dir, "**", "*"), recursive=True))
        raise FileNotFoundError(
            f"No *eval_results.json found in {samples_dir}\nContents:\n{listing}"
        )
    return hits[0]


def normalize_eval_results(
    eval_results_path: str,
    bench_id: str,
    dataset: str,
    metric: str,
    model_path: str,
) -> Dict[str, Any]:
    """Read evalplus eval_results.json → unified benchmark summary dict.

    Unified schema (mirrors the protocol client_runner summary so the report
    treats code and non-code benches the same):
      - ``accuracy``  = the chosen metric's pass@1 (user decision: pass@1 IS accuracy)
      - ``total``     = number of tasks evaluated
      - ``metrics``   = {base_pass_at_1, plus_pass_at_1} (full, lossless)
      - ``runner``/``framework`` so a reader can tell it apart from MC accuracy
    """
    if metric not in VALID_METRICS:
        raise ValueError(f"METRIC must be one of {sorted(VALID_METRICS)}, got {metric!r}")

    with open(eval_results_path, encoding="utf-8") as f:
        results = json.load(f)

    pass_at_k = results.get("pass_at_k", {})
    base_p1 = pass_at_k.get("base", {}).get("pass@1")
    plus_p1 = pass_at_k.get("plus", {}).get("pass@1")
    if base_p1 is None:
        raise ValueError(f"eval_results.json missing pass_at_k.base.pass@1: {eval_results_path}")

    metrics = {"base_pass_at_1": base_p1, "plus_pass_at_1": plus_p1}
    accuracy = metrics[metric]
    if accuracy is None:
        raise ValueError(f"chosen metric {metric!r} is null in {eval_results_path}")

    total = len(results.get("eval", {})) or None

    return {
        "bench_id": bench_id,
        "parent_benchmark": dataset,
        "runner": "external",
        "framework": "evalplus",
        "dataset": dataset,
        "model": model_path,
        "metric": metric,
        "accuracy": accuracy,
        "metrics": metrics,
        "total": total,
        "source_eval_results": eval_results_path,
    }


def _normalize_from_env() -> int:
    """Worker entry: read env, locate evalplus result, write unified result JSON."""
    dataset = os.environ["DATASET"]
    output_dir = os.environ["OUTPUT_DIR"]
    result_path = os.environ["RESULT_PATH"]
    bench_id = os.environ["BENCH_ID"]
    metric = os.environ.get("METRIC", "base_pass_at_1")
    model_path = os.environ.get("MODEL_PATH", "")

    samples_dir = os.path.join(output_dir, dataset)
    eval_results_path = _find_eval_results(samples_dir)
    print(f"[code_eval] eval_results: {eval_results_path}")

    summary = normalize_eval_results(
        eval_results_path=eval_results_path,
        bench_id=bench_id,
        dataset=dataset,
        metric=metric,
        model_path=model_path,
    )

    if summary["accuracy"] < 0:
        raise ValueError(f"accuracy={summary['accuracy']} (negative — eval likely failed)")

    os.makedirs(os.path.dirname(os.path.abspath(result_path)), exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(
        f"[code_eval] {bench_id}: accuracy({metric})={summary['accuracy']:.4f}  "
        f"base={summary['metrics']['base_pass_at_1']}  plus={summary['metrics']['plus_pass_at_1']}"
    )
    print(f"[code_eval] unified result → {result_path}")
    return 0


def main(argv: Optional[list] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] != "normalize":
        print("usage: python -m src.eval.external.code_eval normalize", file=sys.stderr)
        return 2
    return _normalize_from_env()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        raise
