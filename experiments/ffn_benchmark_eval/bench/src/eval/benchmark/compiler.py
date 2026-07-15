"""Benchmark execution plan compiler.

Design (07_benchmark_eval_integration_proposal.md §4 + S2 design notes):
- A suite names benchmark_ids only.
- For each benchmark_id, the compiler loads
  ``datasets/benchmark/normalized/{benchmark_id}/benchmark_meta.json`` and
  resolves each BenchData's protocol via behavior_catalog.get_protocol().
- Each BenchData compiles to exactly 2 Hope jobs: 1 deploy + 1 client.
- Output: execution_plan.json + resolved_run.json.

Fail-fast at plan time (no GPU needed) on:
  1. a suite benchmark_id with no normalized/{id}/benchmark_meta.json
  2. a BenchData whose data_path file does not exist
  3. a prompt_builder_id not in PROMPT_BUILDER_CATALOG
  4. a scorer_id not in SCORER_CATALOG

The compiler does NOT submit Hope jobs; it produces a plan dict.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.data.layout import ArtifactLayout

from ..registry import EvalRun, EvalRunner
from ..suites import BenchmarkSuiteSpec, BenchmarkTaskSpec
from .behavior_catalog import get_prompt_builder, get_protocol, get_scorer
from .loader import ResolvedModel, resolve_model
from .models import BenchData, BenchmarkMeta, BenchmarkProtocol


# Where normalised benchmark directories live (one per benchmark_id).
DEFAULT_NORMALIZED_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "datasets", "benchmark", "normalized",
)


# ── Job descriptors ───────────────────────────────────────────────────────────

@dataclass
class DeployJobDescriptor:
    """Descriptor for a deploy Hope job."""

    bench: str
    model_load_args: Dict[str, Any]
    resources: Dict[str, Any]
    artifact_path: str
    job_type: str = "deploy"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_type": self.job_type,
            "bench": self.bench,
            "model_load_args": self.model_load_args,
            "resources": self.resources,
            "artifact_path": self.artifact_path,
        }


@dataclass
class ClientJobDescriptor:
    """Descriptor for a client Hope job.

    Carries the fully-resolved protocol so the client job is self-contained:
    the runner needs no further catalog lookup beyond the prompt_builder /
    scorer ids it names.
    """

    bench: str
    parent_benchmark: str
    data_path: str
    protocol: Dict[str, Any]
    resources: Dict[str, Any]
    artifact_path: str
    job_type: str = "client"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_type": self.job_type,
            "bench": self.bench,
            "parent_benchmark": self.parent_benchmark,
            "data_path": self.data_path,
            "protocol": self.protocol,
            "resources": self.resources,
            "artifact_path": self.artifact_path,
        }


@dataclass
class BenchJobPair:
    """A deploy + client job pair for one BenchData."""

    bench: str
    deploy: DeployJobDescriptor
    client: ClientJobDescriptor

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bench": self.bench,
            "deploy": self.deploy.to_dict(),
            "client": self.client.to_dict(),
        }


@dataclass
class ExecutionPlan:
    """Full execution plan for a benchmark run.

    ``jobs`` are protocol benches (deploy + client pair each).  ``external_jobs``
    are external-framework benches (one single-GPU job each, e.g. evalplus);
    they carry their own descriptor dicts (see src/eval/external/code_eval.py).
    """

    kind: str
    suite: str
    model_tag: str
    model: Dict[str, Any]
    jobs: List[BenchJobPair]
    external_jobs: List[Any] = None  # List[ExternalJobDescriptor]
    created_at: Optional[str] = None

    def __post_init__(self) -> None:
        if self.external_jobs is None:
            self.external_jobs = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "suite": self.suite,
            "model_tag": self.model_tag,
            "model": self.model,
            "jobs": [j.to_dict() for j in self.jobs],
            "external_jobs": [e.to_dict() for e in self.external_jobs],
            # protocol: deploy+client (×2); external: single job (×1)
            "total_jobs": len(self.jobs) * 2 + len(self.external_jobs),
            "created_at": self.created_at,
        }


# ── Default resource profiles ─────────────────────────────────────────────────

_DEFAULT_DEPLOY_RESOURCES = {"replicas": 1, "gpus_per_job": 4, "port": 8080}
_DEFAULT_CLIENT_RESOURCES = {"replicas": 1, "cpus_per_job": 16, "concurrency": 8, "max_retries": 3}


def _extract_resources(execution_params: dict, role: str) -> dict:
    defaults = _DEFAULT_DEPLOY_RESOURCES if role == "deploy" else _DEFAULT_CLIENT_RESOURCES
    overrides = execution_params.get(role, {}) or {}
    return {**defaults, **overrides}


# ── Sample-spec truncation ────────────────────────────────────────────────────

def _apply_sample_spec(
    bench_data: BenchData,
    task_spec: BenchmarkTaskSpec,
) -> BenchData:
    """Truncate *bench_data* to the row count requested by *task_spec*.

    Rules (nrows takes priority over ratio; both None → full dataset):
    - ``task_spec.nrows`` is set → ``target = min(nrows, total_rows)``
    - ``task_spec.ratio`` is set → ``target = ceil(total_rows * ratio)``
    - both None               → return *bench_data* unchanged

    The truncated JSONL is written next to the original file with a ``__{n}rows``
    suffix so it never overwrites the full dataset.  If the file already exists
    (e.g. a previous plan run) it is reused without re-writing.

    Returns the original *bench_data* when no truncation is needed, or a
    shallow copy with ``data_path`` and ``total_rows`` updated otherwise.
    """
    import copy
    import math

    total = bench_data.total_rows

    # Determine target row count
    if task_spec.nrows is not None:
        target = min(task_spec.nrows, total)
    elif task_spec.ratio is not None:
        target = min(math.ceil(total * task_spec.ratio), total)
    else:
        return bench_data  # no truncation requested

    if target >= total:
        return bench_data  # requested at least as many rows as available

    # Build truncated JSONL path next to the original
    orig_path = bench_data.data_path
    trunc_path = orig_path.replace(".jsonl", f"__{target}rows.jsonl")

    if not os.path.isfile(trunc_path):
        rows_written = 0
        with open(orig_path, encoding="utf-8") as fin, \
             open(trunc_path, "w", encoding="utf-8") as fout:
            for line in fin:
                if rows_written >= target:
                    break
                line = line.strip()
                if line:
                    fout.write(line + "\n")
                    rows_written += 1
        print(
            f"[compiler] truncated {bench_data.bench_id!r}: "
            f"{total} → {rows_written} rows  ({trunc_path})"
        )

    # Return a shallow copy with the updated path/count
    bd2 = copy.copy(bench_data)
    bd2.data_path = trunc_path
    bd2.total_rows = target
    return bd2


# ── Meta loading + protocol resolution (fail-fast) ────────────────────────────

def _load_benchmark_meta(benchmark_id: str, normalized_root: str) -> BenchmarkMeta:
    """Load benchmark_meta.json for *benchmark_id*, failing fast if absent."""
    meta_path = os.path.join(normalized_root, benchmark_id, "benchmark_meta.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(
            f"Suite references benchmark_id={benchmark_id!r} but no "
            f"benchmark_meta.json found at {meta_path!r}. "
            f"Run prepare_bench_data.py to generate it."
        )
    return BenchmarkMeta.load(meta_path)


def _resolve_and_validate_protocol(
    benchmark_id: str,
    bench_data: BenchData,
    meta_protocol: BenchmarkProtocol,
) -> BenchmarkProtocol:
    """Resolve the protocol for *bench_data* and validate the whole contract.

    Priority for fewshot_examples (highest → lowest):
      1. bench_data.fewshot_examples  — subject-specific examples materialised
                                        at prepare time (mmlu / ceval / cmmlu /
                                        mmlu_redux / bbh).
      2. meta_protocol.fewshot_examples — shared examples at the benchmark level
                                          (gsm8k / mmlu_pro / math500).
      3. empty list                   — no few-shot (should not happen for any
                                        benchmark with fewshot > 0).

    Behavior fields (prompt_builder_id, scorer_id, stop_tokens,
    generation_kwargs) always come from the catalog default, optionally
    overridden by bench_data.protocol_override.  This keeps the catalog as the
    single source of truth for behavior while letting the meta carry data assets.
    """
    import dataclasses

    # Step 1: resolve behavior fields from catalog + optional per-bench override.
    protocol = get_protocol(benchmark_id, bench_data.protocol_override)

    # Step 2: inject fewshot_examples from the materialised assets in the meta.
    # bench_data layer (subject-specific) takes priority over protocol layer (shared).
    if bench_data.fewshot_examples:
        protocol = dataclasses.replace(
            protocol, fewshot_examples=bench_data.fewshot_examples
        )
    elif meta_protocol.fewshot_examples:
        protocol = dataclasses.replace(
            protocol, fewshot_examples=meta_protocol.fewshot_examples
        )
    # else: fewshot_examples stays [] — catalog default

    # Step 3: fail-fast validation.
    get_prompt_builder(protocol.prompt_builder_id)   # unknown id → KeyError
    get_scorer(protocol.scorer_id)                   # unknown id → KeyError
    if not bench_data.data_path_exists():
        raise FileNotFoundError(
            f"BenchData {bench_data.bench_id!r} (benchmark {benchmark_id!r}) "
            f"data_path does not exist: {bench_data.data_path!r}"
        )

    return protocol


# ── Compiler ──────────────────────────────────────────────────────────────────

def compile_plan(
    run: EvalRun,
    resolved_model: ResolvedModel,
    created_at: Optional[str] = None,
    normalized_root: str = DEFAULT_NORMALIZED_ROOT,
) -> ExecutionPlan:
    """Compile a benchmark EvalRun into an ExecutionPlan.

    Each BenchData under each benchmark in the suite becomes one BenchJobPair
    (1 deploy + 1 client).
    """
    if run.suite.kind != "benchmark":
        raise ValueError(
            f"compile_plan only handles 'benchmark' suites, got {run.suite.kind!r}"
        )

    layout = ArtifactLayout(run.output_dir)
    suite = run.suite
    spec: BenchmarkSuiteSpec = suite.spec
    model_tag = run.model.model_tag
    exec_params = run.execution.params

    deploy_resources = _extract_resources(exec_params, "deploy")
    client_resources = _extract_resources(exec_params, "client")

    jobs: List[BenchJobPair] = []
    external_jobs: List[Any] = []
    for task_spec in spec.benchmarks:
        # External benches (evalplus etc.) are black boxes with no protocol:
        # build a single-GPU job descriptor and skip the meta/protocol path.
        if task_spec.is_external:
            external_jobs.append(
                _compile_external_job(task_spec, resolved_model, layout, suite.name, model_tag)
            )
            continue

        benchmark_id = task_spec.id
        meta = _load_benchmark_meta(benchmark_id, normalized_root)
        if not meta.bench_data:
            raise ValueError(
                f"benchmark_meta.json for {benchmark_id!r} has no bench_data entries"
            )
        for bench_data in meta.bench_data:
            # Apply nrows/ratio truncation from the suite spec (no-op if both None)
            bench_data = _apply_sample_spec(bench_data, task_spec)
            # Pass meta.protocol so fewshot_examples can be injected from the
            # materialised assets even when bench_data.fewshot_examples is empty.
            protocol = _resolve_and_validate_protocol(
                benchmark_id, bench_data, meta.protocol
            )
            bench = bench_data.bench_id
            deploy = DeployJobDescriptor(
                bench=bench,
                model_load_args=resolved_model.load_args,
                resources=deploy_resources,
                artifact_path=layout.benchmark_deploy_job_path(suite.name, model_tag, bench),
            )
            client = ClientJobDescriptor(
                bench=bench,
                parent_benchmark=bench_data.parent_benchmark,
                data_path=bench_data.data_path,
                protocol=protocol.to_dict(),
                resources=client_resources,
                artifact_path=layout.benchmark_client_job_path(suite.name, model_tag, bench),
            )
            jobs.append(BenchJobPair(bench=bench, deploy=deploy, client=client))

    return ExecutionPlan(
        kind="benchmark",
        suite=suite.name,
        model_tag=model_tag,
        model=resolved_model.load_args,
        jobs=jobs,
        external_jobs=external_jobs,
        created_at=created_at,
    )


def _compile_external_job(
    task_spec: BenchmarkTaskSpec,
    resolved_model: ResolvedModel,
    layout: ArtifactLayout,
    suite_name: str,
    model_tag: str,
) -> Any:
    """Build an ExternalJobDescriptor for an external (framework) bench.

    The unified result lands at the SAME path a protocol bench would use
    (benchmark_task_result_path), so report aggregation is identical.  Raw
    framework artifacts + logs go under the bench's task dir.
    """
    from src.eval.external.code_eval import build_external_job_descriptor

    bench = task_spec.id
    result_path = layout.benchmark_task_result_path(suite_name, model_tag, bench)
    # Raw evalplus artifacts + logs alongside the result (tasks/{bench}/external/).
    output_dir = os.path.join(
        os.path.dirname(result_path), "external"
    )
    return build_external_job_descriptor(
        task=task_spec,
        model_load_args=resolved_model.load_args,
        result_path=result_path,
        output_dir=output_dir,
    )


# ── BenchmarkPlanRunner ───────────────────────────────────────────────────────

class BenchmarkPlanRunner(EvalRunner):
    """EvalRunner for benchmark family: validates model and compiles the plan.

    Does NOT submit Hope jobs.  It:
    1. Resolves + validates the model (pruned → original_path required).
    2. Compiles the execution plan (fails fast on missing data/catalog ids).
    3. Writes resolved_run.json and execution_plan.json to disk.
    """

    def __init__(
        self,
        run: EvalRun,
        created_at: Optional[str] = None,
        normalized_root: str = DEFAULT_NORMALIZED_ROOT,
    ) -> None:
        super().__init__(run)
        if run.suite.kind != "benchmark":
            raise ValueError(
                f"BenchmarkPlanRunner only handles 'benchmark' suites, got {run.suite.kind!r}"
            )
        self.created_at = created_at
        self.normalized_root = normalized_root

    def execute(self) -> Dict[str, Any]:
        layout = ArtifactLayout(self.run.output_dir)
        suite = self.run.suite
        model_cfg = self.run.model
        model_tag = model_cfg.model_tag

        resolved_model = resolve_model(model_cfg)
        print(
            f"[BenchmarkPlanRunner] model resolved: kind={resolved_model.kind!r}, "
            f"tag={model_tag!r}"
        )

        plan = compile_plan(
            self.run, resolved_model,
            created_at=self.created_at, normalized_root=self.normalized_root,
        )
        print(
            f"[BenchmarkPlanRunner] compiled plan: {len(plan.jobs)} bench(es), "
            f"{plan.to_dict()['total_jobs']} jobs total"
        )

        resolved_run = self._build_resolved_run_dict(resolved_model, plan)
        resolved_run_path = layout.benchmark_resolved_run_path(suite.name, model_tag)
        _write_json(resolved_run_path, resolved_run)
        print(f"[BenchmarkPlanRunner] resolved_run → {resolved_run_path}")

        plan_dict = plan.to_dict()
        plan_path = layout.benchmark_execution_plan_path(suite.name, model_tag)
        _write_json(plan_path, plan_dict)
        print(f"[BenchmarkPlanRunner] execution_plan → {plan_path}")

        return plan_dict

    def _build_resolved_run_dict(
        self, resolved_model: ResolvedModel, plan: ExecutionPlan
    ) -> Dict[str, Any]:
        suite = self.run.suite
        spec: BenchmarkSuiteSpec = suite.spec
        model_cfg = self.run.model

        return {
            "kind": "benchmark",
            "suite": suite.name,
            "suite_description": suite.description,
            "suite_source": suite.source_path,
            "model": {
                "path": model_cfg.path,
                "model_tag": model_cfg.model_tag,
                "original_path": model_cfg.original_path,
                "resolved_kind": resolved_model.kind,
            },
            "output_dir": self.run.output_dir,
            "execution": {
                "mode": self.run.execution.mode,
                **self.run.execution.params,
            },
            "benchmarks": spec.benchmark_ids,
            "benches": [pair.bench for pair in plan.jobs],
            "created_at": self.created_at,
        }


def _write_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
