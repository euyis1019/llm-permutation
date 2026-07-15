"""Eval registry: kind → runner dispatch and unified entry points.

Design (from 07_benchmark_eval_integration_proposal.md §2.2):
- EvalRun: one concrete eval request (suite + model + output_dir)
- EvalRunner: abstract base; subclassed by PPLRunner and BenchmarkRunner
- create_runner(run): factory that returns the right runner for the run's suite kind

Run config schema
-----------------
::

    suite: smoke                      # suite name (must exist in configs/eval_suites/{kind}/)
    kind: ppl                         # optional; inferred from suite file if omitted

    model:
      path: /path/to/model
      original_path: /path/to/original   # required for pruned models in benchmark runs
      model_tag: r025                 # stable tag for artifact naming

    output_dir: /path/to/output/olmo3-32b/d1_math

    execution:
      mode: local_model_eval          # ppl default; benchmark default is remote_hf_service
      local:
        gpus_per_job: 4
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .suites import EvalSuite, load_suite, load_suite_by_name

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]


# ── Model config ──────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    """Model specification within a Run."""

    path: str
    model_tag: str
    original_path: Optional[str] = None  # required for pruned models in benchmark

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelConfig":
        path = d.get("path")
        if not path:
            raise ValueError("run.model.path is required")
        model_tag = d.get("model_tag")
        if not model_tag:
            raise ValueError("run.model.model_tag is required")
        return cls(
            path=path,
            model_tag=model_tag,
            original_path=d.get("original_path"),
        )

    @property
    def is_pruned(self) -> bool:
        """Heuristic: a model is considered pruned if its directory contains prune_spec.json."""
        spec = os.path.join(self.path, "prune_spec.json")
        return os.path.isfile(spec)


# ── Execution config ──────────────────────────────────────────────────────────

@dataclass
class ExecutionConfig:
    """Execution mode and resource parameters."""

    mode: str  # "local_model_eval" | "remote_hf_service"
    params: Dict[str, Any] = field(default_factory=dict)

    # Default modes per family
    _DEFAULT_PPL = "local_model_eval"
    _DEFAULT_BENCHMARK = "remote_hf_service"

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]], kind: str) -> "ExecutionConfig":
        if d is None:
            mode = cls._DEFAULT_PPL if kind == "ppl" else cls._DEFAULT_BENCHMARK
            return cls(mode=mode)
        mode = d.get("mode", cls._DEFAULT_PPL if kind == "ppl" else cls._DEFAULT_BENCHMARK)
        params = {k: v for k, v in d.items() if k != "mode"}
        return cls(mode=mode, params=params)


# ── EvalRun ───────────────────────────────────────────────────────────────────

@dataclass
class EvalRun:
    """One concrete eval request.

    Attributes:
        suite:      Loaded EvalSuite (kind + spec)
        model:      ModelConfig
        output_dir: Root output directory (e.g. output/olmo3-32b/d1_math)
        execution:  ExecutionConfig
    """

    suite: EvalSuite
    model: ModelConfig
    output_dir: str
    execution: ExecutionConfig

    @classmethod
    def from_dict(
        cls,
        d: Dict[str, Any],
        suites_root: Optional[str] = None,
    ) -> "EvalRun":
        """Parse a run config dict into an EvalRun.

        The suite is resolved from the config's 'suite' key, optionally with
        an explicit 'kind' key.  If 'kind' is absent, it is inferred from the
        suite YAML.
        """
        suite_name = d.get("suite")
        if not suite_name:
            raise ValueError("run config must have a 'suite' field")

        kind = d.get("kind")  # optional; inferred from suite YAML if absent

        # Try to load suite — if kind is given, use it directly; otherwise
        # try both families.
        if kind:
            suite = load_suite_by_name(kind=kind, name=suite_name, suites_root=suites_root)
        else:
            suite = _load_suite_auto(suite_name, suites_root)

        model_raw = d.get("model")
        if not model_raw:
            raise ValueError("run config must have a 'model' block")
        model = ModelConfig.from_dict(model_raw)

        output_dir = d.get("output_dir")
        if not output_dir:
            raise ValueError("run config must have an 'output_dir' field")

        execution = ExecutionConfig.from_dict(d.get("execution"), kind=suite.kind)

        return cls(
            suite=suite,
            model=model,
            output_dir=output_dir,
            execution=execution,
        )

    @classmethod
    def from_yaml(cls, path: str, suites_root: Optional[str] = None) -> "EvalRun":
        """Load a run config from a YAML file."""
        if yaml is None:
            raise ImportError("PyYAML is required. Install with: pip install pyyaml")
        path = os.path.abspath(path)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Run config not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        if not isinstance(raw, dict):
            raise ValueError(f"Run config YAML must be a mapping, got {type(raw).__name__}")
        return cls.from_dict(raw, suites_root=suites_root)


def _load_suite_auto(name: str, suites_root: Optional[str]) -> EvalSuite:
    """Try each known kind until one succeeds."""
    errors = []
    for kind in ("ppl", "benchmark"):
        try:
            return load_suite_by_name(kind=kind, name=name, suites_root=suites_root)
        except FileNotFoundError as e:
            errors.append(str(e))
    raise FileNotFoundError(
        f"Suite {name!r} not found in any family. Tried:\n" + "\n".join(f"  {e}" for e in errors)
    )


# ── EvalRunner base ───────────────────────────────────────────────────────────

class EvalRunner(ABC):
    """Abstract base for family-specific runners."""

    def __init__(self, run: EvalRun) -> None:
        self.run = run

    @abstractmethod
    def execute(self) -> Dict[str, Any]:
        """Execute the run and return a result summary dict."""


# ── Registry / factory ────────────────────────────────────────────────────────

def create_runner(run: EvalRun) -> EvalRunner:
    """Factory: return the correct EvalRunner subclass for run.suite.kind.

    Dispatches to:
        "ppl"       → src.eval.ppl.runner.PPLRunner
        "benchmark" → src.eval.benchmark.compiler.BenchmarkPlanRunner
    """
    kind = run.suite.kind
    if kind == "ppl":
        from .ppl.runner import PPLRunner
        return PPLRunner(run)
    elif kind == "benchmark":
        from .benchmark.compiler import BenchmarkPlanRunner
        return BenchmarkPlanRunner(run)
    else:
        raise ValueError(f"Unknown suite kind {kind!r}. Valid: 'ppl', 'benchmark'")


def build_run(
    suite_path: str,
    model_path: str,
    model_tag: str,
    output_dir: str,
    original_path: Optional[str] = None,
    execution_mode: Optional[str] = None,
) -> EvalRun:
    """Convenience factory: build an EvalRun from individual arguments.

    Useful for programmatic invocation without a run YAML file.

    Args:
        suite_path:     Path to the suite YAML file.
        model_path:     Path to the model directory.
        model_tag:      Stable tag for artifact naming (e.g. 'r025', 'baseline').
        output_dir:     Root output directory.
        original_path:  Path to original model (required for pruned benchmark runs).
        execution_mode: Override execution mode; defaults to family default.
    """
    suite = load_suite(suite_path)
    model = ModelConfig(
        path=model_path,
        model_tag=model_tag,
        original_path=original_path,
    )
    exec_d = {"mode": execution_mode} if execution_mode else None
    execution = ExecutionConfig.from_dict(exec_d, kind=suite.kind)
    return EvalRun(suite=suite, model=model, output_dir=output_dir, execution=execution)
