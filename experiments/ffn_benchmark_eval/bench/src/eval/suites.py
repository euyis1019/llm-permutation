"""Eval Suite definitions: YAML loading, validation, and internal objects.

Design (from 07_benchmark_eval_integration_proposal.md §2):
- Suite = a named eval preset that answers "what to evaluate"
- Suite YAML lives in configs/eval_suites/{kind}/{name}.yaml
- Suite is immutable once named; Run layer cannot override suite params
- Two families: "ppl" and "benchmark"

Suite YAML format
-----------------
A suite only answers "what to evaluate".  It lists benchmark ids and carries a
description; it carries NO behavior-contract fields (no fewshot, task_type,
data_path, stop_tokens, prompt_builder_id, scorer_id, batch_size).  Behavior
contracts live on each benchmark's ``benchmark_meta.json`` protocol.

benchmark::

    kind: benchmark
    name: smoke
    description: "Smoke test：每个 benchmark 各取 10 道题"

    spec:
      benchmarks:
        - id: mmlu
        - id: mmlu_pro

ppl::

    kind: ppl
    name: smoke

    spec:
      datasets:
        - id: d1_math_holdout
          data_paths:
            - /path/to/d1_math.jsonl
          metrics: [ppl]
          nsamples: 256
          seq_len: 512
          window_policy: prefix_trunc
          batch_size: 8
          include_reasoning: true
          include_tools: true
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]


# ── PPL family ────────────────────────────────────────────────────────────────

@dataclass
class PPLDatasetSpec:
    """One dataset entry in a ppl suite spec."""

    id: str
    data_paths: List[str]
    metrics: List[str] = field(default_factory=lambda: ["ppl"])
    nsamples: int = 256
    seq_len: int = 512
    window_policy: str = "prefix_trunc"
    batch_size: int = 128
    include_reasoning: bool = True
    include_tools: bool = True
    # Only used when window_policy == "sliding_window": stride between windows.
    # None → runner falls back to seq_len // 2 (the standard WikiText protocol).
    stride: Optional[int] = None
    # Only used when window_policy == "sliding_window": cap how many documents
    # are concatenated into the eval stream.  None → no cap (whole corpus, as
    # WikiText does).  Domain holdouts set it (e.g. 2000) as an OOM guard so a
    # suite pointed at a huge raw file can't build an unbounded stream.
    max_docs: Optional[int] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PPLDatasetSpec":
        return cls(
            id=d["id"],
            data_paths=d["data_paths"],
            metrics=d.get("metrics", ["ppl"]),
            nsamples=d.get("nsamples", 256),
            seq_len=d.get("seq_len", 512),
            window_policy=d.get("window_policy", "prefix_trunc"),
            batch_size=d.get("batch_size", 128),
            include_reasoning=d.get("include_reasoning", True),
            include_tools=d.get("include_tools", True),
            stride=d.get("stride", None),
            max_docs=d.get("max_docs", None),
        )


@dataclass
class PPLSuiteSpec:
    datasets: List[PPLDatasetSpec]

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PPLSuiteSpec":
        datasets = [PPLDatasetSpec.from_dict(ds) for ds in d.get("datasets", [])]
        if not datasets:
            raise ValueError("ppl suite spec.datasets must be non-empty")
        return cls(datasets=datasets)


# ── Benchmark family ──────────────────────────────────────────────────────────

@dataclass
class BenchmarkTaskSpec:
    """One benchmark entry in a benchmark suite spec.

    A suite only names *which* benchmarks to evaluate.  The behavior contract
    (prompt builder, scorer, stop tokens, fewshot, generation kwargs) is an
    intrinsic property of the benchmark and lives on its ``benchmark_meta.json``
    protocol — it is NOT settable here.

    Two runner kinds
    ----------------
    - ``runner="protocol"`` (default): a normal benchmark evaluated by our
      protocol-driven client_runner (deploy/client). Behavior comes from the
      benchmark's benchmark_meta.json. ``framework``/``dataset``/``metric`` are
      unused.
    - ``runner="external"``: a benchmark evaluated by a third-party framework
      (e.g. evalplus for HumanEval+/MBPP+). The framework is a black box — it
      owns prompt/generation/scoring — so it has no protocol; instead it needs
      ``framework`` (which scaffold) and ``dataset`` (the scaffold's dataset
      key). See src/eval/external/ and DESIGN_external_eval_integration.md.

    Attributes
    ----------
    id        The bench identifier (protocol: e.g. ``mmlu``; external: e.g.
              ``humaneval_plus``).
    nrows     Optional fixed row count (protocol only). Priority over ``ratio``.
    ratio     Optional fractional sample size in (0, 1.0] (protocol only).
    runner    ``"protocol"`` (default) or ``"external"``.
    framework External only: scaffold name, e.g. ``"evalplus"``.
    dataset   External only: scaffold dataset key, e.g. ``"humaneval"``/``"mbpp"``.
    metric    External only: which framework metric maps to unified ``accuracy``
              (default ``"base_pass_at_1"``).

    Priority for protocol sampling: nrows > ratio > full dataset.
    """

    id: str
    nrows: Optional[int] = None
    ratio: Optional[float] = None
    runner: str = "protocol"
    framework: Optional[str] = None
    dataset: Optional[str] = None
    metric: str = "base_pass_at_1"

    VALID_RUNNERS = frozenset({"protocol", "external"})

    @property
    def is_external(self) -> bool:
        return self.runner == "external"

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BenchmarkTaskSpec":
        nrows = d.get("nrows")
        ratio = d.get("ratio")
        bid = d.get("id")
        runner = d.get("runner", "protocol")
        if runner not in cls.VALID_RUNNERS:
            raise ValueError(
                f"benchmark {bid!r}: runner must be one of {sorted(cls.VALID_RUNNERS)}, "
                f"got {runner!r}"
            )
        if nrows is not None and (not isinstance(nrows, int) or nrows <= 0):
            raise ValueError(
                f"benchmark {bid!r}: nrows must be a positive int, got {nrows!r}"
            )
        if ratio is not None and not (0 < ratio <= 1.0):
            raise ValueError(
                f"benchmark {bid!r}: ratio must be in (0, 1.0], got {ratio!r}"
            )
        framework = d.get("framework")
        dataset = d.get("dataset")
        metric = d.get("metric", "base_pass_at_1")
        if runner == "external":
            if not framework:
                raise ValueError(f"benchmark {bid!r}: runner=external requires 'framework'")
            if not dataset:
                raise ValueError(f"benchmark {bid!r}: runner=external requires 'dataset'")
            if nrows is not None or ratio is not None:
                raise ValueError(
                    f"benchmark {bid!r}: nrows/ratio are not supported for external "
                    f"benches (the framework controls sampling)"
                )
        return cls(
            id=bid, nrows=nrows, ratio=ratio,
            runner=runner, framework=framework, dataset=dataset, metric=metric,
        )


@dataclass
class BenchmarkSuiteSpec:
    """Spec for a benchmark suite.

    ``benchmarks`` answers "what to evaluate" (the behavior contract still lives
    on each benchmark_meta.json). ``resources`` and ``batch_cap`` are *execution*
    knobs — they describe "how to run" rather than "what", so they are optional
    and only consumed by the submitter (submit_eval.py), never by the protocol.

    resources  Optional per-runner resource overrides keyed by runner kind, e.g.
               ``{"protocol": {"tp": 2, "mem": 65536, "gpus": 2}, "external": {...}}``.
               Missing keys fall back to the submitter's protocol defaults;
               command-line flags override these in turn.
    batch_cap  Optional max number of full-prompt-list copies per in-process
               ``generate`` call (KV-cache memory guard). Defaults to 3 at the
               runner. Only meaningful for protocol benches.
    """

    benchmarks: List[BenchmarkTaskSpec]
    resources: Dict[str, Any] = field(default_factory=dict)
    batch_cap: Optional[int] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BenchmarkSuiteSpec":
        benchmarks = [BenchmarkTaskSpec.from_dict(t) for t in d.get("benchmarks", [])]
        if not benchmarks:
            raise ValueError("benchmark suite spec.benchmarks must be non-empty")
        resources = d.get("resources") or {}
        if not isinstance(resources, dict):
            raise ValueError(f"benchmark suite spec.resources must be a mapping, got {type(resources).__name__}")
        batch_cap = d.get("batch_cap")
        if batch_cap is not None and (not isinstance(batch_cap, int) or batch_cap <= 0):
            raise ValueError(f"benchmark suite spec.batch_cap must be a positive int, got {batch_cap!r}")
        return cls(benchmarks=benchmarks, resources=resources, batch_cap=batch_cap)

    @property
    def benchmark_ids(self) -> List[str]:
        return [b.id for b in self.benchmarks]


# ── Unified EvalSuite ─────────────────────────────────────────────────────────

# NOTE: calibration is NOT an eval suite. Calibration sets live in
# configs/calibration/ (a sibling of eval_suites) and are referenced directly
# by HEAPr scoring scripts via --cali-data. See docs/conventions/calibration_conventions.md.

@dataclass
class EvalSuite:
    """Parsed, validated suite object.

    Attributes:
        kind:        "ppl" or "benchmark"
        name:        suite name (must match filename stem)
        spec:        PPLSuiteSpec | BenchmarkSuiteSpec
        description: free-text description of what the suite evaluates
        source_path: absolute path to the YAML file this was loaded from
    """

    kind: str
    name: str
    spec: Any  # PPLSuiteSpec | BenchmarkSuiteSpec
    description: str = ""
    source_path: Optional[str] = None

    VALID_KINDS = frozenset({"ppl", "benchmark"})

    @classmethod
    def from_dict(cls, d: Dict[str, Any], source_path: Optional[str] = None) -> "EvalSuite":
        kind = d.get("kind")
        if kind not in cls.VALID_KINDS:
            raise ValueError(
                f"Invalid suite kind {kind!r}. Valid: {sorted(cls.VALID_KINDS)}"
            )
        name = d.get("name")
        if not name:
            raise ValueError("Suite YAML must have a non-empty 'name' field")

        raw_spec = d.get("spec")
        if not raw_spec:
            raise ValueError("Suite YAML must have a 'spec' block")

        if kind == "ppl":
            spec = PPLSuiteSpec.from_dict(raw_spec)
        elif kind == "benchmark":
            spec = BenchmarkSuiteSpec.from_dict(raw_spec)
        else:
            raise AssertionError(f"Unreachable: {kind!r}")

        return cls(
            kind=kind,
            name=name,
            spec=spec,
            description=d.get("description", ""),
            source_path=source_path,
        )

    @property
    def bench_ids(self) -> List[str]:
        """Return list of benchmark/dataset IDs named in this suite.

        For benchmark suites these are benchmark_ids (e.g. ``mmlu``), not the
        finer-grained BenchData ids — the compiler resolves those from each
        benchmark's benchmark_meta.json.
        """
        if self.kind == "ppl":
            return [ds.id for ds in self.spec.datasets]
        elif self.kind == "benchmark":
            return [b.id for b in self.spec.benchmarks]
        return []


# ── Loading ───────────────────────────────────────────────────────────────────

def load_suite(path: str) -> EvalSuite:
    """Load and validate a suite YAML file.

    Args:
        path: Absolute or relative path to the suite YAML file.

    Returns:
        Validated EvalSuite object.

    Raises:
        FileNotFoundError: if the YAML file does not exist.
        ValueError: if the YAML content is invalid.
        ImportError: if PyYAML is not installed.
    """
    if yaml is None:
        raise ImportError(
            "PyYAML is required to load suite files. Install it with: pip install pyyaml"
        )
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Suite file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Suite YAML must be a mapping, got {type(raw).__name__}")

    return EvalSuite.from_dict(raw, source_path=path)


def load_suite_by_name(
    kind: str,
    name: str,
    suites_root: Optional[str] = None,
) -> EvalSuite:
    """Load a suite by kind + name, searching the standard configs directory.

    Standard location: {repo_root}/configs/eval_suites/{kind}/{name}.yaml

    Args:
        kind:        "ppl" or "benchmark"
        name:        suite name (filename stem, without .yaml)
        suites_root: override the root dir; defaults to
                     {this_file}/../../../configs/eval_suites
    """
    if suites_root is None:
        # src/eval/suites.py → src/eval/ → src/ → repo_root
        repo_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        suites_root = os.path.join(repo_root, "configs", "eval_suites")

    path = os.path.join(suites_root, kind, f"{name}.yaml")
    return load_suite(path)
