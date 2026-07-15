"""Benchmark data models and behavior-contract objects.

This module defines the canonical schema used throughout the benchmark eval
framework.  Three concept layers (see 07_benchmark_eval_integration_proposal.md
and the S2 design notes) map onto the objects here:

- ``BenchmarkProtocol``  the behavior contract of a benchmark: how to build a
  prompt, how to score, where to stop, how many few-shot examples, and the
  generation kwargs.  A protocol is an intrinsic property of a benchmark; a
  suite is never allowed to override it.
- ``BenchData``          the minimal executable unit.  Always a leaf, bound 1:1
  to one JSONL file on disk.  Inherits its parent benchmark's protocol and only
  writes ``protocol_override`` in the rare case it genuinely needs to.
- ``BenchmarkMeta``      one normalised benchmark (e.g. ``mmlu``): its
  description, protocol, and the list of ``BenchData`` under it.  Written as
  ``benchmark_meta.json`` next to the JSONL files.

``EvalRow`` is the JSONL row schema.  A row is self-contained: the client
runner never re-opens upstream source files after prepare time.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ── JSONL Row Schema ──────────────────────────────────────────────────────────

@dataclass
class EvalRow:
    """One row in a benchmark JSONL file.

    Fields
    ------
    sample_id   Globally unique within the bench, e.g. ``gsm8k_smoke_0001``.
    bench_id    The BenchData id this row belongs to, e.g. ``gsm8k`` or
                ``mmlu`` (for smoke each benchmark is a single BenchData).
    question    The question text (already normalised to plain text).
    choices     List of option strings for multiple-choice rows; ``None`` for
                generate-style rows.
    target      Gold answer as a plain string (a letter for MC, a number/string
                for generate).
    metadata    Extra per-row metadata (subject, source_split, source_row, …).
    """

    sample_id: str
    bench_id: str
    question: str
    choices: Optional[List[str]]
    target: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Required (non-empty) fields, validated at load time (fail-fast §5.5).
    _REQUIRED = ("sample_id", "bench_id", "question", "target")

    def validate(self) -> None:
        for f in self._REQUIRED:
            if not getattr(self, f):
                raise ValueError(
                    f"EvalRow is missing required field {f!r} "
                    f"(sample_id={self.sample_id!r}, bench_id={self.bench_id!r})"
                )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "bench_id": self.bench_id,
            "question": self.question,
            "choices": self.choices,
            "target": self.target,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EvalRow":
        return cls(
            sample_id=d["sample_id"],
            bench_id=d["bench_id"],
            question=d["question"],
            choices=d.get("choices"),
            target=str(d["target"]),
            metadata=d.get("metadata", {}),
        )


# ── Benchmark protocol (behavior contract) ────────────────────────────────────

@dataclass
class BenchmarkProtocol:
    """The behavior contract carried by a benchmark.

    Attributes
    ----------
    prompt_builder_id   Key into ``behavior_catalog.PROMPT_BUILDER_CATALOG``.
    scorer_id           Key into ``behavior_catalog.SCORER_CATALOG``.
    stop_tokens         Benchmark-specific stop sequences.  Used both to stop
                        generation and to truncate the output before answer
                        extraction (see design §3.3).
    fewshot             Number of few-shot examples to prepend.
    generation_kwargs   Extra kwargs forwarded to the inference backend.
    fewshot_examples    Shared few-shot examples that live at the benchmark
                        level (e.g. gsm8k, mmlu_pro CoT).  Subject-specific
                        few-shot is materialised on each BenchData instead.
    """

    prompt_builder_id: str
    scorer_id: str
    stop_tokens: List[str] = field(default_factory=list)
    fewshot: int = 0
    generation_kwargs: Dict[str, Any] = field(default_factory=dict)
    fewshot_examples: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_builder_id": self.prompt_builder_id,
            "scorer_id": self.scorer_id,
            "stop_tokens": self.stop_tokens,
            "fewshot": self.fewshot,
            "generation_kwargs": self.generation_kwargs,
            "fewshot_examples": self.fewshot_examples,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BenchmarkProtocol":
        return cls(
            prompt_builder_id=d["prompt_builder_id"],
            scorer_id=d["scorer_id"],
            stop_tokens=list(d.get("stop_tokens", [])),
            fewshot=int(d.get("fewshot", 0)),
            generation_kwargs=dict(d.get("generation_kwargs", {})),
            fewshot_examples=list(d.get("fewshot_examples", [])),
        )

    def merged_with(self, override: Optional[Dict[str, Any]]) -> "BenchmarkProtocol":
        """Return a copy with the override dict applied field-by-field.

        ``override`` is a partial dict (BenchData.protocol_override); ``None``
        leaves the protocol unchanged.  This is the only sanctioned way to
        diverge from the benchmark default, used rarely at the BenchData level.
        """
        if not override:
            return self
        base = self.to_dict()
        base.update(override)
        return BenchmarkProtocol.from_dict(base)


# ── BenchData (minimal executable unit) ───────────────────────────────────────

@dataclass
class BenchData:
    """One executable bench = one JSONL file + provenance + optional override.

    Attributes
    ----------
    bench_id          Unique id within its benchmark, e.g. ``gsm8k`` or, for a
                      subject-split benchmark, ``mmlu`` (smoke uses one BenchData
                      per benchmark).
    parent_benchmark  The logical benchmark this belongs to, e.g. ``mmlu``.
                      Used for report aggregation only; never for dispatch.
    data_path         Absolute path to the bound JSONL file (1:1 contract).
    total_rows        Number of rows in the JSONL.
    fewshot_examples  Materialised few-shot examples for this BenchData
                      (subject-specific case).  Empty when the few-shot lives on
                      the benchmark protocol instead.
    selection_rule    Human-readable description of how rows were selected.
    selections        Per-row provenance records (source file/split/row …).
    protocol_override Partial protocol override; ``None`` in ~99% of cases.
    """

    bench_id: str
    parent_benchmark: str
    data_path: str
    total_rows: int
    fewshot_examples: List[Dict[str, Any]] = field(default_factory=list)
    selection_rule: str = ""
    selections: List[Dict[str, Any]] = field(default_factory=list)
    protocol_override: Optional[Dict[str, Any]] = None

    def validate(self) -> None:
        if not self.bench_id:
            raise ValueError("BenchData.bench_id must be non-empty")
        if not self.data_path:
            raise ValueError(
                f"BenchData {self.bench_id!r} has no data_path — "
                "a BenchData MUST be bound to a JSONL file"
            )

    def data_path_exists(self) -> bool:
        return os.path.isfile(self.data_path)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bench_id": self.bench_id,
            "parent_benchmark": self.parent_benchmark,
            "data_path": self.data_path,
            "total_rows": self.total_rows,
            "fewshot_examples": self.fewshot_examples,
            "selection_rule": self.selection_rule,
            "selections": self.selections,
            "protocol_override": self.protocol_override,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BenchData":
        return cls(
            bench_id=d["bench_id"],
            parent_benchmark=d["parent_benchmark"],
            data_path=d["data_path"],
            total_rows=int(d.get("total_rows", 0)),
            fewshot_examples=list(d.get("fewshot_examples", [])),
            selection_rule=d.get("selection_rule", ""),
            selections=list(d.get("selections", [])),
            protocol_override=d.get("protocol_override"),
        )


# ── BenchmarkMeta (one normalised benchmark directory) ────────────────────────

@dataclass
class BenchmarkMeta:
    """Metadata for one normalised benchmark directory.

    Written as ``benchmark_meta.json`` alongside the JSONL files under
    ``datasets/benchmark/normalized/{benchmark_id}/``.

    Attributes
    ----------
    benchmark_id   The benchmark identifier (= directory name), e.g. ``mmlu``.
    description    Human-readable description.
    created_at     ISO-8601 timestamp (caller-supplied).
    source_note    Free-text note about where the data came from.
    protocol       The benchmark's behavior contract.
    bench_data     The executable BenchData units under this benchmark.
    """

    benchmark_id: str
    description: str
    protocol: BenchmarkProtocol
    bench_data: List[BenchData]
    created_at: Optional[str] = None
    source_note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "description": self.description,
            "created_at": self.created_at,
            "source_note": self.source_note,
            "protocol": self.protocol.to_dict(),
            "bench_data": [bd.to_dict() for bd in self.bench_data],
        }

    def write(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "BenchmarkMeta":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return cls.from_dict(d)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BenchmarkMeta":
        return cls(
            benchmark_id=d["benchmark_id"],
            description=d.get("description", ""),
            protocol=BenchmarkProtocol.from_dict(d["protocol"]),
            bench_data=[BenchData.from_dict(bd) for bd in d.get("bench_data", [])],
            created_at=d.get("created_at"),
            source_note=d.get("source_note", ""),
        )
