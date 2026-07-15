"""Benchmark / Bench hierarchy data models.

Terminology
-----------
Benchmark
    A logical evaluation node.  May be a leaf or may have children.
    - If it has children, it groups sub-benchmarks or leaf benches.
    - A leaf benchmark (no children) is simultaneously a Bench.
    - Examples: "mmlu" (has many subject children), "gsm8k" (leaf, is also a bench).

Bench
    The minimal executable unit.  Always a leaf node.
    - Bound to exactly one standardised JSONL data file on disk.
    - Identified by a bench_id (e.g. "mmlu__slim5", "gsm8k__slim5").
    - The compiler / runner only operate on Bench objects, never on
      intermediate Benchmark nodes.

Design rules
------------
- A BenchmarkNode with children=[] is a leaf; it can be promoted to a LeafBench.
- The tree is traversed with leaf_benches() to get all executable units.
- No "benchmark group" concept — just BenchmarkNode with optional children.
- LeafBench is the only object that carries a data_path.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ── Leaf Bench ────────────────────────────────────────────────────────────────

@dataclass
class LeafBench:
    """The minimal executable benchmark unit.

    Attributes
    ----------
    bench_id        Unique identifier, e.g. ``mmlu__slim5``.
    parent_bench    Logical parent benchmark name, e.g. ``mmlu``.
    task_type       ``multiple_choice`` or ``generate``.
    fewshot         Number of few-shot examples expected at eval time.
    data_path       Absolute path to the standardised JSONL file for this bench.
    generation_kwargs  Extra kwargs for generation (max_new_tokens, do_sample …).
    metadata        Optional extra metadata (selection_rule, subject list …).
    """

    bench_id: str
    parent_bench: str
    task_type: str          # "multiple_choice" | "generate"
    fewshot: int
    data_path: str          # MUST point to a real JSONL file (contract)
    generation_kwargs: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    VALID_TASK_TYPES = frozenset({"multiple_choice", "generate"})

    def validate(self) -> None:
        """Raise ValueError / FileNotFoundError if this LeafBench is invalid."""
        if not self.bench_id:
            raise ValueError("LeafBench.bench_id must be non-empty")
        if self.task_type not in self.VALID_TASK_TYPES:
            raise ValueError(
                f"LeafBench.task_type must be one of {sorted(self.VALID_TASK_TYPES)}, "
                f"got {self.task_type!r} for bench_id={self.bench_id!r}"
            )
        if not self.data_path:
            raise ValueError(
                f"LeafBench {self.bench_id!r} has no data_path — "
                "a leaf bench MUST be bound to a JSONL file"
            )

    def data_path_exists(self) -> bool:
        """Return True if the bound JSONL file exists on disk."""
        return os.path.isfile(self.data_path)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bench_id": self.bench_id,
            "parent_bench": self.parent_bench,
            "task_type": self.task_type,
            "fewshot": self.fewshot,
            "data_path": self.data_path,
            "generation_kwargs": self.generation_kwargs,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LeafBench":
        return cls(
            bench_id=d["bench_id"],
            parent_bench=d["parent_bench"],
            task_type=d["task_type"],
            fewshot=int(d["fewshot"]),
            data_path=d["data_path"],
            generation_kwargs=d.get("generation_kwargs"),
            metadata=d.get("metadata", {}),
        )


# ── Benchmark node ────────────────────────────────────────────────────────────

@dataclass
class BenchmarkNode:
    """A logical benchmark node in the evaluation hierarchy.

    A BenchmarkNode can be:
    - An intermediate node: has children (sub-BenchmarkNodes).
    - A leaf node: no children; can be promoted to a LeafBench via as_leaf_bench().

    The tree structure allows benchmarks like "mmlu" to group many subject
    sub-benches, while "gsm8k" is a leaf that is directly executable.

    Attributes
    ----------
    name        Benchmark name, e.g. ``mmlu``, ``bbh``, ``gsm8k``.
    description Human-readable description (optional).
    children    Sub-BenchmarkNodes (empty → leaf node).
    leaf_bench  Populated only for leaf nodes that have been resolved to a
                LeafBench (i.e. have a bound data_path).
    metadata    Optional extra metadata.
    """

    name: str
    description: str = ""
    children: List["BenchmarkNode"] = field(default_factory=list)
    leaf_bench: Optional[LeafBench] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_leaf(self) -> bool:
        """True if this node has no children (is a leaf benchmark)."""
        return len(self.children) == 0

    def leaf_benches(self) -> List[LeafBench]:
        """Return all leaf benches reachable from this node (depth-first).

        Only nodes that have a bound LeafBench are included.
        """
        if self.is_leaf:
            if self.leaf_bench is not None:
                return [self.leaf_bench]
            return []
        result: List[LeafBench] = []
        for child in self.children:
            result.extend(child.leaf_benches())
        return result

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "is_leaf": self.is_leaf,
            "metadata": self.metadata,
        }
        if self.is_leaf and self.leaf_bench is not None:
            d["leaf_bench"] = self.leaf_bench.to_dict()
        else:
            d["children"] = [c.to_dict() for c in self.children]
        return d


# ── Benchmark metadata file ───────────────────────────────────────────────────

@dataclass
class BenchmarkMeta:
    """Metadata for a normalised benchmark dataset directory.

    This is written as ``benchmark_meta.json`` alongside the JSONL files.
    It describes the benchmark hierarchy, leaf benches, and data provenance.

    Attributes
    ----------
    benchmark_name  Top-level benchmark name (e.g. ``slim_qwen_test``).
    description     Human-readable description.
    root            The root BenchmarkNode (may be a flat list of leaves or a tree).
    created_at      ISO-8601 timestamp (caller-supplied).
    source_note     Free-text note about where the data came from.
    """

    benchmark_name: str
    description: str
    root: BenchmarkNode
    created_at: Optional[str] = None
    source_note: str = ""

    def leaf_benches(self) -> List[LeafBench]:
        """Return all leaf benches in this benchmark (depth-first)."""
        return self.root.leaf_benches()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark_name": self.benchmark_name,
            "description": self.description,
            "created_at": self.created_at,
            "source_note": self.source_note,
            "root": self.root.to_dict(),
            "leaf_benches": [lb.to_dict() for lb in self.leaf_benches()],
        }

    def write(self, path: str) -> None:
        """Write this metadata as JSON to *path*."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "BenchmarkMeta":
        """Load BenchmarkMeta from a benchmark_meta.json file."""
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        root = _node_from_dict(d["root"])
        return cls(
            benchmark_name=d["benchmark_name"],
            description=d.get("description", ""),
            root=root,
            created_at=d.get("created_at"),
            source_note=d.get("source_note", ""),
        )


def _node_from_dict(d: Dict[str, Any]) -> BenchmarkNode:
    """Recursively reconstruct a BenchmarkNode from a dict."""
    leaf_bench = None
    if d.get("is_leaf") and "leaf_bench" in d:
        leaf_bench = LeafBench.from_dict(d["leaf_bench"])
    children = [_node_from_dict(c) for c in d.get("children", [])]
    return BenchmarkNode(
        name=d["name"],
        description=d.get("description", ""),
        children=children,
        leaf_bench=leaf_bench,
        metadata=d.get("metadata", {}),
    )


# ── Convenience: build a flat benchmark from a list of leaf benches ───────────

def flat_benchmark(name: str, leaf_benches: List[LeafBench], description: str = "") -> BenchmarkNode:
    """Build a BenchmarkNode whose children are all leaf nodes.

    Useful for suites like slim_qwen_test where all benches are direct
    leaves with no sub-grouping.

    Each LeafBench becomes a child BenchmarkNode with is_leaf=True.
    """
    children = [
        BenchmarkNode(name=lb.bench_id, leaf_bench=lb)
        for lb in leaf_benches
    ]
    return BenchmarkNode(name=name, description=description, children=children)
