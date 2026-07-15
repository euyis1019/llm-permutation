"""PPL result normalization and disk serialization.

Stable schema for per-dataset PPL results and the run-level summary.
Downstream analysis scripts should read these files, not the raw tensors.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PPLDatasetResult:
    """Normalized result for a single PPL dataset evaluation.

    Attributes:
        dataset_id:   Matches the id in the suite spec.
        ppl:          Perplexity value (primary metric).
        metrics:      Dict of all computed metrics (at minimum {"ppl": <float>}).
        nsamples:     Actual number of samples evaluated.
        seq_len:      Sequence length used.
        window_policy: Window policy used.
        elapsed_sec:  Wall-clock seconds for this dataset.
        error:        Non-None if evaluation failed; ppl will be None.
    """

    dataset_id: str
    ppl: Optional[float]
    metrics: Dict[str, float] = field(default_factory=dict)
    nsamples: Optional[int] = None
    seq_len: Optional[int] = None
    window_policy: Optional[str] = None
    elapsed_sec: Optional[float] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PPLDatasetResult":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


@dataclass
class PPLRunSummary:
    """Run-level summary for a PPL eval run.

    Attributes:
        suite:      Suite name.
        model_tag:  Model tag used.
        model_path: Path to the model directory.
        results:    Per-dataset results (ordered as in suite spec).
        failed_datasets: IDs of datasets that errored.
        created_at: Timestamp string (caller-provided; src/ does not take system time).
    """

    suite: str
    model_tag: str
    model_path: str
    results: List[PPLDatasetResult] = field(default_factory=list)
    failed_datasets: List[str] = field(default_factory=list)
    created_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "suite": self.suite,
            "model_tag": self.model_tag,
            "model_path": self.model_path,
            "created_at": self.created_at,
            "results": {r.dataset_id: r.to_dict() for r in self.results},
            "failed_datasets": self.failed_datasets,
        }
        # Top-level ppl per dataset for quick scanning
        d["ppl_by_dataset"] = {
            r.dataset_id: r.ppl for r in self.results if r.ppl is not None
        }
        return d


# ── Disk I/O helpers ──────────────────────────────────────────────────────────

def write_dataset_result(path: str, result: PPLDatasetResult) -> None:
    """Write a single dataset result to disk."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)


def write_summary(path: str, summary: PPLRunSummary) -> None:
    """Write the run-level summary to disk."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary.to_dict(), f, ensure_ascii=False, indent=2)


def write_resolved_run(path: str, run_dict: Dict[str, Any]) -> None:
    """Write the resolved run config to disk."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(run_dict, f, ensure_ascii=False, indent=2)
