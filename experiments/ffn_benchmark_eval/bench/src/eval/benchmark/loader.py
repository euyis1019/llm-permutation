"""Benchmark model resolution and constraint validation.

Design (from 07_benchmark_eval_integration_proposal.md §1):
- Original models: direct AutoModelForCausalLM.from_pretrained(model_path)
- Pruned models: MUST have original_path; loaded via load_pruned_model(original_path, pruned_dir)
- Benchmark eval does NOT use EvaluationDataset

Model type detection is based on the presence of prune_spec.json in the model directory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from ..registry import ModelConfig


# ── Model resolution ──────────────────────────────────────────────────────────

@dataclass
class ResolvedModel:
    """Fully validated model specification ready for deployment.

    Attributes:
        path:          Path to the model directory (pruned or original).
        model_tag:     Stable tag for artifact naming.
        original_path: Path to the original model (only for pruned models).
        kind:          "original" or "pruned"
    """

    path: str
    model_tag: str
    original_path: Optional[str]
    kind: str  # "original" | "pruned"

    @property
    def load_args(self) -> dict:
        """Arguments to pass to the deploy service for model loading."""
        d = {"model_path": self.path, "model_kind": self.kind}
        if self.kind == "pruned":
            d["original_path"] = self.original_path
        return d


def resolve_model(model_cfg: ModelConfig) -> ResolvedModel:
    """Validate and resolve a ModelConfig for benchmark deployment.

    Rules:
    - If prune_spec.json exists in model_cfg.path → pruned model
      - original_path MUST be set
      - original_path directory MUST exist
    - Otherwise → original model
      - original_path is ignored (may be None)

    Raises:
        ValueError: if pruned model is missing original_path or original_path does not exist
        FileNotFoundError: if model_cfg.path does not exist
    """
    model_path = model_cfg.path

    if not os.path.isdir(model_path):
        raise FileNotFoundError(
            f"Model directory not found: {model_path!r}"
        )

    spec_file = os.path.join(model_path, "prune_spec.json")
    is_pruned = os.path.isfile(spec_file)

    if is_pruned:
        if not model_cfg.original_path:
            raise ValueError(
                f"Model at {model_path!r} is a pruned model "
                f"(prune_spec.json found at {spec_file!r}) but "
                f"original_path is not set in the run config. "
                f"Pruned benchmark models MUST provide original_path."
            )
        if not os.path.isdir(model_cfg.original_path):
            raise FileNotFoundError(
                f"original_path {model_cfg.original_path!r} does not exist or is not a directory"
            )
        return ResolvedModel(
            path=model_path,
            model_tag=model_cfg.model_tag,
            original_path=model_cfg.original_path,
            kind="pruned",
        )
    else:
        return ResolvedModel(
            path=model_path,
            model_tag=model_cfg.model_tag,
            original_path=model_cfg.original_path,  # may be None; not required
            kind="original",
        )
