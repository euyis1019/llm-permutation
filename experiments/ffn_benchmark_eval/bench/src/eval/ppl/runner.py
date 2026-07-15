"""PPL Runner: executes a ppl EvalRun against real model weights.

Flow:
1. Load suite → iterate over spec.datasets
2. For each dataset:
   a. Build EvaluationDataset (tokenize + cache)
   b. Load model (AutoModelForCausalLM)
   c. Call eval_ppl()
   d. Write per-dataset result to eval/ppl/{suite}/{model_tag}/datasets/{id}.json
3. Write resolved_run.json and summary.json

Requires GPU (model inference).  Designed to run inside a Hope job.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional

from ..registry import EvalRunner, EvalRun
from ..suites import PPLSuiteSpec, PPLDatasetSpec
from .results import (
    PPLDatasetResult,
    PPLRunSummary,
    write_dataset_result,
    write_resolved_run,
    write_summary,
)
from src.data.layout import ArtifactLayout


class PPLRunner(EvalRunner):
    """Runs PPL evaluation for a ppl-family EvalRun.

    The runner is designed to be called from a Hope job script.
    It loads the model once and iterates over all datasets in the suite.

    Args:
        run:        EvalRun with suite.kind == "ppl"
        created_at: Timestamp string (caller-provided; runner does not call time.time())
        device:     Torch device string, e.g. "cuda:0" or "cuda"
        dtype:      Model dtype string, e.g. "bfloat16"
        batch_size: PPL evaluation batch size
    """

    def __init__(
        self,
        run: EvalRun,
        created_at: Optional[str] = None,
        device: Optional[str] = None,
        dtype: str = "bfloat16",
        batch_size: int = 8,
    ) -> None:
        super().__init__(run)
        if run.suite.kind != "ppl":
            raise ValueError(
                f"PPLRunner only handles 'ppl' suites, got {run.suite.kind!r}"
            )
        self.created_at = created_at
        self.device = device
        self.dtype = dtype
        self.batch_size = batch_size

    # ── Public API ────────────────────────────────────────────────────────────

    def execute(self) -> Dict[str, Any]:
        """Run all datasets in the suite and write results to disk.

        Returns:
            Summary dict (same content as summary.json).
        """
        layout = ArtifactLayout(self.run.output_dir)
        suite = self.run.suite
        model_cfg = self.run.model
        spec: PPLSuiteSpec = suite.spec

        # Write resolved run first so we have a record even if eval fails mid-way
        resolved_run = self._build_resolved_run_dict()
        resolved_path = layout.eval_resolved_run_path("ppl", suite.name, model_cfg.model_tag)
        write_resolved_run(resolved_path, resolved_run)
        print(f"[PPLRunner] resolved_run → {resolved_path}")

        # Load model once for all datasets (with timing)
        t_load_start = time.monotonic()
        model = self._load_model()
        t_load_end = time.monotonic()
        load_sec = round(t_load_end - t_load_start, 2)
        print(f"[PPLRunner] model loaded in {load_sec}s (device={self.device}, batch_size={self.batch_size})")

        # Evaluate each dataset
        t_eval_start = time.monotonic()
        dataset_results = []
        failed = []
        for ds_spec in spec.datasets:
            result = self._eval_dataset(ds_spec, model, layout, suite.name, model_cfg.model_tag)
            dataset_results.append(result)
            if result.error:
                failed.append(result.dataset_id)
        t_eval_end = time.monotonic()
        eval_sec = round(t_eval_end - t_eval_start, 2)

        # Write summary
        summary = PPLRunSummary(
            suite=suite.name,
            model_tag=model_cfg.model_tag,
            model_path=model_cfg.path,
            results=dataset_results,
            failed_datasets=failed,
            created_at=self.created_at,
        )
        summary_path = layout.eval_summary_path("ppl", suite.name, model_cfg.model_tag)
        write_summary(summary_path, summary)
        print(f"[PPLRunner] summary     → {summary_path}")

        summary_dict = summary.to_dict()

        # Inject timing info into the summary dict (written alongside summary.json)
        timing = {
            "model_load_sec": load_sec,
            "total_eval_sec": eval_sec,
            "total_sec": round(load_sec + eval_sec, 2),
            "config": {
                "device": self.device,
                "batch_size": self.batch_size,
                "dtype": self.dtype,
                "n_gpus": self._detect_gpu_count(),
            },
            "datasets": {r.dataset_id: r.elapsed_sec for r in dataset_results},
        }
        summary_dict["timing"] = timing
        timing_path = os.path.join(os.path.dirname(summary_path), "timing.json")
        with open(timing_path, "w") as f:
            json.dump(timing, f, indent=2)
        print(f"[PPLRunner] timing      → {timing_path}")
        print(f"[PPLRunner] load={load_sec}s eval={eval_sec}s total={timing['total_sec']}s")

        if failed:
            print(f"[PPLRunner] WARNING: {len(failed)} dataset(s) failed: {failed}")
        else:
            print(f"[PPLRunner] All {len(dataset_results)} dataset(s) completed OK")

        return summary_dict

    @staticmethod
    def _detect_gpu_count() -> int:
        """Detect number of visible GPUs."""
        try:
            import torch
            return torch.cuda.device_count()
        except Exception:
            return 0

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _load_model(self):
        """Load the model from model.path (or via load_pruned_model for pruned)."""
        import torch
        from transformers import AutoModelForCausalLM

        model_cfg = self.run.model
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.dtype, torch.bfloat16)

        print(f"[PPLRunner] loading model from {model_cfg.path}")

        # When a specific device is given (e.g. "cuda:0"), load everything onto
        # that single card.  This keeps logits on one GPU so batch_size can be
        # larger without cross-device transfers, and GPU utilisation stays high
        # enough to avoid the platform's auto-kill policy (util < 60%).
        # When no device is given, fall back to device_map="auto" (multi-GPU).
        target_device = self.device  # e.g. "cuda:0" or None

        if model_cfg.is_pruned:
            if not model_cfg.original_path:
                raise ValueError(
                    f"Model at {model_cfg.path!r} appears to be pruned "
                    "(prune_spec.json found) but original_path is not set. "
                    "Pruned models require explicit original_path for loading."
                )
            from src.pruning.heapr import load_pruned_model
            model = load_pruned_model(
                original_model_path=model_cfg.original_path,
                pruned_dir=model_cfg.path,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
            )
            model = model.to(torch_dtype)
            if target_device is not None:
                model = model.to(target_device)
            else:
                from accelerate import infer_auto_device_map, dispatch_model
                device_map = infer_auto_device_map(model)
                model = dispatch_model(model, device_map=device_map)
        else:
            if target_device is not None:
                # Single-card: load directly onto the target device.
                model = AutoModelForCausalLM.from_pretrained(
                    model_cfg.path,
                    torch_dtype=torch_dtype,
                    trust_remote_code=True,
                    device_map=target_device,
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_cfg.path,
                    torch_dtype=torch_dtype,
                    trust_remote_code=True,
                    device_map="auto",
                )

        model.eval()
        return model

    def _eval_dataset(
        self,
        ds_spec: PPLDatasetSpec,
        model,
        layout: ArtifactLayout,
        suite_name: str,
        model_tag: str,
    ) -> PPLDatasetResult:
        """Evaluate one dataset and write its result file."""
        from transformers import AutoTokenizer
        from src.data.evaluation import EvaluationDataset
        from src.eval.perplexity import eval_ppl, eval_ppl_sliding_window

        print(f"[PPLRunner] evaluating dataset: {ds_spec.id}")
        t0 = time.monotonic()

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.run.model.path, trust_remote_code=True
            )
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id

            eval_dataset = EvaluationDataset(
                data_paths=ds_spec.data_paths,
                tokenizer=tokenizer,
                output_dir=self.run.output_dir,
                nsamples=ds_spec.nsamples,
                seq_len=ds_spec.seq_len,
                window_policy=ds_spec.window_policy,
                include_reasoning=ds_spec.include_reasoning,
                include_tools=ds_spec.include_tools,
                max_docs=ds_spec.max_docs,
                created_at=self.created_at,
            )
            data = eval_dataset.load()

            # sliding_window is the standard, full-corpus WikiText PPL protocol:
            # one continuous token stream, scored with overlapping windows so
            # every token has full left context.  All other policies cut the
            # corpus into independent windows and use the plain eval_ppl path.
            if ds_spec.window_policy == "sliding_window":
                ppl_value = eval_ppl_sliding_window(
                    model=model,
                    data=data,
                    batch_size=ds_spec.batch_size,
                    stride=ds_spec.stride,
                    device=self.device,
                    window_size=ds_spec.seq_len,
                )
            else:
                ppl_value = eval_ppl(
                    model=model,
                    data=data,
                    batch_size=ds_spec.batch_size,
                    device=self.device,
                )

            elapsed = time.monotonic() - t0
            result = PPLDatasetResult(
                dataset_id=ds_spec.id,
                ppl=ppl_value,
                metrics={"ppl": ppl_value},
                nsamples=int(data["input_ids"].shape[0]),
                seq_len=ds_spec.seq_len,
                window_policy=ds_spec.window_policy,
                elapsed_sec=round(elapsed, 2),
            )
        except Exception as exc:
            elapsed = time.monotonic() - t0
            import traceback
            tb = traceback.format_exc()
            print(f"[PPLRunner] ERROR on dataset {ds_spec.id}: {exc}\n{tb}")
            result = PPLDatasetResult(
                dataset_id=ds_spec.id,
                ppl=None,
                metrics={},
                elapsed_sec=round(elapsed, 2),
                error=str(exc),
            )

        # Write per-dataset result
        result_path = layout.ppl_dataset_result_path(suite_name, model_tag, ds_spec.id)
        write_dataset_result(result_path, result)
        status = f"ppl={result.ppl:.4f}" if result.ppl is not None else f"ERROR: {result.error}"
        print(f"[PPLRunner]   {ds_spec.id}: {status}  → {result_path}")

        return result

    def _build_resolved_run_dict(self) -> Dict[str, Any]:
        """Build the resolved_run.json content."""
        suite = self.run.suite
        model_cfg = self.run.model
        spec: PPLSuiteSpec = suite.spec

        return {
            "kind": "ppl",
            "suite": suite.name,
            "suite_source": suite.source_path,
            "model": {
                "path": model_cfg.path,
                "model_tag": model_cfg.model_tag,
                "original_path": model_cfg.original_path,
                "is_pruned_detected": model_cfg.is_pruned,
            },
            "output_dir": self.run.output_dir,
            "execution": {
                "mode": self.run.execution.mode,
                **self.run.execution.params,
            },
            "datasets": [
                {
                    "id": ds.id,
                    "data_paths": ds.data_paths,
                    "metrics": ds.metrics,
                    "nsamples": ds.nsamples,
                    "seq_len": ds.seq_len,
                    "window_policy": ds.window_policy,
                    "batch_size": ds.batch_size,
                }
                for ds in spec.datasets
            ],
            "created_at": self.created_at,
        }
