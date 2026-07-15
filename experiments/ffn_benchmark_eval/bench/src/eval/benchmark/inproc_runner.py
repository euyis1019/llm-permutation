"""In-process benchmark runner.

Drop-in replacement for the deploy+client (HTTP server) execution path of
``client_runner.run_bench_from_meta``.  Instead of starting a vLLM HTTP server
and POSTing one prompt at a time, this runner loads vLLM in-process and submits
a whole round of prompts to ``llm.generate()`` at once — vLLM's continuous
batching then keeps the GPU saturated with no client-side concurrency knob.

Why this exists
---------------
The server/client split's entire complexity (``BENCH_CONCURRENCY`` that must be
high or the GPU idles and the job is killed, 600s endpoint-ready wait, the HTTP
client) existed only to feed a separate server.  evalplus already proved the
in-process ``llm.generate(all_prompts)`` pattern saturates the GPU on its own.
The only HTTP coupling in ``client_runner`` was ``_call_vllm`` (one prompt →
``choices[0].text``); everything else (prompt building, stop-token truncation,
scoring, result schema) is pure text and is reused here verbatim.

Prompt/scoring behavior is identical to ``client_runner``: same ``build_prompt``,
same ``effective_stop``, same ``score_response``, same per-sample result dict —
so per-row results match the server path (greedy decoding is deterministic).

Batching & memory
------------------
A single ``generate`` over (n_rows × n_runs) prompts could exhaust KV cache, so
runs are batched: at most ``batch_cap`` copies of the full prompt list per
``generate`` call.  avg@3 with the default cap=3 is exactly one batch; avg@N for
N > cap is ``ceil(N / batch_cap)`` batches run sequentially.  Each run's outputs
are written to ``run_NN/result.json`` and aggregated by ``compute_summary``.
"""

from __future__ import annotations

import copy
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .behavior_catalog import BASE_MODEL_EXTRA_STOPS, get_protocol
from .client_runner import build_prompt, score_response, _build_sample_result
from .models import BenchData, BenchmarkMeta, BenchmarkProtocol, EvalRow

REPO = Path(__file__).resolve().parents[3]
COMPUTE_SUMMARY = REPO / "src/eval/infra/compute_summary.py"


# ── meta → (BenchData, protocol) ──────────────────────────────────────────────


def resolve_bench(meta_path: str, bench_id: Optional[str]) -> Tuple[BenchData, BenchmarkProtocol]:
    """Resolve one BenchData + its protocol from a benchmark_meta.json.

    Mirrors ``client_runner.run_bench_from_meta``'s resolution so the in-process
    path uses the exact same protocol the server path (and the compiler) would.
    """
    meta = BenchmarkMeta.load(meta_path)

    if bench_id is None:
        if len(meta.bench_data) != 1:
            raise ValueError(
                f"{meta_path} has {len(meta.bench_data)} BenchData entries; "
                f"bench_id is required to pick one of "
                f"{[bd.bench_id for bd in meta.bench_data]}"
            )
        bench_data = meta.bench_data[0]
    else:
        matches = [bd for bd in meta.bench_data if bd.bench_id == bench_id]
        if not matches:
            raise ValueError(
                f"No BenchData with bench_id={bench_id!r} in {meta_path}. "
                f"Available: {[bd.bench_id for bd in meta.bench_data]}"
            )
        bench_data = matches[0]

    protocol = get_protocol(meta.benchmark_id, bench_data.protocol_override)
    return bench_data, protocol


def _load_rows(data_path: str, nrows: Optional[int]) -> List[EvalRow]:
    rows: List[EvalRow] = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(EvalRow.from_dict(json.loads(line)))
    if nrows is not None and len(rows) > nrows:
        rows = rows[:nrows]
    if not rows:
        raise ValueError(f"No rows found in bench data file: {data_path}")
    return rows


# ── one run's result.json (same schema/path as the server path) ───────────────


def _write_run_result(
    bench_id: str,
    parent_benchmark: str,
    data_path: str,
    rows: List[EvalRow],
    prompts: List[str],
    responses: List[str],
    protocol: BenchmarkProtocol,
    run_dir: Path,
) -> Dict[str, Any]:
    raw_samples: List[Dict[str, Any]] = []
    n_correct = 0
    errors = 0
    for row, prompt, resp in zip(rows, prompts, responses):
        try:
            correct, extracted = score_response(row, resp, protocol)
            err = None
        except Exception as exc:  # scoring must never abort the whole run
            correct, extracted, err = False, "", repr(exc)
        if correct:
            n_correct += 1
        if err:
            errors += 1
        raw_samples.append(
            _build_sample_result(
                row=row, prompt=prompt, response_text=resp,
                correct=correct, extracted=extracted, elapsed_sec=0.0, error=err,
            )
        )

    accuracy = n_correct / len(rows) if rows else 0.0
    summary = {
        "bench_id": bench_id,
        "parent_benchmark": parent_benchmark,
        "data_path": data_path,
        "mode": "inproc_vllm",
        "fewshot": protocol.fewshot,
        "total": len(rows),
        "correct": n_correct,
        "accuracy": accuracy,
        "errors": errors,
    }

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "raw.json").write_text(
        json.dumps({"bench_id": bench_id, "samples": raw_samples}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (run_dir / "result.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(
        f"[inproc_runner] {bench_id} {run_dir.name}: {n_correct}/{len(rows)} "
        f"correct ({accuracy:.1%})  errors={errors}"
    )
    return summary


# ── main entry ────────────────────────────────────────────────────────────────


def run_bench_inproc(
    meta_path: str,
    bench_id: Optional[str],
    model_path: str,
    output_base: str,
    model_tag: str,
    n_runs: int = 3,
    tp_size: int = 2,
    batch_cap: int = 3,
    nrows: Optional[int] = None,
    run_offset: int = 0,
) -> str:
    """Evaluate one bench in-process for ``n_runs`` greedy runs.

    Outputs ``output_base/{model_tag}/{bench_id}/run_NN/result.json`` for each
    run plus an aggregated ``summary.json`` — the same layout the server path
    produces, so ``collect`` / ``compute_summary`` are unchanged.

    Returns the bench output directory.
    """
    from vllm import LLM, SamplingParams

    bench_data, protocol = resolve_bench(meta_path, bench_id)
    bid = bench_data.bench_id
    rows = _load_rows(bench_data.data_path, nrows)

    effective_stop = list(protocol.stop_tokens) + BASE_MODEL_EXTRA_STOPS
    max_tokens = int(protocol.generation_kwargs.get("max_new_tokens", 256))
    bench_dir = Path(output_base) / model_tag / bid
    bench_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[inproc_runner] bench={bid!r}  rows={len(rows)}  n_runs={n_runs}  "
        f"tp={tp_size}  batch_cap={batch_cap}  max_tokens={max_tokens}  "
        f"stop={effective_stop!r}"
    )

    # Build prompts once — identical across runs (greedy, deterministic).
    prompts = [build_prompt(row, bench_data, protocol) for row in rows]

    llm = LLM(
        model=model_path,
        tensor_parallel_size=tp_size,
        enable_prefix_caching=True,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=0.90,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=max_tokens, stop=effective_stop or None)

    # Batch runs so a single generate() never exceeds batch_cap copies of the
    # full prompt list (KV-cache memory guard).
    run_indices = list(range(run_offset, run_offset + n_runs))
    pos = 0
    while pos < len(run_indices):
        chunk = run_indices[pos : pos + batch_cap]
        batch_prompts = prompts * len(chunk)
        t0 = time.time()
        outputs = llm.generate(batch_prompts, sampling)
        elapsed = time.time() - t0
        texts = [o.outputs[0].text for o in outputs]
        print(
            f"[inproc_runner] generated batch of {len(chunk)} run(s) "
            f"({len(batch_prompts)} prompts) in {elapsed:.1f}s"
        )
        for k, run_idx in enumerate(chunk):
            run_texts = texts[k * len(rows) : (k + 1) * len(rows)]
            _write_run_result(
                bench_id=bid,
                parent_benchmark=bench_data.parent_benchmark,
                data_path=bench_data.data_path,
                rows=rows,
                prompts=prompts,
                responses=run_texts,
                protocol=protocol,
                run_dir=bench_dir / f"run_{run_idx:02d}",
            )
        pos += batch_cap

    # Aggregate mean±std → summary.json (reuse the shared compute_summary.py).
    print(f"[inproc_runner] computing summary over {n_runs} runs ...")
    subprocess.run(
        [sys.executable, str(COMPUTE_SUMMARY), "--bench-dir", str(bench_dir)],
        check=True,
        env={**os.environ, "PYTHONPATH": f"{REPO}:{os.environ.get('PYTHONPATH', '')}"},
    )
    print(f"[inproc_runner] DONE → {bench_dir}")
    return str(bench_dir)


# ── CLI (called by run_eval.sh) ───────────────────────────────────────────────


def _main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="In-process benchmark runner.")
    ap.add_argument("--meta-path", required=True)
    ap.add_argument("--bench-id", default=None)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--output-base", required=True)
    ap.add_argument("--model-tag", required=True)
    ap.add_argument("--n-runs", type=int, default=3)
    ap.add_argument("--tp-size", type=int, default=2)
    ap.add_argument("--batch-cap", type=int, default=3)
    ap.add_argument("--nrows", type=int, default=None)
    ap.add_argument("--run-offset", type=int, default=0)
    args = ap.parse_args()

    try:
        run_bench_inproc(
            meta_path=args.meta_path,
            bench_id=args.bench_id,
            model_path=args.model_path,
            output_base=args.output_base,
            model_tag=args.model_tag,
            n_runs=args.n_runs,
            tp_size=args.tp_size,
            batch_cap=args.batch_cap,
            nrows=args.nrows,
            run_offset=args.run_offset,
        )
    except Exception:
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        raise


if __name__ == "__main__":
    _main()
