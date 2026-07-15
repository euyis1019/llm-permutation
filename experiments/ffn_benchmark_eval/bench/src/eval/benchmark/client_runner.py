"""Benchmark client runner.

This module implements the client-side execution of one BenchData:
1. Load the JSONL data file (EvalRows) for the bench.
2. Build a prompt for each row via the benchmark's prompt builder + few-shot.
3. Send the prompt to the remote inference backend, with stop tokens injected.
4. Truncate the response at the stop tokens, then score it.
5. Write raw responses and scored results to disk.

Behavior is driven entirely by the resolved ``BenchmarkProtocol`` and the
``behavior_catalog`` lookup tables — there is no per-benchmark if/else dispatch
in this file.  The deploy/client split and the protocol contract are described
in 07_benchmark_eval_integration_proposal.md and the S2 design notes.

Supported backends
------------------
- ``remote_vllm_service``  OpenAI-compatible /v1/completions endpoint.
- ``remote_hf_service``    HuggingFace text-generation-inference /generate.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from .behavior_catalog import (
    BASE_MODEL_EXTRA_STOPS,
    get_prompt_builder,
    get_scorer,
    truncate_at_stop,
)
from .models import BenchData, BenchmarkProtocol, EvalRow


# ── Prompt building & scoring (protocol-driven) ───────────────────────────────

def _resolve_fewshot_examples(
    row: EvalRow,
    bench_data: BenchData,
    protocol: BenchmarkProtocol,
) -> List[Dict[str, Any]]:
    """Return the correct few-shot examples for this specific row.

    Two cases:

    1. **Shared fewshot** (gsm8k, mmlu_pro): examples live on ``protocol``
       and are the same for every row.  ``bench_data.fewshot_examples`` is
       empty; we return ``protocol.fewshot_examples`` directly.

    2. **Per-subject/subtask fewshot** (mmlu, mmlu_redux, ceval, cmmlu, bbh):
       examples live on ``bench_data.fewshot_examples`` and each entry carries
       a ``_key`` field (subject or subtask name).  We look up the key from
       ``row.metadata`` (field ``"subject"`` or ``"subtask"``) and return only
       the matching entries, stripping the internal ``_key`` field so builders
       receive clean dicts.
    """
    # Case 1: shared fewshot in protocol
    if not bench_data.fewshot_examples:
        return protocol.fewshot_examples

    # Case 2: per-subject/subtask — find the key for this row
    key = row.metadata.get("subject") or row.metadata.get("subtask") or ""
    matched = [
        {k: v for k, v in ex.items() if k != "_key"}
        for ex in bench_data.fewshot_examples
        if ex.get("_key") == key
    ]
    if not matched and key:
        # Fallback: no match found (e.g. new subject not in dev split).
        # Return empty rather than wrong examples.
        pass
    return matched


def build_prompt(row: EvalRow, bench_data: BenchData, protocol: BenchmarkProtocol) -> str:
    """Build the prompt for *row* using the benchmark's prompt builder.

    Few-shot examples are resolved per-row: for subject/subtask-specific
    benchmarks the correct subset is selected from bench_data.fewshot_examples
    by matching row.metadata["subject"] or row.metadata["subtask"].
    For shared-fewshot benchmarks (gsm8k, mmlu_pro) the protocol examples
    are used directly.
    """
    builder = get_prompt_builder(protocol.prompt_builder_id)
    examples = _resolve_fewshot_examples(row, bench_data, protocol)
    return builder(row, examples, protocol.fewshot)


def score_response(
    row: EvalRow, response_text: str, protocol: BenchmarkProtocol
) -> Tuple[bool, str]:
    """Truncate at stop tokens, then score with the benchmark's scorer."""
    truncated = truncate_at_stop(response_text, protocol.stop_tokens)
    scorer = get_scorer(protocol.scorer_id)
    return scorer(row.target, truncated)


# ── Backend clients ───────────────────────────────────────────────────────────

def _call_vllm(
    prompt: str,
    endpoint: str,
    generation_kwargs: Optional[Dict[str, Any]] = None,
    stop: Optional[List[str]] = None,
    timeout: int = 120,
) -> str:
    """Call vLLM / SGLang OpenAI-compatible /v1/completions endpoint.

    2025-06 change: added ``"model"`` field to the payload.
    -------------------------------------------------------
    The OpenAI Completions spec requires ``model`` as a mandatory field.
    vLLM is lenient and accepts requests without it (falls back to the only
    loaded model), but SGLang / FluentLLM strictly validates the field and
    returns HTTP 400 if it is absent.

    We default to ``"default"`` which is the model name SGLang registers
    when no explicit ``--served-model-name`` is passed at server startup.
    Callers can override via ``generation_kwargs["model"]``.
    """
    import urllib.request

    kwargs = generation_kwargs or {}
    payload = {
        "model": kwargs.get("model", "default"),
        "prompt": prompt,
        "max_tokens": kwargs.get("max_new_tokens", 256),
        "temperature": 0.0 if not kwargs.get("do_sample", False) else 1.0,
        "stop": stop or None,
    }
    url = endpoint.rstrip("/") + "/v1/completions"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    return result["choices"][0]["text"]


def _call_hf_service(
    prompt: str,
    endpoint: str,
    generation_kwargs: Optional[Dict[str, Any]] = None,
    stop: Optional[List[str]] = None,
    timeout: int = 120,
) -> str:
    """Call HuggingFace text-generation-inference /generate endpoint."""
    import urllib.request

    kwargs = generation_kwargs or {}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": kwargs.get("max_new_tokens", 256),
            "do_sample": kwargs.get("do_sample", False),
            "temperature": kwargs.get("temperature", 1.0),
            "stop": stop or None,
        },
    }
    url = endpoint.rstrip("/") + "/generate"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    return result.get("generated_text", "")


def call_backend(
    prompt: str,
    mode: str,
    endpoint: str,
    generation_kwargs: Optional[Dict[str, Any]] = None,
    stop: Optional[List[str]] = None,
    timeout: int = 120,
) -> str:
    """Dispatch to the correct backend based on execution mode.

    Args:
        mode:  ``remote_vllm_service`` or ``remote_hf_service``.
        stop:  Stop sequences to forward to the backend.
    """
    if mode == "remote_vllm_service":
        return _call_vllm(prompt, endpoint, generation_kwargs, stop, timeout)
    elif mode == "remote_hf_service":
        return _call_hf_service(prompt, endpoint, generation_kwargs, stop, timeout)
    else:
        raise ValueError(
            f"Unsupported execution mode: {mode!r}. "
            "Valid: 'remote_vllm_service', 'remote_hf_service'"
        )


# ── Result schema ─────────────────────────────────────────────────────────────

def _build_sample_result(
    row: EvalRow,
    prompt: str,
    response_text: str,
    correct: bool,
    extracted: str,
    elapsed_sec: float,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "sample_id": row.sample_id,
        "bench_id": row.bench_id,
        "prompt": prompt,
        "response": response_text,
        "extracted": extracted,
        "target": row.target,
        "correct": correct,
        "elapsed_sec": elapsed_sec,
        "error": error,
    }


# ── Main runner ───────────────────────────────────────────────────────────────

def run_bench(
    bench_data: BenchData,
    protocol: BenchmarkProtocol,
    mode: str,
    endpoint: str,
    result_path: str,
    raw_path: str,
    timeout: int = 120,
    nrows: Optional[int] = None,
    concurrency: Optional[int] = None,
) -> Dict[str, Any]:
    """Run one BenchData end-to-end against a deployed inference service.

    Args:
        bench_data:  The executable BenchData (bound to a JSONL file).
        protocol:    The resolved BenchmarkProtocol (override already applied).
        mode:        Execution mode (``remote_vllm_service`` / ``remote_hf_service``).
        endpoint:    Base URL of the inference service.
        result_path: Where to write the scored result JSON.
        raw_path:    Where to write the raw per-sample responses JSON.
        timeout:     Per-request timeout in seconds.
        nrows:       If set, truncate to at most this many rows before evaluation.
        concurrency: Number of in-flight requests. Defaults to env
                     ``BENCH_CONCURRENCY`` or 1 (serial). The backend
                     (``_call_vllm``) is stateless, and per-row scoring is
                     independent, so parallel requests yield identical
                     per-sample results; only ordering of network completion
                     differs, and outputs are re-sorted to row order below.

    Returns:
        Summary dict with accuracy and timing statistics.
    """
    rows: List[EvalRow] = []
    with open(bench_data.data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(EvalRow.from_dict(json.loads(line)))

    if nrows is not None and len(rows) > nrows:
        rows = rows[:nrows]

    if not rows:
        raise ValueError(f"No rows found in bench data file: {bench_data.data_path}")

    if concurrency is None:
        concurrency = int(os.environ.get("BENCH_CONCURRENCY", "1") or "1")
    concurrency = max(1, concurrency)

    bench_id = bench_data.bench_id
    effective_stop = list(protocol.stop_tokens) + BASE_MODEL_EXTRA_STOPS
    print(
        f"[client_runner] bench={bench_id!r}  rows={len(rows)}  mode={mode!r}  "
        f"endpoint={endpoint!r}  concurrency={concurrency}  stop={effective_stop!r}"
    )

    def _eval_one(row: "EvalRow") -> Dict[str, Any]:
        prompt = build_prompt(row, bench_data, protocol)
        t0 = time.time()
        response_text = ""
        error_msg = None
        correct = False
        extracted = ""
        try:
            response_text = call_backend(
                prompt=prompt,
                mode=mode,
                endpoint=endpoint,
                generation_kwargs=protocol.generation_kwargs,
                stop=effective_stop,
                timeout=timeout,
            )
            correct, extracted = score_response(row, response_text, protocol)
        except Exception as exc:
            error_msg = repr(exc)
        elapsed = time.time() - t0
        return {
            "row": row, "prompt": prompt, "response_text": response_text,
            "correct": correct, "extracted": extracted, "elapsed": elapsed,
            "error": error_msg,
        }

    results: List[Optional[Dict[str, Any]]] = [None] * len(rows)
    done_count = 0

    if concurrency == 1:
        for i, row in enumerate(rows):
            results[i] = _eval_one(row)
            done_count += 1
            r = results[i]
            status = "✓" if r["correct"] else ("✗ ERR" if r["error"] else "✗")
            print(
                f"  [{done_count}/{len(rows)}] {row.sample_id}  {status}  "
                f"target={row.target!r}  extracted={r['extracted']!r}  {r['elapsed']:.1f}s"
            )
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            fut_to_idx = {ex.submit(_eval_one, row): i for i, row in enumerate(rows)}
            for fut in as_completed(fut_to_idx):
                i = fut_to_idx[fut]
                results[i] = fut.result()
                done_count += 1
                r = results[i]
                status = "✓" if r["correct"] else ("✗ ERR" if r["error"] else "✗")
                print(
                    f"  [{done_count}/{len(rows)}] {rows[i].sample_id}  {status}  "
                    f"target={rows[i].target!r}  extracted={r['extracted']!r}  {r['elapsed']:.1f}s"
                )

    raw_samples: List[Dict[str, Any]] = []
    n_correct = 0
    total_elapsed = 0.0
    errors = 0
    for r in results:
        if r["correct"]:
            n_correct += 1
        if r["error"]:
            errors += 1
        total_elapsed += r["elapsed"]
        raw_samples.append(_build_sample_result(
            row=r["row"], prompt=r["prompt"], response_text=r["response_text"],
            correct=r["correct"], extracted=r["extracted"],
            elapsed_sec=r["elapsed"], error=r["error"],
        ))

    accuracy = n_correct / len(rows) if rows else 0.0
    summary = {
        "bench_id": bench_id,
        "parent_benchmark": bench_data.parent_benchmark,
        "data_path": bench_data.data_path,
        "mode": mode,
        "endpoint": endpoint,
        "fewshot": protocol.fewshot,
        "total": len(rows),
        "correct": n_correct,
        "accuracy": accuracy,
        "errors": errors,
        "elapsed_sec_total": total_elapsed,
        "elapsed_sec_avg": total_elapsed / len(rows) if rows else 0.0,
    }

    os.makedirs(os.path.dirname(os.path.abspath(raw_path)), exist_ok=True)
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump({"bench_id": bench_id, "samples": raw_samples}, f, ensure_ascii=False, indent=2)

    os.makedirs(os.path.dirname(os.path.abspath(result_path)), exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(
        f"[client_runner] {bench_id}: {n_correct}/{len(rows)} correct "
        f"({accuracy:.1%})  errors={errors}  total={total_elapsed:.1f}s"
    )
    print(f"[client_runner] result → {result_path}")
    print(f"[client_runner] raw    → {raw_path}")

    return summary


# ── Worker entry: resolve BenchData + protocol from benchmark_meta.json ───────

def run_bench_from_meta(
    meta_path: str,
    mode: str,
    endpoint: str,
    result_path: str,
    raw_path: str,
    bench_id: Optional[str] = None,
    data_path_override: Optional[str] = None,
    timeout: int = 120,
    nrows: Optional[int] = None,
    concurrency: Optional[int] = None,
) -> Dict[str, Any]:
    """Resolve a BenchData + protocol from a benchmark_meta.json, then run it.

    This is the entry point Hope worker scripts call: given the path to a
    benchmark's ``benchmark_meta.json``, it loads the BenchData (selected by
    *bench_id*, or the sole entry if omitted), resolves its protocol via the
    behavior catalog (applying any protocol_override), and delegates to
    ``run_bench``.  Keeping the resolution here means workers stay thin and use
    the exact same protocol the compiler would.

    Args:
        meta_path: Path to the benchmark's benchmark_meta.json.
        mode:      Execution mode (``remote_vllm_service`` / ``remote_hf_service``).
        endpoint:  Base URL of the inference service.
        bench_id:  Which BenchData to run; defaults to the only one if unique.
        data_path_override: If set, replaces bench_data.data_path. Used by the
                   compiler to point to a truncated JSONL (nrows/ratio from suite).
        nrows:     If set, truncate to at most this many rows before evaluation.
    """
    import copy

    from .behavior_catalog import get_protocol
    from .models import BenchmarkMeta

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

    if data_path_override:
        bench_data = copy.copy(bench_data)
        bench_data.data_path = data_path_override

    protocol = get_protocol(meta.benchmark_id, bench_data.protocol_override)
    return run_bench(
        bench_data=bench_data,
        protocol=protocol,
        mode=mode,
        endpoint=endpoint,
        result_path=result_path,
        raw_path=raw_path,
        timeout=timeout,
        nrows=nrows,
        concurrency=concurrency,
    )
