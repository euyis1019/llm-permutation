#!/usr/bin/env python3
"""Code generation via /v1/completions endpoint (base-model direct completion).

Replaces evalplus --backend vllm (in-process) for models that need a
FluentLLM server. Sends each problem's raw prompt to the running server,
collects completions, and writes a JSONL file in the format evalplus.evaluate
expects:  {"task_id": "HumanEval/0", "solution": "...prompt+completion..."}

Usage (called from src/eval/infra/run_eval_fluentllm.sh for the FluentLLM engine):
  python3 -m src.eval.external.gen_code_completions \
      --dataset humaneval \
      --endpoint http://localhost:8080 \
      --output-jsonl /path/to/humaneval_solutions.jsonl \
      --n-samples 1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

import requests

# ── EOS sequences (mirrored from evalplus/provider/utility.py) ─────────────
BASE_EOS = [
    "<|endoftext|>",
    "<|endofmask|>",
    "</s>",
    "\nif __name__",
    "\ndef main(",
    "\nprint(",
]

EXTRA_EOS = {
    "humaneval": ["\ndef ", "\nclass ", "\nimport ", "\nfrom ", "\nassert "],
    "mbpp":      ['\n"""', "\nassert"],
}


def build_stop_sequences(dataset: str) -> List[str]:
    return BASE_EOS + EXTRA_EOS.get(dataset.lower(), [])


def completion_one(
    prompt: str,
    endpoint: str,
    stop: List[str],
    max_tokens: int = 768,
    timeout: int = 120,
) -> str:
    """Send one completion request; return the generated text."""
    url = f"{endpoint.rstrip('/')}/v1/completions"
    payload = {
        "model": "default",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stop": stop,
    }
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()["choices"][0]["text"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset",      required=True, choices=["humaneval", "mbpp"],
                    help="evalplus dataset key")
    ap.add_argument("--endpoint",     default="http://localhost:8080",
                    help="FluentLLM server base URL")
    ap.add_argument("--output-jsonl", required=True,
                    help="Output JSONL path for evalplus.evaluate")
    ap.add_argument("--n-samples",    type=int, default=1,
                    help="Completions per problem (greedy → always 1)")
    ap.add_argument("--max-tokens",   type=int, default=768)
    ap.add_argument("--workers",      type=int, default=64,
                    help="Concurrent HTTP requests")
    ap.add_argument("--timeout",      type=int, default=180,
                    help="Per-request timeout in seconds")
    args = ap.parse_args()

    # ── Import evalplus data (patched: appdirs/tempdir/wget are now optional) ──
    try:
        from evalplus.data import get_human_eval_plus, get_mbpp_plus
    except ImportError as e:
        print(f"[ERROR] evalplus.data not importable: {e}", file=sys.stderr)
        sys.exit(1)

    # sanitize uses tree_sitter_python (compiled); graceful fallback if absent.
    try:
        from evalplus.sanitize import sanitize as _sanitize
        def sanitize(solution: str, entrypoint: str) -> str:
            try:
                return _sanitize(solution, entrypoint)
            except Exception:
                return solution
    except ImportError:
        print("[gen_code] WARNING: evalplus.sanitize unavailable (tree_sitter_python missing); "
              "using raw solution (may affect scores slightly)")
        def sanitize(solution: str, entrypoint: str) -> str:  # type: ignore
            return solution

    dataset_dict = (
        get_human_eval_plus() if args.dataset == "humaneval" else get_mbpp_plus()
    )
    stop = build_stop_sequences(args.dataset)
    print(f"[gen_code] dataset={args.dataset}  problems={len(dataset_dict)}  "
          f"workers={args.workers}  stop_sequences={len(stop)}")

    # ── Resume: skip already-done task_ids ──────────────────────────────────
    done: dict[str, int] = {}
    if os.path.isfile(args.output_jsonl):
        with open(args.output_jsonl, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                tid = json.loads(line)["task_id"]
                done[tid] = done.get(tid, 0) + 1

    pending = [
        (tid, task)
        for tid, task in dataset_dict.items()
        if done.get(tid, 0) < args.n_samples
    ]
    print(f"[gen_code] already done: {len(done)}  pending: {len(pending)}")

    if not pending:
        print("[gen_code] All solutions already cached. Nothing to do.")
        return

    # ── Generate in parallel ─────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.output_jsonl)), exist_ok=True)

    errors = 0
    with open(args.output_jsonl, "a") as out_f:
        def _generate(tid_task):
            tid, task = tid_task
            prompt = task["prompt"]
            try:
                completion = completion_one(
                    prompt, args.endpoint, stop,
                    max_tokens=args.max_tokens, timeout=args.timeout,
                )
                # direct-completion: prepend prompt so sanitize can find the function
                raw_solution = prompt + completion
                sanitized = sanitize(raw_solution, entrypoint=task["entry_point"])
                return tid, sanitized, None
            except Exception as exc:
                return tid, None, exc

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_generate, item): item[0] for item in pending}
            done_count = 0
            for fut in as_completed(futures):
                tid, solution, err = fut.result()
                done_count += 1
                if err:
                    print(f"[gen_code] ERROR {tid}: {err}", file=sys.stderr)
                    errors += 1
                    continue
                out_f.write(json.dumps({"task_id": tid, "solution": solution}) + "\n")
                out_f.flush()
                if done_count % 50 == 0:
                    print(f"[gen_code] {done_count}/{len(pending)} done ...")

    print(f"[gen_code] Finished. errors={errors}  output={args.output_jsonl}")
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
