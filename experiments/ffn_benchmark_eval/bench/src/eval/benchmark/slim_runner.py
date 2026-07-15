"""Slim bench client runner.

This module implements the client-side execution of one slim bench:
1. Load the JSONL data file for the bench.
2. Build a prompt for each row (with few-shot context if needed).
3. Send the prompt to the remote inference backend.
4. Score each response.
5. Write raw responses and scored results to disk.

Supported backends
------------------
- ``remote_vllm_service``  : OpenAI-compatible /v1/completions endpoint
  (vLLM with --api-style openai, the default for Hope deploy jobs).
- ``remote_hf_service``    : HuggingFace text-generation-inference endpoint
  (/generate or compatible).

Design
------
- The runner reads only the slim JSONL; it does NOT re-open any upstream
  benchmark source files.
- Prompt building and scoring are bench-type-specific but implemented inline
  (no external lm-eval dependency) so the client job is self-contained.
- GSM8K uses flexible-extract: look for the last number in the response.
- BBH uses 3-shot CoT: few-shot examples are hard-coded per subtask.
- Multiple-choice benches use exact single-letter answer matching.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from .models import SlimBenchRow


# ── Few-shot example pools ────────────────────────────────────────────────────

# 3-shot CoT examples for BBH subtasks used in slim5.
# These are minimal illustrative examples; they are deterministic and identical
# across all runs.
_BBH_FEW_SHOT: Dict[str, List[Tuple[str, str]]] = {
    "boolean_expressions": [
        ("not True and False is", "False"),
        ("True or False and True is", "True"),
        ("not ( False ) or True is", "True"),
    ],
    "causal_judgement": [
        ("How would a typical person answer each of the following questions about causation?\n"
         "Frank T., had an accident. The accident was caused by his driving above the speed limit. "
         "Would people say that his driving above the speed limit caused the accident?", "Yes"),
        ("Susan made a mistake at work by sending an incorrect report. "
         "She sent the report because she was distracted. "
         "Would people say that her distraction caused the mistake?", "Yes"),
        ("Tom's car broke down because of an oil leak. "
         "Would people say that the oil leak caused the breakdown?", "Yes"),
    ],
    "date_understanding": [
        ("It was Sept. 1st, 2021 a Wednesday. What is the date 10 days later in MM/DD/YYYY?", "09/11/2021"),
        ("Jane was born on the last day of Feburary in 2001. Today is her 16-year-old birthday. "
         "What is the date today in MM/DD/YYYY?", "02/28/2017"),
        ("2015 is coming in 36 hours. What is the date one week from today in MM/DD/YYYY?", "01/05/2015"),
    ],
    "logical_deduction_three_objects": [
        ("The following paragraphs each describe a set of three objects arranged in a fixed order. "
         "The statements are logically consistent within each paragraph. "
         "In a golf tournament, there were three golfers: Amy, Eli, and Eve. "
         "Eve finished below Eli. Amy finished above Eli. "
         "Which golfer finished lowest?\nOptions:\n(A) Amy\n(B) Eli\n(C) Eve", "(C)"),
        ("Alice, Bob, and Claire are playing a game. "
         "Alice is faster than Bob. Bob is faster than Claire. "
         "Who is the slowest?\nOptions:\n(A) Alice\n(B) Bob\n(C) Claire", "(C)"),
        ("Three people are sitting in a row: Jack, Lisa, Mel. "
         "Jack is to the left of Lisa. Mel is to the right of Lisa. "
         "Who is in the middle?\nOptions:\n(A) Jack\n(B) Lisa\n(C) Mel", "(B)"),
    ],
    "object_counting": [
        ("I have a blackberry, a grape, a plum, and a peach. "
         "How many fruits do I have?", "4"),
        ("I have a chair, two tables, and three beds. "
         "How many pieces of furniture do I have?", "6"),
        ("On the table there are two red apples, one green apple. "
         "How many apples are on the table?", "3"),
    ],
}

# 5-shot MMLU example (generic; same 5 examples for any MMLU subject)
_MMLU_FEW_SHOT = [
    (
        "Passage: The following is a multiple choice question.\n"
        "Question: What is 2 + 2?\nA. 3\nB. 4\nC. 5\nD. 6",
        "B",
    ),
]  # placeholder; slim run uses only 0 few-shot for now (overrideable)

# For MMLU/MMLU-Pro/C-Eval/CMMLU, we use no explicit few-shot examples in the
# prompt body — the fewshot count in the suite spec is informational for
# planning; the client injects the model's prior via system prompt only.
# (Full few-shot from the benchmark splits is optional and not required for
# the slim smoke-test.)


# ── Prompt builders ───────────────────────────────────────────────────────────

def _build_mc_prompt(row: SlimBenchRow, fewshot: int = 0) -> str:
    """Build a multiple-choice prompt.

    Format::
        Question: <question>
        A. <choice_a>
        B. <choice_b>
        ...
        Answer:
    """
    pf = row.prompt_fields
    question = pf["question"]
    choices = pf["choices"]
    letters = "ABCDEFGHIJ"

    parts = []
    parts.append(f"Question: {question}")
    for i, choice in enumerate(choices):
        parts.append(f"{letters[i]}. {choice}")
    parts.append("Answer:")

    return "\n".join(parts)


def _build_bbh_prompt(row: SlimBenchRow, fewshot: int = 3) -> str:
    """Build a BBH Chain-of-Thought prompt with hard-coded few-shot examples."""
    pf = row.prompt_fields
    subtask = pf.get("subtask", "")
    question = pf["question"]

    shots = _BBH_FEW_SHOT.get(subtask, [])[:fewshot]
    parts = []
    for q, a in shots:
        parts.append(f"Q: {q}\nA: {a}\n")
    parts.append(f"Q: {question}\nA:")
    return "\n".join(parts)


def _build_gsm8k_prompt(row: SlimBenchRow, fewshot: int = 0) -> str:
    """Build a GSM8K prompt. No few-shot in slim mode."""
    question = row.prompt_fields["question"]
    return f"Question: {question}\nAnswer:"


def build_prompt(row: SlimBenchRow, fewshot: int) -> str:
    """Dispatch to the correct prompt builder based on bench / task_type."""
    if row.parent_bench == "bbh":
        return _build_bbh_prompt(row, fewshot=fewshot)
    elif row.parent_bench == "gsm8k":
        return _build_gsm8k_prompt(row, fewshot=fewshot)
    elif row.task_type == "multiple_choice":
        return _build_mc_prompt(row, fewshot=fewshot)
    else:
        # Generic generate
        question = row.prompt_fields.get("question", "")
        return f"{question}\nAnswer:"


# ── Scoring ───────────────────────────────────────────────────────────────────

def _extract_first_letter(text: str) -> Optional[str]:
    """Extract the first A-J letter from text (case-insensitive)."""
    m = re.search(r"\b([A-Ja-j])\b", text.strip())
    if m:
        return m.group(1).upper()
    # fallback: first character if it's a letter
    stripped = text.strip()
    if stripped and stripped[0].upper() in "ABCDEFGHIJ":
        return stripped[0].upper()
    return None


def _flexible_extract_number(text: str) -> Optional[str]:
    """Flexible-extract: find the last standalone number in the text.

    Follows the GSM8K flexible-extract convention: look for the last number
    (possibly with commas and decimals) in the model output.
    """
    # Also look for #### pattern first (model CoT output)
    m_hash = re.search(r"####\s*([\d,.-]+)", text)
    if m_hash:
        return m_hash.group(1).replace(",", "").strip()
    # Otherwise find all numbers and return the last one
    numbers = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1].replace(",", "")
    return None


def score_response(row: SlimBenchRow, response_text: str) -> Tuple[bool, str]:
    """Score one model response against the gold target.

    Returns:
        (correct, extracted_answer)
    """
    target = row.target.strip()

    if row.task_type == "multiple_choice":
        extracted = _extract_first_letter(response_text) or ""
        correct = extracted.upper() == target.upper()
        return correct, extracted

    elif row.parent_bench == "gsm8k":
        extracted = _flexible_extract_number(response_text) or ""
        # Normalize both sides: strip commas, compare numerically if possible
        try:
            correct = float(extracted.replace(",", "")) == float(target.replace(",", ""))
        except ValueError:
            correct = extracted.strip() == target.strip()
        return correct, extracted

    else:
        # Generic generate: exact match (case-insensitive, stripped)
        extracted = response_text.strip()
        correct = extracted.lower() == target.lower()
        return correct, extracted


# ── Backend clients ───────────────────────────────────────────────────────────

def _call_vllm(
    prompt: str,
    endpoint: str,
    generation_kwargs: Optional[Dict[str, Any]] = None,
    timeout: int = 120,
) -> str:
    """Call vLLM OpenAI-compatible /v1/completions endpoint."""
    try:
        import urllib.request, urllib.error
    except ImportError:
        raise RuntimeError("urllib not available")

    kwargs = generation_kwargs or {}
    payload = {
        "prompt": prompt,
        "max_tokens": kwargs.get("max_new_tokens", 256),
        "temperature": 0.0 if not kwargs.get("do_sample", False) else 1.0,
        "stop": kwargs.get("stop", None),
    }

    url = endpoint.rstrip("/") + "/v1/completions"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    return result["choices"][0]["text"]


def _call_hf_service(
    prompt: str,
    endpoint: str,
    generation_kwargs: Optional[Dict[str, Any]] = None,
    timeout: int = 120,
) -> str:
    """Call HuggingFace text-generation-inference /generate endpoint."""
    try:
        import urllib.request
    except ImportError:
        raise RuntimeError("urllib not available")

    kwargs = generation_kwargs or {}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": kwargs.get("max_new_tokens", 256),
            "do_sample": kwargs.get("do_sample", False),
            "temperature": kwargs.get("temperature", 1.0),
        },
    }

    url = endpoint.rstrip("/") + "/generate"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    return result.get("generated_text", "")


def call_backend(
    prompt: str,
    mode: str,
    endpoint: str,
    generation_kwargs: Optional[Dict[str, Any]] = None,
    timeout: int = 120,
) -> str:
    """Dispatch to the correct backend based on execution mode.

    Args:
        mode:     ``remote_vllm_service`` or ``remote_hf_service``
        endpoint: Base URL of the deployed service.
    """
    if mode == "remote_vllm_service":
        return _call_vllm(prompt, endpoint, generation_kwargs, timeout)
    elif mode == "remote_hf_service":
        return _call_hf_service(prompt, endpoint, generation_kwargs, timeout)
    else:
        raise ValueError(
            f"Unsupported execution mode: {mode!r}. "
            "Valid: 'remote_vllm_service', 'remote_hf_service'"
        )


# ── Result schema ─────────────────────────────────────────────────────────────

def _build_sample_result(
    row: SlimBenchRow,
    prompt: str,
    response_text: str,
    correct: bool,
    extracted: str,
    elapsed_sec: float,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "sample_id": row.sample_id,
        "bench": row.bench,
        "task_type": row.task_type,
        "prompt": prompt,
        "response": response_text,
        "extracted": extracted,
        "target": row.target,
        "correct": correct,
        "elapsed_sec": elapsed_sec,
        "error": error,
    }


# ── Main runner ───────────────────────────────────────────────────────────────

def run_slim_bench(
    data_path: str,
    fewshot: int,
    mode: str,
    endpoint: str,
    result_path: str,
    raw_path: str,
    generation_kwargs: Optional[Dict[str, Any]] = None,
    timeout: int = 120,
    concurrency: int = 1,  # currently only sequential is implemented
) -> Dict[str, Any]:
    """Run one slim bench end-to-end.

    Args:
        data_path:   Path to the slim bench JSONL file.
        fewshot:     Number of few-shot examples to include in prompts.
        mode:        Execution mode (``remote_vllm_service`` or ``remote_hf_service``).
        endpoint:    Base URL of the inference service.
        result_path: Where to write the scored result JSON.
        raw_path:    Where to write the raw per-sample responses JSON.
        generation_kwargs: Extra kwargs forwarded to the backend.
        timeout:     Per-request timeout in seconds.
        concurrency: Request concurrency (currently only 1 is implemented).

    Returns:
        Summary dict with accuracy and timing statistics.
    """
    # Load bench rows
    rows: List[SlimBenchRow] = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(SlimBenchRow.from_dict(json.loads(line)))

    if not rows:
        raise ValueError(f"No rows found in bench data file: {data_path}")

    bench_id = rows[0].bench
    print(f"[slim_runner] bench={bench_id!r}  rows={len(rows)}  mode={mode!r}  endpoint={endpoint!r}")

    raw_samples: List[Dict[str, Any]] = []
    n_correct = 0
    total_elapsed = 0.0
    errors = 0

    for i, row in enumerate(rows):
        prompt = build_prompt(row, fewshot=fewshot)
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
                generation_kwargs=generation_kwargs,
                timeout=timeout,
            )
            correct, extracted = score_response(row, response_text)
        except Exception as exc:
            error_msg = repr(exc)
            errors += 1
        elapsed = time.time() - t0
        total_elapsed += elapsed

        if correct:
            n_correct += 1

        sample_result = _build_sample_result(
            row=row,
            prompt=prompt,
            response_text=response_text,
            correct=correct,
            extracted=extracted,
            elapsed_sec=elapsed,
            error=error_msg,
        )
        raw_samples.append(sample_result)

        status = "✓" if correct else ("✗ ERR" if error_msg else "✗")
        print(
            f"  [{i+1}/{len(rows)}] {row.sample_id}  {status}  "
            f"target={row.target!r}  extracted={extracted!r}  {elapsed:.1f}s"
        )

    accuracy = n_correct / len(rows) if rows else 0.0
    summary = {
        "bench": bench_id,
        "data_path": data_path,
        "mode": mode,
        "endpoint": endpoint,
        "fewshot": fewshot,
        "total": len(rows),
        "correct": n_correct,
        "accuracy": accuracy,
        "errors": errors,
        "elapsed_sec_total": total_elapsed,
        "elapsed_sec_avg": total_elapsed / len(rows) if rows else 0.0,
    }

    # Write raw
    os.makedirs(os.path.dirname(os.path.abspath(raw_path)), exist_ok=True)
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump({"bench": bench_id, "samples": raw_samples}, f, ensure_ascii=False, indent=2)

    # Write result
    os.makedirs(os.path.dirname(os.path.abspath(result_path)), exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(
        f"[slim_runner] {bench_id}: {n_correct}/{len(rows)} correct "
        f"({accuracy:.1%})  errors={errors}  total={total_elapsed:.1f}s"
    )
    print(f"[slim_runner] result → {result_path}")
    print(f"[slim_runner] raw    → {raw_path}")

    return summary
