"""Explicit behavior catalog for benchmark eval.

This module is the single source of truth for benchmark behavior contracts.
It is a set of *explicit lookup tables* — NOT an auto-registration framework.
There are no decorators, no import side effects, no plugin discovery.  Adding a
benchmark means editing a dict here.

Three tables:

- ``BENCHMARK_DEFAULT_PROTOCOL``  benchmark_id → default BenchmarkProtocol
  (prompt_builder_id, scorer_id, stop_tokens, fewshot, generation_kwargs).
- ``PROMPT_BUILDER_CATALOG``      prompt_builder_id → PromptBuilder callable.
- ``SCORER_CATALOG``             scorer_id → Scorer callable.

The protocol defaults follow ``experiments/slime_qwen_evaluation/benchmark_protocol.md``.

Lookup functions fail fast: an unknown id raises with a message naming the id
and listing the valid ids, so misconfiguration is caught at plan time.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from .models import BenchmarkProtocol, EvalRow


# Appended to every benchmark's stop_tokens at inference time so a base model
# can't run on and "write the next question" (design §3.3 / protocol.md §3.3).
BASE_MODEL_EXTRA_STOPS: List[str] = [
    "\n\n\n",
    "\n\nQuestion",   # "Question: ..." 前有空行
    "\n\nProblem",
    "\n\n[",          # "[Question]", "[Answer]", "[Q]" 等变体（双换行）
    "\n[",            # 同上，单换行变体（base model 有时只输出一个换行就续写）
    "\n\nQ:",         # "Q: ..." 格式（BBH 风格的续写）
    "\n\nA:",         # "A: ..." 格式
]


# ── Output truncation ─────────────────────────────────────────────────────────

def truncate_at_stop(text: str, stop_tokens: List[str]) -> str:
    """Truncate *text* at the earliest occurrence of any stop token.

    Used before answer extraction so the scorer never sees text the model
    produced after it had already "finished" the current question.
    """
    cut = len(text)
    for tok in stop_tokens:
        if not tok:
            continue
        idx = text.find(tok)
        if idx != -1:
            cut = min(cut, idx)
    return text[:cut]


# ── Prompt builders ───────────────────────────────────────────────────────────
# Signature: (row, fewshot_examples, fewshot) -> prompt str
PromptBuilder = Callable[[EvalRow, List[Dict[str, Any]], int], str]

_LETTERS = "ABCDEFGHIJ"


def _render_fewshot_mc(examples: List[Dict[str, Any]], k: int) -> str:
    """Render up to *k* multiple-choice few-shot examples as prompt prefix."""
    parts: List[str] = []
    for ex in examples[:k]:
        parts.append(_format_mc_block(ex["question"], ex.get("choices", []), ex.get("target", "")))
    return "".join(parts)


def _format_mc_block(question: str, choices: List[str], answer: str = "") -> str:
    lines = [f"{question}"]
    for i, choice in enumerate(choices):
        lines.append(f"{_LETTERS[i]}. {choice}")
    lines.append(f"Answer:{(' ' + answer) if answer else ''}")
    return "\n".join(lines) + ("\n\n" if answer else "")


def build_mc_standard(row: EvalRow, fewshot_examples: List[Dict[str, Any]], fewshot: int) -> str:
    """Standard multiple-choice prompt (mmlu / ceval / cmmlu / mmlu_pro)."""
    prefix = _render_fewshot_mc(fewshot_examples, fewshot)
    body = _format_mc_block(row.question, row.choices or [], answer="")
    return prefix + body


def build_math500_cot(row: EvalRow, fewshot_examples: List[Dict[str, Any]], fewshot: int) -> str:
    """MATH-500 prompt: ``Problem: ...\\nAnswer:`` with optional CoT fewshot.

    Follows hendrycks_math lm-eval format exactly.
    """
    parts: List[str] = []
    for ex in fewshot_examples[:fewshot]:
        parts.append(f"Problem: {ex['question']}\nAnswer: {ex.get('target', '')}\n\n")
    parts.append(f"Problem: {row.question}\nAnswer:")
    return "".join(parts)


def build_gsm8k_cot(row: EvalRow, fewshot_examples: List[Dict[str, Any]], fewshot: int) -> str:
    """GSM8K prompt: ``Question: ...\\nAnswer:`` with optional shared CoT shots."""
    parts: List[str] = []
    for ex in fewshot_examples[:fewshot]:
        parts.append(f"Question: {ex['question']}\nAnswer: {ex.get('target', '')}\n\n")
    parts.append(f"Question: {row.question}\nAnswer:")
    return "".join(parts)


def build_bbh_cot(row: EvalRow, fewshot_examples: List[Dict[str, Any]], fewshot: int) -> str:
    """BBH Chain-of-Thought prompt: ``Q: ...\\nA: ...`` shots then the question.

    Few-shot examples are materialised per-subtask on the BenchData (each shot
    carries a full reasoning chain ending in "the answer is X").
    """
    parts: List[str] = []
    for ex in fewshot_examples[:fewshot]:
        parts.append(f"Q: {ex['question']}\nA: {ex.get('target', '')}\n\n")
    parts.append(f"Q: {row.question}\nA:")
    return "".join(parts)


def build_cruxeval_output_v1(
    row: EvalRow, fewshot_examples: List[Dict[str, Any]], fewshot: int
) -> str:
    """CRUXEval output prediction: given function code + input, predict return value.

    Prompt format follows the original CRUXEval paper (Gu et al., 2024):
    [BEGIN PROBLEM] / [END PROBLEM] delimiters bracket each example; the model
    fills in the answer after [BEGIN ANSWER].  The ``[END ANSWER]`` stop token
    signals end-of-answer; everything after it is ignored.

    ``row.question`` is expected to be ``{code}\\n\\nassert f({input}) == ??``,
    pre-formatted by the normalizer.
    """
    header = (
        "You will be given a function f and an input to the function. "
        "Your task is to determine the output of the function.\n\n"
    )
    parts = [header]
    for ex in fewshot_examples[:fewshot]:
        parts.append(f"[BEGIN PROBLEM]\n{ex['question']}\n[END PROBLEM]\n")
        parts.append(f"[BEGIN ANSWER]\n{ex['target']}\n[END ANSWER]\n\n")
    parts.append(f"[BEGIN PROBLEM]\n{row.question}\n[END PROBLEM]\n")
    parts.append("[BEGIN ANSWER]\n")
    return "".join(parts)


PROMPT_BUILDER_CATALOG: Dict[str, PromptBuilder] = {
    "mmlu_standard_v1":      build_mc_standard,
    "ceval_standard_v1":     build_mc_standard,
    "cmmlu_standard_v1":     build_mc_standard,
    "mmlu_pro_cot_v1":       build_mc_standard,
    "gsm8k_cot_v1":          build_gsm8k_cot,
    "bbh_cot_v1":            build_bbh_cot,
    "math500_cot_v1":        build_math500_cot,
    "cruxeval_output_v1":    build_cruxeval_output_v1,
}


# ── Scorers ───────────────────────────────────────────────────────────────────
# Signature: (target, truncated_text) -> (correct, extracted)
Scorer = Callable[[str, str], Tuple[bool, str]]


def _extract_first_letter(text: str, letters: str = _LETTERS) -> Optional[str]:
    """Extract the first standalone A–J letter from *text* (case-insensitive)."""
    m = re.search(rf"\b([{letters}{letters.lower()}])\b", text.strip())
    if m:
        return m.group(1).upper()
    stripped = text.strip()
    if stripped and stripped[0].upper() in letters:
        return stripped[0].upper()
    return None


def score_first_letter_choice(target: str, text: str) -> Tuple[bool, str]:
    """Multiple-choice: first A–J letter equals the gold letter."""
    extracted = _extract_first_letter(text) or ""
    return (extracted.upper() == target.strip().upper(), extracted)


def score_mmlu_pro_regex(target: str, text: str) -> Tuple[bool, str]:
    """MMLU-Pro: regex ``answer is (X)`` then fall back to first A–J letter."""
    m = re.search(r"answer is \(?([A-J])\)?", text, flags=re.IGNORECASE)
    extracted = m.group(1).upper() if m else (_extract_first_letter(text) or "")
    return (extracted.upper() == target.strip().upper(), extracted)


def score_gsm8k_flexible_number(target: str, text: str) -> Tuple[bool, str]:
    """GSM8K flexible-extract: priority ``#### N`` > ``the answer is N`` > last number."""
    m_hash = re.search(r"####\s*([\d,.\-]+)", text)
    if m_hash:
        extracted = m_hash.group(1).replace(",", "").strip()
    else:
        # "The answer is N" — take the FIRST occurrence so base-model run-on text
        # (extra few-shot examples appended after the real answer) doesn't mislead.
        m_ans = re.search(r"[Tt]he answer is\s*(-?[\d,]+(?:\.\d+)?)", text)
        if m_ans:
            extracted = m_ans.group(1).replace(",", "").strip()
        else:
            numbers = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
            extracted = numbers[-1].replace(",", "") if numbers else ""
    try:
        correct = float(extracted) == float(target.replace(",", ""))
    except ValueError:
        correct = extracted.strip() == target.strip()
    return (correct, extracted)


def score_bbh_cot_extract(target: str, text: str) -> Tuple[bool, str]:
    """BBH: prefer ``the answer is X``, else the trimmed text; exact match."""
    m = re.search(r"the answer is\s*(.*?)\s*\.?\s*$", text, flags=re.IGNORECASE | re.DOTALL)
    extracted = (m.group(1).strip() if m else text.strip())
    return (extracted.lower() == target.strip().lower(), extracted)


def score_cruxeval_exact_match_v1(target: str, text: str) -> Tuple[bool, str]:
    """CRUXEval: exact string match on the first non-empty output line.

    The model generates the Python repr answer right after ``[BEGIN ANSWER]``
    and vLLM stops at ``[END ANSWER]``.  We strip surrounding whitespace, then
    take only the first non-empty line (defensive against trailing newlines or
    a ``[END ANSWER]`` that slipped past the stop token), and compare it to the
    gold Python-repr string verbatim.
    """
    answer = text.strip()
    # Defensive: strip [END ANSWER] tag if it slipped past the stop token
    if "[END ANSWER]" in answer:
        answer = answer.split("[END ANSWER]", 1)[0].strip()
    # Take only the first non-empty line
    lines = [ln for ln in answer.split("\n") if ln.strip()]
    answer = lines[0].strip() if lines else ""
    return (answer == target.strip(), answer)


# ── MATH-500 scorer (ported from entropy/exp/01_evaluate/eval_rule.py) ────────

def _extract_all_boxed(text: str) -> List[str]:
    """Extract all \\boxed{...} contents, handling nested braces."""
    results = []
    i = 0
    while i < len(text):
        pos = text.find(r'\boxed{', i)
        if pos == -1:
            break
        depth = 0
        start = pos + len(r'\boxed{')
        j = start
        while j < len(text):
            if text[j] == '{':
                depth += 1
            elif text[j] == '}':
                if depth == 0:
                    results.append(text[start:j].strip())
                    break
                depth -= 1
            j += 1
        i = pos + 1
    return results


def _normalize_math(s: str) -> str:
    """Normalize LaTeX math string for comparison."""
    s = s.strip()
    s = re.sub(r'^\$+|\$+$', '', s).strip()
    # Remove LaTeX display markers
    s = re.sub(r'^\\[\[\]]|\\[\[\]]$', '', s).strip()
    s = s.replace(r'\dfrac', r'\frac')
    s = re.sub(r'\\left\s*', '', s)
    s = re.sub(r'\\right\s*', '', s)
    s = re.sub(r',\s+', ',', s)
    # Strip spaces adjacent to brackets/parens so e.g. "( 3, \pi )" == "(3, \pi)"
    s = re.sub(r'\(\s+', '(', s)
    s = re.sub(r'\s+\)', ')', s)
    s = re.sub(r'\[\s+', '[', s)
    s = re.sub(r'\s+\]', ']', s)
    # Remove variable assignments like "x=", "k=", "N="
    s = re.sub(r'^[a-zA-Z]\s*=\s*', '', s)
    # Remove coordinate labels like "(x,y) = "
    s = re.sub(r'^\([a-z,\s]+\)\s*=\s*', '', s)
    # Remove angle labels like "\angle B = "
    s = re.sub(r'^\\angle\s+[A-Z]\s*=\s*', '', s)
    for cmd in (r'\\text', r'\\mathrm', r'\\mathbf', r'\\mathit'):
        s = re.sub(cmd + r'\{([^}]*)\}', lambda m: m.group(1), s)
    # Remove common units and suffixes (inches, units, degrees, cm, students, outfits, etc.)
    s = re.sub(r'\s+(?:inches|units|degrees|degree|°|cm|m|km|ft|students?|outfits?)\s*$', '', s, flags=re.IGNORECASE)
    # Normalize sqrt: \sqrt2 -> \sqrt{2}, \sqrt{2} -> \sqrt{2}
    s = re.sub(r'\\sqrt(\d)', r'\\sqrt{\1}', s)
    # Normalize pi (both "pi" and "\pi" should match)
    s = re.sub(r'(?<!\\)pi\b', r'\\pi', s)
    # Remove spaces around equals sign
    s = re.sub(r'\s*=\s*', '=', s)
    lower = s.lower().strip()
    if lower in ('yes', 'true'):
        return 'yes'
    if lower in ('no', 'false'):
        return 'no'
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def _try_numeric_math(s: str) -> Optional[float]:
    """Try to parse a LaTeX math string as a float."""
    import math as _math
    s = s.strip()
    try:
        return float(s)
    except ValueError:
        pass
    m = re.match(r'^(-?\d+)\s*/\s*(\d+)$', s)
    if m:
        n, d = int(m.group(1)), int(m.group(2))
        return n / d if d != 0 else None
    m = re.match(r'^\\(?:d?frac)\{(-?\d+)\}\{(\d+)\}$', s.strip())
    if m:
        n, d = int(m.group(1)), int(m.group(2))
        return n / d if d != 0 else None
    return None


def score_math500_boxed(target: str, text: str) -> Tuple[bool, str]:
    """MATH-500: extract last valid \\boxed{} and compare to gold answer.

    Extraction priority (highest → lowest):
      1. Last \\boxed{...} in the response (standard CoT format).
      2. "The answer is X" / "answer is X" in the tail (≤ 400 chars).
      3. Last $...$ expression in the tail (base model often writes ``$42$``).
      4. Last line of the response if it is a bare number/simple expression
         (base model sometimes just outputs the answer on its own line).

    Comparison: exact after _normalize_math, then numeric equivalence.
    """
    # ── Step 1: \boxed{} ──────────────────────────────────────────────────────
    candidates = [c for c in _extract_all_boxed(text) if len(c) <= 120]
    extracted = candidates[-1] if candidates else ""

    # ── Step 2: "The/the answer is X" in tail ────────────────────────────────
    if not extracted:
        tail = text[-400:]
        m = re.search(
            r'(?:[Tt]he\s+(?:final\s+)?answer\s+is|[Aa]nswer\s+is)[:\s]+\**\$?([^\n\$]{1,60})\$?\**',
            tail,
        )
        if m:
            extracted = m.group(1).strip().rstrip('.,').strip()

    # ── Step 3: last $...$ expression in tail ────────────────────────────────
    if not extracted:
        tail = text[-400:]
        # Match $...$ (non-greedy, single-line), prefer the last one
        dollar_matches = re.findall(r'\$([^\$\n]{1,80})\$', tail)
        if dollar_matches:
            extracted = dollar_matches[-1].strip().rstrip('.,').strip()

    # ── Step 4: Extract from first line's $...$ or \boxed{} ───────────────────
    # Handles cases like " $-50$.\nSolution:" - extract $-50$ not the whole line
    if not extracted:
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        if lines:
            first = lines[0].rstrip('.').strip()
            # Handle LaTeX environments like \begin{align*}
            if first.startswith('\\begin{'):
                # Look for \boxed{} or the last line with = sign
                env_content = '\n'.join(lines[:15])  # First 15 lines
                boxed_in_env = _extract_all_boxed(env_content)
                if boxed_in_env:
                    extracted = boxed_in_env[-1]
                else:
                    # Find last line with = in the environment (before \end)
                    in_env = True  # Already inside env since first line is \begin{
                    last_candidate = ""
                    for line in lines[:15]:
                        if '\\end{' in line:
                            break
                        if in_env and '=' in line:
                            # Extract after the last =
                            parts = line.split('=')
                            if len(parts) >= 2:
                                after_eq = parts[-1].strip()
                                # Remove & and everything after it (alignment markers)
                                after_eq = after_eq.split('&')[0].strip()
                                # Remove trailing \\ (line continuation)
                                after_eq = after_eq.rstrip('\\').strip()
                                # Accept if it looks like a math expression (including \frac, \sqrt, etc.)
                                if after_eq and (not after_eq.startswith('\\') or after_eq.startswith('\\frac') or after_eq.startswith('\\sqrt') or after_eq.startswith('\\pi') or after_eq.startswith('\\left') or after_eq.startswith('\\begin')):
                                    last_candidate = after_eq
                    extracted = last_candidate
            # First try to extract \boxed{} from first line
            elif boxed_matches := _extract_all_boxed(first):
                extracted = boxed_matches[-1]
            else:
                # Try to extract $...$ from first line
                dollar_matches = re.findall(r'\$([^\$\n]{1,80})\$', first)
                if dollar_matches:
                    extracted = dollar_matches[-1].strip().rstrip('.,').strip()
                # Try to extract from "... = X ..." pattern (calculation result)
                elif '=' in first:
                    # Extract the number after "= " (before any text like "outfits")
                    m = re.search(r'=\s*(\d+)', first)
                    if m:
                        extracted = m.group(1)
                # Try to extract from "a*b*c = X" pattern (pure calculation)
                elif re.match(r'^[\d\s\+\-\*/=]+$', first):
                    # Just the last number
                    numbers = re.findall(r'\d+', first)
                    if numbers:
                        extracted = numbers[-1]  # Last number is likely the answer
                # Try to extract from "X units" pattern (bare number with units)
                elif re.match(r'^\d+\s+[a-zA-Z]+$', first):
                    m = re.match(r'^(\d+)', first)
                    if m:
                        extracted = m.group(1)
                elif len(first) <= 50:
                    # Single capitalized word like "Carla"
                    if re.match(r'^[A-Z][a-z]+$', first):
                        extracted = first
                    else:
                        # For bare answers, strip explanation suffixes
                        cleaned = re.split(r'\s+(?:Solution|Explanation|Note|Thus|Therefore|So|Hence):', first, flags=re.IGNORECASE)[0]
                        cleaned = cleaned.strip()
                        if cleaned:
                            extracted = cleaned

    # ── Step 5: Last non-empty line if it looks like a bare answer ───────────
    if not extracted:
        lines = [l.strip() for l in text.rstrip().split('\n') if l.strip()]
        if lines:
            last = lines[-1].rstrip('.')
            # Accept bare numbers, simple fractions, or short expressions (≤ 30 chars)
            # but reject lines that look like prose (contain spaces + letters)
            if len(last) <= 30 and not re.search(r'[a-zA-Z]{3,}', last):
                extracted = last

    if not extracted:
        return False, ""

    ext_norm = _normalize_math(extracted)
    gold_norm = _normalize_math(target)

    # Exact after normalization
    if ext_norm == gold_norm:
        return True, extracted

    # Numeric equivalence
    ext_num = _try_numeric_math(ext_norm)
    gold_num = _try_numeric_math(gold_norm)
    if ext_num is not None and gold_num is not None:
        tol = max(1e-6, abs(gold_num) * 1e-4)
        if abs(ext_num - gold_num) <= tol:
            return True, extracted

    return False, extracted


SCORER_CATALOG: Dict[str, Scorer] = {
    "first_letter_choice_v1":      score_first_letter_choice,
    "mmlu_pro_regex_v1":           score_mmlu_pro_regex,
    "gsm8k_flexible_number_v1":    score_gsm8k_flexible_number,
    "bbh_cot_extract_v1":          score_bbh_cot_extract,
    "math500_boxed_v1":            score_math500_boxed,  # Enhanced version with better normalization
    "cruxeval_exact_match_v1":     score_cruxeval_exact_match_v1,
}


# ── Default protocol per benchmark ────────────────────────────────────────────
# Values follow experiments/slime_qwen_evaluation/benchmark_protocol.md.

BENCHMARK_DEFAULT_PROTOCOL: Dict[str, BenchmarkProtocol] = {
    "math500": BenchmarkProtocol(
        prompt_builder_id="math500_cot_v1",
        scorer_id="math500_boxed_v1",
        stop_tokens=["\n\nProblem:", "Problem:"],  # 来自 hendrycks_math lm-eval yaml
        fewshot=4,
        generation_kwargs={"max_new_tokens": 2048, "do_sample": False},
    ),
    "mmlu_redux": BenchmarkProtocol(
        prompt_builder_id="mmlu_standard_v1",   # 同 mmlu，4 选 1 标准 MC 格式
        scorer_id="first_letter_choice_v1",
        stop_tokens=["\n\n"],
        fewshot=5,                               # 5-shot random per-subject (seed=42)
        generation_kwargs={"max_new_tokens": 10, "do_sample": False},  # 基于 MAX_TOKENS_ANALYSIS.md: 99.9% 单字母输出
    ),
    "mmlu": BenchmarkProtocol(
        prompt_builder_id="mmlu_standard_v1",
        scorer_id="first_letter_choice_v1",
        stop_tokens=["\n\n"],   # 防止 base model 答完后续写下一道题
        fewshot=5,
        generation_kwargs={"max_new_tokens": 10, "do_sample": False},  # 基于 MAX_TOKENS_ANALYSIS.md: 100% 单字母输出
    ),
    "mmlu_pro": BenchmarkProtocol(
        prompt_builder_id="mmlu_pro_cot_v1",
        scorer_id="mmlu_pro_regex_v1",
        stop_tokens=["Question:"],
        fewshot=5,
        generation_kwargs={"max_new_tokens": 2048, "do_sample": False},
    ),
    "bbh": BenchmarkProtocol(
        prompt_builder_id="bbh_cot_v1",
        scorer_id="bbh_cot_extract_v1",
        stop_tokens=["Q", "\n\n"],
        fewshot=3,
        generation_kwargs={"max_new_tokens": 1024, "do_sample": False},
    ),
    "gsm8k": BenchmarkProtocol(
        prompt_builder_id="gsm8k_cot_v1",
        scorer_id="gsm8k_flexible_number_v1",
        stop_tokens=["Question:"],
        fewshot=5,
        generation_kwargs={"max_new_tokens": 512, "do_sample": False},
    ),
    "ceval": BenchmarkProtocol(
        prompt_builder_id="ceval_standard_v1",
        scorer_id="first_letter_choice_v1",
        stop_tokens=["\n\n"],   # 同 mmlu，防止续写
        fewshot=5,
        generation_kwargs={"max_new_tokens": 8, "do_sample": False},  # 基于 MAX_TOKENS_ANALYSIS.md: 99.61% 单字母输出
    ),
    "cmmlu": BenchmarkProtocol(
        prompt_builder_id="cmmlu_standard_v1",
        scorer_id="first_letter_choice_v1",
        stop_tokens=["\n\n"],   # 同 mmlu，防止续写
        fewshot=5,
        generation_kwargs={"max_new_tokens": 6, "do_sample": False},  # 基于 MAX_TOKENS_ANALYSIS.md: 98.2% 单字母输出
    ),
    "cruxeval": BenchmarkProtocol(
        prompt_builder_id="cruxeval_output_v1",
        scorer_id="cruxeval_exact_match_v1",
        # [END ANSWER] is the primary stop; BASE_MODEL_EXTRA_STOPS ("\n[" etc.)
        # also fire when a base model tries to continue writing the next example.
        stop_tokens=["[END ANSWER]"],
        fewshot=2,
        generation_kwargs={"max_new_tokens": 128, "do_sample": False},
        fewshot_examples=[
            # sample_3: simple string concat → string output
            {
                "question": (
                    "def f(text, value):\n"
                    "    text_list = list(text)\n"
                    "    text_list.append(value)\n"
                    "    return ''.join(text_list)\n\n"
                    "assert f('bcksrut', 'q') == ??"
                ),
                "target": "'bcksrutq'",
            },
            # sample_17: str.find → int output
            {
                "question": (
                    "def f(text):\n"
                    '    return text.find(",")\n\n'
                    'assert f("There are, no, commas, in this text") == ??'
                ),
                "target": "9",
            },
        ],
    ),
}


# ── Lookups (fail-fast) ───────────────────────────────────────────────────────

def get_protocol(
    benchmark_id: str, override: Optional[Dict[str, Any]] = None
) -> BenchmarkProtocol:
    """Return the default protocol for *benchmark_id*, with *override* applied.

    Raises:
        KeyError: if *benchmark_id* has no entry in BENCHMARK_DEFAULT_PROTOCOL.
    """
    if benchmark_id not in BENCHMARK_DEFAULT_PROTOCOL:
        raise KeyError(
            f"No default protocol for benchmark_id={benchmark_id!r}. "
            f"Known benchmarks: {sorted(BENCHMARK_DEFAULT_PROTOCOL)}"
        )
    return BENCHMARK_DEFAULT_PROTOCOL[benchmark_id].merged_with(override)


def get_prompt_builder(prompt_builder_id: str) -> PromptBuilder:
    """Return the prompt builder for *prompt_builder_id* (fail-fast)."""
    if prompt_builder_id not in PROMPT_BUILDER_CATALOG:
        raise KeyError(
            f"Unknown prompt_builder_id={prompt_builder_id!r}. "
            f"Known: {sorted(PROMPT_BUILDER_CATALOG)}"
        )
    return PROMPT_BUILDER_CATALOG[prompt_builder_id]


def get_scorer(scorer_id: str) -> Scorer:
    """Return the scorer for *scorer_id* (fail-fast)."""
    if scorer_id not in SCORER_CATALOG:
        raise KeyError(
            f"Unknown scorer_id={scorer_id!r}. Known: {sorted(SCORER_CATALOG)}"
        )
    return SCORER_CATALOG[scorer_id]
