#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Normalize CRUXEval raw data to EvalRow format.

Input:
    datasets/benchmark/raw/code/CRUXEval/test_only/test.jsonl
    Schema: {code, input, output, id}

Output:
    datasets/benchmark/normalized/cruxeval/cruxeval.jsonl
    datasets/benchmark/normalized/cruxeval/benchmark_meta.json

EvalRow mapping:
    sample_id  : cruxeval__NNNNN  (zero-padded)
    bench_id   : "cruxeval"
    question   : "{code}\\n\\nassert f({input}) == ??"
    choices    : null  (generative task, not MC)
    target     : raw["output"]  (Python repr string, e.g. "'bcksrutq'", "9")
    metadata   : {code, input, original_id, source_split, source_row}

Usage:
    python src/scripts/suite/normalize_cruxeval.py
    python src/scripts/suite/normalize_cruxeval.py --dry-run   # print first 3 rows
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from src.eval.benchmark.behavior_catalog import BENCHMARK_DEFAULT_PROTOCOL
from src.eval.benchmark.models import BenchData, BenchmarkMeta, EvalRow

# ── Paths ─────────────────────────────────────────────────────────────────────

RAW_PATH = REPO / "datasets" / "benchmark" / "raw" / "code" / "CRUXEval" / "test_only" / "test.jsonl"
OUT_DIR  = REPO / "datasets" / "benchmark" / "normalized" / "cruxeval"
BENCH_ID = "cruxeval"


# ── Core conversion ───────────────────────────────────────────────────────────

def _format_question(code: str, inp: str) -> str:
    """Build the question text for EvalRow from code and input strings.

    The prompt builder (``build_cruxeval_output_v1``) wraps this in
    ``[BEGIN PROBLEM] / [END PROBLEM]`` delimiters; the raw content is just the
    function definition followed by a ``assert f(input) == ??`` line.
    """
    return f"{code}\n\nassert f({inp}) == ??"


def normalize(raw_path: Path = RAW_PATH, dry_run: bool = False) -> list:
    """Load raw JSONL and return a list of EvalRow objects."""
    rows: list[EvalRow] = []
    with open(raw_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            row = EvalRow(
                sample_id=f"{BENCH_ID}__{i:05d}",
                bench_id=BENCH_ID,
                question=_format_question(raw["code"], raw["input"]),
                choices=None,
                target=raw["output"],
                metadata={
                    "code":         raw["code"],
                    "input":        raw["input"],
                    "original_id":  raw["id"],
                    "source_split": "test",
                    "source_row":   i,
                },
            )
            row.validate()
            rows.append(row)
            if dry_run and i >= 2:
                break
    return rows


# ── Writers ───────────────────────────────────────────────────────────────────

def write_jsonl(rows: list[EvalRow], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
    print(f"[normalize] wrote {len(rows)} rows → {out_path}")


def write_meta(rows: list[EvalRow], jsonl_path: Path, raw_path: Path) -> Path:
    """Generate and write benchmark_meta.json alongside the JSONL."""
    protocol = BENCHMARK_DEFAULT_PROTOCOL[BENCH_ID]

    selections = [
        {
            "sample_id":    row.sample_id,
            "source_file":  str(raw_path),
            "source_split": "test",
            "source_row":   row.metadata["source_row"],
            "original_id":  row.metadata["original_id"],
        }
        for row in rows
    ]

    bench_data = BenchData(
        bench_id=BENCH_ID,
        parent_benchmark=BENCH_ID,
        data_path=str(jsonl_path.resolve()),
        total_rows=len(rows),
        fewshot_examples=[],   # shared fewshot lives on the protocol, not per-row
        selection_rule=f"all {len(rows)} test rows from CRUXEval test_only split",
        selections=selections,
    )

    meta = BenchmarkMeta(
        benchmark_id=BENCH_ID,
        description=(
            "CRUXEval：Python 函数输出预测（output prediction）。"
            "给定函数定义和调用输入，预测函数返回值的 Python repr 字符串。"
            "800 条，2-shot，exact match 评分。"
            "原始论文：Gu et al., 2024 (arXiv:2401.03065)。"
        ),
        protocol=protocol,
        bench_data=[bench_data],
        created_at=datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        source_note=f"Normalized from {raw_path}",
    )

    meta_path = jsonl_path.parent / "benchmark_meta.json"
    meta.write(str(meta_path))
    print(f"[normalize] wrote benchmark_meta.json → {meta_path}")
    return meta_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="Print first 3 rows to stdout; do not write files.")
    ap.add_argument("--raw-path", default=str(RAW_PATH),
                    help=f"Path to raw test.jsonl (default: {RAW_PATH})")
    ap.add_argument("--out-dir", default=str(OUT_DIR),
                    help=f"Output directory (default: {OUT_DIR})")
    args = ap.parse_args()

    raw_path = Path(args.raw_path)
    out_dir  = Path(args.out_dir)

    if not raw_path.exists():
        print(f"[normalize] ERROR: raw file not found: {raw_path}", file=sys.stderr)
        sys.exit(1)

    rows = normalize(raw_path, dry_run=args.dry_run)

    if args.dry_run:
        print(f"\n[dry-run] first {len(rows)} rows (not writing):")
        for row in rows:
            d = row.to_dict()
            d["question"] = d["question"][:80] + "…"   # truncate for display
            print(json.dumps(d, ensure_ascii=False, indent=2))
        return

    jsonl_path = out_dir / f"{BENCH_ID}.jsonl"
    write_jsonl(rows, jsonl_path)
    write_meta(rows, jsonl_path, raw_path)
    print(f"[normalize] Done.  benchmark_id={BENCH_ID!r}  total={len(rows)}")


if __name__ == "__main__":
    main()
