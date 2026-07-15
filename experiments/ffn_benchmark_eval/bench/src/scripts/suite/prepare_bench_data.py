#!/usr/bin/env python3
"""Prepare standardised bench JSONL data for a benchmark suite.

This script converts raw benchmark_aligned data into executable bench JSONL
files under datasets/benchmark/normalized/{benchmark_name}/.  Each bench
contains exactly N rows selected by a static, reproducible rule (no runtime
randomness).

After writing JSONL files, the script also generates benchmark_meta.json
describing the benchmark hierarchy and leaf bench bindings.

Supported benchmarks
--------------------
  slim_qwen_test    6 leaf benches × 5 questions each

Usage
-----
  # From repo root:
  python src/scripts/suite/prepare_bench_data.py --benchmark slim_qwen_test

  # Custom output directory:
  python src/scripts/suite/prepare_bench_data.py \\
      --benchmark slim_qwen_test \\
      --output-dir /path/to/output

  # Prepare only specific benches:
  python src/scripts/suite/prepare_bench_data.py \\
      --benchmark slim_qwen_test \\
      --bench mmlu__slim5 mmlu_pro__slim5

  # Micro mode: 2 rows per bench (for acceptance testing):
  python src/scripts/suite/prepare_bench_data.py \\
      --benchmark slim_qwen_test \\
      --micro

  # Custom row count:
  python src/scripts/suite/prepare_bench_data.py \\
      --benchmark slim_qwen_test \\
      --nrows 10
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# ── Repo layout ───────────────────────────────────────────────────────────────

REPO = Path(__file__).resolve().parents[3]  # src/scripts/suite/ → src/ → repo
sys.path.insert(0, str(REPO))

from src.eval.benchmark.models import (
    SlimBenchManifest,
    SlimBenchRow,
    SlimSelectionEntry,
)
from src.eval.benchmark.bench_tree import (
    BenchmarkMeta,
    BenchmarkNode,
    LeafBench,
    flat_benchmark,
)

# Upstream raw benchmark-aligned exports — the input this preparer normalises into
# datasets/benchmark/normalized/. NOT shipped in this bundle (only the normalised
# output is). Override with --data-root or the BENCH_DATA_ROOT env var; to use the
# in-repo default, populate datasets/benchmark/raw/ with the upstream layout
# (mmlu/all__test.jsonl, mmlu_pro/test.jsonl, gsm8k/test.jsonl, …).
DEFAULT_DATA_ROOT = os.environ.get(
    "BENCH_DATA_ROOT", str(REPO / "datasets" / "benchmark" / "raw")
)
DEFAULT_NORM_ROOT = str(REPO / "datasets" / "benchmark" / "normalized")


# ── Static selection tables ───────────────────────────────────────────────────

BBH_SUBTASKS = [
    "boolean_expressions",
    "causal_judgement",
    "date_understanding",
    "logical_deduction_three_objects",
    "object_counting",
]

CEVAL_SUBJECTS = [
    "advanced_mathematics",
    "college_chemistry",
    "high_school_history",
    "law",
    "operating_system",
]

CMMLU_SUBJECTS = [
    "astronomy",
    "computer_science",
    "high_school_mathematics",
    "professional_law",
    "world_history",
]

MMLU_SUBJECTS_TOP5 = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: str, rows: List[dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_manifest(manifest: SlimBenchManifest, output_dir: str) -> str:
    path = os.path.join(output_dir, f"{manifest.bench}.manifest.json")
    manifest.write(path)
    return path


# ── Bench preparers ───────────────────────────────────────────────────────────

def prepare_mmlu(
    data_root: str,
    output_dir: str,
    created_at: Optional[str],
    nrows: int = 5,
) -> LeafBench:
    """Prepare mmlu__slim5: first row from each of 5 fixed subjects."""
    bench_id = "mmlu__slim5"
    src_path = os.path.join(data_root, "mmlu", "all__test.jsonl")
    all_rows = _load_jsonl(src_path)

    by_subject: dict = {}
    for i, r in enumerate(all_rows):
        subj = r["subject"]
        if subj not in by_subject:
            by_subject[subj] = []
        by_subject[subj].append((i, r))

    selected_rows: List[dict] = []
    selections: List[SlimSelectionEntry] = []

    subjects = MMLU_SUBJECTS_TOP5[:nrows]
    for subj in subjects:
        if subj not in by_subject:
            raise RuntimeError(f"MMLU subject not found: {subj!r}")
        src_row_idx, r = by_subject[subj][0]
        choices = r["choices"]
        answer_letter = "ABCD"[r["answer"]]
        slim_id = f"{bench_id}_{subj}_r0"
        slim_row = SlimBenchRow(
            sample_id=slim_id,
            bench=bench_id,
            parent_bench="mmlu",
            task_type="multiple_choice",
            prompt_fields={"question": r["question"], "choices": choices},
            target=answer_letter,
            source_path=src_path,
            source_split="test",
            source_row=src_row_idx,
            metadata={"subject": subj},
        )
        slim_row.validate()
        selected_rows.append(slim_row.to_dict())
        selections.append(SlimSelectionEntry(
            source_file=src_path, source_split="test",
            source_row=src_row_idx, subject=subj, sample_id=slim_id,
        ))

    out_path = os.path.join(output_dir, f"{bench_id}.jsonl")
    _write_jsonl(out_path, selected_rows)
    manifest = SlimBenchManifest(
        bench=bench_id, parent_bench="mmlu", task_type="multiple_choice",
        fewshot=5, total_rows=len(selected_rows),
        selection_rule=(
            f"First row (row 0) from each of the following {len(subjects)} "
            f"subjects (alphabetically first from all__test.jsonl): "
            + ", ".join(subjects)
        ),
        selections=selections,
        data_path=os.path.abspath(out_path),
        created_at=created_at,
    )
    _write_manifest(manifest, output_dir)
    print(f"  [mmlu__slim5]     → {out_path}  ({len(selected_rows)} rows)")
    return LeafBench(
        bench_id=bench_id, parent_bench="mmlu", task_type="multiple_choice",
        fewshot=5, data_path=os.path.abspath(out_path),
        metadata={"selection_rule": manifest.selection_rule},
    )


def prepare_mmlu_pro(
    data_root: str,
    output_dir: str,
    created_at: Optional[str],
    nrows: int = 5,
) -> LeafBench:
    """Prepare mmlu_pro__slim5: first nrows rows from test set."""
    bench_id = "mmlu_pro__slim5"
    src_path = os.path.join(data_root, "mmlu_pro", "test.jsonl")
    all_rows = _load_jsonl(src_path)

    selected_rows: List[dict] = []
    selections: List[SlimSelectionEntry] = []

    for i in range(nrows):
        r = all_rows[i]
        options = r["options"]
        answer_letter = r["answer"]
        slim_id = f"{bench_id}_{i:04d}"
        slim_row = SlimBenchRow(
            sample_id=slim_id, bench=bench_id, parent_bench="mmlu_pro",
            task_type="multiple_choice",
            prompt_fields={"question": r["question"], "choices": options},
            target=answer_letter,
            source_path=src_path, source_split="test", source_row=i,
            metadata={"category": r.get("category", ""), "question_id": r.get("question_id", i)},
        )
        slim_row.validate()
        selected_rows.append(slim_row.to_dict())
        selections.append(SlimSelectionEntry(
            source_file=src_path, source_split="test",
            source_row=i, subject=r.get("category"), sample_id=slim_id,
        ))

    out_path = os.path.join(output_dir, f"{bench_id}.jsonl")
    _write_jsonl(out_path, selected_rows)
    rule = f"First {nrows} rows from mmlu_pro/test.jsonl (rows 0–{nrows-1})"
    manifest = SlimBenchManifest(
        bench=bench_id, parent_bench="mmlu_pro", task_type="multiple_choice",
        fewshot=5, total_rows=len(selected_rows), selection_rule=rule,
        selections=selections, data_path=os.path.abspath(out_path), created_at=created_at,
    )
    _write_manifest(manifest, output_dir)
    print(f"  [mmlu_pro__slim5] → {out_path}  ({len(selected_rows)} rows)")
    return LeafBench(
        bench_id=bench_id, parent_bench="mmlu_pro", task_type="multiple_choice",
        fewshot=5, data_path=os.path.abspath(out_path),
        metadata={"selection_rule": rule},
    )


def prepare_bbh(
    data_root: str,
    output_dir: str,
    created_at: Optional[str],
    nrows: int = 5,
) -> LeafBench:
    """Prepare bbh__slim5: row 0 from each of 5 fixed subtasks."""
    bench_id = "bbh__slim5"
    subtasks = BBH_SUBTASKS[:nrows]

    selected_rows: List[dict] = []
    selections: List[SlimSelectionEntry] = []

    for subtask in subtasks:
        src_path = os.path.join(data_root, "bbh", f"{subtask}__test.jsonl")
        if not os.path.isfile(src_path):
            raise FileNotFoundError(f"BBH subtask file not found: {src_path}")
        all_rows = _load_jsonl(src_path)
        r = all_rows[0]
        slim_id = f"{bench_id}_{subtask}_r0"
        slim_row = SlimBenchRow(
            sample_id=slim_id, bench=bench_id, parent_bench="bbh",
            task_type="generate",
            prompt_fields={"question": r["input"], "subtask": subtask},
            target=str(r["target"]),
            source_path=src_path, source_split="test", source_row=0,
            metadata={"subtask": subtask},
        )
        slim_row.validate()
        selected_rows.append(slim_row.to_dict())
        selections.append(SlimSelectionEntry(
            source_file=src_path, source_split="test",
            source_row=0, subject=subtask, sample_id=slim_id,
        ))

    out_path = os.path.join(output_dir, f"{bench_id}.jsonl")
    _write_jsonl(out_path, selected_rows)
    rule = (
        f"Row 0 from each of the following {len(subtasks)} fixed BBH subtasks: "
        + ", ".join(subtasks)
    )
    manifest = SlimBenchManifest(
        bench=bench_id, parent_bench="bbh", task_type="generate",
        fewshot=3, total_rows=len(selected_rows), selection_rule=rule,
        selections=selections, data_path=os.path.abspath(out_path), created_at=created_at,
    )
    _write_manifest(manifest, output_dir)
    print(f"  [bbh__slim5]      → {out_path}  ({len(selected_rows)} rows)")
    return LeafBench(
        bench_id=bench_id, parent_bench="bbh", task_type="generate",
        fewshot=3, data_path=os.path.abspath(out_path),
        generation_kwargs={"max_new_tokens": 1024, "do_sample": False},
        metadata={"selection_rule": rule},
    )


def prepare_gsm8k(
    data_root: str,
    output_dir: str,
    created_at: Optional[str],
    nrows: int = 5,
) -> LeafBench:
    """Prepare gsm8k__slim5: first nrows rows from test set."""
    bench_id = "gsm8k__slim5"
    src_path = os.path.join(data_root, "gsm8k", "test.jsonl")
    all_rows = _load_jsonl(src_path)

    selected_rows: List[dict] = []
    selections: List[SlimSelectionEntry] = []

    for i in range(nrows):
        r = all_rows[i]
        answer_str = r["answer"]
        if "####" in answer_str:
            numeric_answer = answer_str.split("####")[-1].strip()
        else:
            numeric_answer = answer_str.strip()
        slim_id = f"{bench_id}_{i:04d}"
        slim_row = SlimBenchRow(
            sample_id=slim_id, bench=bench_id, parent_bench="gsm8k",
            task_type="generate",
            prompt_fields={"question": r["question"], "full_answer": r["answer"]},
            target=numeric_answer,
            source_path=src_path, source_split="test", source_row=i,
            metadata={"flexible_extract": True},
        )
        slim_row.validate()
        selected_rows.append(slim_row.to_dict())
        selections.append(SlimSelectionEntry(
            source_file=src_path, source_split="test",
            source_row=i, subject=None, sample_id=slim_id,
        ))

    out_path = os.path.join(output_dir, f"{bench_id}.jsonl")
    _write_jsonl(out_path, selected_rows)
    rule = f"First {nrows} rows from gsm8k/test.jsonl (rows 0–{nrows-1})"
    manifest = SlimBenchManifest(
        bench=bench_id, parent_bench="gsm8k", task_type="generate",
        fewshot=0, total_rows=len(selected_rows), selection_rule=rule,
        selections=selections, data_path=os.path.abspath(out_path), created_at=created_at,
    )
    _write_manifest(manifest, output_dir)
    print(f"  [gsm8k__slim5]    → {out_path}  ({len(selected_rows)} rows)")
    return LeafBench(
        bench_id=bench_id, parent_bench="gsm8k", task_type="generate",
        fewshot=0, data_path=os.path.abspath(out_path),
        generation_kwargs={"max_new_tokens": 512, "do_sample": False},
        metadata={"selection_rule": rule},
    )


def prepare_ceval(
    data_root: str,
    output_dir: str,
    created_at: Optional[str],
    nrows: int = 5,
) -> LeafBench:
    """Prepare ceval__slim5: row 0 from each of 5 fixed subjects."""
    bench_id = "ceval__slim5"
    subjects = CEVAL_SUBJECTS[:nrows]

    selected_rows: List[dict] = []
    selections: List[SlimSelectionEntry] = []

    for subj in subjects:
        src_path = os.path.join(data_root, "ceval", f"{subj}__val.jsonl")
        if not os.path.isfile(src_path):
            raise FileNotFoundError(f"C-Eval subject file not found: {src_path}")
        all_rows = _load_jsonl(src_path)
        r = all_rows[0]
        choices = [r["A"], r["B"], r["C"], r["D"]]
        slim_id = f"{bench_id}_{subj}_r0"
        slim_row = SlimBenchRow(
            sample_id=slim_id, bench=bench_id, parent_bench="ceval",
            task_type="multiple_choice",
            prompt_fields={"question": r["question"], "choices": choices},
            target=r["answer"],
            source_path=src_path, source_split="val", source_row=0,
            metadata={"subject": subj},
        )
        slim_row.validate()
        selected_rows.append(slim_row.to_dict())
        selections.append(SlimSelectionEntry(
            source_file=src_path, source_split="val",
            source_row=0, subject=subj, sample_id=slim_id,
        ))

    out_path = os.path.join(output_dir, f"{bench_id}.jsonl")
    _write_jsonl(out_path, selected_rows)
    rule = (
        f"Row 0 from each of the following {len(subjects)} fixed C-Eval subjects: "
        + ", ".join(subjects)
    )
    manifest = SlimBenchManifest(
        bench=bench_id, parent_bench="ceval", task_type="multiple_choice",
        fewshot=5, total_rows=len(selected_rows), selection_rule=rule,
        selections=selections, data_path=os.path.abspath(out_path), created_at=created_at,
    )
    _write_manifest(manifest, output_dir)
    print(f"  [ceval__slim5]    → {out_path}  ({len(selected_rows)} rows)")
    return LeafBench(
        bench_id=bench_id, parent_bench="ceval", task_type="multiple_choice",
        fewshot=5, data_path=os.path.abspath(out_path),
        metadata={"selection_rule": rule},
    )


def prepare_cmmlu(
    data_root: str,
    output_dir: str,
    created_at: Optional[str],
    nrows: int = 5,
) -> LeafBench:
    """Prepare cmmlu__slim5: row 0 from each of 5 fixed subjects (CSV)."""
    bench_id = "cmmlu__slim5"
    subjects = CMMLU_SUBJECTS[:nrows]

    selected_rows: List[dict] = []
    selections: List[SlimSelectionEntry] = []

    for subj in subjects:
        src_path = os.path.join(data_root, "cmmlu", "raw", "test", f"{subj}.csv")
        if not os.path.isfile(src_path):
            raise FileNotFoundError(f"CMMLU subject CSV not found: {src_path}")
        with open(src_path, encoding="utf-8") as f:
            reader = csv.reader(f)
            csv_rows = list(reader)
        if len(csv_rows) < 2:
            raise RuntimeError(f"CMMLU CSV has no data rows: {src_path}")
        header = csv_rows[0]
        data_row = csv_rows[1]
        h = [c.strip() for c in header]
        row_d = dict(zip(h, data_row))
        question = row_d.get("Question", "")
        choices = [row_d.get(c, "") for c in ["A", "B", "C", "D"]]
        answer = row_d.get("Answer", "").strip()
        slim_id = f"{bench_id}_{subj}_r0"
        slim_row = SlimBenchRow(
            sample_id=slim_id, bench=bench_id, parent_bench="cmmlu",
            task_type="multiple_choice",
            prompt_fields={"question": question, "choices": choices},
            target=answer,
            source_path=src_path, source_split="test", source_row=0,
            metadata={"subject": subj},
        )
        slim_row.validate()
        selected_rows.append(slim_row.to_dict())
        selections.append(SlimSelectionEntry(
            source_file=src_path, source_split="test",
            source_row=0, subject=subj, sample_id=slim_id,
        ))

    out_path = os.path.join(output_dir, f"{bench_id}.jsonl")
    _write_jsonl(out_path, selected_rows)
    rule = (
        f"Row 0 (first data row after CSV header) from each of the "
        f"following {len(subjects)} fixed CMMLU subjects (raw/test/*.csv): "
        + ", ".join(subjects)
    )
    manifest = SlimBenchManifest(
        bench=bench_id, parent_bench="cmmlu", task_type="multiple_choice",
        fewshot=5, total_rows=len(selected_rows), selection_rule=rule,
        selections=selections, data_path=os.path.abspath(out_path), created_at=created_at,
    )
    _write_manifest(manifest, output_dir)
    print(f"  [cmmlu__slim5]    → {out_path}  ({len(selected_rows)} rows)")
    return LeafBench(
        bench_id=bench_id, parent_bench="cmmlu", task_type="multiple_choice",
        fewshot=5, data_path=os.path.abspath(out_path),
        metadata={"selection_rule": rule},
    )


# ── Benchmark registry ────────────────────────────────────────────────────────

SLIM_QWEN_TEST_BENCHES = [
    "mmlu__slim5",
    "mmlu_pro__slim5",
    "bbh__slim5",
    "gsm8k__slim5",
    "ceval__slim5",
    "cmmlu__slim5",
]

SLIM_QWEN_TEST_PREPARERS = {
    "mmlu__slim5":     prepare_mmlu,
    "mmlu_pro__slim5": prepare_mmlu_pro,
    "bbh__slim5":      prepare_bbh,
    "gsm8k__slim5":    prepare_gsm8k,
    "ceval__slim5":    prepare_ceval,
    "cmmlu__slim5":    prepare_cmmlu,
}

BENCHMARK_REGISTRY = {
    "slim_qwen_test": {
        "benches": SLIM_QWEN_TEST_BENCHES,
        "preparers": SLIM_QWEN_TEST_PREPARERS,
        "description": (
            "slim_qwen_test: minimal 6-bench suite for fast Qwen model evaluation. "
            "6 benches × 5 questions = 30 questions total."
        ),
        "source_note": (
            "Data normalised from the upstream benchmark-aligned exports "
            "(see src/scripts/suite/prepare_bench_data.py --data-root); "
            "raw inputs are not shipped in this bundle."
        ),
    },
}


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--benchmark",
        required=True,
        choices=list(BENCHMARK_REGISTRY.keys()),
        help="Which benchmark to prepare.",
    )
    ap.add_argument(
        "--data-root",
        default=DEFAULT_DATA_ROOT,
        help="Path to benchmark_aligned/exports/ directory.",
    )
    ap.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory to write normalised bench JSONL files. "
            "Default: datasets/benchmark/normalized/{benchmark}/"
        ),
    )
    ap.add_argument(
        "--created-at",
        default=None,
        help="ISO-8601 timestamp to embed in manifests (default: now).",
    )
    ap.add_argument(
        "--bench",
        nargs="+",
        default=None,
        help="Which benches to prepare (default: all for the benchmark).",
    )
    ap.add_argument(
        "--micro",
        action="store_true",
        help="Micro-test mode: produce 2 rows per bench instead of the default.",
    )
    ap.add_argument(
        "--nrows",
        type=int,
        default=None,
        help="Number of rows per bench (overrides default and --micro).",
    )
    args = ap.parse_args()

    created_at = args.created_at or datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    nrows = args.nrows if args.nrows is not None else (2 if args.micro else 5)

    bench_reg = BENCHMARK_REGISTRY[args.benchmark]
    all_benches = bench_reg["benches"]
    preparers = bench_reg["preparers"]

    bench_list = args.bench if args.bench else all_benches
    # Validate bench names
    unknown = [b for b in bench_list if b not in preparers]
    if unknown:
        ap.error(f"Unknown bench(es) for {args.benchmark!r}: {unknown}")

    output_dir = args.output_dir or os.path.join(
        DEFAULT_NORM_ROOT, args.benchmark
    )
    os.makedirs(output_dir, exist_ok=True)

    if args.micro:
        print(f"[prepare] ── MICRO mode: {nrows} rows per bench ──")
    print(f"[prepare] benchmark  = {args.benchmark}")
    print(f"[prepare] data_root  = {args.data_root}")
    print(f"[prepare] output_dir = {output_dir}")
    print(f"[prepare] benches    = {bench_list}")
    print()

    leaf_benches = []
    for bench_id in bench_list:
        preparer = preparers[bench_id]
        lb = preparer(
            data_root=args.data_root,
            output_dir=output_dir,
            created_at=created_at,
            nrows=nrows,
        )
        leaf_benches.append(lb)

    # Regenerate benchmark_meta.json if all benches were prepared
    if set(bench_list) == set(all_benches):
        root = flat_benchmark(
            name=args.benchmark,
            leaf_benches=leaf_benches,
            description=bench_reg["description"],
        )
        meta = BenchmarkMeta(
            benchmark_name=args.benchmark,
            description=bench_reg["description"],
            root=root,
            created_at=created_at,
            source_note=bench_reg["source_note"],
        )
        meta_path = os.path.join(output_dir, "benchmark_meta.json")
        meta.write(meta_path)
        print(f"\n[prepare] benchmark_meta.json → {meta_path}")
    else:
        print(
            f"\n[prepare] Partial run ({len(bench_list)}/{len(all_benches)} benches). "
            "benchmark_meta.json NOT regenerated (run all benches to update it)."
        )

    print(f"\n[prepare] Done. {len(bench_list)} bench(es) written to {output_dir}/")


if __name__ == "__main__":
    main()
