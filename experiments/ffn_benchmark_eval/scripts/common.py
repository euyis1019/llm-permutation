"""Shared helpers for the FFN benchmark equivalence experiment.

All prompt-building and scoring logic is imported verbatim from the hard-copied
`bench/` tree (experiments/ffn_benchmark_eval/bench), so the evaluation protocol
is byte-identical to the upstream benchmark contract.  Nothing here reads the
original /nvme0/if/llm-brewing/bench tree at runtime.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

EXP_ROOT = Path(__file__).resolve().parents[1]
BENCH_ROOT = EXP_ROOT / "bench"
CONFIG_PATH = EXP_ROOT / "configs" / "frozen_config.json"

# Make the copied bench importable (import from the copy only).
if str(BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCH_ROOT))

from src.eval.benchmark.behavior_catalog import (  # noqa: E402
    BASE_MODEL_EXTRA_STOPS,
    get_protocol,
)
from src.eval.benchmark.client_runner import (  # noqa: E402
    build_prompt,
    score_response,
)
from src.eval.benchmark.models import (  # noqa: E402
    BenchData,
    BenchmarkMeta,
    BenchmarkProtocol,
    EvalRow,
)

PROTOCOL_BENCHES = ["mmlu", "gsm8k", "ceval", "cmmlu"]
CODE_BENCHES = ["humaneval_plus", "mbpp_plus"]
ALL_BENCHES = PROTOCOL_BENCHES + CODE_BENCHES


def load_config() -> Dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text())


def normalized_dir(benchmark_id: str) -> Path:
    return BENCH_ROOT / "datasets" / "benchmark" / "normalized" / benchmark_id


def meta_path(benchmark_id: str) -> Path:
    return normalized_dir(benchmark_id) / "benchmark_meta.json"


def selected_jsonl_path(benchmark_id: str) -> Path:
    """Path to the frozen 500-sample selection for a protocol benchmark."""
    return normalized_dir(benchmark_id) / f"{benchmark_id}.selected500.jsonl"


def resolve_bench(benchmark_id: str, use_selection: bool = True) -> Tuple[BenchData, BenchmarkProtocol]:
    """Return (BenchData bound to the frozen selection, protocol)."""
    meta = BenchmarkMeta.load(str(meta_path(benchmark_id)))
    assert len(meta.bench_data) == 1, benchmark_id
    bench_data = meta.bench_data[0]
    if use_selection and benchmark_id in PROTOCOL_BENCHES:
        import copy as _copy
        bench_data = _copy.copy(bench_data)
        bench_data.data_path = str(selected_jsonl_path(benchmark_id))
    protocol = get_protocol(meta.benchmark_id, bench_data.protocol_override)
    return bench_data, protocol


def load_rows(data_path: str) -> List[EvalRow]:
    rows: List[EvalRow] = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(EvalRow.from_dict(json.loads(line)))
    return rows


def effective_stop(protocol: BenchmarkProtocol) -> List[str]:
    return list(protocol.stop_tokens) + BASE_MODEL_EXTRA_STOPS


def atomic_write_json(path: str | os.PathLike, obj) -> None:
    import tempfile
    path = str(path)
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
