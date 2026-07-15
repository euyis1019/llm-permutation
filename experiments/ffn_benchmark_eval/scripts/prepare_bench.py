"""Prepare the frozen benchmark selection for the equivalence experiment.

1. Rewrite the stale /mnt/... data_path in each used benchmark_meta.json to the
   local hard-copied absolute path.
2. Select the pre-registered 500-sample subsets:
   - mmlu / ceval / cmmlu : deterministic per-subject stratified (largest
     remainder allocation, then first-k by sorted sample_id within subject).
   - gsm8k : fixed 500 sample IDs via a seeded permutation of sorted IDs.
   HumanEval+ / MBPP+ use the full set (no selection file).
3. Write per-benchmark selection JSONL + a selection manifest with SHA-256.

Deterministic and idempotent: re-running produces byte-identical outputs.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import common

N = 500


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def largest_remainder(counts: dict, total: int) -> dict:
    """Allocate `total` across keys proportional to counts (largest remainder)."""
    grand = sum(counts.values())
    raw = {k: counts[k] / grand * total for k in counts}
    floor = {k: int(raw[k]) for k in counts}
    used = sum(floor.values())
    rem = total - used
    # distribute leftover to largest fractional parts (ties -> sorted key)
    order = sorted(counts, key=lambda k: (-(raw[k] - floor[k]), k))
    for k in order[:rem]:
        floor[k] += 1
    # never allocate more than a subject actually has
    for k in counts:
        floor[k] = min(floor[k], counts[k])
    return floor


def fix_data_path(benchmark_id: str) -> str:
    meta_p = common.meta_path(benchmark_id)
    meta = json.loads(meta_p.read_text())
    local = str(common.normalized_dir(benchmark_id) / f"{benchmark_id}.jsonl")
    changed = False
    for bd in meta["bench_data"]:
        if bd["data_path"] != local:
            bd["data_path"] = local
            changed = True
    if changed:
        common.atomic_write_json(meta_p, meta)
    assert Path(local).is_file(), local
    return local


def select_stratified(benchmark_id: str, local_jsonl: str) -> dict:
    rows = [json.loads(l) for l in open(local_jsonl, encoding="utf-8") if l.strip()]
    by_sub = defaultdict(list)
    for r in rows:
        by_sub[r["metadata"].get("subject", "")].append(r)
    for s in by_sub:
        by_sub[s].sort(key=lambda r: r["sample_id"])
    counts = {s: len(by_sub[s]) for s in by_sub}
    alloc = largest_remainder(counts, N)
    selected = []
    for s in sorted(by_sub):
        selected.extend(by_sub[s][: alloc[s]])
    selected.sort(key=lambda r: r["sample_id"])
    assert len(selected) == N, (benchmark_id, len(selected))
    out = common.selected_jsonl_path(benchmark_id)
    with open(out, "w", encoding="utf-8") as f:
        for r in selected:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {
        "benchmark": benchmark_id,
        "method": "per-subject stratified (largest remainder)",
        "n_selected": len(selected),
        "n_subjects": len(by_sub),
        "alloc": {s: alloc[s] for s in sorted(alloc)},
        "selected_path": str(out),
        "sha256": sha256_file(out),
        "sample_ids_head": [r["sample_id"] for r in selected[:5]],
    }


def select_gsm8k(local_jsonl: str, seed: int) -> dict:
    rows = [json.loads(l) for l in open(local_jsonl, encoding="utf-8") if l.strip()]
    rows.sort(key=lambda r: r["sample_id"])
    ids = [r["sample_id"] for r in rows]
    rng = random.Random(seed)
    chosen = set(rng.sample(ids, N))
    selected = [r for r in rows if r["sample_id"] in chosen]
    selected.sort(key=lambda r: r["sample_id"])
    assert len(selected) == N
    out = common.selected_jsonl_path("gsm8k")
    with open(out, "w", encoding="utf-8") as f:
        for r in selected:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {
        "benchmark": "gsm8k",
        "method": f"fixed sample IDs (Random({seed}).sample over sorted IDs)",
        "n_selected": len(selected),
        "selected_path": str(out),
        "sha256": sha256_file(out),
        "sample_ids_head": [r["sample_id"] for r in selected[:5]],
    }


def main() -> None:
    cfg = common.load_config()
    seed = cfg["sample_selection"]["selection_seed"]
    manifest = {"selection_seed": seed, "N": N, "benchmarks": {}}

    for b in ["mmlu", "ceval", "cmmlu"]:
        local = fix_data_path(b)
        info = select_stratified(b, local)
        manifest["benchmarks"][b] = info
        print(f"[{b}] selected {info['n_selected']} over {info['n_subjects']} subjects  sha={info['sha256'][:12]}")

    local = fix_data_path("gsm8k")
    info = select_gsm8k(local, seed)
    manifest["benchmarks"]["gsm8k"] = info
    print(f"[gsm8k] selected {info['n_selected']}  sha={info['sha256'][:12]}")

    out = common.EXP_ROOT / "configs" / "sample_selection_manifest.json"
    common.atomic_write_json(out, manifest)
    print(f"manifest -> {out}")


if __name__ == "__main__":
    main()
