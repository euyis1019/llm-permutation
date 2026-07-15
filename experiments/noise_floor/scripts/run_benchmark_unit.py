"""Resource-disciplined wrapper for one frozen benchmark unit."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from run_logits_unit import acquire_resources, append_decision


EXP_ROOT = Path(__file__).resolve().parents[1]
BENCH_EXP = EXP_ROOT.parent / "ffn_benchmark_eval"


def complete(path: Path) -> bool:
    try:
        return json.loads(path.read_text()).get("complete") is True
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--model-tag", required=True)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workdir", default=str(EXP_ROOT / "logs" / "code_scratch"))
    ap.add_argument("--decisions", default=str(EXP_ROOT / "DECISIONS.md"))
    ap.add_argument("--progress", default=str(EXP_ROOT / "PROGRESS.md"))
    args = ap.parse_args()

    raw = Path(args.out_dir) / args.model_tag / f"{args.benchmark}.raw.json"
    if complete(raw):
        print(f"[benchmark-unit] {args.model_tag}/{args.benchmark}: complete; skipping")
        return 0
    unit = f"{args.model_tag}/{args.benchmark}"
    attempts = 0
    while True:
        resource = acquire_resources(unit, Path(args.decisions), Path(args.progress))
        if resource is None:
            return 75
        attempts += 1
        log = EXP_ROOT / "logs" / "benchmark" / args.model_tag / f"{args.benchmark}.attempt{attempts}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "/nvme0/if/anaconda3/envs/qwen3/bin/python",
            str(BENCH_EXP / "scripts" / "run_worker.py"),
            "--model-path", str(Path(args.model_path).resolve()),
            "--model-tag", args.model_tag,
            "--benchmarks", args.benchmark,
            "--out-dir", str(Path(args.out_dir).resolve()),
            "--workdir", str(Path(args.workdir).resolve()),
            "--gpu", "0",
            "--gpu-mem-util", str(resource["gpu_memory_utilization"]),
            "--max-model-len", str(resource["max_model_len"]),
            "--max-num-seqs", str(resource["max_num_seqs"]),
            "--engine-init-attempts", "1",
        ]
        env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "0",
            "VLLM_BATCH_INVARIANT": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
        with open(log, "w") as f:
            proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
        if proc.returncode == 0 and complete(raw):
            print(f"[benchmark-unit] {unit}: complete (attempt {attempts})")
            return 0
        tail = log.read_text(errors="replace")[-12000:]
        if "out of memory" in tail.lower() or "oom" in tail.lower():
            append_decision(
                Path(args.decisions),
                f"- {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} `{unit}`: "
                f"attempt {attempts} 运行中 OOM；不计判据失败，按 §1 等待 600 秒后以相同科学参数重试。",
            )
            time.sleep(600)
            continue
        print(tail, file=sys.stderr)
        raise RuntimeError(f"benchmark unit failed (not OOM): {unit}; log={log}")


if __name__ == "__main__":
    raise SystemExit(main())

