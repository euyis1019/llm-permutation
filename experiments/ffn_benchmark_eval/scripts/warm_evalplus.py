"""Pre-warm the evalplus cache single-process.

evalplus.evaluate lazily builds an expected-output .pkl (ground-truth execution)
and a sanitized-mbpp.json in ~/.cache/evalplus on first use.  When two workers
score code benchmarks concurrently they race on those writes and corrupt them.
Running one full evaluate per dataset here (with canonical solutions) populates
every cache artifact so all later parallel workers only read them.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import common

EVALPLUS_DATA = common.BENCH_ROOT / "datasets" / "benchmark" / "evalplus"
os.environ["HUMANEVAL_OVERRIDE_PATH"] = str(EVALPLUS_DATA / "HumanEvalPlus-v0.1.10.jsonl")
os.environ["MBPP_OVERRIDE_PATH"] = str(EVALPLUS_DATA / "MbppPlus-v0.2.0.jsonl")


def warm(dataset: str) -> None:
    from evalplus.data import get_human_eval_plus, get_mbpp_plus
    problems = get_human_eval_plus() if dataset == "humaneval" else get_mbpp_plus()
    d = tempfile.mkdtemp(prefix=f"warm_{dataset}_")
    samples = Path(d) / "samples.jsonl"
    with open(samples, "w") as f:
        for t in problems:
            f.write(json.dumps({"task_id": t,
                                "solution": problems[t]["prompt"] + problems[t]["canonical_solution"]}) + "\n")
    proc = subprocess.run(
        [sys.executable, "-m", "evalplus.evaluate", "--dataset", dataset,
         "--samples", str(samples), "--parallel", "16"],
        env=os.environ, capture_output=True, text=True,
    )
    tail = proc.stdout.strip().splitlines()[-4:]
    print(f"[warm {dataset}] rc={proc.returncode}: {' | '.join(tail)}")
    if proc.returncode != 0:
        print(proc.stderr[-1500:])
        raise SystemExit(1)


if __name__ == "__main__":
    warm("humaneval")
    warm("mbpp")
    print("evalplus cache warmed.")
