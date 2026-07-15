"""Post-hoc supplementary arms S1/S2/S3 (reviewer-initiated, not pre-registered).

S1 sigma-left-edge : scope=all, sigma in {1e-8, 1e-7, 3e-7} x 3 reps, logits units.
S2 scope-matched   : scope=ffn, sigma in {1e-6, 1e-5, 1e-4} x 3 reps, logits units.
S3 behavior at RandOpt sigma : scope=all, sigma in {1e-4, 1e-3} x 5 reps, GSM8K-500.

Seeds: S1 3000+10*idx+rep, S2 4000+10*idx+rep, S3 5000+100*idx+rep.
Reuses the frozen run_logits_unit.py / run_benchmark_unit.py untouched
(same resource discipline, same 32 frozen prompts, one temp checkpoint at a time).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

EXP_ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/nvme0/if/anaconda3/envs/qwen3/bin/python"
SOURCE = "/nvme0/if/models/Qwen3-4B-Base"
TOKENIZED = EXP_ROOT.parent / "ffn_permutation" / "results_base" / "tokenized.json"

S1 = [1e-8, 1e-7, 3e-7]
S2 = [1e-6, 1e-5, 1e-4]
S3 = [1e-4, 1e-3]


def complete(path: Path) -> bool:
    try:
        return json.loads(path.read_text()).get("complete") is True
    except Exception:
        return False


def make_ckpt(temp: Path, sigma: float, seed: int, tag: str, scope: str, stats_copy: Path):
    if temp.exists():
        shutil.rmtree(temp)
    subprocess.run(
        [PYTHON, str(EXP_ROOT / "scripts" / "supp_make_noise_checkpoint.py"),
         "--source", SOURCE, "--out-dir", str(temp),
         "--sigma", repr(sigma), "--seed", str(seed), "--tag", tag, "--scope", scope],
        check=True, env={**os.environ, "NOISE_FLOOR_CPU_THREADS": "8"},
    )
    stats_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(temp / "noise_manifest.json", stats_copy)


def logits_unit(temp: Path, tag: str):
    out = EXP_ROOT / "results" / "supp_units" / tag
    log = EXP_ROOT / "logs" / "logits" / f"{tag}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as f:
        subprocess.run(
            [PYTHON, str(EXP_ROOT / "scripts" / "run_logits_unit.py"),
             "--model-path", str(temp), "--unit-tag", tag,
             "--out-dir", str(out), "--tokenized", str(TOKENIZED)],
            check=True,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "0",
                 "VLLM_BATCH_INVARIANT": "1", "TOKENIZERS_PARALLELISM": "false"},
            stdout=f, stderr=subprocess.STDOUT,
        )


def bench_unit(temp: Path, tag: str, benchmark: str):
    subprocess.run(
        [PYTHON, str(EXP_ROOT / "scripts" / "run_benchmark_unit.py"),
         "--model-path", str(temp), "--model-tag", tag,
         "--benchmark", benchmark,
         "--out-dir", str(EXP_ROOT / "results" / "supp_behavior")],
        check=True,
    )


def main() -> None:
    temp = EXP_ROOT / "tmp" / "supp_noise_checkpoint"
    stats_dir = EXP_ROOT / "results" / "supp_weight_stats"

    for arm, sigmas, scope, base_seed in (("s1", S1, "all", 3000), ("s2", S2, "ffn", 4000)):
        for idx, sigma in enumerate(sigmas):
            for rep in range(3):
                tag = f"supp_{arm}_{scope}_sigma{idx}_rep{rep}"
                summary = EXP_ROOT / "results" / "supp_units" / tag / "summary.json"
                if complete(summary):
                    print(f"[supp] {tag}: complete; skipping")
                    continue
                seed = base_seed + 10 * idx + rep
                try:
                    make_ckpt(temp, sigma, seed, tag, scope, stats_dir / f"{tag}.json")
                    logits_unit(temp, tag)
                    print(f"[supp] {tag}: done sigma={sigma} seed={seed}")
                finally:
                    if temp.exists():
                        shutil.rmtree(temp)

    for idx, sigma in enumerate(S3):
        for rep in range(5):
            tag = f"supp_s3_all_sigma{idx}_rep{rep}"
            raw = EXP_ROOT / "results" / "supp_behavior" / tag / "gsm8k.raw.json"
            if complete(raw):
                print(f"[supp] {tag}: complete; skipping")
                continue
            seed = 5000 + 100 * idx + rep
            try:
                make_ckpt(temp, sigma, seed, tag, "all", stats_dir / f"{tag}.json")
                bench_unit(temp, tag, "gsm8k")
                print(f"[supp] {tag}: done sigma={sigma} seed={seed}")
            finally:
                if temp.exists():
                    shutil.rmtree(temp)

    print("[supp] all units complete")


if __name__ == "__main__":
    main()
