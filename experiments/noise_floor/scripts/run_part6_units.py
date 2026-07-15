"""Run the 30 resumable Gaussian sigma-sweep units with one temp checkpoint."""

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
SIGMAS = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]


def complete(path: Path) -> bool:
    try:
        return json.loads(path.read_text()).get("complete") is True
    except Exception:
        return False


def main() -> None:
    temp = EXP_ROOT / "tmp" / "noise_checkpoint"
    stats_dir = EXP_ROOT / "results" / "part6_weight_stats_units"
    stats_dir.mkdir(parents=True, exist_ok=True)
    for idx, sigma in enumerate(SIGMAS):
        for rep in range(3):
            tag = f"sigma_{idx:02d}_rep{rep}"
            summary = EXP_ROOT / "results" / "units" / tag / "summary.json"
            stats_copy = stats_dir / f"{tag}.json"
            if complete(summary) and stats_copy.is_file():
                print(f"[part6] {tag}: complete; skipping")
                continue
            seed = 1000 + 10 * idx + rep
            if temp.exists():
                shutil.rmtree(temp)
            make_cmd = [
                PYTHON, str(EXP_ROOT / "scripts" / "make_noise_checkpoint.py"),
                "--source", SOURCE, "--out-dir", str(temp),
                "--sigma", repr(sigma), "--seed", str(seed), "--tag", tag,
            ]
            if rep != 0:
                make_cmd.append("--skip-stats")
            env = {**os.environ, "NOISE_FLOOR_CPU_THREADS": "8"}
            try:
                subprocess.run(make_cmd, check=True, env=env)
                shutil.copy2(temp / "noise_manifest.json", stats_copy)
                log = EXP_ROOT / "logs" / "logits" / f"{tag}.log"
                with open(log, "w") as f:
                    subprocess.run(
                        [
                            PYTHON, str(EXP_ROOT / "scripts" / "run_logits_unit.py"),
                            "--model-path", str(temp), "--unit-tag", tag,
                            "--out-dir", str(EXP_ROOT / "results" / "units" / tag),
                            "--tokenized", str(TOKENIZED),
                        ],
                        check=True,
                        env={
                            **os.environ,
                            "CUDA_VISIBLE_DEVICES": "0",
                            "VLLM_BATCH_INVARIANT": "1",
                            "TOKENIZERS_PARALLELISM": "false",
                        },
                        stdout=f, stderr=subprocess.STDOUT,
                    )
                print(f"[part6] {tag}: complete sigma={sigma} seed={seed}")
            finally:
                if temp.exists():
                    shutil.rmtree(temp)


if __name__ == "__main__":
    main()

