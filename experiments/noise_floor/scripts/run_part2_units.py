"""Regenerate, measure, and roll-clean the eight frozen Part 2 anchors."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


EXP_ROOT = Path(__file__).resolve().parents[1]
FFN = EXP_ROOT.parent / "ffn_benchmark_eval"
PYTHON = "/nvme0/if/anaconda3/envs/qwen3/bin/python"
SOURCE = "/nvme0/if/models/Qwen3-4B"
TOKENIZED = EXP_ROOT.parent / "ffn_permutation" / "results" / "tokenized.json"

ANCHORS = [
    ("qwen3_4b__abl_scope_single0_random_s7", "part2_scope_single0"),
    ("qwen3_4b__abl_scope_single17_random_s7", "part2_scope_single17"),
    ("qwen3_4b__abl_scope_single35_random_s7", "part2_scope_single35"),
    ("qwen3_4b__abl_scope_prefix6_random_s7", "part2_scope_prefix6"),
    ("qwen3_4b__abl_scope_prefix18_random_s7", "part2_scope_prefix18"),
    ("qwen3_4b__abl_scope_all36_random_s7", "part2_scope_all36"),
    ("qwen3_4b__abl_mag_adjswap_all36", "part2_mag_adjswap"),
    ("qwen3_4b__abl_mag_reverse_all36", "part2_mag_reverse"),
]


def clean_generated(checkpoint: Path, original_manifest: bytes) -> None:
    for p in checkpoint.iterdir():
        if p.name == "perm_manifest.json":
            continue
        if p.is_file() or p.is_symlink():
            p.unlink()
        elif p.is_dir():
            import shutil
            shutil.rmtree(p)
    (checkpoint / "perm_manifest.json").write_bytes(original_manifest)


def main() -> None:
    for checkpoint_name, unit in ANCHORS:
        out = EXP_ROOT / "results" / "units" / unit / "summary.json"
        if out.is_file() and json.loads(out.read_text()).get("complete") is True:
            print(f"[part2] {unit}: complete; skipping")
            continue
        checkpoint = FFN / "checkpoints" / checkpoint_name
        manifest_path = checkpoint / "perm_manifest.json"
        original = manifest_path.read_bytes()
        man = json.loads(original)
        try:
            subprocess.run(
                [
                    PYTHON, str(FFN / "scripts" / "make_checkpoint.py"),
                    "--source", SOURCE,
                    "--out-dir", str(checkpoint),
                    "--scope", str(man["scope"]),
                    "--kind", str(man["kind"]),
                    "--base-seed", str(man["base_seed"]),
                    "--tag", str(man["tag"]),
                ],
                check=True,
            )
            env = {
                **os.environ,
                "CUDA_VISIBLE_DEVICES": "0",
                "VLLM_BATCH_INVARIANT": "1",
                "TOKENIZERS_PARALLELISM": "false",
            }
            log = EXP_ROOT / "logs" / "logits" / f"{unit}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            with open(log, "w") as f:
                subprocess.run(
                    [
                        PYTHON, str(EXP_ROOT / "scripts" / "run_logits_unit.py"),
                        "--model-path", str(checkpoint),
                        "--unit-tag", unit,
                        "--out-dir", str(EXP_ROOT / "results" / "units" / unit),
                        "--tokenized", str(TOKENIZED),
                    ],
                    check=True, env=env, stdout=f, stderr=subprocess.STDOUT,
                )
            print(f"[part2] {unit}: complete")
        finally:
            clean_generated(checkpoint, original)


if __name__ == "__main__":
    main()

