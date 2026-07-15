"""Run one resumable vLLM logits unit for noise_floor Parts 0/1a/2/6b."""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EXP_ROOT = SCRIPT_DIR.parent
os.environ["PYTHONPATH"] = str(SCRIPT_DIR) + os.pathsep + os.environ.get("PYTHONPATH", "")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def atomic_json(path: Path, obj) -> None:
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def append_decision(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")
        f.flush()
        os.fsync(f.fileno())


def gpu_snapshot() -> dict:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
        "-i",
        "0",
    ]
    line = subprocess.check_output(cmd, text=True).strip()
    parts = [x.strip() for x in line.split(",")]
    return {
        "index": int(parts[0]),
        "name": parts[1],
        "total_mib": int(parts[2]),
        "used_mib": int(parts[3]),
        "free_mib": int(parts[4]),
        "gpu_util_percent": int(parts[5]),
        "checked_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "raw": line,
    }


def acquire_resources(unit_tag: str, decisions: Path, progress: Path) -> dict | None:
    started = time.time()
    while True:
        snap = gpu_snapshot()
        free = snap["free_mib"]
        total = snap["total_mib"]
        if free >= 15 * 1024:
            choice = {
                "mode": "normal",
                "gpu_memory_utilization": 0.28,
                "max_model_len": 4096,
                "max_num_seqs": 256,
                "snapshot": snap,
            }
            break
        if free >= 10 * 1024:
            util = max(0.18, (free - 2 * 1024) / total)
            choice = {
                "mode": "degraded",
                "gpu_memory_utilization": util,
                "max_model_len": 2048,
                "max_num_seqs": 8,
                "snapshot": snap,
            }
            break
        waited = time.time() - started
        append_decision(
            decisions,
            f"- {snap['checked_at_utc']} `{unit_tag}`: GPU 0 free={free} MiB <10 GiB; "
            "按 §1 退避 600 秒。",
        )
        if waited > 4 * 3600:
            atomic_json(
                progress,
                {
                    "status": "resource_wait_exit",
                    "unit": unit_tag,
                    "waited_seconds": waited,
                    "last_gpu_snapshot": snap,
                },
            )
            return None
        time.sleep(600)
    append_decision(
        decisions,
        f"- {choice['snapshot']['checked_at_utc']} `{unit_tag}`: GPU 0 "
        f"free={choice['snapshot']['free_mib']} MiB，采用 {choice['mode']} 配置 "
        f"(util={choice['gpu_memory_utilization']:.6f}, "
        f"max_model_len={choice['max_model_len']}, max_num_seqs={choice['max_num_seqs']})。",
    )
    return choice


def prompt_mean_nll(output, input_ids: list[int]) -> tuple[float, int]:
    entries = output.prompt_logprobs
    if entries is None:
        raise RuntimeError("vLLM returned no prompt_logprobs")
    vals = []
    for pos, item in enumerate(entries):
        if pos == 0 or item is None:
            continue
        token_id = int(input_ids[pos])
        record = item.get(token_id)
        if record is None:
            raise KeyError(f"chosen prompt token {token_id} absent at position {pos}")
        vals.append(-float(record.logprob))
    if not vals:
        raise RuntimeError("no prompt token NLL values")
    return sum(vals) / len(vals), len(vals)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--unit-tag", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tokenized", required=True)
    ap.add_argument(
        "--mutation",
        choices=["identity", "f9_k100", "f10_k100", "f7"],
        default="identity",
    )
    ap.add_argument("--decisions", default=str(EXP_ROOT / "DECISIONS.md"))
    ap.add_argument("--progress", default=str(EXP_ROOT / "PROGRESS.md"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    summary_path = out_dir / "summary.json"
    if summary_path.is_file():
        try:
            if json.loads(summary_path.read_text()).get("complete") is True:
                print(f"[logits-unit] {args.unit_tag}: complete; skipping")
                return 0
        except Exception:
            pass

    resource = acquire_resources(
        args.unit_tag, Path(args.decisions), Path(args.progress)
    )
    if resource is None:
        print(f"[logits-unit] {args.unit_tag}: resource wait exceeded 4h")
        return 75

    out_dir.mkdir(parents=True, exist_ok=True)
    tokenized = json.loads(Path(args.tokenized).read_text())
    prompts = tokenized["prompts"]
    started = time.time()

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=str(Path(args.model_path).resolve()),
        tensor_parallel_size=1,
        dtype="bfloat16",
        gpu_memory_utilization=resource["gpu_memory_utilization"],
        enable_prefix_caching=False,
        max_model_len=resource["max_model_len"],
        max_num_seqs=resource["max_num_seqs"],
        enforce_eager=True,
        trust_remote_code=True,
        logits_processors=["logits_capture:RawLogitsCapture"],
    )

    if args.mutation == "identity":
        mutation_report = {"variant": "identity", "layers": 0, "perm_sha256": []}
    else:
        mutation_report = llm.llm_engine.apply_model(
            functools.partial(
                __import__("model_mutations").apply_vllm_variant,
                variant=args.mutation,
            )
        )[0]

    records = []
    logits_dir = out_dir / "logits"
    logits_dir.mkdir(exist_ok=True)
    for pos, prompt in enumerate(prompts):
        prompt_id = int(prompt["id"])
        ids = [int(x) for x in prompt["input_ids"]]
        stem = logits_dir / f"prompt_{prompt_id:02d}"
        sp = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            prompt_logprobs=1,
            extra_args={"noise_floor_capture": str(stem)},
        )
        outputs = llm.generate(
            {"prompt_token_ids": ids}, sp, use_tqdm=False
        )
        if len(outputs) != 1:
            raise AssertionError(f"expected one request output, got {len(outputs)}")
        mean_nll, n_nll = prompt_mean_nll(outputs[0], ids)
        meta_path = stem.with_suffix(".meta.json")
        if not meta_path.is_file():
            raise RuntimeError(f"capture processor did not write {meta_path}")
        meta = json.loads(meta_path.read_text())
        raw_path = logits_dir / meta["raw_file"]
        f32_path = logits_dir / meta["float32_file"]
        records.append(
            {
                "position": pos,
                "prompt_id": prompt_id,
                "tag": prompt.get("tag"),
                "n_input_tokens": len(ids),
                "mean_nll": mean_nll,
                "n_nll_tokens": n_nll,
                "generated_token_id": int(outputs[0].outputs[0].token_ids[0]),
                "logits": {
                    **meta,
                    "raw_path": str(raw_path.relative_to(out_dir)),
                    "float32_path": str(f32_path.relative_to(out_dir)),
                    "float32_sha256": file_sha256(f32_path),
                },
            }
        )

    import torch
    import transformers
    import vllm

    summary = {
        "complete": True,
        "unit_tag": args.unit_tag,
        "model_path": str(Path(args.model_path).resolve()),
        "mutation": args.mutation,
        "mutation_report": mutation_report,
        "resource": resource,
        "determinism": {
            "VLLM_BATCH_INVARIANT": os.environ.get("VLLM_BATCH_INVARIANT"),
            "enforce_eager": True,
            "enable_prefix_caching": False,
            "dtype": "bfloat16",
            "temperature": 0.0,
        },
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "vllm": vllm.__version__,
        },
        "tokenized_path": str(Path(args.tokenized).resolve()),
        "tokenized_sha256": file_sha256(Path(args.tokenized)),
        "n_prompts": len(records),
        "mean_nll": sum(r["mean_nll"] for r in records) / len(records),
        "records": records,
        "elapsed_seconds": time.time() - started,
    }
    atomic_json(summary_path, summary)
    print(
        f"[logits-unit] {args.unit_tag}: complete prompts={len(records)} "
        f"elapsed={summary['elapsed_seconds']:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
