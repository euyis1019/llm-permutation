"""Stage 1: BF16 single-down-projection geometry law measurements."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
UPSTREAM = ROOT.parent / "ffn_permutation"
sys.path.insert(0, str(HERE))

from backend import matmul
from geometry import geometry_metrics
from perm_families import M, all_specs, make_from_spec
from permutation import diff_metrics, env_info

LAYERS = (0, 17, 35)
INPUTS = ("real", "randn_s7_x1", "randn_s7_x10")
DEFAULT_MODEL = Path("/nvme0/if/models/Qwen3-4B-Base")


def atomic_write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def read_complete_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with path.open() as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except Exception as exc:
                raise RuntimeError(f"invalid existing JSONL at line {lineno}: {exc}") from exc
    return records


def frozen_prompt_text(tokenizer) -> tuple[str, dict]:
    pdata = json.loads((UPSTREAM / "prompts.json").read_text())
    prompt = next(p for p in pdata["prompts"] if p["id"] == 24)
    if tokenizer.chat_template:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt["text"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    else:
        text = prompt["text"]
    return text, prompt


@torch.inference_mode()
def capture_inputs_and_weights(model_path: Path) -> tuple[dict, dict, dict]:
    """CPU capture keeps Stage 1 GPU allocation below 2 GiB."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model.eval()
    if model.config.intermediate_size != M:
        raise AssertionError((model.config.intermediate_size, M))
    text, prompt = frozen_prompt_text(tokenizer)
    enc = tokenizer(text, return_tensors="pt")
    captured: dict[int, torch.Tensor] = {}
    handles = []
    for li in LAYERS:
        def make_hook(layer_idx):
            def hook(module, args):
                # down_proj input is the pre-registered h tensor.
                captured[layer_idx] = args[0].detach().reshape(-1, M).T.contiguous().cpu()
            return hook
        handles.append(model.model.layers[li].mlp.down_proj.register_forward_pre_hook(make_hook(li)))
    model(**enc, use_cache=False)
    for handle in handles:
        handle.remove()
    if set(captured) != set(LAYERS):
        raise RuntimeError(f"capture incomplete: {sorted(captured)}")
    weights = {
        li: model.model.layers[li].mlp.down_proj.weight.detach().cpu().contiguous().clone()
        for li in LAYERS
    }
    meta = {
        "prompt": prompt,
        "rendered_n_tokens": int(enc.input_ids.numel()),
        "model_config": {
            "hidden_size": model.config.hidden_size,
            "intermediate_size": model.config.intermediate_size,
            "num_hidden_layers": model.config.num_hidden_layers,
        },
    }
    del model, tokenizer, enc
    return captured, weights, meta


def synthetic_inputs(n_columns: int = 256) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(7)
    base = torch.randn(M, n_columns, generator=g, dtype=torch.float32)
    return {
        "randn_s7_x1": base.to(torch.bfloat16).contiguous(),
        "randn_s7_x10": (base * 10.0).to(torch.bfloat16).contiguous(),
    }


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--results-dir", type=Path, default=ROOT / "results")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--inversion-pairs", type=int, default=1_000_000)
    args = ap.parse_args()
    out_path = args.results_dir / "stage1_singlelayer.jsonl"
    manifest_path = args.results_dir / "stage1_manifest.json"
    if out_path.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite {out_path}; pass --resume")
    records = read_complete_records(out_path) if args.resume else []
    done = {(r["perm_key"], r["layer"], r["input"]) for r in records}

    t0 = time.time()
    real, weights, capture_meta = capture_inputs_and_weights(args.model_path.resolve())
    synth = synthetic_inputs()
    host_inputs = {li: {"real": real[li], **synth} for li in LAYERS}
    del real, synth

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the pre-registered backend")
    device = torch.device("cuda:0")
    cuda_weights = {li: weights[li].to(device) for li in LAYERS}
    cuda_inputs = {
        li: {name: tensor.to(device) for name, tensor in host_inputs[li].items()}
        for li in LAYERS
    }
    baseline = {
        (li, name): matmul(cuda_weights[li], cuda_inputs[li][name])
        for li in LAYERS for name in INPUTS
    }

    specs = all_specs()
    for spec_i, spec in enumerate(specs, 1):
        perm = make_from_spec(spec, M)
        geom = geometry_metrics(perm, args.inversion_pairs)
        pc = perm.to(device)
        for li in LAYERS:
            wp = cuda_weights[li][:, pc]
            for name in INPUTS:
                key = (spec.key, li, name)
                if key in done:
                    continue
                hp = cuda_inputs[li][name][pc, :]
                yp = matmul(wp, hp, backend="torch_bf16")
                metrics = diff_metrics(yp, baseline[(li, name)])
                record = {
                    "perm_key": spec.key,
                    "family": spec.family,
                    "parameter": spec.parameter,
                    "seed": spec.seed,
                    "layer": li,
                    "input": name,
                    "n_columns": int(hp.shape[1]),
                    "geometry": geom,
                    "drift": metrics,
                    "backend": "torch_bf16",
                }
                records.append(record)
                done.add(key)
                atomic_write_jsonl(out_path, records)
        print(f"[stage1] {spec_i}/{len(specs)} {spec.key} rows={len(records)}", flush=True)

    expected = len(specs) * len(LAYERS) * len(INPUTS)
    if len(records) != expected or len(done) != expected:
        raise RuntimeError(f"incomplete Stage 1: {len(records)} records, expected {expected}")
    manifest = {
        "complete": True,
        "expected_records": expected,
        "actual_records": len(records),
        "families": len(specs),
        "layers": list(LAYERS),
        "inputs": list(INPUTS),
        "capture": capture_meta,
        "inversion_pairs": args.inversion_pairs,
        "environment": env_info(),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    from permutation import atomic_write_json
    atomic_write_json(str(manifest_path), manifest)
    print(f"[stage1] complete: {len(records)} rows in {manifest['elapsed_seconds']}s")


if __name__ == "__main__":
    main()
