"""Amendment v1.1 Stage 1b: fresh-data three-tier single-layer confirmation."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
UPSTREAM = ROOT.parent / "ffn_permutation"
sys.path.insert(0, str(HERE))

from backend import linear
from perm_families import M, all_specs_v11, make_family_perm_v11, predicted_tier_v11
from permutation import atomic_write_json, diff_metrics, env_info

LAYERS = (0, 17, 35)
PROMPT_IDS = (24, 0)
BACKENDS = ("torch_bf16", "vllm_bi")
SHAPES = ("full", "decode1")
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


def read_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with path.open() as f:
        for lineno, line in enumerate(f, 1):
            try:
                records.append(json.loads(line))
            except Exception as exc:
                raise RuntimeError(f"invalid JSONL line {lineno}: {exc}") from exc
    return records


def render_prompt(tokenizer, prompt: dict) -> str:
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt["text"]}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
    return prompt["text"]


@torch.inference_mode()
def capture(model_path: Path) -> tuple[dict, dict, dict]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="cpu",
        low_cpu_mem_usage=True, local_files_only=True,
    ).eval()
    pdata = json.loads((UPSTREAM / "prompts.json").read_text())["prompts"]
    by_id = {int(p["id"]): p for p in pdata}
    expected_second = min(i for i in by_id if i != 24)
    if PROMPT_IDS != (24, expected_second):
        raise AssertionError((PROMPT_IDS, expected_second))

    captured: dict[tuple[int, int], torch.Tensor] = {}
    token_counts: dict[int, int] = {}
    for prompt_id in PROMPT_IDS:
        enc = tokenizer(render_prompt(tokenizer, by_id[prompt_id]), return_tensors="pt")
        # Amendment freezes at most 124 real tokens for the full-shape arm.
        enc.input_ids = enc.input_ids[:, :124]
        enc.attention_mask = enc.attention_mask[:, :124]
        token_counts[prompt_id] = int(enc.input_ids.numel())
        handles = []
        for li in LAYERS:
            def make_hook(layer_idx, pid):
                def hook(module, args):
                    captured[(pid, layer_idx)] = args[0].detach().reshape(-1, M).contiguous().cpu()
                return hook
            handles.append(model.model.layers[li].mlp.down_proj.register_forward_pre_hook(
                make_hook(li, prompt_id)))
        model(**enc, use_cache=False)
        for handle in handles:
            handle.remove()

    weights = {li: model.model.layers[li].mlp.down_proj.weight.detach().cpu().contiguous().clone()
               for li in LAYERS}
    meta = {
        "prompt_ids": list(PROMPT_IDS),
        "prompt_rule": "id=24 and minimum id != 24",
        "full_token_counts_after_cap124": token_counts,
        "model_config": {
            "hidden_size": model.config.hidden_size,
            "intermediate_size": model.config.intermediate_size,
            "num_hidden_layers": model.config.num_hidden_layers,
        },
    }
    del model, tokenizer
    return captured, weights, meta


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--results-dir", type=Path, default=ROOT / "results")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    out_path = args.results_dir / "stage1b_singlelayer.jsonl"
    manifest_path = args.results_dir / "stage1b_manifest.json"
    if out_path.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite {out_path}; use --resume")
    records = read_records(out_path) if args.resume else []
    done = {(r["perm_key"], r["layer"], r["prompt_id"], r["backend"], r["shape"])
            for r in records}

    specs = all_specs_v11()
    perms = {s.key: make_family_perm_v11(s) for s in specs}
    tiers = {key: predicted_tier_v11(p) for key, p in perms.items()}
    for spec in specs:
        expected = ({"F9_inblock_shuffle": "zero", "F10_odd_pairs": "ceil",
                     "F12_win16_aligned": "sub", "F3_scattered_global": "ceil",
                     "F7_global_random": "ceil", "F8_identity": "zero"}.get(spec.family)
                    or ("zero" if int(spec.parameter) == 0 else "ceil"))
        if tiers[spec.key] != expected:
            raise AssertionError(f"label mismatch {spec.key}: {tiers[spec.key]} != {expected}")

    t0 = time.time()
    activations, weights, capture_meta = capture(args.model_path.resolve())
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    dev = torch.device("cuda:0")
    cuda_weights = {li: weights[li].to(dev) for li in LAYERS}
    cuda_x = {}
    for (pid, li), x in activations.items():
        cuda_x[(pid, li, "full")] = x.to(dev)
        cuda_x[(pid, li, "decode1")] = x[-1:, :].to(dev).contiguous()
    baselines = {(pid, li, shape, backend): linear(cuda_x[(pid, li, shape)], cuda_weights[li], backend)
                 for pid in PROMPT_IDS for li in LAYERS for shape in SHAPES for backend in BACKENDS}

    for spec_i, spec in enumerate(specs, 1):
        pc = perms[spec.key].to(dev)
        for li in LAYERS:
            wp = cuda_weights[li][:, pc].contiguous()
            for pid in PROMPT_IDS:
                for backend in BACKENDS:
                    for shape in SHAPES:
                        key = (spec.key, li, pid, backend, shape)
                        if key in done:
                            continue
                        x = cuda_x[(pid, li, shape)]
                        yp = linear(x[:, pc].contiguous(), wp, backend)
                        records.append({
                            "perm_key": spec.key, "family": spec.family,
                            "parameter": spec.parameter, "seed": spec.seed,
                            "layer": li, "prompt_id": pid, "backend": backend,
                            "shape": shape, "T": int(x.shape[0]),
                            "predicted_tier": tiers[spec.key],
                            "drift": diff_metrics(yp, baselines[(pid, li, shape, backend)]),
                        })
                        done.add(key)
                        atomic_write_jsonl(out_path, records)
        print(f"[stage1b] {spec_i}/{len(specs)} {spec.key} rows={len(records)}", flush=True)

    expected = len(specs) * len(LAYERS) * len(PROMPT_IDS) * len(BACKENDS) * len(SHAPES)
    if len(records) != expected or len(done) != expected:
        raise RuntimeError(f"incomplete Stage 1b: {len(records)} vs {expected}")
    atomic_write_json(str(manifest_path), {
        "complete": True, "expected_records": expected, "actual_records": len(records),
        "instances": len(specs), "layers": list(LAYERS), "prompt_ids": list(PROMPT_IDS),
        "backends": list(BACKENDS), "shapes": list(SHAPES), "capture": capture_meta,
        "environment": env_info(), "elapsed_seconds": round(time.time() - t0, 1),
    })
    print(f"[stage1b] complete: {len(records)} rows", flush=True)


if __name__ == "__main__":
    main()
