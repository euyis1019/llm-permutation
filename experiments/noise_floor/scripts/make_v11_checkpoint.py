"""Create one formal Part 1b Base checkpoint with a frozen v1.1 family."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model_mutations import make_perm


def f3_perm(layer_idx: int, m: int) -> torch.Tensor:
    seed = 301 * 1000 + layer_idx
    g = torch.Generator().manual_seed(seed)
    p = torch.arange(m, dtype=torch.int64)
    count = int(0.30 * m)
    chosen = torch.randperm(m, generator=g)[:count]
    base = torch.arange(count)
    while True:
        q = torch.randperm(count, generator=g)
        if not bool((q == base).any()):
            break
    p[chosen] = chosen[q]
    return p


def tensor_sha256(t: torch.Tensor) -> str:
    return hashlib.sha256(
        t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def atomic_json(path: Path, obj) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--variant", choices=["f9_k100", "f10_k100", "f3_k30"], required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    marker = out / "checkpoint_complete.json"
    if marker.is_file() and list(out.glob("*.safetensors")):
        print(f"[make-v11] {args.tag}: complete; skipping")
        return

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.source, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        local_files_only=True,
    )
    layers = model.model.layers
    m = int(model.config.intermediate_size)
    perm_hashes = []
    representative_before = {}
    for i in (0, len(layers) - 1):
        mlp = layers[i].mlp
        representative_before[i] = {
            "gate": mlp.gate_proj.weight.detach().clone(),
            "up": mlp.up_proj.weight.detach().clone(),
            "down": mlp.down_proj.weight.detach().clone(),
        }

    for i, layer in enumerate(layers):
        p = f3_perm(i, m) if args.variant == "f3_k30" else make_perm(args.variant, i, m)
        if not torch.equal(torch.sort(p).values, torch.arange(m)):
            raise AssertionError(f"non-bijection L{i}")
        perm_hashes.append(hashlib.sha256(p.numpy().tobytes()).hexdigest())
        mlp = layer.mlp
        mlp.gate_proj.weight.copy_(mlp.gate_proj.weight[p])
        mlp.up_proj.weight.copy_(mlp.up_proj.weight[p])
        mlp.down_proj.weight.copy_(mlp.down_proj.weight[:, p])

    for i, before in representative_before.items():
        p = f3_perm(i, m) if args.variant == "f3_k30" else make_perm(args.variant, i, m)
        mlp = layers[i].mlp
        assert torch.equal(mlp.gate_proj.weight, before["gate"][p])
        assert torch.equal(mlp.up_proj.weight, before["up"][p])
        assert torch.equal(mlp.down_proj.weight, before["down"][:, p])

    model.save_pretrained(out, safe_serialization=True)
    tok = AutoTokenizer.from_pretrained(args.source, local_files_only=True)
    tok.save_pretrained(out)
    extra = Path(args.source) / "generation_config.json"
    if extra.is_file():
        shutil.copy2(extra, out / extra.name)

    manifest = {
        "complete": True,
        "tag": args.tag,
        "source": str(Path(args.source).resolve()),
        "variant": args.variant,
        "n_layers": len(layers),
        "intermediate_size": m,
        "layer_perm_sha256": perm_hashes,
        "seed_rule": (
            "301*1000+layer" if args.variant == "f3_k30"
            else "401+layer" if args.variant == "f9_k100"
            else "deterministic_all_valid_odd_pairs"
        ),
        "verified_relation_layers": [0, len(layers) - 1],
        "elapsed_seconds": time.time() - t0,
    }
    atomic_json(out / "perm_manifest.json", manifest)
    atomic_json(marker, manifest)
    print(f"[make-v11] {args.tag}: complete in {manifest['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()

