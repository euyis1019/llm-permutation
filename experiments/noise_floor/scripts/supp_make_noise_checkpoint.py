"""Post-hoc supplement: Gaussian-noised Base checkpoint with selectable scope.

Same noise recipe as make_noise_checkpoint.py (single CPU generator, noise in
param dtype, in-place add, named_parameters order), plus --scope:
  all : every floating parameter (identical to the pre-registered Part 6b)
  ffn : only model.layers.<L>.mlp.{gate_proj,up_proj,down_proj}.weight —
        exactly the parameter set the FFN permutation touches.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

FFN_RE = re.compile(r"^model\.layers\.\d+\.mlp\.(gate_proj|up_proj|down_proj)\.weight$")


def atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
    ap.add_argument("--sigma", type=float, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--scope", choices=["all", "ffn"], default="all")
    args = ap.parse_args()

    torch.set_num_threads(int(os.environ.get("NOISE_FLOOR_CPU_THREADS", "8")))
    out = Path(args.out_dir).resolve()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.source, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        local_files_only=True,
    )
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    s = {"n": 0, "changed": 0, "base_sq": 0.0, "diff_sq": 0.0}
    touched = []
    for name, param in model.named_parameters():
        if not param.is_floating_point():
            continue
        if args.scope == "ffn" and not FFN_RE.match(name):
            continue
        touched.append(name)
        before = param.detach().clone()
        noise = torch.randn(param.shape, generator=gen, dtype=param.dtype, device="cpu")
        noise.mul_(args.sigma)
        param.add_(noise)
        s["n"] += param.numel()
        s["changed"] += int((param != before).sum().item())
        s["base_sq"] += float(before.float().square().sum().item())
        s["diff_sq"] += float((param.float() - before.float()).square().sum().item())
        del noise, before

    model.save_pretrained(out, safe_serialization=True)
    tok = AutoTokenizer.from_pretrained(args.source, local_files_only=True)
    tok.save_pretrained(out)
    extra = Path(args.source) / "generation_config.json"
    if extra.is_file():
        shutil.copy2(extra, out / extra.name)

    stats = {
        **s,
        "changed_fraction": s["changed"] / s["n"] if s["n"] else 0.0,
        "weight_rel_l2": math.sqrt(s["diff_sq"] / max(s["base_sq"], 1e-300)),
    }
    manifest = {
        "complete": True,
        "post_hoc_supplement": True,
        "tag": args.tag,
        "source": str(Path(args.source).resolve()),
        "sigma": args.sigma,
        "seed": args.seed,
        "scope": args.scope,
        "n_params_touched": len(touched),
        "dtype": "bfloat16",
        "noise_generation": "torch.randn(param.shape,dtype=param.dtype); noise.mul_(sigma); param.add_(noise)",
        "stats": stats,
        "elapsed_seconds": time.time() - t0,
    }
    atomic_json(out / "noise_manifest.json", manifest)
    print(
        f"[supp-noise] {args.tag}: scope={args.scope} sigma={args.sigma} "
        f"changed={stats['changed_fraction']:.6f} rel={stats['weight_rel_l2']:.6e} "
        f"elapsed={manifest['elapsed_seconds']:.1f}s"
    )


if __name__ == "__main__":
    main()
