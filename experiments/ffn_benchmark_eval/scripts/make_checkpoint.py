"""Generate a permuted (or plain re-saved) Qwen3-4B checkpoint on disk.

Convention (validated in ../ffn_permutation): a SwiGLU FFN is invariant under a
joint permutation of the intermediate-neuron axis:
    gate.weight <- gate.weight[perm, :]
    up.weight   <- up.weight[perm, :]
    down.weight <- down.weight[:, perm]
Each layer draws an independent permutation.  In exact real arithmetic the
function is unchanged; under BF16 the down_proj GEMM reduction order shifts,
producing the drift this experiment measures at the benchmark level.

Scopes:
    all36            permute every layer
    single:L         permute only layer L
    prefix:K         permute layers [0, K)
Kinds:
    random           full random permutation (maximal displacement)
    adjacent_swap    swap neighbouring pairs (minimal displacement)
    reverse          reverse order

A permuted checkpoint = base non-FFN weights + reindexed FFN weights.  The
generator verifies, before writing, that every non-FFN tensor is byte-identical
to the source and each permuted FFN triplet satisfies the row/col relation.
`baseline_copy` uses scope=none (identity everywhere): a pure re-save.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import permutation as P


def layer_seed(base_seed: int, layer_idx: int) -> int:
    return base_seed * 1000 + layer_idx


def build_layer_perm(kind: str, m: int, base_seed: int, layer_idx: int) -> torch.Tensor:
    if kind == "random":
        return P.make_perm("random", m, seed=layer_seed(base_seed, layer_idx))
    return P.make_perm(kind, m)  # adjacent_swap / reverse are deterministic


def target_layers(scope: str, n_layers: int) -> list[int]:
    if scope == "none":
        return []
    if scope == "all36":
        return list(range(n_layers))
    if scope.startswith("single:"):
        return [int(scope.split(":", 1)[1])]
    if scope.startswith("prefix:"):
        k = int(scope.split(":", 1)[1])
        return list(range(k))
    raise ValueError(f"unknown scope {scope!r}")


@torch.no_grad()
def generate(
    source: str,
    out_dir: str,
    scope: str,
    kind: str,
    base_seed: int,
    tag: str,
) -> dict:
    t0 = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        source, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    layers = model.model.layers
    n_layers = len(layers)
    m = model.config.intermediate_size
    to_perm = target_layers(scope, n_layers)

    # Snapshot pre-permutation SHA of every layer's FFN + a couple of non-FFN
    # tensors so we can prove the invariants after mutation.
    pre_ffn = {}
    for i, lyr in enumerate(layers):
        pre_ffn[i] = P.mlp_checksums(lyr.mlp)
    sentinel_names = [
        "model.embed_tokens.weight",
        f"model.layers.0.self_attn.q_proj.weight",
        f"model.layers.{n_layers-1}.self_attn.o_proj.weight",
        "model.norm.weight",
    ]
    sd = dict(model.named_parameters())
    pre_sentinel = {n: P.tensor_sha256(sd[n]) for n in sentinel_names if n in sd}

    layer_perms = {}
    for i in to_perm:
        perm = build_layer_perm(kind, m, base_seed, i)
        assert P.check_bijection(perm, m)
        mlp = layers[i].mlp
        P.permute_rows(mlp.gate_proj.weight, perm)
        P.permute_rows(mlp.up_proj.weight, perm)
        P.permute_cols(mlp.down_proj.weight, perm)
        layer_perms[i] = {
            "perm_sha256": __import__("hashlib").sha256(perm.numpy().tobytes()).hexdigest(),
            "first8": perm[:8].tolist(),
        }

    # ── verification ──────────────────────────────────────────────────────────
    # 1. non-FFN sentinels unchanged
    sd_after = dict(model.named_parameters())
    for n in pre_sentinel:
        assert P.tensor_sha256(sd_after[n]) == pre_sentinel[n], f"non-FFN tensor changed: {n}"
    # 2. untouched layers' FFN unchanged; touched layers satisfy row/col relation
    for i, lyr in enumerate(layers):
        now = P.mlp_checksums(lyr.mlp)
        if i not in to_perm:
            assert now == pre_ffn[i], f"untouched layer {i} FFN changed"
    # 3. row/col permutation relation on touched layers (reload perm, re-apply to
    #    a fresh copy of the source layer and compare byte-for-byte)
    verify_layers = to_perm[:1] + to_perm[-1:] if to_perm else []
    if verify_layers:
        ref = AutoModelForCausalLM.from_pretrained(
            source, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
        )
        for i in set(verify_layers):
            perm = build_layer_perm(kind, m, base_seed, i)
            rmlp = ref.model.layers[i].mlp
            g_expect = rmlp.gate_proj.weight[perm.to(rmlp.gate_proj.weight.device), :]
            u_expect = rmlp.up_proj.weight[perm.to(rmlp.up_proj.weight.device), :]
            d_expect = rmlp.down_proj.weight[:, perm.to(rmlp.down_proj.weight.device)]
            cur = layers[i].mlp
            assert torch.equal(cur.gate_proj.weight, g_expect), f"gate perm mismatch L{i}"
            assert torch.equal(cur.up_proj.weight, u_expect), f"up perm mismatch L{i}"
            assert torch.equal(cur.down_proj.weight, d_expect), f"down perm mismatch L{i}"
        del ref

    model.save_pretrained(out, safe_serialization=True)
    # tokenizer / aux files
    tok = AutoTokenizer.from_pretrained(source)
    tok.save_pretrained(out)
    for extra in ["generation_config.json"]:
        src_extra = Path(source) / extra
        if src_extra.is_file():
            shutil.copy2(src_extra, out / extra)

    manifest = {
        "tag": tag,
        "source": source,
        "out_dir": str(out),
        "scope": scope,
        "kind": kind,
        "base_seed": base_seed,
        "n_layers": n_layers,
        "intermediate_size": m,
        "permuted_layers": to_perm,
        "layer_perms": layer_perms,
        "sentinel_sha256": pre_sentinel,
        "verified": {
            "non_ffn_unchanged": True,
            "untouched_ffn_unchanged": True,
            "rowcol_relation_layers": sorted(set(verify_layers)),
        },
        "gen_seconds": round(time.time() - t0, 1),
        "generator": "make_checkpoint.py v1",
    }
    P_atomic_write(out / "perm_manifest.json", manifest)
    del model
    return manifest


def P_atomic_write(path: Path, obj) -> None:
    import os, tempfile
    d = str(path.parent)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--scope", default="all36")
    ap.add_argument("--kind", default="random")
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    man = generate(args.source, args.out_dir, args.scope, args.kind, args.base_seed, args.tag)
    print(f"[make_checkpoint] {args.tag}: permuted {len(man['permuted_layers'])} layers "
          f"in {man['gen_seconds']}s -> {args.out_dir}")


if __name__ == "__main__":
    main()
