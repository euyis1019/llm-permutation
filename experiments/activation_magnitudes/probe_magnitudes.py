"""Experiment 3 — activation & computation magnitude profile of Qwen3-4B (BF16).

Question: how large are the numbers the model actually computes with, per
layer and per stream, and what does that imply for the BF16 grid (ulp) —
i.e. ground the permutation-drift story in measured magnitudes.

Inputs: 16 prompts sampled (seed 42) from each of 8 normalized benchmarks in
/nvme0/if/llm-brewing/bench/datasets/benchmark/normalized (128 prompts total),
chat-templated, truncated to MAX_TOKENS.

Measured per layer (36) per stream, over all prompt tokens:
    embed_out (once), block_in==prev block_out, attn_out, mlp_in,
    gate_out (pre-SiLU), up_out, h (down_proj input), mlp_out, block_out,
    final_norm_out, logits
Stats: rms, abs_mean, abs_p50, abs_p99, abs_max  (float32 accumulation).

Static weight stats per layer: rms / abs_max of gate/up/down weights.

Down-GEMM deep dive (layers 0/17/35, 16 sampled tokens from 4 prompts):
  - cancellation ratio  Σ_k|W[i,k]·h[k]| / |y_i|   (per output element)
  - max-term ratio      max_k|W[i,k]·h[k]| / |y_i|
  - intrinsic BF16 GEMM rounding: y_bf16 vs exact fp64 GEMM on the same
    bf16 values, rel_l2 per token.

Read-only on model; no weight modification.
"""

import argparse
import json
import math
import os
import random
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "ffn_permutation"))
from permutation import atomic_write_json, env_info

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(HERE)), "models", "Qwen3-4B"
)
DEFAULT_BENCH_DIR = "/nvme0/if/llm-brewing/bench/datasets/benchmark/normalized"
DEFAULT_RESULTS_DIR = os.path.join(HERE, "results")

BENCHES = ["mmlu", "mmlu_pro", "mmlu_redux", "ceval", "cmmlu", "gsm8k", "math500", "bbh", "cruxeval"]
PER_BENCH = 16
MAX_TOKENS = 512
SEED = 42
DEEP_LAYERS = [0, 17, 35]
DEEP_PROMPTS = 4          # first N prompts used for the down-GEMM deep dive
DEEP_TOKENS = 16          # tokens sampled per prompt per layer

STREAMS = ["attn_out", "mlp_in", "gate_out", "up_out", "h", "mlp_out", "block_out"]


def build_prompts(bench_dir):
    rng = random.Random(SEED)
    prompts = []
    for bench in BENCHES:
        path = os.path.join(bench_dir, bench, f"{bench}.jsonl")
        if not os.path.exists(path):
            print(f"skip missing bench {bench}")
            continue
        with open(path) as f:
            rows = [json.loads(l) for l in f]
        for r in rng.sample(rows, PER_BENCH):
            q = r["question"]
            if r.get("choices"):
                opts = "\n".join(
                    f"{chr(65+i)}. {c}" for i, c in enumerate(r["choices"])
                )
                text = f"{q}\n{opts}\n请选择正确选项。" if bench in ("ceval", "cmmlu") \
                    else f"{q}\n{opts}\nChoose the correct option."
            else:
                text = q
            prompts.append({"bench": bench, "sample_id": r["sample_id"], "text": text})
    return prompts


def tensor_stats(t: torch.Tensor) -> dict:
    x = t.detach().float().flatten()
    ax = x.abs()
    n = ax.numel()
    if n > 1_000_000:
        step = n // 1_000_000 + 1
        ax_q = ax[::step]
    else:
        ax_q = ax
    q = torch.quantile(ax_q, torch.tensor([0.5, 0.99], device=ax.device))
    return {
        "rms": x.pow(2).mean().sqrt().item(),
        "abs_mean": ax.mean().item(),
        "abs_p50": q[0].item(),
        "abs_p99": q[1].item(),
        "abs_max": ax.max().item(),
        "n": n,
    }


def merge_stats(acc: dict, s: dict):
    """Accumulate token-weighted moments + max across prompts."""
    n = s["n"]
    acc["n"] += n
    acc["sum_sq"] += s["rms"] ** 2 * n
    acc["sum_abs"] += s["abs_mean"] * n
    acc["p50s"].append(s["abs_p50"])
    acc["p99s"].append(s["abs_p99"])
    acc["abs_max"] = max(acc["abs_max"], s["abs_max"])


def finalize(acc: dict) -> dict:
    import statistics as st
    return {
        "rms": math.sqrt(acc["sum_sq"] / acc["n"]),
        "abs_mean": acc["sum_abs"] / acc["n"],
        "abs_p50_median": st.median(acc["p50s"]),
        "abs_p99_median": st.median(acc["p99s"]),
        "abs_max": acc["abs_max"],
        "n_values": acc["n"],
    }


def new_acc():
    return {"n": 0, "sum_sq": 0.0, "sum_abs": 0.0, "p50s": [], "p99s": [], "abs_max": 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    ap.add_argument("--bench-dir", default=DEFAULT_BENCH_DIR)
    ap.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    args = ap.parse_args()
    model_path = os.path.abspath(args.model_path)
    bench_dir = os.path.abspath(args.bench_dir)
    results_dir = os.path.abspath(args.results_dir)

    device = "cuda"
    os.makedirs(results_dir, exist_ok=True)
    prompts = build_prompts(bench_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map=device, local_files_only=True
    )
    model.eval()
    n_layers = model.config.num_hidden_layers

    # tokenize once, persist
    tok_prompts = []
    for i, p in enumerate(prompts):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": p["text"]}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        ids = tokenizer(text, truncation=True, max_length=MAX_TOKENS)["input_ids"]
        tok_prompts.append({**p, "id": i, "n_tokens": len(ids), "input_ids": ids})
    atomic_write_json(
        os.path.join(results_dir, "prompts_used.json"),
        [{k: v for k, v in p.items() if k != "input_ids"} for p in tok_prompts],
    )

    # static weight stats
    weight_stats = {}
    for li in range(n_layers):
        mlp = model.model.layers[li].mlp
        weight_stats[li] = {
            k: {
                "rms": getattr(mlp, f"{k}_proj").weight.float().pow(2).mean().sqrt().item(),
                "abs_max": getattr(mlp, f"{k}_proj").weight.abs().max().item(),
            }
            for k in ("gate", "up", "down")
        }

    # accumulators
    acc = {(li, s): new_acc() for li in range(n_layers) for s in STREAMS}
    acc_embed, acc_norm, acc_logits = new_acc(), new_acc(), new_acc()
    margins = []

    # deep-dive storage: h vectors for selected layers/prompts
    deep_h = {li: [] for li in DEEP_LAYERS}

    cap = {}
    hooks = []

    def mk(key):
        def f(module, args, out=None):
            t = args[0] if out is None else (out[0] if isinstance(out, tuple) else out)
            cap[key] = t.detach()
        return f

    hooks.append(model.model.embed_tokens.register_forward_hook(mk("embed")))
    hooks.append(model.model.norm.register_forward_hook(mk("final_norm")))
    for li in range(n_layers):
        layer = model.model.layers[li]
        hooks.append(layer.self_attn.register_forward_hook(mk((li, "attn_out"))))
        hooks.append(layer.mlp.register_forward_pre_hook(mk((li, "mlp_in"))))
        hooks.append(layer.mlp.gate_proj.register_forward_hook(mk((li, "gate_out"))))
        hooks.append(layer.mlp.up_proj.register_forward_hook(mk((li, "up_out"))))
        hooks.append(layer.mlp.down_proj.register_forward_pre_hook(mk((li, "h"))))
        hooks.append(layer.mlp.register_forward_hook(mk((li, "mlp_out"))))
        hooks.append(layer.register_forward_hook(mk((li, "block_out"))))

    with torch.inference_mode():
        for pi, p in enumerate(tok_prompts):
            ids = torch.tensor([p["input_ids"]], device=device)
            out = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                        use_cache=False)
            for li in range(n_layers):
                for s in STREAMS:
                    merge_stats(acc[(li, s)], tensor_stats(cap[(li, s)]))
            merge_stats(acc_embed, tensor_stats(cap["embed"]))
            merge_stats(acc_norm, tensor_stats(cap["final_norm"]))
            logits = out.logits[0].float()
            merge_stats(acc_logits, tensor_stats(logits))
            top2 = logits.topk(2, dim=-1).values
            margins.append((top2[:, 0] - top2[:, 1]).median().item())

            if pi < DEEP_PROMPTS:
                g = torch.Generator().manual_seed(SEED + pi)
                T = cap[(0, "h")].shape[1]
                tok_idx = torch.randperm(T, generator=g)[:DEEP_TOKENS]
                for li in DEEP_LAYERS:
                    deep_h[li].append(cap[(li, "h")][0, tok_idx].clone())
            if (pi + 1) % 16 == 0:
                print(f"{pi+1}/{len(tok_prompts)} prompts", flush=True)

    for h in hooks:
        h.remove()

    # ---- down-GEMM deep dive -------------------------------------------
    deep = {}
    with torch.inference_mode():
        for li in DEEP_LAYERS:
            W = model.model.layers[li].mlp.down_proj.weight        # [d, m] bf16
            Wa = W.abs().float()
            H = torch.cat(deep_h[li], dim=0)                       # [N, m] bf16
            y = torch.nn.functional.linear(H, W).float()           # bf16 GEMM
            abs_sum = H.abs().float() @ Wa.T                       # Σ|terms|
            eps = 1e-12
            cancel = abs_sum / y.abs().clamp_min(eps)
            y64 = (H.double() @ W.double().T)                      # exact on same values
            rel_err = (y.double() - y64).norm(dim=-1) / y64.norm(dim=-1)
            max_term = []
            for r in range(0, H.shape[0], 4):                      # chunked elementwise max
                chunk = (H[r:r+4].abs().float().unsqueeze(1) * Wa.unsqueeze(0)).max(dim=-1).values
                max_term.append(chunk)
            max_term = torch.cat(max_term, dim=0)
            term_rms = (H.float().pow(2).mean(dim=-1).sqrt()
                        * W.float().pow(2).mean().sqrt())
            deep[li] = {
                "n_tokens": int(H.shape[0]),
                "y_abs_p50": y.abs().median().item(),
                "y_abs_p99": y.abs().flatten().quantile(0.99).item(),
                "abs_term_sum_p50": abs_sum.median().item(),
                "cancellation_ratio_p50": cancel.median().item(),
                "cancellation_ratio_p99": cancel.flatten().quantile(0.99).item(),
                "max_term_over_y_p50": (max_term / y.abs().clamp_min(eps)).median().item(),
                "term_rms_p50": term_rms.median().item(),
                "bf16_gemm_rel_err_median": rel_err.median().item(),
                "bf16_gemm_rel_err_max": rel_err.max().item(),
            }

    # ---- finalize --------------------------------------------------------
    per_layer = {
        s: [finalize(acc[(li, s)]) for li in range(n_layers)] for s in STREAMS
    }
    import statistics as st
    result = {
        "env": env_info(),
        "model_path": model_path,
        "bench_dir": bench_dir,
        "n_prompts": len(tok_prompts),
        "n_tokens_total": sum(p["n_tokens"] for p in tok_prompts),
        "benches": {b: sum(1 for p in tok_prompts if p["bench"] == b) for b in BENCHES},
        "embed_out": finalize(acc_embed),
        "final_norm_out": finalize(acc_norm),
        "logits": finalize(acc_logits),
        "logits_margin_median": st.median(margins),
        "per_layer": per_layer,
        "weight_stats": weight_stats,
        "down_gemm_deep_dive": deep,
    }
    atomic_write_json(os.path.join(results_dir, "magnitudes.json"), result)
    print("Experiment 3 complete.", flush=True)


if __name__ == "__main__":
    main()
