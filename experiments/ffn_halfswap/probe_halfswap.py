"""Experiment 2 — half-swap: exchange the first and second half of every
FFN's intermediate neurons in all 36 layers of Qwen3-4B.

Permutation: perm = [m/2 .. m-1, 0 .. m/2-1]  (block swap, an involution),
applied as the validated coupled triplet (gate/up rows + down columns) to
every layer. BF16 throughout, same as experiment 1.

Protocol:
  - Inputs: the same 32 tokenized prompts as experiment 1
    (../ffn_permutation/results/tokenized.json), reused bit-identically.
  - Forward: 2 repeats per condition (determinism check); full-logits and
    last-token metrics vs baseline.
  - Generation: temperature-0 greedy (do_sample=False), 64 new tokens,
    8 independent repeats per prompt per condition. Within-condition
    repeats must be token-identical for cross-condition diffs to be
    attributable to the permutation.
  - Weights restored via inverse permutation (block swap is its own
    inverse; invert_case uses argsort as usual) and verified byte-exactly
    against a CPU master copy.
"""

import argparse
import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
EXP1 = os.path.join(os.path.dirname(HERE), "ffn_permutation")
sys.path.insert(0, EXP1)
from permutation import (
    append_jsonl,
    apply_case,
    atomic_write_json,
    env_info,
    invert_case,
)
from probe_full_model import (
    logits_comparison,
    snapshot_mlp_weights_cpu,
    verify_restore,
    all_mlp_sha,
)

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(HERE)), "models", "Qwen3-4B"
)
DEFAULT_RESULTS_DIR = os.path.join(HERE, "results")
DEFAULT_TOKENIZED = os.path.join(EXP1, "results", "tokenized.json")

N_GEN_REPEATS = 8
GEN_TOKENS = 64
FWD_REPEATS = 2


@torch.inference_mode()
def forward_logits(model, input_ids):
    out = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        use_cache=False,
    )
    return out.logits[0].to("cpu")  # [T, V]


@torch.inference_mode()
def greedy_generate(model, tokenizer, input_ids):
    out = model.generate(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        max_new_tokens=GEN_TOKENS,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        pad_token_id=tokenizer.eos_token_id,
    )
    return out[0, input_ids.shape[1]:].tolist()


def run_condition(model, tokenizer, prompts, device):
    """Forward (FWD_REPEATS×) + generation (N_GEN_REPEATS×) for all prompts
    under the model's current weights."""
    cond = {}
    for p in prompts:
        ids = torch.tensor([p["input_ids"]], device=device)
        logits_runs = [forward_logits(model, ids) for _ in range(FWD_REPEATS)]
        gen_runs = [
            greedy_generate(model, tokenizer, ids) for _ in range(N_GEN_REPEATS)
        ]
        cond[p["id"]] = {
            "logits": logits_runs[0],
            "forward_repeats_bitwise": all(
                torch.equal(l, logits_runs[0]) for l in logits_runs[1:]
            ),
            "gen_runs": gen_runs,
            "gen_repeats_identical": all(g == gen_runs[0] for g in gen_runs[1:]),
        }
        print(f"  prompt {p['id']:2d}: fwd_det={cond[p['id']]['forward_repeats_bitwise']} "
              f"gen_det={cond[p['id']]['gen_repeats_identical']}", flush=True)
    return cond


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    ap.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    ap.add_argument("--tokenized", default=DEFAULT_TOKENIZED)
    args = ap.parse_args()
    model_path = os.path.abspath(args.model_path)
    results_dir = os.path.abspath(args.results_dir)
    tokenized = os.path.abspath(args.tokenized)

    device = "cuda"
    os.makedirs(results_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map=device, local_files_only=True
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    m = model.config.intermediate_size

    with open(tokenized) as f:
        prompts = json.load(f)["prompts"]

    # block-swap permutation: first half <-> second half
    half = m // 2
    perm = torch.cat([torch.arange(half, m), torch.arange(0, half)])
    assert torch.equal(torch.sort(perm).values, torch.arange(m))
    assert torch.equal(torch.argsort(perm), perm), "block swap must be an involution"

    master = snapshot_mlp_weights_cpu(model)
    sha_start = all_mlp_sha(model)
    manifest = {
        "env": env_info(),
        "model_path": model_path,
        "tokenized_path": tokenized,
        "m": m,
        "half": half,
        "n_layers": n_layers,
        "n_gen_repeats": N_GEN_REPEATS,
        "gen_tokens": GEN_TOKENS,
        "sha_all_mlp_start": sha_start,
    }

    t0 = time.time()
    print("=== condition: baseline ===", flush=True)
    base = run_condition(model, tokenizer, prompts, device)
    print(f"baseline done in {time.time()-t0:.0f}s", flush=True)

    case = {li: {"gate": perm, "up": perm, "down": perm} for li in range(n_layers)}
    t0 = time.time()
    print("=== condition: half-swap (all 36 layers) ===", flush=True)
    try:
        for li, cd in case.items():
            apply_case(model.model.layers[li].mlp, cd)
        swap = run_condition(model, tokenizer, prompts, device)
    finally:
        for li, cd in case.items():
            invert_case(model.model.layers[li].mlp, cd)
    print(f"half-swap done in {time.time()-t0:.0f}s", flush=True)

    restore_ok = verify_restore(model, master, list(range(n_layers)))
    sha_end = all_mlp_sha(model)
    manifest["restore_equal"] = restore_ok
    manifest["sha_all_mlp_end"] = sha_end
    manifest["sha_match"] = sha_end == sha_start
    print(f"restore_equal={restore_ok} sha_match={manifest['sha_match']}", flush=True)

    # ---- comparisons -------------------------------------------------------
    out_path = os.path.join(results_dir, "halfswap.jsonl")
    if os.path.exists(out_path):
        os.unlink(out_path)
    for p in prompts:
        b, s = base[p["id"]], swap[p["id"]]
        gen_b, gen_s = b["gen_runs"][0], s["gen_runs"][0]
        div = next(
            (i for i, (x, y) in enumerate(zip(gen_b, gen_s)) if x != y), None
        )
        if div is None and len(gen_b) != len(gen_s):
            div = min(len(gen_b), len(gen_s))
        rec = {
            "id": p["id"],
            "tag": p["tag"],
            "baseline_fwd_det": b["forward_repeats_bitwise"],
            "halfswap_fwd_det": s["forward_repeats_bitwise"],
            "baseline_gen_det": b["gen_repeats_identical"],
            "halfswap_gen_det": s["gen_repeats_identical"],
            "logits": logits_comparison(s["logits"], b["logits"]),
            "gen_exact_match": gen_b == gen_s,
            "gen_first_divergence": div,
            "baseline_text": tokenizer.decode(gen_b, skip_special_tokens=True),
            "halfswap_text": tokenizer.decode(gen_s, skip_special_tokens=True),
        }
        append_jsonl(out_path, rec)

    atomic_write_json(os.path.join(results_dir, "manifest.json"), manifest)
    print("Experiment 2 complete.", flush=True)
    if not (restore_ok and manifest["sha_match"]):
        sys.exit(2)


if __name__ == "__main__":
    main()
