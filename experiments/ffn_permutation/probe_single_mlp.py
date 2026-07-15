"""Stage B — isolated real-MLP experiments on Qwen3-4B (layers 0 / 17 / 35).

For each sampled layer, each input, and each permutation case of the
pre-registered 10-case control table, run the actual `Qwen3MLP` module with
in-place permuted weights and measure output drift vs. the unpermuted
baseline. All computation in BF16 on the GPU. Weights are restored in a
try/finally by inverse indexing and verified against a byte-exact master
copy plus SHA-256 after every case; the run aborts on any mismatch.

Extra structural checks for valid-triplet cases:
  - gate/up/product coordinate alignment (bitwise);
  - canonical-down path (un-permute h, then original down) must be bitwise
    equal to baseline, isolating GEMM reduction order as the error source.
"""

import argparse
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from permutation import (
    append_jsonl,
    apply_case,
    atomic_write_json,
    diff_metrics,
    env_info,
    inverse_perm,
    invert_case,
    make_perm,
    mlp_checksums,
    tensor_sha256,
)

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(HERE)), "models", "Qwen3-4B"
)
DEFAULT_RESULTS_DIR = os.path.join(HERE, "results")

LAYERS = [0, 17, 35]
PERM_SEEDS = [42, 43, 44, 45, 46]
SPECIAL_PERMS = ["identity", "adjacent_swap", "reverse"]
REAL_PROMPT = (
    "请阅读下面的短文并回答问题。李华是一名中学教师，他每天早上六点起床，"
    "先去学校旁边的公园跑步半小时，然后回家吃早餐。问题：李华早上做什么运动？"
)


def build_inputs(hidden_size: int, device) -> dict:
    inputs = {}
    for name, seed, scale in [
        ("randn-s7-x1", 7, 1.0),
        ("randn-s8-x1", 8, 1.0),
        ("randn-s7-x0.1", 7, 0.1),
        ("randn-s7-x10", 7, 10.0),
    ]:
        g = torch.Generator().manual_seed(seed)
        x = (torch.randn(4, 64, hidden_size, generator=g) * scale).bfloat16()
        inputs[name] = x.to(device)
    return inputs


def capture_real_mlp_inputs(model, tokenizer, layers, device) -> dict:
    """Real MLP inputs (post attention + layernorm) for a fixed prompt."""
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": REAL_PROMPT}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    enc = tokenizer(text, return_tensors="pt").to(device)
    captured = {}
    hooks = []
    for li in layers:
        def mk(li):
            def pre_hook(module, args):
                captured[li] = args[0].detach().clone()
            return pre_hook
        hooks.append(model.model.layers[li].mlp.register_forward_pre_hook(mk(li)))
    with torch.inference_mode():
        model(**enc, use_cache=False)
    for h in hooks:
        h.remove()
    return captured


def control_table(m: int, seed: int) -> list:
    perm = make_perm("random", m, seed)
    inv = inverse_perm(perm)
    pg = make_perm("random", m, seed + 100)
    pu = make_perm("random", m, seed + 200)
    pd = make_perm("random", m, seed + 300)
    return [
        ("baseline-repeat", {"gate": None, "up": None, "down": None}),
        ("valid-triplet", {"gate": perm, "up": perm, "down": perm}),
        ("gate-only", {"gate": perm, "up": None, "down": None}),
        ("up-only", {"gate": None, "up": perm, "down": None}),
        ("down-only", {"gate": None, "up": None, "down": perm}),
        ("gate+up", {"gate": perm, "up": perm, "down": None}),
        ("gate+down", {"gate": perm, "up": None, "down": perm}),
        ("up+down", {"gate": None, "up": perm, "down": perm}),
        ("independent-triplet", {"gate": pg, "up": pu, "down": pd}),
        ("wrong-direction", {"gate": perm, "up": perm, "down": inv}),
    ]


@torch.inference_mode()
def mlp_parts(mlp, x):
    g = torch.nn.functional.silu(
        torch.nn.functional.linear(x, mlp.gate_proj.weight)
    )
    u = torch.nn.functional.linear(x, mlp.up_proj.weight)
    h = g * u
    y = torch.nn.functional.linear(h, mlp.down_proj.weight)
    return g, u, h, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    ap.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    args = ap.parse_args()
    model_path = os.path.abspath(args.model_path)
    results_dir = os.path.abspath(args.results_dir)
    results = os.path.join(results_dir, "single_mlp.jsonl")
    manifest_path = os.path.join(results_dir, "single_mlp_manifest.json")
    os.makedirs(results_dir, exist_ok=True)

    done = set()
    if args.resume and os.path.exists(results):
        with open(results) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["case_key"])
                except Exception:
                    pass

    device = "cuda"
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map=device, local_files_only=True
    )
    model.eval()
    m = model.config.intermediate_size

    inputs = build_inputs(model.config.hidden_size, device)
    real = capture_real_mlp_inputs(model, tokenizer, LAYERS, device)

    manifest = {
        "env": env_info(), "model_path": model_path,
        "cases": {}, "aborted": False,
    }

    for li in LAYERS:
        mlp = model.model.layers[li].mlp
        master = {
            "gate": mlp.gate_proj.weight.detach().clone(),
            "up": mlp.up_proj.weight.detach().clone(),
            "down": mlp.down_proj.weight.detach().clone(),
        }
        sha0 = mlp_checksums(mlp)
        layer_inputs = dict(inputs)
        layer_inputs["real-hidden"] = real[li].bfloat16()

        # baselines per input (parts + module output), computed once
        base = {}
        with torch.inference_mode():
            for iname, x in layer_inputs.items():
                g0, u0, h0, y0 = mlp_parts(mlp, x)
                y_mod = mlp(x)
                assert torch.equal(y_mod, y0), "functional path != module path"
                y0b = mlp(x)
                base[iname] = {
                    "g": g0, "u": u0, "h": h0, "y": y0,
                    "deterministic": bool(torch.equal(y0b, y0)),
                }

        case_specs = []
        for kind in SPECIAL_PERMS:
            p = make_perm(kind, m)
            case_specs.append(
                (f"L{li}:{kind}:valid-triplet",
                 {"gate": p, "up": p, "down": p}, kind, "valid-triplet")
            )
        for seed in PERM_SEEDS:
            for cname, cdict in control_table(m, seed):
                case_specs.append(
                    (f"L{li}:random-s{seed}:{cname}", cdict, f"random-s{seed}", cname)
                )

        for case_key, cdict, perm_name, cname in case_specs:
            if case_key in done:
                manifest["cases"][case_key] = "skipped(resume)"
                continue
            rec = {
                "case_key": case_key, "layer": li,
                "perm": perm_name, "control": cname, "inputs": {},
            }
            try:
                apply_case(mlp, cdict)
                with torch.inference_mode():
                    for iname, x in layer_inputs.items():
                        b = base[iname]
                        y1 = mlp(x)
                        entry = {
                            "vs_baseline": diff_metrics(y1, b["y"]),
                            "baseline_deterministic": b["deterministic"],
                        }
                        if cname == "valid-triplet":
                            perm = cdict["gate"].to(device)
                            inv = inverse_perm(cdict["gate"]).to(device)
                            g1, u1, h1, _ = mlp_parts(mlp, x)
                            y_canon = torch.nn.functional.linear(
                                h1[..., inv], master["down"]
                            )
                            entry.update({
                                "gate_coordinate_equal": bool(
                                    torch.equal(g1, b["g"][..., perm])),
                                "up_coordinate_equal": bool(
                                    torch.equal(u1, b["u"][..., perm])),
                                "product_coordinate_equal": bool(
                                    torch.equal(h1, b["h"][..., perm])),
                                "canonical_down_bitwise_equal": bool(
                                    torch.equal(y_canon, b["y"])),
                                "canonical_vs_baseline": diff_metrics(y_canon, b["y"]),
                            })
                        rec["inputs"][iname] = entry
            finally:
                invert_case(mlp, cdict)

            restore_ok = all(
                torch.equal(getattr(mlp, f"{k}_proj").weight, master[k])
                for k in master
            )
            sha_now = mlp_checksums(mlp)
            rec["restore_equal"] = restore_ok
            rec["restore_sha_match"] = sha_now == sha0
            append_jsonl(results, rec)
            manifest["cases"][case_key] = "done"
            if not (restore_ok and rec["restore_sha_match"]):
                manifest["aborted"] = True
                atomic_write_json(manifest_path, manifest)
                print(f"FATAL: restore mismatch after {case_key}; aborting.")
                sys.exit(2)
            print(f"done {case_key}")

        del master
        torch.cuda.empty_cache()

    atomic_write_json(manifest_path, manifest)
    print("Stage B complete.")


if __name__ == "__main__":
    main()
