"""Stage C — full-model propagation experiments on Qwen3-4B (BF16).

Per pre-registered plan §5C:
  - 32 fixed prompts, tokenized once and persisted; every case reuses the
    exact same input_ids (per-prompt forward, no padding).
  - Cases: baseline-repeat; valid-triplet permutation of layer {0|17|35},
    layers 0-5, 0-17, 0-35, each with seeds 42/43/44; negative controls
    (gate-only / up-only / down-only / independent-triplet) at layer 17.
  - Hooks capture per-layer MLP input, MLP output and decoder-block output;
    we record the first place a difference appears and per-layer drift.
  - Full logits compared over all tokens (online, never written to disk);
    last-token logits metrics; top-1 / top-5 agreement; flip margins.
  - Greedy generation (do_sample=False) for baseline and all-36 cases.
  - After every case, weights are restored by inverse permutation and
    verified byte-exactly against a CPU master copy; abort on mismatch.
"""

import argparse
import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from permutation import (
    append_jsonl,
    apply_case,
    atomic_write_json,
    diff_metrics,
    env_info,
    invert_case,
    make_perm,
    tensor_sha256,
)

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(HERE)), "models", "Qwen3-4B"
)
DEFAULT_RESULTS_DIR = os.path.join(HERE, "results")
PROMPTS = os.path.join(HERE, "prompts.json")

SEEDS = [42, 43, 44]
GEN_TOKENS = 64


# ---------------------------------------------------------------------------
# case construction
# ---------------------------------------------------------------------------

def layer_perms(m: int, layers: list, seed: int) -> dict:
    """One independent random perm per layer, drawn sequentially from a
    generator seeded with `seed` (deterministic, layer order ascending)."""
    g = torch.Generator().manual_seed(seed)
    return {li: torch.randperm(m, generator=g) for li in sorted(layers)}


def build_cases(m: int) -> list:
    """Returns list of (case_key, control, seed, {layer: case_dict})."""
    cases = []
    ranges = [
        ("one-layer-first", [0]),
        ("one-layer-middle", [17]),
        ("one-layer-last", [35]),
        ("prefix-6", list(range(6))),
        ("half-18", list(range(18))),
        ("all-36", list(range(36))),
    ]
    for name, layers in ranges:
        for seed in SEEDS:
            perms = layer_perms(m, layers, seed)
            cd = {
                li: {"gate": p, "up": p, "down": p} for li, p in perms.items()
            }
            cases.append((f"{name}:s{seed}", "valid-triplet", seed, cd))
    # negative controls, layer 17 only, seed 42
    p = layer_perms(m, [17], 42)[17]
    pg = make_perm("random", m, 42 + 100)
    pu = make_perm("random", m, 42 + 200)
    pd = make_perm("random", m, 42 + 300)
    for cname, cdict in [
        ("gate-only", {"gate": p, "up": None, "down": None}),
        ("up-only", {"gate": None, "up": p, "down": None}),
        ("down-only", {"gate": None, "up": None, "down": p}),
        ("independent-triplet", {"gate": pg, "up": pu, "down": pd}),
    ]:
        cases.append((f"neg-L17-{cname}:s42", cname, 42, {17: cdict}))
    return cases


# ---------------------------------------------------------------------------
# hooks
# ---------------------------------------------------------------------------

class StreamCapture:
    def __init__(self, model):
        self.model = model
        self.n_layers = model.config.num_hidden_layers
        self.data = {}
        self.handles = []

    def __enter__(self):
        for li in range(self.n_layers):
            layer = self.model.model.layers[li]

            def mk_pre(li):
                def f(module, a):
                    self.data[(li, "mlp_in")] = a[0].detach()
                return f

            def mk_post(li):
                def f(module, a, out):
                    self.data[(li, "mlp_out")] = out.detach()
                return f

            def mk_block(li):
                def f(module, a, out):
                    o = out[0] if isinstance(out, tuple) else out
                    self.data[(li, "block_out")] = o.detach()
                return f

            self.handles.append(layer.mlp.register_forward_pre_hook(mk_pre(li)))
            self.handles.append(layer.mlp.register_forward_hook(mk_post(li)))
            self.handles.append(layer.register_forward_hook(mk_block(li)))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []


@torch.inference_mode()
def forward_with_streams(model, input_ids):
    cap_out = {}
    with StreamCapture(model) as cap:
        out = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            use_cache=False,
        )
        for k, v in cap.data.items():
            cap_out[k] = v.to("cpu")
    return out.logits[0].to("cpu"), cap_out  # logits: [T, V]


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def logits_comparison(lg_case: torch.Tensor, lg_base: torch.Tensor) -> dict:
    a = lg_case.float()
    b = lg_base.float()
    full = diff_metrics(lg_case, lg_base)

    top1_b = b.argmax(-1)
    top1_a = a.argmax(-1)
    agree = top1_a == top1_b
    top2v_b = b.topk(2, dim=-1).values
    margin_b = (top2v_b[:, 0] - top2v_b[:, 1])  # baseline top1-top2 margin
    flips = (~agree).nonzero().flatten()
    top5_b = b.topk(5, dim=-1).indices
    top5_a = a.topk(5, dim=-1).indices
    jacc = []
    exact5 = 0
    for i in range(a.shape[0]):
        sa, sb = set(top5_a[i].tolist()), set(top5_b[i].tolist())
        jacc.append(len(sa & sb) / len(sa | sb))
        exact5 += int(sa == sb)

    lt = diff_metrics(lg_case[-1], lg_base[-1])
    return {
        "full": full,
        "n_tokens": int(a.shape[0]),
        "top1_agreement": agree.float().mean().item(),
        "n_top1_flips": int(flips.numel()),
        "flip_positions": flips[:50].tolist(),
        "flip_baseline_margins": margin_b[flips][:50].tolist(),
        "median_baseline_margin": margin_b.median().item(),
        "top5_jaccard_mean": sum(jacc) / len(jacc),
        "top5_exact_frac": exact5 / a.shape[0],
        "last_token": {
            **lt,
            "top1_same": bool(top1_a[-1] == top1_b[-1]),
            "top5_same": bool(
                set(top5_a[-1].tolist()) == set(top5_b[-1].tolist())
            ),
            "baseline_margin": margin_b[-1].item(),
        },
    }


def stream_comparison(cap_case: dict, cap_base: dict, n_layers: int) -> dict:
    per_layer = {s: [] for s in ("mlp_in", "mlp_out", "block_out")}
    bitwise = {s: [] for s in ("mlp_in", "mlp_out", "block_out")}
    first_diff = None
    for li in range(n_layers):
        for stream in ("mlp_in", "mlp_out", "block_out"):
            a = cap_case[(li, stream)]
            b = cap_base[(li, stream)]
            eq = torch.equal(a, b)
            bitwise[stream].append(eq)
            d = (a.float() - b.float()).norm().item()
            n = max(b.float().norm().item(), 1e-12)
            per_layer[stream].append(d / n)
            if not eq and first_diff is None:
                tok = int(
                    (a != b).any(-1).flatten().nonzero().flatten()[0].item()
                )
                first_diff = {"layer": li, "stream": stream, "first_token": tok}
    return {
        "first_diff": first_diff,
        "per_layer_rel_l2": per_layer,
        "per_layer_bitwise": bitwise,
    }


# ---------------------------------------------------------------------------
# weight management
# ---------------------------------------------------------------------------

def snapshot_mlp_weights_cpu(model) -> dict:
    out = {}
    for li in range(model.config.num_hidden_layers):
        mlp = model.model.layers[li].mlp
        out[li] = {
            "gate": mlp.gate_proj.weight.detach().cpu().clone(),
            "up": mlp.up_proj.weight.detach().cpu().clone(),
            "down": mlp.down_proj.weight.detach().cpu().clone(),
        }
    return out


def verify_restore(model, master: dict, layers: list) -> bool:
    for li in layers:
        mlp = model.model.layers[li].mlp
        for k in ("gate", "up", "down"):
            if not torch.equal(
                getattr(mlp, f"{k}_proj").weight.detach().cpu(), master[li][k]
            ):
                return False
    return True


def all_mlp_sha(model) -> str:
    h = []
    for li in range(model.config.num_hidden_layers):
        mlp = model.model.layers[li].mlp
        for k in ("gate", "up", "down"):
            h.append(tensor_sha256(getattr(mlp, f"{k}_proj").weight))
    import hashlib

    return hashlib.sha256("".join(h).encode()).hexdigest()


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------

@torch.inference_mode()
def greedy_generate(model, tokenizer, input_ids) -> dict:
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
    new = out[0, input_ids.shape[1]:]
    return {
        "token_ids": new.tolist(),
        "text": tokenizer.decode(new, skip_special_tokens=True),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="4 prompts, baseline + one-layer-middle:s42 only")
    ap.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    ap.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    args = ap.parse_args()
    model_path = os.path.abspath(args.model_path)
    results_dir = os.path.abspath(args.results_dir)
    results = os.path.join(results_dir, "full_model.jsonl")
    gen_results = os.path.join(results_dir, "full_model_generation.jsonl")
    manifest_path = os.path.join(results_dir, "full_model_manifest.json")
    tokenized = os.path.join(results_dir, "tokenized.json")
    os.makedirs(results_dir, exist_ok=True)

    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map=device, local_files_only=True
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    m = model.config.intermediate_size

    # -- tokenization (once, persisted, reused bit-identically) -------------
    if os.path.exists(tokenized):
        with open(tokenized) as f:
            tok = json.load(f)
    else:
        with open(PROMPTS) as f:
            pdata = json.load(f)
        tok = {"prompts": []}
        for p in pdata["prompts"]:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": p["text"]}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            ids = tokenizer(text)["input_ids"]
            tok["prompts"].append(
                {"id": p["id"], "tag": p["tag"], "input_ids": ids}
            )
        atomic_write_json(tokenized, tok)
    prompts = tok["prompts"]
    if args.smoke:
        prompts = prompts[:4]

    done = set()
    if args.resume and os.path.exists(results):
        with open(results) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["case_key"])
                except Exception:
                    pass
    gen_done = set()
    if args.resume and os.path.exists(gen_results):
        with open(gen_results) as f:
            for line in f:
                try:
                    gen_done.add(json.loads(line)["case_key"])
                except Exception:
                    pass

    master = snapshot_mlp_weights_cpu(model)
    sha_start = all_mlp_sha(model)
    manifest = {
        "env": env_info(),
        "model_path": model_path,
        "sha_all_mlp_start": sha_start,
        "n_prompts": len(prompts),
        "cases": {},
    }
    atomic_write_json(manifest_path, manifest)

    # -- baseline capture ----------------------------------------------------
    t0 = time.time()
    base = {}
    for p in prompts:
        ids = torch.tensor([p["input_ids"]], device=device)
        logits, streams = forward_with_streams(model, ids)
        base[p["id"]] = {"logits": logits, "streams": streams}
    print(f"baseline capture done in {time.time()-t0:.1f}s", flush=True)

    # -- baseline-repeat -----------------------------------------------------
    if "baseline-repeat" not in done:
        rec = {"case_key": "baseline-repeat", "control": "baseline",
               "seed": None, "layers": [], "prompts": []}
        for p in prompts:
            ids = torch.tensor([p["input_ids"]], device=device)
            logits, streams = forward_with_streams(model, ids)
            b = base[p["id"]]
            streams_eq = all(
                torch.equal(streams[k], b["streams"][k]) for k in streams
            )
            rec["prompts"].append({
                "id": p["id"],
                "logits_bitwise": bool(torch.equal(logits, b["logits"])),
                "streams_bitwise": bool(streams_eq),
                "logits": logits_comparison(logits, b["logits"]),
            })
        append_jsonl(results, rec)
        manifest["cases"]["baseline-repeat"] = "done"
        atomic_write_json(manifest_path, manifest)
        print("baseline-repeat done", flush=True)

    # -- permutation cases ---------------------------------------------------
    cases = build_cases(m)
    if args.smoke:
        cases = [c for c in cases if c[0] == "one-layer-middle:s42"]

    for case_key, control, seed, layer_cases in cases:
        if case_key in done:
            continue
        t0 = time.time()
        rec = {
            "case_key": case_key,
            "control": control,
            "seed": seed,
            "layers": sorted(layer_cases.keys()),
            "prompts": [],
        }
        try:
            for li, cd in layer_cases.items():
                apply_case(model.model.layers[li].mlp, cd)
            for p in prompts:
                ids = torch.tensor([p["input_ids"]], device=device)
                logits, streams = forward_with_streams(model, ids)
                b = base[p["id"]]
                rec["prompts"].append({
                    "id": p["id"],
                    **stream_comparison(streams, b["streams"], n_layers),
                    "logits": logits_comparison(logits, b["logits"]),
                })
        finally:
            for li, cd in layer_cases.items():
                invert_case(model.model.layers[li].mlp, cd)

        ok = verify_restore(model, master, sorted(layer_cases.keys()))
        rec["restore_equal"] = ok
        rec["elapsed_s"] = time.time() - t0
        append_jsonl(results, rec)
        manifest["cases"][case_key] = "done"
        atomic_write_json(manifest_path, manifest)
        print(f"case {case_key} done in {rec['elapsed_s']:.1f}s restore={ok}",
              flush=True)
        if not ok:
            print("FATAL: restore mismatch; aborting.")
            sys.exit(2)

    # -- greedy generation: baseline + all-36 seeds --------------------------
    gen_base = {}
    if "gen-baseline" not in gen_done:
        rec = {"case_key": "gen-baseline", "prompts": []}
        for p in prompts:
            ids = torch.tensor([p["input_ids"]], device=device)
            g = greedy_generate(model, tokenizer, ids)
            gen_base[p["id"]] = g
            rec["prompts"].append({"id": p["id"], **g})
        append_jsonl(gen_results, rec)
        print("gen-baseline done", flush=True)
    else:
        with open(gen_results) as f:
            for line in f:
                r = json.loads(line)
                if r["case_key"] == "gen-baseline":
                    gen_base = {q["id"]: q for q in r["prompts"]}

    gen_cases = [] if args.smoke else [
        (f"gen-all-36:s{seed}", layer_perms(m, list(range(n_layers)), seed))
        for seed in SEEDS
    ]
    for case_key, perms in gen_cases:
        if case_key in gen_done:
            continue
        layer_cases = {
            li: {"gate": p, "up": p, "down": p} for li, p in perms.items()
        }
        rec = {"case_key": case_key, "prompts": []}
        try:
            for li, cd in layer_cases.items():
                apply_case(model.model.layers[li].mlp, cd)
            for p in prompts:
                ids = torch.tensor([p["input_ids"]], device=device)
                g = greedy_generate(model, tokenizer, ids)
                gb = gen_base[p["id"]]
                ids_a, ids_b = g["token_ids"], gb["token_ids"]
                div = next(
                    (i for i, (x, y) in enumerate(zip(ids_a, ids_b)) if x != y),
                    None,
                )
                if div is None and len(ids_a) != len(ids_b):
                    div = min(len(ids_a), len(ids_b))
                rec["prompts"].append({
                    "id": p["id"],
                    "exact_match": ids_a == ids_b,
                    "first_divergence": div,
                    "text": g["text"],
                    "baseline_text": gb["text"],
                })
        finally:
            for li, cd in layer_cases.items():
                invert_case(model.model.layers[li].mlp, cd)
        ok = verify_restore(model, master, list(layer_cases.keys()))
        rec["restore_equal"] = ok
        append_jsonl(gen_results, rec)
        n_match = sum(q["exact_match"] for q in rec["prompts"])
        print(f"{case_key} done: exact_match {n_match}/{len(rec['prompts'])} "
              f"restore={ok}", flush=True)
        if not ok:
            sys.exit(2)

    sha_end = all_mlp_sha(model)
    manifest["sha_all_mlp_end"] = sha_end
    manifest["sha_match"] = sha_end == sha_start
    atomic_write_json(manifest_path, manifest)
    print(f"Stage C complete. sha_match={manifest['sha_match']}")
    if not manifest["sha_match"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
