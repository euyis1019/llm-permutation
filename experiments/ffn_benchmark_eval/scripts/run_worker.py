"""GPU worker: load one checkpoint once, run the requested benchmarks in-process.

Loads a vLLM engine for the given model directory, then runs any subset of the
six benchmarks (mmlu, gsm8k, ceval, cmmlu, humaneval_plus, mbpp_plus) with a
single weight load.  Every benchmark writes a per-sample raw JSON with enough
detail for the paired analysis (text, extracted answer, correctness).

Determinism / config are frozen in configs/frozen_config.json and identical for
baseline and permuted checkpoints of the same model family.  Greedy decoding
(temperature 0, n_runs=1); no sampling, no avg@k.

Resumable: a benchmark whose raw file already exists and parses is skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Reduce fragmentation-driven OOM under GPU sharing.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import common  # noqa: E402

EVALPLUS_DATA = common.BENCH_ROOT / "datasets" / "benchmark" / "evalplus"
HE_PATH = EVALPLUS_DATA / "HumanEvalPlus-v0.1.10.jsonl"
MBPP_PATH = EVALPLUS_DATA / "MbppPlus-v0.2.0.jsonl"

# Set the dataset overrides globally so both the in-process loader and the
# evalplus.evaluate subprocess read the vendored files.  The evalplus expected-
# output .pkl cache is pre-warmed single-process (scripts/warm_evalplus.py)
# before any parallel worker runs, so workers only ever read the cache.
os.environ["HUMANEVAL_OVERRIDE_PATH"] = str(HE_PATH)
os.environ["MBPP_OVERRIDE_PATH"] = str(MBPP_PATH)

# code-generation stop sequences (mirrored from bench gen_code_completions.py)
BASE_EOS = ["<|endoftext|>", "<|endofmask|>", "</s>", "\nif __name__", "\ndef main(", "\nprint("]
EXTRA_EOS = {
    "humaneval": ["\ndef ", "\nclass ", "\nimport ", "\nfrom ", "\nassert "],
    "mbpp": ['\n"""', "\nassert"],
}
CODE_MAX_TOKENS = 768


def raw_path(out_dir: str, model_tag: str, benchmark: str) -> Path:
    return Path(out_dir) / model_tag / f"{benchmark}.raw.json"


def _done(p: Path) -> bool:
    if not p.is_file():
        return False
    try:
        d = json.loads(p.read_text())
        return d.get("complete") is True
    except Exception:
        return False


# ── protocol benchmarks ───────────────────────────────────────────────────────

def run_protocol(llm, benchmark: str, out_dir: str, model_tag: str) -> dict:
    from vllm import SamplingParams

    bench_data, protocol = common.resolve_bench(benchmark)
    rows = common.load_rows(bench_data.data_path)
    prompts = [common.build_prompt(r, bench_data, protocol) for r in rows]
    stop = common.effective_stop(protocol)
    max_tokens = int(protocol.generation_kwargs.get("max_new_tokens", 256))
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, stop=stop or None)

    t0 = time.time()
    outs = llm.generate(prompts, sp)
    gen_s = time.time() - t0
    texts = [o.outputs[0].text for o in outs]

    samples = []
    n_correct = 0
    for row, prompt, resp in zip(rows, prompts, texts):
        try:
            correct, extracted = common.score_response(row, resp, protocol)
        except Exception as exc:
            correct, extracted = False, f"<scorer-error {exc!r}>"
        n_correct += int(correct)
        samples.append({
            "sample_id": row.sample_id,
            "target": row.target,
            "response": resp,
            "extracted": extracted,
            "correct": bool(correct),
        })
    result = {
        "benchmark": benchmark,
        "model_tag": model_tag,
        "kind": "protocol",
        "total": len(rows),
        "correct": n_correct,
        "accuracy": n_correct / len(rows),
        "gen_seconds": round(gen_s, 1),
        "max_tokens": max_tokens,
        "stop": stop,
        "complete": True,
        "samples": samples,
    }
    common.atomic_write_json(raw_path(out_dir, model_tag, benchmark), result)
    print(f"[worker] {model_tag}/{benchmark}: {n_correct}/{len(rows)} = "
          f"{result['accuracy']:.3%}  ({gen_s:.1f}s)")
    return result


# ── code benchmarks (evalplus scoring) ────────────────────────────────────────

def _load_evalplus(dataset: str):
    os.environ["HUMANEVAL_OVERRIDE_PATH"] = str(HE_PATH)
    os.environ["MBPP_OVERRIDE_PATH"] = str(MBPP_PATH)
    from evalplus.data import get_human_eval_plus, get_mbpp_plus
    return get_human_eval_plus() if dataset == "humaneval" else get_mbpp_plus()


def run_code(llm, benchmark: str, out_dir: str, model_tag: str, workdir: str) -> dict:
    from vllm import SamplingParams
    from evalplus.sanitize import sanitize

    dataset = "humaneval" if benchmark == "humaneval_plus" else "mbpp"
    problems = _load_evalplus(dataset)
    task_ids = list(problems)
    prompts = [problems[t]["prompt"] for t in task_ids]
    entry_points = [problems[t]["entry_point"] for t in task_ids]
    stop = BASE_EOS + EXTRA_EOS[dataset]
    sp = SamplingParams(temperature=0.0, max_tokens=CODE_MAX_TOKENS, stop=stop)

    t0 = time.time()
    outs = llm.generate(prompts, sp)
    gen_s = time.time() - t0
    completions = [o.outputs[0].text for o in outs]

    wd = Path(workdir) / model_tag / benchmark
    wd.mkdir(parents=True, exist_ok=True)
    samples_path = wd / "samples.jsonl"
    raw_solutions = {}
    with open(samples_path, "w") as f:
        for t, prompt, comp, ep in zip(task_ids, prompts, completions, entry_points):
            full = prompt + comp
            solution = sanitize(full, ep)
            raw_solutions[t] = {"completion": comp, "solution": solution}
            f.write(json.dumps({"task_id": t, "solution": solution}) + "\n")

    # run evalplus.evaluate (subprocess: it uses multiprocessing + Fire)
    env = {**os.environ, "HUMANEVAL_OVERRIDE_PATH": str(HE_PATH), "MBPP_OVERRIDE_PATH": str(MBPP_PATH)}
    proc = subprocess.run(
        [sys.executable, "-m", "evalplus.evaluate", "--dataset", dataset,
         "--samples", str(samples_path), "--parallel", "16"],
        env=env, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"evalplus.evaluate failed for {benchmark}:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")

    res_file = samples_path.with_name("samples_eval_results.json")
    ev = json.loads(res_file.read_text())["eval"]

    samples = []
    n_base = n_plus = 0
    for t in task_ids:
        entry = ev[t][0]
        base_ok = entry.get("base_status") == "pass"
        plus_ok = entry.get("plus_status") == "pass"
        n_base += int(base_ok)
        n_plus += int(plus_ok)
        samples.append({
            "sample_id": t,
            "target": "pass",
            "response": raw_solutions[t]["completion"],
            "solution": raw_solutions[t]["solution"],
            "base_pass": base_ok,
            "plus_pass": plus_ok,
            # unified 'correct' = plus pass@1 (harder EvalPlus tests)
            "correct": bool(plus_ok),
        })
    result = {
        "benchmark": benchmark,
        "model_tag": model_tag,
        "kind": "code",
        "total": len(task_ids),
        "correct": n_plus,
        "accuracy": n_plus / len(task_ids),
        "base_pass_at_1": n_base / len(task_ids),
        "plus_pass_at_1": n_plus / len(task_ids),
        "gen_seconds": round(gen_s, 1),
        "max_tokens": CODE_MAX_TOKENS,
        "stop": stop,
        "complete": True,
        "samples": samples,
    }
    common.atomic_write_json(raw_path(out_dir, model_tag, benchmark), result)
    print(f"[worker] {model_tag}/{benchmark}: base={result['base_pass_at_1']:.3%} "
          f"plus={result['plus_pass_at_1']:.3%}  ({gen_s:.1f}s)")
    return result


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--model-tag", required=True)
    ap.add_argument("--benchmarks", required=True, help="comma-separated")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workdir", default=None, help="scratch for code eval")
    ap.add_argument("--gpu", default=None)
    ap.add_argument("--gpu-mem-util", type=float, default=None,
                    help="override gpu_memory_utilization (for GPU sharing)")
    ap.add_argument("--max-model-len", type=int, default=None,
                    help="resource-only override for shared GPU operation")
    ap.add_argument("--max-num-seqs", type=int, default=None,
                    help="resource-only override for shared GPU operation")
    ap.add_argument("--engine-init-attempts", type=int, default=6,
                    help="wrapper-controlled retries; science parameters unchanged")
    ap.add_argument("--nrows", type=int, default=None, help="smoke: cap protocol rows")
    args = ap.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    workdir = args.workdir or str(common.EXP_ROOT / "logs" / "code_scratch")

    cfg = common.load_config()
    benches = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    todo = [b for b in benches if not _done(raw_path(args.out_dir, args.model_tag, b))]
    if not todo:
        print(f"[worker] {args.model_tag}: all benchmarks already complete, skipping load")
        return
    print(f"[worker] {args.model_tag}: to run {todo} on GPU {os.environ.get('CUDA_VISIBLE_DEVICES')}")

    v = cfg["vllm"]
    if v.get("batch_invariant"):
        os.environ["VLLM_BATCH_INVARIANT"] = "1"
    mem_util = args.gpu_mem_util if args.gpu_mem_util is not None else v["gpu_memory_utilization"]
    from vllm import LLM
    # Engine init can fail transiently under GPU sharing (another process holds
    # memory during the KV-cache profiling step); retry with backoff.
    llm = None
    last_exc = None
    for attempt in range(args.engine_init_attempts):
        try:
            llm = LLM(
                model=args.model_path,
                tensor_parallel_size=1,
                dtype="bfloat16",
                gpu_memory_utilization=mem_util,
                enable_prefix_caching=v.get("enable_prefix_caching", False),
                max_model_len=(args.max_model_len if args.max_model_len is not None
                               else v["max_model_len"]),
                max_num_seqs=(args.max_num_seqs if args.max_num_seqs is not None
                              else v.get("max_num_seqs", 256)),
                enforce_eager=v.get("enforce_eager", True),
                trust_remote_code=True,
            )
            break
        except Exception as exc:
            last_exc = exc
            if attempt + 1 < args.engine_init_attempts:
                print(f"[worker] engine init attempt {attempt+1} failed: {exc!r}; retrying in 45s", flush=True)
                time.sleep(45)
    if llm is None:
        raise RuntimeError(f"engine init failed after {args.engine_init_attempts} retries: {last_exc!r}")

    # apply nrows cap for smoke by truncating the selection at read time
    if args.nrows is not None:
        _orig = common.load_rows
        def _capped(path, _cap=args.nrows, _o=_orig):
            return _o(path)[:_cap]
        common.load_rows = _capped

    for b in todo:
        if b in common.PROTOCOL_BENCHES:
            run_protocol(llm, b, args.out_dir, args.model_tag)
        elif b in common.CODE_BENCHES:
            run_code(llm, b, args.out_dir, args.model_tag, workdir)
        else:
            raise ValueError(f"unknown benchmark {b}")


if __name__ == "__main__":
    main()
