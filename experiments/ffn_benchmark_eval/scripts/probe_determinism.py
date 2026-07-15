"""Probe run-to-run determinism of one benchmark under a given vLLM config.

Runs the first `--nrows` prompts of a protocol benchmark once, writes responses
to a file keyed by a --slot label.  Invoke twice (slot A and B, separate
processes) with the same config, then diff the two files to measure cross-run
response/correctness divergence.
"""
from __future__ import annotations
import argparse, json, os, time
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import common


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="gsm8k")
    ap.add_argument("--nrows", type=int, default=100)
    ap.add_argument("--slot", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--prefix-caching", type=int, default=1)
    ap.add_argument("--eager", type=int, default=0)
    ap.add_argument("--batch-invariant", type=int, default=0)
    ap.add_argument("--max-num-seqs", type=int, default=0)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.batch_invariant:
        os.environ["VLLM_BATCH_INVARIANT"] = "1"

    from vllm import LLM, SamplingParams
    bench_data, protocol = common.resolve_bench(args.benchmark)
    rows = common.load_rows(bench_data.data_path)[: args.nrows]
    prompts = [common.build_prompt(r, bench_data, protocol) for r in rows]
    stop = common.effective_stop(protocol)
    max_tokens = int(protocol.generation_kwargs.get("max_new_tokens", 256))

    kw = dict(model="/nvme0/if/models/Qwen3-4B", tensor_parallel_size=1, dtype="bfloat16",
              gpu_memory_utilization=0.90, max_model_len=4096,
              enable_prefix_caching=bool(args.prefix_caching),
              enforce_eager=bool(args.eager), trust_remote_code=True)
    if args.max_num_seqs:
        kw["max_num_seqs"] = args.max_num_seqs
    t0 = time.time()
    llm = LLM(**kw)
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, stop=stop or None)
    outs = llm.generate(prompts, sp)
    gen_s = time.time() - t0
    recs = {}
    for row, o in zip(rows, outs):
        resp = o.outputs[0].text
        corr, ext = common.score_response(row, resp, protocol)
        recs[row.sample_id] = {"response": resp, "correct": bool(corr)}
    json.dump({"gen_s": gen_s, "recs": recs}, open(args.out, "w"))
    print(f"[probe {args.slot}] {args.benchmark} n={len(rows)} gen={gen_s:.1f}s wrote {args.out}")


if __name__ == "__main__":
    main()
