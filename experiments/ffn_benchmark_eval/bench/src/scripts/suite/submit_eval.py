#!/usr/bin/env python3
"""Unified eval entry point: submit + collect.

The single sanctioned way to run the benchmark eval suite. It reads a suite
YAML (the protocol's single source of truth, validated by src.eval.suites) and,
for each bench, selects the right driver, injects the right env, and applies the
right resources — so callers never choose a driver, never forget DATASET/METRIC,
never hardcode a per-bench resource. The exact class of low-level mistakes that
hand-written hopes / the old generate_eval_hopes.py kept reproducing.

Two engines, picked by ``--engine`` (a property of the whole submission, not
per-bench; REQUIRED — no default, the submitter states it explicitly):
  vllm       in-process unified driver ``run_eval.sh`` (one job per bench, vLLM
             in-process, no server/client, no BENCH_CONCURRENCY). Within a job,
             dispatch is by the suite's ``runner`` field: protocol →
             inproc_runner, external → run_code_eval.
  fluentllm  ``run_eval_fluentllm.sh`` — starts an SGLang/FluentLLM server then
             drives it over HTTP with BENCH_CONCURRENCY=500 (protocol via
             client_runner, code via gen_code_completions + evalplus).

Resource precedence (lowest → highest): runner protocol default < suite
``spec.resources[runner]`` < command-line flag.

Subcommands
-----------
  submit   Generate + submit one hope per bench. Returns immediately with a
           {bench → run_id → expected summary.json} manifest. Never blocks.
  collect  Idempotently scan the expected summary.json files; print an
           accuracy_mean (avg@N) table when complete, list missing benches and
           exit non-zero otherwise. Meant to be polled by an agent via cron.

Usage
-----
  python -m src.scripts.suite.submit_eval submit \\
      --suite base_model_eval_v1 \\
      --model-path /path/to/pruned_model \\
      --model-tag 04b_twostage_r028 \\
      --output-dir /path/to/eval_output \\
      --engine vllm|fluentllm \\
      [--bench gsm8k mmlu ...] [--tp N] [--mem MB] [--gpus N] \\
      [--batch-cap N] [--n-runs N] [--dry-run]

  python -m src.scripts.suite.submit_eval collect \\
      --suite base_model_eval_v1 \\
      --output-dir /path/to/eval_output \\
      --model-tag 04b_twostage_r028
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from src.eval.suites import BenchmarkTaskSpec, load_suite_by_name

# ── Constants ───────────────────────────────────────────────────────────────

QUEUE = "root.zw05_training_cluster.hadoop-llm.pool4"
USERGROUP = "hadoop-mtai"

# ── Engines ───────────────────────────────────────────────────────────────────
# An engine is a property of the whole submission (not per-bench): it picks the
# driver, docker image, hope template, and resource defaults. Behavior contracts
# (prompt/scorer/stop, in benchmark_meta.json) are engine-independent.
#   vllm       in-process vLLM (run_eval.sh): one job evaluates one bench with
#              vLLM in-process, no server/client, no BENCH_CONCURRENCY.
#   fluentllm  SGLang server + HTTP client (run_eval_fluentllm.sh): starts a
#              FluentLLM server then drives it with BENCH_CONCURRENCY=500.
VALID_ENGINES = ("vllm", "fluentllm")


def _detect_engine_for_model(model_path: str) -> Optional[str]:
    """Auto-detect engine based on model type.

    LongCAT/Flash-3B models require fluentllm engine.
    Returns None if cannot detect (caller should require explicit --engine).
    """
    path_lower = model_path.lower()
    # Check path keywords
    if any(kw in path_lower for kw in ["flash3b", "flash-3b", "longcat", "long-cat"]):
        return "fluentllm"

    # Check config.json if exists
    config_path = Path(model_path) / "config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
            model_type = config.get("model_type", "").lower()
            archs = [a.lower() for a in config.get("architectures", [])]
            if any(x in model_type for x in ["longcat", "flash3b"]):
                return "fluentllm"
            if any("longcat" in a or "flash3b" in a for a in archs):
                return "fluentllm"
        except Exception:
            pass

    return None

DOCKER_IMAGE = {
    "vllm": (
        "registry-offlinebiz.sankuai.com/custom_prod/"
        "com.sankuai.data.hadoop.gpu/hmart-fsp-ml/"
        "serving_ubuntu22_cu12.9_python3.12_torch2.11_vllm_2e2a9606:1.0.2"
    ),
    "fluentllm": (
        "registry-offlinebiz.sankuai.com/custom_prod/"
        "com.sankuai.data.hadoop.gpu/efficient-llm/"
        "serving_fluentllm_master_20260408_bdb108a3:1.0.3"
    ),
}

DRIVER_SH = {
    "vllm": str(REPO / "src/eval/infra/run_eval.sh"),
    "fluentllm": str(REPO / "src/eval/infra/run_eval_fluentllm.sh"),
}

# Resource defaults per (engine, runner). vLLM splits protocol (light, 2-GPU) vs
# external (4-GPU); FluentLLM always runs a 4-GPU server so both runners share.
RUNNER_DEFAULTS = {
    "vllm": {
        "protocol": {"tp": 2, "mem": 65536, "vcore": 48, "gpus": 2},
        "external": {"tp": 4, "mem": 400000, "vcore": 64, "gpus": 4},
    },
    "fluentllm": {
        "protocol": {"tp": 4, "mem": 128000, "vcore": 36, "gpus": 4},
        "external": {"tp": 4, "mem": 128000, "vcore": 36, "gpus": 4},
    },
}

# Two hope templates differ only in the [docker]/[failover]/[config]/[others]
# sections (image, failover policy, SSHD port, SHM size, FluentLLM-specific env).
# The [base]/[resource]/[user_args]/[am]/[tensorboard]/[data] prefix is shared.
_HOPE_PREFIX = """\
[base]
type = ml-easy-job
afo.app.name = {app_name}

[resource]
usergroup = {usergroup}
queue = {queue}

[roles]
workers = 1
worker.memory = {mem}
worker.vcore = {vcore}
worker.gcores80g = {gpus}
worker.script = bash {script}

[user_args]
{user_args}

[am]
afo.app.am.resource.mb = 4096

[tensorboard]
with.tensor.board = false

[docker]
afo.docker.image.name = {docker_image}

[data]
afo.data.prefetch = false
"""

_HOPE_SUFFIX = {
    "vllm": """\

[failover]
afo.app.support.engine.failover = false

[conda]

[config]

[others]
afo.app.env.YARN_CONTAINER_RUNTIME_DOCKER_SHM_SIZE_BYTES = 10737418240
afo.role.worker.env.INIT_SCRIPT_SSHD_ENABLED = true
afo.role.worker.env.INIT_SCRIPT_SSHD_PASSWORD = abc123
afo.role.worker.env.INIT_SCRIPT_SSHD_PROT = 22
afo.xm.notice.receivers.account =
with_requirements = false
""",
    "fluentllm": """\

[failover]
afo.app.support.engine.failover = true
afo.role.worker.not.nccl_not_ready = true

[conda]

[config]
afo.role.worker.env.INIT_SCRIPT_SSHD_ENABLED = true
afo.role.worker.env.INIT_SCRIPT_SSHD_PASSWORD = abc123
afo.role.worker.env.INIT_SCRIPT_SSHD_PROT = 8022

[others]
afo.app.env.YARN_CONTAINER_RUNTIME_DOCKER_SHM_SIZE_BYTES = 343597383680
afo.app.env.YARN_CONTAINER_RUNTIME_DOCKER_ULIMITS = memlock=unlimited
afo.xm.notice.receivers.account =
with_requirements = false
afo.afo-base.image.version = llm_sup
afo.app.env.need_check_gid = false
""",
}


def render_hope(engine: str, **fields: Any) -> str:
    return _HOPE_PREFIX.format(**fields) + _HOPE_SUFFIX[engine]


# ── Resource resolution (runner default < suite < flag) ───────────────────────

def resolve_resources(
    engine: str,
    runner: str,
    suite_resources: Dict[str, Any],
    flags: Dict[str, Optional[int]],
) -> Dict[str, int]:
    res = dict(RUNNER_DEFAULTS[engine][runner])
    res.update({k: v for k, v in (suite_resources.get(runner) or {}).items() if v is not None})
    res.update({k: v for k, v in flags.items() if v is not None})
    return res


# ── env construction per bench ────────────────────────────────────────────────

def build_env(
    engine: str,
    bench: BenchmarkTaskSpec,
    model_path: str,
    model_tag: str,
    output_dir: str,
    n_runs: int,
    tp: int,
    batch_cap: Optional[int],
) -> Dict[str, str]:
    """Build the per-bench env injected into the hope worker.

    The shared keys (MODEL_PATH/MODEL_TAG/BENCH_ID/OUTPUT_BASE) are common to
    both drivers. The rest is engine-specific so the env matches exactly what
    each driver reads:
      vllm       RUNNER + N_RUNS + TP_SIZE (+ BATCH_CAP protocol / DATASET+METRIC external)
      fluentllm  N_RUNS protocol / DATASET+METRIC+N_CODE_RUNS external. No RUNNER
                 (driver dispatches by BENCH_ID), no TP_SIZE/BATCH_CAP (server
                 TP is fixed at 4 inside the driver).
    """
    env: Dict[str, str] = {
        "MODEL_PATH": model_path,
        "MODEL_TAG": model_tag,
        "BENCH_ID": bench.id,
        "OUTPUT_BASE": output_dir,
    }
    if engine == "vllm":
        env["RUNNER"] = "external" if bench.is_external else "protocol"
        env["N_RUNS"] = str(n_runs)
        env["TP_SIZE"] = str(tp)
        if bench.is_external:
            env["DATASET"] = bench.dataset   # validated non-empty by suites.py
            env["METRIC"] = bench.metric
        elif batch_cap is not None:
            env["BATCH_CAP"] = str(batch_cap)
    elif engine == "fluentllm":
        if bench.is_external:
            env["DATASET"] = bench.dataset
            env["METRIC"] = bench.metric
            env["N_CODE_RUNS"] = str(n_runs)
        else:
            env["N_RUNS"] = str(n_runs)
    return env


def _user_args_str(env: Dict[str, str]) -> str:
    return "\n".join(f"afo.role.worker.env.{k} = {v}" for k, v in env.items())


# ── hope submit (run_id capture, minimal-cwd staging — 踩坑 §8.1) ─────────────

def _submit(hope_path: Path, dry_run: bool, hope_cmd: str) -> Optional[str]:
    cmd = [hope_cmd, "run", hope_path.name, "--force"]
    if dry_run:
        print(f"  [dry-run] would submit (cwd={hope_path.parent}): {' '.join(cmd)}")
        return None
    result = subprocess.run(
        cmd, cwd=str(hope_path.parent),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=180,
    )
    out = result.stdout
    run_id = None
    for line in out.splitlines():
        if "run_id:" in line:
            run_id = line.split("run_id:")[-1].strip()
            break
    if run_id is None:
        print(f"  [submit] FAILED:\n{out[-500:]}")
    return run_id


# ── summary path / collect ─────────────────────────────────────────────────────

def summary_path(output_dir: str, model_tag: str, bench_id: str) -> Path:
    return Path(output_dir) / model_tag / bench_id / "summary.json"


def cmd_submit(args: argparse.Namespace) -> int:
    # Hope jobs run on compute nodes whose CWD is NOT the repo root, so every
    # path baked into a .hope must be absolute. Normalize here so a relative
    # --model-path / --output-dir (convenient on CodeLab) can't silently produce
    # hopes that fail to resolve the model / write outputs on the node.
    args.model_path = os.path.abspath(args.model_path)
    args.output_dir = os.path.abspath(args.output_dir)
    if not os.path.exists(args.model_path):
        print(f"ERROR: --model-path does not exist: {args.model_path}", file=sys.stderr)
        return 2

    suite = load_suite_by_name("benchmark", args.suite)
    benches: List[BenchmarkTaskSpec] = list(suite.spec.benchmarks)
    if args.bench:
        wanted = set(args.bench)
        benches = [b for b in benches if b.id in wanted]
        missing = wanted - {b.id for b in benches}
        if missing:
            print(f"ERROR: --bench ids not in suite {args.suite!r}: {sorted(missing)}", file=sys.stderr)
            return 2
    batch_cap = args.batch_cap if args.batch_cap is not None else suite.spec.batch_cap

    hopes_dir = Path(args.hopes_dir) if args.hopes_dir else Path(args.output_dir) / "generated_hopes"
    hopes_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect engine if not specified
    engine = args.engine
    if engine is None:
        detected = _detect_engine_for_model(args.model_path)
        if detected:
            engine = detected
            print(f"[submit] auto-detected engine='{engine}' for model: {args.model_path}")
        else:
            print(f"ERROR: --engine not specified and could not auto-detect for model: {args.model_path}",
                  file=sys.stderr)
            print(f"       Please specify --engine ({'/'.join(VALID_ENGINES)})", file=sys.stderr)
            return 2
    script = DRIVER_SH[engine]
    docker_image = args.docker_image or DOCKER_IMAGE[engine]
    # suite.spec.resources are tuned for the vLLM engine (the suite's reference
    # execution). They do not transfer to FluentLLM (different server topology),
    # so for fluentllm we ignore them and use engine defaults + CLI flags only.
    suite_resources = suite.spec.resources if engine == "vllm" else {}

    print(f"[submit] suite={args.suite}  engine={engine}  model_tag={args.model_tag}  "
          f"benches={[b.id for b in benches]}")
    manifest = []
    for b in benches:
        runner = "external" if b.is_external else "protocol"
        res = resolve_resources(
            engine, runner, suite_resources,
            {"tp": args.tp, "mem": args.mem, "vcore": args.vcore, "gpus": args.gpus},
        )
        env = build_env(engine, b, args.model_path, args.model_tag, args.output_dir,
                        args.n_runs, res["tp"], batch_cap)
        # Human-readable Hope job name so the N jobs of a tag are distinguishable
        # on the platform (otherwise only opaque run_ids show). One per (tag,bench).
        app_name = f"eval_{args.model_tag}_{b.id}"
        content = render_hope(
            engine,
            usergroup=USERGROUP, queue=QUEUE, script=script,
            mem=res["mem"], vcore=res["vcore"], gpus=res["gpus"],
            user_args=_user_args_str(env),
            docker_image=docker_image,
            app_name=app_name,
        )
        hope_path = hopes_dir / f"{b.id}.hope"
        hope_path.write_text(content)
        run_id = _submit(hope_path, args.dry_run, args.hope_cmd)
        sp = summary_path(args.output_dir, args.model_tag, b.id)
        manifest.append((b.id, runner, res, run_id, sp))

    print(f"\n[submit] {'(dry-run) ' if args.dry_run else ''}manifest (engine={engine}):")
    print(f"  {'bench':18s} {'runner':9s} {'tp/gpu/mem':16s} {'run_id':12s} expected summary.json")
    for bid, runner, res, run_id, sp in manifest:
        rs = f"{res['tp']}/{res['gpus']}/{res['mem']}"
        print(f"  {bid:18s} {runner:9s} {rs:16s} {str(run_id or '-'):12s} {sp}")
    print(f"\n[submit] {len(manifest)} job(s) {'generated (not submitted)' if args.dry_run else 'submitted'}.")
    print("[submit] Poll completion with:  submit_eval collect "
          f"--suite {args.suite} --output-dir {args.output_dir} --model-tag {args.model_tag}")
    return 0


def cmd_collect(args: argparse.Namespace) -> int:
    suite = load_suite_by_name("benchmark", args.suite)
    benches = list(suite.spec.benchmarks)
    if args.bench:
        wanted = set(args.bench)
        benches = [b for b in benches if b.id in wanted]

    done, missing = [], []
    for b in benches:
        sp = summary_path(args.output_dir, args.model_tag, b.id)
        if sp.is_file():
            try:
                d = json.loads(sp.read_text())
                done.append((b.id, d.get("accuracy_mean"), d.get("n_runs")))
            except Exception as exc:
                missing.append((b.id, f"unreadable: {exc!r}"))
        else:
            missing.append((b.id, "no summary.json"))

    print(f"[collect] suite={args.suite}  model_tag={args.model_tag}  "
          f"{len(done)}/{len(benches)} complete")
    if done:
        print(f"\n  {'bench':18s} accuracy_mean (avg@N)")
        accs = []
        for bid, acc, n in sorted(done):
            if acc is not None:
                accs.append(acc)
                print(f"  {bid:18s} {acc*100:.1f}   (avg@{n})")
            else:
                print(f"  {bid:18s} (no accuracy_mean)")
        if missing == [] and accs:
            print(f"\n  {'AVG':18s} {sum(accs)/len(accs)*100:.1f}")
    if missing:
        print(f"\n[collect] missing {len(missing)}:")
        for bid, why in missing:
            print(f"  {bid:18s} {why}")
        return 1
    print("\n[collect] all benches complete.")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="subcmd", required=True)

    sp = sub.add_parser("submit", help="Generate + submit one hope per bench.")
    sp.add_argument("--suite", required=True)
    sp.add_argument("--model-path", required=True)
    sp.add_argument("--model-tag", required=True)
    sp.add_argument("--output-dir", required=True)
    sp.add_argument("--bench", nargs="+", default=None, help="Only these bench ids.")
    sp.add_argument("--n-runs", type=int, default=3)
    sp.add_argument("--tp", type=int, default=None, help="Override TP_SIZE.")
    sp.add_argument("--mem", type=int, default=None, help="Override worker memory (MB).")
    sp.add_argument("--vcore", type=int, default=None, help="Override worker vcore.")
    sp.add_argument("--gpus", type=int, default=None, help="Override GPU count.")
    sp.add_argument("--batch-cap", type=int, default=None, help="Override protocol batch_cap (vllm only).")
    sp.add_argument("--engine", choices=VALID_ENGINES, default=None,
                    help="Inference engine: vllm (in-process) or fluentllm "
                         "(SGLang server + client). Picks driver/image/resources. "
                         "Auto-detected for known models (LongCAT/Flash-3B → fluentllm); "
                         "if not detected, must be specified explicitly.")
    sp.add_argument("--hopes-dir", default=None)
    sp.add_argument("--hope-cmd", default=os.environ.get("HOPE_CMD", "hope"))
    sp.add_argument("--docker-image", default=None,
                    help="Override docker image (default: project serving image). "
                         "Use zhangsiyuan22's vllm_latest image for LongCat MoE.")
    sp.add_argument("--dry-run", action="store_true")
    sp.set_defaults(func=cmd_submit)

    cp = sub.add_parser("collect", help="Scan summary.json; print accuracy_mean table.")
    cp.add_argument("--suite", required=True)
    cp.add_argument("--model-tag", required=True)
    cp.add_argument("--output-dir", required=True)
    cp.add_argument("--bench", nargs="+", default=None)
    cp.set_defaults(func=cmd_collect)

    args = ap.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
