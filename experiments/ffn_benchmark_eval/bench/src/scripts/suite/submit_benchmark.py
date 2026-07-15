#!/usr/bin/env python3
"""Submit benchmark Hope jobs for a suite.

For each bench in the suite (or the specified subset), this script:
  1. Calls plan() to generate/refresh the execution plan.
  2. For each bench pair (deploy + client):
     a. Generates a deploy .hope file and submits the deploy job.
     b. Generates a client .hope file and submits the client job.

The deploy job starts vLLM (or custom HF service); the client job waits
for ENDPOINT to be ready then runs slim_runner.run_slim_bench().

Usage
-----
  # Dry-run (generate .hope files, don't submit):
  python src/scripts/suite/submit_benchmark.py \\
      --suite slim_qwen_test \\
      --model-path /path/to/Qwen3-14B-Base \\
      --model-tag baseline \\
      --output-dir /path/to/output/qwen3-14b \\
      --slim-data-dir datasets/benchmark/normalized/slim_qwen_test \\
      --gpus 4 \\
      --dry-run

  # Submit a single bench:
  python src/scripts/suite/submit_benchmark.py \\
      --suite slim_qwen_test \\
      --model-path /path/to/model \\
      --model-tag baseline \\
      --output-dir /path/to/output \\
      --slim-data-dir datasets/benchmark/normalized/slim_qwen_test \\
      --gpus 4 \\
      --bench mmlu__slim5
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from src.scripts.suite.plan_benchmark import plan as do_plan

# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_DOCKER_IMAGE = (
    "registry-offlinebiz.sankuai.com/custom_prod/"
    "com.sankuai.data.hadoop.gpu/hmart-fsp-ml/"
    "serving_ubuntu22_cu12.9_python3.12_torch2.11_vllm_2e2a9606:1.0.2"
)
DEFAULT_QUEUE = "root.zw05_training_cluster.hadoop-llm.pool"
DEFAULT_USERGROUP = "hadoop-mtai"
DEFAULT_DEPLOY_PORT = 8080

# Worker scripts (relative to repo root)
DEPLOY_SCRIPT = str(REPO / "experiments" / "05_framework_evaluate" / "run_deploy.sh")
CLIENT_SCRIPT = str(REPO / "experiments" / "05_framework_evaluate" / "run_client.sh")


# ── Hope templates ─────────────────────────────────────────────────────────────

def _hope_deploy_template(gpus: int) -> str:
    return textwrap.dedent(f"""\
        [base]
        type = ml-easy-job

        [resource]
        usergroup = {{usergroup}}
        queue = {{queue}}

        [roles]
        workers = 1
        worker.memory = 1500000
        worker.vcore = 64
        worker.gcores80g = {gpus}
        worker.script = bash {{script}}

        [user_args]
        {{user_args}}

        [am]
        afo.app.am.resource.mb = 4096

        [tensorboard]
        with.tensor.board = false

        [docker]
        afo.docker.image.name = {{docker_image}}

        [data]
        afo.data.prefetch = false

        [failover]
        afo.app.support.engine.failover = true

        [conda]

        [config]

        [others]
        afo.app.env.YARN_CONTAINER_RUNTIME_DOCKER_SHM_SIZE_BYTES = 10737418240
        afo.role.worker.env.INIT_SCRIPT_SSHD_ENABLED = true
        afo.role.worker.env.INIT_SCRIPT_SSHD_PASSWORD = abc123
        afo.role.worker.env.INIT_SCRIPT_SSHD_PROT = 22
        afo.xm.notice.receivers.account =
        with_requirements = false
    """)


def _hope_external_template(gpus: int) -> str:
    """Single-GPU job template for an external-framework bench (e.g. evalplus).

    No deploy/client split: the framework loads vLLM in-process. SSHD/shm are
    enabled like deploy (8-GPU NCCL needs shm; Ubuntu22 needs SSHD shadow fix).
    """
    return textwrap.dedent(f"""\
        [base]
        type = ml-easy-job

        [resource]
        usergroup = {{usergroup}}
        queue = {{queue}}

        [roles]
        workers = 1
        worker.memory = 900000
        worker.vcore = 64
        worker.gcores80g = {gpus}
        worker.script = bash {{script}}

        [user_args]
        {{user_args}}

        [am]
        afo.app.am.resource.mb = 4096

        [tensorboard]
        with.tensor.board = false

        [docker]
        afo.docker.image.name = {{docker_image}}

        [data]
        afo.data.prefetch = false

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
    """)


def _hope_client_template() -> str:
    return textwrap.dedent("""\
        [base]
        type = ml-easy-job

        [resource]
        usergroup = {usergroup}
        queue = {queue}

        [roles]
        workers = 1
        worker.memory = 200000
        worker.vcore = 16
        worker.gcores80g = 0
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

        [failover]
        afo.app.support.engine.failover = false

        [conda]

        [config]

        [others]
        afo.xm.notice.receivers.account =
        with_requirements = false
    """)


def _user_args_str(env: dict) -> str:
    return "\n".join(f"afo.role.worker.env.{k} = {v}" for k, v in env.items())


def _write_hope(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    print(f"  [hope] written: {path}")


def _submit(hope_path: Path, dry_run: bool, hope_cmd: str, cwd: Optional[str] = None) -> Optional[str]:
    cmd = [hope_cmd, "run", str(hope_path.resolve())]
    print(f"  [submit] {' '.join(cmd)}" + (f"  (cwd={cwd})" if cwd else ""))
    if dry_run:
        print("  [dry-run] not submitted")
        return None
    # cwd matters: hope stages the *submission cwd subtree*. For external jobs we
    # submit from the tiny worker-script dir so staging is minimal (踩坑 §8.1).
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=cwd)
    output = result.stdout.decode("utf-8", errors="replace")
    print(output)
    if result.returncode != 0 and "run_id:" not in output:
        raise RuntimeError(f"hope run failed (exit {result.returncode}):\n{output}")
    for line in output.splitlines():
        if "run_id:" in line.lower() or "jobid" in line.lower():
            return line.strip()
    return output.strip()


# ── Per-bench submission ──────────────────────────────────────────────────────

def submit_bench_pair(
    bench: str,
    client_descriptor: dict,
    model_path: str,
    gpus: int,
    mode: str,
    output_dir: str,
    slim_data_dir: str,
    suite: str,
    model_tag: str,
    created_at: str,
    hopes_dir: Path,
    docker_image: str,
    queue: str,
    usergroup: str,
    port: int,
    dry_run: bool,
    hope_cmd: str,
) -> None:
    """Submit one deploy + client job pair for a single bench."""
    from src.data.layout import ArtifactLayout
    layout = ArtifactLayout(output_dir)
    result_path = layout.benchmark_task_result_path(suite, model_tag, bench)
    raw_path = layout.benchmark_task_raw_path(suite, model_tag, bench)

    data_path = os.path.join(slim_data_dir, f"{bench}.jsonl")
    fewshot = client_descriptor.get("fewshot", 0)

    print(f"\n{'='*60}")
    print(f"  Bench:       {bench}")
    print(f"  Deploy GPUs: {gpus}")
    print(f"  Data:        {data_path}")
    print(f"  Mode:        {mode}")
    print(f"  Result:      {result_path}")
    print(f"{'='*60}")

    # ── Deploy hope ──────────────────────────────────────────────────────────
    deploy_env = {
        "MODEL_PATH": model_path,
        "BENCH": bench,
        "DEPLOY_PORT": str(port),
    }
    deploy_hope_path = hopes_dir / f"{bench}_deploy.hope"
    deploy_content = _hope_deploy_template(gpus).format(
        usergroup=usergroup,
        queue=queue,
        script=DEPLOY_SCRIPT,
        user_args=_user_args_str(deploy_env),
        docker_image=docker_image,
    )
    _write_hope(deploy_hope_path, deploy_content)
    deploy_job_id = _submit(deploy_hope_path, dry_run, hope_cmd)

    endpoint_placeholder = f"http://${{DEPLOY_HOST}}:{port}"

    # ── Client hope ──────────────────────────────────────────────────────────
    gen_kwargs = client_descriptor.get("generation_kwargs") or {}
    max_new_tokens = gen_kwargs.get("max_new_tokens", gen_kwargs.get("max_gen_toks", 512))

    client_env = {
        "BENCH_ID": bench,
        "DATA_PATH": data_path,
        "ENDPOINT": endpoint_placeholder,
        "FEWSHOT": str(fewshot),
        "MODE": mode,
        "RESULT_PATH": result_path,
        "RAW_PATH": raw_path,
        "MAX_NEW_TOKENS": str(max_new_tokens),
        "CREATED_AT": created_at,
    }
    client_hope_path = hopes_dir / f"{bench}_client.hope"
    client_content = _hope_client_template().format(
        usergroup=usergroup,
        queue=queue,
        script=CLIENT_SCRIPT,
        user_args=_user_args_str(client_env),
        docker_image=docker_image,
    )
    _write_hope(client_hope_path, client_content)

    if not dry_run and deploy_job_id:
        print(f"  [note] Deploy job submitted: {deploy_job_id}")
        print(f"  [note] Before submitting client, set DEPLOY_HOST to the deploy job's IP.")
        print(f"  [note] Then re-run: hope run {client_hope_path}")
    else:
        _submit(client_hope_path, dry_run, hope_cmd)


def submit_external_bench(
    ext: dict,
    model_path: str,
    hopes_dir: Path,
    docker_image: str,
    queue: str,
    usergroup: str,
    dry_run: bool,
    hope_cmd: str,
    repeat: int = 1,
    gpu_override: Optional[int] = None,
) -> None:
    """Submit one external-framework bench as a single GPU job (e.g. evalplus).

    ``ext`` is an ExternalJobDescriptor dict from the plan. We render a single
    job .hope (no client) and submit it FROM the worker-script's directory so
    hope stages a minimal subtree (踩坑 §8.1) — the worker uses absolute NFS
    paths so the staged copy is never read at runtime.

    When ``repeat > 1``, submits N independent jobs with output isolated under
    run_01/, run_02/, ... subdirectories for variance quantification.
    """
    bench = ext["bench"]
    gpus = gpu_override if gpu_override is not None else ext["resources"].get("gpus_per_job", 2)
    worker_script = ext["worker_script"]

    base_output_dir = str(Path(ext["output_dir"]).resolve())
    base_result_path = str(Path(ext["result_path"]).resolve())

    iterations = range(1, repeat + 1) if repeat > 1 else [None]

    for run_idx in iterations:
        if run_idx is not None:
            # repeat mode: tasks/{bench}/run_XX/external and tasks/{bench}/run_XX/result.json
            task_dir = str(Path(base_result_path).parent / f"run_{run_idx:02d}")
            output_dir = str(Path(task_dir) / "external")
            result_path = str(Path(task_dir) / "result.json")
            suffix = f"_run{run_idx:02d}"
        else:
            output_dir = base_output_dir
            result_path = base_result_path
            suffix = ""

        print(f"\n{'='*60}")
        print(f"  External bench: {bench}{suffix}")
        print(f"  Framework:      {ext['framework']}  dataset={ext['dataset']}  metric={ext['metric']}")
        print(f"  GPUs:           {gpus}")
        print(f"  Result:         {result_path}")
        print(f"{'='*60}")

        env = {
            "MODEL_PATH": model_path,
            "DATASET": ext["dataset"],
            "BENCH_ID": bench,
            "METRIC": ext["metric"],
            "OUTPUT_DIR": output_dir,
            "RESULT_PATH": result_path,
        }
        hope_path = hopes_dir / f"{bench}_external{suffix}.hope"
        content = _hope_external_template(gpus).format(
            usergroup=usergroup,
            queue=queue,
            script=worker_script,
            user_args=_user_args_str(env),
            docker_image=docker_image,
        )
        _write_hope(hope_path, content)
        _submit(hope_path, dry_run, hope_cmd, cwd=str(Path(worker_script).parent))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--suite", required=True, help="Suite name (e.g. slim_qwen_test).")
    ap.add_argument("--model-path", required=True, help="Path to the model directory.")
    ap.add_argument("--model-tag", required=True, help="Stable tag (e.g. baseline, r025).")
    ap.add_argument("--output-dir", required=True, help="Root output directory.")
    ap.add_argument(
        "--slim-data-dir",
        default=str(REPO / "datasets" / "benchmark" / "normalized" / "slim_qwen_test"),
        help="Directory containing slim bench JSONL files.",
    )
    ap.add_argument(
        "--gpus", type=int, default=4,
        help="Number of GPUs for the deploy job.",
    )
    ap.add_argument(
        "--original-path", default=None,
        help="Path to original model (required for pruned models).",
    )
    ap.add_argument(
        "--bench", nargs="+", default=None,
        help="Only submit these bench IDs (default: all).",
    )
    ap.add_argument(
        "--backend",
        default="remote_vllm_service",
        choices=["remote_vllm_service", "remote_hf_service"],
        help="Execution backend mode.",
    )
    ap.add_argument(
        "--port", type=int, default=DEFAULT_DEPLOY_PORT,
        help=f"Deploy service port (default: {DEFAULT_DEPLOY_PORT}).",
    )
    ap.add_argument("--docker-image", default=DEFAULT_DOCKER_IMAGE)
    ap.add_argument("--queue", default=DEFAULT_QUEUE)
    ap.add_argument("--usergroup", default=DEFAULT_USERGROUP)
    ap.add_argument(
        "--hopes-dir", default=None,
        help="Directory for generated .hope files (default: <output-dir>/generated_hopes/).",
    )
    ap.add_argument("--dry-run", action="store_true", help="Generate hope files but do not submit.")
    ap.add_argument(
        "--repeat", type=int, default=1,
        help="Submit N independent runs per external bench (for variance quantification).",
    )
    ap.add_argument(
        "--external-gpus", type=int, default=None,
        help="Override GPU count for external jobs (default: use plan value).",
    )
    ap.add_argument(
        "--micro", action="store_true",
        help="Micro-test mode: use slim_data/micro/ for data.",
    )
    ap.add_argument(
        "--hope-cmd",
        default=os.environ.get("HOPE_CMD", "hope"),
        help="Path or name of the hope CLI binary.",
    )
    args = ap.parse_args()

    created_at = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    slim_data_dir = args.slim_data_dir
    if args.micro:
        slim_data_dir = str(Path(slim_data_dir) / "micro")
        print(f"[submit] Micro mode: data dir = {slim_data_dir}")

    hopes_dir = Path(args.hopes_dir) if args.hopes_dir else Path(args.output_dir) / "generated_hopes"

    print(f"[submit] Planning {args.suite} for model_tag={args.model_tag}...")
    plan_dict = do_plan(
        suite_name=args.suite,
        model_path=args.model_path,
        model_tag=args.model_tag,
        output_dir=args.output_dir,
        original_path=args.original_path,
        benchmark_filter=args.bench,
        backend_mode=args.backend,
        created_at=created_at,
    )

    model_tag = plan_dict["model_tag"]
    suite_name = plan_dict["suite"]
    benches = plan_dict["jobs"]
    external = plan_dict.get("external_jobs", [])
    print(f"[submit] {len(benches)} protocol bench(es) + {len(external)} external bench(es)")

    # ── Protocol benches: deploy + client pair each ───────────────────────────
    for job_pair in benches:
        submit_bench_pair(
            bench=job_pair["bench"],
            client_descriptor=job_pair["client"],
            model_path=args.model_path,
            gpus=args.gpus,
            mode=args.backend,
            output_dir=args.output_dir,
            slim_data_dir=slim_data_dir,
            suite=suite_name,
            model_tag=model_tag,
            created_at=created_at,
            hopes_dir=hopes_dir,
            docker_image=args.docker_image,
            queue=args.queue,
            usergroup=args.usergroup,
            port=args.port,
            dry_run=args.dry_run,
            hope_cmd=args.hope_cmd,
        )

    # ── External benches: single GPU job each ─────────────────────────────────
    for ext in external:
        submit_external_bench(
            ext=ext,
            model_path=args.model_path,
            hopes_dir=hopes_dir,
            docker_image=args.docker_image,
            queue=args.queue,
            usergroup=args.usergroup,
            dry_run=args.dry_run,
            hope_cmd=args.hope_cmd,
            repeat=args.repeat,
            gpu_override=args.external_gpus,
        )

    total = len(benches) + len(external)
    if args.dry_run:
        print(f"\n[dry-run] {total} bench(es) hope files generated, nothing submitted.")
    else:
        print(f"\n[done] Submitted {len(benches)} protocol pair(s) + {len(external)} external job(s).")


if __name__ == "__main__":
    main()
