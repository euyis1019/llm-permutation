"""Two-GPU scheduler with rolling checkpoint generation.

Builds a job list of (family, checkpoint) units.  Each unit loads once in a
worker and runs the six benchmarks.  Baseline-original jobs point directly at
the read-only source model; baseline-copy and permutation jobs generate an HF
checkpoint on disk just-in-time, run, then have their weights removed once every
benchmark raw file is complete (manifests + results are always kept).

Two worker threads bind to CUDA_VISIBLE_DEVICES 0 and 1; each pulls the next job
from a shared queue.  Checkpoint generation (CPU/IO) overlaps with the other
GPU's generation.  Fully resumable: a job whose raw files already exist is
skipped without reloading; an interrupted job's checkpoint is regenerated
deterministically on the next run.

Usage:
    python scheduler.py --stage stage1
    python scheduler.py --stage stage2
    python scheduler.py --stage ablation
    python scheduler.py --stage all
"""

from __future__ import annotations

import argparse
import json
import queue
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import common

EXP = common.EXP_ROOT
CKPT_ROOT = EXP / "checkpoints"
RESULTS_RAW = EXP / "results" / "raw"
LOG_JOBS = EXP / "logs" / "jobs"
SCRIPTS = EXP / "scripts"

BENCHES = ["mmlu", "gsm8k", "ceval", "cmmlu", "humaneval_plus", "mbpp_plus"]

_print_lock = threading.Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(f"[sched {time.strftime('%H:%M:%S')}] {msg}", flush=True)


@dataclass
class Job:
    family: str
    tag: str                 # full model_tag e.g. qwen3_4b__perm_all36_s42
    kind: str                # "original" | "copy" | "perm"
    source: str              # source model dir
    scope: str = "none"      # for perm/copy checkpoint gen
    perm_kind: str = "random"
    seed: int = 0
    benchmarks: List[str] = field(default_factory=lambda: list(BENCHES))

    @property
    def needs_ckpt(self) -> bool:
        return self.kind in ("copy", "perm")

    @property
    def ckpt_dir(self) -> Path:
        return CKPT_ROOT / self.tag

    @property
    def model_path(self) -> str:
        return str(self.ckpt_dir) if self.needs_ckpt else self.source


def raw_complete(tag: str, bench: str) -> bool:
    p = RESULTS_RAW / tag / f"{bench}.raw.json"
    if not p.is_file():
        return False
    try:
        return json.loads(p.read_text()).get("complete") is True
    except Exception:
        return False


def job_complete(job: Job) -> bool:
    return all(raw_complete(job.tag, b) for b in job.benchmarks)


def ensure_checkpoint(job: Job) -> None:
    man = job.ckpt_dir / "perm_manifest.json"
    weights_ok = (job.ckpt_dir / "model.safetensors").is_file() or \
                 any(job.ckpt_dir.glob("model-*.safetensors"))
    if man.is_file() and weights_ok:
        return
    if job.ckpt_dir.exists():
        shutil.rmtree(job.ckpt_dir)
    log(f"generating checkpoint {job.tag} (scope={job.scope} kind={job.perm_kind} seed={job.seed})")
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS / "make_checkpoint.py"),
         "--source", job.source, "--out-dir", str(job.ckpt_dir),
         "--scope", job.scope, "--kind", job.perm_kind,
         "--base-seed", str(job.seed), "--tag", job.tag],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"make_checkpoint failed for {job.tag}:\n{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}")
    log(f"checkpoint {job.tag} ready in {time.time()-t0:.0f}s")


def cleanup_checkpoint(job: Job) -> None:
    """Delete only the weight shards; keep perm_manifest.json + config/tokenizer."""
    if not job.needs_ckpt:
        return
    for f in list(job.ckpt_dir.glob("model-*.safetensors")) + \
             list(job.ckpt_dir.glob("model.safetensors")):
        try:
            f.unlink()
        except FileNotFoundError:
            pass
    # keep index json so provenance stays, drop nothing else
    log(f"cleaned weights for {job.tag}")


MEM_UTIL = None  # set in main() from config


def run_worker(job: Job, gpu: int) -> None:
    LOG_JOBS.mkdir(parents=True, exist_ok=True)
    logf = LOG_JOBS / f"{job.tag}.gpu{gpu}.log"
    cmd = [sys.executable, str(SCRIPTS / "run_worker.py"),
           "--model-path", job.model_path, "--model-tag", job.tag,
           "--benchmarks", ",".join(job.benchmarks),
           "--out-dir", str(RESULTS_RAW), "--gpu", str(gpu)]
    if MEM_UTIL is not None:
        cmd += ["--gpu-mem-util", str(MEM_UTIL)]
    with open(logf, "a") as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"worker failed for {job.tag} (see {logf})")


def worker_loop(gpu: int, jobs: List[Job], errors: list) -> None:
    """Process this GPU's job list sequentially (family pinned to the device)."""
    for job in jobs:
        try:
            if job_complete(job):
                log(f"SKIP {job.tag} (already complete)")
            else:
                if job.needs_ckpt:
                    ensure_checkpoint(job)
                log(f"GPU{gpu} running {job.tag}")
                t0 = time.time()
                run_worker(job, gpu)
                log(f"GPU{gpu} DONE {job.tag} in {time.time()-t0:.0f}s")
            if job_complete(job):
                cleanup_checkpoint(job)
        except Exception as exc:
            log(f"ERROR {job.tag}: {exc!r}")
            errors.append((job.tag, repr(exc)))


# ── job list construction ─────────────────────────────────────────────────────

def families(cfg):
    return {
        "qwen3_4b": cfg["models"]["qwen3_4b"]["original_path"],
        "qwen3_4b_base": cfg["models"]["qwen3_4b_base"]["original_path"],
    }


def stage1_jobs(cfg) -> List[Job]:
    jobs = []
    for fam, src in families(cfg).items():
        jobs.append(Job(fam, f"{fam}__baseline_original_run1", "original", src))
        jobs.append(Job(fam, f"{fam}__baseline_original_run2", "original", src))
        jobs.append(Job(fam, f"{fam}__baseline_copy", "copy", src, scope="none"))
        for s in cfg["stage1_seeds"]:
            jobs.append(Job(fam, f"{fam}__perm_all36_s{s}", "perm", src,
                            scope="all36", perm_kind="random", seed=s))
    return jobs


def stage2_jobs(cfg) -> List[Job]:
    jobs = []
    for fam, src in families(cfg).items():
        for s in cfg["stage2_seeds"]:
            jobs.append(Job(fam, f"{fam}__perm_all36_s{s}", "perm", src,
                            scope="all36", perm_kind="random", seed=s))
    return jobs


def noise_jobs(cfg) -> List[Job]:
    """Baseline-rerun repeats: identical weights, fresh process, to measure the
    same-function inference noise floor the permutation deltas are judged against.
    run1/run2/baseline_copy already exist; add rep02..rep09 for a null of ~10
    same-weights points per family (comparable to the 20 permutation seeds)."""
    jobs = []
    for fam, src in families(cfg).items():
        for i in range(2, 10):
            jobs.append(Job(fam, f"{fam}__baseline_rep{i:02d}", "original", src))
    return jobs


def ablation_jobs(cfg) -> List[Job]:
    """Directly answer 'local vs global vs large-magnitude'.

    Scope axis   (how many layers permuted): single L0, single L17, single L35,
                 prefix:6, prefix:18, all36 — all with full random perms.
    Magnitude axis (per-neuron displacement at fixed all-36 scope):
                 adjacent_swap (minimal), reverse (structured large), random
                 (maximal, = stage sweep).
    Run on qwen3_4b only (the instruct model) to bound cost; seeds fixed.
    """
    fam = "qwen3_4b"
    src = families(cfg)[fam]
    jobs = []
    # scope axis, random perms, seed 7
    for scope in ["single:0", "single:17", "single:35", "prefix:6", "prefix:18"]:
        stag = scope.replace(":", "")
        jobs.append(Job(fam, f"{fam}__abl_scope_{stag}_random_s7", "perm", src,
                        scope=scope, perm_kind="random", seed=7))
    # magnitude axis at all-36 scope
    jobs.append(Job(fam, f"{fam}__abl_mag_adjswap_all36", "perm", src,
                    scope="all36", perm_kind="adjacent_swap", seed=0))
    jobs.append(Job(fam, f"{fam}__abl_mag_reverse_all36", "perm", src,
                    scope="all36", perm_kind="reverse", seed=0))
    # all36 random @ seed7 as the shared anchor point for both axes
    jobs.append(Job(fam, f"{fam}__abl_scope_all36_random_s7", "perm", src,
                    scope="all36", perm_kind="random", seed=7))
    return jobs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["stage1", "stage2", "noise", "ablation", "all"], required=True)
    ap.add_argument("--gpus", default="0,1")
    args = ap.parse_args()
    cfg = common.load_config()
    global MEM_UTIL
    MEM_UTIL = cfg["vllm"].get("gpu_memory_utilization_shared")

    jobs: List[Job] = []
    if args.stage in ("stage1", "all"):
        jobs += stage1_jobs(cfg)
    if args.stage in ("stage2", "all"):
        jobs += stage2_jobs(cfg)
    if args.stage in ("noise", "all"):
        jobs += noise_jobs(cfg)
    if args.stage in ("ablation", "all"):
        jobs += ablation_jobs(cfg)

    # de-dup by tag (stage1/stage2 share baseline seeds only if overlapping; they don't)
    seen = set()
    uniq = []
    for j in jobs:
        if j.tag not in seen:
            seen.add(j.tag)
            uniq.append(j)
    jobs = uniq

    pending = [j for j in jobs if not job_complete(j)]
    log(f"stage={args.stage}: {len(jobs)} jobs, {len(pending)} pending")

    # Ensure the evalplus expected-output cache is warm (single-process) so the
    # parallel code-eval workers never race on building it.
    cache_dir = Path("/nvme0/if/.cache/evalplus")
    if not any(cache_dir.glob("*.pkl")) if cache_dir.exists() else True:
        log("warming evalplus cache (single-process)")
        wp = subprocess.run([sys.executable, str(SCRIPTS / "warm_evalplus.py")],
                            capture_output=True, text=True)
        if wp.returncode != 0:
            raise RuntimeError(f"warm_evalplus failed:\n{wp.stdout[-1000:]}\n{wp.stderr[-1000:]}")
        log("evalplus cache warmed")

    # Pin each family to a fixed GPU: batch-invariant kernels are bitwise
    # deterministic within a device but the two 4090s are not bit-identical, so
    # a family's baseline and all its permutations must share one GPU.
    pin = cfg["gpu_pinning"]
    by_gpu: dict[int, List[Job]] = {}
    for j in pending:
        g = pin[j.family]
        by_gpu.setdefault(g, []).append(j)
    for g, js in by_gpu.items():
        log(f"GPU{g}: {len(js)} jobs")
        for j in js:
            log(f"  GPU{g} pending: {j.tag}")

    # Retry passes: under GPU sharing a job may fail transiently. Re-run any
    # still-incomplete jobs up to `max_passes` times; the worker itself already
    # retries engine init internally.
    max_passes = 6
    remaining = pending
    for p in range(max_passes):
        errors: list = []
        by_gpu = {}
        for j in remaining:
            by_gpu.setdefault(pin[j.family], []).append(j)
        threads = [threading.Thread(target=worker_loop, args=(g, js, errors), daemon=True)
                   for g, js in by_gpu.items()]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        remaining = [j for j in jobs if not job_complete(j)]
        if not remaining:
            log(f"stage={args.stage} ALL DONE, no incomplete jobs (pass {p+1})")
            return
        log(f"pass {p+1}: {len(remaining)} jobs still incomplete; retrying in 60s")
        time.sleep(60)
    log(f"COMPLETED WITH {len(remaining)} INCOMPLETE JOBS after {max_passes} passes:")
    for j in remaining:
        log(f"  incomplete: {j.tag}")
    sys.exit(1)


if __name__ == "__main__":
    main()
