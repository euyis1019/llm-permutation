"""Reviewer-side analyses for the noise_floor final round (zero GPU).

Part 3  null selection test   : pick best-of-20 functionally-identical models on
                                half A, measure the "gain" on half B (must vanish).
Part 4  regression-to-mean    : where does the baseline sit inside the 20-seed
                                neighborhood distribution? Does rank explain sign?
Part 5  majority vote         : vote over the 20 seeds; does it beat baseline?
Part 7  ZO probe noise floor  : |dNLL| of Gaussian sigma-probes vs the dNLL that
                                pure permutation numerical noise already causes.

Inputs: existing per-sample records in ffn_benchmark_eval/results/raw/ and the
noise_floor Part 1a / Part 6 jsonl files. Output: reviewer_analysis/part{3,4,5,7}.json
"""
from __future__ import annotations
import json, os, random, statistics
from collections import Counter, defaultdict

ROOT = "/nvme0/if/permutation/experiments"
RAW = f"{ROOT}/ffn_benchmark_eval/results/raw"
OUT = f"{ROOT}/noise_floor/reviewer_analysis"
os.makedirs(OUT, exist_ok=True)

BENCHES = ["mmlu", "gsm8k", "ceval", "cmmlu", "humaneval_plus", "mbpp_plus"]
MC_LIKE = {"mmlu", "gsm8k", "ceval", "cmmlu"}  # vote on extracted answer
SEEDS = list(range(1000, 1020))
FAMILIES = {
    "base": {"prefix": "qwen3_4b_base", "baseline": "qwen3_4b_base__baseline_original_run1"},
    "instruct": {"prefix": "qwen3_4b", "baseline": "qwen3_4b__baseline_original_run1"},
}
BASELINE_REPS = ["baseline_original_run1", "baseline_original_run2", "baseline_copy"] + [
    f"baseline_rep{i:02d}" for i in range(2, 10)
]
N_SPLITS = 200
SPLIT_SEED = 12345


def load_samples(tag: str, bench: str):
    path = f"{RAW}/{tag}/{bench}.raw.json"
    with open(path) as f:
        d = json.load(f)
    recs = {}
    for s in d["samples"]:
        recs[s["sample_id"]] = {
            "correct": bool(s["correct"]),
            "extracted": s.get("extracted"),
            "response": s.get("response"),
        }
    return recs


def load_family(fam):
    prefix = FAMILIES[fam]["prefix"]
    models = {}
    for s in SEEDS:
        tag = f"{prefix}__perm_all36_s{s}"
        models[s] = {b: load_samples(tag, b) for b in BENCHES}
    baseline = {b: load_samples(FAMILIES[fam]["baseline"], b) for b in BENCHES}
    return models, baseline


def acc(recs, ids):
    return sum(recs[i]["correct"] for i in ids) / len(ids)


# ---------------- Part 3: null selection ----------------

def part3(models, baseline):
    rng = random.Random(SPLIT_SEED)
    sample_ids = {b: sorted(baseline[b].keys()) for b in BENCHES}
    per_bench = {b: {"gain_A": [], "gain_B": [], "gain_B_vs_baseline": []} for b in BENCHES}
    suite = {"gain_A": [], "gain_B": [], "gain_B_vs_baseline": []}
    for _ in range(N_SPLITS):
        halves = {}
        for b in BENCHES:
            ids = sample_ids[b][:]
            rng.shuffle(ids)
            halves[b] = (ids[: len(ids) // 2], ids[len(ids) // 2 :])
        # per-benchmark selection
        for b in BENCHES:
            A, B = halves[b]
            accA = {s: acc(models[s][b], A) for s in SEEDS}
            accB = {s: acc(models[s][b], B) for s in SEEDS}
            best = max(SEEDS, key=lambda s: accA[s])
            meanA = statistics.mean(accA.values())
            meanB = statistics.mean(accB.values())
            per_bench[b]["gain_A"].append(accA[best] - meanA)
            per_bench[b]["gain_B"].append(accB[best] - meanB)
            per_bench[b]["gain_B_vs_baseline"].append(accB[best] - acc(baseline[b], B))
        # suite-macro selection (the realistic scenario)
        macroA = {s: statistics.mean(acc(models[s][b], halves[b][0]) for b in BENCHES) for s in SEEDS}
        macroB = {s: statistics.mean(acc(models[s][b], halves[b][1]) for b in BENCHES) for s in SEEDS}
        blA = statistics.mean(acc(baseline[b], halves[b][0]) for b in BENCHES)
        blB = statistics.mean(acc(baseline[b], halves[b][1]) for b in BENCHES)
        best = max(SEEDS, key=lambda s: macroA[s])
        suite["gain_A"].append(macroA[best] - statistics.mean(macroA.values()))
        suite["gain_B"].append(macroB[best] - statistics.mean(macroB.values()))
        suite["gain_B_vs_baseline"].append(macroB[best] - blB)
    def summ(v):
        return {"mean_pp": 100 * statistics.mean(v), "sd_pp": 100 * statistics.pstdev(v),
                "frac_positive": sum(x > 0 for x in v) / len(v)}
    return {
        "n_splits": N_SPLITS, "split_seed": SPLIT_SEED,
        "suite_macro": {k: summ(v) for k, v in suite.items()},
        "per_benchmark": {b: {k: summ(v) for k, v in d.items()} for b, d in per_bench.items()},
    }


# ---------------- Part 4: regression to mean ----------------

def part4(models, baseline, fam):
    prefix = FAMILIES[fam]["prefix"]
    out = {"per_benchmark": {}, "suite_macro": {}, "baseline_reps": {}}
    ids = {b: sorted(baseline[b].keys()) for b in BENCHES}
    seed_macro, base_macro = {}, None
    for b in BENCHES:
        seed_accs = sorted(acc(models[s][b], ids[b]) for s in SEEDS)
        ba = acc(baseline[b], ids[b])
        rank = sum(a < ba for a in seed_accs)  # #seeds strictly below baseline
        out["per_benchmark"][b] = {
            "baseline_acc": ba,
            "seed_mean": statistics.mean(seed_accs),
            "seed_sd": statistics.pstdev(seed_accs),
            "seed_min": seed_accs[0], "seed_max": seed_accs[-1],
            "baseline_rank_of_20": rank,
            "delta_mean_minus_baseline_pp": 100 * (statistics.mean(seed_accs) - ba),
        }
    for s in SEEDS:
        seed_macro[s] = statistics.mean(acc(models[s][b], ids[b]) for b in BENCHES)
    base_macro = statistics.mean(acc(baseline[b], ids[b]) for b in BENCHES)
    sm = sorted(seed_macro.values())
    out["suite_macro"] = {
        "baseline": base_macro,
        "seed_mean": statistics.mean(sm), "seed_sd": statistics.pstdev(sm),
        "seed_min": sm[0], "seed_max": sm[-1],
        "baseline_rank_of_20": sum(a < base_macro for a in sm),
        "delta_mean_minus_baseline_pp": 100 * (statistics.mean(sm) - base_macro),
    }
    # baseline replica runs: run-to-run spread of the un-permuted model
    rep_macros = []
    for rep in BASELINE_REPS:
        tag = f"{prefix}__{rep}"
        if not os.path.isdir(f"{RAW}/{tag}"):
            continue
        try:
            m = statistics.mean(acc(load_samples(tag, b), ids[b]) for b in BENCHES)
            rep_macros.append({"tag": rep, "suite_macro": m})
        except FileNotFoundError:
            continue
    vals = [r["suite_macro"] for r in rep_macros]
    out["baseline_reps"] = {
        "runs": rep_macros,
        "n": len(vals),
        "spread_pp": 100 * (max(vals) - min(vals)) if vals else None,
        "sd_pp": 100 * statistics.pstdev(vals) if len(vals) > 1 else None,
    }
    return out


# ---------------- Part 5: majority vote ----------------

def part5(models, baseline):
    out = {}
    for b in BENCHES:
        ids = sorted(baseline[b].keys())
        voted_correct = 0
        oracle_majority = 0
        for i in ids:
            if b in MC_LIKE:
                votes = Counter()
                for s in SEEDS:
                    e = models[s][b][i]["extracted"]
                    votes[e if e is not None else "__none__"] += 1
                win, _ = votes.most_common(1)[0]
                # winner correct iff some seed with that extraction was correct
                ok = any(
                    models[s][b][i]["extracted"] == win and models[s][b][i]["correct"]
                    for s in SEEDS
                ) if win != "__none__" else False
                voted_correct += ok
            else:
                votes = Counter(models[s][b][i]["response"] for s in SEEDS)
                win, _ = votes.most_common(1)[0]
                ok = any(
                    models[s][b][i]["response"] == win and models[s][b][i]["correct"]
                    for s in SEEDS
                )
                voted_correct += ok
            oracle_majority += sum(models[s][b][i]["correct"] for s in SEEDS) > 10
        n = len(ids)
        seed_accs = [acc(models[s][b], ids) for s in SEEDS]
        out[b] = {
            "vote_acc": voted_correct / n,
            "majority_correct_acc": oracle_majority / n,
            "baseline_acc": acc(baseline[b], ids),
            "seed_mean": statistics.mean(seed_accs),
            "seed_max": max(seed_accs),
            "vote_minus_baseline_pp": 100 * (voted_correct / n - acc(baseline[b], ids)),
            "vote_minus_seedmean_pp": 100 * (voted_correct / n - statistics.mean(seed_accs)),
        }
    macro = lambda k: statistics.mean(out[b][k] for b in BENCHES)
    out["suite_macro"] = {k: macro(k) for k in
                          ["vote_acc", "majority_correct_acc", "baseline_acc", "seed_mean"]}
    return out


# ---------------- Part 7: ZO probe noise floor ----------------

def part7():
    nf = f"{ROOT}/noise_floor/results"
    # permutation floor from part1a (per-prompt paired NLL)
    per_variant = defaultdict(list)
    with open(f"{nf}/part1a_logits.jsonl") as f:
        for line in f:
            d = json.loads(line)
            per_variant[d["variant"]].append(d["mean_nll"] - d["baseline_mean_nll"])
    perm_floor = {}
    for v, deltas in per_variant.items():
        perm_floor[v] = {
            "mean_abs_dnll": statistics.mean(abs(x) for x in deltas),
            "mean_dnll": statistics.mean(deltas),
            "max_abs_dnll": max(abs(x) for x in deltas),
        }
    # sigma sweep (aggregate dNLL per unit, 3 reps per sigma)
    sweep = defaultdict(list)
    with open(f"{nf}/part6_sigma_sweep.jsonl") as f:
        for line in f:
            d = json.loads(line)
            sweep[d["sigma"]].append(d["mean_nll"] - d["baseline_mean_nll"])
    floor = perm_floor["F7-all36"]["mean_abs_dnll"]
    curve = []
    for sig in sorted(sweep):
        ds = sweep[sig]
        curve.append({
            "sigma": sig,
            "dnll_reps": ds,
            "mean_abs_dnll": statistics.mean(abs(x) for x in ds),
            "mean_dnll": statistics.mean(ds),
            "ratio_to_perm_floor": statistics.mean(abs(x) for x in ds) / floor if floor else None,
        })
    return {"perm_floor_per_variant": perm_floor,
            "perm_floor_used": floor,
            "note": "dNLL = mean over 32 prompts of (case mean NLL - baseline mean NLL); "
                    "part6 units store only the 32-prompt aggregate, so reps are n=3 per sigma.",
            "sigma_curve": curve}


def main():
    p7 = part7()
    json.dump(p7, open(f"{OUT}/part7_zo_floor.json", "w"), indent=2)
    print("== part7 written ==")
    results = {}
    for fam in FAMILIES:
        print(f"loading family {fam} ...")
        models, baseline = load_family(fam)
        results[fam] = {
            "part3": part3(models, baseline),
            "part4": part4(models, baseline, fam),
            "part5": part5(models, baseline),
        }
        print(f"== {fam} done ==")
    for part in ["part3", "part4", "part5"]:
        json.dump({fam: results[fam][part] for fam in results},
                  open(f"{OUT}/{part}.json", "w"), indent=2)
    # console digest
    for fam in results:
        r = results[fam]
        s3 = r["part3"]["suite_macro"]
        print(f"\n[{fam}] part3 suite: gain_A={s3['gain_A']['mean_pp']:.3f}pp "
              f"gain_B={s3['gain_B']['mean_pp']:.3f}pp (sd {s3['gain_B']['sd_pp']:.3f}) "
              f"gain_B_vs_baseline={s3['gain_B_vs_baseline']['mean_pp']:.3f}pp")
        s4 = r["part4"]["suite_macro"]
        print(f"[{fam}] part4 suite: baseline={100*s4['baseline']:.3f} "
              f"seeds {100*s4['seed_mean']:.3f}±{100*s4['seed_sd']:.3f} "
              f"range [{100*s4['seed_min']:.3f},{100*s4['seed_max']:.3f}] "
              f"rank={s4['baseline_rank_of_20']}/20  reps n={r['part4']['baseline_reps']['n']} "
              f"spread={r['part4']['baseline_reps']['spread_pp']}pp")
        s5 = r["part5"]["suite_macro"]
        print(f"[{fam}] part5 suite: vote={100*s5['vote_acc']:.3f} "
              f"baseline={100*s5['baseline_acc']:.3f} seed_mean={100*s5['seed_mean']:.3f}")


if __name__ == "__main__":
    main()
