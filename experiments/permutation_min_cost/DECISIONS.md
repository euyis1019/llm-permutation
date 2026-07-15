# Implementation Decisions

This log records implementation-level choices without changing any pre-registered family, metric, seed, model, benchmark, threshold, or decision rule.

## 2026-07-11

- Stage 1 uses prompt `id=24` from the frozen 32-prompt file. It is the same fixed Chinese passage used by the upstream single-MLP probe, while avoiding an unregistered aggregation rule across prompts.
- Real inputs are captured as the input to each selected layer's `down_proj`, which is exactly `h = silu(gate(x)) * up(x)`. The Base model is loaded on CPU for capture so GPU usage remains below the Stage 1 resource envelope; only BF16 `down_proj` GEMMs run on GPU 0.
- Synthetic `randn seed=7` inputs have shape `[intermediate_size, 256]`, matching the upstream synthetic probe's 4x64 token count, at scales x1 and x10.
- F2/F6 block starts and all random selections use a local Torch generator. F3 uses rejection-sampled random derangements. F5 builds a maximum matching on residue-class paths for edges `(i, i+D)`, then samples the required number of disjoint edges.
- The inversion metric samples one million uniformly distributed unordered distinct index pairs. Its PRNG seed is derived from the permutation SHA-256 so the estimate is deterministic.
- The plan's Stage 1 size expression is arithmetically inconsistent with its fixed families/seeds: those rules yield 145 seeded instances plus two seedless F8 instances, hence `3 layers x 3 inputs x 147 = 1323` rows. Execution follows the fixed rules and does not manufacture additional seeds.

## 2026-07-12 — Amendment v1.1 / Stage 1b

- The second real prompt follows the amendment's implementation-level freedom and is the minimum frozen prompt id other than 24: `id=0`. The rendered token counts are 124 for id 24 and 25 for id 0; the full-shape arm caps at 124 tokens, and the decode arm uses the final captured token (`T=1`).
- All v1.1 labels are computed from each realized permutation with the frozen pointwise block-membership rules, rather than inferred only from the family name. Static validation confirms 21 zero-predicted, 5 sub-predicted, and 26 ceil-predicted instances.
- `F10 K=100%` has 4,863 valid odd-aligned candidate pairs for even `M=9728`, while the literal `ceil(K*M/2)` is 4,864 and therefore impossible. The amendment explicitly freezes this arm as deterministic; it is implemented as all 4,863 valid candidates, leaving endpoints 0 and 9727 fixed (9,726 moved coordinates). No extra pair, seed, or instance is introduced.
- Stage 1b captures the two prompts on CPU and places only the three BF16 down-projection weights and activations on GPU 0, keeping the formal measurement below the inherited 2 GiB preflight threshold. The runtime operation is exactly `linear(x[:, p], W[:, p])` for both backends.
- For the vLLM measured-tier boundary, "corresponding backend median ceiling" is operationalized without post-hoc family selection as the median `rel_l2` of all pre-registered predicted-ceil measurements in that backend; the vLLM sub/ceil threshold is one third of that value. Torch retains the explicit fixed `3e-4` threshold. This affects only S1b-2/3/4/5 reporting, not the bitwise hard criterion S1b-1.
- Formal Stage 1b completed once with 1,248/1,248 unique measurements in 33.1 seconds. S1b-1 failed (431/504 free-predicted records bitwise equal; 73 failures), so the amendment's hard-stop rule was triggered immediately. No Stage 2b or Stage 3b measurement was started, and no formal seed/configuration was rerun.
