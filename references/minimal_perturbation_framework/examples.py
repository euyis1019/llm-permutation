"""Configurations that expose the family resemblance between the methods.

These functions intentionally stop before supplying a torch/JAX/vLLM backend.
Reading their return values is the point: policy + reducer determines the
algorithm, while evaluator/committer determines how it runs.
"""

from perturbation_framework import (
    AntitheticPairs,
    CentralTwoPoint,
    CompatibilityCollapse,
    NoiseFamily,
    ParameterScope,
    Population,
    PopulationES,
    TargetSpace,
    TopKCommittee,
    TopKDistill,
    low_rank_family,
    quantized_code_family,
)


def mezo_like():
    """One antithetic pair, then a central-difference weight update."""
    return (
        AntitheticPairs(pairs=1, radius=1e-3),
        CentralTwoPoint(step_size=1e-5),
    )


def es_at_scale_like():
    """One-sided full-weight population, z-score reward, replayed update."""
    return (
        Population(size=30, radii=(1e-3,)),
        PopulationES(step_size=1e-3, shaping="zscore", normalization="code"),
    )


def eggroll_like():
    """The same population loop, but directions are functional low-rank operators."""
    return (
        Population(size=30, radii=(1e-3,), family=low_rank_family(rank=1)),
        PopulationES(step_size=1e-3, shaping="zscore", normalization="code"),
    )


def randopt_like():
    """No center update: preserve top candidates as an output committee."""
    family = NoiseFamily(
        family="isotropic_gaussian",
        scope=ParameterScope(
            name="language_model_only",
            exclude_prefixes=("visual.", "model.visual."),
        ),
        # Mirrors the currently published worker and makes its quirk explicit.
        rng_scheme="cuda_per_tensor_reset",
        version="randopt-worker-v1",
    )
    return (
        Population(size=5_000, radii=(1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2), family=family),
        TopKCommittee(top_k=50),
    )


def iterative_randopt_like():
    """The search is gradient-free; this deployment plan explicitly is not."""
    return (
        Population(size=30, radii=(1e-3,)),
        TopKDistill(top_k=8, recipe="SFT/KD then use checkpoint as next anchor"),
    )


def corp_like():
    """Collapse rewarded candidates into one symbolic weight delta."""
    return (
        Population(size=500, radii=(1e-3,)),
        CompatibilityCollapse(elite_quantile=0.8, reward_temperature=5.0),
    )


def qes_like():
    """Same ES control plane; a bounded integer backend owns residual/rounding state."""
    return (
        Population(size=30, radii=(1e-2,), family=quantized_code_family(bits=4, signed=False)),
        PopulationES(step_size=5e-4, shaping="centered_rank", normalization="score_function"),
    )


def qzo_like():
    """Two-point ZO in quantization-scale coordinates, not integer-code space."""
    scales = NoiseFamily(
        family="isotropic_gaussian",
        target=TargetSpace.QUANT_SCALE,
        scope=ParameterScope(name="quantization_scales"),
    )
    return (
        AntitheticPairs(pairs=1, radius=1e-3, family=scales),
        CentralTwoPoint(step_size=1e-5),
    )

