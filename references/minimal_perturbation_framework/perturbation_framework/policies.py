"""Small proposal policies.

The direction family is intentionally monolithic.  A real SubZero, LOZO, or
learned-scale policy owns slow state (bases, factors, or learned scales) and
may implement the same ``SearchPolicy`` without decomposing itself into a
dozen premature interfaces.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, replace
from typing import Sequence

from .core import Candidate, Direction, NoiseRef, ParameterScope, SearchPolicy, SearchState, TargetSpace


@dataclass(frozen=True)
class NoiseFamily:
    family: str = "isotropic_gaussian"
    target: TargetSpace = TargetSpace.FLOAT_WEIGHT
    scope: ParameterScope = field(default_factory=ParameterScope)
    rng_scheme: str = "global_stream"
    version: str = "v1"
    options: tuple[tuple[str, str], ...] = ()

    def ref(self, seed: int) -> NoiseRef:
        return NoiseRef(
            family=self.family,
            seed=int(seed),
            target=self.target,
            scope=self.scope,
            rng_scheme=self.rng_scheme,
            version=self.version,
            options=self.options,
        )


def _step_rng(state: SearchState, salt: int) -> random.Random:
    # This selects replay keys; it is deliberately independent from the model RNG.
    return random.Random((state.root_seed << 32) ^ (state.step << 8) ^ salt)


@dataclass(frozen=True)
class AntitheticPairs(SearchPolicy):
    """The proposal side of central two-point ZO or mirror-sampled ES."""

    pairs: int
    radius: float
    family: NoiseFamily = field(default_factory=NoiseFamily)

    def ask(self, state: SearchState) -> Sequence[Candidate]:
        rng = _step_rng(state, salt=11)
        result: list[Candidate] = []
        for index in range(self.pairs):
            seed = rng.randrange(1, 2**31)
            pair_id = f"step-{state.step}-pair-{index}"
            unit = Direction.one(self.family.ref(seed))
            for role, sign in (("plus", 1.0), ("minus", -1.0)):
                result.append(
                    Candidate(
                        uid=f"{pair_id}-{role}",
                        anchor_id=state.anchor_id,
                        direction=unit.scaled(sign * self.radius),
                        role=role,
                        pair_id=pair_id,
                        metadata={"radius": self.radius, "sign": sign},
                    )
                )
        return result


@dataclass(frozen=True)
class Population(SearchPolicy):
    """One-sided population sampling used by RandOpt and basic OpenAI-style ES."""

    size: int
    radii: tuple[float, ...]
    family: NoiseFamily = field(default_factory=NoiseFamily)

    def ask(self, state: SearchState) -> Sequence[Candidate]:
        if not self.radii:
            raise ValueError("Population requires at least one radius")
        rng = _step_rng(state, salt=23)
        seeds: set[int] = set()
        result: list[Candidate] = []
        while len(result) < self.size:
            seed = rng.randrange(1, 2**31)
            if seed in seeds:
                continue
            seeds.add(seed)
            radius = rng.choice(self.radii)
            result.append(
                Candidate(
                    uid=f"step-{state.step}-member-{len(result)}",
                    anchor_id=state.anchor_id,
                    direction=Direction.one(self.family.ref(seed), radius),
                    role="member",
                    metadata={"radius": radius},
                )
            )
        return result


@dataclass(frozen=True)
class OneSidedWithBaseline(SearchPolicy):
    """FZOO-like N one-sided probes plus one unperturbed baseline query."""

    size: int
    radius: float
    family: NoiseFamily = field(
        default_factory=lambda: NoiseFamily(family="rademacher")
    )

    def ask(self, state: SearchState) -> Sequence[Candidate]:
        population = Population(self.size, (self.radius,), self.family).ask(state)
        baseline = Candidate(
            uid=f"step-{state.step}-baseline",
            anchor_id=state.anchor_id,
            direction=Direction(),
            role="baseline",
            metadata={"radius": 0.0},
        )
        return (*population, baseline)


def low_rank_family(rank: int, *, scope: ParameterScope | None = None) -> NoiseFamily:
    """EGGROLL/LOZO-shaped lazy directions, realized by a backend operator hook."""

    return NoiseFamily(
        family="low_rank_gaussian",
        target=TargetSpace.FUNCTIONAL_OPERATOR,
        scope=scope or ParameterScope(),
        rng_scheme="per_parameter_hashed_stream",
        version="mental-model-v1",
        options=(("rank", str(rank)),),
    )


def quantized_code_family(*, bits: int, signed: bool) -> NoiseFamily:
    """A QES-shaped direction; boundary masks/residuals remain backend state."""

    return NoiseFamily(
        family="stochastically_rounded_gaussian",
        target=TargetSpace.INTEGER_CODE,
        rng_scheme="noise_and_rounding_streams",
        version="mental-model-v1",
        options=(("bits", str(bits)), ("signed", str(signed).lower())),
    )

