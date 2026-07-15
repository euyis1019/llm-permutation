"""The small intermediate representation shared by perturbation methods.

This module deliberately contains no torch, JAX, vLLM, or distributed code.
Its purpose is to name the pieces that the surveyed repositories currently
mix together inside their trainers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence, TypeAlias


class TargetSpace(str, Enum):
    """The coordinates in which a direction is interpreted."""

    FLOAT_WEIGHT = "float_weight"
    ADAPTER = "adapter"
    QUANT_SCALE = "quant_scale"
    INTEGER_CODE = "integer_code"
    FUNCTIONAL_OPERATOR = "functional_operator"


@dataclass(frozen=True)
class ParameterScope:
    """A persistent, inspectable replacement for ad-hoc parameter filters."""

    name: str = "all_trainable"
    include_prefixes: tuple[str, ...] = ()
    exclude_prefixes: tuple[str, ...] = ()
    trainable_only: bool = True


@dataclass(frozen=True)
class NoiseRef:
    """A replayable *unit* direction; it does not contain sigma.

    A seed alone is not a replay contract.  ``rng_scheme`` and ``version``
    distinguish, for example, a global RNG stream from RandOpt's current
    per-tensor-reset CUDA stream.  ``options`` records small family-specific
    facts such as a low-rank value or a quantization rounding stream.
    """

    family: str
    seed: int
    target: TargetSpace = TargetSpace.FLOAT_WEIGHT
    scope: ParameterScope = field(default_factory=ParameterScope)
    rng_scheme: str = "global_stream"
    version: str = "v1"
    options: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class DirectionTerm:
    """One replayable direction with its physical additive amplitude."""

    noise: NoiseRef
    amplitude: float


@dataclass(frozen=True)
class Direction:
    """A lazy linear combination of replayable directions.

    Keeping a direction symbolic is the seed-replay trick: an update can be
    stored as a few seeds and coefficients rather than a model-sized tensor.
    """

    terms: tuple[DirectionTerm, ...] = ()

    @classmethod
    def one(cls, noise: NoiseRef, amplitude: float = 1.0) -> "Direction":
        return cls((DirectionTerm(noise, float(amplitude)),)).canonical()

    @classmethod
    def combine(cls, directions: Sequence["Direction"]) -> "Direction":
        return cls(tuple(term for direction in directions for term in direction.terms)).canonical()

    def scaled(self, coefficient: float) -> "Direction":
        return Direction(
            tuple(DirectionTerm(term.noise, term.amplitude * coefficient) for term in self.terms)
        ).canonical()

    def canonical(self, atol: float = 1e-15) -> "Direction":
        amplitudes: dict[NoiseRef, float] = {}
        for term in self.terms:
            amplitudes[term.noise] = amplitudes.get(term.noise, 0.0) + term.amplitude
        ordered = sorted(
            ((noise, amplitude) for noise, amplitude in amplitudes.items() if abs(amplitude) > atol),
            key=lambda item: (
                item[0].target.value,
                item[0].scope.name,
                item[0].family,
                item[0].seed,
                item[0].rng_scheme,
                item[0].version,
                item[0].options,
            ),
        )
        return Direction(tuple(DirectionTerm(noise, amplitude) for noise, amplitude in ordered))

    def negated(self) -> "Direction":
        return self.scaled(-1.0)

    @property
    def is_zero(self) -> bool:
        return not self.terms


@dataclass(frozen=True)
class Candidate:
    """A model view ``anchor + direction`` that should be scored once."""

    uid: str
    anchor_id: str
    direction: Direction
    role: str = "member"  # plus, minus, member, or baseline
    pair_id: str | None = None
    group_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Trial:
    """The observation produced by evaluating one candidate.

    ``utility`` is always higher-is-better.  A loss objective should expose
    ``utility = -loss`` at the adapter boundary so reducers cannot silently
    disagree about signs.
    """

    candidate: Candidate
    utility: float
    per_example_utilities: tuple[float, ...] = ()
    metrics: Mapping[str, float] = field(default_factory=dict)
    output_ref: str | None = None


@dataclass(frozen=True)
class SearchState:
    """Model-anchor state and proposal-distribution state are intentionally separate."""

    anchor_id: str
    step: int = 0
    root_seed: int = 0
    search_state: Mapping[str, Any] = field(default_factory=dict)


# A reducer can produce three qualitatively different deployment objects.


@dataclass(frozen=True)
class WeightDeltaPlan:
    delta: Direction
    reason: str
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CommitteePlan:
    members: tuple[Trial, ...]
    aggregation: str = "majority_vote"
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DistillPlan:
    """A behavior-level commit: teachers produce a new model anchor via distillation."""

    teachers: tuple[Trial, ...]
    recipe: str = "sft_or_logit_kd"
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RejectPlan:
    reason: str
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


DeploymentPlan: TypeAlias = WeightDeltaPlan | CommitteePlan | DistillPlan | RejectPlan


@dataclass(frozen=True)
class Reduction:
    """A deployment decision plus the next proposal-distribution state.

    The search distribution may adapt even when an acceptance gate rejects a
    model update.  CoRP has exactly this separation.
    """

    plan: DeploymentPlan
    next_search_state: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CommitResult:
    anchor_id: str
    artifact: Any = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StepTrace:
    previous_state: SearchState
    candidates: tuple[Candidate, ...]
    trials: tuple[Trial, ...]
    proposed_plan: DeploymentPlan
    committed_plan: DeploymentPlan
    result: CommitResult
    next_state: SearchState


class SearchPolicy(Protocol):
    def ask(self, state: SearchState) -> Sequence[Candidate]:
        """Create symbolic candidates around ``state.anchor_id``."""


class CandidateEvaluator(Protocol):
    def evaluate(
        self,
        state: SearchState,
        candidates: Sequence[Candidate],
        *,
        split: str,
    ) -> Sequence[Trial]:
        """Score candidates without permanently changing the anchor."""


class Consolidator(Protocol):
    def reduce(self, state: SearchState, trials: Sequence[Trial]) -> Reduction:
        """Turn black-box observations into an update, committee, or teachers."""


class AcceptanceGate(Protocol):
    def review(
        self,
        state: SearchState,
        plan: DeploymentPlan,
        evaluator: CandidateEvaluator,
    ) -> DeploymentPlan:
        """Optionally validate a proposed deployment on a disjoint split."""


class Committer(Protocol):
    def commit(self, state: SearchState, plan: DeploymentPlan) -> CommitResult:
        """Materialize the chosen deployment object and return its anchor/artifact id."""


class PostStepHook(Protocol):
    def after_step(
        self,
        state: SearchState,
        trials: Sequence[Trial],
        plan: DeploymentPlan,
        result: CommitResult,
    ) -> CommitResult:
        """Optionally run a slower controller after the main commit.

        DiZO's periodic low-dimensional radius optimization belongs here; it
        is not a different Gaussian direction sampler in the outer loop.
        """
