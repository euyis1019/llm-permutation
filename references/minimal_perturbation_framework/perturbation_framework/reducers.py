"""Reducers: the place where scores become deployment plans."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Mapping, Protocol, Sequence

from .core import (
    CommitteePlan,
    Direction,
    DistillPlan,
    Reduction,
    RejectPlan,
    SearchState,
    Trial,
    WeightDeltaPlan,
)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: Sequence[float], eps: float = 1e-12) -> float:
    if not values:
        return 0.0
    center = _mean(values)
    return math.sqrt(_mean([(value - center) ** 2 for value in values]) + eps)


def _softmax(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    shift = max(values)
    exps = [math.exp(value - shift) for value in values]
    total = sum(exps)
    return [value / total for value in exps]


@dataclass(frozen=True)
class CentralTwoPoint:
    """Canonical two-point ascent on a higher-is-better utility.

    A plus candidate stores the *physical* offset ``sigma * epsilon``.
    Therefore multiplying it by ``(u+ - u-) / (2 sigma^2)`` yields the usual
    ``((u+ - u-) / (2 sigma)) * epsilon`` estimate.  Repositories that absorb
    sigma into learning rate should implement a different reducer explicitly.
    """

    step_size: float

    def reduce(self, state: SearchState, trials: Sequence[Trial]) -> Reduction:
        pairs: dict[str, dict[str, Trial]] = {}
        for trial in trials:
            if trial.candidate.pair_id is None:
                continue
            pairs.setdefault(trial.candidate.pair_id, {})[trial.candidate.role] = trial

        updates: list[Direction] = []
        differences: list[float] = []
        for pair_id, pair in pairs.items():
            if "plus" not in pair or "minus" not in pair:
                raise ValueError(f"Incomplete antithetic pair: {pair_id}")
            plus, minus = pair["plus"], pair["minus"]
            sigma = float(plus.candidate.metadata["radius"])
            difference = plus.utility - minus.utility
            differences.append(difference)
            updates.append(plus.candidate.direction.scaled(difference / (2.0 * sigma * sigma)))

        if not updates:
            return Reduction(RejectPlan("no complete antithetic pairs"), state.search_state)
        delta = Direction.combine(updates).scaled(self.step_size / len(updates))
        return Reduction(
            WeightDeltaPlan(
                delta,
                reason="central two-point utility estimate",
                diagnostics={"pair_differences": differences},
            ),
            state.search_state,
        )


@dataclass(frozen=True)
class PopulationES:
    """Fitness-weighted population update.

    ``normalization='code'`` mirrors several surveyed implementations: reward
    shaping and learning rate absorb sigma conventions.  ``'score_function'``
    applies the canonical 1/sigma^2 factor to each physical offset.  Making
    this choice visible is safer than a framework silently correcting papers.
    """

    step_size: float
    shaping: Literal["zscore", "centered_rank"] = "zscore"
    normalization: Literal["code", "score_function"] = "code"

    def _weights(self, trials: Sequence[Trial]) -> list[float]:
        utilities = [trial.utility for trial in trials]
        if self.shaping == "zscore":
            center, scale = _mean(utilities), _std(utilities)
            return [(value - center) / scale for value in utilities]
        order = sorted(range(len(trials)), key=lambda index: trials[index].utility)
        ranks = [0] * len(trials)
        for rank, index in enumerate(order):
            ranks[index] = rank
        denominator = max(len(trials) - 1, 1)
        return [rank / denominator - 0.5 for rank in ranks]

    def reduce(self, state: SearchState, trials: Sequence[Trial]) -> Reduction:
        members = [trial for trial in trials if trial.candidate.role != "baseline"]
        if not members:
            return Reduction(RejectPlan("empty population"), state.search_state)
        weights = self._weights(members)
        directions: list[Direction] = []
        for trial, weight in zip(members, weights):
            factor = self.step_size * weight / len(members)
            if self.normalization == "score_function":
                radius = abs(float(trial.candidate.metadata["radius"]))
                factor /= radius * radius
            directions.append(trial.candidate.direction.scaled(factor))
        return Reduction(
            WeightDeltaPlan(
                Direction.combine(directions),
                reason=f"population ES ({self.shaping}, {self.normalization})",
                diagnostics={"fitness_weights": weights},
            ),
            state.search_state,
        )


@dataclass(frozen=True)
class OneSidedNormalized:
    """A compact FZOO-shaped reducer, preserving its code-level scale convention."""

    step_size: float

    def reduce(self, state: SearchState, trials: Sequence[Trial]) -> Reduction:
        baseline = next((trial for trial in trials if trial.candidate.role == "baseline"), None)
        members = [trial for trial in trials if trial.candidate.role == "member"]
        if baseline is None or not members:
            return Reduction(RejectPlan("one-sided probes require a baseline"), state.search_state)
        scale = _std([trial.utility for trial in members])
        updates = [
            trial.candidate.direction.scaled(
                self.step_size * (trial.utility - baseline.utility) / (len(members) * scale)
            )
            for trial in members
        ]
        return Reduction(
            WeightDeltaPlan(
                Direction.combine(updates),
                reason="batched one-sided normalized probes",
                diagnostics={"baseline_utility": baseline.utility, "population_std": scale},
            ),
            state.search_state,
        )


@dataclass(frozen=True)
class TopKCommittee:
    top_k: int
    aggregation: str = "majority_vote"

    def reduce(self, state: SearchState, trials: Sequence[Trial]) -> Reduction:
        winners = tuple(sorted(trials, key=lambda trial: trial.utility, reverse=True)[: self.top_k])
        return Reduction(
            CommitteePlan(
                winners,
                aggregation=self.aggregation,
                diagnostics={"selected_utilities": [trial.utility for trial in winners]},
            ),
            state.search_state,
        )


@dataclass(frozen=True)
class TopKDistill:
    """The outer commit used by iterative RandOpt; distillation itself may use gradients."""

    top_k: int
    recipe: str = "sft_or_logit_kd"

    def reduce(self, state: SearchState, trials: Sequence[Trial]) -> Reduction:
        teachers = tuple(sorted(trials, key=lambda trial: trial.utility, reverse=True)[: self.top_k])
        return Reduction(DistillPlan(teachers, recipe=self.recipe), state.search_state)


class DirectionGeometry(Protocol):
    def cosine(self, first: Direction, second: Direction) -> float: ...

    def residual_norm_sq(self, direction: Direction, axis: Direction) -> float: ...


class SymbolicSeedGeometry:
    """Treat distinct NoiseRefs as orthogonal coordinates.

    This is sufficient for a mental model.  A real CoRP adapter should replace
    it with coordinate sketches or materialized dot products.
    """

    @staticmethod
    def _coords(direction: Direction) -> dict[object, float]:
        return {term.noise: term.amplitude for term in direction.canonical().terms}

    def _inner(self, first: Direction, second: Direction) -> float:
        a, b = self._coords(first), self._coords(second)
        if len(a) > len(b):
            a, b = b, a
        return sum(value * b.get(key, 0.0) for key, value in a.items())

    def cosine(self, first: Direction, second: Direction) -> float:
        aa, bb = self._inner(first, first), self._inner(second, second)
        if aa <= 1e-15 or bb <= 1e-15:
            return 0.0
        return self._inner(first, second) / math.sqrt(aa * bb)

    def residual_norm_sq(self, direction: Direction, axis: Direction) -> float:
        axis_norm = self._inner(axis, axis)
        if axis_norm <= 1e-15:
            return self._inner(direction, direction)
        projection = self._inner(direction, axis) / axis_norm
        residual = Direction.combine([direction, axis.scaled(-projection)])
        return self._inner(residual, residual)


@dataclass(frozen=True)
class CompatibilityCollapse:
    """A readable, reduced CoRP-style two-pass consolidation."""

    elite_quantile: float = 0.8
    reward_temperature: float = 5.0
    step_scale: float = 1.0
    geometry: DirectionGeometry = SymbolicSeedGeometry()

    def reduce(self, state: SearchState, trials: Sequence[Trial]) -> Reduction:
        if not trials:
            return Reduction(RejectPlan("empty population"), state.search_state)
        ordered = sorted(trials, key=lambda trial: trial.utility)
        start = min(int(self.elite_quantile * len(ordered)), len(ordered) - 1)
        elite = ordered[start:]

        pass1 = _softmax([self.reward_temperature * trial.utility for trial in elite])
        provisional = Direction.combine(
            [trial.candidate.direction.scaled(weight) for trial, weight in zip(elite, pass1)]
        )
        alignments = [self.geometry.cosine(trial.candidate.direction, provisional) for trial in elite]
        dispersions = [
            self.geometry.residual_norm_sq(trial.candidate.direction, provisional) for trial in elite
        ]
        alignment_mean, alignment_std = _mean(alignments), _std(alignments)
        dispersion_mean, dispersion_std = _mean(dispersions), _std(dispersions)
        logits = [
            self.reward_temperature * trial.utility
            + (alignment - alignment_mean) / alignment_std
            - (dispersion - dispersion_mean) / dispersion_std
            for trial, alignment, dispersion in zip(elite, alignments, dispersions)
        ]
        pass2 = _softmax(logits)
        delta = Direction.combine(
            [trial.candidate.direction.scaled(weight) for trial, weight in zip(elite, pass2)]
        ).scaled(self.step_scale)
        next_search_state = {
            **state.search_state,
            "last_elite_fraction": len(elite) / len(trials),
            "last_merge_terms": len(delta.terms),
        }
        return Reduction(
            WeightDeltaPlan(
                delta,
                reason="two-pass reward/compatibility population collapse",
                diagnostics={
                    "elite_ids": [trial.candidate.uid for trial in elite],
                    "pass1_weights": pass1,
                    "pass2_weights": pass2,
                    "alignments": alignments,
                    "dispersions": dispersions,
                },
            ),
            next_search_state,
        )

