"""A tiny orchestration layer and backend adapter skeletons."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, ContextManager, Mapping, Protocol, Sequence

from .core import (
    AcceptanceGate,
    Candidate,
    CandidateEvaluator,
    CommitResult,
    CommitteePlan,
    Committer,
    DeploymentPlan,
    Direction,
    DistillPlan,
    PostStepHook,
    RejectPlan,
    SearchPolicy,
    SearchState,
    StepTrace,
    Trial,
    WeightDeltaPlan,
    Consolidator,
)


@dataclass(frozen=True)
class ObjectiveResult:
    utility: float
    per_example_utilities: tuple[float, ...] = ()
    metrics: Mapping[str, float] | None = None
    output_ref: str | None = None


class CandidateBackend(Protocol):
    def candidate(self, anchor_id: str, direction: Direction) -> ContextManager[Any]:
        """Return an exact transactional view of ``anchor + direction``.

        The context must restore from an anchor snapshot on exit.  Merely
        subtracting a bf16 perturbation is not an exact transaction.  A
        functional JAX backend may return a lazy operator view; an integer-code
        backend may unpack, mutate, and repack while retaining boundary tokens.
        """


class Objective(Protocol):
    def score(
        self,
        model_view: Any,
        *,
        split: str,
        candidate: Candidate,
    ) -> ObjectiveResult:
        """Run only forward/generation/verifier work and return higher-is-better utility."""


@dataclass
class SequentialEvaluator(CandidateEvaluator):
    """Reference evaluator; Ray/JAX implementations can replace only this class."""

    backend: CandidateBackend
    objective: Objective

    def evaluate(
        self,
        state: SearchState,
        candidates: Sequence[Candidate],
        *,
        split: str,
    ) -> Sequence[Trial]:
        trials: list[Trial] = []
        for candidate in candidates:
            if candidate.anchor_id != state.anchor_id:
                raise ValueError("candidate was proposed around a stale anchor")
            with self.backend.candidate(state.anchor_id, candidate.direction) as model_view:
                result = self.objective.score(model_view, split=split, candidate=candidate)
            trials.append(
                Trial(
                    candidate=candidate,
                    utility=float(result.utility),
                    per_example_utilities=result.per_example_utilities,
                    metrics=result.metrics or {},
                    output_ref=result.output_ref,
                )
            )
        return trials


class DeploymentBackend(CandidateBackend, Protocol):
    def commit_delta(self, anchor_id: str, delta: Direction) -> str:
        """Create a new anchor and return its id."""

    def register_committee(self, anchor_id: str, plan: CommitteePlan) -> Any:
        """Persist replay specs and the output aggregation rule."""

    def distill(self, anchor_id: str, plan: DistillPlan) -> str:
        """Run an explicitly separate SFT/KD stage and return its new anchor id."""


@dataclass
class BackendCommitter(Committer):
    backend: DeploymentBackend

    def commit(self, state: SearchState, plan: DeploymentPlan) -> CommitResult:
        if isinstance(plan, WeightDeltaPlan):
            anchor_id = self.backend.commit_delta(state.anchor_id, plan.delta)
            return CommitResult(anchor_id, diagnostics={"kind": "weight_delta"})
        if isinstance(plan, CommitteePlan):
            artifact = self.backend.register_committee(state.anchor_id, plan)
            return CommitResult(state.anchor_id, artifact, {"kind": "committee"})
        if isinstance(plan, DistillPlan):
            anchor_id = self.backend.distill(state.anchor_id, plan)
            return CommitResult(anchor_id, diagnostics={"kind": "distilled_anchor"})
        if isinstance(plan, RejectPlan):
            return CommitResult(state.anchor_id, diagnostics={"kind": "rejected", "reason": plan.reason})
        raise TypeError(f"unknown plan type: {type(plan)!r}")


class AcceptAll(AcceptanceGate):
    def review(
        self,
        state: SearchState,
        plan: DeploymentPlan,
        evaluator: CandidateEvaluator,
    ) -> DeploymentPlan:
        return plan


class NoPostStep:
    def after_step(
        self,
        state: SearchState,
        trials: Sequence[Trial],
        plan: DeploymentPlan,
        result: CommitResult,
    ) -> CommitResult:
        return result


@dataclass(frozen=True)
class HeldOutImprovementGate(AcceptanceGate):
    """Validate a weight merge against its unperturbed anchor before committing."""

    split: str = "gate"
    minimum_gain: float = 0.0

    def review(
        self,
        state: SearchState,
        plan: DeploymentPlan,
        evaluator: CandidateEvaluator,
    ) -> DeploymentPlan:
        if not isinstance(plan, WeightDeltaPlan):
            return plan
        baseline = Candidate(
            uid=f"step-{state.step}-gate-baseline",
            anchor_id=state.anchor_id,
            direction=Direction(),
            role="baseline",
        )
        proposal = Candidate(
            uid=f"step-{state.step}-gate-proposal",
            anchor_id=state.anchor_id,
            direction=plan.delta,
            role="merged",
        )
        baseline_trial, proposal_trial = evaluator.evaluate(
            state, (baseline, proposal), split=self.split
        )
        gain = proposal_trial.utility - baseline_trial.utility
        if gain < self.minimum_gain:
            return RejectPlan(
                "held-out gate rejected proposed delta",
                diagnostics={
                    "baseline_utility": baseline_trial.utility,
                    "proposal_utility": proposal_trial.utility,
                    "gain": gain,
                },
            )
        return replace(
            plan,
            diagnostics={**plan.diagnostics, "held_out_gain": gain},
        )


@dataclass
class StepEngine:
    """The complete mental model: ask -> evaluate -> reduce -> gate -> commit."""

    policy: SearchPolicy
    evaluator: CandidateEvaluator
    consolidator: Consolidator
    committer: Committer
    gate: AcceptanceGate = AcceptAll()
    post_step: PostStepHook = NoPostStep()
    selection_split: str = "selection"

    def step(self, state: SearchState) -> StepTrace:
        candidates = tuple(self.policy.ask(state))
        trials = tuple(self.evaluator.evaluate(state, candidates, split=self.selection_split))
        reduction = self.consolidator.reduce(state, trials)
        committed_plan = self.gate.review(state, reduction.plan, self.evaluator)
        result = self.committer.commit(state, committed_plan)
        result = self.post_step.after_step(state, trials, committed_plan, result)
        next_state = SearchState(
            anchor_id=result.anchor_id,
            step=state.step + 1,
            root_seed=state.root_seed,
            # This advances even if the model update was rejected.
            search_state=reduction.next_search_state,
        )
        return StepTrace(
            previous_state=state,
            candidates=candidates,
            trials=trials,
            proposed_plan=reduction.plan,
            committed_plan=committed_plan,
            result=result,
            next_state=next_state,
        )
