from perturbation_framework import (
    AntitheticPairs,
    CentralTwoPoint,
    CommitteePlan,
    Population,
    SearchState,
    TopKCommittee,
    WeightDeltaPlan,
)
from perturbation_framework.core import Trial


def test_antithetic_pair_reuses_one_noise_ref():
    plus, minus = AntitheticPairs(pairs=1, radius=0.1).ask(
        SearchState("base", root_seed=7)
    )
    assert plus.pair_id == minus.pair_id
    assert plus.direction.terms[0].noise == minus.direction.terms[0].noise
    assert plus.direction.terms[0].amplitude == -minus.direction.terms[0].amplitude


def test_two_point_produces_a_weight_delta():
    state = SearchState("base", root_seed=7)
    plus, minus = AntitheticPairs(pairs=1, radius=0.1).ask(state)
    trials = (Trial(plus, utility=2.0), Trial(minus, utility=1.0))
    plan = CentralTwoPoint(step_size=0.01).reduce(state, trials).plan
    assert isinstance(plan, WeightDeltaPlan)
    assert not plan.delta.is_zero


def test_randopt_keeps_candidates_instead_of_moving_center():
    state = SearchState("base", root_seed=7)
    candidates = Population(size=3, radii=(0.1,)).ask(state)
    trials = tuple(Trial(candidate, utility=float(index)) for index, candidate in enumerate(candidates))
    plan = TopKCommittee(top_k=2).reduce(state, trials).plan
    assert isinstance(plan, CommitteePlan)
    assert [member.utility for member in plan.members] == [2.0, 1.0]

