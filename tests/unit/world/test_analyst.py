"""
Tests for the simulated analyst (world/analyst.py, amendment A-022).
Covers that the decision is a pure function, that accuracy falls with weak evidence and with a longer queue, that
measured accuracy matches the formula, that automation bias copies the score, that the floor applies and is
visible, that capacity follows its formula, that bad inputs are refused and that drawing parameters always takes
six draws. Every expected value is computed in the test from the A-022 formula and from parameters built here.

TERM: queue load: open cases divided by the team's daily capacity in cases.
TERM: overload fraction: 0 when load is at most 1, otherwise (load - 1) / load.
TERM: automation bias: the analyst copying the model's score instead of judging the case.
"""

import copy
import math
from typing import Any

import numpy as np
import pytest

from world.analyst import (
    AnalystParams,
    daily_capacity_cases,
    decide_case,
    draw_analyst_params,
    expected_accuracy,
    load_analyst_config,
    overload_fraction,
    validate_analyst_config,
)
from world.config_reader import read_world_ranges
from world.label_records import load_labels_config

SPLIT = "training"
SAMPLE_SIZE = 20_000


def make_params(
    base: float = 0.9, weak_drop: float = 0.2, overload_drop: float = 0.15, follows: float = 0.0,
    minutes_per_case: int = 20, analysts: int = 5, shift: int = 480, floor: float = 0.5, threshold: float = 0.5,
) -> AnalystParams:
    """Builds analyst parameters directly, so each test controls exactly the numbers it relies on."""
    return AnalystParams(base, weak_drop, overload_drop, follows, minutes_per_case, analysts, shift, floor, threshold)


def formula_accuracy(params: AnalystParams, evidence: float, load: float) -> float:
    """The A-022 accuracy formula written out independently of world/analyst.py, before the floor."""
    overload = 0.0 if load <= 1.0 else (load - 1.0) / load
    return params.accuracy_base - params.drop_weak_evidence * (1.0 - evidence) - params.drop_overload * overload


@pytest.fixture(scope="module")
def analyst_config() -> dict[str, Any]:
    """Loads and validates the analyst config once for the module."""
    return load_analyst_config()


def test_decide_case_is_pure_and_uses_no_global_random_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Checks the same inputs give an equal outcome twice, even when numpy's global generators would raise."""
    def refuse(*_args: Any, **_kwargs: Any) -> None:
        """Stands in for a random source and fails the test if anything calls it."""
        raise AssertionError("decide_case must not use a random generator")

    monkeypatch.setattr(np.random, "default_rng", refuse)
    monkeypatch.setattr(np.random, "random", refuse)
    params = make_params(follows=0.3)

    first = decide_case(True, 0.4, 2.0, 0.7, params, 0.5, 0.6)
    second = decide_case(True, 0.4, 2.0, 0.7, params, 0.5, 0.6)

    assert first == second


def test_accuracy_falls_with_load_and_with_weaker_evidence() -> None:
    """Checks expected accuracy never rises as load grows or evidence weakens, and load up to 1 changes nothing."""
    params = make_params()
    loads = [0.5, 1.0, 2.0, 4.0, 8.0]
    by_load = [expected_accuracy(params, 0.7, load)[1] for load in loads]
    assert all(later <= earlier for earlier, later in zip(by_load, by_load[1:]))
    assert by_load[-1] < by_load[1]
    assert expected_accuracy(params, 0.7, 0.5) == expected_accuracy(params, 0.7, 1.0)
    assert overload_fraction(0.0) == 0.0 and overload_fraction(1.0) == 0.0

    evidence_levels = [1.0, 0.5, 0.0]
    by_evidence = [expected_accuracy(params, evidence, 1.0)[1] for evidence in evidence_levels]
    assert all(later <= earlier for earlier, later in zip(by_evidence, by_evidence[1:]))
    assert by_evidence[-1] < by_evidence[0]


@pytest.mark.parametrize("load", [1.0, 4.0])
def test_measured_accuracy_matches_the_formula(load: float) -> None:
    """Checks the share of correct verdicts over 20,000 cases is within the configured standard errors."""
    params = make_params()
    evidence = 0.5
    expected = max(params.accuracy_floor, formula_accuracy(params, evidence, load))
    draws = np.random.default_rng(31).random((SAMPLE_SIZE, 2))

    correct = 0
    for case_index in range(SAMPLE_SIZE):
        is_mule = case_index % 2 == 0
        bias_draw = float(draws[case_index, 0])
        accuracy_draw = float(draws[case_index, 1])
        outcome = decide_case(is_mule, evidence, load, None, params, bias_draw, accuracy_draw)
        if outcome.verdict_is_mule == is_mule:
            correct += 1

    standard_error = math.sqrt(expected * (1 - expected) / SAMPLE_SIZE)
    tolerance = load_labels_config()["rate_check_standard_errors"] * standard_error
    assert abs(correct / SAMPLE_SIZE - expected) <= tolerance


def test_automation_bias_copies_the_score_and_a_zero_share_ignores_it() -> None:
    """Checks follows share 1.0 always returns score >= threshold, even when wrong, and share 0.0 ignores the score."""
    always_follows = make_params(follows=1.0, threshold=0.5)
    never_follows = make_params(follows=0.0, threshold=0.5)
    rng = np.random.default_rng(41)
    saw_wrong_verdict = False
    for _case in range(500):
        is_mule = bool(rng.random() < 0.5)
        score = float(rng.random())
        bias_draw, accuracy_draw = float(rng.random()), float(rng.random())

        followed = decide_case(is_mule, 1.0, 0.5, score, always_follows, bias_draw, accuracy_draw)
        assert followed.followed_score is True
        assert followed.verdict_is_mule == (score >= always_follows.score_threshold)
        saw_wrong_verdict = saw_wrong_verdict or followed.verdict_is_mule != is_mule

        ignored = decide_case(is_mule, 1.0, 0.5, score, never_follows, bias_draw, accuracy_draw)
        without_score = decide_case(is_mule, 1.0, 0.5, None, never_follows, bias_draw, accuracy_draw)
        assert ignored == without_score
        assert ignored.followed_score is False
    assert saw_wrong_verdict


def test_the_floor_applies_and_the_unfloored_value_is_visible() -> None:
    """Checks accuracy is raised to the floor while unfloored_accuracy shows the lower formula value."""
    params = make_params(base=0.6, weak_drop=0.3, overload_drop=0.3, floor=0.5)
    evidence, load = 0.0, 8.0

    outcome = decide_case(True, evidence, load, None, params, 0.0, 0.0)

    expected_unfloored = 0.6 - 0.3 * (1.0 - 0.0) - 0.3 * ((8.0 - 1.0) / 8.0)
    assert expected_unfloored < params.accuracy_floor
    assert outcome.unfloored_accuracy == pytest.approx(expected_unfloored)
    assert outcome.accuracy == params.accuracy_floor


def test_capacity_follows_the_formula_and_bad_inputs_are_refused(analyst_config: dict[str, Any]) -> None:
    """Checks daily capacity equals analysts x shift / minutes, and negative load, bad evidence or draws raise."""
    params = make_params(minutes_per_case=20, analysts=7, shift=480)
    assert daily_capacity_cases(params) == 7 * 480 / 20

    with pytest.raises(ValueError, match="queue_load"):
        overload_fraction(-0.1)
    with pytest.raises(ValueError, match="evidence_strength"):
        expected_accuracy(params, 1.5, 1.0)
    with pytest.raises(ValueError, match="bias_draw"):
        decide_case(True, 0.5, 1.0, 0.5, params, 1.0, 0.5)
    with pytest.raises(ValueError, match="accuracy_draw"):
        decide_case(True, 0.5, 1.0, None, params, 0.5, 1.0)
    with pytest.raises(ValueError, match="model_score"):
        decide_case(True, 0.5, 1.0, 1.2, params, 0.5, 0.5)

    missing_key = copy.deepcopy(analyst_config)
    del missing_key["accuracy_floor"]
    with pytest.raises(ValueError, match="accuracy_floor"):
        validate_analyst_config(missing_key)
    bad_floor = copy.deepcopy(analyst_config)
    bad_floor["accuracy_floor"]["value"] = 1.5
    with pytest.raises(ValueError, match="accuracy_floor"):
        validate_analyst_config(bad_floor)


def test_drawing_parameters_takes_exactly_six_draws_whatever_the_config(analyst_config: dict[str, Any]) -> None:
    """Checks two different configurations leave the generator in the same state, six draws further on."""
    world_ranges = read_world_ranges()
    other_config = copy.deepcopy(analyst_config)
    other_config["minutes_per_case"]["training"] = [5, 500]
    other_config["analysts_on_shift"]["training"] = [1, 90]

    rng_a, rng_b, reference = np.random.default_rng(51), np.random.default_rng(51), np.random.default_rng(51)
    params_a = draw_analyst_params(world_ranges, SPLIT, rng_a, analyst_config)
    draw_analyst_params(world_ranges, SPLIT, rng_b, other_config)
    reference.random(6)

    assert rng_a.random() == rng_b.random() == reference.random()
    low, high = analyst_config["minutes_per_case"][SPLIT]
    assert low <= params_a.minutes_per_case <= high
