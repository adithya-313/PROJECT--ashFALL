"""
WHAT: Unit tests for the pure rules-baseline metrics in evaluation/rules_metrics.py.

WHY: Each metric protects a specific honesty property of the baseline report (recall at
capacity, yield inflation, harm to genuine people, hidden depth recall, detection speed,
analyst load, cross-world spread). The expected values below are hand-computed from the
scenario in each test, never produced by the function under test.
"""

from datetime import date, datetime

import pytest

from evaluation.rules_metrics import (
    analyst_load,
    genuine_harm,
    hidden_mule_recall_by_depth,
    percentile_linear,
    recall_at_capacity,
    summarise_across_worlds,
    time_to_detection,
    yield_rate,
)

FIRST_DAY = date(2025, 1, 1)
SECOND_DAY = date(2025, 1, 2)
NOT_RUN = {"status": "not run", "reason": "needs liens/warnings from later levels"}


def test_percentile_linear_single_value_is_that_value() -> None:
    """A one-value list returns that value for any fraction inside the range."""
    assert percentile_linear([5.0], 0.9) == 5.0


def test_percentile_linear_interpolates_between_two_values() -> None:
    """Two values at fraction 0.5 interpolate to the midpoint: 1.0 + 0.5 * 1.0 = 1.5."""
    assert percentile_linear([1.0, 2.0], 0.5) == 1.5


def test_percentile_linear_at_fraction_one_returns_last_value() -> None:
    """Fraction 1.0 selects the final rank, so the last value is returned."""
    assert percentile_linear([1.0, 4.0, 9.0], 1.0) == 9.0


def test_percentile_linear_rejects_empty_list() -> None:
    """A percentile over no values is undefined and raises ValueError."""
    with pytest.raises(ValueError):
        percentile_linear([], 0.5)


def test_percentile_linear_rejects_fraction_above_one() -> None:
    """A fraction outside [0.0, 1.0] raises ValueError."""
    with pytest.raises(ValueError):
        percentile_linear([1.0, 2.0], 1.5)


def test_recall_at_capacity_counts_a_mule_once_across_days() -> None:
    """Four mules, two of them escalated across two days, give recall 2/4 = 0.5."""
    daily_escalations = {
        FIRST_DAY: ["m1", "g1"],
        SECOND_DAY: ["m2", "m1"],
    }
    assert recall_at_capacity(daily_escalations, {"m1", "m2", "m3", "m4"}) == 0.5


def test_recall_at_capacity_with_no_escalations_is_zero() -> None:
    """A day with an empty escalation list recalls nothing: 0/1 = 0.0."""
    assert recall_at_capacity({FIRST_DAY: []}, {"m1"}) == 0.0


def test_recall_at_capacity_with_no_days_is_zero() -> None:
    """No days of escalations at all still scores recall 0.0 for a non-empty mule set."""
    assert recall_at_capacity({}, {"m1"}) == 0.0


def test_recall_at_capacity_rejects_empty_mule_set() -> None:
    """Recall against no mules is undefined and raises ValueError."""
    with pytest.raises(ValueError):
        recall_at_capacity({FIRST_DAY: ["g1"]}, set())


def test_yield_rate_measured_and_reweighted_hand_computed() -> None:
    """3 mule escalations and 1 genuine one give 0.75 measured, 0.3/1.3 reweighted at f=10."""
    escalations = [
        ("m1", FIRST_DAY),
        ("g1", FIRST_DAY),
        ("m2", SECOND_DAY),
        ("m1", SECOND_DAY),
    ]

    def is_mule(account_id: str, escalation_date: date) -> bool:
        return account_id in {"m1", "m2"}

    result = yield_rate(escalations, is_mule, 10.0)
    assert result["measured"] == 0.75
    assert result["reweighted"] == pytest.approx(0.3 / 1.3, abs=1e-6)


def test_yield_rate_without_escalations_has_no_values() -> None:
    """An empty escalation set has undefined yield, reported as None for both figures."""

    def is_mule(account_id: str, escalation_date: date) -> bool:
        return False

    result = yield_rate([], is_mule, 5.0)
    assert result == {"measured": None, "reweighted": None}


def test_yield_rate_rejects_zero_inflation_factor() -> None:
    """A zero inflation factor is invalid and raises ValueError."""

    def is_mule(account_id: str, escalation_date: date) -> bool:
        return False

    with pytest.raises(ValueError):
        yield_rate([], is_mule, 0.0)


def test_genuine_harm_counts_only_active_accounts_with_persona_weights() -> None:
    """Retiree restricted twice and shopkeeper once: 2 accounts, 3 days, weight 2*2+1*1=5."""
    restricted_days = {
        "g1": {FIRST_DAY, SECOND_DAY},
        "g2": {FIRST_DAY},
        "g9": {FIRST_DAY},
    }
    personas = {"g1": "retiree", "g2": "small_shopkeeper"}

    def persona_of(account_id: str) -> str:
        # g9 is not active, so looking it up here would raise and fail the test.
        return personas[account_id]

    result = genuine_harm(
        restricted_days,
        {"g1", "g2", "g3", "g4"},
        persona_of,
        {"retiree": 2, "small_shopkeeper": 1},
        1,
    )
    assert result["restricted_accounts"] == 2
    assert result["restriction_days"] == 3
    assert result["weighted_restriction_days"] == 5
    assert result["genuine_restriction_rate"] == 0.5
    assert result["unknown_persona_count"] == 0
    assert result["downstream_shops"] == NOT_RUN
    assert result["lien_release_wait_days"] == NOT_RUN
    assert result["warnings_to_genuine_payers"] == NOT_RUN


def test_genuine_harm_uses_default_weight_for_unknown_persona() -> None:
    """A restricted active account with an unlisted persona uses the default weight."""
    result = genuine_harm(
        {"g1": {FIRST_DAY}},
        {"g1"},
        lambda account_id: "mystery",
        {},
        3,
    )
    assert result["unknown_persona_count"] == 1
    assert result["weighted_restriction_days"] == 3
    assert result["genuine_restriction_rate"] == 1.0


def test_genuine_harm_rejects_no_genuinely_active_accounts() -> None:
    """A harm rate over no active people is undefined and raises ValueError."""
    with pytest.raises(ValueError):
        genuine_harm({"g1": {FIRST_DAY}}, set(), lambda account_id: "retiree", {"retiree": 2}, 1)


def test_hidden_mule_recall_by_depth_excludes_reported_and_depth_one() -> None:
    """Depth 2 has one of two hidden mules flagged; depth 3 has none; reported depth 1 is absent."""
    result = hidden_mule_recall_by_depth(
        {"m1", "m2"},
        {"m1": 1, "m2": 2, "m3": 2, "m4": 3},
        {"m1"},
    )
    assert result == {2: 0.5, 3: 0.0}
    assert 1 not in result


def test_hidden_mule_recall_ignores_flagged_non_mules() -> None:
    """A flagged genuine account that is not a mule contributes nothing to any depth."""
    assert hidden_mule_recall_by_depth({"g1"}, {"m1": 2}, set()) == {2: 0.0}


def test_time_to_detection_median_and_p90_hand_computed() -> None:
    """12h and 24h detections give median 18.0 and linear p90 22.8; one mule stays undetected."""
    first_illicit_credit = {
        "m1": datetime(2025, 1, 1, 8, 0),
        "m2": datetime(2025, 1, 1, 0, 0),
        "m3": datetime(2025, 1, 1, 0, 0),
    }
    first_flag = {
        "m1": datetime(2025, 1, 1, 20, 0),
        "m2": datetime(2025, 1, 2, 0, 0),
    }
    result = time_to_detection(first_illicit_credit, first_flag)
    assert result["median"] == pytest.approx(18.0, abs=1e-9)
    assert result["p90"] == pytest.approx(22.8, abs=1e-9)
    assert result["detected_count"] == 2
    assert result["undetected_count"] == 1


def test_time_to_detection_flag_before_credit_counts_as_zero_hours() -> None:
    """A flag two hours before the first illicit credit counts as 0 hours, not negative."""
    first_illicit_credit = {"m4": datetime(2025, 1, 1, 10, 0)}
    first_flag = {"m4": datetime(2025, 1, 1, 8, 0)}
    result = time_to_detection(first_illicit_credit, first_flag)
    assert result["median"] == 0.0
    assert result["p90"] == 0.0
    assert result["detected_count"] == 1
    assert result["undetected_count"] == 0


def test_time_to_detection_without_any_detection_has_no_percentiles() -> None:
    """Two mules and no flags leave the percentiles None and count both as undetected."""
    first_illicit_credit = {
        "m1": datetime(2025, 1, 1, 8, 0),
        "m2": datetime(2025, 1, 1, 9, 0),
    }
    result = time_to_detection(first_illicit_credit, {})
    assert result == {"median": None, "p90": None, "detected_count": 0, "undetected_count": 2}


def test_analyst_load_mean_max_and_capacity_share() -> None:
    """Two days with 2 and 1 cases at capacity 2 give mean 1.5, max 2, share 0.75."""
    daily_escalations = {
        FIRST_DAY: ["a1", "a2"],
        SECOND_DAY: ["a3"],
    }
    result = analyst_load(daily_escalations, 2)
    assert result == {"mean": 1.5, "max": 2, "mean_share_of_capacity": 0.75}


def test_analyst_load_rejects_zero_capacity() -> None:
    """A zero daily capacity cannot be shared out and raises ValueError."""
    with pytest.raises(ValueError):
        analyst_load({FIRST_DAY: ["a1"]}, 0)


def test_analyst_load_without_days_reports_zero_load() -> None:
    """No days of escalations report zero mean, zero peak, and zero capacity share."""
    result = analyst_load({}, 2)
    assert result == {"mean": 0.0, "max": 0, "mean_share_of_capacity": 0.0}


def test_summarise_across_worlds_reports_percentiles_with_enough_worlds() -> None:
    """Twenty worlds from 0.05 to 1.00 give median 0.525, p10 0.145, p90 0.905."""
    values = [0.05 * step for step in range(1, 21)]
    result = summarise_across_worlds(values, 20)
    assert result["world_count"] == 20
    assert result["median"] == pytest.approx(0.525, abs=1e-9)
    assert result["p10"] == pytest.approx(0.145, abs=1e-9)
    assert result["p90"] == pytest.approx(0.905, abs=1e-9)
    assert "spread" not in result


def test_summarise_across_worlds_with_too_few_worlds_reports_not_enough() -> None:
    """Nineteen worlds under a minimum of twenty report the spread as unavailable."""
    values = [0.05 * step for step in range(1, 20)]
    result = summarise_across_worlds(values, 20)
    assert result["world_count"] == 19
    assert result["median"] == pytest.approx(0.5, abs=1e-9)
    assert result["spread"] == "not enough worlds"
    assert "p10" not in result
    assert "p90" not in result
