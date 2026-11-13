"""
WHAT: Unit tests for evaluation/level17_evaluation.py.

WHY: The evaluation must refuse untuned thresholds, start detection clocks only at real mule
fraud credits, and report the same hand-checkable metrics for all three scorers.
"""

from datetime import date, datetime
from typing import Any

import pytest

from core.event_clock import IST
from decisions.rules_config import load_rules_config, with_thresholds
from evaluation.level17_evaluation import (
    evaluate_world,
    first_flag_times,
    first_illicit_credit,
    require_tuned,
)
from evaluation.level17_scoring import ScoredDay

DAY_ONE = date(2025, 1, 1)
DAY_TWO = date(2025, 1, 2)
ACCOUNTS = ("mule_a", "mule_b", "g1", "g2", "g3", "g4")
RESTRICTED_ON_DAY_ONE = {"mule_a", "g1"}


def _scored_day(account_id: str, day: date, restricted: bool) -> ScoredDay:
    """Builds one account-day that is either restricted (rule fired, high score) or not."""
    if account_id == "mule_a":
        role, depth = "mule", 1
    elif account_id == "mule_b":
        role, depth = "mule", 2
    else:
        role, depth = "genuine", None
    return ScoredDay(
        bank_id="bank_a",
        account_id=account_id,
        day=day,
        rules_score=1 if restricted else 0,
        rules_fired=("RB3_MANY_PAYERS_ONE_DAY",) if restricted else (),
        anomaly_score=0.9 if restricted else 0.1,
        role=role,
        depth=depth,
    )


def _two_day_world() -> list[ScoredDay]:
    """Six accounts over two days; only mule_a and g1 are restricted, only on day one."""
    scored: list[ScoredDay] = []
    for day in (DAY_ONE, DAY_TWO):
        for account_id in ACCOUNTS:
            restricted = day == DAY_ONE and account_id in RESTRICTED_ON_DAY_ONE
            scored.append(_scored_day(account_id, day, restricted))
    return scored


def test_require_tuned_refuses_untuned_thresholds() -> None:
    """The shipped untuned_start config is refused with the exact guard message."""
    with pytest.raises(ValueError, match="thresholds not tuned on validation"):
        require_tuned(load_rules_config())


def test_require_tuned_accepts_a_fully_tuned_config() -> None:
    """Replacing every threshold value marks all of them tuned and passes the guard."""
    config = load_rules_config()
    tuned_values = {key: threshold.value for key, threshold in config.thresholds.items()}

    require_tuned(with_thresholds(config, tuned_values, "tuned on validation dataset test"))


def test_first_illicit_credit_ignores_genuine_stories_and_non_mules() -> None:
    """Only fraud-origin payments to an account that was a mule at that moment start the clock."""
    early_fraud = datetime(2025, 1, 1, 10, 0, tzinfo=IST)
    late_fraud = datetime(2025, 1, 1, 12, 0, tzinfo=IST)
    genuine_before = datetime(2025, 1, 1, 9, 0, tzinfo=IST)
    collector_fraud = datetime(2025, 1, 1, 11, 0, tzinfo=IST)
    rows: list[dict[str, Any]] = [
        {
            "payment_id": "p_late",
            "receiver_account_id": "mule_a",
            "decision_time": late_fraud,
            "truth_role_receiver": "mule",
        },
        {
            "payment_id": "p_early",
            "receiver_account_id": "mule_a",
            "decision_time": early_fraud,
            "truth_role_receiver": "mule",
        },
        {
            "payment_id": "p_genuine_before",
            "receiver_account_id": "mule_a",
            "decision_time": genuine_before,
            "truth_role_receiver": "mule",
        },
        {
            "payment_id": "p_non_mule",
            "receiver_account_id": "g1",
            "decision_time": genuine_before,
            "truth_role_receiver": "genuine",
        },
        {
            "payment_id": "p_collector",
            "receiver_account_id": "mule_b",
            "decision_time": collector_fraud,
            "truth_role_receiver": "collector",
        },
    ]
    stories = {
        "p_late": {"origin": "fraud"},
        "p_early": {"origin": "fraud"},
        "p_genuine_before": {"origin": "genuine"},
        "p_non_mule": {"origin": "fraud"},
        "p_collector": {"origin": "fraud"},
    }

    first_credit = first_illicit_credit(rows, lambda payment_id: stories[payment_id])

    assert first_credit == {"mule_a": early_fraud, "mule_b": collector_fraud}


@pytest.mark.parametrize("scorer", ["rules", "anomaly", "rules_and_anomaly"])
def test_evaluate_world_reports_hand_computed_metrics_for_every_scorer(scorer: str) -> None:
    """
    Six accounts over two days, capacity 100%, only mule_a and g1 restricted on day one:
    two mules give recall 1/2; one true and one false escalation give measured yield 1/2;
    exactly one genuinely active account is restricted.
    """
    scored = _two_day_world()
    first_credit = {
        "mule_a": datetime(2025, 1, 1, 0, 0, tzinfo=IST),
        "mule_b": datetime(2025, 1, 2, 0, 0, tzinfo=IST),
    }
    snapshot_times = {
        (scored_day.account_id, scored_day.day): datetime(
            scored_day.day.year, scored_day.day.month, scored_day.day.day, 21, 0, tzinfo=IST
        )
        for scored_day in scored
    }

    result = evaluate_world(
        scored=scored,
        scorer=scorer,
        anomaly_threshold=0.5,
        capacity_pct=100.0,
        inflation_factor=2.0,
        persona_of=lambda account_id: "person",
        weights={"person": 1},
        default_weight=1,
        snapshot_time_by_account_day=snapshot_times,
        first_credit=first_credit,
        reported=set(),
    )

    assert result["recall_at_capacity"] == 0.5
    assert result["yield_rate"]["measured"] == 0.5
    assert result["yield_rate"]["reweighted"] == pytest.approx(1.0 / 3.0)
    assert result["genuine_harm"]["restricted_accounts"] == 1
    assert result["hidden_mule_recall_by_depth"] == {1: 1.0, 2: 0.0}


def test_first_flag_times_uses_the_snapshot_time_of_the_escalated_day() -> None:
    """
    A credit at 10:00 and an escalation whose snapshot is 21:00 give 11.0 hours; a flag anchored
    at midnight would give 0 hours and fail the final assertion.
    """
    credit = datetime(2025, 1, 1, 10, 0, tzinfo=IST)
    snapshot = datetime(2025, 1, 1, 21, 0, tzinfo=IST)

    first_flag = first_flag_times(
        {DAY_ONE: ["mule_a"]},
        _two_day_world(),
        {("mule_a", DAY_ONE): snapshot},
    )

    assert first_flag == {"mule_a": snapshot}
    hours = (first_flag["mule_a"] - credit).total_seconds() / 3600.0
    assert hours == 11.0
