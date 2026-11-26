"""
WHAT: Hand examples for the Level 18 serving comparison.

WHY: The decision between two models rests on these three functions, so each is checked on numbers
small enough to work out by hand.
"""

from datetime import date

import pytest

from evaluation.level17_scoring import ScoredDay
from evaluation.level18_serving_validation import median_defined_recall, serving_decision, world_recall

DAYS = (date(2026, 9, 1), date(2026, 9, 2))


def _day(account_id: str, day: date, score: float, role: str) -> ScoredDay:
    """Builds one account-day with no rule fired, so ranking follows the model score."""
    return ScoredDay(
        world_id="world_a",
        bank_id="bank_a",
        account_id=account_id,
        day=day,
        rules_score=0,
        rules_fired=(),
        anomaly_score=score,
        role=role,
        depth=1 if role == "mule" else None,
    )


def test_one_slot_a_day_catches_two_of_three_mules() -> None:
    """Three accounts give one slot a day at 0.5%; two days escalate the two highest mules."""
    scored = [
        _day(account_id, day, score, "mule")
        for day in DAYS
        for account_id, score in (("acc_a", 0.9), ("acc_b", 0.8), ("acc_c", 0.7))
    ]

    assert world_recall(scored, 0.5, "once_per_world") == pytest.approx(2 / 3)
    assert world_recall(scored, 0.85, "once_per_world") == pytest.approx(1 / 3)


def test_capacity_is_half_a_percent_of_the_bank() -> None:
    """201 accounts give ceil(1.005) = 2 slots at 0.5%, so both mules are caught on one day."""
    scored = [_day("acc_m1", DAYS[0], 0.9, "mule"), _day("acc_m2", DAYS[0], 0.8, "mule")]
    scored += [_day(f"acc_g{index:03d}", DAYS[0], 0.1, "genuine") for index in range(199)]

    assert len(scored) == 201
    assert world_recall(scored, 0.5, "once_per_world") == 1.0


def test_world_without_mules_is_undefined_and_left_out_of_the_median() -> None:
    """A mule-free world has no recall, and the median uses only the defined worlds."""
    assert world_recall([_day("acc_g", DAYS[0], 0.9, "genuine")], 0.5, "once_per_world") is None
    assert median_defined_recall({"w1": 0.25, "w2": None, "w3": 0.75, "w4": 1.0}) == 0.75
    assert median_defined_recall({"w1": 0.25, "w2": 0.75}) == 0.5
    with pytest.raises(ValueError, match="No world has a defined recall"):
        median_defined_recall({"w1": None})


def test_run_a_is_served_only_when_it_beats_run_b() -> None:
    """An exact tie goes to Run B."""
    assert serving_decision(0.75, 0.5) == "run_a"
    assert serving_decision(0.5, 0.75) == "run_b"
    assert serving_decision(0.5, 0.5) == "run_b"
