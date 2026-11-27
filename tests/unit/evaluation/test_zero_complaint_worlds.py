"""
WHAT: Tests that a world with zero complaints gets undefined E1-re-weighted metrics, never zero or a floor.

WHY: The re-weighted yield divides by the world's fraud inflation factor, which is its complaint rate over
the real-world ratio. A world with no complaints has no measured inflation, so dividing is impossible and a
floor would be a made-up number. Measured metrics must not move, and the per-world summaries must use only
the worlds that defined the value and say how many did not.
"""

from datetime import date, datetime, timezone
from typing import Any

import pytest

from evaluation.level17_evaluation import evaluate_world
from evaluation.level17_scoring import ScoredDay, account_key
from evaluation.rules_metrics import REWEIGHTED_NOT_DEFINED_REASON, yield_rate
from evaluation.run_level_17 import _world_inflation_factor, summarise_worlds

DAY = date(2026, 9, 1)
REASON = "not defined: world has 0 complaints, so the E1 inflation factor is undefined"


def _card(complaints: Any, factor: Any = 19.1831801876115) -> dict[str, Any]:
    """A world card with the two fields the factor reads."""
    return {"complaints": complaints, "fraud_inflation_factor": factor}


def test_the_reason_text_is_the_ruled_wording() -> None:
    assert REWEIGHTED_NOT_DEFINED_REASON == REASON


def test_a_zero_complaint_card_gives_none() -> None:
    assert _world_inflation_factor(_card(0, factor=0.0), "run_x") is None


def test_a_positive_card_returns_its_factor_exactly() -> None:
    assert _world_inflation_factor(_card(3), "run_x") == 19.1831801876115


@pytest.mark.parametrize("bad_complaints", [None, -1, "3", True])
def test_a_missing_negative_or_non_numeric_complaint_count_raises(bad_complaints: Any) -> None:
    with pytest.raises(ValueError, match="complaints count"):
        _world_inflation_factor(_card(bad_complaints), "run_x")


def test_a_card_without_the_complaints_field_raises() -> None:
    with pytest.raises(ValueError, match="complaints count"):
        _world_inflation_factor({"fraud_inflation_factor": 19.0}, "run_x")


@pytest.mark.parametrize("bad_factor", [None, 0.0, -2.0, "19"])
def test_a_world_with_complaints_still_needs_a_positive_factor(bad_factor: Any) -> None:
    with pytest.raises(ValueError, match="positive fraud_inflation_factor"):
        _world_inflation_factor(_card(3, factor=bad_factor), "run_x")


def _scored_world() -> list[ScoredDay]:
    """One world: mule m1 and genuine g1, g2; m1 and g1 fire a rule."""

    def day(account_id: str, role: str, rules_score: int) -> ScoredDay:
        return ScoredDay(
            world_id="world_a",
            bank_id="bank_a",
            account_id=account_id,
            day=DAY,
            rules_score=rules_score,
            rules_fired=("RB3_MANY_PAYERS_ONE_DAY",) if rules_score else (),
            anomaly_score=None,
            role=role,
            depth=1 if role == "mule" else None,
        )

    return [day("m1", "mule", 1), day("g1", "genuine", 1), day("g2", "genuine", 0)]


def _evaluate(inflation_factor: float | None) -> dict[str, Any]:
    """Runs evaluate_world with the rules scorer on the fixed world and the given factor."""
    scored = _scored_world()
    seen = datetime(2026, 9, 1, 21, 0, tzinfo=timezone.utc)
    return evaluate_world(
        scored=scored,
        scorer="rules",
        anomaly_threshold=None,
        capacity_pct=100.0,
        inflation_factor=inflation_factor,
        persona_of=lambda account_id: "person",
        weights={"person": 1},
        default_weight=1,
        snapshot_time_by_account_day={(account_key(day), day.day): seen for day in scored},
        first_credit={"m1": datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)},
        reported=set(),
    )


def test_evaluate_world_with_an_undefined_factor_leaves_measured_metrics_identical() -> None:
    undefined = _evaluate(_world_inflation_factor(_card(0, factor=0.0), "run_x"))
    defined = _evaluate(_world_inflation_factor(_card(3), "run_x"))

    assert undefined["yield_rate"]["reweighted"] is None
    assert undefined["yield_rate"]["reweighted_reason"] == REASON
    assert defined["yield_rate"]["reweighted"] is not None
    assert "reweighted_reason" not in defined["yield_rate"]
    # Escalations are the mule and g1: one true and one false positive, whatever the factor.
    assert undefined["yield_rate"]["measured"] == defined["yield_rate"]["measured"] == 0.5
    for metric in (
        "recall_at_capacity",
        "genuine_harm",
        "hidden_mule_recall_by_depth",
        "time_to_detection",
        "analyst_load",
    ):
        assert undefined[metric] == defined[metric]


def test_yield_rate_with_no_escalations_and_an_undefined_factor_says_why() -> None:
    result = yield_rate([], lambda account, day: False, None)

    assert result == {"measured": None, "reweighted": None, "reweighted_reason": REASON}


def _world_block(split: str, reweighted: float | None, measured: float = 0.5) -> dict[str, Any]:
    """A world block whose reweighted yield is the given number, or undefined when None."""
    yield_block: dict[str, Any] = {"measured": measured, "reweighted": reweighted}
    if reweighted is None:
        yield_block["reweighted_reason"] = REASON
    scorers = {
        scorer: {
            "recall_at_capacity": 0.5,
            "yield_rate": dict(yield_block),
            "genuine_harm": {"genuine_restriction_rate": 0.0},
            "time_to_detection": {"median": 1.0},
        }
        for scorer in ("rules", "anomaly", "rules_and_anomaly")
    }
    return {"run_id": f"run_{split}", "split": split, "scorers": scorers}


def test_summary_over_two_defined_and_one_undefined_world_uses_the_two() -> None:
    per_world = {
        "test_a": _world_block("test", 0.2),
        "test_b": _world_block("test", 0.6),
        "test_c": _world_block("test", None),
    }

    summary = summarise_worlds(per_world, min_worlds=2)["test"]["rules"]

    reweighted = summary["yield_reweighted"]
    assert reweighted["median"] == pytest.approx(0.4)
    assert reweighted["world_count"] == 2
    assert reweighted["worlds_defined"] == 2
    assert reweighted["worlds_undefined"] == 1
    # The measured yield still covers all three worlds and carries no defined/undefined counts.
    assert summary["yield_measured"]["world_count"] == 3
    assert "worlds_defined" not in summary["yield_measured"]


def test_summary_where_every_world_is_undefined_has_a_none_median_and_the_reason() -> None:
    per_world = {"test_a": _world_block("test", None), "test_b": _world_block("test", None)}

    summary = summarise_worlds(per_world, min_worlds=2)["test"]["rules"]["yield_reweighted"]

    assert summary["median"] is None
    assert summary["reason"] == REASON
    assert summary["worlds_defined"] == 0
    assert summary["worlds_undefined"] == 2
    assert summary["world_count"] == 0
