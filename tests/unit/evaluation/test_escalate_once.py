"""
WHAT: Unit tests for the A-043 escalation repeat policy: the Level 17 every_day behaviour stays
exact, once_per_world frees the slot an earlier day's account held, banks stay independent, and an
unknown policy is refused.

WHY: A-043 replaces re-queuing a known account with review that lasts for the rest of the world, and
the switch must be provably inert until a Level 18 caller asks for it.
"""

from datetime import date, datetime, timezone

import pytest

from evaluation.level17_evaluation import evaluate_world
from evaluation.level17_scoring import ScoredDay, daily_escalations
from evaluation.rules_metrics import recall_at_capacity

DAYS = [date(2026, 9, 1), date(2026, 9, 2), date(2026, 9, 3)]


def _mule(account_id: str, day: date, bank_id: str = "bank_a") -> ScoredDay:
    """Builds one mule account-day that fires a rule."""
    return ScoredDay(
        bank_id=bank_id,
        account_id=account_id,
        day=day,
        rules_score=1,
        rules_fired=("RB3_MANY_PAYERS_ONE_DAY",),
        anomaly_score=None,
        role="mule",
        depth=1,
    )


def _two_mule_days(bank_id: str = "bank_a") -> list[ScoredDay]:
    """Builds three days of two mules who each fire a rule, so ranking ties on account id."""
    return [_mule(account_id, day, bank_id) for day in DAYS for account_id in ("acc_a", "acc_b")]


def test_every_day_repeats_the_top_account_and_once_per_world_moves_the_slot() -> None:
    """Capacity one: every_day escalates acc_a three times; once_per_world escalates acc_a, acc_b, nobody."""
    scored = _two_mule_days()

    every_day = daily_escalations(scored, "rules", None, 10.0)
    once_per_world = daily_escalations(scored, "rules", None, 10.0, repeat_policy="once_per_world")

    assert every_day == {DAYS[0]: ["acc_a"], DAYS[1]: ["acc_a"], DAYS[2]: ["acc_a"]}
    assert once_per_world == {DAYS[0]: ["acc_a"], DAYS[1]: ["acc_b"], DAYS[2]: []}
    assert recall_at_capacity(every_day, {"acc_a", "acc_b"}) == 0.5
    assert recall_at_capacity(once_per_world, {"acc_a", "acc_b"}) == 1.0


def test_once_per_world_keeps_banks_independent() -> None:
    """An account escalated at bank one is still a candidate at bank two on the same and later days."""
    scored = _two_mule_days("bank_a") + _two_mule_days("bank_b")

    escalations = daily_escalations(scored, "rules", None, 10.0, repeat_policy="once_per_world")

    assert escalations == {
        DAYS[0]: ["acc_a", "acc_a"],
        DAYS[1]: ["acc_b", "acc_b"],
        DAYS[2]: [],
    }


def test_unknown_repeat_policy_is_refused() -> None:
    """An unrecognised policy is an error, never a silent default."""
    with pytest.raises(ValueError, match="Unknown repeat policy"):
        daily_escalations(_two_mule_days(), "rules", None, 10.0, repeat_policy="never_again")


def test_evaluate_world_forwards_the_repeat_policy() -> None:
    """The C1 world gives recall 0.5 under every_day and 1.0 under once_per_world through evaluate_world."""
    scored = _two_mule_days() + [
        ScoredDay(
            bank_id="bank_a",
            account_id="acc_g",
            day=day,
            rules_score=0,
            rules_fired=(),
            anomaly_score=None,
            role="genuine",
            depth=None,
        )
        for day in DAYS
    ]
    snapshot_times = {
        (scored_day.account_id, scored_day.day): datetime(2026, 9, scored_day.day.day, 12, 0, tzinfo=timezone.utc)
        for scored_day in scored
    }
    first_credit = {
        "acc_a": datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc),
        "acc_b": datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc),
    }

    every_day = evaluate_world(
        scored,
        "rules",
        None,
        10.0,
        1.0,
        lambda _account: "genuine",
        {},
        1,
        snapshot_times,
        first_credit,
        set(),
    )
    once_per_world = evaluate_world(
        scored,
        "rules",
        None,
        10.0,
        1.0,
        lambda _account: "genuine",
        {},
        1,
        snapshot_times,
        first_credit,
        set(),
        repeat_policy="once_per_world",
    )

    assert every_day["recall_at_capacity"] == 0.5
    assert once_per_world["recall_at_capacity"] == 1.0
