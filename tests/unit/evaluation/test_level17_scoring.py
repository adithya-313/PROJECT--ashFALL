"""
WHAT: Unit tests for evaluation/level17_scoring.py, the per-account-day scoring and ground-set
layer used by the Level 17 evaluation.

WHY: Each test pins a hand-computed outcome: which snapshot survives dedupe, that scoring never
sees hidden truth, that a victim counts as genuine harm, that capacity is a hard ceiling, and
that ranking ties are broken deterministically.
"""

from datetime import date, datetime
from typing import Any

import pytest

from core.event_clock import IST
from decisions.rules_config import load_rules_config
from decisions.rules_engine import RulesDecision
from evaluation import level17_scoring
from evaluation.level17_scoring import (
    ScoredDay,
    daily_escalations,
    feature_values_only,
    genuine_restricted_days,
    ground_sets,
    one_row_per_account_day,
    rules_restricted_days,
    score_account_days,
)
from features.account_snapshots import account_feature_names

FIRST_DAY = date(2025, 1, 1)
SECOND_DAY = date(2025, 1, 2)


def _row(**overrides: Any) -> dict[str, Any]:
    """Builds one snapshot row with sensible defaults for the tests below."""
    row: dict[str, Any] = {
        "bank_id": "bank_a",
        "account_id": "acct_1",
        "snapshot_time": datetime(2025, 1, 1, 12, 0, tzinfo=IST),
        "snapshot_kind": "daily",
        "truth_role": "genuine",
        "truth_depth": None,
        "truth_owner": "owner_1",
    }
    row.update(overrides)
    return row


# (a) One row per account and IST day: the latest snapshot wins, and a 'daily' row wins an
# exact-time tie against a 'full' row.
def test_one_row_per_account_day_keeps_latest_snapshot() -> None:
    """09:00 full and 21:00 daily on the same day collapse to the 21:00 row."""
    early = _row(
        snapshot_time=datetime(2025, 1, 1, 9, 0, tzinfo=IST),
        snapshot_kind="full",
    )
    late = _row(
        snapshot_time=datetime(2025, 1, 1, 21, 0, tzinfo=IST),
        snapshot_kind="daily",
    )

    kept = one_row_per_account_day([early, late])

    assert kept == [late]


def test_one_row_per_account_day_prefers_daily_on_exact_time_tie() -> None:
    """Two rows at 12:00 collapse to the daily one; a later full row would still win."""
    full = _row(snapshot_kind="full")
    daily = _row(snapshot_kind="daily")

    kept = one_row_per_account_day([full, daily])

    assert kept == [daily]
    later_full = _row(
        account_id="acct_2",
        snapshot_time=datetime(2025, 1, 1, 22, 0, tzinfo=IST),
        snapshot_kind="full",
    )
    kept_with_later_full = one_row_per_account_day(
        [
            _row(account_id="acct_2", snapshot_kind="daily"),
            later_full,
        ]
    )
    assert kept_with_later_full == [later_full]


# (b) Scoring receives feature values only: never identifiers and never hidden truth.
def test_score_account_days_hides_truth_and_identifiers(monkeypatch: pytest.MonkeyPatch) -> None:
    """The rules call sees canonical features only; no truth_ or identifier key is passed."""
    captured: list[dict[str, Any]] = []

    def fake_evaluate_rules(
        features: dict[str, Any], config: Any
    ) -> RulesDecision:
        captured.append(dict(features))
        return RulesDecision(
            fired=(("rule_3", "RB3_MANY_PAYERS_ONE_DAY"),),
            not_evaluable=(),
            score=1,
        )

    monkeypatch.setattr(level17_scoring, "evaluate_rules", fake_evaluate_rules)

    row = _row(account_age_days=30, distinct_payers_24h=10)
    scored = score_account_days([row], load_rules_config(), detector=None)

    assert len(captured) == 1
    seen = captured[0]
    assert set(seen) <= set(account_feature_names())
    assert seen == {"account_age_days": 30, "distinct_payers_24h": 10}
    assert not any(name.startswith("truth_") for name in seen)
    assert "bank_id" not in seen and "account_id" not in seen
    assert scored[0].rules_score == 1
    assert scored[0].rules_fired == ("RB3_MANY_PAYERS_ONE_DAY",)
    assert feature_values_only(row) == seen


# (c) An account can be genuine one day and a mule another: restricting the genuine day is
# genuine harm, and the mule side supplies the depth.
def test_ground_sets_and_genuine_harm_use_per_day_truth_roles() -> None:
    """Rule 3 fires on both days; only the genuine day is genuine harm and depth comes from d2."""
    genuine_day = _row(
        account_id="acct_1",
        snapshot_time=datetime(2025, 1, 1, 10, 0, tzinfo=IST),
        truth_role="genuine",
        truth_depth=None,
        distinct_payers_24h=10,
    )
    mule_day = _row(
        account_id="acct_1",
        snapshot_time=datetime(2025, 1, 2, 10, 0, tzinfo=IST),
        truth_role="mule",
        truth_depth=2,
        distinct_payers_24h=10,
    )

    scored = score_account_days(
        one_row_per_account_day([genuine_day, mule_day]),
        load_rules_config(),
        detector=None,
    )

    restricted = rules_restricted_days(scored)
    assert restricted == {"acct_1": {FIRST_DAY, SECOND_DAY}}

    genuine_restricted = genuine_restricted_days(restricted, scored)
    assert genuine_restricted == {"acct_1": {FIRST_DAY}}

    mules, genuine_active, depth_of, is_mule = ground_sets(scored)
    assert mules == {"acct_1"}
    assert genuine_active == {"acct_1"}
    assert depth_of == {"acct_1": 2}
    assert is_mule("acct_1", SECOND_DAY) is True
    assert is_mule("acct_1", FIRST_DAY) is False


def test_ground_sets_count_collectors_and_cash_outs_without_a_depth() -> None:
    """Collectors and cash-outs are mules for recall but carry no depth; truth requires depth None."""
    scored = [
        ScoredDay(
            bank_id="bank_a",
            account_id="collector_1",
            day=FIRST_DAY,
            rules_score=0,
            rules_fired=(),
            anomaly_score=None,
            role="collector",
            depth=None,
        ),
        ScoredDay(
            bank_id="bank_a",
            account_id="cash_out_1",
            day=FIRST_DAY,
            rules_score=0,
            rules_fired=(),
            anomaly_score=None,
            role="cash_out",
            depth=None,
        ),
    ]

    mules, genuine_active, depth_of, is_mule = ground_sets(scored)

    assert mules == {"collector_1", "cash_out_1"}
    assert genuine_active == set()
    assert depth_of == {}
    assert is_mule("collector_1", FIRST_DAY) is True
    assert is_mule("cash_out_1", FIRST_DAY) is True


def test_ground_sets_refuse_a_mule_day_without_a_depth() -> None:
    """A role-"mule" day with depth None still raises, because its depth would be unknown."""
    scored = [
        ScoredDay(
            bank_id="bank_a",
            account_id="mule_1",
            day=FIRST_DAY,
            rules_score=0,
            rules_fired=(),
            anomaly_score=None,
            role="mule",
            depth=None,
        ),
    ]

    with pytest.raises(ValueError, match="Mule account mule_1 has no truth_depth"):
        ground_sets(scored)


# (d) Capacity is ceil(active * pct / 100), floored at one escalation per bank-day.
def test_daily_escalations_apply_bank_day_capacity() -> None:
    """1000 active accounts at 0.25% allow 3 escalations; 10 allow the floor of 1."""
    thousand = [
        ScoredDay(
            bank_id="bank_a",
            account_id=f"acct_{index:04d}",
            day=FIRST_DAY,
            rules_score=1,
            rules_fired=("RB3_MANY_PAYERS_ONE_DAY",),
            anomaly_score=None,
            role="genuine",
            depth=None,
        )
        for index in range(1000)
    ]
    assert len(daily_escalations(thousand, "rules", None, 0.25)[FIRST_DAY]) == 3

    ten = thousand[:10]
    assert len(daily_escalations(ten, "rules", None, 0.25)[FIRST_DAY]) == 1


# (e) Ranking ties are broken by account id, and anomaly modes refuse rows without scores.
def test_daily_escalations_breaks_score_ties_by_account_id() -> None:
    """Two equal-score candidates at capacity one resolve to the lexicographically first id."""
    tied = [
        ScoredDay(
            bank_id="bank_a",
            account_id="acct_z",
            day=FIRST_DAY,
            rules_score=1,
            rules_fired=("RB3_MANY_PAYERS_ONE_DAY",),
            anomaly_score=None,
            role="genuine",
            depth=None,
        ),
        ScoredDay(
            bank_id="bank_a",
            account_id="acct_a",
            day=FIRST_DAY,
            rules_score=1,
            rules_fired=("RB3_MANY_PAYERS_ONE_DAY",),
            anomaly_score=None,
            role="genuine",
            depth=None,
        ),
    ]

    escalations = daily_escalations(tied, "rules", None, 10.0)

    assert escalations == {FIRST_DAY: ["acct_a"]}


def test_daily_escalations_anomaly_mode_requires_scores() -> None:
    """Mode 'anomaly' raises rather than treating a missing score as zero."""
    row = ScoredDay(
        bank_id="bank_a",
        account_id="acct_1",
        day=FIRST_DAY,
        rules_score=0,
        rules_fired=(),
        anomaly_score=None,
        role="genuine",
        depth=None,
    )

    with pytest.raises(ValueError, match="requires anomaly scores on every row"):
        daily_escalations([row], "anomaly", 0.5, 10.0)
