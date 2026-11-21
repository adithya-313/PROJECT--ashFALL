"""
WHAT: Tests that two worlds reusing the same account ids and dates stay two sets of accounts.

WHY: Account ids repeat across simulated worlds. Keying metrics by the bare account id merged different
worlds' accounts, so validation tuning saw 382 "accounts" instead of 42,279 and could not meet its
budget. Every fixture here has world 1 and world 2 share the ids "acc_1" and "acc_2" and the same
calendar day, with acc_1 a mule in world 1 and genuine in world 2, so any bare-id keying fails.
"""

from datetime import date, datetime, timezone

import pytest

from evaluation.level17_evaluation import evaluate_world, first_flag_times
from evaluation.level17_scoring import ScoredDay, account_key, daily_escalations, ground_sets
from evaluation.rules_tuning import _account_peak_scores, score_cut_curve

DAY = date(2026, 9, 1)
SECOND_DAY = date(2026, 9, 2)
WORLD_ONE = "world_1"
WORLD_TWO = "world_2"


def _day(
    world_id: str,
    account_id: str,
    role: str,
    anomaly_score: float | None = None,
    rules_score: int = 0,
    day: date = DAY,
) -> ScoredDay:
    """Builds one scored account-day at bank_a; a mule gets depth 1, anyone else no depth."""
    return ScoredDay(
        world_id=world_id,
        bank_id="bank_a",
        account_id=account_id,
        day=day,
        rules_score=rules_score,
        rules_fired=("RB3_MANY_PAYERS_ONE_DAY",) if rules_score else (),
        anomaly_score=anomaly_score,
        role=role,
        depth=1 if role == "mule" else None,
    )


def _two_worlds_with_shared_ids() -> list[ScoredDay]:
    """
    World 1: acc_1 is a mule (score 0.9), acc_2 genuine (0.1).
    World 2: acc_1 is genuine (0.4), acc_2 genuine (0.2). World 2 comes last so a bare-id map would
    let world 2's genuine role overwrite world 1's mule role on the same day.
    """
    return [
        _day(WORLD_ONE, "acc_1", "mule", anomaly_score=0.9, rules_score=1),
        _day(WORLD_ONE, "acc_2", "genuine", anomaly_score=0.1),
        _day(WORLD_TWO, "acc_1", "genuine", anomaly_score=0.4),
        _day(WORLD_TWO, "acc_2", "genuine", anomaly_score=0.2),
    ]


def test_account_peak_scores_keep_the_two_worlds_apart() -> None:
    genuine_peak, mule_peak = _account_peak_scores(_two_worlds_with_shared_ids())

    assert mule_peak == {(WORLD_ONE, "bank_a", "acc_1"): 0.9}
    assert genuine_peak == {
        (WORLD_ONE, "bank_a", "acc_2"): 0.1,
        (WORLD_TWO, "bank_a", "acc_1"): 0.4,
        (WORLD_TWO, "bank_a", "acc_2"): 0.2,
    }


def test_score_cut_curve_matches_the_per_world_hand_computed_rate_and_recall() -> None:
    # 3 genuine accounts (peaks 0.1, 0.2, 0.4) and 1 mule account (peak 0.9). A cut restricts the
    # genuine accounts whose peak is at or above it; the mule is above every cut.
    curve = score_cut_curve(_two_worlds_with_shared_ids())

    assert [cut.cut for cut in curve] == [0.1, 0.2, 0.4]
    assert [cut.validation_rate_pct for cut in curve] == pytest.approx([100.0, 200.0 / 3.0, 100.0 / 3.0])
    assert [cut.validation_recall for cut in curve] == [1.0, 1.0, 1.0]


def test_ground_sets_count_each_worlds_account_separately() -> None:
    mules, genuine_active, depth_of, is_mule = ground_sets(_two_worlds_with_shared_ids())

    mule_account = (WORLD_ONE, "bank_a", "acc_1")
    genuine_account = (WORLD_TWO, "bank_a", "acc_1")
    assert mules == {mule_account}
    assert genuine_account in genuine_active and mule_account not in genuine_active
    assert len(genuine_active) == 3
    assert depth_of == {mule_account: 1}
    assert is_mule(mule_account, DAY) is True
    assert is_mule(genuine_account, DAY) is False


def _evaluate_two_worlds() -> dict[str, object]:
    """Runs evaluate_world with the rules scorer over both worlds' rows at 100% capacity."""
    scored = _two_worlds_with_shared_ids()
    flag_time = datetime(2026, 9, 1, 21, 0, tzinfo=timezone.utc)
    snapshot_times = {(account_key(scored_day), scored_day.day): flag_time for scored_day in scored}
    return evaluate_world(
        scored=scored,
        scorer="rules",
        anomaly_threshold=None,
        capacity_pct=100.0,
        inflation_factor=1.0,
        persona_of=lambda account_id: "person",
        weights={"person": 1},
        default_weight=1,
        snapshot_time_by_account_day=snapshot_times,
        first_credit={"acc_1": datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)},
        reported=set(),
    )


def test_daily_capacity_counts_both_worlds_accounts() -> None:
    # 4 distinct accounts at 100% capacity -> daily capacity 4. One escalation a day, so the share of
    # capacity used is 1/4. Keyed by bare id there would be 2 accounts and the share would be 1/2.
    result = _evaluate_two_worlds()

    assert result["analyst_load"] == {"mean": 1.0, "max": 1, "mean_share_of_capacity": 0.25}


def test_escalation_counts_the_mule_and_not_the_genuine_account_with_the_same_id() -> None:
    result = _evaluate_two_worlds()

    # Only world 1's acc_1 fires a rule and it is a mule: one true positive, no false positive.
    assert result["yield_rate"] == {"measured": 1.0, "reweighted": 1.0}
    assert result["recall_at_capacity"] == 1.0
    # World 2's genuine acc_1 has the same id and day but fired nothing, so there is no genuine harm.
    harm = result["genuine_harm"]
    assert isinstance(harm, dict) and harm["restricted_accounts"] == 0


def test_time_to_detection_uses_the_flagging_worlds_time() -> None:
    result = _evaluate_two_worlds()

    assert result["time_to_detection"] == {"median": 11.0, "p90": 11.0, "detected_count": 1, "undetected_count": 0}


def test_first_flag_times_gives_each_world_its_own_first_flag() -> None:
    scored = _two_worlds_with_shared_ids() + [
        _day(WORLD_ONE, "acc_1", "mule", day=SECOND_DAY),
        _day(WORLD_TWO, "acc_1", "genuine", day=SECOND_DAY),
    ]
    world_one = (WORLD_ONE, "bank_a", "acc_1")
    world_two = (WORLD_TWO, "bank_a", "acc_1")
    first_seen = datetime(2026, 9, 1, 9, 0, tzinfo=timezone.utc)
    later_seen = datetime(2026, 9, 2, 9, 0, tzinfo=timezone.utc)
    snapshot_times = {
        (world_one, DAY): first_seen,
        (world_two, DAY): first_seen,
        (world_one, SECOND_DAY): later_seen,
        (world_two, SECOND_DAY): later_seen,
    }

    # World 2's acc_1 is escalated one day after world 1's: each keeps its own first flag.
    first_flag = first_flag_times({DAY: [world_one], SECOND_DAY: [world_two]}, scored, snapshot_times)

    assert first_flag == {world_one: first_seen, world_two: later_seen}


def test_ties_between_worlds_with_the_same_id_are_both_escalated() -> None:
    # Two worlds hold acc_1 on the same bank-day with equal scores: both must be chosen and named.
    scored = [
        _day(WORLD_ONE, "acc_1", "genuine", rules_score=1),
        _day(WORLD_TWO, "acc_1", "genuine", rules_score=1),
    ]

    escalations = daily_escalations(scored, "rules", None, 100.0)

    assert escalations == {DAY: [(WORLD_ONE, "bank_a", "acc_1"), (WORLD_TWO, "bank_a", "acc_1")]}
