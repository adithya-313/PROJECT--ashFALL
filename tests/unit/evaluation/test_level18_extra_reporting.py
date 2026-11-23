"""
WHAT: Unit tests for evaluation/level18_extra_reporting.py: the frozen-model guards, the mandatory
consistency gate, the pure account metrics, the normalized capacity ratio and held-out family recall.

WHY: This module reports on a model that was already frozen, so every guard and ratio must fail
loudly on drift; a silent fallback would produce numbers nobody can trace back to the checkpoint.
"""

import hashlib
from datetime import date
from pathlib import Path

import pytest

from evaluation.level17_scoring import AccountKey, ScoredDay
from evaluation.level18_extra_reporting import (
    PINNED_ACCOUNT_LABELS_CHANGED_LINES,
    _pure_metrics,
    _WorldEvidence,
    family_recall,
    normalized_recall,
    require_allowed_drift,
    require_consistency,
    require_mule_model_config_sha,
)
from evaluation.truth import TruthStore

DAY = date(2026, 1, 1)
WORLD = "world_1"
BANK = "bank_1"


def _key(account_id: str) -> AccountKey:
    """Builds the (world, bank, account) key the metric functions count accounts by."""
    return (WORLD, BANK, account_id)


def _scored(account_id: str, role: str, probability: float, depth: int | None) -> ScoredDay:
    """Builds one scored account-day carrying a model probability and a hidden truth role."""
    return ScoredDay(
        world_id=WORLD,
        bank_id=BANK,
        account_id=account_id,
        day=DAY,
        rules_score=0,
        rules_fired=(),
        anomaly_score=probability,
        role=role,
        depth=depth,
    )


def _family_row(account_id: str, role: str, owner: str | None) -> dict[str, object]:
    """Builds one joined snapshot row carrying its hidden role and owning gang."""
    return {
        "world_id": WORLD,
        "bank_id": BANK,
        "account_id": account_id,
        "truth_role": role,
        "truth_owner": owner,
    }


def test_normalized_recall_is_none_at_zero_oracle_and_keeps_above_one() -> None:
    """
    WHAT: normalized_recall returns None when the oracle reached nothing, splits the other way
    when the oracle reached everything, and keeps a value above 1.0.
    WHY: The brief allows a model to over-reach a capped oracle and forbids dividing by zero.
    """
    assert normalized_recall(0.4, 0.0) is None
    assert normalized_recall(0.5, 1.0) == 0.5
    assert normalized_recall(1.0, 0.5) == 2.0


def test_pure_metrics_on_a_hand_example() -> None:
    """
    WHAT: Pure recall, precision and genuine flag rate on four accounts where exactly one mule and
    one genuine account reached the cut.
    WHY: Each ratio must divide by the right population: mules for recall, flags for precision and
    genuine accounts for the genuine flag rate.
    """
    scored = [
        _scored("acc_mule_1", "mule", 0.90, 1),
        _scored("acc_mule_2", "mule", 0.20, 2),
        _scored("acc_genuine_1", "genuine", 0.80, None),
        _scored("acc_genuine_2", "genuine", 0.10, None),
    ]
    evidence = _WorldEvidence(
        world_id=WORLD,
        run_id="run_1",
        xgboost_cut=0.5,
        scored=scored,
        probabilities=[0.90, 0.20, 0.80, 0.10],
        day_rows=[],
        mules={_key("acc_mule_1"), _key("acc_mule_2")},
        genuine_active={_key("acc_genuine_1"), _key("acc_genuine_2")},
        depth_of={},
        flagged_at_cut={_key("acc_mule_1"), _key("acc_genuine_1")},
        reported=set(),
    )

    pure = _pure_metrics(evidence)

    assert pure["recall_at_cut"] == 0.5
    assert pure["precision_at_cut"] == 0.5
    assert pure["genuine_flag_rate"] == 0.5
    assert 0.0 <= pure["account_day_average_precision"] <= 1.0


def test_family_recall_on_a_hand_example() -> None:
    """
    WHAT: Family recall counts each held-out family's own mules and divides by that family alone,
    and a family absent from the world is omitted.
    WHY: Pooling families would hide a weak tactic behind a strong one; the hand example makes each
    family's numerator and denominator visible.
    """
    store = TruthStore(
        roles=[],
        device_control=[],
        payment_stories={},
        gangs=[
            {"gang_id": "gang_slow", "kind": "seen", "family": "slow_drip"},
            {"gang_id": "gang_dev", "kind": "held_out", "family": "device_rotation"},
        ],
        personas={},
    )
    day_rows = [
        _family_row("acc_slow_1", "mule", "gang_slow"),
        _family_row("acc_slow_2", "collector", "gang_slow"),
        _family_row("acc_dev_1", "mule", "gang_dev"),
        _family_row("acc_dev_2", "cash_out", "gang_dev"),
        _family_row("acc_genuine", "genuine", None),
    ]
    flagged = {_key("acc_slow_1"), _key("acc_dev_1")}
    escalations = {DAY: [_key("acc_slow_1"), _key("acc_dev_2")]}

    result = family_recall(day_rows, flagged, escalations, store, ["slow_drip", "device_rotation", "absent_family"])

    assert result == {
        "slow_drip": {"accounts": 2, "recall_at_capacity": 0.5, "recall_at_cut": 0.5},
        "device_rotation": {"accounts": 2, "recall_at_capacity": 0.5, "recall_at_cut": 0.5},
    }


def test_family_recall_refuses_a_mule_row_without_an_owner_gang() -> None:
    """
    WHAT: A mule-role row with no owning gang stops the run.
    WHY: Without an owner the family is unknown, and guessing it would misattribute the account.
    """
    store = TruthStore(roles=[], device_control=[], payment_stories={}, gangs=[], personas={})

    with pytest.raises(ValueError, match="has no owner gang"):
        family_recall([_family_row("acc_mule", "mule", None)], set(), {}, store, ["slow_drip"])


def test_consistency_gate_stops_on_a_mismatch() -> None:
    """
    WHAT: Matching stored and recomputed recalls pass, and one differing world stops the gate with
    both values named.
    WHY: Equality with the frozen run's stored value is the proof that this report scored the same
    model on the same rows.
    """
    require_consistency([("world_a", 0.5, 0.5), ("world_b", 0.25, 0.25)])

    with pytest.raises(ValueError) as error:
        require_consistency([("world_a", 0.5, 0.5), ("world_c", 0.25, 0.5)])

    message = str(error.value)
    assert "world_c" in message
    assert "0.25" in message
    assert "0.5" in message


def test_drift_guard_refuses_a_changed_path_outside_the_allow_list() -> None:
    """
    WHAT: The expected post-freeze paths pass, and any other path is refused by name.
    WHY: The frozen numbers are only comparable while no offline scoring code changed after the fit.
    """
    require_allowed_drift(
        [
            "evaluation/level18_extra_reporting.py",
            "evaluation/truth.py",
            "docs/research/amendments.md",
            "tests/unit/evaluation/test_level18_extra_reporting.py",
        ],
        None,
    )

    with pytest.raises(ValueError, match="api/server.py"):
        require_allowed_drift(["evaluation/level18_extra_reporting.py", "api/server.py"], None)


def test_account_labels_guard_accepts_the_pinned_diff_and_refuses_an_extra_line() -> None:
    """
    WHAT: features/account_labels.py passes only with exactly the pinned log-level swap lines, and
    one extra changed line is refused.
    WHY: The file sits in the scoring path's import chain, so its allowed change is pinned to the
    exact bytes rather than to the path alone.
    """
    require_allowed_drift(["features/account_labels.py"], PINNED_ACCOUNT_LABELS_CHANGED_LINES)

    with pytest.raises(ValueError, match="beyond the pinned log-level swap"):
        require_allowed_drift(
            ["features/account_labels.py"],
            [*PINNED_ACCOUNT_LABELS_CHANGED_LINES, "+        logger.warning("],
        )


def test_config_sha_guard_refuses_a_mismatch(tmp_path: Path) -> None:
    """
    WHAT: The guard returns the config's real digest and refuses any other expected digest.
    WHY: The mule model config decides the feature contract and the rules-table switch, so a
    different file would score the frozen model on different inputs.
    """
    config = tmp_path / "final_mule_model.yaml"
    body = b"pu_method: bagging\n"
    config.write_bytes(body)
    actual = hashlib.sha256(body).hexdigest()

    assert require_mule_model_config_sha(config, actual) == actual

    with pytest.raises(ValueError, match="checkpoint was fitted with"):
        require_mule_model_config_sha(config, "0" * 64)
