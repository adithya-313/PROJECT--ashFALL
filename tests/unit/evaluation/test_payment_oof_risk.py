"""
WHAT: Tests the out-of-fold payee-risk module: fold assignment, the strict earlier-than lookup, the
same-bank/NaN rule, and the training/validation-only split guard.

WHY: These are the pre-registered invariants of A-055. A snapshot at exactly the decision time must
never be used, a cross-bank receiver must be missing rather than scored, and every world must land
in exactly one fold, or the payment model would see a receiver score that leaked its own world.
"""

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pytest

from evaluation.payment_oof_risk import (
    REASON_BANK_UNKNOWN,
    REASON_NO_PRIOR_DAY,
    REASON_OTHER_BANK,
    AccountDayScore,
    assign_folds,
    latest_prior_score,
    payee_risk,
    validate_dataset_split,
)

UTC = timezone.utc


def _moment(hour: int) -> datetime:
    """Returns a UTC datetime on 2026-09-10 at the given hour."""
    return datetime(2026, 9, 10, hour, 0, 0, tzinfo=UTC)


def _write_manifest(dataset_dir: Path, split: str) -> None:
    """Writes a minimal manifest.json with the given split so the guard can read it."""
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "manifest.json").write_text(
        json.dumps({"split": split, "worlds": [{"world_id": "w-0", "run_id": "r-0"}]}),
        encoding="utf-8",
    )


def test_assign_folds_puts_every_world_in_exactly_one_fold() -> None:
    world_ids = [f"training-b02-{index:04d}" for index in range(320)]

    folds = assign_folds(world_ids)

    assert sorted(folds) == sorted(world_ids)
    assert all(0 <= fold < 5 for fold in folds.values())
    ordered = sorted(world_ids)
    for position, world_id in enumerate(ordered):
        assert folds[world_id] == position % 5
    sizes = {fold: sum(1 for value in folds.values() if value == fold) for fold in range(5)}
    assert sizes == {0: 64, 1: 64, 2: 64, 3: 64, 4: 64}


def test_assign_folds_is_deterministic_regardless_of_input_order() -> None:
    world_ids = ["world-c", "world-a", "world-b", "world-e", "world-d"]

    forward = assign_folds(world_ids)
    backward = assign_folds(list(reversed(world_ids)))

    assert forward == backward
    assert forward == {"world-a": 0, "world-b": 1, "world-c": 2, "world-d": 3, "world-e": 4}


def test_assign_folds_rejects_a_duplicate_world() -> None:
    with pytest.raises(ValueError, match="unique"):
        assign_folds(["world-a", "world-a"])


def test_snapshot_at_exactly_decision_time_is_not_used() -> None:
    earlier = AccountDayScore(_moment(9), 0.10)
    at_decision = AccountDayScore(_moment(12), 0.90)

    prior = latest_prior_score([earlier, at_decision], _moment(12))
    assert prior is earlier

    value, reason = payee_risk([earlier, at_decision], _moment(12), "bank-a", "bank-a")
    assert reason == ""
    assert value == 0.10


def test_latest_prior_picks_the_closest_earlier_day() -> None:
    scores = [
        AccountDayScore(_moment(8), 0.20),
        AccountDayScore(_moment(10), 0.40),
        AccountDayScore(_moment(11), 0.60),
        AccountDayScore(_moment(13), 0.99),
    ]

    prior = latest_prior_score(scores, _moment(12))

    assert prior is scores[2]
    value, reason = payee_risk(scores, _moment(12), "bank-a", "bank-a")
    assert (value, reason) == (0.60, "")


def test_cross_bank_receiver_is_nan() -> None:
    scores = [AccountDayScore(_moment(9), 0.70)]

    value, reason = payee_risk(scores, _moment(12), "bank-other", "bank-a")

    assert math.isnan(value)
    assert reason == REASON_OTHER_BANK


def test_unknown_receiver_bank_is_nan() -> None:
    value, reason = payee_risk([], _moment(12), None, "bank-a")

    assert math.isnan(value)
    assert reason == REASON_BANK_UNKNOWN


def test_same_bank_receiver_with_no_earlier_day_is_nan() -> None:
    scores = [AccountDayScore(_moment(12), 0.70), AccountDayScore(_moment(13), 0.80)]

    value, reason = payee_risk(scores, _moment(12), "bank-a", "bank-a")

    assert math.isnan(value)
    assert reason == REASON_NO_PRIOR_DAY


def test_split_guard_rejects_test_and_stress(tmp_path: Path) -> None:
    test_dir = tmp_path / "test-dataset"
    stress_dir = tmp_path / "stress-dataset"
    _write_manifest(test_dir, "test")
    _write_manifest(stress_dir, "stress")

    with pytest.raises(ValueError, match="never read test or stress"):
        validate_dataset_split(test_dir, "training")
    with pytest.raises(ValueError, match="never read test or stress"):
        validate_dataset_split(stress_dir, "training")


def test_split_guard_rejects_a_folder_on_the_wrong_side(tmp_path: Path) -> None:
    validation_dir = tmp_path / "validation-dataset"
    _write_manifest(validation_dir, "validation")

    manifest = validate_dataset_split(validation_dir, "validation")
    assert manifest["split"] == "validation"
    with pytest.raises(ValueError, match="was passed as 'training'"):
        validate_dataset_split(validation_dir, "training")
