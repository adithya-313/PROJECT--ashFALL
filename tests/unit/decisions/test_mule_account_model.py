"""
WHAT: Unit tests for the Level 18 pilot account model: feature exclusion, label-day truncation,
unlabelled negatives, training guards, and sidecar-checked persistence.

WHY: The pilot model must never learn from hidden truth or from account-days after the bank already
knew about an account, and a reloaded model must score exactly the matrix it was trained on.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from decisions.mule_account_model import (
    account_model_feature_names,
    build_training_rows,
    load_model,
    predict_account_days,
    save_model,
    train_account_model,
)


def _snapshot(world_id: str, account_id: str, day: int, value: float) -> dict[str, object]:
    """Builds one daily snapshot row for the given account and September 2026 day."""
    return {
        "world_id": world_id,
        "bank_id": "bank_0",
        "account_id": account_id,
        "snapshot_time": datetime(2026, 9, day, 12, 0, tzinfo=timezone.utc),
        "snapshot_kind": "daily",
        "distinct_payers_24h": value,
        "account_age_days": day,
    }


def _label(world_id: str, account_id: str, day: int) -> dict[str, object]:
    """Builds one bank-visible fraud label row arriving on the given September 2026 day."""
    return {
        "world_id": world_id,
        "account_id": account_id,
        "source": "complaint",
        "verdict": "fraud",
        "arrival_time": datetime(2026, 9, day, 10, 0, tzinfo=timezone.utc),
    }


def _training_fixture() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Builds three positive and ten unlabelled accounts over three days."""
    snapshots: list[dict[str, object]] = []
    labels: list[dict[str, object]] = []
    for index in range(3):
        account_id = f"acc_positive_{index}"
        labels.append(_label("world-0", account_id, 2))
        for day in (1, 2, 3):
            snapshots.append(_snapshot("world-0", account_id, day, float(index + day)))
    for index in range(10):
        account_id = f"acc_negative_{index}"
        for day in (1, 2, 3):
            snapshots.append(_snapshot("world-0", account_id, day, float(index * 3 + day)))
    return snapshots, labels


def test_excluded_feature_never_appears_in_matrix() -> None:
    """The anomaly detector's excluded features must not be model inputs."""
    names = account_model_feature_names()
    assert "account_type" not in names
    assert "hops_to_confirmed_label" not in names
    assert "shared_counterparties_with_flagged_7d" not in names
    assert "hops_to_complained_account" not in names
    assert "registry_match_count" not in names
    assert "registry_match_age_days" not in names


def test_positive_account_days_after_first_label_day_are_dropped() -> None:
    """A positive account keeps only days on or before its first label's IST day."""
    snapshots = [
        _snapshot("world-0", "acc_positive", 1, 1.0),
        _snapshot("world-0", "acc_positive", 2, 2.0),
        _snapshot("world-0", "acc_positive", 3, 3.0),
    ]
    labels = [_label("world-0", "acc_positive", 2)]

    matrix, targets, row_keys = build_training_rows(snapshots, labels)

    assert len(matrix) == 2
    assert targets == [1, 1]
    assert [key[3].isoformat() for key in row_keys] == ["2026-09-01", "2026-09-02"]


def test_unlabelled_account_keeps_all_its_days() -> None:
    """An account with no label keeps every account-day as a negative."""
    snapshots = [
        _snapshot("world-0", "acc_positive", 2, 2.0),
        _snapshot("world-0", "acc_unlabelled", 1, 1.0),
        _snapshot("world-0", "acc_unlabelled", 2, 2.0),
        _snapshot("world-0", "acc_unlabelled", 3, 3.0),
    ]
    labels = [_label("world-0", "acc_positive", 2)]

    matrix, targets, row_keys = build_training_rows(snapshots, labels)

    unlabelled_keys = [key for key in row_keys if key[2] == "acc_unlabelled"]
    unlabelled_targets = [
        target for target, key in zip(targets, row_keys, strict=True) if key[2] == "acc_unlabelled"
    ]
    assert len(unlabelled_keys) == 3
    assert unlabelled_targets == [0, 0, 0]


def test_training_without_positives_raises() -> None:
    """The model refuses to fit when there is no positive account-day."""
    with pytest.raises(ValueError, match="at least one positive"):
        train_account_model([[1.0]], [0], seed=1)


def test_training_without_negatives_raises() -> None:
    """The model refuses to fit when there is no negative account-day."""
    with pytest.raises(ValueError, match="at least one negative"):
        train_account_model([[1.0]], [1], seed=1)


def test_save_then_load_gives_identical_predictions(tmp_path: Path) -> None:
    """A saved model reloads with the same feature order and the same probabilities."""
    snapshots, labels = _training_fixture()
    matrix, targets, _ = build_training_rows(snapshots, labels)
    model, scale_pos_weight = train_account_model(matrix, targets, seed=7, n_estimators=10)

    day_rows = [
        _snapshot("world-0", "acc_negative_0", 1, 1.0),
        _snapshot("world-0", "acc_positive_0", 1, 2.0),
    ]
    before = predict_account_days(model, day_rows)

    model_path = tmp_path / "account_model.ubj"
    save_model(
        model,
        model_path,
        {
            "seed": 7,
            "scale_pos_weight": scale_pos_weight,
            "training_dataset_ids": ["fixture-training"],
        },
    )
    loaded = load_model(model_path)
    after = predict_account_days(loaded, day_rows)

    assert after == pytest.approx(before)


def test_feature_order_mismatch_on_load_raises(tmp_path: Path) -> None:
    """A sidecar whose feature order changed is refused instead of scoring the wrong columns."""
    snapshots, labels = _training_fixture()
    matrix, targets, _ = build_training_rows(snapshots, labels)
    model, scale_pos_weight = train_account_model(matrix, targets, seed=7, n_estimators=10)

    model_path = tmp_path / "account_model.ubj"
    save_model(
        model,
        model_path,
        {
            "seed": 7,
            "scale_pos_weight": scale_pos_weight,
            "training_dataset_ids": ["fixture-training"],
        },
    )

    sidecar_path = Path(str(model_path) + ".json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["feature_names"] = list(reversed(sidecar["feature_names"]))
    sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="feature order"):
        load_model(model_path)
