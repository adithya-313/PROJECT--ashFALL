"""
WHAT: Unit tests for the Level 18a4 label-cut switch: the config default, the exact arrival-time cut,
its inclusive boundary, and refusal of unknown cuts or malformed config files.

WHY: A training row taken after the bank learned about an account leaks the label, so the builder
must cut at the label's exact arrival time by default and must never silently accept an unknown cut.
"""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from decisions.mule_account_model import (
    build_training_rows,
    load_mule_model_config,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config" / "mule_model.yaml"


def _snapshot(account_id: str, moment: datetime, value: float) -> dict[str, object]:
    """Builds one daily snapshot row for the given account and moment."""
    return {
        "world_id": "world-0",
        "bank_id": "bank_0",
        "account_id": account_id,
        "snapshot_time": moment,
        "snapshot_kind": "daily",
        "distinct_payers_24h": value,
        "account_age_days": moment.day,
    }


def _label(account_id: str, moment: datetime) -> dict[str, object]:
    """Builds one bank-visible fraud label row arriving at the given moment."""
    return {
        "world_id": "world-0",
        "account_id": account_id,
        "source": "complaint",
        "verdict": "fraud",
        "arrival_time": moment,
    }


def _write_config(path: Path, body: str) -> None:
    """Writes a config file body verbatim."""
    path.write_text(body, encoding="utf-8")


def test_config_default_cut_is_arrival_time() -> None:
    """The committed config selects the exact arrival-time cut and leaves maturity off."""
    assert load_mule_model_config().label_cut == "arrival_time"
    assert load_mule_model_config(CONFIG_PATH).label_cut == "arrival_time"
    assert load_mule_model_config().label_maturity_days is None


def test_default_cut_drops_label_day_snapshot_after_arrival() -> None:
    """Without an explicit cut the builder uses the config and drops the snapshot after arrival."""
    snapshots = [
        _snapshot("acc_labelled", datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc), 1.0),
        _snapshot("acc_labelled", datetime(2026, 9, 2, 12, 0, tzinfo=timezone.utc), 2.0),
    ]
    labels = [_label("acc_labelled", datetime(2026, 9, 2, 10, 0, tzinfo=timezone.utc))]

    matrix, targets, row_keys = build_training_rows(snapshots, labels)

    assert len(matrix) == 1
    assert targets == [1]
    assert row_keys[0][3].isoformat() == "2026-09-01"


def test_snapshot_exactly_at_arrival_time_is_kept() -> None:
    """A snapshot at the same instant as the first label is still a row the bank could have seen."""
    snapshots = [
        _snapshot("acc_labelled", datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc), 1.0),
        _snapshot("acc_labelled", datetime(2026, 9, 2, 10, 0, tzinfo=timezone.utc), 2.0),
    ]
    labels = [_label("acc_labelled", datetime(2026, 9, 2, 10, 0, tzinfo=timezone.utc))]

    matrix, targets, row_keys = build_training_rows(snapshots, labels, label_cut="arrival_time")

    assert len(matrix) == 2
    assert targets == [1, 1]
    assert [key[3].isoformat() for key in row_keys] == ["2026-09-01", "2026-09-02"]


def test_unknown_label_cut_is_refused() -> None:
    """An unrecognised cut is an error, never a silent default."""
    with pytest.raises(ValueError, match="label_cut must be one of"):
        build_training_rows([], [], label_cut="yesterday")


def test_maturity_drops_each_worlds_own_trailing_days() -> None:
    """The maturity window is applied against each world's last day, never a global one."""
    snapshots: list[dict[str, object]] = []
    for snapshot_day in range(1, 11):
        snapshots.append(
            _snapshot("acc_long", datetime(2026, 9, snapshot_day, 12, 0, tzinfo=timezone.utc), float(snapshot_day))
        )
    for snapshot_day in range(1, 6):
        row = _snapshot("acc_short", datetime(2026, 9, snapshot_day, 12, 0, tzinfo=timezone.utc), float(snapshot_day))
        row["world_id"] = "world-1"
        snapshots.append(row)

    _, _, row_keys = build_training_rows(snapshots, [], label_maturity_days=2)

    kept: dict[str, list[str]] = {}
    for world_id, _, _, day in row_keys:
        kept.setdefault(world_id, []).append(day.isoformat())
    assert kept == {
        "world-0": [f"2026-09-{day:02d}" for day in range(1, 9)],
        "world-1": ["2026-09-01", "2026-09-02", "2026-09-03"],
    }


def test_negative_label_maturity_is_refused() -> None:
    """A negative maturity window is an error, never a silent no-op."""
    with pytest.raises(ValueError, match="label_maturity_days must be a non-negative integer"):
        build_training_rows([], [], label_maturity_days=-1)


def test_config_loader_refuses_unknown_value(tmp_path: Path) -> None:
    """A config whose label_cut is not an allowed cut is refused."""
    config_path = tmp_path / "mule_model.yaml"
    _write_config(
        config_path,
        "parameters:\n"
        "  label_cut:\n"
        "    label: policy\n"
        "    value: next_week\n"
        "    derived_from: test\n"
        "    reason: test\n"
        "  label_maturity_days:\n"
        "    label: policy\n"
        "    value: null\n"
        "    not_found: true\n"
        "    derived_from: test\n"
        "    reason: test\n",
    )

    with pytest.raises(ValueError, match="label_cut .* must be one of"):
        load_mule_model_config(config_path)


def test_config_loader_refuses_missing_file(tmp_path: Path) -> None:
    """A missing config file is an error, never an empty default."""
    with pytest.raises(FileNotFoundError, match="Mule model configuration not found"):
        load_mule_model_config(tmp_path / "absent.yaml")
