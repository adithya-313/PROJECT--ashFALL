# File: tests/unit/test_account_snapshots.py
# Purpose: Unit tests for account snapshots configuration, feature naming, and watermark gating.
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import fakeredis
import pytest

from features.account_snapshots import (
    NoWatermarkError,
    SnapshotSettings,
    account_feature_names,
    account_feature_names_sha256,
    full_snapshot_due,
    load_snapshot_settings,
    to_account_ordered_values,
    write_account_snapshots,
)
from features.feature_log import canonical_feature_names
from features.feature_set import load_feature_set_m1


def test_account_feature_names_equals_canonical_minus_payment_level_in_order() -> None:
    """Verifies account_feature_names() matches canonical names minus payment_level in identical order."""
    feature_set = load_feature_set_m1()
    payment_names = {f.name for f in feature_set.features_by_group("payment_level")}
    expected_names = tuple(f.name for f in feature_set.features if f.name not in payment_names)

    names = account_feature_names()
    assert names == expected_names
    assert len(names) == len(canonical_feature_names()) - len(payment_names)
    assert len(names) == 44

    expected_sha = hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()
    assert account_feature_names_sha256() == expected_sha


def test_full_snapshot_due_cadence() -> None:
    """Verifies full_snapshot_due returns True on day intervals and False otherwise."""
    settings = SnapshotSettings(full_interval_days=7)
    assert full_snapshot_due(0, settings) is True
    assert full_snapshot_due(7, settings) is True
    assert full_snapshot_due(14, settings) is True
    assert full_snapshot_due(6, settings) is False
    assert full_snapshot_due(1, settings) is False
    assert full_snapshot_due(8, settings) is False


def test_load_snapshot_settings_valid() -> None:
    """Verifies load_snapshot_settings loads configured interval from account_features.yaml."""
    settings = load_snapshot_settings()
    assert settings.full_interval_days == 7


def test_load_snapshot_settings_custom_file(tmp_path: Path) -> None:
    """Verifies loading custom snapshot settings file."""
    cfg_file = tmp_path / "custom.yaml"
    cfg_file.write_text(
        "account_snapshots:\n  full_interval_days:\n    value: 10\n",
        encoding="utf-8",
    )
    settings = load_snapshot_settings(cfg_file)
    assert settings.full_interval_days == 10


def test_to_account_ordered_values_validation() -> None:
    """Verifies schema validation and exact ordering in to_account_ordered_values."""
    names = account_feature_names()
    valid_dict: dict[str, Any] = {name: i for i, name in enumerate(names)}
    ordered = to_account_ordered_values(valid_dict)
    assert ordered == list(range(len(names)))

    # Missing key raises ValueError
    incomplete_dict = dict(valid_dict)
    del incomplete_dict[names[0]]
    with pytest.raises(ValueError, match="Missing"):
        to_account_ordered_values(incomplete_dict)

    # Unexpected key raises ValueError
    extra_dict = dict(valid_dict)
    extra_dict["unexpected_feat"] = 123
    with pytest.raises(ValueError, match="Unexpected"):
        to_account_ordered_values(extra_dict)


@pytest.mark.anyio
async def test_write_account_snapshots_no_watermark_raises() -> None:
    """Verifies NoWatermarkError is raised and nothing is written when bank watermark is None."""
    fake_pg = AsyncMock()
    fake_redis = fakeredis.FakeRedis()  # No watermarks recorded

    now = datetime(2026, 10, 31, 12, 0, tzinfo=timezone.utc)
    with pytest.raises(NoWatermarkError, match="No watermark available for bank 'bank_0'"):
        await write_account_snapshots(
            pg=fake_pg,
            redis=fake_redis,
            run_id="66db9aa439b092ee",
            bank_id="bank_0",
            snapshot_time=now,
            snapshot_kind="daily",
            account_ids=["acc_1", "acc_2"],
        )

    # Ensure no database queries were executed
    fake_pg.execute.assert_not_called()

