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


def test_sizing_estimates_arithmetic_and_formatting() -> None:
    """Verifies resource estimation arithmetic and summary table formatting with fixed inputs."""
    from scripts.feature_log_size_estimate import (
        compute_preset_estimates,
        format_estimates_table,
    )
    from world.estimates import MachineProfile

    fixed_profile = MachineProfile(
        payments_per_second=5000.0,
        payments_per_person_day=0.8,
        file_bytes_per_payment=350.0,
        memory_bytes_per_payment=1000.0,
        calibration_payments=2915,
    )
    fixed_sizing_config = {
        "payment_count_margin": {"value": 1.15},
        "presets": {
            "tiny": {"n_people": 60, "n_shops": 12, "n_days": 61},
            "small": {"n_people": 300, "n_shops": 40, "n_days": 61},
            "medium": {"n_people": 1000, "n_shops": 120, "n_days": 61},
            "large": {"n_people": 2000, "n_shops": 250, "n_days": 61},
        },
    }

    f_row_bytes = 650.0
    s_row_bytes = 530.0
    tiny_accounts = 72

    estimates = compute_preset_estimates(
        sizing_config=fixed_sizing_config,
        profile=fixed_profile,
        feature_log_row_bytes=f_row_bytes,
        snapshot_row_bytes=s_row_bytes,
        tiny_accounts=tiny_accounts,
    )

    assert len(estimates) == 4
    tiny_est = estimates[0]
    assert tiny_est.name == "tiny"
    assert tiny_est.n_people == 60
    assert tiny_est.n_shops == 12
    assert tiny_est.n_days == 61
    assert tiny_est.estimated_accounts == 72
    # payments = ceil(60 * 61 * 0.8 * 1.15) = ceil(3367.2) = 3368
    assert tiny_est.total_payments == 3368
    expected_decisions_day = 3368 / 61
    assert abs(tiny_est.decisions_per_day - expected_decisions_day) < 1e-6
    expected_f_mb = (expected_decisions_day * f_row_bytes) / (1024 * 1024)
    assert abs(tiny_est.feature_log_mb_per_day - expected_f_mb) < 1e-6
    expected_s_mb = (72 * s_row_bytes) / (1024 * 1024)
    assert abs(tiny_est.daily_snapshot_mb_per_day - expected_s_mb) < 1e-6
    assert abs(tiny_est.full_snapshot_mb_per_week - expected_s_mb) < 1e-6

    # Verify table formatting
    table = format_estimates_table(estimates, f_row_bytes, s_row_bytes, tiny_accounts)
    assert "LEVEL 15c FEATURE LOG & ACCOUNT SNAPSHOTS SIZE ESTIMATES" in table
    assert "tiny" in table
    assert "small" in table
    assert "medium" in table
    assert "large" in table
    assert "DERIVATIONS & RATIONALE:" in table
    assert "[TAG: ESTIMATE]" in table
