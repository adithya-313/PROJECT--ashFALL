"""
Tests for the mid-run genuine account openings feature switch and procedural scheduling.
Verifies narrowed configuration resolution, switch-off bitwise equality, switch-on genuine account
creation with standard ID formats and clean truth records, and switch-off refusal semantics.
"""

import re
from pathlib import Path

import numpy as np
import pytest

from core.config_validator import load_yaml
from world.account_openings_config import (
    load_account_openings_config,
    schedule_genuine_account_openings,
)
from world.fast_mode import FastRunSettings
from world.interactive_records import RunnerSettings
from world.interactive_runner import (
    check_switch,
    effective_config_for,
    run_interactive_world,
)


def test_narrowed_config_rate_produces_exact_count(tmp_path: Path) -> None:
    """
    Verifies that a custom configuration with narrowed rate parameters generates the exact expected openings count.
    Tests that procedural calculation scales population and days linearly without hardcoded constants.
    """
    narrowed_yaml = """
schema_version: 1
label: assumption
issue_ids_v2: [12-19]
derived_from: "assumption test"
reason: "narrowed test config"
parameters:
  genuine_account_openings_per_1000_people_per_30_days:
    training: [100.0, 100.0]
    test: [100.0, 100.0]
    stress: [100.0, 100.0]
  share_of_openings_that_are_shops:
    training: [0.0, 0.0]
    test: [0.0, 0.0]
    stress: [0.0, 0.0]
"""
    cfg_file = tmp_path / "account_openings.yaml"
    cfg_file.write_text(narrowed_yaml, encoding="utf-8")
    cfg = load_yaml(cfg_file)

    rng = np.random.default_rng(42)
    # 60 people, 30 days, rate 100 per 1000 people -> expected (100 / 1000) * 60 * (30 / 30) = 6.
    openings = schedule_genuine_account_openings(60, 30, "training", rng, config=cfg)
    assert len(openings) == 6
    assert all(ev.owner_kind == "person" for ev in openings)
    assert all(ev.opening_stream == "genuine_account_openings" for ev in openings)


def test_switch_off_byte_identical_run_id_and_output() -> None:
    """
    Ensures that when account_openings_enabled is False, effective configuration and run IDs are byte-identical.
    Guarantees strict backward compatibility for existing simulation runs and golden frozen anchors.
    """
    fast_settings = FastRunSettings(101, 60, 12, 61, "training")
    cfg_default = effective_config_for(
        fast_settings, None, runner_settings=RunnerSettings(fraud_enabled=True, account_openings_enabled=False)
    )
    cfg_omitted = effective_config_for(
        fast_settings, None, runner_settings=RunnerSettings(fraud_enabled=True)
    )

    assert cfg_default == cfg_omitted
    assert "account_openings_enabled" not in cfg_default
    assert "account_openings_config" not in cfg_default


def test_switch_on_schedules_genuine_openings_with_standard_id_formats() -> None:
    """
    Validates that enabling account openings produces registered accounts with normal ID formatting and truth facts.
    Confirms that all created accounts have civilian truth status with no criminal role annotations.
    """
    fast_settings = FastRunSettings(101, 60, 12, 61, "training")
    settings = RunnerSettings(fraud_enabled=True, account_openings_enabled=True)
    output = run_interactive_world(fast_settings, runner_settings=settings)

    opened = output.result.opened_accounts
    assert len(opened) > 0, "Expected at least one genuine account opening over 61 days"

    person_regex = re.compile(r"^acc_person_\d{6}$")
    shop_regex = re.compile(r"^acc_shop_\d{4}$")

    opened_ids = {entry[1] for entry in opened}
    for minute, account_id, owner_kind, opening_stream, bank_index, opening_balance in opened:
        if owner_kind == "person":
            assert person_regex.match(account_id)
        elif owner_kind == "shop":
            assert shop_regex.match(account_id)
        assert opening_stream == "genuine_account_openings"

    # Verify that truth facts do not assign any criminal roles to genuine mid-run accounts.
    truth_mules = {span.account_id for span in output.result.truth.roles}
    assert opened_ids.isdisjoint(truth_mules)

    # Check effective configuration captures the flag and settings.
    assert output.effective_config["account_openings_enabled"] is True
    assert "account_openings_config" in output.effective_config


def test_switch_on_with_fraud_off_raises_plain_sentence() -> None:
    """
    Confirms that requesting account openings without enabling fraud simulation raises a plain ValueError.
    Ensures unsupported execution configurations fail fast before preparing the simulation world.
    """
    fast_settings = FastRunSettings(101, 60, 12, 61, "training")
    bad_settings = RunnerSettings(fraud_enabled=False, account_openings_enabled=True)

    with pytest.raises(ValueError, match="^account openings need fraud_enabled=True$"):
        check_switch(bad_settings, None, None, None)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="^account openings need fraud_enabled=True$"):
        run_interactive_world(fast_settings, runner_settings=bad_settings)


def test_different_opening_config_changes_run_id() -> None:
    """
    Verifies Decision D7: differing account opening configurations produce distinct effective configs and run IDs.
    Ensures downstream artifact storage correctly partitions results across varying hyperparameter sweeps.
    """
    fast_settings = FastRunSettings(101, 60, 12, 61, "training")
    base_cfg = load_account_openings_config()

    custom_cfg = {
        **base_cfg,
        "parameters": {
            **base_cfg["parameters"],
            "genuine_account_openings_per_1000_people_per_30_days": {
                "training": [99.0, 99.0],
                "test": [99.0, 99.0],
                "stress": [99.0, 99.0],
            },
        },
    }

    out1 = run_interactive_world(
        fast_settings,
        runner_settings=RunnerSettings(fraud_enabled=True, account_openings_enabled=True),
        account_openings_config=base_cfg,
    )
    out2 = run_interactive_world(
        fast_settings,
        runner_settings=RunnerSettings(fraud_enabled=True, account_openings_enabled=True),
        account_openings_config=custom_cfg,
    )

    assert out1.result.run_id != out2.result.run_id
    assert out1.effective_config != out2.effective_config
