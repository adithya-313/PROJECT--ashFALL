"""
WHAT: Unit tests for loading tuned serving rules configuration from config/rules_tuned.yaml.

WHY: Serving rules must be strictly validated at load time to ensure all 6 baseline thresholds
are fully tuned with proven values, preventing untuned defaults or misconfigured keys in production.
"""

from pathlib import Path

import pytest
import yaml

from decisions.rules_config import (
    all_thresholds_tuned,
    load_rules_config,
    load_serving_rules_config,
)

EXPECTED_TUNED_VALUES: dict[str, int | float] = {
    "large_credit_amount_paise": 5000000,
    "near_zero_balance_paise": 10000,
    "new_account_age_days": 30,
    "outflow_share": 0.9,
    "payer_count": 40,
    "time_window_seconds": 7200,
}


def test_serving_rules_config_loads_and_is_fully_tuned() -> None:
    """
    WHAT: Tests that the committed config/rules_tuned.yaml loads successfully and is fully tuned.

    WHY: Live decision serving requires an all-tuned rules table where no threshold remains at
    untuned_start.
    """
    config = load_serving_rules_config()
    assert all_thresholds_tuned(config) is True
    assert len(config.thresholds) == 6


def test_serving_rules_values_match_file() -> None:
    """
    WHAT: Tests that each threshold value in the loaded config equals the value in config/rules_tuned.yaml.

    WHY: Ensures with_thresholds correctly mapped every single tuned value onto the rules table.
    """
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    tuned_yaml_path = repo_root / "config" / "rules_tuned.yaml"
    file_data = yaml.safe_load(tuned_yaml_path.read_text(encoding="utf-8"))
    file_thresholds = file_data["thresholds"]

    config = load_serving_rules_config()
    for key, expected_val in file_thresholds.items():
        assert key in config.thresholds
        threshold = config.thresholds[key]
        assert threshold.value == expected_val
        assert threshold.status == "tuned"
        assert threshold.value == EXPECTED_TUNED_VALUES[key]


def test_serving_rules_missing_threshold_raises(tmp_path: Path) -> None:
    """
    WHAT: Tests that a serving rules YAML missing one threshold raises ValueError.

    WHY: Refuses partial tuning configuration so untuned defaults cannot silently persist.
    """
    partial_thresholds = dict(EXPECTED_TUNED_VALUES)
    del partial_thresholds["payer_count"]

    bad_yaml = tmp_path / "missing_threshold.yaml"
    bad_yaml.write_text(yaml.safe_dump({"thresholds": partial_thresholds}), encoding="utf-8")

    with pytest.raises(ValueError, match="Missing threshold names in serving rules config"):
        load_serving_rules_config(serving_rules_path=bad_yaml)


def test_serving_rules_unknown_threshold_raises(tmp_path: Path) -> None:
    """
    WHAT: Tests that a serving rules YAML with an unknown threshold name raises ValueError.

    WHY: Prevents typos or extraneous extension parameters from entering the baseline rules table.
    """
    unknown_thresholds = dict(EXPECTED_TUNED_VALUES)
    unknown_thresholds["unrecognized_threshold_paise"] = 99999

    bad_yaml = tmp_path / "unknown_threshold.yaml"
    bad_yaml.write_text(yaml.safe_dump({"thresholds": unknown_thresholds}), encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown threshold names in serving rules config"):
        load_serving_rules_config(serving_rules_path=bad_yaml)


def test_shipped_baseline_remains_untuned() -> None:
    """
    WHAT: Tests that loading the baseline configuration still yields untuned_start thresholds.

    WHY: Guarantees that config/rules_baseline.yaml and load_rules_config() behaviour remains
    unchanged, as required by existing Level 17 evaluations and pinning tests.
    """
    baseline = load_rules_config()
    assert all_thresholds_tuned(baseline) is False
    for threshold in baseline.thresholds.values():
        assert threshold.status == "untuned_start"
