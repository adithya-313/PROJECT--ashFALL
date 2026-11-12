"""
Unit tests for decisions.rules_config loader and validation routines.
"""

from pathlib import Path

import pytest
import yaml

from decisions.rules_config import (
    all_thresholds_tuned,
    load_rules_config,
)


def _write_yaml(path: Path, data: dict) -> Path:
    """Helper writing a dict to a temporary YAML file."""
    path.write_text(yaml.dump(data), encoding="utf-8")
    return path


def _base_valid_config() -> dict:
    """Helper returning a minimal valid config dict for mutation."""
    return {
        "thresholds": {
            "large_credit_amount_paise": {
                "value": 1000000,
                "label": "policy",
                "reason": "Test reason",
                "derived_from": "test spec",
                "status": "untuned_start",
            },
        },
        "rules": [
            {
                "id": "rule_1",
                "reason_code": "RB1_TEST",
                "plain_english": "Test rule",
                "conditions": [
                    {
                        "feature": "largest_credit_paise_24h",
                        "op": ">=",
                        "threshold": "large_credit_amount_paise",
                    }
                ],
            }
        ],
    }


def test_load_committed_rules_baseline() -> None:
    """Verifies that config/rules_baseline.yaml loads cleanly with 5 rules and 6 thresholds."""
    config = load_rules_config()

    # 1. Assert exactly 5 rules and 6 thresholds
    assert len(config.rules) == 5
    assert len(config.thresholds) == 6

    # 2. Assert initial tuning status is untuned_start (all_thresholds_tuned is False)
    assert all_thresholds_tuned(config) is False
    assert config.all_thresholds_tuned() is False

    # 3. Assert rule IDs and reason codes match table order
    expected_rules = [
        ("rule_1", "RB1_FAST_PASS_THROUGH"),
        ("rule_2", "RB2_LARGE_CREDIT_NEW_ACCOUNT"),
        ("rule_3", "RB3_MANY_PAYERS_ONE_DAY"),
        ("rule_4", "RB4_NEAR_ZERO_AFTER_OUTFLOW"),
        ("rule_5", "RB5_NEW_DEVICE_MONEY_MOVED"),
    ]
    actual_rules = [(r.id, r.reason_code) for r in config.rules]
    assert actual_rules == expected_rules

    # 4. Assert threshold names
    expected_threshold_keys = {
        "large_credit_amount_paise",
        "outflow_share",
        "time_window_seconds",
        "new_account_age_days",
        "payer_count",
        "near_zero_balance_paise",
    }
    assert set(config.thresholds.keys()) == expected_threshold_keys


def test_all_thresholds_tuned_returns_true_when_all_tuned(tmp_path: Path) -> None:
    """Verifies all_thresholds_tuned returns True only when all threshold statuses are 'tuned'."""
    data = _base_valid_config()
    data["thresholds"]["large_credit_amount_paise"]["status"] = "tuned"
    cfg_file = _write_yaml(tmp_path / "rules.yaml", data)

    config = load_rules_config(cfg_file)
    assert all_thresholds_tuned(config) is True
    assert config.all_thresholds_tuned() is True


def test_validation_missing_file_raises(tmp_path: Path) -> None:
    """Verifies FileNotFoundError when path does not exist."""
    missing = tmp_path / "nonexistent.yaml"
    with pytest.raises(FileNotFoundError, match="Rules configuration file not found"):
        load_rules_config(missing)


def test_validation_unknown_feature_name_raises(tmp_path: Path) -> None:
    """Verifies ValueError when condition references an unknown feature name."""
    data = _base_valid_config()
    data["rules"][0]["conditions"][0]["feature"] = "completely_invented_feature_xyz"
    cfg_file = _write_yaml(tmp_path / "rules.yaml", data)

    with pytest.raises(ValueError, match="Unknown feature name 'completely_invented_feature_xyz'"):
        load_rules_config(cfg_file)


def test_validation_unknown_threshold_key_raises(tmp_path: Path) -> None:
    """Verifies ValueError when condition references an unknown threshold key."""
    data = _base_valid_config()
    data["rules"][0]["conditions"][0]["threshold"] = "nonexistent_threshold_key"
    cfg_file = _write_yaml(tmp_path / "rules.yaml", data)

    with pytest.raises(ValueError, match="Unknown threshold key 'nonexistent_threshold_key'"):
        load_rules_config(cfg_file)


def test_validation_unknown_operator_raises(tmp_path: Path) -> None:
    """Verifies ValueError when condition uses an invalid comparison operator."""
    data = _base_valid_config()
    data["rules"][0]["conditions"][0]["op"] = "<>"
    cfg_file = _write_yaml(tmp_path / "rules.yaml", data)

    with pytest.raises(ValueError, match="Unknown op '<>'"):
        load_rules_config(cfg_file)


def test_validation_duplicate_rule_id_raises(tmp_path: Path) -> None:
    """Verifies ValueError when multiple rules share the same id."""
    data = _base_valid_config()
    data["rules"].append(
        {
            "id": "rule_1",
            "reason_code": "RB2_OTHER",
            "plain_english": "Duplicate id",
            "conditions": [
                {
                    "feature": "largest_credit_paise_24h",
                    "op": ">=",
                    "threshold": "large_credit_amount_paise",
                }
            ],
        }
    )
    cfg_file = _write_yaml(tmp_path / "rules.yaml", data)

    with pytest.raises(ValueError, match="Duplicate rule id 'rule_1'"):
        load_rules_config(cfg_file)


def test_validation_duplicate_reason_code_raises(tmp_path: Path) -> None:
    """Verifies ValueError when multiple rules share the same reason code."""
    data = _base_valid_config()
    data["rules"].append(
        {
            "id": "rule_2",
            "reason_code": "RB1_TEST",
            "plain_english": "Duplicate reason code",
            "conditions": [
                {
                    "feature": "largest_credit_paise_24h",
                    "op": ">=",
                    "threshold": "large_credit_amount_paise",
                }
            ],
        }
    )
    cfg_file = _write_yaml(tmp_path / "rules.yaml", data)

    with pytest.raises(ValueError, match="Duplicate reason code 'RB1_TEST'"):
        load_rules_config(cfg_file)


def test_validation_rule_with_no_conditions_raises(tmp_path: Path) -> None:
    """Verifies ValueError when a rule defines an empty conditions list."""
    data = _base_valid_config()
    data["rules"][0]["conditions"] = []
    cfg_file = _write_yaml(tmp_path / "rules.yaml", data)

    with pytest.raises(ValueError, match="Rule 'rule_1' has no conditions configured"):
        load_rules_config(cfg_file)


@pytest.mark.parametrize("missing_field", ["label", "derived_from", "status"])
def test_validation_threshold_missing_metadata_raises(tmp_path: Path, missing_field: str) -> None:
    """Verifies ValueError when a threshold is missing label, derived_from, or status."""
    data = _base_valid_config()
    del data["thresholds"]["large_credit_amount_paise"][missing_field]
    cfg_file = _write_yaml(tmp_path / "rules.yaml", data)

    with pytest.raises(ValueError, match=f"missing or empty required field '{missing_field}'"):
        load_rules_config(cfg_file)

