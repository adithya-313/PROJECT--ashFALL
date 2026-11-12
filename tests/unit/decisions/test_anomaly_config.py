"""
Unit tests for decisions.anomaly_config loader and validation rules.
"""

from pathlib import Path

import pytest
import yaml

from decisions.anomaly_config import (
    input_feature_names,
    load_anomaly_detector_config,
    load_cold_start_config,
)


def _write_yaml(path: Path, data: dict) -> Path:
    """Helper writing dictionary data to a YAML file."""
    path.write_text(yaml.dump(data), encoding="utf-8")
    return path


def _base_valid_anomaly_data() -> dict:
    """Helper returning valid anomaly detector configuration dictionary."""
    return {
        "excluded_features": {
            "label": "policy",
            "reason": "Excluded features",
            "derived_from": "test spec",
            "value": {
                "hops_to_confirmed_label": "built from labels",
                "shared_counterparties_with_flagged_7d": "built from labels",
                "hops_to_complained_account": "built from labels",
                "registry_match_count": "registry listings come from complaints",
                "registry_match_age_days": "registry listings come from complaints",
                "account_type": "text category",
            },
        },
        "early_window_days": {
            "value": 14,
            "label": "policy",
            "reason": "early window",
            "derived_from": "test spec",
        },
        "n_estimators": {
            "value": 200,
            "label": "policy",
            "reason": "trees count",
            "derived_from": "test spec",
        },
        "max_samples": {
            "value": "auto",
            "label": "policy",
            "reason": "sample size",
            "derived_from": "test spec",
        },
        "contamination": {
            "value": "auto",
            "label": "policy",
            "reason": "contamination rate",
            "derived_from": "test spec",
        },
        "random_state": {
            "value": 1717,
            "label": "policy",
            "reason": "random seed",
            "derived_from": "test spec",
        },
        "allowed_training_splits": {
            "value": ["training"],
            "label": "policy",
            "reason": "splits",
            "derived_from": "test spec",
        },
    }


def _base_valid_cold_start_data() -> dict:
    """Helper returning valid cold start configuration dictionary."""
    return {
        "min_positive_labels_per_bank": {
            "value": 200,
            "label": "policy_placeholder",
            "reason": "Threshold count",
            "derived_from": "test spec",
        },
        "counted_verdict": {
            "value": "fraud",
            "label": "policy",
            "reason": "Verdict type",
            "derived_from": "test spec",
        },
    }


def test_load_committed_configs_and_input_feature_names() -> None:
    """Verifies committed config files load cleanly and input_feature_names holds exactly 39 features."""
    anomaly_cfg = load_anomaly_detector_config()
    cold_cfg = load_cold_start_config()

    # 1. Assert committed anomaly hyperparameters
    assert anomaly_cfg.early_window_days == 14
    assert anomaly_cfg.n_estimators == 200
    assert anomaly_cfg.max_samples == "auto"
    assert anomaly_cfg.contamination == "auto"
    assert anomaly_cfg.random_state == 1717
    assert anomaly_cfg.allowed_training_splits == ("training",)
    assert len(anomaly_cfg.excluded_features) == 6

    # 2. Assert committed cold-start values
    assert cold_cfg.min_positive_labels_per_bank == 200
    assert cold_cfg.counted_verdict == "fraud"

    # 3. Assert input_feature_names() count and exclusion
    feature_names = input_feature_names(anomaly_cfg)
    assert len(feature_names) == 39
    for excluded in anomaly_cfg.excluded_features:
        assert excluded not in feature_names

    # Check default no-arg invocation
    assert input_feature_names() == feature_names


def test_validation_missing_config_files_raise(tmp_path: Path) -> None:
    """Verifies FileNotFoundError on nonexistent configuration paths."""
    missing = tmp_path / "missing.yaml"
    with pytest.raises(FileNotFoundError):
        load_anomaly_detector_config(missing)
    with pytest.raises(FileNotFoundError):
        load_cold_start_config(missing)


def test_validation_excluded_feature_not_in_account_features_raises(tmp_path: Path) -> None:
    """Verifies ValueError when an excluded feature is not in canonical account_feature_names."""
    data = _base_valid_anomaly_data()
    data["excluded_features"]["value"]["invalid_feature_name_xyz"] = "Invalid"
    cfg_file = _write_yaml(tmp_path / "cfg.yaml", data)

    with pytest.raises(ValueError, match="is not in canonical account_feature_names"):
        load_anomaly_detector_config(cfg_file)


def test_validation_unknown_keys_raise(tmp_path: Path) -> None:
    """Verifies ValueError when unknown root keys exist in anomaly or cold-start YAML."""
    anomaly_data = _base_valid_anomaly_data()
    anomaly_data["unknown_extra_key"] = {"value": 1, "label": "policy", "derived_from": "test"}
    f_anom = _write_yaml(tmp_path / "anom.yaml", anomaly_data)

    with pytest.raises(ValueError, match="Unknown keys in .*unknown_extra_key"):
        load_anomaly_detector_config(f_anom)

    cold_data = _base_valid_cold_start_data()
    cold_data["unknown_cold_key"] = {"value": 1, "label": "policy", "derived_from": "test"}
    f_cold = _write_yaml(tmp_path / "cold.yaml", cold_data)

    with pytest.raises(ValueError, match="Unknown keys in .*unknown_cold_key"):
        load_cold_start_config(f_cold)


@pytest.mark.parametrize("missing_field", ["label", "derived_from"])
def test_validation_missing_metadata_raises(tmp_path: Path, missing_field: str) -> None:
    """Verifies ValueError when a configuration block is missing label or derived_from."""
    data = _base_valid_anomaly_data()
    del data["early_window_days"][missing_field]
    cfg_file = _write_yaml(tmp_path / "cfg.yaml", data)

    with pytest.raises(ValueError, match=f"missing or empty required field '{missing_field}'"):
        load_anomaly_detector_config(cfg_file)


@pytest.mark.parametrize("invalid_days", [0, -1, -14])
def test_validation_early_window_days_non_positive_raises(tmp_path: Path, invalid_days: int) -> None:
    """Verifies ValueError when early_window_days <= 0."""
    data = _base_valid_anomaly_data()
    data["early_window_days"]["value"] = invalid_days
    cfg_file = _write_yaml(tmp_path / "cfg.yaml", data)

    with pytest.raises(ValueError, match="early_window_days must be > 0"):
        load_anomaly_detector_config(cfg_file)


@pytest.mark.parametrize("forbidden_split", ["test", "stress"])
def test_validation_allowed_training_splits_forbidden_splits_raise(tmp_path: Path, forbidden_split: str) -> None:
    """Verifies ValueError when allowed_training_splits includes test or stress."""
    data = _base_valid_anomaly_data()
    data["allowed_training_splits"]["value"] = ["training", forbidden_split]
    cfg_file = _write_yaml(tmp_path / "cfg.yaml", data)

    with pytest.raises(ValueError, match=f"allowed_training_splits .* cannot contain '{forbidden_split}'"):
        load_anomaly_detector_config(cfg_file)

