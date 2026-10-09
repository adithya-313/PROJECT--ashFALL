"""Checks that config/evaluation.yaml keeps its pre-registered shape and values.

The rules in that file are committed before any new-engine model exists, so this
test fails loudly if a label, reason or key number is removed or changed by accident.
"""
import copy
from pathlib import Path

import pytest
import yaml

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "evaluation.yaml"

ALLOWED_LABELS = {"assumption", "policy"}

# The decision blocks every one of which must carry a label and a reason.
DECISION_BLOCKS = ["capacity", "fair_comparison", "ml_beats_rules", "latency", "held_out_tactics"]

EXPECTED_LABELS = {
    "capacity": "assumption",
    "latency": "assumption",
    "fair_comparison": "policy",
    "ml_beats_rules": "policy",
    "held_out_tactics": "policy",
}


def load_config():
    """Read the YAML file into a plain dictionary."""
    with open(CONFIG_PATH, encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def check_config(config):
    """Raise ValueError with a clear message if the config is not well formed."""
    for block_name in DECISION_BLOCKS:
        block = config[block_name]

        # Every decision must say whether it is an assumption or a policy.
        if block.get("label") not in ALLOWED_LABELS:
            raise ValueError(f"{block_name}: label must be one of {sorted(ALLOWED_LABELS)}, got {block.get('label')!r}")

        # Every decision must explain itself in words.
        reason = block.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"{block_name}: reason must be a non-empty string")

    # The headline capacity must sit inside the swept range.
    capacity = config["capacity"]
    low = capacity["sweep_min_pct_active_accounts_per_day"]
    high = capacity["sweep_max_pct_active_accounts_per_day"]
    headline = capacity["headline_pct_active_accounts_per_day"]
    if not low <= headline <= high:
        raise ValueError(f"capacity: headline {headline} is outside the sweep range {low} to {high}")


def test_file_loads_and_is_pre_registered():
    config = load_config()
    assert config["meta"]["status"] == "pre-registered"


def test_labels_and_reasons_are_present():
    # Passes quietly when every block is well formed.
    check_config(load_config())


def test_labels_are_the_expected_ones():
    config = load_config()
    for block_name, expected_label in EXPECTED_LABELS.items():
        assert config[block_name]["label"] == expected_label, block_name


def test_key_values_are_unchanged():
    config = load_config()
    capacity = config["capacity"]
    assert capacity["headline_pct_active_accounts_per_day"] == 0.25
    assert capacity["sweep_min_pct_active_accounts_per_day"] == 0.05
    assert capacity["sweep_max_pct_active_accounts_per_day"] == 1.0
    assert config["fair_comparison"]["main_genuine_restriction_rate_pct"] == 0.5
    assert config["ml_beats_rules"]["min_median_recall_difference_points"] == 5
    assert config["ml_beats_rules"]["min_share_of_test_worlds_ml_wins_pct"] == 70
    assert config["ml_beats_rules"]["confidence_level_pct"] == 95
    assert config["latency"]["max_ms"] == 100
    assert config["latency"]["feature_computation_max_ms"] == 50


def test_held_out_tactics():
    held_out = load_config()["held_out_tactics"]
    assert len(held_out["tactics"]) == 4
    assert "basic_merchant_mule" not in held_out["tactics"]


def test_missing_reason_is_rejected():
    config = copy.deepcopy(load_config())
    del config["latency"]["reason"]
    with pytest.raises(ValueError, match="latency: reason must be a non-empty string"):
        check_config(config)


def test_bad_label_is_rejected():
    config = copy.deepcopy(load_config())
    config["capacity"]["label"] = "guess"
    with pytest.raises(ValueError, match="capacity: label must be one of"):
        check_config(config)


def test_headline_outside_sweep_is_rejected():
    config = copy.deepcopy(load_config())
    config["capacity"]["headline_pct_active_accounts_per_day"] = 2.0
    with pytest.raises(ValueError, match="outside the sweep range"):
        check_config(config)
