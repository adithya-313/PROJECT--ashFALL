"""
WHAT: Unit tests for the label-network feature switch on the Level 18 account model.

WHY: The model may only add catalogue features built from labels that have already arrived, and the
switch-off contract must stay byte-for-byte the previous 39-feature list so a pilot becomes
comparable only when the switch is made on deliberately.
"""

from pathlib import Path

import pytest
import yaml

from decisions.anomaly_config import load_anomaly_detector_config
from decisions.mule_account_model import (
    account_model_feature_names,
    load_mule_model_config,
)
from features.account_snapshots import account_feature_names

REPO_ROOT = Path(__file__).resolve().parents[2]
MULE_MODEL_CONFIG = REPO_ROOT / "config" / "mule_model.yaml"

LABEL_FEATURES = (
    "hops_to_complained_account",
    "registry_match_count",
    "registry_match_age_days",
)
FEEDBACK_FEATURES = (
    "hops_to_confirmed_label",
    "shared_counterparties_with_flagged_7d",
)


def _config_body(
    label_features: tuple[str, ...] = LABEL_FEATURES,
    feedback_features: tuple[str, ...] = FEEDBACK_FEATURES,
) -> str:
    """Returns a complete mule model config body listing the given network feature names."""
    label_lines = "".join(f"      - {name}\n" for name in label_features)
    feedback_lines = "".join(f"      - {name}\n" for name in feedback_features)
    return (
        "parameters:\n"
        "  label_cut:\n"
        "    label: policy\n"
        "    value: arrival_time\n"
        "    derived_from: test\n"
        "    reason: test\n"
        "  label_maturity_days:\n"
        "    label: policy\n"
        "    value: null\n"
        "    not_found: true\n"
        "    derived_from: test\n"
        "    reason: test\n"
        "  feature_fill_flag_share:\n"
        "    label: policy\n"
        "    value: 0.999\n"
        "    derived_from: test\n"
        "    reason: test\n"
        "  use_label_network_features:\n"
        "    label: policy\n"
        "    value: true\n"
        "    derived_from: test\n"
        "    reason: test\n"
        "  label_network_features:\n"
        "    label: policy\n"
        "    value:\n"
        f"{label_lines}"
        "    derived_from: test\n"
        "    reason: test\n"
        "  feedback_network_features_not_used:\n"
        "    label: policy\n"
        "    value:\n"
        f"{feedback_lines}"
        "    derived_from: test\n"
        "    reason: test\n"
    )


def _catalogue_minus(exclusions: set[str], admitted: set[str]) -> tuple[str, ...]:
    """Returns canonical names outside the exclusions except those the switch admits."""
    return tuple(name for name in account_feature_names() if name not in exclusions or name in admitted)


def test_switch_off_keeps_the_39_feature_contract() -> None:
    """With the switch off the model uses exactly the previous canonical-minus-exclusions list."""
    excluded = set(load_anomaly_detector_config().excluded_features)
    expected = tuple(name for name in account_feature_names() if name not in excluded)

    names = account_model_feature_names(use_label_network_features=False)

    assert len(names) == 39
    assert names == expected


def test_switch_on_adds_the_three_features_in_catalogue_order() -> None:
    """With the switch on the three label features enter at their catalogue positions."""
    excluded = set(load_anomaly_detector_config().excluded_features)
    expected = _catalogue_minus(excluded, set(LABEL_FEATURES))

    names = account_model_feature_names(use_label_network_features=True)

    assert len(names) == 42
    assert names == expected
    assert list(names[-4:]) == [
        "local_cluster_density",
        "hops_to_complained_account",
        "registry_match_count",
        "registry_match_age_days",
    ]


def test_switch_on_exposes_three_and_hides_feedback_and_account_type() -> None:
    """The switch admits the three label features and neither feedback feature nor account_type."""
    names = account_model_feature_names(use_label_network_features=True)

    for feature in LABEL_FEATURES:
        assert feature in names
    for feature in (*FEEDBACK_FEATURES, "account_type"):
        assert feature not in names


def test_unknown_feature_name_is_refused(tmp_path: Path) -> None:
    """A listed name missing from the canonical catalogue is refused and named."""
    config_path = tmp_path / "mule_model.yaml"
    config_path.write_text(_config_body(label_features=(*LABEL_FEATURES, "not_a_feature")), encoding="utf-8")

    with pytest.raises(ValueError, match="'not_a_feature' .* not in the canonical account feature catalogue"):
        load_mule_model_config(config_path)


def test_feature_outside_the_anomaly_exclusions_is_refused(tmp_path: Path) -> None:
    """A canonical feature the anomaly detector still uses is refused and named."""
    config_path = tmp_path / "mule_model.yaml"
    config_path.write_text(_config_body(label_features=(*LABEL_FEATURES, "local_cluster_density")), encoding="utf-8")

    with pytest.raises(ValueError, match="'local_cluster_density' .* not in the anomaly exclusion list"):
        load_mule_model_config(config_path)


def test_rules_plus_labels_and_registry_parses_from_the_committed_config() -> None:
    """The committed config carries the rule_6 switch and it is off."""
    config = load_mule_model_config(MULE_MODEL_CONFIG)

    assert config.rules_plus_labels_and_registry is False


def test_rules_plus_labels_and_registry_rejects_a_non_boolean(tmp_path: Path) -> None:
    """A non-boolean value for the rule_6 switch is refused and the key is named."""
    raw = yaml.safe_load(MULE_MODEL_CONFIG.read_text(encoding="utf-8"))
    raw["parameters"]["rules_plus_labels_and_registry"]["value"] = "yes"
    config_path = tmp_path / "mule_model_bad_rules_switch.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="rules_plus_labels_and_registry .* must be a boolean"):
        load_mule_model_config(config_path)
