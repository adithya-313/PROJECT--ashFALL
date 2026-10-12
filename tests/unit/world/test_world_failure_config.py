"""
WHAT: Unit tests for failure realism configuration and provenance validation.
WHY: Ensures that simulated failure parameters adhere to empirical sources, strict assumptions,
     and the shared config validator schema (issue R7).
ISSUE IDs: R7 (failure domain and outage tracking).
TERM: failure realism: Configuration parameters defining rates and durations of payment outages and drops.
"""

from pathlib import Path

import numpy as np
import pytest
import yaml

from core.config_validator import load_yaml, validate_file
from world.background_config import read_background_realism
from world.config_reader import draw_in_range, get_parameter_entry

EXPECTED_FAILURE_KEYS = (
    "npci_incident_count_per_30_days",
    "npci_incident_duration_minutes",
    "npci_incident_success_rate",
    "bank_downtime_count_per_bank_per_30_days",
    "bank_downtime_duration_minutes",
    "bank_downtime_success_rate",
    "technical_decline_rate",
    "customer_wrong_pin_rate",
    "app_network_failure_rate",
)

REPORTED_KEYS = (
    "bank_downtime_duration_minutes",
    "technical_decline_rate",
)

ASSUMPTION_KEYS = (
    "npci_incident_count_per_30_days",
    "npci_incident_duration_minutes",
    "npci_incident_success_rate",
    "bank_downtime_count_per_bank_per_30_days",
    "bank_downtime_success_rate",
    "customer_wrong_pin_rate",
    "app_network_failure_rate",
)


def test_y1_failure_realism_validates() -> None:
    """
    WHAT: Validates config/failure_realism.yaml using the shared config validator.
    WHY: Enforces that all 9 failure parameter entries follow label and range rules.
    """
    config_path = Path("config/failure_realism.yaml")
    # Verify zero errors returned by validator.
    errors = validate_file(config_path)
    assert errors == []


def test_y2_keys_labels_and_ranges() -> None:
    """
    WHAT: Checks the exact 9 keys, their labels, source metadata, and range nesting.
    WHY: Verifies empirical provenance (2 reported, 7 assumptions) and containment.
    """
    data = load_yaml("config/failure_realism.yaml")
    params = data.get("parameters", {})
    assert set(params.keys()) == set(EXPECTED_FAILURE_KEYS)
    for key in REPORTED_KEYS:
        entry = params[key]
        assert entry["label"] == "reported"
        assert entry["source"]["url"].startswith("http")
        assert entry["source"]["date"] != ""
    for key in ASSUMPTION_KEYS:
        entry = params[key]
        assert entry["label"] == "assumption"
        assert "source" not in entry
    for entry in params.values():
        tr, te, st = entry["training"], entry["test"], entry["stress"]
        assert tr[0] < tr[1] and te[0] <= tr[0] and tr[1] <= te[1]
        assert te[0] < te[1] and st[0] <= te[0] and te[1] <= st[1]


def test_y3_negative_tampered_configs_rejected(tmp_path: Path) -> None:
    """
    WHAT: Tests that removing a reported entry source or reason triggers validator errors.
    WHY: Proves the config validator actively defends against unverified failure numbers.
    """
    data = load_yaml("config/failure_realism.yaml")
    # Tamper 1: remove source from reported entry.
    t1_data = dict(data)
    t1_data["parameters"] = dict(data["parameters"])
    rep_key = "bank_downtime_duration_minutes"
    t1_entry = dict(t1_data["parameters"][rep_key])
    del t1_entry["source"]
    t1_data["parameters"][rep_key] = t1_entry
    t1_path = tmp_path / "missing_source.yaml"
    with open(t1_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(t1_data, f)
    assert len(validate_file(t1_path)) > 0

    # Tamper 2: remove reason from an assumption entry.
    t2_data = dict(data)
    t2_data["parameters"] = dict(data["parameters"])
    assump_key = "customer_wrong_pin_rate"
    t2_entry = dict(t2_data["parameters"][assump_key])
    del t2_entry["reason"]
    t2_data["parameters"][assump_key] = t2_entry
    t2_path = tmp_path / "missing_reason.yaml"
    with open(t2_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(t2_data, f)
    assert len(validate_file(t2_path)) > 0


def test_y4_reader_integration_and_draw_bounds() -> None:
    """
    WHAT: Loads failure config via read_background_realism and draws within bounds.
    WHY: Guarantees interoperability with existing world drawing functions without file edits.
    """
    data = read_background_realism("config/failure_realism.yaml")
    rng = np.random.default_rng(12345)
    for key in EXPECTED_FAILURE_KEYS:
        entry = get_parameter_entry(data, key)
        val = draw_in_range(entry, rng, range_name="training")
        bounds = entry["training"]
        assert bounds[0] <= val <= bounds[1]

    # Test unknown key rejection.
    with pytest.raises(ValueError, match="non_existent_key"):
        get_parameter_entry(data, "non_existent_key")

    # Test unknown range name rejection.
    with pytest.raises(ValueError, match="unknown_range"):
        draw_in_range(get_parameter_entry(data, "customer_wrong_pin_rate"), rng, range_name="unknown_range")

    # Read-only check: background_realism.yaml still contains exactly 17 keys.
    bg_data = read_background_realism()
    assert len(bg_data["parameters"]) == 17
