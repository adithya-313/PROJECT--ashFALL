"""
WHAT: Unit tests for background realism configuration and its reader module.
WHY: Proves that background_realism.yaml is valid, all 17 assumption keys exist with nested ranges,
     and the reader integrates with draw_in_range while strictly validating missing or malformed inputs.
ISSUE IDs: R2 (background realism config).
TERM: background realism: Simulation parameters controlling temporal rhythms such as festivals and peak hours.
"""

from pathlib import Path

import numpy as np
import pytest
import yaml

from core.config_validator import validate_file
from world.background_config import EXPECTED_BACKGROUND_KEYS, draw_whole_number, read_background_realism
from world.config_reader import draw_in_range, get_parameter_entry

CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "background_realism.yaml"


def test_k1_background_realism_file_valid() -> None:
    """
    WHAT: Tests that config/background_realism.yaml passes validate_file with zero errors.
    WHY: Enforces that the new configuration adheres to the repository-wide labelled-block schema.
    """
    errors = validate_file(CONFIG_PATH)
    assert errors == [], f"Validation errors found in background_realism.yaml: {errors}"


def test_k2_all_seventeen_assumption_keys_exist() -> None:
    """
    WHAT: Asserts that all 17 expected keys exist, are labelled assumption, and have no source block.
    WHY: Guarantees full coverage of background rhythm parameters without unverified sources.
    """
    data = read_background_realism(CONFIG_PATH)
    params = data["parameters"]
    assert len(params) == 17
    for key in EXPECTED_BACKGROUND_KEYS:
        assert key in params, f"Missing parameter key: {key}"
        entry = params[key]
        assert entry["label"] == "assumption", f"Expected label 'assumption' for {key}"
        assert isinstance(entry.get("reason"), str) and len(entry["reason"].strip()) > 0
        assert "source" not in entry, f"Source block unexpectedly present in {key}"
        for split in ["training", "test", "stress"]:
            assert split in entry, f"Missing split {split} in {key}"


def test_k3_negative_missing_reason_and_bad_label(tmp_path: Path) -> None:
    """
    WHAT: Asserts that removing a reason or setting an unsourced label produces validation errors.
    WHY: Verifies that validator enforces schema rules on modified copies of background_realism.yaml.
    """
    raw_text = CONFIG_PATH.read_text(encoding="utf-8")

    # Corrupt copy 1: remove reason from first parameter
    corrupt_1 = yaml.safe_load(raw_text)
    first_key = next(iter(corrupt_1["parameters"]))
    del corrupt_1["parameters"][first_key]["reason"]
    p1 = tmp_path / "corrupt_1.yaml"
    p1.write_text(yaml.dump(corrupt_1), encoding="utf-8")
    assert len(validate_file(p1)) > 0, "Validator failed to flag missing reason"

    # Corrupt copy 2: set label to sourced without source dictionary
    corrupt_2 = yaml.safe_load(raw_text)
    corrupt_2["parameters"][first_key]["label"] = "sourced"
    p2 = tmp_path / "corrupt_2.yaml"
    p2.write_text(yaml.dump(corrupt_2), encoding="utf-8")
    assert len(validate_file(p2)) > 0, "Validator failed to flag missing source block on sourced label"


def test_k4_reader_integration_and_draw_bounds() -> None:
    """
    WHAT: Tests drawing values inside named bounds and verifying refusal of unknown keys or splits.
    WHY: Ensures background configuration entries can be drawn via draw_in_range and draw_whole_number.
    """
    data = read_background_realism(CONFIG_PATH)
    rng = np.random.default_rng(12345)

    entry = get_parameter_entry(data, "festival_uplift_fraction")
    draw_val = draw_in_range(entry, rng, range_name="training")
    assert entry["training"][0] <= draw_val <= entry["training"][1]

    # Test draw_whole_number helper on day bounds
    day_entry = get_parameter_entry(data, "festival_peak_day_in_world")
    whole_val = draw_whole_number(day_entry, rng, range_name="training")
    assert isinstance(whole_val, int)
    assert day_entry["training"][0] <= whole_val <= day_entry["training"][1]


def test_k4_refusal_on_missing_file_and_unknown_key() -> None:
    """
    WHAT: Tests that missing files, unknown keys, and unknown splits raise descriptive errors.
    WHY: Confirms robust fail-fast error handling on invalid config paths or parameter accesses.
    """
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        read_background_realism("non_existent_file.yaml")

    data = read_background_realism(CONFIG_PATH)
    with pytest.raises(ValueError, match="Unknown config parameter key: 'unknown_key'"):
        get_parameter_entry(data, "unknown_key")

    entry = get_parameter_entry(data, "festival_uplift_fraction")
    rng = np.random.default_rng(12345)
    with pytest.raises(ValueError, match="Missing range split 'invalid_split'"):
        draw_in_range(entry, rng, range_name="invalid_split")
