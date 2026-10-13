"""
WHAT: Unit tests V1-V5 for device realism configuration parameters.
WHY: Ensures phone replacement cycles, upgrade uplifts, reinstalls, and OS adoption bands validate
     strictly (issues A1, R5, R2).
ISSUE IDs: A1 (episode and device realism), R5 (OS updates and device rules), R2 (background realism validation).
TERM: device realism: Parameter distributions governing hardware lifespans, app reinstall velocity,
      and OS rollout waves.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from core.config_validator import validate_file
from world.background_config import read_background_realism
from world.config_reader import draw_in_range, get_parameter_entry

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CONFIG_PATH = REPO_ROOT / "config" / "device_realism.yaml"
BG_CONFIG_PATH = REPO_ROOT / "config" / "background_realism.yaml"
FAILURE_CONFIG_PATH = REPO_ROOT / "config" / "failure_realism.yaml"

EXPECTED_KEYS = [
    "phone_replacement_months",
    "phone_change_festival_sale_uplift",
    "phone_change_festival_sale_days",
    "app_reinstall_per_person_per_year",
    "os_wave_count_per_world",
    "os_wave_length_days",
    "os_wave_final_share_of_eligible",
]


def test_v1_device_realism_validates() -> None:
    """
    WHAT: Asserts that config/device_realism.yaml passes validate_file with zero errors.
    WHY: Verifies the configuration satisfies all schema and nesting rules.
    """
    errors = validate_file(CONFIG_PATH)
    assert errors == []


def test_v2_keys_labels_and_ranges() -> None:
    """
    WHAT: Tests key presence, label partitioning (1 reported, 6 assumptions), and interval nesting.
    WHY: Enforces that only phone_replacement_months carries an empirical citation and all ranges nest.
    """
    data = read_background_realism(CONFIG_PATH)
    params = data.get("parameters", {})
    assert sorted(list(params.keys())) == sorted(EXPECTED_KEYS)

    for key in EXPECTED_KEYS:
        block = params[key]
        if key == "phone_replacement_months":
            assert block["label"] == "reported"
            src = block.get("source", {})
            assert src.get("url", "").startswith("http")
            assert "date" in src
        else:
            assert block["label"] == "assumption"
            assert "source" not in block

        tr = block["training"]
        te = block["test"]
        st = block["stress"]
        assert tr[0] < tr[1] and te[0] < te[1] and st[0] < st[1]
        assert st[0] <= te[0] <= tr[0] and tr[1] <= te[1] <= st[1]


def test_v3_negative_tampered_configs_rejected(tmp_path: Path) -> None:
    """
    WHAT: Asserts validate_file rejects configurations missing citations or reasons.
    WHY: Proves negative validation defenses block malformed or unsourced reported entries.
    """
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    # Missing source on reported entry.
    t1 = tmp_path / "no_source.yaml"
    raw1 = yaml.safe_load(yaml.dump(raw))
    raw1["parameters"]["phone_replacement_months"].pop("source")
    t1.write_text(yaml.dump(raw1), encoding="utf-8")
    assert len(validate_file(t1)) > 0

    # Missing reason on assumption entry.
    t2 = tmp_path / "no_reason.yaml"
    raw2 = yaml.safe_load(yaml.dump(raw))
    raw2["parameters"]["os_wave_length_days"].pop("reason")
    t2.write_text(yaml.dump(raw2), encoding="utf-8")
    assert len(validate_file(t2)) > 0


def test_v4_reader_integration_and_draw_bounds() -> None:
    """
    WHAT: Tests draw_in_range draws within bounds for all keys, plus refusals and regression keys.
    WHY: Guarantees existing config readers load device_realism cleanly without breaking existing configs.
    """
    data = read_background_realism(CONFIG_PATH)
    rng = np.random.default_rng(42)

    for key in EXPECTED_KEYS:
        entry = get_parameter_entry(data, key)
        val = draw_in_range(entry, rng, range_name="training", param_name=key)
        assert entry["training"][0] <= val <= entry["training"][1]

    with pytest.raises(ValueError, match="Unknown config parameter key"):
        get_parameter_entry(data, "non_existent_key")
    with pytest.raises(ValueError, match="Missing range split"):
        draw_in_range(get_parameter_entry(data, EXPECTED_KEYS[0]), rng, range_name="invalid")

    # Read-only checks on pre-existing configs.
    assert len(read_background_realism(BG_CONFIG_PATH)["parameters"]) == 17
    assert len(read_background_realism(FAILURE_CONFIG_PATH)["parameters"]) == 9


def _check_settings_range(settings: Any, data: dict[str, Any], range_name: str) -> None:
    """
    WHAT: Validates that drawn device settings fields fall strictly within configured boundaries.
    WHY: Verifies boundary confinement across training, test, and stress distributions.
    """
    p = data["parameters"]
    p_repl = p["phone_replacement_months"][range_name]
    p_sale = p["phone_change_festival_sale_uplift"][range_name]
    p_days = p["phone_change_festival_sale_days"][range_name]
    p_rein = p["app_reinstall_per_person_per_year"][range_name]
    assert p_repl[0] <= settings.replacement_months <= p_repl[1]
    assert p_sale[0] <= settings.sale_uplift <= p_sale[1]
    assert p_days[0] <= settings.sale_days <= p_days[1]
    assert p_rein[0] <= settings.reinstall_per_year <= p_rein[1]
    assert isinstance(settings.sale_days, int) and not isinstance(settings.sale_days, bool)


def test_v5_draw_device_settings() -> None:
    """
    WHAT: Tests draw_device_settings across 50 seeds, checking bounds and range refusals.
    WHY: Verifies correct random drawing and type checking for all device setting fields.
    """
    from world.device_events import draw_device_settings

    data = read_background_realism(CONFIG_PATH)
    for seed in range(50):
        rng = np.random.default_rng(seed)
        s_train = draw_device_settings(data, rng, range_name="training")
        _check_settings_range(s_train, data, "training")
        s_test = draw_device_settings(data, rng, range_name="test")
        _check_settings_range(s_test, data, "test")
        s_stress = draw_device_settings(data, rng, range_name="stress")
        _check_settings_range(s_stress, data, "stress")

    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="Unknown range name|Missing range split"):
        draw_device_settings(data, rng, range_name="unknown_split")
