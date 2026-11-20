"""
WHAT: Unit tests for the label trust curve, the trust switch and the tie tolerance in the mule
model config, and for the script that generates the curve.

WHY: A wrong curve silently changes every training weight, so its shape and its provenance guards
are pinned here.
"""

import hashlib
from pathlib import Path

import orjson
import pytest
import yaml

from decisions.mule_account_model import LABEL_TRUST_CURVE_LENGTH, load_mule_model_config
from scripts.build_label_trust_curve import pool_delays, trust_curve, verify_report_hash
from tests.unit.mule_model_config_fixture import write_mule_model_config

REPO_ROOT = Path(__file__).resolve().parents[2]
COMMITTED_CONFIG = REPO_ROOT / "config" / "mule_model.yaml"


def _config_with_block_value(tmp_path: Path, key: str, value: object) -> Path:
    """Returns a copy of the committed config whose given block carries the given value."""
    raw = yaml.safe_load(COMMITTED_CONFIG.read_text(encoding="utf-8"))
    raw["parameters"][key]["value"] = value
    path = tmp_path / f"mule_model_{key}.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def test_committed_config_carries_valid_trust_settings() -> None:
    config = load_mule_model_config(COMMITTED_CONFIG)

    assert len(config.label_trust_curve) == LABEL_TRUST_CURVE_LENGTH
    assert config.label_trust_weighting is False
    assert config.maturity_tie_tolerance == 0.002


def test_fixture_config_parses_with_trust_settings(tmp_path: Path) -> None:
    config = load_mule_model_config(write_mule_model_config(tmp_path / "mule_model.yaml"))

    assert config.label_trust_curve[0] == 0.0
    assert config.label_trust_curve[-1] == 1.0


def test_curve_with_wrong_length_is_refused(tmp_path: Path) -> None:
    path = _config_with_block_value(tmp_path, "label_trust_curve", [0.5] * 90)

    with pytest.raises(ValueError, match="must be a list of 91 numbers"):
        load_mule_model_config(path)


def test_curve_outside_unit_range_is_refused(tmp_path: Path) -> None:
    values = [0.5] * 90 + [1.5]
    path = _config_with_block_value(tmp_path, "label_trust_curve", values)

    with pytest.raises(ValueError, match="outside"):
        load_mule_model_config(path)


def test_decreasing_curve_is_refused(tmp_path: Path) -> None:
    values = [0.5] * 89 + [0.4, 0.5]
    path = _config_with_block_value(tmp_path, "label_trust_curve", values)

    with pytest.raises(ValueError, match="decreases at day 89"):
        load_mule_model_config(path)


def test_trust_switch_must_be_boolean(tmp_path: Path) -> None:
    path = _config_with_block_value(tmp_path, "label_trust_weighting", "yes")

    with pytest.raises(ValueError, match="must be a boolean"):
        load_mule_model_config(path)


@pytest.mark.parametrize("bad_value", [0, -0.1, float("inf"), "x"])
def test_tolerance_must_be_positive_and_finite(tmp_path: Path, bad_value: object) -> None:
    path = _config_with_block_value(tmp_path, "maturity_tie_tolerance", bad_value)

    with pytest.raises(ValueError, match="positive finite number"):
        load_mule_model_config(path)


def test_trust_curve_counts_negative_delays_as_day_zero() -> None:
    curve = trust_curve([-5.0, 0.5, 1.0, 3.0], max_days=3)

    # Delays clamped to 0, 0.5, 1.0, 3.0: day 0 holds one, day 1 holds three, day 3 holds all four.
    assert curve == [0.25, 0.75, 0.75, 1.0]


def test_trust_curve_refuses_no_delays() -> None:
    with pytest.raises(ValueError, match="No delays"):
        trust_curve([], max_days=3)


def test_report_hash_mismatch_is_refused(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    report_path.write_bytes(b"{}")

    with pytest.raises(ValueError, match="does not equal"):
        verify_report_hash(report_path, "0" * 64)
    verify_report_hash(report_path, hashlib.sha256(b"{}").hexdigest())


def test_pool_delays_refuses_wrong_world_count() -> None:
    report = {"by_world": [{"delays_days": [1.0]}], "pooled": {}}

    with pytest.raises(ValueError, match="expected 160"):
        pool_delays(report)


def test_committed_curve_matches_the_measured_report_when_present() -> None:
    report_path = REPO_ROOT / "runs" / "full" / "label_delay_r7.json"
    if not report_path.exists():
        pytest.skip("measured label delay report is not in this checkout")
    report = orjson.loads(report_path.read_bytes())

    committed = load_mule_model_config(COMMITTED_CONFIG).label_trust_curve

    assert list(committed) == trust_curve(pool_delays(report), max_days=LABEL_TRUST_CURVE_LENGTH - 1)
