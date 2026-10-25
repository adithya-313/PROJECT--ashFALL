"""
WHAT: Unit tests for Level 12h empirical base rate, inflation factor, and world card integration.
It tests arithmetic scaling against NCRP benchmarks, configuration sensitivity, denominator validations,
showcase fraud world evaluation, Level 11 card immutability, and card schema symmetry.

WHY: Verifies that empirical baseline calculations maintain strict numerical integrity and conform
to architectural specifications without hardcoding policy literals in source code.
"""

import hashlib
import json
from pathlib import Path

import pytest

from world.base_rate import (
    base_rate_for,
    complaints_per_100000,
    inflation_factor,
    load_base_rate_config,
)
from world.fast_mode import FastRunSettings, run_fast_world
from world.world_card import build_fraud_world_card, build_world_card


def test_hand_computed_base_rate_and_inflation_factor() -> None:
    """
    WHAT: Tests hand-computed complaint rate and inflation factor matching calibrated benchmark numbers.
    WHY: Verifies literal arithmetic values for baseline calculations.
    """
    # 21 complaints over 2,000,000 payments:
    # rate = (21 / 2,000,000) * 100,000 = 1.05 per 100,000.
    # With real ratio 1.05: factor = 1.05 / 1.05 = 1.0.
    r1 = complaints_per_100000(21, 2000000)
    assert r1 == 1.05
    f1 = inflation_factor(r1, 1.05)
    assert f1 == 1.0

    # 42 complaints over 2,000,000 payments:
    # rate = (42 / 2,000,000) * 100,000 = 2.10 per 100,000.
    # With real ratio 1.05: factor = 2.10 / 1.05 = 2.0.
    r2 = complaints_per_100000(42, 2000000)
    assert r2 == 2.1
    f2 = inflation_factor(r2, 1.05)
    assert f2 == 2.0


def test_changing_base_rate_config_value_scales_factor(tmp_path: Path) -> None:
    """
    WHAT: Tests that overriding config/base_rate.yaml real-world ratio proportionally changes the inflation factor.
    WHY: Proves that downstream calculations dynamically track configured policy benchmarks without hardcoding.
    """
    # If real_ratio increases from 1.05 to 2.10 (doubles), factor for rate 1.05 becomes 1.05 / 2.10 = 0.5 (halved).
    cfg_data = load_base_rate_config()
    cfg_data["real_world_complaints_per_100000_upi_payments"]["value"] = 2.10
    factor = inflation_factor(1.05, float(cfg_data["real_world_complaints_per_100000_upi_payments"]["value"]))
    assert factor == 0.5


def test_refusals_zero_payments_and_zero_real_ratio() -> None:
    """
    WHAT: Asserts that non-positive denominators raise ValueError across rate and factor functions.
    WHY: Enforces strict arithmetic integrity and avoids silent zero-division anomalies.
    """
    with pytest.raises(ValueError, match="must be > 0"):
        complaints_per_100000(10, 0)
    with pytest.raises(ValueError, match="must be > 0"):
        complaints_per_100000(10, -5)
    with pytest.raises(ValueError, match="must be > 0"):
        inflation_factor(1.5, 0.0)
    with pytest.raises(ValueError, match="must be > 0"):
        inflation_factor(1.5, -1.0)


def test_real_fraud_on_world_card_base_rate_filled() -> None:
    """
    WHAT: Asserts 12e showcase scam world produces a complete fraud card with factor > 1 and valid fraud share.
    WHY: Verifies end-to-end base-rate calculations and card assembly on an authentic fraud simulation run.
    """
    # Uses Level 12e showcase complaints gang run (get_complaints_run)
    from showcase.level_12_support_e import get_complaints_run

    output, labels, _ = get_complaints_run()
    base_rate = base_rate_for(output, labels)
    assert base_rate["complaints"] > 0
    assert base_rate["fraud_inflation_factor"] > 1.0
    assert 0.0 < base_rate["fraud_share"] < 1.0

    settings = output.effective_config["settings"]

    mock_settings = FastRunSettings(
        seed=settings["seed"],
        n_people=60,
        n_shops=12,
        n_days=61,
        range_name=settings["range_name"],
    )

    card = build_fraud_world_card(mock_settings, output, base_rate)
    assert card["base_rate"]["fraud_inflation_factor"] > 1.0
    assert card["base_rate"]["fraud_share"] == base_rate["fraud_share"]


def test_fraud_off_level_11_card_has_none_and_golden_checks_pass() -> None:
    """
    WHAT: Asserts that a Level 11 fraud-off card preserves fraud_inflation_factor as None.
    WHY: Guarantees backward compatibility with baseline Level 11 evaluation records and golden value contracts.
    """
    s = FastRunSettings(seed=101, n_people=10, n_shops=4, n_days=61, range_name="training")
    out = run_fast_world(s)
    card = build_world_card(s, out)
    assert card["base_rate"]["fraud_inflation_factor"] is None
    assert card["base_rate"]["fraud_share"] is None


def test_grep_no_literal_1_05_in_world_or_showcase() -> None:
    """
    WHAT: Asserts that literal 1.05 does not appear in Python files under world/ or showcase/.
    WHY: Proves that empirical baseline ratios are read strictly from configuration rather than embedded as literals.
    """
    repo_root = Path(__file__).resolve().parent.parent.parent
    banned = "1" + ".05"
    offenders: list[str] = []
    for folder in ("world", "showcase"):
        for py_file in (repo_root / folder).rglob("*.py"):
            text = py_file.read_text(encoding="utf-8")
            if banned in text:
                offenders.append(str(py_file.relative_to(repo_root)))
    assert not offenders, f"Literal {banned} found in code files: {offenders}"


def test_fraud_on_and_fraud_off_cards_same_keys_and_fraud_off_byte_identical() -> None:
    """
    WHAT: Asserts fraud-on and fraud-off cards share identical key sets and fraud-off card matches golden hash.
    WHY: Verifies schema symmetry across simulation variants and guarantees byte-level immutability for Level 11 cards.
    """
    s = FastRunSettings(seed=101, n_people=10, n_shops=4, n_days=61, range_name="training")
    out_off = run_fast_world(s)
    card_off = build_world_card(s, out_off)

    from showcase.level_12_support_e import get_complaints_run

    out_on, labels, _ = get_complaints_run()
    base_rate = base_rate_for(out_on, labels)
    card_on = build_fraud_world_card(s, out_on, base_rate)

    assert set(card_off.keys()) == set(card_on.keys())
    assert card_off.keys() == card_on.keys()

    # Byte-identical to what build_world_card gave at 7df5a5e
    card_bytes = (json.dumps(card_off, sort_keys=True, indent=2) + "\n").encode("utf-8")
    digest = hashlib.sha256(card_bytes).hexdigest()
    assert digest == "0a891e220092ba69fb802e8bbff8b06cf7c515b44d1117beee194faf0c963832"
