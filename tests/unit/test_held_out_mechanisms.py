"""
WHAT: Unit tests for held-out mechanism validation and status verification (Level 12g C8, A-025).
Asserts that active held-out families have implementing block options, deferred families have none
and cite existing amendments, and CLI invocations correctly return formatted status lines.

WHY: Ground rule 17 and amendment A-025 require mechanism-level checks ensuring that active held-out
evasions are implemented while deferred families remain unbuilt and transparently documented.

TERM: held-out mechanism: a concrete parameter option in tactic_blocks.yaml implementing a held-out family.
TERM: deferred family: a pre-registered evasion postponed to later levels with explicit return conditions.
"""

import copy
import subprocess
import sys

import pytest

from world.tactic_config import load_tactic_blocks
from world.tactic_families import (
    check_held_out_mechanisms,
    load_evaluation_config,
    load_tactic_families,
)


def test_real_config_passes_and_returns_four_status_lines() -> None:
    """
    WHAT: Asserts that production configuration files pass mechanism verification and return 4 status lines.
    Verifies that slow_drip and device_rotation are ACTIVE, and fake_routine_life and
    pseudo_merchant_camouflage are DEFERRED.

    WHY: Amendment A-025 requires exactly two active held-out families and two deferred families.

    Example:
        >>> test_real_config_passes_and_returns_four_status_lines()
    """
    families = load_tactic_families()
    evaluation = load_evaluation_config()
    blocks = load_tactic_blocks()

    lines = check_held_out_mechanisms(families, evaluation, blocks)
    assert len(lines) == 4

    active_lines = [line for line in lines if "ACTIVE" in line]
    deferred_lines = [line for line in lines if "DEFERRED" in line]
    assert len(active_lines) == 2
    assert len(deferred_lines) == 2

    assert any(line.startswith("slow_drip: ACTIVE") for line in active_lines)
    assert any(line.startswith("device_rotation: ACTIVE") for line in active_lines)
    assert any(line.startswith("fake_routine_life: DEFERRED (A-025)") for line in deferred_lines)
    assert any(line.startswith("pseudo_merchant_camouflage: DEFERRED (A-025)") for line in deferred_lines)


def test_active_family_without_implementing_block_option_fails() -> None:
    """
    WHAT: Moving a deferred family (fake_routine_life) to the active tactics list without implementing options fails.
    Verifies that check_held_out_mechanisms raises ValueError with 'has no implementing block option'.

    WHY: Ground rule 17 and amendment A-025 require every active held-out family to have at least one block option.

    Example:
        >>> test_active_family_without_implementing_block_option_fails()
    """
    families = load_tactic_families()
    evaluation = copy.deepcopy(load_evaluation_config())
    blocks = load_tactic_blocks()

    # Move fake_routine_life from deferred to tactics
    eval_held = evaluation["held_out_tactics"]
    eval_held["tactics"].append("fake_routine_life")
    eval_held["deferred"] = [d for d in eval_held["deferred"] if d["name"] != "fake_routine_life"]

    err_msg = "held-out family 'fake_routine_life' is active but has no implementing block option"
    with pytest.raises(ValueError, match=err_msg):
        check_held_out_mechanisms(families, evaluation, blocks)


def test_deferred_item_citing_nonexistent_amendment_fails() -> None:
    """
    WHAT: Asserts that a deferred family citing a nonexistent amendment ID (A-999) raises ValueError.
    Verifies that citations are verified against docs/research/amendments.md.

    WHY: Ground rule 1 and amendment A-025 mandate that deferrals reference authentic, existing amendment records.

    Example:
        >>> test_deferred_item_citing_nonexistent_amendment_fails()
    """
    families = load_tactic_families()
    evaluation = copy.deepcopy(load_evaluation_config())
    blocks = load_tactic_blocks()

    # Point deferred amendment to A-999
    evaluation["held_out_tactics"]["deferred"][0]["amendment"] = "A-999"

    with pytest.raises(ValueError, match="cites amendment 'A-999' which does not appear"):
        check_held_out_mechanisms(families, evaluation, blocks)


def test_deferred_item_missing_returns_when_fails() -> None:
    """
    WHAT: Asserts that a deferred family record missing the returns_when field raises ValueError.
    Verifies that return conditions cannot be silently omitted.

    WHY: Amendment A-025 requires every deferred evasion to specify exact return conditions.

    Example:
        >>> test_deferred_item_missing_returns_when_fails()
    """
    families = load_tactic_families()
    evaluation = copy.deepcopy(load_evaluation_config())
    blocks = load_tactic_blocks()

    del evaluation["held_out_tactics"]["deferred"][0]["returns_when"]

    with pytest.raises(ValueError, match="missing required field 'returns_when'"):
        check_held_out_mechanisms(families, evaluation, blocks)


def test_block_option_naming_inactive_held_out_family_fails() -> None:
    """
    WHAT: Asserts that a block option in tactic_blocks.yaml specifying an inactive held-out family raises ValueError.
    Verifies that held_out_family markers can only name active families.

    WHY: A deferred or uncatalogued evasion must not carry active configuration mechanisms.

    Example:
        >>> test_block_option_naming_inactive_held_out_family_fails()
    """
    families = load_tactic_families()
    evaluation = load_evaluation_config()
    blocks = copy.deepcopy(load_tactic_blocks())

    # Point route.drip_piece_max_inr to fake_routine_life (which is deferred, not active)
    route_params = blocks["tactic_blocks"]["blocks"]["route"]["parameters"]
    route_params["drip_piece_max_inr"]["held_out_family"] = "fake_routine_life"

    err_msg = (
        "held_out_family marker on 'route.drip_piece_max_inr' "
        "names family 'fake_routine_life' which is not active"
    )
    with pytest.raises(ValueError, match=err_msg):
        check_held_out_mechanisms(families, evaluation, blocks)


def test_cli_execution_exits_zero_on_real_config() -> None:
    """
    WHAT: Executes `py -3.12 -m world.tactic_families` via subprocess and asserts returncode 0 and status lines.

    WHY: Amendment A-025 requires the module to be directly runnable as a verification script.

    Example:
        >>> test_cli_execution_exits_zero_on_real_config()
    """
    result = subprocess.run(
        [sys.executable, "-m", "world.tactic_families"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "device_rotation: ACTIVE" in result.stdout
    assert "slow_drip: ACTIVE" in result.stdout
    assert "fake_routine_life: DEFERRED" in result.stdout
    assert "pseudo_merchant_camouflage: DEFERRED" in result.stdout
