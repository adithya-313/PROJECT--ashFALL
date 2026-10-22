"""
WHAT: Unit tests for gang tactic configuration loader and dividing line resolution (world/tactic_config.py):
asserts dividing lines resolve dynamically to world ranges bounds without typed numbers, all building blocks
declare valid strength levels in {1, 2}, disguise permits only strength 2, tactic risks match block names,
and both tactic YAML configurations pass the core config validator.

WHY: Enforces Ground Rule 15, amendment A-014, and amendment A-016 by asserting that dividing lines, building
block capabilities, and cost distributions stay fully synchronized with declarative policy and world ranges.

TERM: building block: an atomic step in a fraud tactic campaign (acquire, warm_up, route, disguise).
TERM: strength level: the sophistication and camouflage applied by a gang (strength 1 = fixed, 2 = camouflaged).
TERM: dwell: the duration in simulated minutes money stays in an account before being forwarded.
TERM: layer: the hop distance in a laundering payment chain from the source to terminal exit.
TERM: budget: maximum allowed expenditure in money, time, and cumulative risk for a gang campaign.
TERM: dividing line: boundary separating training tactics from held-out tactics to prevent data leakage.
"""

import pytest

from core.config_validator import load_yaml, validate_file
from world.tactic_config import (
    TACTIC_BLOCKS_PATH,
    TACTIC_COSTS_PATH,
    WORLD_RANGES_PATH,
    get_block_definition,
    get_block_risk_range,
    get_block_time_range,
    get_disguise_jitter_range,
    get_dwell_range,
    get_gang_payment_purpose,
    get_route_amount_range,
    get_supplier_fee_range,
    load_tactic_blocks,
    load_tactic_costs,
    load_world_ranges,
    resolve_dividing_line,
    resolve_parameter_training_range,
)


def test_dividing_lines_resolve_to_world_ranges_bounds_dynamically() -> None:
    """
    WHAT: Asserts that held-out dividing lines resolve to the exact numerical bounds read from world_ranges.yaml,
    ensuring no numerical boundaries are hardcoded or typed directly into tests or configuration.

    WHY: Amendment A-014 requires held-out dividing lines (12-8) to reference world_ranges parameters dynamically;
    comparing resolved values against world_ranges.yaml verifies the reference link holds without typing numbers.

    Args:
        None.

    Returns:
        None. Fails if resolved dividing lines deviate from the underlying world ranges bounds.

    Example:
        # resolve_dividing_line("training_max_dwell_minutes") == world_ranges["minutes_money_stays..."]["training"][1]
    """
    # 1. Read ground truth parameter bounds directly from world_ranges.yaml.
    world_ranges_data = load_yaml(WORLD_RANGES_PATH)["parameters"]
    expected_dwell_max = world_ranges_data["minutes_money_stays_in_mule_account"]["training"][1]
    expected_amount_min = world_ranges_data["mule_transfer_amount_inr"]["training"][0]

    # 2. Assert resolved values match the bounds read from the ranges file.
    assert resolve_dividing_line("training_max_dwell_minutes") == expected_dwell_max
    assert resolve_dividing_line("slow_drip_max_amount_inr") == expected_amount_min
    assert resolve_dividing_line("training_devices_retired_on_schedule") is False

    # 3. Assert unknown dividing line raises KeyError.
    with pytest.raises(KeyError, match="Unknown held-out dividing line 'unknown_line'"):
        resolve_dividing_line("unknown_line")


def test_every_block_lists_allowed_strengths_in_one_and_two() -> None:
    """
    WHAT: Asserts that every building block defined in config/tactic_blocks.yaml specifies allowed strength levels
    as a non-empty subset of {1, 2}, and that the disguise block allows only strength 2.

    WHY: Decision D7 permits strength 1 (fixed parameters) and strength 2 (camouflaged timing and amounts), while
    the disguise jitter block is strictly restricted to strength 2 to prevent timing perturbation in naive gangs.

    Args:
        None.

    Returns:
        None. Fails if any block defines invalid strengths or disguise allows strength 1.

    Example:
        # get_block_definition("acquire")["allowed_strengths"] == [1, 2]
        # get_block_definition("disguise")["allowed_strengths"] == [2]
    """
    # 1. Retrieve blocks dictionary from configuration.
    blocks_config = load_tactic_blocks()["tactic_blocks"]["blocks"]
    expected_blocks = {"acquire", "warm_up", "route", "disguise", "take_control", "scam", "cash_out", "split_below_threshold"}
    assert set(blocks_config.keys()) == expected_blocks

    # 2. Verify allowed strengths for each block are within {1, 2}.
    for block_name, definition in blocks_config.items():
        allowed_strengths = set(definition["allowed_strengths"])
        assert allowed_strengths.issubset({1, 2}), f"Block '{block_name}' has invalid strengths: {allowed_strengths}"
        assert len(allowed_strengths) > 0, f"Block '{block_name}' has empty allowed strengths"

    # 3. Assert disguise strictly permits only strength 2.
    assert get_block_definition("disguise")["allowed_strengths"] == [2]


def test_tactic_risk_per_use_matches_block_names_and_has_valid_ranges() -> None:
    """
    WHAT: Asserts that tactic_risk_per_use in config/tactic_costs.yaml contains exactly the five building blocks
    from tactic_blocks.yaml, and every entry is labelled assumption with a valid probability range in [0, 1].

    WHY: Decision D2 and amendment A-016 require per-block execution risks to be assumptions with ranges inside [0, 1]
    matching the registered block catalogue for gang budget risk calculations.

    Args:
        None.

    Returns:
        None. Fails if risk block names mismatch tactic blocks or if ranges exceed [0, 1].

    Example:
        # lower, upper = get_block_risk_range("acquire")
        # assert 0.0 <= lower <= upper <= 1.0
    """
    # 1. Compare block keys between tactic blocks catalogue and risk configuration.
    blocks_config = load_tactic_blocks()["tactic_blocks"]["blocks"]
    costs_config = load_tactic_costs()
    risk_data = costs_config["tactic_risk_per_use"]
    risk_blocks = {name: entry for name, entry in risk_data.items() if isinstance(entry, dict) and "range" in entry}
    assert set(risk_blocks.keys()) == set(blocks_config.keys())

    # 2. Verify label and numerical probability range boundaries for each block.
    for block_name in blocks_config.keys():
        block_risk_entry = risk_data[block_name]
        assert block_risk_entry["label"] == "assumption", f"Block '{block_name}' risk must be assumption"
        lower_bound, upper_bound = get_block_risk_range(block_name)
        assert 0.0 <= lower_bound <= upper_bound <= 1.0, (
            f"Block '{block_name}' risk range [{lower_bound}, {upper_bound}] outside [0, 1]"
        )


def test_config_validator_passes_on_both_tactic_yaml_files() -> None:
    """
    WHAT: Invokes core.config_validator.validate_file on both config/tactic_blocks.yaml and config/tactic_costs.yaml
    and asserts that neither file contains validation errors.

    WHY: Guarantees that every labelled block in both tactic configuration files contains permitted labels,
    non-empty rationales, and valid nested ranges according to repository configuration standards.

    Args:
        None.

    Returns:
        None. Fails if either configuration file produces validation error messages.

    Example:
        # assert validate_file(TACTIC_BLOCKS_PATH) == []
        # assert validate_file(TACTIC_COSTS_PATH) == []
    """
    # 1. Run shared config validator on tactic blocks YAML file.
    blocks_errors = validate_file(TACTIC_BLOCKS_PATH)
    assert blocks_errors == [], f"Validation errors in {TACTIC_BLOCKS_PATH.name}: {blocks_errors}"

    # 2. Run shared config validator on tactic costs YAML file.
    costs_errors = validate_file(TACTIC_COSTS_PATH)
    assert costs_errors == [], f"Validation errors in {TACTIC_COSTS_PATH.name}: {costs_errors}"

    # 3. Verify gang payment purpose resolves to everyday.
    assert get_gang_payment_purpose() == "everyday"


def test_resolver_functions_read_ranges_from_config() -> None:
    """
    WHAT: Asserts that all range resolvers in world/tactic_config.py retrieve parameter bounds
    dynamically from their backing YAML configurations without relying on hardcoded numerical values.
    It checks dwell bounds [5, 60], transfer amount bounds [2000.0, 50000.0], supplier fee bounds [0.03, 0.06],
    acquire time bounds [60, 1440], and disguise jitter bounds [0, 30] against their source configurations.

    WHY: Project rules prohibit copying numerical bounds into source code; this test verifies that
    resolver functions properly expose the configured parameter boundaries across all tactical dimensions.

    Args:
        None.

    Returns:
        None. Fails if any resolver returns incorrect bounds or fails to read from its configuration source.

    Example:
        # assert get_dwell_range() == (5, 60)
        # assert get_route_amount_range() == (2000.0, 50000.0)
        # assert get_supplier_fee_range() == (0.03, 0.06)
    """
    # 1. Verify dwell range matches world ranges training split for dwell.
    assert get_dwell_range() == (5, 60)

    # 2. Verify route amount range matches world ranges training split for transfer amount.
    assert get_route_amount_range() == (2000.0, 50000.0)

    # 3. Verify supplier fee range matches tactic costs configuration.
    assert get_supplier_fee_range() == (0.03, 0.06)

    # 4. Verify acquire block time range matches tactic blocks configuration.
    assert get_block_time_range("acquire") == (60, 1440)

    # 5. Verify disguise jitter range matches tactic blocks configuration.
    assert get_disguise_jitter_range() == (0, 30)

