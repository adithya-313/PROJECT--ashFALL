"""
WHAT: Unit tests for gang tactical configuration validation and plain serialization (world/gang_config.py):
asserts that one passing and one failing example per validation rule are verified (unknown blocks, invalid strength
capabilities, illegal step ordering, dividing line violations for dwell and amount, invalid layers, and unknown params).

WHY: Fraud tactics must adhere strictly to policy catalogues, strength capabilities, and held-out dividing lines.
Testing each rule with concrete passing and failing cases guarantees that invalid campaigns are rejected cleanly.

TERM: gang configuration: specification of a criminal gang including budget, source account, strength, and tactic steps.
TERM: gang budget: three-dimensional spending cap in money (paise), simulated time (minutes), and cumulative risk.
TERM: fixture source: initial collector account pre-funded with opening money from which fraud routing originates.
TERM: block step: single parametrized tactic building block in a gang's sequential action plan.
TERM: strength level: the sophistication and camouflage applied by a gang (strength 1 = fixed, 2 = camouflaged).
TERM: dividing line: boundary separating training tactics from held-out tactics to prevent data leakage.
"""

import pytest
import yaml

from core.config_validator import load_yaml
from world.gang_config import (
    BlockStep,
    FixtureSource,
    GangBudget,
    GangConfig,
    gang_config_to_plain,
    validate_gang_config,
)
import world.tactic_config as tactic_config
from world.tactic_config import TACTIC_BLOCKS_PATH, resolve_dividing_line

DEFAULT_BUDGET = GangBudget(money_paise=1_000_000, time_minutes=5000, max_risk=0.1)
DEFAULT_SOURCE = FixtureSource("acc_collector_01", 0, 10_000_000, 100)


def test_unknown_block_is_refused_or_passed() -> None:
    """
    WHAT: Asserts that a gang configuration with valid blocks passes validation, while a configuration specifying
    an unrecognized block name raises ValueError naming the unknown block.

    WHY: Syndicates can only execute registered building blocks; unrecognized block identifiers indicate typing
    mistakes or unsupported behaviors that must be caught before simulation planning.

    Args:
        None.

    Returns:
        None. Fails if valid blocks are rejected or unknown blocks are accepted.

    Example:
        # validate_gang_config(valid_config) passes cleanly.
        # validate_gang_config(config_with_teleport) raises ValueError.
    """
    # 1. Validate that standard acquire and route steps pass cleanly.
    valid_steps = (BlockStep("acquire", (("count", 5),)), BlockStep("route", (("layers", 2),)))
    valid_config = GangConfig("gang_valid", 1, DEFAULT_BUDGET, DEFAULT_SOURCE, valid_steps)
    validate_gang_config(valid_config)

    # 2. Validate that an unknown block raises ValueError.
    invalid_steps = (BlockStep("teleport", (("speed", 10),)),)
    invalid_config = GangConfig("gang_invalid", 1, DEFAULT_BUDGET, DEFAULT_SOURCE, invalid_steps)
    with pytest.raises(ValueError, match="Gang 'gang_invalid' uses unknown block 'teleport'."):
        validate_gang_config(invalid_config)


def test_strength_not_allowed_for_block_is_refused_or_passed() -> None:
    """
    WHAT: Asserts that the disguise block passes validation at strength 2, but raises ValueError when configured
    for a gang operating at strength 1.

    WHY: Decision D7 permits disguise timing jitter exclusively at strength 2 (camouflage); strength 1 gangs
    are naive and must use fixed uncamouflaged intervals without perturbation.

    Args:
        None.

    Returns:
        None. Fails if disguise is permitted at strength 1 or rejected at strength 2.

    Example:
        # validate_gang_config(strength_2_with_disguise) passes.
        # validate_gang_config(strength_1_with_disguise) raises ValueError.
    """
    # 1. Validate that disguise passes at strength 2 after acquiring accounts.
    steps = (BlockStep("acquire", (("count", 2),)), BlockStep("disguise", (("jitter_minutes", 10),)))
    valid_config = GangConfig("gang_str2", 2, DEFAULT_BUDGET, DEFAULT_SOURCE, steps)
    validate_gang_config(valid_config)

    # 2. Validate that disguise is refused at strength 1.
    invalid_config = GangConfig("gang_str1", 1, DEFAULT_BUDGET, DEFAULT_SOURCE, steps)
    with pytest.raises(ValueError, match="Gang 'gang_str1' strength 1 is not allowed for block 'disguise'"):
        validate_gang_config(invalid_config)


def test_step_order_rules_are_enforced_or_passed() -> None:
    """
    WHAT: Asserts that procedural step ordering rules (acquire before route, route needs acquire, warm_up after acquire)
    pass when respected and raise ValueError when violated.

    WHY: Logical causality requires mule accounts to exist before money can be routed through them or accounts
    can be warmed up; out-of-order execution represents an incoherent tactical campaign.

    Args:
        None.

    Returns:
        None. Fails if improper step sequences pass or valid sequences fail.

    Example:
        # acquire -> warm_up -> route passes cleanly.
        # route before acquire raises ValueError.
    """
    # 1. Validate that acquire -> warm_up -> route sequence passes cleanly.
    valid_steps = (
        BlockStep("acquire", (("count", 3),)),
        BlockStep("warm_up", (("days", 3),)),
        BlockStep("route", (("layers", 1),)),
    )
    validate_gang_config(GangConfig("gang_order", 1, DEFAULT_BUDGET, DEFAULT_SOURCE, valid_steps))

    # 2. Validate that route before acquire is rejected.
    reversed_steps = (BlockStep("route", (("layers", 1),)), BlockStep("acquire", (("count", 3),)))
    with pytest.raises(ValueError, match="Gang 'gang_bad_route' must acquire accounts before routing money."):
        validate_gang_config(GangConfig("gang_bad_route", 1, DEFAULT_BUDGET, DEFAULT_SOURCE, reversed_steps))

    # 3. Validate that warm_up before acquire is rejected.
    early_warmup = (BlockStep("warm_up", (("days", 3),)), BlockStep("acquire", (("count", 3),)))
    with pytest.raises(ValueError, match="Gang 'gang_bad_warm' cannot warm up before acquiring accounts."):
        validate_gang_config(GangConfig("gang_bad_warm", 1, DEFAULT_BUDGET, DEFAULT_SOURCE, early_warmup))


def test_route_dividing_line_dwell_is_enforced_or_passed() -> None:
    """
    WHAT: Asserts that route dwell times within training_max_dwell_minutes pass validation, while dwell times
    exceeding the dividing line boundary raise ValueError.

    WHY: Amendment A-014 reserves extended dwell periods for held-out tactic evaluations; training gangs must
    strictly forward money within the training maximum dwell window.

    Args:
        None.

    Returns:
        None. Fails if excessive dwell times pass or permitted dwell times fail.

    Example:
        # Dwell of 30 minutes <= 60 passes.
        # Dwell of 90 minutes > 60 raises ValueError.
    """
    # 1. Resolve dynamic training dwell upper bound from world ranges.
    max_dwell = resolve_dividing_line("training_max_dwell_minutes")

    # 2. Assert route with dwell within boundary passes.
    valid_steps = (BlockStep("acquire", (("count", 2),)), BlockStep("route", (("dwell_minutes", max_dwell),)))
    validate_gang_config(GangConfig("gang_dwell_ok", 1, DEFAULT_BUDGET, DEFAULT_SOURCE, valid_steps))

    # 3. Assert route with dwell exceeding boundary raises ValueError.
    excess_dwell = int(max_dwell) + 30
    bad_steps = (BlockStep("acquire", (("count", 2),)), BlockStep("route", (("dwell_minutes", excess_dwell),)))
    with pytest.raises(ValueError, match=f"route dwell {excess_dwell} exceeds training maximum dwell of {max_dwell}"):
        validate_gang_config(GangConfig("gang_dwell_bad", 1, DEFAULT_BUDGET, DEFAULT_SOURCE, bad_steps))


def test_route_dividing_line_amount_is_enforced_or_passed() -> None:
    """
    WHAT: Asserts that route payment amounts at or above slow_drip_max_amount_inr pass validation, while amounts
    below the dividing line boundary raise ValueError.

    WHY: Micro-transfers below the slow drip dividing line are held-out tactics; training routes must never send
    amounts below the lower bound of mule_transfer_amount_inr.

    Args:
        None.

    Returns:
        None. Fails if amounts below slow drip threshold pass or valid amounts fail.

    Example:
        # Amount 5000 INR >= 2000 INR passes.
        # Amount 1000 INR < 2000 INR raises ValueError.
    """
    # 1. Resolve dynamic slow drip amount lower bound from world ranges.
    min_amount = resolve_dividing_line("slow_drip_max_amount_inr")

    # 2. Assert route with amount at or above boundary passes.
    valid_steps = (BlockStep("acquire", (("count", 2),)), BlockStep("route", (("amount_inr", min_amount),)))
    validate_gang_config(GangConfig("gang_amt_ok", 1, DEFAULT_BUDGET, DEFAULT_SOURCE, valid_steps))

    # 3. Assert route with amount below boundary raises ValueError.
    low_amount = int(min_amount) - 500
    bad_steps = (BlockStep("acquire", (("count", 2),)), BlockStep("route", (("amount_inr", low_amount),)))
    with pytest.raises(ValueError, match=f"route amount {low_amount} INR is below training minimum slow drip amount"):
        validate_gang_config(GangConfig("gang_amt_bad", 1, DEFAULT_BUDGET, DEFAULT_SOURCE, bad_steps))


def test_route_layers_must_be_at_least_one() -> None:
    """
    WHAT: Asserts that route steps specifying layers >= 1 pass validation, while layers < 1 raise ValueError.

    WHY: A laundering chain must have at least one hop; zero or negative layers represent invalid configurations.

    Args:
        None.

    Returns:
        None. Fails if layers < 1 pass or layers >= 1 fail.

    Example:
        # layers = 2 passes.
        # layers = 0 raises ValueError.
    """
    # 1. Assert route with positive layers passes.
    valid_steps = (BlockStep("acquire", (("count", 2),)), BlockStep("route", (("layers", 2),)))
    validate_gang_config(GangConfig("gang_layers_ok", 1, DEFAULT_BUDGET, DEFAULT_SOURCE, valid_steps))

    # 2. Assert route with zero layers raises ValueError.
    bad_steps = (BlockStep("acquire", (("count", 2),)), BlockStep("route", (("layers", 0),)))
    with pytest.raises(ValueError, match="route layers must be at least 1, got 0."):
        validate_gang_config(GangConfig("gang_layers_bad", 1, DEFAULT_BUDGET, DEFAULT_SOURCE, bad_steps))


def test_unknown_parameter_name_is_refused_or_passed() -> None:
    """
    WHAT: Asserts that known block parameters pass validation, while unrecognized parameter names raise ValueError.

    WHY: Guarantees that only declared block schema parameters can be configured, preventing typos and silent errors.

    Args:
        None.

    Returns:
        None. Fails if unknown parameter names are accepted.

    Example:
        # acquire with 'count' passes.
        # acquire with 'invalid_param' raises ValueError.
    """
    # 1. Assert acquire with valid 'count' parameter passes.
    valid_steps = (BlockStep("acquire", (("count", 5),)),)
    validate_gang_config(GangConfig("gang_param_ok", 1, DEFAULT_BUDGET, DEFAULT_SOURCE, valid_steps))

    # 2. Assert acquire with unrecognized parameter raises ValueError.
    bad_steps = (BlockStep("acquire", (("invalid_param", 5),)),)
    with pytest.raises(ValueError, match="block 'acquire' has unknown parameter 'invalid_param'."):
        validate_gang_config(GangConfig("gang_param_bad", 1, DEFAULT_BUDGET, DEFAULT_SOURCE, bad_steps))


def test_gang_config_to_plain_serializes_canonical_dictionary() -> None:
    """
    WHAT: Asserts that gang_config_to_plain produces a complete, canonical plain dictionary representation
    matching the input GangConfig fields.

    WHY: Decision D5 requires gang configurations to be included in effective_config in canonical plain-data form
    to deterministically differentiate run IDs.

    Args:
        None.

    Returns:
        None. Fails if serialized dictionary deviates from canonical schema.

    Example:
        # plain = gang_config_to_plain(config)
        # assert plain["gang_id"] == "gang_alpha"
    """
    # 1. Construct representative gang configuration.
    steps = (BlockStep("acquire", (("count", 4),)), BlockStep("route", (("layers", 2),)))
    config = GangConfig("gang_alpha", 2, DEFAULT_BUDGET, DEFAULT_SOURCE, steps)

    # 2. Serialize to plain dictionary and verify all sections.
    plain = gang_config_to_plain(config)
    assert plain["gang_id"] == "gang_alpha"
    assert plain["strength"] == 2
    assert plain["budget"]["money_paise"] == 1_000_000
    assert plain["source"]["account_id"] == "acc_collector_01"
    assert len(plain["steps"]) == 2
    assert plain["steps"][0] == {"block": "acquire", "params": [("count", 4)]}


def test_take_control_devices_range_follows_config(tmp_path, monkeypatch) -> None:
    """
    WHAT: Asserts that take_control devices validation bounds are read dynamically from configuration.
    It writes a temporary narrowed tactic_blocks.yaml with devices range [2, 2], points tactic_config at it,
    and checks that devices=1 is refused while devices=2 is accepted.

    WHY: Project rules prohibit hardcoding numerical bounds into validation logic; this test proves that
    updating the configuration file directly controls validator acceptance thresholds.

    Args:
        tmp_path: Pytest temporary directory fixture.
        monkeypatch: Pytest monkeypatch fixture to override TACTIC_BLOCKS_PATH.

    Returns:
        None. Fails if devices=1 is accepted or devices=2 is refused under range [2, 2].

    Example:
        # With range [2, 2], validate_gang_config refuses devices=1 and accepts devices=2.
    """
    # 1. Create narrowed tactic_blocks.yaml with devices range [2, 2].
    blocks_data = load_yaml(TACTIC_BLOCKS_PATH)
    blocks_data["tactic_blocks"]["blocks"]["take_control"]["parameters"]["devices"]["range"] = [2, 2]
    temp_blocks_path = tmp_path / "tactic_blocks.yaml"
    with open(temp_blocks_path, "w", encoding="utf-8") as file_handle:
        yaml.safe_dump(blocks_data, file_handle)

    # 2. Monkeypatch TACTIC_BLOCKS_PATH in tactic_config.
    monkeypatch.setattr(tactic_config, "TACTIC_BLOCKS_PATH", temp_blocks_path)

    # 3. Assert devices=1 is refused.
    steps_1 = (BlockStep("acquire", (("count", 2),)), BlockStep("take_control", (("devices", 1),)))
    with pytest.raises(ValueError, match=r"take_control devices must be an integer in \[2, 2\], got 1"):
        validate_gang_config(GangConfig("gang_dev1", 1, DEFAULT_BUDGET, DEFAULT_SOURCE, steps_1))

    # 4. Assert devices=2 is accepted.
    steps_2 = (BlockStep("acquire", (("count", 2),)), BlockStep("take_control", (("devices", 2),)))
    validate_gang_config(GangConfig("gang_dev2", 1, DEFAULT_BUDGET, DEFAULT_SOURCE, steps_2))
