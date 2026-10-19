"""
WHAT: Configuration loader and dividing line resolution module for gang tactic building blocks
(config/tactic_blocks.yaml). Loads block specifications, allowed strength levels, parameter bounds by reference,
per-block risk distributions, and resolves training-versus-held-out dividing lines dynamically from world ranges.

WHY: Hardcoding numerical limits or copying parameter bounds into source code creates synchronization gaps.
Resolving dividing lines by dynamic reference to config/world_ranges.yaml guarantees that changes to underlying
ranges propagate automatically while enforcing architectural separation per amendment A-014 and A-016.

TERM: building block: an atomic step in a fraud tactic campaign (acquire, warm_up, route, disguise).
TERM: strength level: the sophistication and camouflage applied by a gang (strength 1 = fixed, 2 = camouflaged).
TERM: dwell: the duration in simulated minutes money stays in an account before being forwarded.
TERM: layer: the hop distance in a laundering payment chain from the source to terminal exit.
TERM: budget: maximum allowed expenditure in money, time, and cumulative risk for a gang campaign.
TERM: dividing line: boundary separating training tactics from held-out tactics to prevent data leakage.
"""

from pathlib import Path
from typing import Any

from core.config_validator import load_yaml
from world.run_identity import REPO_ROOT

TACTIC_BLOCKS_PATH: Path = REPO_ROOT / "config" / "tactic_blocks.yaml"
TACTIC_COSTS_PATH: Path = REPO_ROOT / "config" / "tactic_costs.yaml"
WORLD_RANGES_PATH: Path = REPO_ROOT / "config" / "world_ranges.yaml"


def load_tactic_blocks(config_path: Path = TACTIC_BLOCKS_PATH) -> dict[str, Any]:
    """
    WHAT: Reads config/tactic_blocks.yaml and returns the entire parsed YAML dictionary.

    WHY: Provides a centralized entry point for inspecting tactic building blocks, dividing lines, and payment purposes.

    Args:
        config_path: Filesystem Path pointing to tactic_blocks.yaml (defaults to repository path).

    Returns:
        Dictionary holding top-level sections: tactic_blocks, held_out_dividing_lines, and gang_payment_purpose.

    Example:
        # blocks_config = load_tactic_blocks()
        # "acquire" in blocks_config["tactic_blocks"]["blocks"]
    """
    # 1. Load and return YAML configuration dictionary.
    return load_yaml(config_path)


def load_tactic_costs(costs_path: Path = TACTIC_COSTS_PATH) -> dict[str, Any]:
    """
    WHAT: Reads config/tactic_costs.yaml and returns the entire parsed YAML dictionary.

    WHY: Provides cost units, fee parameters, and per-block risk distributions for gang budget accounting.

    Args:
        costs_path: Filesystem Path pointing to tactic_costs.yaml (defaults to repository path).

    Returns:
        Dictionary holding tactic_cost_units and tactic_risk_per_use sections.

    Example:
        # costs_config = load_tactic_costs()
        # "tactic_risk_per_use" in costs_config
    """
    # 1. Load and return YAML configuration dictionary.
    return load_yaml(costs_path)


def resolve_dividing_line(
    name: str,
    config_path: Path = TACTIC_BLOCKS_PATH,
    world_ranges_path: Path = WORLD_RANGES_PATH,
) -> int | float | bool:
    """
    WHAT: Resolves a named held-out dividing line dynamically from config/world_ranges.yaml by reference.
    Returns the resolved boundary number (for dwell or amount limits) or boolean rule value (for device retirement).

    WHY: Ground rule 15 and amendment A-014 require dividing lines to reference world_ranges bounds without copying
    numbers into code, ensuring training bounds never leak or drift from reality assumptions.

    Args:
        name: String name of the dividing line (e.g., 'training_max_dwell_minutes').
        config_path: Filesystem Path to tactic_blocks.yaml.
        world_ranges_path: Filesystem Path to world_ranges.yaml.

    Returns:
        Numeric boundary value (int or float) or boolean flag.

    Raises:
        KeyError: If the dividing line name is not found in held_out_dividing_lines.

    Example:
        # resolve_dividing_line("training_max_dwell_minutes") == 60
        # resolve_dividing_line("slow_drip_max_amount_inr") == 2000
    """
    # 1. Look up dividing line specification in tactic blocks configuration.
    dividing_lines = load_tactic_blocks(config_path).get("held_out_dividing_lines", {})
    if name not in dividing_lines:
        raise KeyError(f"Unknown held-out dividing line '{name}'.")

    spec = dividing_lines[name]

    # 2. Return literal rule value if present (e.g., boolean flags).
    if "value" in spec:
        return bool(spec["value"])

    # 3. Resolve parameter bound dynamically from world_ranges.yaml.
    parameter_name = spec["parameter_reference"]
    bound_type = spec["bound"]
    split = spec.get("split", "training")

    ranges_data = load_yaml(world_ranges_path)["parameters"][parameter_name]
    split_bounds = ranges_data[split]
    bound_index = 1 if bound_type == "upper" else 0
    return split_bounds[bound_index]


def get_block_definition(block_name: str, config_path: Path = TACTIC_BLOCKS_PATH) -> dict[str, Any]:
    """
    WHAT: Retrieves the specification dictionary for a building block from config/tactic_blocks.yaml.

    WHY: Enables the gang validator and planner to inspect allowed strength levels, parameter schemas, and timings.

    Args:
        block_name: Identifier of the building block (acquire, warm_up, route, disguise).
        config_path: Filesystem Path to tactic_blocks.yaml.

    Returns:
        Dictionary containing block metadata, allowed strengths, and parameter definitions.

    Raises:
        KeyError: If block_name is not registered in tactic_blocks.yaml.

    Example:
        # acquire_block = get_block_definition("acquire")
        # acquire_block["allowed_strengths"] == [1, 2]
    """
    # 1. Fetch blocks dictionary and verify presence of block_name.
    blocks = load_tactic_blocks(config_path)["tactic_blocks"]["blocks"]
    if block_name not in blocks:
        raise KeyError(f"Unknown tactic building block '{block_name}'.")
    return blocks[block_name]


def get_block_risk_range(block_name: str, costs_path: Path = TACTIC_COSTS_PATH) -> tuple[float, float]:
    """
    WHAT: Retrieves the [min, max] probability-per-use risk range for a tactic building block.

    WHY: Used by the gang planner to sample stochastic execution risk for cumulative budget evaluation.

    Args:
        block_name: Identifier of the building block (acquire, warm_up, route, disguise).
        costs_path: Filesystem Path to tactic_costs.yaml.

    Returns:
        Two-element float tuple representing (lower_bound, upper_bound) of probability per use.

    Raises:
        KeyError: If block_name is not registered under tactic_risk_per_use.

    Example:
        # lower, upper = get_block_risk_range("acquire")
        # lower == 0.001 and upper == 0.01
    """
    # 1. Fetch risk distributions dictionary from tactic costs configuration.
    risk_data = load_tactic_costs(costs_path)["tactic_risk_per_use"]
    if block_name not in risk_data:
        raise KeyError(f"No risk per use configuration found for block '{block_name}'.")

    # 2. Return validated range as float tuple.
    range_list = risk_data[block_name]["range"]
    return float(range_list[0]), float(range_list[1])


def get_gang_payment_purpose(config_path: Path = TACTIC_BLOCKS_PATH) -> str:
    """
    WHAT: Returns the operational payment purpose assigned to simulated gang payments.

    WHY: Decision D4 standardizes gang payments on 'everyday' in Level 12b until specialized purposes are added in 12c.

    Args:
        config_path: Filesystem Path to tactic_blocks.yaml.

    Returns:
        String payment purpose (e.g., 'everyday').

    Example:
        # get_gang_payment_purpose() == "everyday"
    """
    # 1. Extract purpose string from gang_payment_purpose policy section.
    policy = load_tactic_blocks(config_path).get("gang_payment_purpose", {})
    return str(policy.get("purpose", "everyday"))
