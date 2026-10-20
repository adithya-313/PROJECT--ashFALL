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


def load_tactic_blocks(config_path: Path | None = None) -> dict[str, Any]:
    """
    WHAT: Reads config/tactic_blocks.yaml and returns the entire parsed YAML dictionary.

    WHY: Provides a centralized entry point for inspecting tactic building blocks, dividing lines, and payment purposes.

    Args:
        config_path: Optional filesystem Path to tactic_blocks.yaml (defaults to TACTIC_BLOCKS_PATH).

    Returns:
        Dictionary holding top-level sections: tactic_blocks, held_out_dividing_lines, and gang_payment_purpose.

    Example:
        # blocks_config = load_tactic_blocks()
        # assert "acquire" in blocks_config["tactic_blocks"]["blocks"]
    """
    # 1. Resolve path dynamically and load YAML dictionary.
    resolved_path = config_path if config_path is not None else TACTIC_BLOCKS_PATH
    return load_yaml(resolved_path)


def load_tactic_costs(costs_path: Path | None = None) -> dict[str, Any]:
    """
    WHAT: Reads config/tactic_costs.yaml and returns the entire parsed YAML dictionary.

    WHY: Provides cost units, fee parameters, and per-block risk distributions for gang budget accounting.

    Args:
        costs_path: Optional filesystem Path to tactic_costs.yaml (defaults to TACTIC_COSTS_PATH).

    Returns:
        Dictionary holding tactic_cost_units and tactic_risk_per_use sections.

    Example:
        # costs_config = load_tactic_costs()
        # assert "tactic_risk_per_use" in costs_config
    """
    # 1. Resolve path dynamically and load YAML dictionary.
    resolved_path = costs_path if costs_path is not None else TACTIC_COSTS_PATH
    return load_yaml(resolved_path)


def load_world_ranges(world_ranges_path: Path | None = None) -> dict[str, Any]:
    """
    WHAT: Reads config/world_ranges.yaml and returns the parsed parameters dictionary.

    WHY: Provides parameter ranges across training, test, and stress splits to resolve dividing lines and bounds.

    Args:
        world_ranges_path: Optional filesystem Path to world_ranges.yaml (defaults to WORLD_RANGES_PATH).

    Returns:
        Dictionary containing parameters specification and calibrated range bounds.

    Example:
        # ranges_dict = load_world_ranges()
        # assert "parameters" in ranges_dict
    """
    # 1. Resolve path dynamically and load YAML dictionary.
    resolved_path = world_ranges_path if world_ranges_path is not None else WORLD_RANGES_PATH
    return load_yaml(resolved_path)


def resolve_dividing_line(
    name: str,
    config_path: Path | None = None,
    world_ranges_path: Path | None = None,
) -> int | float | bool:
    """
    WHAT: Resolves a named held-out dividing line dynamically from config/world_ranges.yaml by reference.
    Returns the resolved boundary number (for dwell or amount limits) or boolean rule value (for device retirement).

    WHY: Ground rule 15 and amendment A-014 require dividing lines to reference world_ranges bounds without copying
    numbers into code, ensuring training bounds never leak or drift from reality assumptions.

    Args:
        name: String name of the dividing line (e.g., 'training_max_dwell_minutes').
        config_path: Optional filesystem Path to tactic_blocks.yaml.
        world_ranges_path: Optional filesystem Path to world_ranges.yaml.

    Returns:
        Numeric boundary value (int or float) or boolean flag.

    Raises:
        KeyError: If the dividing line name is not found in held_out_dividing_lines.

    Example:
        # assert resolve_dividing_line("training_max_dwell_minutes") == 60
        # assert resolve_dividing_line("slow_drip_max_amount_inr") == 2000
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

    ranges_data = load_world_ranges(world_ranges_path)["parameters"][parameter_name]
    split_bounds = ranges_data[split]
    bound_index = 1 if bound_type == "upper" else 0
    return split_bounds[bound_index]


def get_block_definition(block_name: str, config_path: Path | None = None) -> dict[str, Any]:
    """
    WHAT: Retrieves the specification dictionary for a building block from config/tactic_blocks.yaml.

    WHY: Enables the gang validator and planner to inspect allowed strength levels, parameter schemas, and timings.

    Args:
        block_name: Identifier of the building block (acquire, warm_up, route, disguise).
        config_path: Optional filesystem Path to tactic_blocks.yaml.

    Returns:
        Dictionary containing block metadata, allowed strengths, and parameter definitions.

    Raises:
        KeyError: If block_name is not registered in tactic_blocks.yaml.

    Example:
        # acquire_block = get_block_definition("acquire")
        # assert acquire_block["allowed_strengths"] == [1, 2]
    """
    # 1. Fetch blocks dictionary and verify presence of block_name.
    blocks = load_tactic_blocks(config_path)["tactic_blocks"]["blocks"]
    if block_name not in blocks:
        raise KeyError(f"Unknown tactic building block '{block_name}'.")
    return blocks[block_name]


def resolve_parameter_training_range(
    block_name: str,
    parameter_name: str,
    config_path: Path | None = None,
    world_ranges_path: Path | None = None,
) -> tuple[float, float] | tuple[int, int]:
    """
    WHAT: Resolves the calibrated training range for a tactic block parameter referenced in config/world_ranges.yaml.
    It inspects the parameter definition in config/tactic_blocks.yaml and fetches the corresponding split bounds.

    WHY: Ground Rule 15 forbids hardcoding parameter ranges; resolving through world ranges ensures synchronization.

    Args:
        block_name: Building block identifier (e.g., 'route').
        parameter_name: Parameter name inside the block (e.g., 'dwell_minutes', 'amount_inr').
        config_path: Optional path override for tactic_blocks.yaml.
        world_ranges_path: Optional path override for world_ranges.yaml.

    Returns:
        Two-element tuple representing the lower and upper bounds of the parameter's training range.

    Raises:
        KeyError: If block, parameter, or world ranges reference cannot be found.

    Example:
        # min_dwell, max_dwell = resolve_parameter_training_range("route", "dwell_minutes")
        # assert min_dwell == 5 and max_dwell == 60
    """
    # 1. Inspect block parameter definition in tactic blocks configuration.
    block_def = get_block_definition(block_name, config_path)
    param_spec = block_def.get("parameters", {}).get(parameter_name, {})
    if "world_ranges_reference" in param_spec:
        ref_name = param_spec["world_ranges_reference"]
        split = param_spec.get("split", "training")
        ranges_data = load_world_ranges(world_ranges_path)["parameters"][ref_name]
        split_bounds = ranges_data[split]
        return split_bounds[0], split_bounds[1]

    if "range" in param_spec:
        range_bounds = param_spec["range"]
        return range_bounds[0], range_bounds[1]

    raise KeyError(f"Parameter '{parameter_name}' of block '{block_name}' lacks range or reference.")


def get_dwell_range(
    config_path: Path | None = None,
    world_ranges_path: Path | None = None,
) -> tuple[int, int]:
    """
    WHAT: Resolves the [min, max] dwell duration range in simulated minutes for routing hops.

    WHY: Ensures intermediate account holding windows match the calibrated training distribution in world ranges.

    Args:
        config_path: Optional path override for tactic_blocks.yaml.
        world_ranges_path: Optional path override for world_ranges.yaml.

    Returns:
        Two-element integer tuple (min_dwell, max_dwell).

    Example:
        # min_dwell, max_dwell = get_dwell_range()
        # assert min_dwell == 5 and max_dwell == 60
    """
    # 1. Resolve dwell minutes training range from world ranges reference.
    bounds = resolve_parameter_training_range("route", "dwell_minutes", config_path, world_ranges_path)
    return int(bounds[0]), int(bounds[1])


def get_route_amount_range(
    config_path: Path | None = None,
    world_ranges_path: Path | None = None,
) -> tuple[float, float]:
    """
    WHAT: Resolves the [min, max] transfer amount range in Indian Rupees (INR) for routing payments.

    WHY: Ensures stochastic hop amounts match the calibrated training distribution in world ranges.

    Args:
        config_path: Optional path override for tactic_blocks.yaml.
        world_ranges_path: Optional path override for world_ranges.yaml.

    Returns:
        Two-element float tuple (min_inr, max_inr).

    Example:
        # min_amt, max_amt = get_route_amount_range()
        # assert min_amt == 2000.0 and max_amt == 50000.0
    """
    # 1. Resolve transfer amount training range from world ranges reference.
    bounds = resolve_parameter_training_range("route", "amount_inr", config_path, world_ranges_path)
    return float(bounds[0]), float(bounds[1])


def get_supplier_fee_range(costs_path: Path | None = None) -> tuple[float, float]:
    """
    WHAT: Retrieves the [min, max] account supplier fee share fraction from config/tactic_costs.yaml.

    WHY: Used by the tactical gang planner to sample money costs incurred for renting citizen mule accounts.

    Args:
        costs_path: Optional path override for tactic_costs.yaml.

    Returns:
        Two-element float tuple (min_fee_share, max_fee_share).

    Example:
        # min_fee, max_fee = get_supplier_fee_range()
        # assert min_fee == 0.03 and max_fee == 0.06
    """
    # 1. Extract fee share range from tactic cost units parameter configuration.
    params = load_tactic_costs(costs_path)["tactic_cost_units"]["parameters"]
    fee_range = params["account_supplier_fee_share"]["range"]
    return float(fee_range[0]), float(fee_range[1])


def get_block_time_range(block_name: str, config_path: Path | None = None) -> tuple[int, int]:
    """
    WHAT: Retrieves the [min, max] execution duration range in simulated minutes for a tactic block.

    WHY: Provides duration bounds for stochastic time expenditure sampling during gang campaign synthesis.

    Args:
        block_name: Identifier of the tactic building block (e.g., 'acquire').
        config_path: Optional path override for tactic_blocks.yaml.

    Returns:
        Two-element integer tuple (min_minutes, max_minutes).

    Example:
        # min_time, max_time = get_block_time_range("acquire")
        # assert min_time == 60 and max_time == 1440
    """
    # 1. Fetch time minutes specification from block definition.
    definition = get_block_definition(block_name, config_path)
    time_spec = definition["time_minutes_per_use"]
    return int(time_spec[0]), int(time_spec[1])


def get_disguise_jitter_range(config_path: Path | None = None) -> tuple[int, int]:
    """
    WHAT: Retrieves the [min, max] minute jitter range for timing camouflage from config/tactic_blocks.yaml.

    WHY: Used by disguise step planning and send minute computation to bound dispatch perturbation intervals.

    Args:
        config_path: Optional path override for tactic_blocks.yaml.

    Returns:
        Two-element integer tuple (min_jitter, max_jitter).

    Example:
        # min_jitter, max_jitter = get_disguise_jitter_range()
        # assert min_jitter == 0 and max_jitter == 30
    """
    # 1. Fetch jitter range from disguise block parameters.
    definition = get_block_definition("disguise", config_path)
    jitter_spec = definition["parameters"]["jitter_minutes"]["range"]
    return int(jitter_spec[0]), int(jitter_spec[1])


def get_block_risk_range(block_name: str, costs_path: Path | None = None) -> tuple[float, float]:
    """
    WHAT: Retrieves the [min, max] probability-per-use risk range for a tactic building block.

    WHY: Used by the gang planner to sample stochastic execution risk for cumulative budget evaluation.

    Args:
        block_name: Identifier of the building block (acquire, warm_up, route, disguise).
        costs_path: Optional filesystem Path to tactic_costs.yaml.

    Returns:
        Two-element float tuple representing (lower_bound, upper_bound) of probability per use.

    Raises:
        KeyError: If block_name is not registered under tactic_risk_per_use.

    Example:
        # lower, upper = get_block_risk_range("acquire")
        # assert lower == 0.001 and upper == 0.01
    """
    # 1. Fetch risk distributions dictionary from tactic costs configuration.
    risk_data = load_tactic_costs(costs_path)["tactic_risk_per_use"]
    if block_name not in risk_data:
        raise KeyError(f"No risk per use configuration found for block '{block_name}'.")

    # 2. Return validated range as float tuple.
    range_list = risk_data[block_name]["range"]
    return float(range_list[0]), float(range_list[1])


def get_gang_payment_purpose(config_path: Path | None = None) -> str:
    """
    WHAT: Returns the operational payment purpose assigned to simulated gang payments.

    WHY: Decision D4 standardizes gang payments on 'everyday' in Level 12b until specialized purposes are added in 12c.

    Args:
        config_path: Optional filesystem Path to tactic_blocks.yaml.

    Returns:
        String payment purpose (e.g., 'everyday').

    Example:
        # assert get_gang_payment_purpose() == "everyday"
    """
    # 1. Extract purpose string from gang_payment_purpose policy section.
    policy = load_tactic_blocks(config_path).get("gang_payment_purpose", {})
    return str(policy.get("purpose", "everyday"))


def get_gang_device_prefix(config_path: Path | None = None) -> str:
    """
    WHAT: Returns the neutral identifier prefix used when synthesizing gang-controlled devices.
    It reads the device_id_prefix string from the gang_devices policy section in config/tactic_blocks.yaml.

    WHY: Decision E2 requires device identifiers to be neutral (such as 'dev_ext_') so that the ID string
    itself never leaks the device's illicit role or gang affiliation to simulated banking systems.

    Args:
        config_path: Optional filesystem Path to tactic_blocks.yaml.

    Returns:
        String prefix for external devices (e.g., 'dev_ext_').

    Example:
        # assert get_gang_device_prefix() == "dev_ext_"
    """
    # 1. Extract device identifier prefix from gang_devices configuration section.
    section = load_tactic_blocks(config_path).get("gang_devices", {})
    return str(section.get("device_id_prefix", "dev_ext_"))


def get_gang_device_event_kinds(config_path: Path | None = None) -> list[str]:
    """
    WHAT: Returns the list of permitted event kind names for gang device lifecycle and compromise telemetry.
    It extracts the allowed kinds from the gang_devices policy section in config/tactic_blocks.yaml.

    WHY: Decision E4 defines distinct telemetry event kinds for device acquisition and account takeover
    flags (SIM swap, linked number switch, emulator, remote access) to model external device events.

    Args:
        config_path: Optional filesystem Path to tactic_blocks.yaml.

    Returns:
        List of event kind string names.

    Example:
        # assert "device_acquired" in get_gang_device_event_kinds()
        # assert "sim_swap" in get_gang_device_event_kinds()
    """
    # 1. Extract device event kinds list from gang_devices configuration section.
    section = load_tactic_blocks(config_path).get("gang_devices", {})
    kinds = section.get("device_event_kinds", [])
    return [str(item) for item in kinds]


def get_take_control_devices_range(config_path: Path | None = None) -> tuple[int, int]:
    """
    WHAT: Retrieves the [min, max] device count range for take_control from config/tactic_blocks.yaml.

    WHY: Used by gang configuration validation to ensure device counts lie within configured bounds without hardcoding.

    Args:
        config_path: Optional path override for tactic_blocks.yaml.

    Returns:
        Two-element integer tuple (min_devices, max_devices).

    Example:
        # min_devices, max_devices = get_take_control_devices_range()
        # assert min_devices == 1 and max_devices == 5
    """
    # 1. Fetch devices parameter range from take_control block definition.
    definition = get_block_definition("take_control", config_path)
    devices_range = definition["parameters"]["devices"]["range"]
    return int(devices_range[0]), int(devices_range[1])


def get_scam_receiving_mules_range(config_path: Path | None = None) -> tuple[int, int]:
    """
    WHAT: Retrieves the [min, max] count of receiving mules a scam step may name, from config/tactic_blocks.yaml.

    WHY: Gang configuration validation checks the scam step's receiving_mules against this range, so the limit is
    never typed in code.

    Args:
        config_path: Optional path override for tactic_blocks.yaml.

    Returns:
        Two-element integer tuple (min_mules, max_mules).

    Example:
        # min_mules, max_mules = get_scam_receiving_mules_range()
        # assert min_mules == 1 and max_mules == 10
    """
    # 1. Fetch the receiving_mules parameter range from the scam block definition.
    definition = get_block_definition("scam", config_path)
    mules_range = definition["parameters"]["receiving_mules"]["range"]
    return int(mules_range[0]), int(mules_range[1])


def get_scam_victims_per_mule_range(
    config_path: Path | None = None, world_ranges_path: Path | None = None
) -> tuple[int, int]:
    """
    WHAT: Resolves the [min, max] count of victims per receiving mule from the world ranges the scam block names.

    WHY: The scam step draws each mule's victim count from this range, and validation checks an explicit count
    against it.

    Args:
        config_path: Optional path override for tactic_blocks.yaml.
        world_ranges_path: Optional path override for world_ranges.yaml.

    Returns:
        Two-element integer tuple (min_victims, max_victims).

    Example:
        # min_victims, max_victims = get_scam_victims_per_mule_range()
        # assert min_victims == 1 and max_victims == 5
    """
    # 1. Resolve the training range through the block parameter's world_ranges_reference.
    bounds = resolve_parameter_training_range("scam", "victims_per_receiving_mule", config_path, world_ranges_path)
    return int(bounds[0]), int(bounds[1])

