"""
WHAT: Held-out tactic option inspection, policy validation, and range safety assertions.

WHY: Level 12g requires held-out tactics (slow_drip and device_rotation) to be impossible
in training and validation worlds. This module verifies that gangs attempting to use held-out
parameters are executed exclusively in exempt worlds (test and stress splits). It also proves
from configuration that held-out parameter ranges never overlap training ranges.

ISSUE IDs: 12-8
TERM: held-out option: a tactic block parameter flagged in tactic_blocks.yaml with held_out_family.
TERM: exempt split: an evaluation split (test or stress) permitted to run held-out tactics.
"""

from pathlib import Path
from typing import Any

from world.gang_config import GangConfig
from world.tactic_config import get_block_definition, load_tactic_blocks, resolve_dividing_line
from world.truth_records import GangTruth


def find_held_out_options(config: GangConfig) -> list[tuple[str, str, str]]:
    """
    WHAT: Discovers all held-out tactic parameters explicitly declared across a gang's configured steps.

    WHY: Connects individual step parameter assignments to tactic family classification schemas in tactic_blocks.yaml.

    Args:
        config: GangConfig specification containing campaign steps and parameters.

    Returns:
        List of (block_name, parameter_name, family_name) tuples for all held-out parameters configured.

    Raises:
        ValueError: If configuration inspection fails.

    Example:
        # options = find_held_out_options(gang_config)
        # assert ("route", "drip_piece_max_inr", "slow_drip") in options
    """
    held_out_found: list[tuple[str, str, str]] = []
    for step in config.steps:
        try:
            block_def = get_block_definition(step.block)
        except KeyError:
            # Unknown blocks or unregistered plugins are not built-in and possess no held-out schemas.
            continue

        param_specs = block_def.get("parameters", {})
        for param_name, _ in step.params:
            if param_name not in param_specs:
                continue
            spec = param_specs[param_name]
            if isinstance(spec, dict) and "held_out_family" in spec:
                family_name = str(spec["held_out_family"])
                held_out_found.append((step.block, param_name, family_name))

    return held_out_found


def load_exempt_splits(config_path: Path | None = None) -> tuple[str, ...]:
    """
    WHAT: Retrieves the tuple of evaluation split names permitted to execute held-out tactics.

    WHY: Ground Rule 15 requires policy constants to reside in YAML configuration rather than code.

    Args:
        config_path: Optional filesystem Path to tactic_blocks.yaml.

    Returns:
        Tuple of string split identifiers allowed to run held-out tactics.

    Raises:
        KeyError: If held_out_dividing_lines or exempt_splits is missing.

    Example:
        # assert load_exempt_splits() == ("test", "stress")
    """
    dividing_lines = load_tactic_blocks(config_path).get("held_out_dividing_lines", {})
    if "exempt_splits" not in dividing_lines:
        raise KeyError("Missing 'exempt_splits' in held_out_dividing_lines configuration.")
    exempt_spec = dividing_lines["exempt_splits"]
    raw_splits = exempt_spec.get("value", ())
    return tuple(str(s) for s in raw_splits)


def check_held_out_options_allowed(config: GangConfig, world_split: str | None) -> None:
    """
    WHAT: Enforces that gangs with held-out tactic parameters run only within exempt evaluation splits.

    WHY: Prevents held-out tactics from contaminating training and validation worlds (Level 12g fail-closed policy).

    Args:
        config: GangConfig instance being validated for planning.
        world_split: String split identifier (e.g. 'training', 'test', 'stress') or None if unassigned.

    Returns:
        None. Passes cleanly if no held-out options are present or if the split is exempt.

    Raises:
        ValueError: If held-out options are present and world_split is None or non-exempt.

    Example:
        # check_held_out_options_allowed(drip_gang, "training")  # raises ValueError
    """
    held_out = find_held_out_options(config)
    if not held_out:
        return

    exempt_splits = load_exempt_splits()
    _, first_option, first_family = held_out[0]

    if world_split is None:
        raise ValueError(
            f"Gang '{config.gang_id}' uses held-out option {first_option} (family {first_family}), "
            "which is allowed only in test and stress worlds; no split was given."
        )

    if world_split not in exempt_splits:
        raise ValueError(
            f"Gang '{config.gang_id}' uses held-out option {first_option} (family {first_family}), "
            f"which is allowed only in test and stress worlds; this world's split is {world_split}."
        )


def check_held_out_option_ranges(config_path: Path | None = None) -> None:
    """
    WHAT: Verifies that held-out configuration ranges maintain strict mathematical separation from training bounds.

    WHY: Proves from configuration files that held-out parameter values can never overlap training parameter values.

    Args:
        config_path: Optional filesystem Path to tactic_blocks.yaml.

    Returns:
        None. Passes cleanly if dividing line boundaries are strictly respected.

    Raises:
        ValueError: If drip_piece_max_inr upper bound >= slow_drip_max_amount_inr,
            or if drip_hold_minutes lower bound <= training_max_dwell_minutes.

    Example:
        # check_held_out_option_ranges()  # passes on valid tactic_blocks.yaml
    """
    route_def = get_block_definition("route", config_path)
    route_params = route_def.get("parameters", {})

    drip_piece_spec = route_params.get("drip_piece_max_inr", {})
    piece_range = drip_piece_spec.get("range", [])
    if len(piece_range) != 2:
        raise ValueError("Parameter 'drip_piece_max_inr' must have a 2-element range.")
    piece_max_upper = piece_range[1]

    drip_hold_spec = route_params.get("drip_hold_minutes", {})
    hold_range = drip_hold_spec.get("range", [])
    if len(hold_range) != 2:
        raise ValueError("Parameter 'drip_hold_minutes' must have a 2-element range.")
    hold_min_lower = hold_range[0]

    amount_limit = resolve_dividing_line("slow_drip_max_amount_inr", config_path=config_path)
    dwell_limit = resolve_dividing_line("training_max_dwell_minutes", config_path=config_path)

    if piece_max_upper >= amount_limit:
        raise ValueError(
            f"Held-out drip_piece_max_inr upper bound ({piece_max_upper}) must be strictly below "
            f"slow_drip_max_amount_inr dividing line ({amount_limit})."
        )

    if hold_min_lower <= dwell_limit:
        raise ValueError(
            f"Held-out drip_hold_minutes lower bound ({hold_min_lower}) must be strictly above "
            f"training_max_dwell_minutes dividing line ({dwell_limit})."
        )


def gang_truth_for(config: GangConfig, families: dict[str, Any]) -> GangTruth:
    """
    WHAT: Derives the ground truth GangTruth classification record for a gang configuration based
    on its executed options.

    WHY: Ground truth evaluations in Level 12g require verifying that a gang's taxonomy label reflects its
    actual mechanism parameters rather than unverified user-declared tags.

    Args:
        config: GangConfig instance declaring gang steps, parameters, and optional family tag.
        families: Dictionary parsed from tactic_families.yaml containing training and held_out family specifications.

    Returns:
        GangTruth: Validated truth record indicating whether the gang is seen or held-out.

    Raises:
        ValueError: If a gang configures options from multiple held-out families, if a declared family tag
            contradicts detected held-out options, or if a held-out family tag is used without its mechanism.

    Example:
        # gt = gang_truth_for(drip_gang_config, families)
        # assert gt.kind == "held_out"
    """
    # Discover all parameters across gang steps mapped to held-out tactic families.
    options = find_held_out_options(config)
    distinct_families = sorted(set(family for _, _, family in options))
    held_out_options = tuple(sorted(f"{b}.{p}" for b, p, _ in options))

    # Reject mixing multiple held-out families to keep evaluation results attributable per family.
    if len(distinct_families) > 1:
        raise ValueError(
            f"Gang '{config.gang_id}' uses options from multiple held-out families "
            f"{distinct_families}; one held-out family per gang, so results stay per family."
        )

    # If exactly one held-out family is configured, the gang is held-out and its tag must agree.
    if len(distinct_families) == 1:
        family_name = distinct_families[0]
        if config.family is not None and config.family != family_name:
            raise ValueError(
                f"Gang '{config.gang_id}' is tagged with family '{config.family}' but uses options from "
                f"held-out family '{family_name}'; tag disagrees with behaviour."
            )
        return GangTruth(
            gang_id=config.gang_id,
            family=family_name,
            kind="held_out",
            held_out_options=held_out_options,
        )

    # When no held-out options are present, disallow claiming a held-out family name without its mechanism.
    held_out_names = set(families.get("held_out", {}).keys())
    if config.family in held_out_names:
        raise ValueError(
            f"Gang '{config.gang_id}' is tagged held-out family '{config.family}' but uses none of its options; "
            "a held-out name needs its mechanism."
        )

    # Ordinary seen gang with no held-out options configured.
    return GangTruth(
        gang_id=config.gang_id,
        family=config.family,
        kind="seen",
        held_out_options=(),
    )

