"""
WHAT: Gang reaction decision engine ("the gang mind") for Level 12d reactive criminal syndicates.
Maintains per-gang state including rule adherence and stopped accounts, updates state upon
receiving delayed observations from the world feed, and answers whether accounts remain usable.

WHY: Realistic syndicates respond to operational disruptions: when an account is liened,
a gang following the stop rule stops moving money through it to avoid losing stolen funds.
Strict isolation prevents the gang mind from accessing ground truth or bank ledgers.

TERM: gang mind: isolated per-gang state machine updating solely on delivered GangObservation items.
TERM: stop rule: fixed policy of halting all incoming, outgoing, and cash-out activity on a compromised account.
TERM: stopped account: an account marked unusable by the gang mind following an observed lien.
TERM: rotation rule: policy of redirecting payments from stopped mules to operational spare mules at the same depth.
TERM: split rule: policy of splitting transfers above a threshold into equal legs below it upon observing trouble.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from core.config_validator import load_yaml
from world.gang_feed import REPO_ROOT, WORLD_RANGES_PATH, GangObservation, load_gang_reactions_config

TACTIC_BLOCKS_PATH: Path = REPO_ROOT / "config" / "tactic_blocks.yaml"


@dataclass
class GangReactionState:
    """
    WHAT: In-memory operational decision state for one criminal syndicate.
    Tracks whether the gang follows the stop-account rule, rotation rule, and splitting rule,
    along with stopped accounts, split threshold, and whether friction trouble was observed.

    WHY: Ground rule A-017 requires per-gang reaction tracking driven purely by observable
    friction events, isolating gang intelligence from simulation ground truth.

    Args:
        gang_id: Unique identifier for the criminal syndicate.
        follows_stop_rule: Boolean flag indicating if this gang halts activity on liened accounts.
        stopped_accounts: Set of account identifiers marked unusable by this gang mind.
        follows_rotation_rule: Boolean flag indicating if this gang rotates to spare accounts.
        follows_split_rule: Boolean flag indicating if this gang splits transfers above threshold.
        split_threshold_paise: Drawn threshold in integer paise for payment splitting.
        has_seen_trouble: Boolean flag indicating if this gang has observed any friction event.

    Returns:
        An initialized GangReactionState instance.

    Example:
        GangReactionState(gang_id="g1", follows_stop_rule=True)
    """

    gang_id: str
    follows_stop_rule: bool
    stopped_accounts: set[str] = field(default_factory=set)
    follows_rotation_rule: bool = False
    follows_split_rule: bool = False
    split_threshold_paise: int | None = None
    has_seen_trouble: bool = False


def resolve_stop_rule_share_range(
    split: str,
    config: dict[str, Any] | None = None,
    world_ranges_path: Path | None = None,
) -> tuple[float, float]:
    """
    WHAT: Resolves the probability range for adopting the liened account stop rule on a given split.
    Dereferences world_ranges parameter gang_stops_using_liened_account_share.

    WHY: Decouples rule adoption probabilities from hardcoded values, supporting fixture overrides.

    Args:
        split: Dataset split ('training', 'test', or 'stress').
        config: Optional configuration dictionary override.
        world_ranges_path: Optional path override for world_ranges.yaml.

    Returns:
        Tuple of (min_share, max_share).
    """
    if (
        config is not None
        and "parameters" in config
        and "gang_stops_using_liened_account_share" in config["parameters"]
    ):
        param_entry = config["parameters"]["gang_stops_using_liened_account_share"]
        if isinstance(param_entry, dict) and split in param_entry:
            bounds = param_entry[split]
            return float(bounds[0]), float(bounds[1])

    cfg = config if config is not None else load_gang_reactions_config()
    reaction_cfg = cfg.get("gang_reactions", cfg)
    rule_entry = reaction_cfg.get("gang_stops_using_liened_account_share", {})

    if split in rule_entry and isinstance(rule_entry[split], (list, tuple)):
        bounds = rule_entry[split]
        return float(bounds[0]), float(bounds[1])

    ref_key = rule_entry.get("world_ranges_reference", "gang_stops_using_liened_account_share")
    world_ranges_file = world_ranges_path if world_ranges_path is not None else WORLD_RANGES_PATH
    world_ranges = load_yaml(world_ranges_file)
    bounds = world_ranges["parameters"][ref_key][split]
    return float(bounds[0]), float(bounds[1])


def draw_stop_rule_flag(
    split: str,
    rng: np.random.Generator,
    config: dict[str, Any] | None = None,
    world_ranges_path: Path | None = None,
) -> bool:
    """
    WHAT: Draws whether one gang follows the liened-account stop rule using child 15 (gang_reactions).
    First draws the gang's adherence probability from world_ranges, then evaluates a Bernoulli trial.

    WHY: Amendment A-017 specifies two-stage fixed reaction drawing: a probability from world_ranges
    followed by an adherence decision per syndicate.

    Args:
        split: Dataset split name.
        rng: Random number generator dedicated to child 15 (gang_reactions).
        config: Optional configuration override.
        world_ranges_path: Optional path override for world_ranges.yaml.

    Returns:
        Boolean indicating whether the gang adheres to the stop rule.
    """
    bounds = resolve_stop_rule_share_range(split, config, world_ranges_path)
    low_share, high_share = bounds[0], bounds[1]
    rule_prob = float(rng.uniform(low_share, high_share))

    # Second draw: Bernoulli trial against the drawn rule probability.
    decision_draw = float(rng.uniform(0.0, 1.0))
    return decision_draw < rule_prob


def resolve_rotation_share_range(
    split: str,
    config: dict[str, Any] | None = None,
    world_ranges_path: Path | None = None,
) -> tuple[float, float]:
    """
    WHAT: Resolves the probability range for adopting the spare-account rotation rule on a given split.
    Dereferences world_ranges parameter gang_rotates_to_spare_account_share.

    WHY: Decouples rule adoption probabilities from hardcoded values, supporting fixture overrides.

    Args:
        split: Dataset split ('training', 'test', or 'stress').
        config: Optional configuration dictionary override.
        world_ranges_path: Optional path override for world_ranges.yaml.

    Returns:
        Tuple of (min_share, max_share).
    """
    if (
        config is not None
        and "parameters" in config
        and "gang_rotates_to_spare_account_share" in config["parameters"]
    ):
        param_entry = config["parameters"]["gang_rotates_to_spare_account_share"]
        if isinstance(param_entry, dict) and split in param_entry:
            bounds = param_entry[split]
            return float(bounds[0]), float(bounds[1])

    cfg = config if config is not None else load_gang_reactions_config()
    reaction_cfg = cfg.get("gang_reactions", cfg)
    rule_entry = reaction_cfg.get("gang_rotates_to_spare_account_share", {})

    if split in rule_entry and isinstance(rule_entry[split], (list, tuple)):
        bounds = rule_entry[split]
        return float(bounds[0]), float(bounds[1])

    ref_key = rule_entry.get("world_ranges_reference", "gang_rotates_to_spare_account_share")
    world_ranges_file = world_ranges_path if world_ranges_path is not None else WORLD_RANGES_PATH
    world_ranges = load_yaml(world_ranges_file)
    bounds = world_ranges["parameters"][ref_key][split]
    return float(bounds[0]), float(bounds[1])


def draw_rotation_flag(
    split: str,
    rng: np.random.Generator,
    config: dict[str, Any] | None = None,
    world_ranges_path: Path | None = None,
) -> bool:
    """
    WHAT: Draws whether one gang follows the spare-account rotation rule using child 15 (gang_reactions).
    First draws the gang's adherence probability from world_ranges, then evaluates a Bernoulli trial.

    WHY: Amendment A-017 specifies two-stage fixed reaction drawing: a probability from world_ranges
    followed by an adherence decision per syndicate.

    Args:
        split: Dataset split name.
        rng: Random number generator dedicated to child 15 (gang_reactions).
        config: Optional configuration override.
        world_ranges_path: Optional path override for world_ranges.yaml.

    Returns:
        Boolean indicating whether the gang adheres to the rotation rule.
    """
    bounds = resolve_rotation_share_range(split, config, world_ranges_path)
    low_share, high_share = bounds[0], bounds[1]
    rule_prob = float(rng.uniform(low_share, high_share))

    # Second draw: Bernoulli trial against the drawn rule probability.
    decision_draw = float(rng.uniform(0.0, 1.0))
    return decision_draw < rule_prob


def resolve_split_share_range(
    split: str,
    config: dict[str, Any] | None = None,
    world_ranges_path: Path | None = None,
) -> tuple[float, float]:
    """
    WHAT: Resolves the probability range for adopting the amount-splitting rule on a given split.
    Dereferences world_ranges parameter gang_splits_amounts_share.

    WHY: Decouples rule adoption probabilities from hardcoded values, supporting fixture overrides.

    Args:
        split: Dataset split ('training', 'test', or 'stress').
        config: Optional configuration dictionary override.
        world_ranges_path: Optional path override for world_ranges.yaml.

    Returns:
        Tuple of (min_share, max_share).
    """
    if (
        config is not None
        and "parameters" in config
        and "gang_splits_amounts_share" in config["parameters"]
    ):
        param_entry = config["parameters"]["gang_splits_amounts_share"]
        if isinstance(param_entry, dict) and split in param_entry:
            bounds = param_entry[split]
            return float(bounds[0]), float(bounds[1])

    cfg = config if config is not None else load_gang_reactions_config()
    reaction_cfg = cfg.get("gang_reactions", cfg)
    rule_entry = reaction_cfg.get("gang_splits_amounts_share", {})

    if split in rule_entry and isinstance(rule_entry[split], (list, tuple)):
        bounds = rule_entry[split]
        return float(bounds[0]), float(bounds[1])

    ref_key = rule_entry.get("world_ranges_reference", "gang_splits_amounts_share")
    world_ranges_file = world_ranges_path if world_ranges_path is not None else WORLD_RANGES_PATH
    world_ranges = load_yaml(world_ranges_file)
    bounds = world_ranges["parameters"][ref_key][split]
    return float(bounds[0]), float(bounds[1])


def draw_split_flag(
    split: str,
    rng: np.random.Generator,
    config: dict[str, Any] | None = None,
    world_ranges_path: Path | None = None,
) -> bool:
    """
    WHAT: Draws whether one gang follows the amount-splitting rule using child 15 (gang_reactions).
    First draws the gang's adherence probability from world_ranges, then evaluates a Bernoulli trial.

    WHY: Amendment A-017 specifies two-stage fixed reaction drawing: a probability from world_ranges
    followed by an adherence decision per syndicate.

    Args:
        split: Dataset split name.
        rng: Random number generator dedicated to child 15 (gang_reactions).
        config: Optional configuration override.
        world_ranges_path: Optional path override for world_ranges.yaml.

    Returns:
        Boolean indicating whether the gang adheres to the splitting rule.
    """
    bounds = resolve_split_share_range(split, config, world_ranges_path)
    low_share, high_share = bounds[0], bounds[1]
    rule_prob = float(rng.uniform(low_share, high_share))

    # Second draw: Bernoulli trial against the drawn rule probability.
    decision_draw = float(rng.uniform(0.0, 1.0))
    return decision_draw < rule_prob


def resolve_split_threshold_range(
    split: str,
    config: dict[str, Any] | None = None,
    tactic_blocks_path: Path | None = None,
) -> tuple[int, int]:
    """
    WHAT: Resolves the minimum and maximum threshold in INR for payment splitting on a given split.
    Dereferences tactic_blocks building block split_below_threshold.parameters.threshold_inr.

    WHY: Decouples splitting thresholds from hardcoded values, supporting fixture overrides.

    Args:
        split: Dataset split ('training', 'test', or 'stress').
        config: Optional configuration dictionary override.
        tactic_blocks_path: Optional path override for tactic_blocks.yaml.

    Returns:
        Tuple of (min_threshold_inr, max_threshold_inr).
    """
    if (
        config is not None
        and "parameters" in config
        and "gang_split_threshold_inr" in config["parameters"]
    ):
        param_entry = config["parameters"]["gang_split_threshold_inr"]
        if isinstance(param_entry, dict) and split in param_entry:
            bounds = param_entry[split]
            return int(bounds[0]), int(bounds[1])

    cfg = config if config is not None else load_gang_reactions_config()
    reaction_cfg = cfg.get("gang_reactions", cfg)
    entry = reaction_cfg.get("gang_split_threshold_inr", {})

    if split in entry and isinstance(entry[split], (list, tuple)):
        bounds = entry[split]
        return int(bounds[0]), int(bounds[1])

    ref_key = entry.get("tactic_blocks_reference", "split_below_threshold.threshold_inr")
    tb_file = tactic_blocks_path if tactic_blocks_path is not None else TACTIC_BLOCKS_PATH
    tb_data = load_yaml(tb_file)
    block_key, param_key = ref_key.split(".", 1)
    block = tb_data["tactic_blocks"]["blocks"][block_key]
    bounds = block["parameters"][param_key][split]
    return int(bounds[0]), int(bounds[1])


def draw_split_threshold_paise(
    split: str,
    rng: np.random.Generator,
    config: dict[str, Any] | None = None,
    tactic_blocks_path: Path | None = None,
) -> int:
    """
    WHAT: Draws a whole-rupee splitting threshold uniformly in [min_inr, max_inr] and converts to paise.
    WHY: Evaluates threshold ceiling for amount splitting under child 15 (gang_reactions).

    Args:
        split: Dataset split name.
        rng: Random number generator dedicated to child 15 (gang_reactions).
        config: Optional configuration override.
        tactic_blocks_path: Optional path override for tactic_blocks.yaml.

    Returns:
        Integer threshold in paise units (whole rupees * 100).
    """
    bounds = resolve_split_threshold_range(split, config, tactic_blocks_path)
    low_inr, high_inr = bounds[0], bounds[1]
    if low_inr == high_inr:
        _ = rng.integers(0, 2)
        inr = low_inr
    else:
        inr = int(rng.integers(low_inr, high_inr + 1))
    return inr * 100


def update_gang_reaction(state: GangReactionState, observation: GangObservation) -> None:
    """
    WHAT: Updates a gang's mental reaction state upon receiving a delivered observation.
    Any delivered observation addressed to this gang flips has_seen_trouble to True.
    If the gang follows the stop rule and the observation reports a lien placed, marks the account stopped.

    WHY: Implements reactive syndicates: observing friction triggers vigilance (trouble observed)
    and abandons accounts known to have suffered government encumbrances under the stop rule.

    Args:
        state: Mutable GangReactionState for the observing syndicate.
        observation: Delivered GangObservation record.

    Returns:
        None. Modifies state in-place.
    """
    # Ignore telemetry addressed to other syndicates.
    if observation.gang_id != state.gang_id:
        return

    # Delivered observation addressed to this gang confirms external friction or trouble.
    state.has_seen_trouble = True

    # Under the stop rule, an observed lien halts all further operations on that account.
    if state.follows_stop_rule and observation.kind == "lien_placed":
        state.stopped_accounts.add(observation.account_id)


def choose_spare_account(
    state: GangReactionState,
    candidates: Sequence[str],
) -> str | None:
    """
    WHAT: Pure selection function returning the first candidate account not marked stopped.
    WHY: Implements account rotation: when a mule account is stopped, the syndicate reroutes incoming
    funds to an operational spare mule holding the same layer depth in the laundering structure.

    Args:
        state: GangReactionState containing current stopped accounts.
        candidates: Sequence of candidate account identifiers supplied by the caller in gang_holdings order.

    Returns:
        First candidate not present in state.stopped_accounts, or None if all candidates are stopped.
    """
    # Select the first viable candidate account not present in the stopped_accounts set.
    for candidate in candidates:
        if candidate not in state.stopped_accounts:
            return candidate
    return None


def may_move_money(state: GangReactionState, account_id: str) -> bool:
    """
    WHAT: Answers whether a gang is willing to move money through an account at the current moment.
    WHY: Pre-transaction validation consults this decision before executing payments or cash withdrawals.

    Args:
        state: GangReactionState for the executing syndicate.
        account_id: Account identifier proposed for transfer or withdrawal.

    Returns:
        True if the account is not stopped; False if marked stopped by the gang mind.
    """
    # Stopped accounts are strictly rejected by the gang mind.
    return account_id not in state.stopped_accounts

