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
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from core.config_validator import load_yaml
from world.gang_feed import WORLD_RANGES_PATH, GangObservation, load_gang_reactions_config


@dataclass
class GangReactionState:
    """
    WHAT: In-memory operational decision state for one criminal syndicate.
    Tracks whether the gang follows the stop-account rule and which accounts have been stopped.

    WHY: Ground rule A-017 requires per-gang reaction tracking driven purely by observable
    friction events, isolating gang intelligence from simulation ground truth.

    Args:
        gang_id: Unique identifier for the criminal syndicate.
        follows_stop_rule: Boolean flag indicating if this gang halts activity on liened accounts.
        stopped_accounts: Set of account identifiers marked unusable by this gang mind.

    Returns:
        An initialized GangReactionState instance.

    Example:
        GangReactionState(gang_id="g1", follows_stop_rule=True)
    """

    gang_id: str
    follows_stop_rule: bool
    stopped_accounts: set[str] = field(default_factory=set)


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
    # Allow fixture configs to force the flag directly.
    if config is not None and "follows_stop_rule" in config:
        return bool(config["follows_stop_rule"])

    bounds = resolve_stop_rule_share_range(split, config, world_ranges_path)
    low_share, high_share = bounds[0], bounds[1]
    if low_share == high_share:
        rule_prob = float(low_share)
    else:
        rule_prob = float(rng.uniform(low_share, high_share))

    # Second draw: Bernoulli trial against the drawn rule probability.
    decision_draw = float(rng.uniform(0.0, 1.0))
    return decision_draw < rule_prob


def update_gang_reaction(state: GangReactionState, observation: GangObservation) -> None:
    """
    WHAT: Updates a gang's mental reaction state upon receiving a delivered observation.
    If the gang follows the stop rule and the observation reports a lien placed, marks the account stopped.

    WHY: Implements the fixed stop rule: syndicates protect remaining assets by immediately abandoning
    accounts known to have suffered government encumbrances.

    Args:
        state: Mutable GangReactionState for the observing syndicate.
        observation: Delivered GangObservation record.

    Returns:
        None. Modifies state.stopped_accounts in-place.
    """
    # Ignore telemetry addressed to other syndicates.
    if observation.gang_id != state.gang_id:
        return

    # Under the stop rule, an observed lien halts all further operations on that account.
    if state.follows_stop_rule and observation.kind == "lien_placed":
        state.stopped_accounts.add(observation.account_id)


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
