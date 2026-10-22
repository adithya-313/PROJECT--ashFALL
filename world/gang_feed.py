"""
WHAT: Gang observation feed and holding records for Level 12d reactive criminal syndicates.
Defines immutable representations for account holdings and external event observations,
utility functions to resolve which gang controls an account at any simulation minute,
and random draw resolvers for observation delivery delay and event loss rates.

WHY: Criminal syndicates operate under realistic market frictions: they cannot inspect
bank ledgers or ground truth. Instead, gangs receive an imperfect, time-delayed observation
feed of external friction (liens, holds, and declined payments) touching accounts they hold.

TERM: gang holding: a temporal interval [valid_from, valid_to) during which a gang controls an account.
TERM: gang observation: a delayed and lossy notification delivered to a gang about friction on its accounts.
TERM: miss share: probability that an external friction event is dropped and never delivered to the gang.
TERM: observation delay: elapsed simulated minutes between when an event happened and when the gang sees it.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core.config_validator import load_yaml

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
GANG_REACTIONS_CONFIG_PATH: Path = REPO_ROOT / "config" / "gang_reactions.yaml"
WORLD_RANGES_PATH: Path = REPO_ROOT / "config" / "world_ranges.yaml"

# Valid observation categories that may be delivered through a gang observation feed.
OBSERVATION_KINDS: tuple[str, ...] = (
    "lien_placed",
    "debit_hold_placed",
    "released",
    "payment_declined",
    "payment_held",
    "payment_blocked_by_hold",
)


@dataclass(frozen=True)
class GangHolding:
    """
    WHAT: Represents a temporal span of operational control by a gang over an account.
    Captures gang identifier, account identifier, and half-open validity window [valid_from, valid_to).

    WHY: Ground rule A-017 dictates that gangs only receive feed items concerning accounts they
    currently hold. Victim accounts are never held by gangs.

    Args:
        gang_id: Identifier of the gang holding operational control.
        account_id: Identifier of the held account.
        valid_from: Starting minute of control (inclusive, >= 0).
        valid_to: Ending minute of control (exclusive, or None if unending).

    Returns:
        A frozen GangHolding record.

    Example:
        GangHolding("gang_alpha", "acc_person_000001", 1440, 2880)
    """

    gang_id: str
    account_id: str
    valid_from: int
    valid_to: int | None

    def __post_init__(self) -> None:
        """
        WHAT: Validates temporal validity and non-empty identity fields.
        WHY: Rejects corrupted holdings where valid_to <= valid_from or identifiers are empty.
        """
        if not self.gang_id:
            raise ValueError("GangHolding gang_id must be a non-empty string.")
        if not self.account_id:
            raise ValueError("GangHolding account_id must be a non-empty string.")
        if self.valid_from < 0:
            raise ValueError(f"GangHolding valid_from ({self.valid_from}) cannot be negative.")
        if self.valid_to is not None and self.valid_to <= self.valid_from:
            raise ValueError(
                f"GangHolding valid_to ({self.valid_to}) must be greater than valid_from ({self.valid_from})."
            )

    def covers(self, minute: int) -> bool:
        """
        WHAT: Checks whether this holding span covers the given simulation minute.
        WHY: Used by finding and filtering routines to verify active control at a specific time.

        Args:
            minute: Integer simulation minute to check.

        Returns:
            True if valid_from <= minute and (valid_to is None or minute < valid_to).
        """
        # Half-open temporal interval check [valid_from, valid_to).
        return self.valid_from <= minute and (self.valid_to is None or minute < self.valid_to)


@dataclass(frozen=True)
class GangObservation:
    """
    WHAT: One observation item queued for or delivered to a gang's decision-making state.
    Specifies target gang, account involved, friction event kind, when it happened, and when seen.

    WHY: Telemetry delivery is separated from occurrence by an observation delay, reflecting
    real-world information latency before a gang learns an account has been encumbered or declined.

    Args:
        gang_id: Identifier of the observing gang.
        account_id: Account experiencing the external friction event.
        kind: Category of external friction (one of OBSERVATION_KINDS).
        happened_minute: Simulation minute when the event actually occurred.
        seen_minute: Simulation minute when the observation reaches the gang.

    Returns:
        A frozen GangObservation record.

    Example:
        GangObservation("g1", "acc_1", "lien_placed", 100, 145)
    """

    gang_id: str
    account_id: str
    kind: str
    happened_minute: int
    seen_minute: int

    def __post_init__(self) -> None:
        """
        WHAT: Validates that observation kind is one of the authorized categories.
        WHY: Rejects uncatalogued friction events early to prevent corrupted state updates.
        """
        if self.kind not in OBSERVATION_KINDS:
            raise ValueError(
                f"GangObservation kind '{self.kind}' is invalid; must be one of {OBSERVATION_KINDS}."
            )
        if self.seen_minute < self.happened_minute:
            raise ValueError(
                f"GangObservation seen_minute ({self.seen_minute}) cannot precede "
                f"happened_minute ({self.happened_minute})."
            )


@dataclass(frozen=True)
class GangFeedSettings:
    """
    WHAT: Per-gang observation feed parameters drawn at simulation initialization.
    Encapsulates observation latency in minutes and observation packet loss share.

    WHY: Each syndicate has distinct operational vigilance, drawn once at startup
    from empirical world ranges without draw-order entanglement across gangs.

    Args:
        delay_minutes: Integer minutes of delay before an observation is delivered.
        miss_share: Float probability [0.0, 1.0] that an observation is dropped.

    Returns:
        A frozen GangFeedSettings record.
    """

    delay_minutes: int
    miss_share: float


def validate_gang_reactions_config(config: dict[str, Any]) -> None:
    """
    WHAT: Validates that gang reactions configuration complies with schema constraints.
    WHY: Enforces that fixture overrides pass identical validation to file configs on disk.

    Args:
        config: Gang reactions configuration dictionary to validate.

    Raises:
        ValueError: If required keys are missing, values fall outside valid bounds, or label is missing.
    """
    if not isinstance(config, dict):
        raise ValueError("gang_reactions_config must be a dictionary.")

    cfg = config.get("gang_reactions", config)
    if not isinstance(cfg, dict):
        raise ValueError("gang_reactions section must be a dictionary.")

    label = cfg.get("label")
    if not label or not isinstance(label, str) or not label.strip():
        raise ValueError("gang_reactions config must contain a non-empty 'label' attribute.")

    required_keys = (
        "derived_from",
        "reason",
        "issue_ids_v2",
        "gang_feed_delay_minutes",
        "gang_feed_miss_share",
        "gang_stops_using_liened_account_share",
    )
    for key in required_keys:
        if key not in cfg:
            raise ValueError(f"gang_reactions config missing required key '{key}'.")

    metadata_keys = ("label", "reason", "world_ranges_reference", "derived_from")

    # Validate delay entries if explicit bounds are supplied
    delay_entry = cfg["gang_feed_delay_minutes"]
    if isinstance(delay_entry, dict):
        for split, bounds in delay_entry.items():
            if split in metadata_keys:
                continue
            if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                raise ValueError(f"delay range for split '{split}' must be a pair [lower, upper].")
            lower, upper = bounds
            if not isinstance(lower, int) or not isinstance(upper, int) or lower < 0 or lower > upper:
                raise ValueError(
                    f"delay range for split '{split}' must be non-negative integers with lower <= upper, got {bounds}."
                )

    # Validate miss share entries if explicit bounds are supplied
    miss_entry = cfg["gang_feed_miss_share"]
    if isinstance(miss_entry, dict):
        for split, bounds in miss_entry.items():
            if split in metadata_keys:
                continue
            if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                raise ValueError(f"miss share range for split '{split}' must be a pair [lower, upper].")
            lower, upper = bounds
            if not (0.0 <= lower <= 1.0) or not (0.0 <= upper <= 1.0) or lower > upper:
                raise ValueError(
                    f"miss share range for split '{split}' must be floats in [0, 1] with lower <= upper, got {bounds}."
                )

    # Validate stop rule share entries if explicit bounds are supplied
    stop_entry = cfg["gang_stops_using_liened_account_share"]
    if isinstance(stop_entry, dict):
        for split, bounds in stop_entry.items():
            if split in metadata_keys:
                continue
            if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                raise ValueError(f"stop rule share range for split '{split}' must be a pair [lower, upper].")
            lower, upper = bounds
            if not (0.0 <= lower <= 1.0) or not (0.0 <= upper <= 1.0) or lower > upper:
                raise ValueError(
                    f"stop rule share range for split '{split}' must be floats in [0, 1] "
                    f"with lower <= upper, got {bounds}."
                )


def load_gang_reactions_config(path: Path | None = None) -> dict[str, Any]:
    """
    WHAT: Reads config/gang_reactions.yaml, returning its parsed dictionary.
    WHY: Isolates configuration file reading and path resolution from simulation routines.

    Args:
        path: Optional file path override for testing.

    Returns:
        Parsed and validated configuration dictionary.
    """
    resolved_path = path if path is not None else GANG_REACTIONS_CONFIG_PATH
    cfg = load_yaml(resolved_path)
    validate_gang_reactions_config(cfg)
    return cfg


def resolve_feed_delay_range(
    split: str,
    config: dict[str, Any] | None = None,
    world_ranges_path: Path | None = None,
) -> tuple[int, int]:
    """
    WHAT: Resolves the minimum and maximum observation feed delay minutes for a dataset split.
    Reads from config/gang_reactions.yaml and dereferences config/world_ranges.yaml.

    WHY: Decouples simulation code from hardcoded parameter bounds, supporting test fixtures.

    Args:
        split: Dataset split ('training', 'test', or 'stress').
        config: Optional configuration dictionary override.
        world_ranges_path: Optional path override for world_ranges.yaml.

    Returns:
        Tuple of (min_delay_minutes, max_delay_minutes).
    """
    if config is not None and "parameters" in config and "gang_feed_delay_minutes" in config["parameters"]:
        param_entry = config["parameters"]["gang_feed_delay_minutes"]
        if isinstance(param_entry, dict) and split in param_entry:
            bounds = param_entry[split]
            return int(bounds[0]), int(bounds[1])

    cfg = config if config is not None else load_gang_reactions_config()
    reaction_cfg = cfg.get("gang_reactions", cfg)
    delay_entry = reaction_cfg.get("gang_feed_delay_minutes", {})

    # If bounds are supplied directly in config, use them.
    if split in delay_entry and isinstance(delay_entry[split], (list, tuple)):
        bounds = delay_entry[split]
        return int(bounds[0]), int(bounds[1])

    ref_key = delay_entry.get("world_ranges_reference", "gang_feed_delay_minutes")
    world_ranges_file = world_ranges_path if world_ranges_path is not None else WORLD_RANGES_PATH
    world_ranges = load_yaml(world_ranges_file)
    bounds = world_ranges["parameters"][ref_key][split]
    return int(bounds[0]), int(bounds[1])


def resolve_feed_miss_share_range(
    split: str,
    config: dict[str, Any] | None = None,
    world_ranges_path: Path | None = None,
) -> tuple[float, float]:
    """
    WHAT: Resolves the observation feed packet miss share bounds [min_share, max_share] for a split.
    Dereferences world_ranges parameter gang_feed_miss_share.

    WHY: Guarantees loss rate ranges originate from empirically catalogued configuration.

    Args:
        split: Dataset split ('training', 'test', or 'stress').
        config: Optional configuration dictionary override.
        world_ranges_path: Optional path override for world_ranges.yaml.

    Returns:
        Tuple of (min_miss_share, max_miss_share).
    """
    if config is not None and "parameters" in config and "gang_feed_miss_share" in config["parameters"]:
        param_entry = config["parameters"]["gang_feed_miss_share"]
        if isinstance(param_entry, dict) and split in param_entry:
            bounds = param_entry[split]
            return float(bounds[0]), float(bounds[1])

    cfg = config if config is not None else load_gang_reactions_config()
    reaction_cfg = cfg.get("gang_reactions", cfg)
    miss_entry = reaction_cfg.get("gang_feed_miss_share", {})

    if split in miss_entry and isinstance(miss_entry[split], (list, tuple)):
        bounds = miss_entry[split]
        return float(bounds[0]), float(bounds[1])

    ref_key = miss_entry.get("world_ranges_reference", "gang_feed_miss_share")
    world_ranges_file = world_ranges_path if world_ranges_path is not None else WORLD_RANGES_PATH
    world_ranges = load_yaml(world_ranges_file)
    bounds = world_ranges["parameters"][ref_key][split]
    return float(bounds[0]), float(bounds[1])


def draw_gang_feed_settings(
    split: str,
    rng: np.random.Generator,
    config: dict[str, Any] | None = None,
    world_ranges_path: Path | None = None,
) -> GangFeedSettings:
    """
    WHAT: Draws observation latency (whole minutes) and miss share for one gang from child 14.
    WHY: Amendment A-017 requires drawing latency uniformly in minutes and miss share as a fraction.

    Args:
        split: Simulation split name.
        rng: Random number generator dedicated to child 14 (gang_observation_feed).
        config: Optional configuration override.
        world_ranges_path: Optional world ranges file path override.

    Returns:
        Instantiated GangFeedSettings with drawn parameters.
    """
    delay_bounds = resolve_feed_delay_range(split, config, world_ranges_path)
    miss_bounds = resolve_feed_miss_share_range(split, config, world_ranges_path)

    # Delay is an integer number of whole minutes drawn uniformly across the range bounds.
    low_delay, high_delay = delay_bounds[0], delay_bounds[1]
    if low_delay == high_delay:
        delay = low_delay
    else:
        delay = int(rng.integers(low_delay, high_delay + 1))

    # Miss share is a continuous float fraction in [low_miss, high_miss].
    low_miss, high_miss = miss_bounds[0], miss_bounds[1]
    if low_miss == high_miss:
        miss_share = float(low_miss)
    else:
        miss_share = float(rng.uniform(low_miss, high_miss))

    return GangFeedSettings(delay_minutes=delay, miss_share=miss_share)


def find_holding_gang(
    holdings: Sequence[GangHolding],
    account_id: str,
    minute: int,
) -> str | None:
    """
    WHAT: Determines which criminal syndicate holds operational control of an account at a given minute.
    WHY: Feed dispatchers consult this check to direct external friction telemetry to the proper gang.

    Args:
        holdings: Sequence of active and planned GangHolding records.
        account_id: Account identifier to inspect.
        minute: Current simulation minute.

    Returns:
        String gang_id if an active holding covers the account, otherwise None.
    """
    # Linear scan across configured gang holdings for a temporal match.
    for holding in holdings:
        if holding.account_id == account_id and holding.covers(minute):
            return holding.gang_id
    return None


def build_gang_holdings_from_roles(roles: Sequence[Any]) -> tuple[GangHolding, ...]:
    """
    WHAT: Constructs GangHolding records from truth role spans for mules, collectors, and cash-outs.
    Victim role spans are strictly excluded.

    WHY: Gangs hold and monitor their own money-moving infrastructure; victim accounts remain
    outside gang ownership and do not produce gang telemetry.

    Args:
        roles: Sequence of RoleSpan instances from planned gang campaigns.

    Returns:
        Tuple of initialized GangHolding records.
    """
    holdings: list[GangHolding] = []
    # Only mule infrastructure generates gang ownership; victims are excluded by definition.
    for r in roles:
        if getattr(r, "role", None) in ("mule", "collector", "cash_out"):
            holdings.append(
                GangHolding(
                    gang_id=r.gang_id,
                    account_id=r.account_id,
                    valid_from=r.valid_from,
                    valid_to=r.valid_to,
                )
            )
    return tuple(holdings)
