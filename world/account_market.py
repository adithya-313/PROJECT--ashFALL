"""
WHAT: Account market resale planner simulating secondary circulation of rented mule accounts (Level 12d, G5).
Provides pure planning functions to validate market configuration, draw resale fractions, identify
reusable burned mule accounts, plan resales across gangs, and apply identity substitution to gang plans.

WHY: Police and intelligence reports document account suppliers reselling compromised bank account kits
to subsequent criminal syndicates once an initial campaign completes, causing syndicate networks to overlap.

TERM: account market = secondary marketplace where suppliers resell used rented mule accounts to later gangs.
TERM: resale offer = burned rented mule account from an earlier gang eligible for resale to subsequent gangs.
TERM: resale = substitution of a planned temporary account by an operational resold account from an earlier gang.
"""

from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

from world.gang_planner import GangPlan
from world.interactive_records import Resale
from world.truth_records import RoleSpan

ACCOUNT_MARKET_CONFIG_PATH: Path = Path("config/account_market.yaml")
WORLD_RANGES_PATH: Path = Path("config/world_ranges.yaml")
ALLOWED_SPLITS: tuple[str, ...] = ("training", "test", "stress")


def validate_account_market_config(config: dict[str, Any]) -> None:
    """
    WHAT: Validates that an account market configuration dictionary complies with structural constraints.
    Checks that the root dictionary and account_market block contain non-empty label, reason, issue IDs,
    and valid resale share range specifications.

    WHY: Fixture overrides and configuration files must pass identical validation before being applied
    to prevent malformed or unverified assumptions from driving simulation outcomes.

    Args:
        config: Dictionary containing account market configuration parameters.

    Raises:
        ValueError: If config is not a dict, required keys are missing, or numeric ranges are invalid.
    """
    if not isinstance(config, dict):
        raise ValueError("account_market config must be a dictionary.")

    cfg = config.get("account_market", config)
    if not isinstance(cfg, dict):
        raise ValueError("account_market section must be a dictionary.")

    label = cfg.get("label")
    if not label or not isinstance(label, str) or not label.strip():
        raise ValueError("account_market config must contain a non-empty 'label' attribute.")

    required_keys = (
        "derived_from",
        "reason",
        "issue_ids_v2",
        "share_of_accounts_sold_to_several_gangs",
    )
    for key in required_keys:
        if key not in cfg:
            raise ValueError(f"account_market config missing required key '{key}'.")

    share_entry = cfg["share_of_accounts_sold_to_several_gangs"]
    metadata_keys = ("label", "reason", "world_ranges_reference", "derived_from")

    if isinstance(share_entry, dict):
        for split, bounds in share_entry.items():
            if split in metadata_keys:
                continue
            if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                raise ValueError(f"resale share range for split '{split}' must be a pair [lower, upper].")
            lower, upper = bounds
            if (
                not isinstance(lower, (int, float))
                or not isinstance(upper, (int, float))
                or isinstance(lower, bool)
                or isinstance(upper, bool)
            ):
                raise ValueError(f"resale share bounds for split '{split}' must be numeric floats.")
            if not (0.0 <= lower <= 1.0) or not (0.0 <= upper <= 1.0) or lower > upper:
                raise ValueError(
                    f"resale share range for split '{split}' must be floats in [0, 1] "
                    f"with lower <= upper, got {bounds}."
                )
    elif isinstance(share_entry, (list, tuple)):
        if len(share_entry) != 2:
            raise ValueError("resale share range must be a pair [lower, upper].")
        lower, upper = share_entry
        if (
            not isinstance(lower, (int, float))
            or not isinstance(upper, (int, float))
            or isinstance(lower, bool)
            or isinstance(upper, bool)
        ):
            raise ValueError("resale share bounds must be numeric floats.")
        if not (0.0 <= lower <= 1.0) or not (0.0 <= upper <= 1.0) or lower > upper:
            raise ValueError(
                f"resale share range must be floats in [0, 1] with lower <= upper, got {share_entry}."
            )


def load_account_market_config(path: Path | None = None) -> dict[str, Any]:
    """
    WHAT: Reads config/account_market.yaml, returning its validated parsed dictionary.
    WHY: Isolates configuration file reading and path resolution from simulation planning routines.

    Args:
        path: Optional file path override for testing.

    Returns:
        Validated dictionary parsed from the account market YAML configuration file.
    """
    config_path = path if path is not None else ACCOUNT_MARKET_CONFIG_PATH
    if not config_path.is_file():
        raise ValueError(f"account market config file not found at '{config_path}'.")

    with config_path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"account market config at '{config_path}' must be a dictionary.")

    validate_account_market_config(data)
    return data


def resolve_resale_share_range(
    split: str,
    config: dict[str, Any] | None = None,
    world_ranges_path: Path | None = None,
) -> tuple[float, float]:
    """
    WHAT: Resolves the [lower, upper] interval for resale share under the specified evaluation split.
    Prefers explicit parameters from config when provided, otherwise resolves from config/world_ranges.yaml.

    WHY: Guarantees simulation parameter ranges align with calibrated research ranges while supporting
    deterministic test overrides.

    Args:
        split: Environment split name ('training', 'test', or 'stress').
        config: Optional configuration dictionary containing override ranges.
        world_ranges_path: Optional path override for world_ranges.yaml.

    Returns:
        Tuple of (lower, upper) float probabilities representing the valid draw interval.
    """
    if split not in ALLOWED_SPLITS:
        raise ValueError(f"Invalid split '{split}', must be one of {ALLOWED_SPLITS}.")

    if (
        config is not None
        and "parameters" in config
        and "share_of_accounts_sold_to_several_gangs" in config["parameters"]
    ):
        param_entry = config["parameters"]["share_of_accounts_sold_to_several_gangs"]
        if isinstance(param_entry, dict) and split in param_entry:
            bounds = param_entry[split]
            return float(bounds[0]), float(bounds[1])

    cfg = config if config is not None else load_account_market_config()
    market_cfg = cfg.get("account_market", cfg)
    share_entry = market_cfg.get("share_of_accounts_sold_to_several_gangs", {})

    if isinstance(share_entry, dict) and split in share_entry and isinstance(share_entry[split], (list, tuple)):
        bounds = share_entry[split]
        return float(bounds[0]), float(bounds[1])
    if isinstance(share_entry, (list, tuple)) and len(share_entry) == 2:
        return float(share_entry[0]), float(share_entry[1])

    ref_key = share_entry.get("world_ranges_reference", "share_of_accounts_sold_to_several_gangs")
    ranges_file = world_ranges_path if world_ranges_path is not None else WORLD_RANGES_PATH
    if not ranges_file.is_file():
        raise ValueError(f"world_ranges config not found at '{ranges_file}'.")

    with ranges_file.open(encoding="utf-8") as f:
        world_ranges = yaml.safe_load(f)

    parameters = world_ranges.get("parameters", world_ranges)
    if ref_key not in parameters or split not in parameters[ref_key]:
        raise ValueError(
            f"Key '{ref_key}.{split}' not found in '{ranges_file}'."
        )

    raw_bounds = parameters[ref_key][split]
    return float(raw_bounds[0]), float(raw_bounds[1])


def draw_resale_share(
    split: str,
    rng: np.random.Generator,
    config: dict[str, Any] | None = None,
) -> float:
    """
    WHAT: Draws the fraction of rented mule accounts resold to subsequent gangs for a simulation run.
    Always draws from the random number generator without skipping on degenerate bounds.

    WHY: Ground rule G5 requires a constant number of random draws decoupled from configuration bounds.

    Args:
        split: Environment split name ('training', 'test', or 'stress').
        rng: Random number generator dedicated to account market operations (child stream 16).
        config: Optional configuration dictionary containing override bounds.

    Returns:
        Drawn float resale probability in [0.0, 1.0].
    """
    bounds = resolve_resale_share_range(split, config)
    return float(rng.uniform(bounds[0], bounds[1]))


def find_resale_offers(earlier_roles: Sequence[RoleSpan]) -> list[RoleSpan]:
    """
    WHAT: Scans role spans from previously planned gangs to identify candidate mule accounts available for resale.
    Filters for spans with role 'mule', origin 'rented', burned_from not None, and valid_to None.

    WHY: Suppliers only resell accounts they control (rented) that have completed their original active run
    (burned) and remain open-ended (valid_to None) until reassigned to a subsequent syndicate.

    Args:
        earlier_roles: Sequence of role spans established by gangs planned earlier in the simulation.

    Returns:
        List of eligible RoleSpan records preserved in original role order.
    """
    offers: list[RoleSpan] = []
    for span in earlier_roles:
        if (
            span.role == "mule"
            and span.origin == "rented"
            and span.burned_from is not None
            and span.valid_to is None
        ):
            offers.append(span)
    return offers


def plan_resales(
    new_gang_roles: Sequence[RoleSpan],
    offers: Sequence[RoleSpan],
    share: float,
    rng: np.random.Generator,
) -> list[Resale]:
    """
    WHAT: Evaluates resale decisions for a new gang's rented mule accounts against available supplier offers.
    For each rented mule span in role order, draws a uniform random number. If draw < share, matches the first
    unused offer whose burned_from <= that span's valid_from.

    WHY: Replicates empirical supplier behavior where a fraction of rented accounts are recycled across gangs,
    ensuring every rented mule advances the random stream exactly once and offers are never reused.

    Args:
        new_gang_roles: Sequence of role spans planned for the new gang.
        offers: Sequence of candidate resale offers from earlier gangs.
        share: Drawn resale probability fraction governing buyer acceptance.
        rng: Random number generator dedicated to account market operations (child stream 16).

    Returns:
        List of Resale audit records detailing matched account transfers.
    """
    resales: list[Resale] = []
    used_offer_indices: set[int] = set()

    for span in new_gang_roles:
        if span.role != "mule" or span.origin != "rented":
            continue

        # Each rented mule takes exactly one draw whether or not an offer exists.
        draw = float(rng.uniform(0.0, 1.0))
        if draw < share:
            for idx, offer in enumerate(offers):
                if (
                    idx not in used_offer_indices
                    and offer.burned_from is not None
                    and offer.burned_from <= span.valid_from
                ):
                    used_offer_indices.add(idx)
                    resales.append(
                        Resale(
                            new_gang=span.gang_id,
                            new_account=span.account_id,
                            sold_account=offer.account_id,
                            previous_gang=offer.gang_id,
                            resale_minute=span.valid_from,
                        )
                    )
                    break

    return resales


def apply_resale(plan: GangPlan, resale: Resale) -> GangPlan:
    """
    WHAT: Replaces all occurrences of new_account with sold_account across every field in a GangPlan.
    Rewrites payment endpoints, truth role spans, device control spans, cash withdrawals, money rules,
    gang device events, and outside events.

    WHY: Embeds the resold account identity into the tactical plan so downstream simulation, execution,
    and forensic truth consistently reflect the physical reused account across the laundering chain.

    Args:
        plan: Existing immutable GangPlan scheduled for the new gang.
        resale: Resale audit record specifying replacement of new_account by sold_account.

    Returns:
        New GangPlan instance with all references to new_account substituted by sold_account.
    """
    old_acc = resale.new_account
    new_acc = resale.sold_account

    # 1. Planned payments (from_account, to_account)
    new_payments = []
    for p in plan.payments:
        from_a = new_acc if p.from_account == old_acc else p.from_account
        to_a = new_acc if p.to_account == old_acc else p.to_account
        if from_a != p.from_account or to_a != p.to_account:
            p = replace(p, from_account=from_a, to_account=to_a)
        new_payments.append(p)

    # 2. Truth roles (account_id)
    new_roles = []
    for r in plan.truth.roles:
        if r.account_id == old_acc:
            r = replace(r, account_id=new_acc)
        new_roles.append(r)

    # 3. Truth device control (account_id)
    new_device_control = []
    for d in plan.truth.device_control:
        if d.account_id == old_acc:
            d = replace(d, account_id=new_acc)
        new_device_control.append(d)

    # 4. Gang device events (account_id)
    new_gang_device_events = []
    for dev_ev in plan.gang_device_events:
        if dev_ev.account_id == old_acc:
            dev_ev = replace(dev_ev, account_id=new_acc)
        new_gang_device_events.append(dev_ev)

    # 5. Money rules (requires_credit_from)
    new_money_rules = []
    for m in plan.money_rules:
        if m.requires_credit_from == old_acc:
            m = replace(m, requires_credit_from=new_acc)
        new_money_rules.append(m)

    # 6. Outside events (account_id)
    new_outside_events = []
    for out_ev in plan.outside_events:
        if out_ev.account_id == old_acc:
            out_ev = replace(out_ev, account_id=new_acc)
        new_outside_events.append(out_ev)

    # 7. Cash withdrawals (account_id)
    new_withdrawals = []
    for w in plan.withdrawals:
        if w.account_id == old_acc:
            w = replace(w, account_id=new_acc)
        new_withdrawals.append(w)

    new_truth = replace(
        plan.truth,
        roles=tuple(new_roles),
        device_control=tuple(new_device_control),
    )

    return replace(
        plan,
        payments=tuple(new_payments),
        truth=new_truth,
        gang_device_events=tuple(new_gang_device_events),
        money_rules=tuple(new_money_rules),
        outside_events=tuple(new_outside_events),
        withdrawals=tuple(new_withdrawals),
    )
