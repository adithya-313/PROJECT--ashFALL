# File: showcase/level_16_world.py
# Purpose: Level 16 showcase - the committed world, its pins, and the guarantee counts.
"""
WHAT: Loads config/showcase_level_16.yaml, builds the pinned world configs and the gang, plans (runs) the world and
counts the five things the Level 16 checks need: a bulk confirmation, a cross-bank complaint, a recycled number listed
before day 0, one account with complaints from several victims over weeks (G3), and a falsely reported account.

WHY: One module defines the showcase world, so the seed scan, the unit test and the checks all see the same world.

TERM: guarantee: something the showcase world must contain, with a floor in config/showcase_level_16.yaml.
TERM: pin: a world_ranges parameter set to one value for this showcase world only.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from core.config_validator import load_yaml
from core.event_clock import ManualClock
from nationalsim.keyed_hash import compute_identifier_hash
from replay.registry_driver import build_registry_driver
from replay.registry_replay import RegistryReplaySettings, load_registry_replay_settings
from world.edge_cases import apply_edge_case_layer
from world.fast_mode import FastRunSettings
from world.gang_config import GangConfig, gang_config_from_plain
from world.interactive_records import RunnerSettings
from world.interactive_runner import run_interactive_world
from world.label_builder import build_bulk_confirmations, build_label_events
from world.phone_book import build_phone_book
from world.replay_recording import RecordedWorld
from world.world_configs import WorldConfigs, read_world_configs

__all__ = [
    "GuaranteeCounts",
    "load_level_16_config",
    "level_16_world_configs",
    "level_16_gang",
    "level_16_registry_settings",
    "level_16_fast_settings",
    "plan_level_16_world",
    "guarantee_counts",
    "unmet_guarantees",
]

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config" / "showcase_level_16.yaml"
GANG_PATH = REPO_ROOT / "showcase" / "fixtures" / "level_16" / "gang_stacking.yaml"
MINUTES_PER_DAY = 1440


@dataclass(frozen=True)
class GuaranteeCounts:
    """
    WHAT: The measured counts of the five guarantees in one planned world.
    Args:
        bulk_confirmations_with_accounts: Crackdowns that confirmed at least one account.
        cross_bank_complaints: Complaints whose reported account sits at another bank than the complainant.
        recycled_numbers_listed: Recycled phone numbers with a pre-world registry listing.
        g3_complaints: Most complaints naming one account.
        g3_victims: Distinct complainants on that account.
        g3_spread_days: Days between the first and last complaint on that account.
        false_reported_accounts: Distinct reported accounts that hold no mule-side role.
        hold_requests: Hold requests that reach a receiving bank (cross-bank complaints whose hold is not missed).
        innocent_holder_listings: Pre-world listings whose recycled phone now belongs to a non-mule account at a
            DPIP participant bank, so the bank labels an innocent holder (N1).
    Example:
        GuaranteeCounts(1, 2, 1, 4, 4, 20.5, 1, 3, 1)
    """

    bulk_confirmations_with_accounts: int
    cross_bank_complaints: int
    recycled_numbers_listed: int
    g3_complaints: int
    g3_victims: int
    g3_spread_days: float
    false_reported_accounts: int
    hold_requests: int
    innocent_holder_listings: int


def load_level_16_config(path: Path | None = None) -> dict[str, Any]:
    """
    WHAT: Reads config/showcase_level_16.yaml.
    WHY: The seed, sizes, pins and floors live in the file, not in code.
    Args:
        path: Optional other file (tests).
    Returns:
        The parsed dictionary.
    Example:
        load_level_16_config()["world"]["seed"]
    """
    loaded = load_yaml(path or CONFIG_PATH)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path or CONFIG_PATH} must hold a mapping")
    return loaded


def level_16_world_configs(cfg: dict[str, Any]) -> WorldConfigs:
    """
    WHAT: Reads the four world configs and applies the showcase pins through the edge-case layer.
    WHY: The pins reach the world through world config only; nothing is patched later.
    Args:
        cfg: The parsed showcase config.
    Returns:
        The pinned WorldConfigs.
    Example:
        level_16_world_configs(load_level_16_config())
    """
    layer = {"world_ranges": {name: pin["values"] for name, pin in cfg["pins"].items()}}
    return apply_edge_case_layer(read_world_configs(), layer)


def level_16_gang() -> GangConfig:
    """
    WHAT: Loads the showcase's scam gang fixture.
    WHY: The gang is what makes mule accounts, victims and complaints in the showcase world.
    Returns:
        The GangConfig.
    Example:
        level_16_gang().family
    """
    return gang_config_from_plain(load_yaml(GANG_PATH))


def level_16_registry_settings(cfg: dict[str, Any]) -> RegistryReplaySettings:
    """
    WHAT: The registry replay settings of the showcase: the file settings for the split, with the banks named under
    registry.non_participants switched to dpip_participation false.
    WHY: Check 9 needs a bank that does not take part in DPIP; config/nationalsim.yaml enrols every bank and is not
    edited, so the showcase replay passes its own settings (showcase-only).
    Args:
        cfg: The parsed showcase config.
    Returns:
        RegistryReplaySettings with the non-participants switched off.
    Example:
        level_16_registry_settings(cfg).participants == ["bank_0", "bank_1"]
    """
    base = load_registry_replay_settings(str(cfg["world"]["range_name"]))
    off = set(cfg["registry"]["non_participants"])
    banks = {b: ({**s, "dpip_participation": False} if b in off else dict(s)) for b, s in base.bank_settings.items()}
    return replace(base, bank_settings=banks)


def level_16_fast_settings(cfg: dict[str, Any], seed: int | None = None) -> FastRunSettings:
    """
    WHAT: Builds the FastRunSettings of the showcase world (optionally for another seed, for the scan).
    Args:
        cfg: The parsed showcase config.
        seed: Seed override.
    Returns:
        FastRunSettings(seed, people, shops, days, range).
    Example:
        level_16_fast_settings(cfg, 7)
    """
    w = cfg["world"]
    return FastRunSettings(int(w["seed"] if seed is None else seed), int(w["n_people"]), int(w["n_shops"]),
                           int(w["n_days"]), str(w["range_name"]))


def plan_level_16_world(cfg: dict[str, Any], seed: int | None = None) -> RecordedWorld:
    """
    WHAT: Runs the pinned world for the seed with fraud on and the showcase gang, and returns it as a RecordedWorld
    with no recorded decision calls (enough to count the guarantees; no replay and no bank connectors).
    WHY: The scan and the unit test only need the planned world, which is cheap.
    Args:
        cfg: The parsed showcase config.
        seed: Seed override.
    Returns:
        A RecordedWorld with an empty calls tuple and a phone book.
    Example:
        plan_level_16_world(cfg).output.result.run_id
    """
    settings = level_16_fast_settings(cfg, seed)
    gang = level_16_gang()
    output = run_interactive_world(
        settings, RunnerSettings(fraud_enabled=True), configs=level_16_world_configs(cfg), gang_configs=(gang,),
        world_split=settings.range_name,
    )
    return RecordedWorld(
        output=output, calls=(), world_id=output.result.run_id, split=settings.range_name, seed=settings.seed,
        gang_families=(gang.family,), phone_book=build_phone_book(output, run_seed=settings.seed),
    )


def guarantee_counts(world: RecordedWorld, registry: RegistryReplaySettings | None = None) -> GuaranteeCounts:
    """
    WHAT: Counts the five guarantees in a planned world.
    WHY: Truth is read from the in-memory world only to say which reported accounts hold no mule-side role.
    Args:
        world: A planned or recorded world.
        registry: The showcase registry settings (default: the config file settings).
    Returns:
        GuaranteeCounts.
    Example:
        guarantee_counts(plan_level_16_world(cfg)).cross_bank_complaints
    """
    output = world.output
    complaints = [e for e in build_label_events(output) if e.kind == "complaint"]
    bulk = [c for c in build_bulk_confirmations(output).confirmations if c.account_ids]
    mule_roles = {"mule", "collector", "cash_out"}
    mule_accounts = {s.account_id for s in output.result.truth.roles if s.role in mule_roles}
    by_account: dict[str, list[Any]] = defaultdict(list)
    for e in complaints:
        by_account[e.reported_account].append(e)
    best = max(by_account.values(), key=lambda v: (len(v), len({e.complainant_account for e in v})), default=[])
    spread = 0.0
    if best:
        spread = (max(e.arrival_minute for e in best) - min(e.arrival_minute for e in best)) / MINUTES_PER_DAY
    driver = build_registry_driver(world, ManualClock(), registry)
    participants = set((registry or load_registry_replay_settings(str(world.split))).participants)
    bank_of = {acc: f"bank_{ledger.bank_index}" for ledger in output.world.ledgers for acc in ledger.accounts}
    phone_hashes = {
        compute_identifier_hash(driver.key, output.result.run_id, "phone", r.phone_number): r
        for r in (world.phone_book.records.values() if world.phone_book else ())
        if r.is_recycled
    }
    innocent = 0
    for listing in driver.seed:
        holder = phone_hashes.get(listing.item.identifier_hash)
        if holder and holder.account_id not in mule_accounts and bank_of.get(holder.account_id) in participants:
            innocent += 1
    return GuaranteeCounts(
        bulk_confirmations_with_accounts=len(bulk),
        cross_bank_complaints=sum(1 for e in complaints if e.complainant_bank_index != e.reported_bank_index),
        recycled_numbers_listed=len(driver.seed),
        g3_complaints=len(best),
        g3_victims=len({e.complainant_account for e in best}),
        g3_spread_days=spread,
        false_reported_accounts=len({a for a in by_account if a not in mule_accounts}),
        hold_requests=len(driver.holds),
        innocent_holder_listings=innocent,
    )


def unmet_guarantees(counts: GuaranteeCounts, cfg: dict[str, Any]) -> list[str]:
    """
    WHAT: Names every guarantee whose count is below its floor in the config.
    Args:
        counts: Measured counts.
        cfg: The parsed showcase config.
    Returns:
        Names of unmet guarantees (empty when all are met).
    Example:
        unmet_guarantees(counts, cfg) == []
    """
    m = cfg["minimums"]
    checks = {
        "bulk_confirmations_with_accounts":
            counts.bulk_confirmations_with_accounts >= m["bulk_confirmations_with_accounts"],
        "cross_bank_complaints": counts.cross_bank_complaints >= m["cross_bank_complaints"],
        "recycled_numbers_listed_before_day_0":
            counts.recycled_numbers_listed >= m["recycled_numbers_listed_before_day_0"],
        "g3_complaints_on_one_account": counts.g3_complaints >= m["g3_complaints_on_one_account"],
        "g3_spread_days": counts.g3_spread_days >= m["g3_spread_days"],
        "false_reported_accounts": counts.false_reported_accounts >= m["false_reported_accounts"],
        "hold_requests": counts.hold_requests >= m["hold_requests"],
        "innocent_holder_listings": counts.innocent_holder_listings >= m["innocent_holder_listings"],
    }
    return [name for name, ok in checks.items() if not ok]
