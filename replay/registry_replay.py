"""
WHAT: Plans what the national registry receives during a replay: the pre-world seed (old listings of phones that were
later recycled to innocent people) and the in-world reports banks make, with the minute each entry becomes visible.

WHY: Replay drives a real in-process RegistryCore. Planning is pure (no I/O) so it can be tested without the stack,
and it uses its own random stream so no other world draw moves.

TERM: seed listing: a registry entry made before day 0 for the phone of a previous (mule) holder; the phone is later
      recycled to an innocent person, so the entry matches them.
TERM: visible minute: when an entry can be seen by banks: ingest time plus the listing delay, never before day 0.
TERM: DPIP participant: a bank that shares and reads registry entries (config/nationalsim.yaml banks).
"""

import hashlib
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from core.config_validator import load_yaml
from core.event_clock import IST
from nationalsim.adapter import DEFAULT_FRAUD_REASON_CODE, IDENTIFIER_TYPE_PHONE, IngestionItem
from nationalsim.keyed_hash import compute_identifier_hash
from world.phone_book import PhoneBook

__all__ = [
    "REGISTRY_INGEST",
    "REGISTRY_VISIBLE",
    "RegistryReplaySettings",
    "RegistryListing",
    "BulkEvent",
    "load_registry_replay_settings",
    "replay_dpip_key",
    "plan_registry_seed",
    "visible_world_time",
]

REGISTRY_INGEST = "registry_ingest"
REGISTRY_VISIBLE = "registry_visible"
NATIONALSIM_PATH = Path(__file__).resolve().parents[1] / "config" / "nationalsim.yaml"
REPORTER_CHOICE_UNIFORM = "uniform_over_dpip_participants"


@dataclass(frozen=True)
class RegistryReplaySettings:
    """
    WHAT: The registry numbers a replay needs from config/nationalsim.yaml.
    WHY: One validated place for the listing delay, bank enrolment and pre-world seed assumptions.
    Args:
        listing_delay: Time between ingest and visibility.
        dispute_resolution_delay: Time to uphold a dispute.
        bank_settings: bank id -> settings (dpip_participation).
        seed_share_range: (low, high) of the pre-world listing share for the run's split.
        reporter_choice: How the seed reporter is chosen; only uniform over participants is supported.
        screen_delay_ms: (low, high) of the per-question screen delay (config screen_added_delay_ms).
    Example:
        RegistryReplaySettings(timedelta(minutes=15), timedelta(minutes=60), {"bank_0": {...}}, (0.2, 0.6), "x")
    """

    listing_delay: timedelta
    dispute_resolution_delay: timedelta
    bank_settings: dict[str, dict[str, Any]]
    seed_share_range: tuple[float, float]
    reporter_choice: str
    screen_delay_ms: tuple[float, float] = (2.0, 15.0)

    @property
    def participants(self) -> list[str]:
        """Sorted ids of banks with dpip_participation true."""
        return sorted(b for b, s in self.bank_settings.items() if s.get("dpip_participation") is True)


@dataclass(frozen=True)
class RegistryListing:
    """
    WHAT: One planned registry entry: the ingestion item and the world time it becomes visible.
    WHY: Replay turns each listing into a registry_visible step at visible_time.
    Args:
        item: The pre-hashed ingestion item.
        visible_time: Aware IST time banks can first see the entry.
        seed: True for a pre-world seed listing.
    Example:
        RegistryListing(item, visible_time, True)
    """

    item: IngestionItem
    visible_time: datetime
    seed: bool


@dataclass(frozen=True)
class BulkEvent:
    """
    WHAT: One police crackdown confirmation of one account, as a registry.events event for the holding bank.
    WHY: A police bulk confirmation is not a DPIP report: it goes only to the bank that holds the account, whether or
    not that bank takes part in DPIP, so it is staged as an event and never enters the registry's listings.
    Args:
        item: Ingestion item (account type, the account's hash with its own bank index, reason police_bulk).
        visible_time: Aware IST time the confirmation arrives (the crackdown's arrival minute).
        batch_id: The crackdown's batch id.
        region: The crackdown's region.
    Example:
        BulkEvent(item, arrival, "r-B000001", 2)
    """

    item: IngestionItem
    visible_time: datetime
    batch_id: str
    region: int


def replay_dpip_key(run_id: str) -> str:
    """
    WHAT: Returns the DPIP key a replay uses, derived from the run id.
    WHY: The registry feed and every bank-side consumer must hash with the same key; a replay is synthetic, so a
    key derived from the run id is reproducible and holds no secret.
    Args:
        run_id: The world's run id.
    Returns:
        A 64-character hex key.
    Example:
        replay_dpip_key("r1") == replay_dpip_key("r1")
    """
    return hashlib.sha256(f"ashfall-replay-dpip|{run_id}".encode()).hexdigest()


def load_registry_replay_settings(split: str, path: Path | None = None) -> RegistryReplaySettings:
    """
    WHAT: Reads the registry replay settings for a split from config/nationalsim.yaml.
    WHY: Refuses a missing value or split instead of falling back silently.
    Args:
        split: Range split name (training, test or stress).
        path: Optional config path.
    Returns:
        The validated settings.
    Raises:
        ValueError: If a value, the split range or the reporter choice is missing or unsupported.
    Example:
        load_registry_replay_settings("training").listing_delay == timedelta(minutes=15)
    """
    block = load_yaml(path if path is not None else NATIONALSIM_PATH)["nationalsim"]
    share_entry = block.get("pre_world_listing_share")
    if not isinstance(share_entry, dict) or split not in share_entry:
        raise ValueError(f"pre_world_listing_share has no range for split '{split}' in config/nationalsim.yaml")
    low, high = (float(x) for x in share_entry[split])
    if not 0.0 <= low <= high <= 1.0:
        raise ValueError(f"pre_world_listing_share for '{split}' must satisfy 0 <= low <= high <= 1")
    choice = str(block.get("pre_world_reporter_choice", {}).get("value"))
    if choice != REPORTER_CHOICE_UNIFORM:
        raise ValueError(f"pre_world_reporter_choice must be '{REPORTER_CHOICE_UNIFORM}', got '{choice}'")
    settings = RegistryReplaySettings(
        listing_delay=timedelta(minutes=int(block["listing_delay_minutes"]["value"])),
        dispute_resolution_delay=timedelta(minutes=int(block["dispute_resolution_minutes"]["value"])),
        bank_settings={str(k): dict(v) for k, v in block["banks"].items()},
        seed_share_range=(low, high),
        reporter_choice=choice,
        screen_delay_ms=(float(block["screen_added_delay_ms"]["min"]), float(block["screen_added_delay_ms"]["max"])),
    )
    if not settings.participants:
        raise ValueError("config/nationalsim.yaml lists no DPIP participant bank")
    return settings


def visible_world_time(ingest_time: datetime, listing_delay: timedelta, world_start: date) -> datetime:
    """
    WHAT: Returns when an entry ingested at ingest_time can first be seen: ingest time plus the listing delay, but
    never before minute 0 of the world.
    WHY: A pre-world listing becomes visible at day 0 at the earliest, because nothing happens before the world.
    Args:
        ingest_time: Aware time the registry received the report.
        listing_delay: The registry's listing delay.
        world_start: World start date.
    Returns:
        Aware IST datetime.
    Example:
        visible_world_time(day0 - timedelta(days=5), timedelta(minutes=15), day0.date()) == day0
    """
    start = datetime(world_start.year, world_start.month, world_start.day, tzinfo=IST)
    return max(ingest_time.astimezone(IST) + listing_delay, start)


def plan_registry_seed(
    phone_book: PhoneBook,
    settings: RegistryReplaySettings,
    rng: np.random.Generator,
    run_id: str,
    key: str,
    world_start: date,
) -> list[RegistryListing]:
    """
    WHAT: Plans the pre-world registry seed. It draws one listing share in the split's range, then for each recycled
    phone (sorted by holder account id) one uniform and one reporter choice. A phone is listed when its uniform is
    below the share; the reporter is a uniform DPIP participant; the report time is the minute its previous holder
    was deactivated (before day 0); the entry is visible at max(that time + listing delay, day 0).
    WHY: The draw count is fixed (1 + 2 per recycled phone) so the stream stays aligned whatever the share is.
    Args:
        phone_book: The recorded world's phone book.
        settings: Registry replay settings.
        rng: The seed's own random stream.
        run_id: The world's run id.
        key: The DPIP key.
        world_start: World start date.
    Returns:
        Listings in report order (time, then holder account id).
    Example:
        plan_registry_seed(book, settings, np.random.default_rng(1), "r1", key, date(2026, 9, 1))
    """
    low, high = settings.seed_share_range
    share = float(rng.uniform(low, high))
    participants = settings.participants
    midnight = datetime(world_start.year, world_start.month, world_start.day, tzinfo=IST)
    listings: list[tuple[datetime, str, RegistryListing]] = []
    recycled = sorted(
        (r for r in phone_book.records.values() if r.reassigned_minute is not None),
        key=lambda r: r.account_id,
    )
    for position, record in enumerate(recycled):
        draw = float(rng.uniform())
        reporter = participants[int(rng.integers(len(participants)))]
        if draw >= share or record.previous_holder_deactivated_minute is None:
            continue
        reported_at = midnight + timedelta(minutes=record.previous_holder_deactivated_minute)
        item = IngestionItem(
            bank_id=reporter,
            report_id=f"{run_id}-S{position:06d}",
            identifier_hash=compute_identifier_hash(key, run_id, IDENTIFIER_TYPE_PHONE, record.phone_number),
            identifier_type=IDENTIFIER_TYPE_PHONE,
            reason_code=DEFAULT_FRAUD_REASON_CODE,
            reported_at=reported_at,
            run_id=run_id,
            linked_account_hash=None,
            reported_minute=record.previous_holder_deactivated_minute,
        )
        visible = visible_world_time(reported_at, settings.listing_delay, world_start)
        listings.append((reported_at, record.account_id, RegistryListing(item, visible, True)))
    listings.sort(key=lambda t: (t[0], t[1]))
    return [t[2] for t in listings]
