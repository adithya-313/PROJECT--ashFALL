"""
WHAT: Drives the in-process national registry during a replay: plans the pre-world seed and the in-world reports,
turns them into replay steps, ingests them into a RegistryCore on the replay world clock, and stages a registry.events
row in the workspace national_outbox when an entry becomes visible.

WHY: Registry labels must come through the real path (registry -> national_outbox -> workspace topic -> bank-side
consumer), at the minute the registry would show them, so a decision never sees a listing from its own future.

TERM: registry_ingest step: a bank's report reaches the registry (world minute = reported minute); order 4.
TERM: registry_visible step: the entry becomes visible after the listing delay; order 5 (so a zero delay still works).
TERM: seed: pre-world listings, ingested before the first step with the clock set to their pre-world report time.

Order and reason: at one world minute the account events, payments, withdrawals and complaints come first (orders
0-3) because a report is made after the complaints that confirmed it; ingest (4) precedes visible (5) so an entry is
never shown before it is received. Revoke and dispute are hand-made actions on these two kinds only; the world makes
none (a later issue tracks that).
"""

import uuid
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from core.event_clock import IST, ManualClock
from features.registry_prefetch import load_registry_timeout_seconds
from nationalsim.adapter import IngestionItem, registry_report_to_ingestions
from nationalsim.national_outbox import write_event
from nationalsim.registry import RegistryCore, RegistryEntry
from replay.ordering import FeedStep
from replay.registry_replay import (
    REGISTRY_INGEST,
    REGISTRY_VISIBLE,
    RegistryListing,
    RegistryReplaySettings,
    load_registry_replay_settings,
    plan_registry_seed,
    replay_dpip_key,
    visible_world_time,
)
from replay.registry_screener import CoreRegistryScreener, ScreenStats
from world.interactive_loop import build_component_streams
from world.label_builder import build_label_events, build_registry_reports
from world.replay_recording import RecordedWorld
from world.run_identity import load_runner_config

__all__ = ["RegistryDriver", "build_registry_driver", "registry_event_payload"]

SEED_STREAM_NAME = "registry_seed"
SCREEN_DELAY_STREAM_NAME = "registry_screen_delay"


def registry_event_payload(item: IngestionItem, entry: RegistryEntry) -> dict[str, Any]:
    """
    WHAT: Builds the registry.events payload for a visible entry: the ingestion item's dictionary (reporter bank id,
    hashes, linked account hash, report minute) plus when and how the entry became visible.
    WHY: The bank-side consumer needs the reporter and the linked account hash to apply its skip rule.
    Args:
        item: The ingestion item that was reported.
        entry: The registry entry the registry made from it.
    Returns:
        A JSON-ready dictionary; it holds hashes only, never a raw identifier.
    Example:
        registry_event_payload(item, entry)["bank_id"] == item.bank_id
    """
    payload = item.to_dict()
    payload["visible_at"] = entry.visible_at.isoformat()
    payload["status"] = entry.status
    # A deterministic event id and the world time keep the staged row reproducible and idempotent.
    payload["event_id"] = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{item.run_id}:{item.report_id}:{item.identifier_type}"))
    payload["occurred_at"] = entry.visible_at.isoformat()
    return payload


@dataclass
class RegistryDriver:
    """
    WHAT: Holds the registry, its plan and the replay steps made from the plan.
    WHY: One object the replay runner calls for the seed, for each registry step, and for the key.
    Args:
        core: The in-process registry, on the replay world clock.
        key: The DPIP key of the run.
        seed: Pre-world listings.
        world: In-world listings.
        world_start: World start date.
    Example:
        driver = build_registry_driver(recorded_world, world_clock)
    """

    core: RegistryCore
    key: str
    seed: list[RegistryListing]
    world: list[RegistryListing]
    world_start: date
    run_id: str = ""
    clock: Any = None
    screen_rng: Any = None
    screen_delay_ms: tuple[float, float] = (2.0, 15.0)
    screen_timeout_ms: float = 33.19
    stats: ScreenStats = field(default_factory=ScreenStats)

    def screener_for(self, bank_id: str) -> CoreRegistryScreener | None:
        """
        WHAT: The screener for one bank, or None for a bank that is not a DPIP participant.
        WHY: A non-participant never screens; its registry features stay None by design.
        Args:
            bank_id: The bank id, e.g. "bank_0".
        Returns:
            A CoreRegistryScreener, or None.
        Example:
            driver.screener_for("bank_0")
        """
        if not self.core.is_participant(bank_id):
            return None
        return CoreRegistryScreener(
            self.core, bank_id, self.run_id, self.clock, self.screen_rng, self.screen_delay_ms,
            self.screen_timeout_ms, self.stats,
        )

    def steps(self) -> list[FeedStep]:
        """The registry_ingest steps of the in-world reports and the registry_visible steps of every listing."""
        out: list[FeedStep] = []
        for listing in self.world:
            out.append(
                FeedStep(listing.item.reported_at.astimezone(IST), REGISTRY_INGEST, listing.item.bank_id,
                         listing.item.report_id + ":" + listing.item.identifier_type, listing)
            )
        for listing in [*self.seed, *self.world]:
            out.append(
                FeedStep(listing.visible_time, REGISTRY_VISIBLE, listing.item.bank_id,
                         listing.item.report_id + ":" + listing.item.identifier_type, listing)
            )
        return out

    def _ingest(self, item: IngestionItem) -> RegistryEntry:
        """Ingests one item (idempotent: a repeated call returns the existing entry)."""
        return self.core.ingest(
            item.bank_id, item.report_id, item.identifier_hash, item.identifier_type, item.reason_code,
            item.reported_at, item.run_id, linked_account_hash=item.linked_account_hash,
            reported_minute=item.reported_minute,
        )

    def ingest_seed(self, clock: ManualClock, start_time: datetime) -> int:
        """
        WHAT: Ingests every seed listing with the clock at its pre-world report time, then puts the clock back.
        WHY: The registry received those reports before day 0; its received_at must say so.
        Args:
            clock: The replay world clock the registry reads.
            start_time: The clock time to restore afterwards.
        Returns:
            Number of seed listings ingested.
        Example:
            driver.ingest_seed(world_clock, initial_time)
        """
        for listing in self.seed:
            clock.set_time(listing.item.reported_at)
            self._ingest(listing.item)
        clock.set_time(start_time)
        return len(self.seed)

    async def run_step(self, step: FeedStep, conn: Any) -> None:
        """
        WHAT: Executes one registry step. Ingest: the registry receives the report at the current clock. Visible:
        the entry is looked up and a registry.events row is staged in the workspace national_outbox.
        WHY: The drain relays staged rows to the workspace registry topic.
        Args:
            step: A registry_ingest or registry_visible step.
            conn: Workspace database connection (has national_outbox).
        Example:
            await driver.run_step(step, conn)
        """
        listing: RegistryListing = step.record
        entry = self._ingest(listing.item)
        if step.kind == REGISTRY_VISIBLE:
            bank_index = int(listing.item.bank_id.removeprefix("bank_"))
            await write_event(
                conn, "registry.events", listing.item.run_id, bank_index, registry_event_payload(listing.item, entry)
            )


def build_registry_driver(
    recorded_world: RecordedWorld, clock: ManualClock, settings: RegistryReplaySettings | None = None
) -> RegistryDriver:
    """
    WHAT: Plans the registry side of a replay for a recorded world: the seed (own stream, child registry_seed), the
    in-world reports converted to hashed ingestion items with the phone book, and the registry on the given clock.
    WHY: Everything the registry will do is decided up front and deterministically.
    Args:
        recorded_world: The recorded world (needs its phone book).
        clock: The replay world clock.
        settings: Optional registry settings (default: read from config for the world's split).
    Returns:
        A RegistryDriver.
    Raises:
        ValueError: If the recorded world has no phone book.
    Example:
        driver = build_registry_driver(recorded_world, ManualClock(start))
    """
    if recorded_world.phone_book is None:
        raise ValueError("world registry reporting needs the phone book; record the world with replay recording")
    output = recorded_world.output
    effective = output.effective_config
    split = effective["settings"]["range_name"]
    reg = settings if settings is not None else load_registry_replay_settings(split)
    run_id = output.result.run_id
    world_start = output.plan.clock.start_date
    if isinstance(world_start, datetime):
        world_start = world_start.astimezone(IST).date()
    key = replay_dpip_key(run_id)

    children = load_runner_config()["random_stream_children"]
    stream = build_component_streams(effective["settings"]["seed"], {SEED_STREAM_NAME: children[SEED_STREAM_NAME]})
    seed = plan_registry_seed(recorded_world.phone_book, reg, stream[SEED_STREAM_NAME], run_id, key, world_start)

    reports = build_registry_reports(output, build_label_events(output))
    world: list[RegistryListing] = []
    for report in reports:
        for item in registry_report_to_ingestions(report, run_id, key, world_start, recorded_world.phone_book):
            visible = visible_world_time(item.reported_at, reg.listing_delay, world_start)
            world.append(RegistryListing(item, visible, False))

    core = RegistryCore(clock, reg.listing_delay, reg.bank_settings, reg.dispute_resolution_delay)
    delay_stream = build_component_streams(
        effective["settings"]["seed"], {SCREEN_DELAY_STREAM_NAME: children[SCREEN_DELAY_STREAM_NAME]}
    )[SCREEN_DELAY_STREAM_NAME]
    return RegistryDriver(
        core, key, seed, world, world_start, run_id=run_id, clock=clock, screen_rng=delay_stream,
        screen_delay_ms=reg.screen_delay_ms, screen_timeout_ms=load_registry_timeout_seconds() * 1000.0,
    )
