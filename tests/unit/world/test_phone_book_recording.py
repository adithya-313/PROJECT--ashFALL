"""
Tests that the recording step builds the phone book once, on its own seeded stream (world/replay_recording.py).

WHY: Banks will later see account phones at opening; the book must be reproducible and must not move any other world
number.

TERM: phone book: the synthetic directory mapping each person account to a phone number (world/phone_book.py).
"""

import asyncio

from tests.integration.test_replay_recording import RejectingTransport
from world.bank_connectors import MockBankConnectors, load_bank_modes
from world.bank_feed import resolve_bank_id
from world.bank_feed_config import load_bank_feed_config
from world.interactive_records import RunnerSettings
from world.interactive_runner import FastRunSettings, prepare_world, run_interactive_world
from world.phone_book import build_phone_book
from world.replay_recording import RecordedWorld, record_world_for_replay

SETTINGS = FastRunSettings(101, 20, 6, 51, "training")


def record() -> RecordedWorld:
    """Records the small world with rejecting transports, no network."""
    cfg = load_bank_feed_config()
    modes = load_bank_modes()
    prepared = prepare_world(SETTINGS, None)
    account_bank_map = {
        acc: resolve_bank_id(ledger.bank_index, cfg) for ledger in prepared.world.ledgers for acc in ledger.accounts
    }
    loop = asyncio.new_event_loop()
    try:
        connectors = MockBankConnectors(
            transports={b: RejectingTransport() for b in modes}, modes=modes, account_bank_map=account_bank_map,
            clock_start=prepared.plan.clock.start_date, feed_config=cfg, decision_timeout_seconds=5.0, loop=loop,
        )
        return record_world_for_replay(SETTINGS, RunnerSettings(fraud_enabled=True), connectors)
    finally:
        loop.close()


def test_same_seed_gives_an_identical_phone_book() -> None:
    """Records the same seed twice and checks every phone record is equal."""
    first, second = record(), record()
    assert first.phone_book is not None and second.phone_book is not None
    assert first.phone_book.records == second.phone_book.records
    assert first.phone_book.records


def test_recorded_book_is_the_one_build_phone_book_makes() -> None:
    """Checks the recorder uses build_phone_book itself (same function the edge-case check calls)."""
    recorded = record()
    assert recorded.phone_book is not None
    assert recorded.phone_book.records == build_phone_book(recorded.output, run_seed=SETTINGS.seed).records


def test_building_the_book_changes_no_other_world_number() -> None:
    """Checks the recorded world's run id and event list equal a plain run of the same settings."""
    recorded = record()
    plain = run_interactive_world(SETTINGS, RunnerSettings(fraud_enabled=True))
    assert recorded.output.result.run_id == plain.result.run_id
    assert recorded.output.result.run_result.events == plain.result.run_result.events
