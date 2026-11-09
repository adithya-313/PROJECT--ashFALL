"""
Test of the N1 path end to end on a recorded world, without the stack: a recycled phone number whose previous holder
was listed in the pre-world registry seed produces a registry label on the CURRENT (innocent) holder's account.

WHY: N1 says matches are not filtered for recycled numbers; this proves the wrong label lands on the right account.

TERM: recycled number: a phone once held by a mule, now held by an innocent person.
TERM: current holder: the account whose KYC phone is the recycled number today.
"""

from datetime import datetime, timedelta

from core.event_clock import IST, ManualClock
from features.registry_label_consumer import build_own_identifiers, match_registry_event, registry_label_rows
from replay.registry_driver import build_registry_driver
from replay.registry_replay import RegistryReplaySettings
from tests.unit.world.test_phone_book_recording import record
from world.bank_feed import resolve_bank_id
from world.bank_feed_config import load_bank_feed_config

MIDNIGHT = datetime(2026, 9, 1, tzinfo=IST)


def test_seeded_recycled_number_labels_the_current_holder_account() -> None:
    """Checks the registry label for a seeded recycled phone is on the current holder's account id, day 0 or later."""
    recorded = record()
    book = recorded.phone_book
    assert book is not None
    (holder,) = [r for r in book.records.values() if r.is_recycled]
    settings = RegistryReplaySettings(
        timedelta(minutes=15), timedelta(minutes=60), {f"bank_{i}": {"dpip_participation": True} for i in range(3)},
        (1.0, 1.0), "uniform_over_dpip_participants",
    )
    driver = build_registry_driver(recorded, ManualClock(MIDNIGHT), settings)
    (listing,) = driver.seed
    payload = listing.item.to_dict()

    cfg = load_bank_feed_config()
    holder_bank = next(
        resolve_bank_id(ledger.bank_index, cfg)
        for ledger in recorded.output.world.ledgers
        if holder.account_id in ledger.accounts
    )
    run_id = recorded.output.result.run_id
    own = build_own_identifiers(
        holder_bank, int(holder_bank.removeprefix("bank_")), True, [holder.account_id], {},
        {holder.account_id: holder.phone_number}, driver.key, run_id,
    )
    accounts = match_registry_event(payload, own)
    assert accounts == [holder.account_id]
    rows = registry_label_rows(payload, accounts, own, run_id, listing.visible_time)
    assert [r.account_id for r in rows] == [holder.account_id]
    assert rows[0].arrival_time >= MIDNIGHT
