"""
Tests for police_bulk labels in the registry consumer (features/registry_label_consumer.py) and the bulk events the
registry driver stages (replay/registry_driver.py).

WHY: A police crackdown reaches the bank that holds the account whether or not it takes part in DPIP; no other bank
learns of it, and the label keeps the batch and region.

TERM: police_bulk: a label source for accounts confirmed by a police crackdown (one batch, one region).
"""

from datetime import datetime

from core.event_clock import IST, ManualClock
from features.registry_label_consumer import build_own_identifiers, match_registry_event, registry_label_rows
from nationalsim.keyed_hash import compute_identifier_hash
from replay.registry_driver import build_registry_driver
from replay.registry_replay import BulkEvent, replay_dpip_key
from tests.unit.world.test_phone_book_recording import record

RUN = "0123456789abcdef"
KEY = replay_dpip_key(RUN)
ARRIVAL = datetime(2026, 9, 12, 9, 30, tzinfo=IST)


def bulk_event(account: str, holding_index: int) -> dict[str, object]:
    """The registry.events payload of one police-bulk confirmation of an account at its own bank."""
    return {
        "report_id": f"{RUN}-B000001-A00000", "identifier_type": "account", "bank_id": f"bank_{holding_index}",
        "identifier_hash": compute_identifier_hash(KEY, RUN, "account", account, bank_index=holding_index),
        "reason_code": "police_bulk", "event_type": "police_bulk", "batch_id": f"{RUN}-B000001", "region": 3,
        "linked_account_hash": None, "run_id": RUN,
    }


def own(bank_index: int, participant: bool) -> object:
    """A bank holding account 'mule_1' when bank_index is 1."""
    accounts = ("mule_1",) if bank_index == 1 else ("other",)
    return build_own_identifiers(f"bank_{bank_index}", bank_index, participant, accounts, {}, {}, KEY, RUN)


def test_non_participant_holder_is_still_labelled() -> None:
    """Checks the holding bank gets the label although it is not a DPIP participant."""
    holder = own(1, participant=False)
    assert match_registry_event(bulk_event("mule_1", 1), holder) == ["mule_1"]  # type: ignore[arg-type]


def test_other_banks_get_nothing() -> None:
    """Checks banks that do not hold the account get no label, participant or not."""
    event = bulk_event("mule_1", 1)
    assert match_registry_event(event, own(0, participant=True)) == []  # type: ignore[arg-type]
    assert match_registry_event(event, own(2, participant=False)) == []  # type: ignore[arg-type]


def test_label_is_police_bulk_with_batch_and_region_and_no_via() -> None:
    """Checks source police_bulk, via None, and the batch id and region are kept."""
    holder = own(1, participant=False)
    event = bulk_event("mule_1", 1)
    (row,) = registry_label_rows(event, ["mule_1"], holder, RUN, ARRIVAL)  # type: ignore[arg-type]
    assert (row.source, row.via, row.region, row.batch_id) == ("police_bulk", None, 3, f"{RUN}-B000001")
    assert (row.bank_id, row.account_id, row.verdict, row.arrival_time) == ("bank_1", "mule_1", "fraud", ARRIVAL)


def test_the_driver_stages_one_bulk_event_per_confirmed_account() -> None:
    """Checks every bulk step carries a police_bulk item, a batch id and the account's own holding bank."""
    driver = build_registry_driver(record(), ManualClock(ARRIVAL))
    for event in driver.bulk:
        assert isinstance(event, BulkEvent)
        assert event.item.reason_code == "police_bulk" and event.item.identifier_type == "account"
        assert event.item.bank_id.startswith("bank_") and event.batch_id.startswith(driver.run_id + "-B")
    steps = [s for s in driver.steps() if isinstance(s.record, BulkEvent)]
    assert len(steps) == len(driver.bulk)
