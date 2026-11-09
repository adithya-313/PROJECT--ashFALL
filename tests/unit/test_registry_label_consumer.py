"""
Tests for the bank-side registry label consumer's pure matching (features/registry_label_consumer.py), with
hand-made hashes and no database or Kafka.

WHY: Who gets a registry label (participants only, never the reported account at its own bank, shared phones and
devices and recycled numbers included) decides what the dataset's registry labels mean.

TERM: reporter bank: the bank that filed the registry report.
TERM: reported account: the account the report is about; its own bank skips it.
TERM: recycled number: a phone once held by a mule, now held by an innocent person (issue N1: still labelled).
"""

from datetime import date, datetime, timedelta

import numpy as np

from core.event_clock import IST
from features.registry_label_consumer import (
    OwnIdentifiers,
    build_own_identifiers,
    match_registry_event,
    registry_label_rows,
)
from nationalsim.keyed_hash import compute_identifier_hash
from replay.registry_replay import RegistryReplaySettings, plan_registry_seed, replay_dpip_key
from world.phone_book import PhoneBook, PhoneRecord

RUN = "0123456789abcdef"
KEY = replay_dpip_key(RUN)
MIDNIGHT = datetime(2026, 9, 1, tzinfo=IST)


def own(
    bank: str,
    index: int,
    participant: bool = True,
    accounts: tuple[str, ...] = ("a1", "a2"),
    fps: dict[str, list[str]] | None = None,
    phones: dict[str, str] | None = None,
) -> OwnIdentifiers:
    """Builds one bank's own identifiers."""
    return build_own_identifiers(bank, index, participant, accounts, fps or {}, phones or {}, KEY, RUN)


def event(kind: str, raw: str, reporter: str, linked: str | None, bank_index: int | None = None) -> dict[str, object]:
    """Builds a registry.events payload for one identifier."""
    return {
        "report_id": f"{RUN}-R000001", "identifier_type": kind, "bank_id": reporter, "linked_account_hash": linked,
        "identifier_hash": compute_identifier_hash(KEY, RUN, kind, raw, bank_index=bank_index), "run_id": RUN,
    }


def test_participant_gets_a_label_and_non_participant_none() -> None:
    """Checks a participant matching a phone is labelled and a non-participant with the same phone gets nothing."""
    ev = event("phone", "sim-1", "bank_0", None)
    assert match_registry_event(ev, own("bank_1", 1, True, phones={"a1": "sim-1"})) == ["a1"]
    assert match_registry_event(ev, own("bank_1", 1, False, phones={"a1": "sim-1"})) == []


def test_reporter_bank_skips_only_the_reported_account() -> None:
    """Checks the reporter's own reported account gets none, for the account item and for its phone item."""
    reported = compute_identifier_hash(KEY, RUN, "account", "a1", bank_index=0)
    bank = own("bank_0", 0, phones={"a1": "sim-1"})
    assert match_registry_event(event("account", "a1", "bank_0", reported, bank_index=0), bank) == []
    assert match_registry_event(event("phone", "sim-1", "bank_0", reported), bank) == []


def test_a_different_account_at_the_reporter_sharing_the_phone_is_labelled() -> None:
    """Checks a second account at the reporting bank with the same phone gets the label."""
    reported = compute_identifier_hash(KEY, RUN, "account", "a1", bank_index=0)
    bank = own("bank_0", 0, phones={"a1": "sim-1", "a2": "sim-1"})
    assert match_registry_event(event("phone", "sim-1", "bank_0", reported), bank) == ["a2"]


def test_pre_world_seed_is_never_skipped() -> None:
    """Checks a listing with no linked account (a seed) labels even a holder at the reporting bank."""
    bank = own("bank_0", 0, phones={"a1": "sim-1"})
    assert match_registry_event(event("phone", "sim-1", "bank_0", None), bank) == ["a1"]


def test_device_match_at_another_bank_is_labelled() -> None:
    """Checks a device hash matches an account at a different bank than the reporter."""
    bank = own("bank_2", 2, fps={"a2": ["fp-9"]})
    assert match_registry_event(event("device", "fp-9", "bank_0", "x" * 64), bank) == ["a2"]


def test_account_hash_of_another_bank_index_does_not_match() -> None:
    """Checks a reported account hash (namespaced by the reporter's index) matches nobody at another bank."""
    ev = event("account", "a1", "bank_0", None, bank_index=0)
    assert match_registry_event(ev, own("bank_1", 1)) == []


def test_recycled_number_labels_the_innocent_current_holder_after_day_zero() -> None:
    """Checks a pre-world seed for a recycled phone labels its current holder with an arrival time at or after day 0."""
    records = {"innocent": PhoneRecord("innocent", "sim-77", True, -120 * 1440, -20 * 1440, False, None)}
    book = PhoneBook(records, {"sim-77": [records["innocent"]]}, 1, 1, 0)
    settings = RegistryReplaySettings(
        timedelta(minutes=15), timedelta(minutes=60), {f"bank_{i}": {"dpip_participation": True} for i in range(3)},
        (1.0, 1.0), "uniform_over_dpip_participants",
    )
    (listing,) = plan_registry_seed(book, settings, np.random.default_rng(3), RUN, KEY, date(2026, 9, 1))
    payload = listing.item.to_dict()
    holder_bank = own("bank_1", 1, phones={"innocent": "sim-77"})
    accounts = match_registry_event(payload, holder_bank)
    assert accounts == ["innocent"]
    rows = registry_label_rows(payload, accounts, holder_bank, RUN, listing.visible_time)
    got = [(r.source, r.account_id, r.verdict, r.bank_id) for r in rows]
    assert got == [("registry", "innocent", "fraud", "bank_1")]
    assert rows[0].arrival_time >= MIDNIGHT
    assert rows[0].label_id == f"{listing.item.report_id}:phone"
