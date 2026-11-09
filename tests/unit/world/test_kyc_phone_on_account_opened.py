"""
Tests for the KYC phone on account_opened events (world/bank_feed.py, switch kyc_phone_on_account_opened).

WHY: Banks can only match phone hashes in the registry if they hold the phone of their own account holders. The phone
rides on account_opened behind a switch; with the switch off the feed must not change at all.

TERM: KYC phone: the phone number on the account holder's know-your-customer record, from the phone book.
TERM: day-0 opening: an account_opened event for an account that existed when the run started.
TERM: mid-run opening: an account_opened event for an account opened during the run.
"""

import copy
import dataclasses
from typing import Any

import pytest

from world.bank_feed import ACCOUNT_OPENED_DETAIL_FIELDS, BankFeed, project_run
from world.bank_feed_config import load_bank_feed_config
from world.interactive_records import InteractiveWorldOutput, RunnerSettings
from world.interactive_runner import FastRunSettings, run_interactive_world
from world.phone_book import PhoneBook, build_phone_book

SETTINGS = FastRunSettings(101, 20, 6, 51, "training")


@pytest.fixture(scope="module")
def world() -> tuple[InteractiveWorldOutput, PhoneBook]:
    """Runs the small world once and builds its phone book."""
    output = run_interactive_world(SETTINGS, RunnerSettings(fraud_enabled=True))
    return output, build_phone_book(output, run_seed=SETTINGS.seed)


def openings(feed: BankFeed) -> list[Any]:
    """Lists every account_opened event of a feed."""
    return [e for events in feed.account_events_by_bank.values() for e in events if e.kind == "account_opened"]


def strip_phone(feed: BankFeed) -> BankFeed:
    """Returns a copy of a feed with the phone key removed from every account_opened detail."""
    by_bank = {}
    for bank, events in feed.account_events_by_bank.items():
        cleaned = []
        for event in events:
            details = {k: v for k, v in event.details.items() if k != "phone"}
            cleaned.append(dataclasses.replace(event, details=details))
        by_bank[bank] = tuple(cleaned)
    return dataclasses.replace(feed, account_events_by_bank=by_bank)


def test_live_default_is_off() -> None:
    """Checks config/bank_feed.yaml ships the switch off."""
    assert load_bank_feed_config().kyc_phone_on_account_opened is False


def test_off_carries_no_phone_and_changes_nothing(world: tuple[InteractiveWorldOutput, PhoneBook]) -> None:
    """Checks the default feed has no phone key anywhere and equals the ON feed once phones are stripped."""
    output, book = world
    off = project_run(output)
    on = project_run(output, phone_book=book, kyc_phone=True)
    assert all("phone" not in e.details for events in off.account_events_by_bank.values() for e in events)
    assert off == project_run(output, phone_book=book, kyc_phone=False)
    assert strip_phone(on) == off
    assert copy.deepcopy(off) == off


def test_on_every_opening_carries_the_book_phone_or_none(world: tuple[InteractiveWorldOutput, PhoneBook]) -> None:
    """Checks each day-0 and mid-run opening has a phone key equal to the phone book's, None when it has none."""
    output, book = world
    on = project_run(output, phone_book=book, kyc_phone=True)
    mid_ids = {opened[1] for opened in output.result.opened_accounts}
    events = openings(on)
    assert events
    for event in events:
        assert "phone" in event.details
        record = book.get_record(event.account_id)
        assert event.details["phone"] == (record.phone_number if record is not None else None)
        assert event.details["phone"] != ""
    day0 = [e for e in events if e.account_id not in mid_ids]
    with_phone = [e for e in day0 if e.details["phone"] is not None]
    assert with_phone


def test_on_without_a_phone_book_is_refused(world: tuple[InteractiveWorldOutput, PhoneBook]) -> None:
    """Checks switching the phone on with no phone book is a plain error."""
    output, _ = world
    with pytest.raises(ValueError, match="no phone book"):
        project_run(output, kyc_phone=True)


def test_phone_is_not_an_allowed_detail_when_off() -> None:
    """Checks the allowed opening fields are unchanged, so a stray phone would still be caught."""
    assert ACCOUNT_OPENED_DETAIL_FIELDS == frozenset({"home_region", "account_type"})
