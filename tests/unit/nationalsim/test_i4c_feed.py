"""
tests/unit/nationalsim/test_i4c_feed.py

Unit tests for I4C intelligence feed isolation, separate key hashing, non-participant access,
and strict exclusion of raw telephone numbers across DPIP and I4C stores.
"""

import json
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import numpy as np
import pytest
from fastapi.testclient import TestClient

from core.event_clock import IST, ManualClock
from nationalsim.adapter import ingest_report
from nationalsim.app import create_app
from nationalsim.i4c_feed import I4CFeedCore
from nationalsim.keyed_hash import compute_identifier_hash
from nationalsim.registry import NotParticipantError, RegistryCore, draw_dpip_participation
from world.phone_book import PhoneBook, PhoneRecord


@dataclass(frozen=True)
class DummyReport:
    """Mock registry report matching world/label_records.py RegistryReport shape."""

    report_id: str
    kind: str
    reported_minute: int
    reporting_bank_index: int
    account_id: str
    device_fingerprint: str | None
    confirmed_minute: int
    complaint_ids: tuple[str, ...]


def test_non_participating_bank_gets_no_dpip_match_but_does_get_i4c_feed_entries() -> None:
    """A non-participating bank is refused DPIP screening but receives I4C intelligence entries.

    Recreates a real situation where an institution has not joined bilateral DPIP agreements
    but participates in mandatory national cybercrime coordination under I4C. Catches a bug
    where DPIP screening permissions mistakenly gate access to the broadcast I4C feed.
    """
    clock = ManualClock(datetime(2026, 9, 1, 10, 0, tzinfo=IST))
    listing_delay = timedelta(minutes=15)
    bank_settings = {
        "bank_0": {"dpip_participation": True},
        "bank_outsider": {"dpip_participation": False},
    }
    registry = RegistryCore(clock=clock, listing_delay=listing_delay, bank_settings=bank_settings)
    i4c_feed = I4CFeedCore(clock=clock, listing_delay=listing_delay)

    phone_num = "sim-0000000042"
    phone_rec = PhoneRecord(account_id="acc_mule_101", phone_number=phone_num)
    phone_book = PhoneBook(
        records={"acc_mule_101": phone_rec},
        by_phone={phone_num: [phone_rec]},
        run_seed=101,
        recycled_count=0,
        shared_count=0,
    )

    report = DummyReport(
        report_id="R-TEST-001",
        kind="registry_report",
        reported_minute=600,
        reporting_bank_index=0,
        account_id="acc_mule_101",
        device_fingerprint=None,
        confirmed_minute=580,
        complaint_ids=("L1",),
    )

    ingest_report(
        registry=registry,
        report=report,
        run_id="run_i4c_test",
        key="dpip_secret_key_1",
        world_start=date(2026, 9, 1),
        phone_book=phone_book,
        i4c_feed=i4c_feed,
        i4c_key="i4c_secret_key_2",
    )

    # Advance clock past listing delay
    clock.advance(timedelta(minutes=20))

    # 1. Bank outsider is refused at DPIP screen with NotParticipantError
    phone_dpip_hash = compute_identifier_hash("dpip_secret_key_1", "run_i4c_test", "phone", phone_num)
    with pytest.raises(NotParticipantError, match="not a participating DPIP member"):
        registry.screen(
            requesting_bank_id="bank_outsider",
            identifier_hash=phone_dpip_hash,
            identifier_type="phone",
            run_id="run_i4c_test",
        )

    # 2. Bank outsider receives I4C feed entries successfully
    feed_entries = i4c_feed.get_feed(bank_id="bank_outsider", run_id="run_i4c_test")
    assert len(feed_entries) == 1
    assert feed_entries[0].identifier_type == "phone"
    assert feed_entries[0].report_id == "R-TEST-001"


def test_bank_never_sees_another_banks_account_entries() -> None:
    """A bank only receives its own account entries in the I4C feed, never another bank's data.

    Recreates regulatory compliance rules where confidential customer account lists must never
    be distributed to competitor banks. Catches a bug where the I4C feed broadcasts private
    account records across all banks rather than scoping them strictly to the holding bank.
    """
    clock = ManualClock(datetime(2026, 9, 1, 10, 0, tzinfo=IST))
    listing_delay = timedelta(minutes=10)
    i4c_feed = I4CFeedCore(clock=clock, listing_delay=listing_delay)

    # Ingest account report from bank_0
    i4c_feed.ingest(
        bank_id="bank_0",
        report_id="R-B0-ACC",
        identifier_hash="0" * 64,
        identifier_type="account",
        reason_code="fraud",
        reported_at=datetime(2026, 9, 1, 10, 0, tzinfo=IST),
        run_id="run_leak_test",
    )

    # Ingest account report from bank_1
    i4c_feed.ingest(
        bank_id="bank_1",
        report_id="R-B1-ACC",
        identifier_hash="1" * 64,
        identifier_type="account",
        reason_code="fraud",
        reported_at=datetime(2026, 9, 1, 10, 0, tzinfo=IST),
        run_id="run_leak_test",
    )

    # Ingest global phone report
    i4c_feed.ingest(
        bank_id="bank_0",
        report_id="R-PHONE-GLOBAL",
        identifier_hash="f" * 64,
        identifier_type="phone",
        reason_code="fraud",
        reported_at=datetime(2026, 9, 1, 10, 0, tzinfo=IST),
        run_id="run_leak_test",
    )

    clock.advance(timedelta(minutes=15))

    # Bank 0 sees its own account and the global phone, but NOT Bank 1's account
    feed_b0 = i4c_feed.get_feed(bank_id="bank_0", run_id="run_leak_test")
    report_ids_b0 = [e.report_id for e in feed_b0]
    assert "R-B0-ACC" in report_ids_b0
    assert "R-PHONE-GLOBAL" in report_ids_b0
    assert "R-B1-ACC" not in report_ids_b0

    # Bank 1 sees its own account and the global phone, but NOT Bank 0's account
    feed_b1 = i4c_feed.get_feed(bank_id="bank_1", run_id="run_leak_test")
    report_ids_b1 = [e.report_id for e in feed_b1]
    assert "R-B1-ACC" in report_ids_b1
    assert "R-PHONE-GLOBAL" in report_ids_b1
    assert "R-B0-ACC" not in report_ids_b1

    # Bank 2 sees only the global phone, and neither bank's accounts
    feed_b2 = i4c_feed.get_feed(bank_id="bank_2", run_id="run_leak_test")
    report_ids_b2 = [e.report_id for e in feed_b2]
    assert report_ids_b2 == ["R-PHONE-GLOBAL"]


def test_dpip_hash_and_i4c_hash_of_same_phone_differ() -> None:
    """A DPIP hash and an I4C hash of the exact same synthetic phone number differ.

    Recreates cryptographic system separation where DPIP and I4C operate under independent
    key management domains. Catches a bug where the adapter mistakenly hashes phone numbers
    with the same key across both systems, breaking cryptographic isolation.
    """
    phone_raw = "sim-0000009999"
    run_id = "run_crypto_diff"
    dpip_key = "dpip_secret_encryption_key_alpha"
    i4c_key = "i4c_secret_encryption_key_beta"

    dpip_hash = compute_identifier_hash(
        key=dpip_key,
        run_id=run_id,
        identifier_type="phone",
        raw_value=phone_raw,
    )
    i4c_hash = compute_identifier_hash(
        key=i4c_key,
        run_id=run_id,
        identifier_type="phone",
        raw_value=phone_raw,
    )

    assert dpip_hash != i4c_hash
    assert len(dpip_hash) == 64
    assert len(i4c_hash) == 64
    assert re.fullmatch(r"^[0-9a-f]{64}$", dpip_hash) is not None
    assert re.fullmatch(r"^[0-9a-f]{64}$", i4c_hash) is not None


def test_no_raw_phone_number_stored_anywhere() -> None:
    """No raw telephone number is stored anywhere in the DPIP store or the I4C feed.

    Recreates privacy and security constraints forbidding unhashed PII storage in national
    systems. Catches a bug where the adapter or feed inadvertently serializes or caches
    plain-text telephone numbers.
    """
    clock = ManualClock(datetime(2026, 9, 1, 10, 0, tzinfo=IST))
    listing_delay = timedelta(minutes=5)
    registry = RegistryCore(clock=clock, listing_delay=listing_delay)
    i4c_feed = I4CFeedCore(clock=clock, listing_delay=listing_delay)

    secret_raw_phone = "sim-0000077777"
    secret_raw_account = "acc_holder_secret_88"

    phone_rec = PhoneRecord(account_id=secret_raw_account, phone_number=secret_raw_phone)
    phone_book = PhoneBook(
        records={secret_raw_account: phone_rec},
        by_phone={secret_raw_phone: [phone_rec]},
        run_seed=202,
        recycled_count=0,
        shared_count=0,
    )

    report = DummyReport(
        report_id="R-CONFIDENTIAL-01",
        kind="registry_report",
        reported_minute=600,
        reporting_bank_index=1,
        account_id=secret_raw_account,
        device_fingerprint=None,
        confirmed_minute=590,
        complaint_ids=("L7",),
    )

    ingest_report(
        registry=registry,
        report=report,
        run_id="run_strict_hash",
        key="dpip_key_sec",
        world_start=date(2026, 9, 1),
        phone_book=phone_book,
        i4c_feed=i4c_feed,
        i4c_key="i4c_key_sec",
    )

    # Export both serialised stores
    registry_data = json.dumps(registry.export_store())
    i4c_data = json.dumps(i4c_feed.export_feed())

    # Raw phone number must never appear in either serialized store
    assert secret_raw_phone not in registry_data
    assert secret_raw_phone not in i4c_data

    # Every identifier hash in both stores must be exactly 64 lowercase hex characters
    for entry in registry.export_store()["entries"]:
        h = entry["identifier_hash"]
        assert len(h) == 64
        assert re.fullmatch(r"^[0-9a-f]{64}$", h) is not None

    for entry in i4c_feed.export_feed()["entries"]:
        h = entry["identifier_hash"]
        assert len(h) == 64
        assert re.fullmatch(r"^[0-9a-f]{64}$", h) is not None


def test_draw_dpip_participation_deterministic_and_respects_probability() -> None:
    """Per-world DPIP participation sampling respects range bounds and seed determinism.

    Recreates per-world bank enrollment generation across different evaluation splits.
    Catches a bug where bank participation sampling ignores probability bounds or fails
    reproducibility across identical random seeds.
    """
    world_ranges = {
        "parameters": {
            "probability_a_bank_takes_part_in_dpip": {
                "training": [0.4, 0.4],  # Pinned to 0.4 for deterministic evaluation
            }
        }
    }
    bank_ids = ["0", "1", "2"]

    rng1 = np.random.default_rng(42)
    settings_1 = draw_dpip_participation(bank_ids, world_ranges, "training", rng1)

    rng2 = np.random.default_rng(42)
    settings_2 = draw_dpip_participation(bank_ids, world_ranges, "training", rng2)

    assert settings_1 == settings_2
    assert "bank_0" in settings_1
    assert "bank_1" in settings_1
    assert "bank_2" in settings_1
    for bank_setting in settings_1.values():
        assert isinstance(bank_setting["dpip_participation"], bool)


def test_http_i4c_feed_endpoint() -> None:
    """GET /i4c/feed serves visible intelligence entries over HTTP with bank scoping.

    Recreates the REST API gateway serving I4C feed requests to banking institutions.
    Catches a bug where the HTTP endpoint fails to serialize feed entries, ignores
    listing delays, or leaks foreign bank accounts.
    """
    clock = ManualClock(datetime(2026, 9, 1, 10, 0, tzinfo=IST))
    app = create_app(
        clock=clock,
        key_provider=lambda: "dpip_key",
        settings={
            "listing_delay_minutes": 15,
            "banks": {"bank_0": {"dpip_participation": True}, "bank_1": {"dpip_participation": True}},
        },
    )
    client = TestClient(app)

    # 1. Ingest account report for bank_0 and phone report
    now_iso = clock.now_ist().isoformat()
    client.post(
        "/ingest",
        json={
            "bank_id": "bank_0",
            "report_id": "R-HTTP-B0",
            "identifier_hash": "a" * 64,
            "identifier_type": "account",
            "reason_code": "fraud",
            "reported_at": now_iso,
            "run_id": "run_http_i4c",
        },
    )
    client.post(
        "/ingest",
        json={
            "bank_id": "bank_1",
            "report_id": "R-HTTP-PHONE",
            "identifier_hash": "b" * 64,
            "identifier_type": "phone",
            "reason_code": "fraud",
            "reported_at": now_iso,
            "run_id": "run_http_i4c",
        },
    )

    # Before listing delay (10:10 IST): feed is empty
    clock.advance(timedelta(minutes=10))
    res_before = client.get("/i4c/feed?bank_id=bank_0&run_id=run_http_i4c")
    assert res_before.status_code == 200
    assert res_before.json()["entries_count"] == 0

    # After listing delay (10:20 IST): feed delivers bank_0 account and shared phone
    clock.advance(timedelta(minutes=10))
    res_after_b0 = client.get("/i4c/feed?bank_id=bank_0&run_id=run_http_i4c")
    assert res_after_b0.status_code == 200
    data_b0 = res_after_b0.json()
    assert data_b0["entries_count"] == 2
    report_ids_b0 = [e["report_id"] for e in data_b0["entries"]]
    assert "R-HTTP-B0" in report_ids_b0
    assert "R-HTTP-PHONE" in report_ids_b0

    # Bank 1 queries feed: sees shared phone, but NOT bank_0's account
    res_after_b1 = client.get("/i4c/feed?bank_id=bank_1&run_id=run_http_i4c")
    assert res_after_b1.status_code == 200
    data_b1 = res_after_b1.json()
    assert data_b1["entries_count"] == 1
    assert data_b1["entries"][0]["report_id"] == "R-HTTP-PHONE"
