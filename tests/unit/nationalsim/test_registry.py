"""
tests/unit/nationalsim/test_registry.py

Unit tests for nationalsim registry core, keyed hashing, and adapter translation.
Covers screening visibility boundary, revocation isolation, dispute status, strict raw
identifier exclusion, non-participant refusal, wrong-key isolation, idempotency, future
report rejection, adapter emission counts, and 1x vs 100x replay clock consistency.
"""

import json
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import pytest

from core.event_clock import IST, ManualClock, ReplayClock
from nationalsim.adapter import (
    ingest_report,
    registry_report_to_ingestions,
)
from nationalsim.keyed_hash import compute_identifier_hash
from nationalsim.registry import NotParticipantError, RegistryCore

DEFAULT_TEST_BANK_SETTINGS = {
    "bank_0": {"dpip_participation": True},
    "bank_1": {"dpip_participation": True},
}



class FakeWallSource:
    """Monotonic wall clock source for controllable time advancement."""

    def __init__(self, start: float = 1000.0) -> None:
        self.current: float = start

    def __call__(self) -> float:
        return self.current

    def tick(self, seconds: float) -> None:
        self.current += seconds


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


def test_screened_identifier_visibility_boundary() -> None:
    """
    A screened identifier does not match at visible_at minus 1 minute, and matches at visible_at.
    """
    clock = ManualClock(datetime(2026, 9, 1, 10, 0, tzinfo=IST))
    listing_delay = timedelta(minutes=15)
    registry = RegistryCore(clock=clock, listing_delay=listing_delay, bank_settings=DEFAULT_TEST_BANK_SETTINGS)

    hash_val = compute_identifier_hash("test-key", "run1", "account", "acc_100", 0)
    reported_at = datetime(2026, 9, 1, 10, 0, tzinfo=IST)

    entry = registry.ingest(
        bank_id="bank_0",
        report_id="R001",
        identifier_hash=hash_val,
        identifier_type="account",
        reason_code="fraud_confirmed",
        reported_at=reported_at,
        run_id="run1",
    )
    assert entry.visible_at == datetime(2026, 9, 1, 10, 15, tzinfo=IST)

    # 1. At visible_at minus 1 minute (10:14 IST), screening returns no match
    clock.set_time(datetime(2026, 9, 1, 10, 14, tzinfo=IST))
    result_before = registry.screen(
        requesting_bank_id="bank_0",
        identifier_hash=hash_val,
        identifier_type="account",
        run_id="run1",
    )
    assert result_before.match is False
    assert result_before.status == "none"
    assert result_before.distinct_accounts_linked == 0

    # 2. At visible_at (10:15 IST), screening matches
    clock.set_time(datetime(2026, 9, 1, 10, 15, tzinfo=IST))
    result_at = registry.screen(
        requesting_bank_id="bank_0",
        identifier_hash=hash_val,
        identifier_type="account",
        run_id="run1",
    )
    assert result_at.match is True
    assert result_at.status == "listed"
    assert result_at.distinct_accounts_linked == 1


def test_revoke_behavior_and_authorization() -> None:
    """
    Revoke removes the match; revoke by another bank is refused;
    with two entries on the same hash, revoking one still matches.
    """
    clock = ManualClock(datetime(2026, 9, 1, 10, 0, tzinfo=IST))
    listing_delay = timedelta(minutes=15)
    registry = RegistryCore(clock=clock, listing_delay=listing_delay, bank_settings=DEFAULT_TEST_BANK_SETTINGS)

    hash_val = compute_identifier_hash("test-key", "run1", "account", "acc_mule_shared", 0)
    now = datetime(2026, 9, 1, 10, 0, tzinfo=IST)

    # Ingest two reports against the same identifier hash (e.g., from bank_0 and bank_1)
    registry.ingest(
        bank_id="bank_0",
        report_id="R-001",
        identifier_hash=hash_val,
        identifier_type="account",
        reason_code="fraud",
        reported_at=now,
        run_id="run1",
    )
    registry.ingest(
        bank_id="bank_1",
        report_id="R-002",
        identifier_hash=hash_val,
        identifier_type="account",
        reason_code="fraud",
        reported_at=now,
        run_id="run1",
    )

    clock.set_time(datetime(2026, 9, 1, 10, 30, tzinfo=IST))
    assert registry.screen("bank_0", hash_val, "account", "run1").match is True

    # Bank 1 cannot revoke Bank 0's report R-001
    with pytest.raises(ValueError, match="cannot revoke report 'R-001' ingested by bank 'bank_0'"):
        registry.revoke(bank_id="bank_1", report_id="R-001", run_id="run1")

    # Bank 0 revokes R-001; hash still matches because R-002 from bank_1 is active
    registry.revoke(bank_id="bank_0", report_id="R-001", run_id="run1")
    assert registry.screen("bank_0", hash_val, "account", "run1").match is True

    # Bank 1 revokes R-002; now all entries are revoked, so screening returns no match
    registry.revoke(bank_id="bank_1", report_id="R-002", run_id="run1")
    assert registry.screen("bank_0", hash_val, "account", "run1").match is False


def test_dispute_marks_entry_contested() -> None:
    """
    Dispute marks the entry contested and screening says 'contested'.
    """
    clock = ManualClock(datetime(2026, 9, 1, 10, 0, tzinfo=IST))
    registry = RegistryCore(
        clock=clock, listing_delay=timedelta(minutes=10), bank_settings=DEFAULT_TEST_BANK_SETTINGS
    )

    hash_val = compute_identifier_hash("key", "run1", "account", "acc_victim_falsely_listed", 0)
    registry.ingest(
        bank_id="bank_0",
        report_id="R-100",
        identifier_hash=hash_val,
        identifier_type="account",
        reason_code="fraud",
        reported_at=datetime(2026, 9, 1, 10, 0, tzinfo=IST),
        run_id="run1",
    )
    clock.set_time(datetime(2026, 9, 1, 10, 20, tzinfo=IST))

    # Before dispute, screening reports "listed"
    assert registry.screen("bank_0", hash_val, "account", "run1").status == "listed"

    # Customer disputes through their bank
    registry.dispute(
        bank_id="bank_0",
        identifier_hash=hash_val,
        reason_code="customer_not_involved",
        run_id="run1",
    )

    # After dispute, screening returns match=True with status="contested"
    res = registry.screen("bank_0", hash_val, "account", "run1")
    assert res.match is True
    assert res.status == "contested"


def test_no_raw_identifiers_stored_in_registry() -> None:
    """
    No raw identifier is stored: ingest real raw values through the adapter,
    serialise the whole store, and assert that none of the raw strings appears
    and every stored identifier is 64 lowercase hex.
    """
    clock = ManualClock(datetime(2026, 9, 1, 10, 0, tzinfo=IST))
    registry = RegistryCore(clock=clock, listing_delay=timedelta(minutes=5))

    secret_raw_account = "acc_ultra_confidential_987654"
    secret_raw_device = "fingerprint_device_hardware_abcdef123"

    report = DummyReport(
        report_id="R-SECRET-01",
        kind="registry_report",
        reported_minute=600,  # 10:00 on 2026-09-01
        reporting_bank_index=2,
        account_id=secret_raw_account,
        device_fingerprint=secret_raw_device,
        confirmed_minute=550,
        complaint_ids=("L001",),
    )

    ingest_report(
        registry=registry,
        report=report,
        run_id="run_crypto_shield",
        key="super_secret_hmac_key",
        world_start=date(2026, 9, 1),
    )

    # Export full serialised store representation
    store = registry.export_store()
    serialized = json.dumps(store)

    # 1. Assert raw strings never appear anywhere in the serialized store
    assert secret_raw_account not in serialized
    assert secret_raw_device not in serialized

    # 2. Assert every stored identifier is exactly 64 lowercase hex characters
    entries = store["entries"]
    assert len(entries) == 2  # 1 account, 1 device
    for entry in entries:
        ident_hash = entry["identifier_hash"]
        assert isinstance(ident_hash, str)
        assert len(ident_hash) == 64
        assert re.fullmatch(r"^[0-9a-f]{64}$", ident_hash) is not None
        if entry["linked_account_hash"]:
            assert re.fullmatch(r"^[0-9a-f]{64}$", entry["linked_account_hash"]) is not None


def test_unlisted_bank_refused_at_screen() -> None:
    """An unlisted bank is refused at screen with NotParticipantError.

    Recreates a situation where an institution that never joined DPIP attempts to query
    the shared registry. Catches a bug where missing bank configuration silently defaults
    to granting screening access.
    """
    clock = ManualClock()
    registry = RegistryCore(
        clock=clock,
        listing_delay=timedelta(minutes=5),
        bank_settings={"bank_0": {"dpip_participation": True}},
    )
    valid_hash = "a" * 64

    with pytest.raises(NotParticipantError, match="not a participating DPIP member"):
        registry.screen(
            requesting_bank_id="bank_outsider",
            identifier_hash=valid_hash,
            identifier_type="account",
            run_id="run1",
        )


def test_listed_bank_set_to_false_refused_at_screen() -> None:
    """A listed bank set to false is refused at screen with NotParticipantError.

    Recreates a situation where a bank has opted out or had its participation revoked.
    Catches a bug where having a dictionary entry for a bank mistakenly grants screening
    privileges regardless of its boolean participation setting.
    """
    clock = ManualClock()
    registry = RegistryCore(
        clock=clock,
        listing_delay=timedelta(minutes=5),
        bank_settings={"bank_inactive": {"dpip_participation": False}},
    )
    valid_hash = "a" * 64

    with pytest.raises(NotParticipantError, match="not a participating DPIP member"):
        registry.screen(
            requesting_bank_id="bank_inactive",
            identifier_hash=valid_hash,
            identifier_type="account",
            run_id="run1",
        )


def test_listed_bank_set_to_true_allowed_at_screen() -> None:
    """A listed bank set to true is allowed to screen identifiers without refusal.

    Recreates standard screening by an enrolled member bank.
    Catches a bug where explicit participation settings are ignored or rejected.
    """
    clock = ManualClock()
    registry = RegistryCore(
        clock=clock,
        listing_delay=timedelta(minutes=5),
        bank_settings={"bank_0": {"dpip_participation": True}},
    )
    valid_hash = "a" * 64

    result = registry.screen(
        requesting_bank_id="bank_0",
        identifier_hash=valid_hash,
        identifier_type="account",
        run_id="run1",
    )
    assert result.match is False
    assert result.status == "none"


def test_wrong_key_hash_never_matches() -> None:
    """
    A hash made with the wrong key never matches.
    """
    clock = ManualClock(datetime(2026, 9, 1, 10, 0, tzinfo=IST))
    registry = RegistryCore(
        clock=clock, listing_delay=timedelta(minutes=10), bank_settings=DEFAULT_TEST_BANK_SETTINGS
    )

    correct_hash = compute_identifier_hash("correct_key", "run1", "account", "acc_target", 0)
    wrong_hash = compute_identifier_hash("wrong_key", "run1", "account", "acc_target", 0)
    assert correct_hash != wrong_hash

    registry.ingest(
        bank_id="bank_0",
        report_id="R-KEY",
        identifier_hash=correct_hash,
        identifier_type="account",
        reason_code="fraud",
        reported_at=datetime(2026, 9, 1, 10, 0, tzinfo=IST),
        run_id="run1",
    )
    clock.advance(timedelta(minutes=20))

    # Screening with wrong key hash produces no match
    res_wrong = registry.screen("bank_0", wrong_hash, "account", "run1")
    assert res_wrong.match is False
    assert res_wrong.status == "none"

    # Screening with correct key hash matches
    res_correct = registry.screen("bank_0", correct_hash, "account", "run1")
    assert res_correct.match is True


def test_idempotent_reingest() -> None:
    """
    Idempotent re-ingest: re-ingesting the same (run, bank, report_id) returns the existing entry.
    """
    clock = ManualClock()
    registry = RegistryCore(clock=clock, listing_delay=timedelta(minutes=5))
    h = "c" * 64
    now = clock.now_ist()

    entry_1 = registry.ingest(
        bank_id="bank_0",
        report_id="R-IDEM",
        identifier_hash=h,
        identifier_type="account",
        reason_code="fraud",
        reported_at=now,
        run_id="run1",
    )
    entry_2 = registry.ingest(
        bank_id="bank_0",
        report_id="R-IDEM",
        identifier_hash=h,
        identifier_type="account",
        reason_code="fraud",
        reported_at=now,
        run_id="run1",
    )
    assert entry_1 is entry_2
    assert len(registry.export_store()["entries"]) == 1


def test_future_dated_report_is_refused() -> None:
    """
    A report dated in the future of the clock is refused.
    """
    clock = ManualClock(datetime(2026, 9, 1, 10, 0, tzinfo=IST))
    registry = RegistryCore(clock=clock, listing_delay=timedelta(minutes=5))

    future_time = datetime(2026, 9, 1, 10, 5, tzinfo=IST)
    with pytest.raises(ValueError, match="in the future relative to clock"):
        registry.ingest(
            bank_id="bank_0",
            report_id="R-FUT",
            identifier_hash="d" * 64,
            identifier_type="account",
            reason_code="fraud",
            reported_at=future_time,
            run_id="run1",
        )


def test_adapter_emits_one_or_two_ingestions() -> None:
    """
    The adapter emits 1 ingestion when device_fingerprint is None and 2 when it is not.
    """
    world_start = date(2026, 9, 1)

    # 1. Report with no device fingerprint
    report_no_device = DummyReport(
        report_id="R-1",
        kind="registry_report",
        reported_minute=100,
        reporting_bank_index=0,
        account_id="acc_no_dev",
        device_fingerprint=None,
        confirmed_minute=80,
        complaint_ids=("L1",),
    )
    items_1 = registry_report_to_ingestions(report_no_device, "run1", "key", world_start)
    assert len(items_1) == 1
    assert items_1[0].identifier_type == "account"
    assert items_1[0].bank_id == "bank_0"

    # 2. Report with device fingerprint
    report_with_device = DummyReport(
        report_id="R-2",
        kind="registry_report",
        reported_minute=100,
        reporting_bank_index=1,
        account_id="acc_with_dev",
        device_fingerprint="fp_device_77",
        confirmed_minute=80,
        complaint_ids=("L2",),
    )
    items_2 = registry_report_to_ingestions(report_with_device, "run1", "key", world_start)
    assert len(items_2) == 2
    assert items_2[0].identifier_type == "account"
    assert items_2[1].identifier_type == "device"
    assert items_2[1].bank_id == "bank_1"


def test_1x_vs_100x_replay_clock_consistency() -> None:
    """
    The 1x vs 100x consistency: the same scripted scenario under two ReplayClocks gives
    identical simulated match times.
    """
    sim_start = datetime(2026, 9, 1, 10, 0, tzinfo=IST)
    listing_delay = timedelta(minutes=15)

    wall_source_1x = FakeWallSource(start=100.0)
    clock_1x = ReplayClock(simulated_start=sim_start, speed=1.0, wall_source=wall_source_1x)
    registry_1x = RegistryCore(
        clock=clock_1x, listing_delay=listing_delay, bank_settings=DEFAULT_TEST_BANK_SETTINGS
    )

    wall_source_100x = FakeWallSource(start=500.0)
    clock_100x = ReplayClock(simulated_start=sim_start, speed=100.0, wall_source=wall_source_100x)
    registry_100x = RegistryCore(
        clock=clock_100x, listing_delay=listing_delay, bank_settings=DEFAULT_TEST_BANK_SETTINGS
    )

    target_hash = compute_identifier_hash("shared-key", "run1", "account", "acc_sync_test", 0)

    # Ingest at simulated start in both
    entry_1x = registry_1x.ingest("bank_0", "R-SYNC", target_hash, "account", "f", sim_start, "run1")
    entry_100x = registry_100x.ingest("bank_0", "R-SYNC", target_hash, "account", "f", sim_start, "run1")
    assert entry_1x.visible_at == entry_100x.visible_at

    # Advance to 14 minutes simulated time (1 minute before visible_at):
    # For 1x: 14 minutes = 840 seconds wall time
    wall_source_1x.tick(840.0)
    # For 100x: 14 minutes = 840 simulated seconds / 100 = 8.4 seconds wall time
    wall_source_100x.tick(8.4)

    assert clock_1x.now_ist() == clock_100x.now_ist()
    assert registry_1x.screen("bank_0", target_hash, "account", "run1").match is False
    assert registry_100x.screen("bank_0", target_hash, "account", "run1").match is False

    # Advance 1 more minute simulated time (reaching exact visible_at):
    # For 1x: 1 minute = 60 seconds wall time
    wall_source_1x.tick(60.0)
    # For 100x: 1 minute = 60 simulated seconds / 100 = 0.6 seconds wall time
    wall_source_100x.tick(0.6)

    assert clock_1x.now_ist() == clock_100x.now_ist()
    res_1x = registry_1x.screen("bank_0", target_hash, "account", "run1")
    res_100x = registry_100x.screen("bank_0", target_hash, "account", "run1")

    assert res_1x.match is True
    assert res_100x.match is True
    assert res_1x.status == res_100x.status == "listed"


def test_simulation_runs_are_strictly_isolated() -> None:
    """Proves that fraud entries from one simulation run never leak into another run.

    In the simulator, different runs represent parallel worlds or evaluation experiments.
    If run isolation failed, an account screened in a clean scenario could wrongly match a fraud
    report that was ingested during a completely separate run.
    """
    clock = ManualClock(datetime(2026, 9, 1, 10, 0, tzinfo=IST))
    registry = RegistryCore(
        clock=clock, listing_delay=timedelta(minutes=10), bank_settings=DEFAULT_TEST_BANK_SETTINGS
    )

    shared_hash = "f" * 64
    now = clock.now_ist()

    # Ingest the identifier in run_A
    registry.ingest(
        bank_id="bank_0",
        report_id="R-RUN-A",
        identifier_hash=shared_hash,
        identifier_type="account",
        reason_code="fraud",
        reported_at=now,
        run_id="run_A",
    )

    # Advance clock past listing delay
    clock.advance(timedelta(minutes=15))

    # Screening in run_A matches
    result_a = registry.screen(
        requesting_bank_id="bank_0",
        identifier_hash=shared_hash,
        identifier_type="account",
        run_id="run_A",
    )
    assert result_a.match is True
    assert result_a.status == "listed"

    # Screening the exact same hash in run_B returns no match
    result_b = registry.screen(
        requesting_bank_id="bank_0",
        identifier_hash=shared_hash,
        identifier_type="account",
        run_id="run_B",
    )
    assert result_b.match is False
    assert result_b.status == "none"

