# File: tests/unit/test_registry_as_of.py
# Purpose: Unit tests proving the Level 13 registry as_of past-time replay extension.
# Issue: Level 15b intelligence features and replay support (architect rules 1 and 2).
"""
WHAT: Unit tests verifying RegistryCore as_of replay evaluation, immutability,
temporal boundaries, revoked/contested handling, and FastAPI /screen 400 mapping.

WHY: Fraud intelligence features evaluated during historical simulation replay
must query registry truth as of the payment decision time, without future leakage
and without mutating live in-memory registry entries.
"""

from copy import deepcopy
from datetime import datetime, timedelta
from typing import Any

import pytest
from starlette.testclient import TestClient

from core.event_clock import IST, ManualClock
from nationalsim.app import create_app
from nationalsim.registry import RegistryCore

RUN_ID = "66db9aa439b092ee"
BANK_0 = "bank_0"
IDENTIFIER_HASH_1 = "a" * 64
IDENTIFIER_HASH_2 = "b" * 64


def _create_registry(
    clock: ManualClock,
    listing_delay_hours: int = 2,
    dispute_resolution_delay_hours: int = 2,
) -> RegistryCore:
    return RegistryCore(
        clock=clock,
        listing_delay=timedelta(hours=listing_delay_hours),
        bank_settings={BANK_0: {"dpip_participation": True}},
        dispute_resolution_delay=timedelta(hours=dispute_resolution_delay_hours),
    )


def test_as_of_visibility_before_and_after() -> None:
    """
    CRITICAL: Validates visible_at threshold evaluation in replay:
      - Entry ingested at T with visible_at = T+2h.
      - as_of T+1h -> no match (status 'none', first_visible_at None).
      - as_of T+3h -> match (status 'listed', first_visible_at = T+2h).
    """
    t_start = datetime(2026, 9, 1, 10, 0, tzinfo=IST)
    clock = ManualClock(t_start)
    core = _create_registry(clock, listing_delay_hours=2)

    entry = core.ingest(
        bank_id=BANK_0,
        report_id=f"{RUN_ID}-rep-01",
        identifier_hash=IDENTIFIER_HASH_1,
        identifier_type="account",
        reason_code="mule_report",
        reported_at=t_start,
        run_id=RUN_ID,
    )
    t_visible = t_start + timedelta(hours=2)
    assert entry.visible_at == t_visible

    # Advance clock to T+3h so replay queries up to T+3h are temporally legal
    clock.set_time(t_start + timedelta(hours=3))

    # 1. Query as_of T+1h (before visibility) -> no match
    res_before = core.screen(
        requesting_bank_id=BANK_0,
        identifier_hash=IDENTIFIER_HASH_1,
        identifier_type="account",
        run_id=RUN_ID,
        as_of=t_start + timedelta(hours=1),
    )
    assert res_before.match is False
    assert res_before.status == "none"
    assert res_before.first_visible_at is None

    # 2. Query as_of T+3h (after visibility) -> match
    res_after = core.screen(
        requesting_bank_id=BANK_0,
        identifier_hash=IDENTIFIER_HASH_1,
        identifier_type="account",
        run_id=RUN_ID,
        as_of=t_start + timedelta(hours=3),
    )
    assert res_after.match is True
    assert res_after.status == "listed"
    assert res_after.first_visible_at == t_visible


def test_as_of_revocation_before_and_after() -> None:
    """
    CRITICAL: Validates revocation timing in replay:
      - Entry ingested at T, visible at T+2h.
      - Revoked at T+5h.
      - as_of T+4h -> match (was still listed prior to revocation).
      - as_of T+6h -> no match (revocation took effect at T+5h).
      - Live call (no as_of) after revoke -> no match.
    """
    t_start = datetime(2026, 9, 1, 10, 0, tzinfo=IST)
    clock = ManualClock(t_start)
    core = _create_registry(clock, listing_delay_hours=2)

    core.ingest(
        bank_id=BANK_0,
        report_id=f"{RUN_ID}-rep-02",
        identifier_hash=IDENTIFIER_HASH_1,
        identifier_type="account",
        reason_code="mule_report",
        reported_at=t_start,
        run_id=RUN_ID,
    )

    # Revoke at T+5h
    t_revoke = t_start + timedelta(hours=5)
    clock.set_time(t_revoke)
    core.revoke(bank_id=BANK_0, report_id=f"{RUN_ID}-rep-02", run_id=RUN_ID)

    # Advance clock to T+6h
    clock.set_time(t_start + timedelta(hours=6))

    # 1. as_of T+4h (before revocation) -> match
    res_prior = core.screen(
        requesting_bank_id=BANK_0,
        identifier_hash=IDENTIFIER_HASH_1,
        identifier_type="account",
        run_id=RUN_ID,
        as_of=t_start + timedelta(hours=4),
    )
    assert res_prior.match is True
    assert res_prior.status == "listed"
    assert res_prior.first_visible_at == t_start + timedelta(hours=2)

    # 2. as_of T+6h (after revocation) -> no match
    res_post = core.screen(
        requesting_bank_id=BANK_0,
        identifier_hash=IDENTIFIER_HASH_1,
        identifier_type="account",
        run_id=RUN_ID,
        as_of=t_start + timedelta(hours=6),
    )
    assert res_post.match is False
    assert res_post.status == "none"
    assert res_post.first_visible_at is None

    # 3. Live call without as_of -> no match
    res_live = core.screen(
        requesting_bank_id=BANK_0,
        identifier_hash=IDENTIFIER_HASH_1,
        identifier_type="account",
        run_id=RUN_ID,
    )
    assert res_live.match is False
    assert res_live.status == "none"


def test_as_of_dispute_contested_and_resolved() -> None:
    """
    CRITICAL: Validates dispute adjudication lifecycle in replay:
      - Entry ingested at T (minute 0).
      - Disputed at T+7h with reassigned_minute=10 (eligible for auto-resolution in 2h).
      - Dispute resolved_at scheduled for T+9h.
      - as_of T+8h -> contested.
      - as_of T+10h -> resolved, no match (status 'resolved').
    """
    t_start = datetime(2026, 9, 1, 10, 0, tzinfo=IST)
    clock = ManualClock(t_start)
    core = _create_registry(clock, listing_delay_hours=1, dispute_resolution_delay_hours=2)

    core.ingest(
        bank_id=BANK_0,
        report_id=f"{RUN_ID}-rep-03",
        identifier_hash=IDENTIFIER_HASH_2,
        identifier_type="phone",
        reason_code="reported_phone",
        reported_at=t_start,
        run_id=RUN_ID,
        reported_minute=0,
    )

    # Dispute at T+7h
    t_dispute = t_start + timedelta(hours=7)
    clock.set_time(t_dispute)
    core.dispute(
        bank_id=BANK_0,
        identifier_hash=IDENTIFIER_HASH_2,
        reason_code="recycled_number_dispute",
        run_id=RUN_ID,
        reassigned_minute=10,
    )

    # Advance clock to T+10h
    clock.set_time(t_start + timedelta(hours=10))

    # 1. as_of T+8h -> contested
    res_contested = core.screen(
        requesting_bank_id=BANK_0,
        identifier_hash=IDENTIFIER_HASH_2,
        identifier_type="phone",
        run_id=RUN_ID,
        as_of=t_start + timedelta(hours=8),
    )
    assert res_contested.match is True
    assert res_contested.status == "contested"

    # 2. as_of T+10h -> resolved, no match
    res_resolved = core.screen(
        requesting_bank_id=BANK_0,
        identifier_hash=IDENTIFIER_HASH_2,
        identifier_type="phone",
        run_id=RUN_ID,
        as_of=t_start + timedelta(hours=10),
    )
    assert res_resolved.match is False
    assert res_resolved.status == "resolved"


def test_as_of_future_raises_value_error_and_app_returns_400() -> None:
    """
    CRITICAL: Validates Rule 2 (no future answers):
      - as_of > registry clock raises ValueError("as_of is later than the registry clock").
      - Through FastAPI HTTP endpoint /screen, returns status code 400 Bad Request.
    """
    t_start = datetime(2026, 9, 1, 10, 0, tzinfo=IST)
    clock = ManualClock(t_start)
    core = _create_registry(clock, listing_delay_hours=1)

    core.ingest(
        bank_id=BANK_0,
        report_id=f"{RUN_ID}-rep-04",
        identifier_hash=IDENTIFIER_HASH_1,
        identifier_type="account",
        reason_code="fraud",
        reported_at=t_start,
        run_id=RUN_ID,
    )

    # Direct RegistryCore call with future as_of raises ValueError
    t_future = t_start + timedelta(seconds=1)
    with pytest.raises(ValueError, match="as_of is later than the registry clock"):
        core.screen(
            requesting_bank_id=BANK_0,
            identifier_hash=IDENTIFIER_HASH_1,
            identifier_type="account",
            run_id=RUN_ID,
            as_of=t_future,
        )

    # Naive datetime also raises ValueError
    with pytest.raises(ValueError, match="as_of must be timezone-aware"):
        core.screen(
            requesting_bank_id=BANK_0,
            identifier_hash=IDENTIFIER_HASH_1,
            identifier_type="account",
            run_id=RUN_ID,
            as_of=datetime(2026, 9, 1, 10, 0),
        )

    # Test via FastAPI HTTP application
    app = create_app(
        clock=clock,
        key_provider=lambda: "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        settings={
            "listing_delay_minutes": 60,
            "banks": {BANK_0: {"dpip_participation": True}},
        },
    )
    app.state.registry = core
    client = TestClient(app)

    response = client.post(
        "/screen",
        json={
            "requesting_bank_id": BANK_0,
            "identifier_hash": IDENTIFIER_HASH_1,
            "identifier_type": "account",
            "run_id": RUN_ID,
            "as_of": t_future.isoformat(),
        },
    )
    assert response.status_code == 400
    assert "as_of is later than the registry clock" in response.json()["detail"]


def test_as_of_call_never_mutates_registry_entries() -> None:
    """
    CRITICAL: Validates that as_of queries compute verdicts from timestamps only
    and NEVER mutate internal registry entries (statuses, timestamps, resolution state).
    """
    t_start = datetime(2026, 9, 1, 10, 0, tzinfo=IST)
    clock = ManualClock(t_start)
    core = _create_registry(clock, listing_delay_hours=1, dispute_resolution_delay_hours=2)

    core.ingest(
        bank_id=BANK_0,
        report_id=f"{RUN_ID}-rep-05",
        identifier_hash=IDENTIFIER_HASH_2,
        identifier_type="phone",
        reason_code="reported_phone",
        reported_at=t_start,
        run_id=RUN_ID,
        reported_minute=0,
    )

    # Dispute at T+2h (scheduled resolution at T+4h)
    clock.set_time(t_start + timedelta(hours=2))
    core.dispute(
        bank_id=BANK_0,
        identifier_hash=IDENTIFIER_HASH_2,
        reason_code="recycled_number_dispute",
        run_id=RUN_ID,
        reassigned_minute=10,
    )

    # Clock at T+3h: entry status is still 'contested' in live memory
    clock.set_time(t_start + timedelta(hours=3))

    # Snapshot entries before any as_of query
    entries_before: dict[Any, Any] = deepcopy(core.export_store())

    # Execute multiple as_of queries across past timestamps
    core.screen(
        requesting_bank_id=BANK_0,
        identifier_hash=IDENTIFIER_HASH_2,
        identifier_type="phone",
        run_id=RUN_ID,
        as_of=t_start + timedelta(hours=1),
    )
    core.screen(
        requesting_bank_id=BANK_0,
        identifier_hash=IDENTIFIER_HASH_2,
        identifier_type="phone",
        run_id=RUN_ID,
        as_of=t_start + timedelta(hours=2, minutes=30),
    )

    entries_after: dict[Any, Any] = deepcopy(core.export_store())
    assert entries_before == entries_after


def test_no_as_of_preserves_exact_legacy_behavior() -> None:
    """
    CRITICAL: Validates that live queries (as_of=None) retain byte-for-byte identical
    legacy behavior, including the in-place contested->resolved transition.
    """
    t_start = datetime(2026, 9, 1, 10, 0, tzinfo=IST)
    clock = ManualClock(t_start)
    core = _create_registry(clock, listing_delay_hours=1, dispute_resolution_delay_hours=2)

    entry = core.ingest(
        bank_id=BANK_0,
        report_id=f"{RUN_ID}-rep-06",
        identifier_hash=IDENTIFIER_HASH_2,
        identifier_type="phone",
        reason_code="reported_phone",
        reported_at=t_start,
        run_id=RUN_ID,
        reported_minute=0,
    )

    # Advance to T+2h (entry is visible and listed)
    clock.set_time(t_start + timedelta(hours=2))
    res_listed = core.screen(
        requesting_bank_id=BANK_0,
        identifier_hash=IDENTIFIER_HASH_2,
        identifier_type="phone",
        run_id=RUN_ID,
    )
    assert res_listed.match is True
    assert res_listed.status == "listed"

    # Dispute at T+2h (resolution at T+4h)
    core.dispute(
        bank_id=BANK_0,
        identifier_hash=IDENTIFIER_HASH_2,
        reason_code="recycled_number_dispute",
        run_id=RUN_ID,
        reassigned_minute=10,
    )
    assert entry.status == "contested"

    # Advance to T+5h (past resolution delay)
    clock.set_time(t_start + timedelta(hours=5))
    res_resolved = core.screen(
        requesting_bank_id=BANK_0,
        identifier_hash=IDENTIFIER_HASH_2,
        identifier_type="phone",
        run_id=RUN_ID,
    )
    assert res_resolved.match is False
    assert res_resolved.status == "resolved"
    # Live screen transitioned entry.status in place
    assert entry.status == "resolved"

