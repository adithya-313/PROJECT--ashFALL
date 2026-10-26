"""
tests/unit/nationalsim/test_dispute_resolution.py

Unit tests for telecom recycled and shared-household phone dispute resolution,
SLA resolution timing (minutes_to_resolve), and post-reassignment fraud protection (Issue 13-10, N4).
"""

from datetime import datetime, timedelta

from fastapi.testclient import TestClient

from core.event_clock import IST, ManualClock
from nationalsim.app import create_app
from nationalsim.keyed_hash import compute_identifier_hash
from nationalsim.registry import RegistryCore


def test_recycled_number_wrongly_matches_then_dispute_resolves_with_exact_minutes() -> None:
    """A recycled number wrongly matches a genuine customer until a dispute resolves it after the configured delay.

    Recreates a real situation where an innocent customer is assigned a recycled phone number
    previously flagged for mule activity. Catches a bug where the registry either fails to
    uphold pre-reassignment disputes or fails to accurately log the elapsed resolution minutes for N4 metrics.
    """
    clock = ManualClock(datetime(2026, 9, 1, 10, 0, tzinfo=IST))
    listing_delay = timedelta(minutes=15)
    dispute_delay = timedelta(minutes=60)
    bank_settings = {"bank_0": {"dpip_participation": True}}
    registry = RegistryCore(
        clock=clock,
        listing_delay=listing_delay,
        bank_settings=bank_settings,
        dispute_resolution_delay=dispute_delay,
    )

    phone_num = "sim-0000000777"
    key = "dpip_secret_key_recycled"
    run_id = "run_recycled_test"
    phone_hash = compute_identifier_hash(key, run_id, "phone", phone_num)

    # 1. Old mule listing reported before day 0 (minute -200)
    # Number was reassigned at minute -10 (after 90 days statutory quarantine)
    registry.ingest(
        bank_id="bank_0",
        report_id="R-MULE-PRE-WORLD",
        identifier_hash=phone_hash,
        identifier_type="phone",
        reason_code="fraud_complaint_confirmed",
        reported_at=datetime(2026, 9, 1, 9, 0, tzinfo=IST),
        run_id=run_id,
        reported_minute=-200,
    )
    # Register public telecom reassignment
    registry.register_telecom_reassignment(
        identifier_hash=phone_hash,
        reassigned_minute=-10,
        run_id=run_id,
    )

    # Advance clock past listing delay
    clock.advance(timedelta(minutes=20))

    # Screening shows wrong match against the innocent holder
    screen_before = registry.screen(
        requesting_bank_id="bank_0",
        identifier_hash=phone_hash,
        identifier_type="phone",
        run_id=run_id,
    )
    assert screen_before.match is True
    assert screen_before.status == "listed"

    # 2. Customer disputes through their bank
    dispute_filed_time = clock.now_ist()
    contested = registry.dispute(
        bank_id="bank_0",
        identifier_hash=phone_hash,
        reason_code="recycled_number_dispute",
        run_id=run_id,
        filed_by="customer_via_bank",
    )
    assert len(contested) == 1

    # Right after dispute: screening gives contested match
    screen_contested = registry.screen(
        requesting_bank_id="bank_0",
        identifier_hash=phone_hash,
        identifier_type="phone",
        run_id=run_id,
    )
    assert screen_contested.match is True
    assert screen_contested.status == "contested"

    # 3. Advance clock by 59 minutes (before resolution delay): still contested
    clock.advance(timedelta(minutes=59))
    screen_still_contested = registry.screen(
        requesting_bank_id="bank_0",
        identifier_hash=phone_hash,
        identifier_type="phone",
        run_id=run_id,
    )
    assert screen_still_contested.match is True
    assert screen_still_contested.status == "contested"

    # 4. Advance clock by 1 more minute (total 60 minutes): dispute resolved
    clock.advance(timedelta(minutes=1))
    screen_resolved = registry.screen(
        requesting_bank_id="bank_0",
        identifier_hash=phone_hash,
        identifier_type="phone",
        run_id=run_id,
    )
    assert screen_resolved.match is False
    assert screen_resolved.status == "resolved"
    assert screen_resolved.risk_score == 0.0

    # 5. Check resolution audit log
    resolutions = registry.get_resolutions()
    assert len(resolutions) == 1
    res = resolutions[0]
    assert res.status == "resolved"
    assert res.filed_by == "customer_via_bank"
    assert res.filed_at == dispute_filed_time
    assert res.resolved_at == dispute_filed_time + timedelta(minutes=60)
    assert res.minutes_to_resolve == 60


def test_shared_household_dispute_stays_contested() -> None:
    """A shared-household dispute is NOT auto-upheld and remains contested indefinitely.

    Recreates a dispute filed for an active telephone number shared between a mule and a genuine
    family member. Catches a bug where the registry mistakenly auto-upholds a dispute on an
    active, current illicit listing.
    """
    clock = ManualClock(datetime(2026, 9, 1, 10, 0, tzinfo=IST))
    listing_delay = timedelta(minutes=10)
    dispute_delay = timedelta(minutes=60)
    bank_settings = {"bank_0": {"dpip_participation": True}}
    registry = RegistryCore(
        clock=clock,
        listing_delay=listing_delay,
        bank_settings=bank_settings,
        dispute_resolution_delay=dispute_delay,
    )

    phone_hash = "1" * 64
    run_id = "run_shared_dispute"

    # Fraud report during the run at minute 120 (no telecom reassignment)
    registry.ingest(
        bank_id="bank_0",
        report_id="R-MULE-SHARED",
        identifier_hash=phone_hash,
        identifier_type="phone",
        reason_code="fraud_complaint_confirmed",
        reported_at=datetime(2026, 9, 1, 10, 0, tzinfo=IST),
        run_id=run_id,
        reported_minute=120,
    )

    clock.advance(timedelta(minutes=15))

    # Customer disputes through bank
    registry.dispute(
        bank_id="bank_0",
        identifier_hash=phone_hash,
        reason_code="shared_household_dispute",
        run_id=run_id,
        filed_by="customer_via_bank",
    )

    # Advance clock far beyond resolution delay
    clock.advance(timedelta(minutes=500))

    screen_after = registry.screen(
        requesting_bank_id="bank_0",
        identifier_hash=phone_hash,
        identifier_type="phone",
        run_id=run_id,
    )
    # Stays contested; not auto-upheld
    assert screen_after.match is True
    assert screen_after.status == "contested"

    resolutions = registry.get_resolutions()
    assert len(resolutions) == 1
    assert resolutions[0].status == "contested"
    assert resolutions[0].resolved_at is None
    assert resolutions[0].minutes_to_resolve is None


def test_dispute_about_listing_made_after_reassignment_not_upheld() -> None:
    """A dispute about a fraud listing reported after the reassignment date is not upheld.

    Recreates a case where a customer claims a number was recycled, but the fraud report was
    logged after they took ownership of the number. Catches a bug where reassignment logic
    ignores report timestamps and excuses post-reassignment fraud.
    """
    clock = ManualClock(datetime(2026, 9, 1, 10, 0, tzinfo=IST))
    listing_delay = timedelta(minutes=10)
    dispute_delay = timedelta(minutes=60)
    bank_settings = {"bank_0": {"dpip_participation": True}}
    registry = RegistryCore(
        clock=clock,
        listing_delay=listing_delay,
        bank_settings=bank_settings,
        dispute_resolution_delay=dispute_delay,
    )

    phone_hash = "2" * 64
    run_id = "run_post_reassign_dispute"

    # Number was reassigned at minute -50
    registry.register_telecom_reassignment(
        identifier_hash=phone_hash,
        reassigned_minute=-50,
        run_id=run_id,
    )

    # Fraud report was made at minute 10 (AFTER reassignment minute -50)
    registry.ingest(
        bank_id="bank_0",
        report_id="R-POST-REASSIGN",
        identifier_hash=phone_hash,
        identifier_type="phone",
        reason_code="fraud_complaint_confirmed",
        reported_at=datetime(2026, 9, 1, 10, 0, tzinfo=IST),
        run_id=run_id,
        reported_minute=10,
    )

    clock.advance(timedelta(minutes=15))

    # Customer disputes
    registry.dispute(
        bank_id="bank_0",
        identifier_hash=phone_hash,
        reason_code="recycled_number_dispute",
        run_id=run_id,
        filed_by="customer_via_bank",
    )

    # Advance clock past resolution delay
    clock.advance(timedelta(minutes=100))

    screen_after = registry.screen(
        requesting_bank_id="bank_0",
        identifier_hash=phone_hash,
        identifier_type="phone",
        run_id=run_id,
    )
    # Not upheld because report occurred after reassignment
    assert screen_after.match is True
    assert screen_after.status == "contested"

    resolutions = registry.get_resolutions()
    assert len(resolutions) == 1
    assert resolutions[0].status == "contested"
    assert resolutions[0].resolved_at is None
    assert resolutions[0].minutes_to_resolve is None


def test_http_dispute_endpoint_lifecycle() -> None:
    """HTTP POST /dispute correctly schedules resolution and returns minutes_to_resolve.

    Recreates API client dispute submission over REST gateway. Catches a bug where
    HTTP serialization drops minutes_to_resolve or fails to transition resolved status.
    """
    clock = ManualClock(datetime(2026, 9, 1, 10, 0, tzinfo=IST))
    app = create_app(
        clock=clock,
        key_provider=lambda: "test_dpip_key",
        settings={
            "listing_delay_minutes": 15,
            "dispute_resolution_minutes": 45,
            "banks": {"bank_0": {"dpip_participation": True}},
        },
    )
    client = TestClient(app)

    phone_hash = "3" * 64
    run_id = "run_http_dispute"

    # Register reassignment on app.state.registry
    app.state.registry.register_telecom_reassignment(
        identifier_hash=phone_hash,
        reassigned_minute=-20,
        run_id=run_id,
    )

    # Ingest old report reported before reassignment
    client.post(
        "/ingest",
        json={
            "bank_id": "bank_0",
            "report_id": "R-DISP-01",
            "identifier_hash": phone_hash,
            "identifier_type": "phone",
            "reason_code": "fraud",
            "reported_at": clock.now_ist().isoformat(),
            "run_id": run_id,
        },
    )
    # Update entry's reported_minute to pre-reassignment minute (-100)
    for entry in app.state.registry._entries.values():
        if entry.identifier_hash == phone_hash:
            entry.reported_minute = -100

    clock.advance(timedelta(minutes=20))

    # Submit dispute via HTTP
    resp = client.post(
        "/dispute",
        json={
            "bank_id": "bank_0",
            "identifier_hash": phone_hash,
            "reason_code": "recycled_number_dispute",
            "run_id": run_id,
            "filed_by": "customer_via_bank",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "resolved"
    assert data["contested_count"] == 1
    assert data["minutes_to_resolve"] == 45

    # Screen before resolution delay
    clock.advance(timedelta(minutes=30))
    resp_screen_before = client.post(
        "/screen",
        json={
            "requesting_bank_id": "bank_0",
            "identifier_hash": phone_hash,
            "identifier_type": "phone",
            "run_id": run_id,
        },
    )
    assert resp_screen_before.status_code == 200
    assert resp_screen_before.json()["match"] is True
    assert resp_screen_before.json()["status"] == "contested"

    # Screen after resolution delay (total 50 minutes > 45 minutes)
    clock.advance(timedelta(minutes=20))
    resp_screen_after = client.post(
        "/screen",
        json={
            "requesting_bank_id": "bank_0",
            "identifier_hash": phone_hash,
            "identifier_type": "phone",
            "run_id": run_id,
        },
    )
    assert resp_screen_after.status_code == 200
    assert resp_screen_after.json()["match"] is False
    assert resp_screen_after.json()["status"] == "resolved"
