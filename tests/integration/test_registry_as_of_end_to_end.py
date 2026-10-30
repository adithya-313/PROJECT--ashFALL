# File: tests/integration/test_registry_as_of_end_to_end.py
# Purpose: Integration tests proving Nationalsim as_of historical screening end-to-end over HTTP ASGI transport.
#
# Proves:
#   a) An entry reported at t0, visible at t0+2m, is NOT returned to screen(as_of=t0+1m)
#      and IS returned to screen(as_of=t0+3m).
#   b) An entry revoked at t0+10m is returned to screen(as_of=t0+5m) and NOT returned to screen(as_of=t0+15m).
#   c) A 403 response (unknown participant bank) produces status="unavailable" on the bank client
#      without raising an unhandled exception.

from datetime import datetime, timedelta

import httpx
import pytest

from core.event_clock import IST, ManualClock
from nationalsim.app import create_app
from nationalsim.bank_client import NationalsimBankClient, ScreenResult

pytestmark = [pytest.mark.anyio, pytest.mark.needs_stack]


@pytest.fixture
def nationalsim_setup() -> tuple[ManualClock, httpx.ASGITransport]:
    """
    Initializes a live FastAPI nationalsim application configured with a 2-minute listing delay
    and bank_0 / bank_1 as participating banks.
    """
    t0 = datetime(2026, 9, 1, 10, 0, tzinfo=IST)
    clock = ManualClock(t0)
    settings = {
        "listing_delay_minutes": 2,
        "dispute_resolution_minutes": 60,
        "banks": {
            "bank_0": {"dpip_participation": True},
            "bank_1": {"dpip_participation": True},
        },
    }
    app = create_app(
        clock=clock,
        key_provider=lambda: "test_shared_secret",
        settings=settings,
    )
    transport = httpx.ASGITransport(app=app)
    return clock, transport


async def test_as_of_visibility_delay_end_to_end(
    nationalsim_setup: tuple[ManualClock, httpx.ASGITransport],
) -> None:
    """
    Proves that an entry reported at t0 with a 2m delay is invisible at t0+1m and visible at t0+3m
    when queried through NationalsimBankClient against the live ASGI app.
    """
    clock, transport = nationalsim_setup
    t0 = clock.now_ist()
    test_hash = "c" * 64

    # 1. Ingest report at t0 via HTTP endpoint
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http_client:
        ingest_resp = await http_client.post(
            "/ingest",
            json={
                "bank_id": "bank_0",
                "report_id": "rep_vis_001",
                "identifier_hash": test_hash,
                "identifier_type": "account",
                "reason_code": "mule_account",
                "reported_at": t0.isoformat(),
                "run_id": "run_e2e_001",
            },
        )
        assert ingest_resp.status_code == 200

    # 2. Advance clock to t0+10m so that current time is past visibility
    clock.set_time(t0 + timedelta(minutes=10))

    client = NationalsimBankClient(
        base_url="http://test",
        bank_id="bank_1",
        run_id="run_e2e_001",
        clock=clock,
        transport=transport,
    )

    # 3. Query as_of=t0+1m (before visibility): must be clean / no match
    res_1m: ScreenResult = await client.screen(
        identifier_hash=test_hash,
        identifier_type="account",
        as_of=t0 + timedelta(minutes=1),
    )
    assert res_1m.status == "no_match"
    assert res_1m.match is False
    assert res_1m.is_unavailable is False
    assert res_1m.first_visible_at is None

    # 4. Query as_of=t0+3m (after visibility): must be listed / match
    res_3m: ScreenResult = await client.screen(
        identifier_hash=test_hash,
        identifier_type="account",
        as_of=t0 + timedelta(minutes=3),
    )
    assert res_3m.status == "match"
    assert res_3m.match is True
    assert res_3m.is_unavailable is False
    assert res_3m.first_visible_at == t0 + timedelta(minutes=2)


async def test_as_of_revocation_end_to_end(
    nationalsim_setup: tuple[ManualClock, httpx.ASGITransport],
) -> None:
    """
    Proves that an entry revoked at t0+10m is returned to screen(as_of=t0+5m) and not returned
    to screen(as_of=t0+15m) when queried through NationalsimBankClient against the live ASGI app.
    """
    clock, transport = nationalsim_setup
    t0 = clock.now_ist()
    test_hash = "d" * 64

    # 1. Ingest report at t0
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http_client:
        ingest_resp = await http_client.post(
            "/ingest",
            json={
                "bank_id": "bank_0",
                "report_id": "rep_rev_001",
                "identifier_hash": test_hash,
                "identifier_type": "account",
                "reason_code": "mule_account",
                "reported_at": t0.isoformat(),
                "run_id": "run_e2e_001",
            },
        )
        assert ingest_resp.status_code == 200

        # 2. Advance clock to t0+10m and revoke the report
        clock.set_time(t0 + timedelta(minutes=10))
        revoke_resp = await http_client.post(
            "/revoke",
            json={
                "bank_id": "bank_0",
                "report_id": "rep_rev_001",
                "run_id": "run_e2e_001",
            },
        )
        assert revoke_resp.status_code == 200
        assert revoke_resp.json()["status"] == "revoked"

    # 3. Advance clock to t0+20m so queries are strictly in the past
    clock.set_time(t0 + timedelta(minutes=20))

    client = NationalsimBankClient(
        base_url="http://test",
        bank_id="bank_1",
        run_id="run_e2e_001",
        clock=clock,
        transport=transport,
    )

    # 4. as_of=t0+5m (active, visible, not yet revoked) -> listed / match
    res_5m: ScreenResult = await client.screen(
        identifier_hash=test_hash,
        identifier_type="account",
        as_of=t0 + timedelta(minutes=5),
    )
    assert res_5m.status == "match"
    assert res_5m.match is True
    assert res_5m.is_unavailable is False
    assert res_5m.first_visible_at == t0 + timedelta(minutes=2)

    # 5. as_of=t0+15m (after revocation) -> clean / no match
    res_15m: ScreenResult = await client.screen(
        identifier_hash=test_hash,
        identifier_type="account",
        as_of=t0 + timedelta(minutes=15),
    )
    assert res_15m.status == "no_match"
    assert res_15m.match is False
    assert res_15m.is_unavailable is False


async def test_non_participant_403_returns_unavailable_without_raising(
    nationalsim_setup: tuple[ManualClock, httpx.ASGITransport],
) -> None:
    """
    Proves that when a non-participating bank receives HTTP 403 Forbidden from the registry,
    the bank client returns status="unavailable" and match=None without raising an unhandled exception.
    """
    clock, transport = nationalsim_setup
    test_hash = "e" * 64

    # bank_non_participant is not configured in settings.banks
    client = NationalsimBankClient(
        base_url="http://test",
        bank_id="bank_non_participant",
        run_id="run_e2e_001",
        clock=clock,
        transport=transport,
    )

    # Calling screen() should NOT raise NotParticipantError or HTTPStatusError;
    # it must gracefully return an unavailable ScreenResult.
    result: ScreenResult = await client.screen(
        identifier_hash=test_hash,
        identifier_type="account",
    )

    assert result.status == "unavailable"
    assert result.match is None
    assert result.is_unavailable is True
    assert len(client.outages) == 1
    assert client.outages[0].bank_id == "bank_non_participant"

