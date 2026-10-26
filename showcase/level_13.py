"""
showcase/level_13.py

Showcase verification suite for Level 13: Mock national systems and event clock (Part 13a).
Runs all verification checks in-process against the real FastAPI application over httpx.ASGITransport.
The checks evaluate report listing delays, customer disputes, revocation authorization,
time replay consistency between 1x and 100x speeds, non-participant refusal, wrong-key isolation,
and exact single-minute visibility boundaries.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from core.event_clock import IST, ManualClock, ReplayClock
from nationalsim.adapter import (
    DEFAULT_FRAUD_REASON_CODE,
    IDENTIFIER_TYPE_ACCOUNT,
    IDENTIFIER_TYPE_PHONE,
)
from nationalsim.app import create_app
from nationalsim.keyed_hash import compute_identifier_hash
from showcase.framework import CheckDefinition, CheckOutcome

LEVEL_NUMBER = 13
LEVEL_TITLE = "Mock national systems and event clock (built: parts 13a, 13b)"
NEEDS_DATABASE = False

WHAT_YOU_SAW = (
    "Level 13 introduces the national systems simulator (nationalsim) and an injectable event clock. "
    "Part 13a adds the DPIP keyed registry mock, a 15-minute listing delay, customer disputes, and speed-invariant "
    "clock replays. Part 13b introduces registry intelligence: synthetic telecom recycling with 90-day statutory "
    "quarantines, dispute resolution timelines logged for N4 metrics, independent I4C intelligence feeds for all "
    "licensed banks, and emergent registry coverage across mule routing depths."
)

REPO_ROOT = Path(__file__).resolve().parent.parent
REQUIRED_FILES = [
    REPO_ROOT / "config" / "nationalsim.yaml",
    REPO_ROOT / "core" / "event_clock.py",
    REPO_ROOT / "nationalsim" / "app.py",
    REPO_ROOT / "nationalsim" / "keyed_hash.py",
    REPO_ROOT / "nationalsim" / "registry.py",
    REPO_ROOT / "nationalsim" / "adapter.py",
    REPO_ROOT / "evaluation" / "registry_coverage.py",
    REPO_ROOT / "world" / "phone_book.py",
    REPO_ROOT / "scenarios" / "edge_cases" / "13-recycled_phone_number.yaml",
]


class FakeMonotonicSource:
    """Controllable monotonic time source for ReplayClock testing."""

    def __init__(self, start_seconds: float = 1000.0) -> None:
        self.current_seconds = start_seconds

    def __call__(self) -> float:
        return self.current_seconds

    def advance(self, seconds: float) -> None:
        self.current_seconds += seconds


def find_missing_preconditions() -> list[str]:
    """Check that all required configuration and implementation files exist before running checks."""
    missing: list[str] = []
    for file_path in REQUIRED_FILES:
        if not file_path.is_file():
            missing.append(f"Missing required file: {file_path.relative_to(REPO_ROOT)}")
    return missing


async def check_1_report_then_screen(workspace: Any) -> CheckOutcome:
    """Proves that a fraud report is hidden during the listing delay and becomes visible right after.

    What it is for:
        When a bank confirms fraud, national registry rules impose a listing delay for validation.
        This check proves that the HTTP /screen endpoint reports 'no match' before the delay expires
        and matches immediately after the delay passes.

    Inputs:
        workspace: Unused placeholder provided by the showcase runner framework.

    What it returns:
        CheckOutcome recording the tested delay and matching status before and after.

    How it works:
        1. Set up a ManualClock starting at 10:00 IST on 2026-09-01.
        2. Create the real FastAPI app with a test HMAC key.
        3. Send a POST /ingest request reporting a mule bank account.
        4. Advance simulated time to 10:14 IST and verify POST /screen returns match=False.
        5. Advance simulated time to 10:15 IST and verify POST /screen returns match=True.

    Errors and side effects:
        Fails assertion if screening matches early or fails to match after the delay.

    Limits:
        Runs in-process via httpx ASGITransport without opening local network ports.
    """
    sim_time = datetime(2026, 9, 1, 10, 0, tzinfo=IST)
    clock = ManualClock(sim_time)
    test_key = "showcase_test_secret_key"
    app = create_app(
        clock=clock,
        key_provider=lambda: test_key,
        settings={
            "listing_delay_minutes": 15,
            "banks": {"bank_0": {"dpip_participation": True}},
        },
    )

    ident_hash = compute_identifier_hash(
        key=test_key,
        run_id="showcase_run_1",
        identifier_type=IDENTIFIER_TYPE_ACCOUNT,
        raw_value="acc_mule_scam_01",
        bank_index=0,
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://nationalsim.local") as client:
        # 1. Ingest report
        ingest_res = await client.post(
            "/ingest",
            json={
                "bank_id": "bank_0",
                "report_id": "REP-SHOW-01",
                "identifier_hash": ident_hash,
                "identifier_type": IDENTIFIER_TYPE_ACCOUNT,
                "reason_code": DEFAULT_FRAUD_REASON_CODE,
                "reported_at": sim_time.isoformat(),
                "run_id": "showcase_run_1",
            },
        )
        assert ingest_res.status_code == 200

        # 2. Check before listing delay (10:14 IST)
        clock.set_time(datetime(2026, 9, 1, 10, 14, tzinfo=IST))
        screen_before = await client.post(
            "/screen",
            json={
                "requesting_bank_id": "bank_0",
                "identifier_hash": ident_hash,
                "identifier_type": IDENTIFIER_TYPE_ACCOUNT,
                "run_id": "showcase_run_1",
            },
        )
        assert screen_before.status_code == 200
        before_data = screen_before.json()
        assert before_data["match"] is False

        # 3. Check at listing delay (10:15 IST)
        clock.set_time(datetime(2026, 9, 1, 10, 15, tzinfo=IST))
        screen_after = await client.post(
            "/screen",
            json={
                "requesting_bank_id": "bank_0",
                "identifier_hash": ident_hash,
                "identifier_type": IDENTIFIER_TYPE_ACCOUNT,
                "run_id": "showcase_run_1",
            },
        )
        assert screen_after.status_code == 200
        after_data = screen_after.json()
        assert after_data["match"] is True
        assert after_data["status"] == "listed"

    key_number = (
        f"delay 15m: at 10:14 match={before_data['match']}, at 10:15 match={after_data['match']}"
    )
    return CheckOutcome(
        key_number=key_number,
        evidence={
            "report_id": "REP-SHOW-01",
            "listing_delay_minutes": 15,
            "before_visible_match": before_data["match"],
            "at_visible_match": after_data["match"],
            "at_visible_status": after_data["status"],
        },
    )


async def check_2_revoke_and_dispute(workspace: Any) -> CheckOutcome:
    """Proves that a customer dispute marks an entry contested and revocation removes the match.

    What it is for:
        When a victim or falsely accused customer files a dispute through their bank, the registry
        must flag the listing as contested without dropping the match. If the reporting bank later
        revokes the report, the listing is completely deactivated and no longer matches.

    Inputs:
        workspace: Unused showcase runner parameter.

    What it returns:
        CheckOutcome recording the contested status after dispute and no-match after revocation.

    How it works:
        1. Ingest a report for an identifier hash and advance time past the listing delay.
        2. Submit a POST /dispute request; verify POST /screen returns status='contested' with match=True.
        3. Submit a POST /revoke request from the reporting bank.
        4. Verify POST /screen now returns match=False with status='none'.

    Errors and side effects:
        Raises AssertionError if dispute fails to flag contested or revoke fails to deactivate the match.

    Limits:
        Evaluates one dispute and revocation cycle on a single identifier.
    """
    clock = ManualClock(datetime(2026, 9, 1, 10, 0, tzinfo=IST))
    test_key = "showcase_test_secret_key"
    app = create_app(
        clock=clock,
        key_provider=lambda: test_key,
        settings={
            "listing_delay_minutes": 5,
            "banks": {"bank_0": {"dpip_participation": True}},
        },
    )

    ident_hash = compute_identifier_hash(test_key, "showcase_run_2", "account", "acc_victim_disputed", 0)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://nationalsim.local") as client:
        # Ingest report
        await client.post(
            "/ingest",
            json={
                "bank_id": "bank_0",
                "report_id": "REP-SHOW-02",
                "identifier_hash": ident_hash,
                "identifier_type": "account",
                "reason_code": "fraud",
                "reported_at": clock.now_ist().isoformat(),
                "run_id": "showcase_run_2",
            },
        )
        clock.advance(timedelta(minutes=10))

        # Dispute entry
        disp_res = await client.post(
            "/dispute",
            json={
                "bank_id": "bank_0",
                "identifier_hash": ident_hash,
                "reason_code": "customer_affidavit_fraud_claim",
                "run_id": "showcase_run_2",
            },
        )
        assert disp_res.status_code == 200

        # Screen after dispute
        screen_disp = await client.post(
            "/screen",
            json={
                "requesting_bank_id": "bank_0",
                "identifier_hash": ident_hash,
                "identifier_type": "account",
                "run_id": "showcase_run_2",
            },
        )
        disp_data = screen_disp.json()
        assert disp_data["match"] is True
        assert disp_data["status"] == "contested"

        # Revoke report
        revoke_res = await client.post(
            "/revoke",
            json={
                "bank_id": "bank_0",
                "report_id": "REP-SHOW-02",
                "run_id": "showcase_run_2",
            },
        )
        assert revoke_res.status_code == 200

        # Screen after revoke
        screen_rev = await client.post(
            "/screen",
            json={
                "requesting_bank_id": "bank_0",
                "identifier_hash": ident_hash,
                "identifier_type": "account",
                "run_id": "showcase_run_2",
            },
        )
        rev_data = screen_rev.json()
        assert rev_data["match"] is False
        assert rev_data["status"] == "none"

    key_number = f"disputed status={disp_data['status']}, revoked match={rev_data['match']}"
    return CheckOutcome(
        key_number=key_number,
        evidence={
            "disputed_status": disp_data["status"],
            "disputed_match": disp_data["match"],
            "revoked_status": rev_data["status"],
            "revoked_match": rev_data["match"],
        },
    )


async def check_3_clock_consistency(workspace: Any) -> CheckOutcome:
    """Proves that simulating time at 1x speed and 100x speed gives identical match times.

    What it is for:
        Fast simulation runs must run 100 times faster than wall time without changing any
        fraud detection answers or decision order. This check proves ReplayClock guarantees
        identical event-driven outcomes across different replay multipliers.

    Inputs:
        workspace: Unused showcase runner parameter.

    What it returns:
        CheckOutcome recording the identical simulated match timestamp at 1x and 100x.

    How it works:
        1. Set up two separate ReplayClock instances starting at 10:00 IST on 2026-09-01.
        2. Wire one instance to run at speed 1.0 and the second to run at speed 100.0.
        3. Advance wall time for 1x by 15 minutes (900 seconds) and for 100x by 9 seconds.
        4. Assert that both clocks report the exact same simulated IST time and screening answers.

    Errors and side effects:
        Fails assertion if 1x and 100x clocks disagree on time or screening visibility.

    Limits:
        Driven by fake wall clocks; does not sleep or depend on computer hardware performance.
    """
    sim_start = datetime(2026, 9, 1, 10, 0, tzinfo=IST)
    test_key = "showcase_speed_key"

    source_1x = FakeMonotonicSource(start_seconds=100.0)
    clock_1x = ReplayClock(simulated_start=sim_start, speed=1.0, wall_source=source_1x)
    app_1x = create_app(
        clock=clock_1x,
        key_provider=lambda: test_key,
        settings={
            "listing_delay_minutes": 15,
            "banks": {"bank_0": {"dpip_participation": True}},
        },
    )

    source_100x = FakeMonotonicSource(start_seconds=500.0)
    clock_100x = ReplayClock(simulated_start=sim_start, speed=100.0, wall_source=source_100x)
    app_100x = create_app(
        clock=clock_100x,
        key_provider=lambda: test_key,
        settings={
            "listing_delay_minutes": 15,
            "banks": {"bank_0": {"dpip_participation": True}},
        },
    )

    target_hash = compute_identifier_hash(test_key, "showcase_run_3", "account", "acc_speed_sync", 0)

    transport_1x = httpx.ASGITransport(app=app_1x)
    transport_100x = httpx.ASGITransport(app=app_100x)

    async with (
        httpx.AsyncClient(transport=transport_1x, base_url="http://nationalsim-1x.local") as client_1x,
        httpx.AsyncClient(transport=transport_100x, base_url="http://nationalsim-100x.local") as client_100x,
    ):
        # Ingest in both at initial simulated time
        payload = {
            "bank_id": "bank_0",
            "report_id": "REP-SPEED-01",
            "identifier_hash": target_hash,
            "identifier_type": "account",
            "reason_code": "fraud",
            "reported_at": sim_start.isoformat(),
            "run_id": "showcase_run_3",
        }
        await client_1x.post("/ingest", json=payload)
        await client_100x.post("/ingest", json=payload)

        # Advance 14 minutes simulated time (840s wall time at 1x; 8.4s wall time at 100x)
        source_1x.advance(840.0)
        source_100x.advance(8.4)
        assert clock_1x.now_ist() == clock_100x.now_ist()

        res_14m_1x = (await client_1x.post(
            "/screen",
            json={"requesting_bank_id": "bank_0", "identifier_hash": target_hash,
                  "identifier_type": "account", "run_id": "showcase_run_3"},
        )).json()
        res_14m_100x = (await client_100x.post(
            "/screen",
            json={"requesting_bank_id": "bank_0", "identifier_hash": target_hash,
                  "identifier_type": "account", "run_id": "showcase_run_3"},
        )).json()
        assert res_14m_1x["match"] is False
        assert res_14m_100x["match"] is False

        # Advance 1 more minute simulated time (60s wall time at 1x; 0.6s wall time at 100x)
        source_1x.advance(60.0)
        source_100x.advance(0.6)
        assert clock_1x.now_ist() == clock_100x.now_ist()

        res_15m_1x = (await client_1x.post(
            "/screen",
            json={"requesting_bank_id": "bank_0", "identifier_hash": target_hash,
                  "identifier_type": "account", "run_id": "showcase_run_3"},
        )).json()
        res_15m_100x = (await client_100x.post(
            "/screen",
            json={"requesting_bank_id": "bank_0", "identifier_hash": target_hash,
                  "identifier_type": "account", "run_id": "showcase_run_3"},
        )).json()
        assert res_15m_1x["match"] is True
        assert res_15m_100x["match"] is True
        assert res_15m_1x["status"] == res_15m_100x["status"] == "listed"

    sim_match_time = clock_1x.now_ist().isoformat()
    key_number = f"1x and 100x match identical at {sim_match_time}"
    return CheckOutcome(
        key_number=key_number,
        evidence={
            "simulated_match_time": sim_match_time,
            "1x_match": res_15m_1x["match"],
            "100x_match": res_15m_100x["match"],
        },
    )


async def check_4_non_participant_and_wrong_key(workspace: Any) -> CheckOutcome:
    """Proves that non-participating banks are refused and wrong secret keys prevent matches.

    What it is for:
        DPIP operates on shared secrets held only by enrolled institutions. Banks without the key
        must be denied access with HTTP 403, and calculations using an incorrect HMAC key must
        never produce matching hashes against the registry.

    Inputs:
        workspace: Unused showcase runner parameter.

    What it returns:
        CheckOutcome recording the 403 status code for the outsider and no-match for wrong keys.

    How it works:
        1. Set up an app where bank_outsider has dpip_participation=False.
        2. Attempt screening from bank_outsider and verify HTTP 403 Forbidden is returned.
        3. Ingest a report using correct_key.
        4. Screen using a hash generated with wrong_key and verify match=False.

    Errors and side effects:
        Fails assertion if an outsider gains access or a mismatched key mistakenly matches.

    Limits:
        Uses a static two-bank test configuration.
    """
    clock = ManualClock()
    correct_key = "correct_secret_key"
    wrong_key = "wrong_attacker_key"

    app = create_app(
        clock=clock,
        key_provider=lambda: correct_key,
        settings={
            "listing_delay_minutes": 15,
            "banks": {
                "bank_0": {"dpip_participation": True},
                "bank_outsider": {"dpip_participation": False},
            },
        },
    )

    correct_hash = compute_identifier_hash(correct_key, "showcase_run_4", "account", "acc_shared_target", 0)
    wrong_hash = compute_identifier_hash(wrong_key, "showcase_run_4", "account", "acc_shared_target", 0)
    assert correct_hash != wrong_hash

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://nationalsim.local") as client:
        # Ingest report with correct hash
        await client.post(
            "/ingest",
            json={
                "bank_id": "bank_0",
                "report_id": "REP-SHOW-04",
                "identifier_hash": correct_hash,
                "identifier_type": "account",
                "reason_code": "fraud",
                "reported_at": clock.now_ist().isoformat(),
                "run_id": "showcase_run_4",
            },
        )
        clock.advance(timedelta(minutes=20))

        # 1. Non-participating bank screening attempt is refused with 403
        outsider_res = await client.post(
            "/screen",
            json={
                "requesting_bank_id": "bank_outsider",
                "identifier_hash": correct_hash,
                "identifier_type": "account",
                "run_id": "showcase_run_4",
            },
        )
        assert outsider_res.status_code == 403
        outsider_detail = outsider_res.json()["detail"]

        # 2. Screening with wrong key hash produces no match
        wrong_res = await client.post(
            "/screen",
            json={
                "requesting_bank_id": "bank_0",
                "identifier_hash": wrong_hash,
                "identifier_type": "account",
                "run_id": "showcase_run_4",
            },
        )
        assert wrong_res.status_code == 200
        wrong_data = wrong_res.json()
        assert wrong_data["match"] is False

    key_number = f"outsider status={outsider_res.status_code}, wrong key match={wrong_data['match']}"
    return CheckOutcome(
        key_number=key_number,
        evidence={
            "outsider_http_status": outsider_res.status_code,
            "outsider_detail": outsider_detail,
            "wrong_key_match": wrong_data["match"],
        },
    )


async def check_5_boundary_minute(workspace: Any) -> CheckOutcome:
    """Proves that screening is strictly bounded at the exact minute of visibility.

    What it is for:
        Delay enforcement must not drift by a single minute. Screening at visible_at - 1 minute
        must return no match, while screening at visible_at must return an active match.

    Inputs:
        workspace: Unused showcase runner parameter.

    What it returns:
        CheckOutcome recording the boundary transition between minute 14 and minute 15.

    How it works:
        1. Ingest a report at 10:00 IST with a 15-minute listing delay.
        2. Set clock to 10:14 IST (visible_at - 1 minute) and assert match=False.
        3. Advance clock by exactly 60 seconds to 10:15 IST and assert match=True.

    Errors and side effects:
        Raises AssertionError if the transition happens too early or fails at the exact boundary.

    Limits:
        Tests one single-minute boundary step.
    """
    sim_start = datetime(2026, 9, 1, 10, 0, tzinfo=IST)
    clock = ManualClock(sim_start)
    test_key = "showcase_boundary_key"
    app = create_app(
        clock=clock,
        key_provider=lambda: test_key,
        settings={
            "listing_delay_minutes": 15,
            "banks": {"bank_0": {"dpip_participation": True}},
        },
    )

    ident_hash = compute_identifier_hash(test_key, "showcase_run_5", "account", "acc_boundary_target", 0)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://nationalsim.local") as client:
        # Ingest report
        await client.post(
            "/ingest",
            json={
                "bank_id": "bank_0",
                "report_id": "REP-BOUND-01",
                "identifier_hash": ident_hash,
                "identifier_type": "account",
                "reason_code": "fraud",
                "reported_at": sim_start.isoformat(),
                "run_id": "showcase_run_5",
            },
        )

        # 1. Exactly 1 minute before visibility (10:14 IST)
        clock.set_time(datetime(2026, 9, 1, 10, 14, tzinfo=IST))
        res_before = await client.post(
            "/screen",
            json={
                "requesting_bank_id": "bank_0",
                "identifier_hash": ident_hash,
                "identifier_type": "account",
                "run_id": "showcase_run_5",
            },
        )
        before_data = res_before.json()
        assert before_data["match"] is False

        # 2. Exactly at visibility (10:15 IST)
        clock.set_time(datetime(2026, 9, 1, 10, 15, tzinfo=IST))
        res_at = await client.post(
            "/screen",
            json={
                "requesting_bank_id": "bank_0",
                "identifier_hash": ident_hash,
                "identifier_type": "account",
                "run_id": "showcase_run_5",
            },
        )
        at_data = res_at.json()
        assert at_data["match"] is True
        assert at_data["status"] == "listed"

    key_number = f"boundary minute 14 match={before_data['match']}, minute 15 match={at_data['match']}"
    return CheckOutcome(
        key_number=key_number,
        evidence={
            "minute_14_match": before_data["match"],
            "minute_15_match": at_data["match"],
            "minute_15_status": at_data["status"],
        },
    )


async def check_6_recycled_phone_dispute(workspace: Any) -> CheckOutcome:
    """Proves that a recycled phone wrongly matches a genuine person until a customer dispute resolves it.

    What it is for:
        When a telecom provider reassigns a disconnected mobile number, innocent new subscribers
        risk inheriting collateral fraud suspicion from prior mule operators. This check proves that
        while the registry initially matches the historical listing, filing a dispute on behalf of the customer
        automatically schedules resolution after the configured delay and transitions screening to resolved.

    Inputs:
        workspace: Unused placeholder provided by the showcase runner framework.

    What it returns:
        CheckOutcome recording match before dispute, contested status, resolved status, and exact minutes_to_resolve.

    How it works:
        1. Set up a ManualClock starting at 10:00 IST on 2026-09-01.
        2. Create the FastAPI app with bank_0 enrolled in DPIP and dispute_resolution_minutes configured to 60.
        3. Register a public telecom reassignment for a phone hash at simulation minute -10.
        4. Ingest an old mule report reported before reassignment (at minute -200).
        5. Advance time by 20 minutes; verify screening matches with status 'listed'.
        6. Submit POST /dispute on behalf of the customer via bank_0; verify response returns status 'resolved'
           and minutes_to_resolve=60.
        7. Advance time by 59 minutes (before delay expires); verify screening is status 'contested'.
        8. Advance time by 1 more minute (60 total); verify screening returns match=False and status 'resolved'.

    Errors and side effects:
        Raises AssertionError if the dispute fails to schedule resolution or screening fails to transition.

    Limits:
        Runs in-process against the real FastAPI endpoints via httpx.ASGITransport.
    """
    sim_time = datetime(2026, 9, 1, 10, 0, tzinfo=IST)
    clock = ManualClock(sim_time)
    test_key = "showcase_test_secret_key"
    app = create_app(
        clock=clock,
        key_provider=lambda: test_key,
        settings={
            "listing_delay_minutes": 15,
            "dispute_resolution_minutes": 60,
            "banks": {"bank_0": {"dpip_participation": True}},
        },
    )

    phone_num = "sim-0000000888"
    run_id = "showcase_run_13b"
    phone_hash = compute_identifier_hash(test_key, run_id, IDENTIFIER_TYPE_PHONE, phone_num)

    # Register telecom reassignment at minute -10 (after 90 days statutory quarantine)
    app.state.registry.register_telecom_reassignment(
        identifier_hash=phone_hash,
        reassigned_minute=-10,
        run_id=run_id,
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # Ingest old report reported before reassignment (minute -200)
        await client.post(
            "/ingest",
            json={
                "bank_id": "bank_0",
                "report_id": "R-SHOWCASE-OLD-MULE",
                "identifier_hash": phone_hash,
                "identifier_type": IDENTIFIER_TYPE_PHONE,
                "reason_code": DEFAULT_FRAUD_REASON_CODE,
                "reported_at": (sim_time - timedelta(minutes=200)).isoformat(),
                "run_id": run_id,
            },
        )
        # Tag entry's reported_minute to pre-reassignment minute (-200)
        for entry in app.state.registry._entries.values():
            if entry.identifier_hash == phone_hash:
                entry.reported_minute = -200

        clock.advance(timedelta(minutes=20))

        # 1. Screen initially matches against the innocent customer
        res_before = await client.post(
            "/screen",
            json={
                "requesting_bank_id": "bank_0",
                "identifier_hash": phone_hash,
                "identifier_type": IDENTIFIER_TYPE_PHONE,
                "run_id": run_id,
            },
        )
        before_data = res_before.json()
        assert before_data["match"] is True
        assert before_data["status"] == "listed"

        # 2. File dispute on behalf of customer via bank
        res_disp = await client.post(
            "/dispute",
            json={
                "bank_id": "bank_0",
                "identifier_hash": phone_hash,
                "reason_code": "recycled_number_dispute",
                "run_id": run_id,
                "filed_by": "customer_via_bank",
            },
        )
        disp_data = res_disp.json()
        assert disp_data["status"] == "resolved"
        assert disp_data["minutes_to_resolve"] == 60

        # 3. Before 60m delay expires (at 59m): still contested
        clock.advance(timedelta(minutes=59))
        res_mid = await client.post(
            "/screen",
            json={
                "requesting_bank_id": "bank_0",
                "identifier_hash": phone_hash,
                "identifier_type": IDENTIFIER_TYPE_PHONE,
                "run_id": run_id,
            },
        )
        mid_data = res_mid.json()
        assert mid_data["match"] is True
        assert mid_data["status"] == "contested"

        # 4. At 60m (1 minute later): resolved match=False
        clock.advance(timedelta(minutes=1))
        res_after = await client.post(
            "/screen",
            json={
                "requesting_bank_id": "bank_0",
                "identifier_hash": phone_hash,
                "identifier_type": IDENTIFIER_TYPE_PHONE,
                "run_id": run_id,
            },
        )
        after_data = res_after.json()
        assert after_data["match"] is False
        assert after_data["status"] == "resolved"

    key_number = f"wrong match resolved in {disp_data['minutes_to_resolve']}m; match before=True, after=False"
    return CheckOutcome(
        key_number=key_number,
        evidence={
            "phone_hash": phone_hash,
            "status_before": before_data["status"],
            "minutes_to_resolve": disp_data["minutes_to_resolve"],
            "status_mid": mid_data["status"],
            "match_after": after_data["match"],
            "status_after": after_data["status"],
        },
    )


async def check_7_non_participant_i4c_feed(workspace: Any) -> CheckOutcome:
    """Proves that a non-participating bank gets no DPIP match but still sees the I4C feed.

    What it is for:
        DPIP screening requires formal bilateral enrollment with keyed HMAC hashes, while
        the I4C national suspect registry feed is broadcast to all licensed financial institutions.
        This check proves that a non-participating bank is refused screening with HTTP 403 but
        successfully pulls broadcast telephone records from the shared I4C feed endpoint.

    Inputs:
        workspace: Unused placeholder provided by the showcase runner framework.

    What it returns:
        CheckOutcome recording 403 refusal on DPIP screening and 200 success on I4C feed ingestion.

    How it works:
        1. Set up app with bank_0 enrolled in DPIP and bank_2 not enrolled.
        2. Ingest a fraud report via bank_0 into DPIP (which automatically mirrors to I4C feed).
        3. Advance time past listing delay.
        4. Bank_2 calls POST /screen; verify HTTP 403 is returned.
        5. Bank_2 calls GET /i4c/feed; verify HTTP 200 is returned with populated entries.

    Errors and side effects:
        Raises AssertionError if screening succeeds or I4C feed is refused.

    Limits:
        Evaluates HTTP gateway access controls in-process.
    """
    sim_time = datetime(2026, 9, 1, 10, 0, tzinfo=IST)
    clock = ManualClock(sim_time)
    test_key = "showcase_dpip_key"
    i4c_key = "showcase_i4c_key"
    app = create_app(
        clock=clock,
        key_provider=lambda: test_key,
        i4c_key_provider=lambda: i4c_key,
        settings={
            "listing_delay_minutes": 15,
            "banks": {
                "bank_0": {"dpip_participation": True},
                "bank_2": {"dpip_participation": False},
            },
        },
    )

    run_id = "showcase_run_i4c"
    phone_num = "sim-0000000777"
    phone_hash = compute_identifier_hash(test_key, run_id, IDENTIFIER_TYPE_PHONE, phone_num)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # Ingest report through enrolled bank_0
        res_ingest = await client.post(
            "/ingest",
            json={
                "bank_id": "bank_0",
                "report_id": "R-I4C-001",
                "identifier_hash": phone_hash,
                "identifier_type": IDENTIFIER_TYPE_PHONE,
                "reason_code": DEFAULT_FRAUD_REASON_CODE,
                "reported_at": sim_time.isoformat(),
                "run_id": run_id,
            },
        )
        assert res_ingest.status_code == 200

        clock.advance(timedelta(minutes=20))

        # Non-participating bank_2 attempts DPIP screening -> 403
        res_screen = await client.post(
            "/screen",
            json={
                "requesting_bank_id": "bank_2",
                "identifier_hash": phone_hash,
                "identifier_type": IDENTIFIER_TYPE_PHONE,
                "run_id": run_id,
            },
        )
        assert res_screen.status_code == 403
        assert "not a participating DPIP member" in res_screen.json()["detail"]

        # Non-participating bank_2 accesses I4C feed -> 200
        res_feed = await client.get("/i4c/feed", params={"bank_id": "bank_2", "run_id": run_id})
        assert res_feed.status_code == 200
        feed_data = res_feed.json()
        assert feed_data["entries_count"] >= 1
        assert any(item["identifier_hash"] == phone_hash for item in feed_data["entries"])

    key_number = f"DPIP screen 403 forbidden, I4C feed 200 OK ({feed_data['entries_count']} items)"
    return CheckOutcome(
        key_number=key_number,
        evidence={
            "screen_status_code": res_screen.status_code,
            "feed_status_code": res_feed.status_code,
            "feed_item_count": feed_data["entries_count"],
        },
    )


async def check_8_registry_coverage_by_mule_depth(workspace: Any) -> CheckOutcome:
    """Proves that registry coverage is measured across mule depths and satisfies layer-1 dominance.

    What it is for:
        Victim complaints emerge almost exclusively at layer 1 (the initial mule accounts victims pay directly).
        This check measures emergent coverage across depths in a fraud-on world and confirms that observed
        coverage for layer 1 strictly exceeds deeper mule coverage, displaying configured bounds beside it.

    Inputs:
        workspace: Unused placeholder provided by the showcase runner framework.

    What it returns:
        CheckOutcome recording observed fractions, configured reference ranges, and scatter chart data.

    How it works:
        1. Executes a fast fraud-on world with a scam-and-route syndicate (seed 102, 60 people, 61 days).
        2. Evaluates registry coverage across mule depth tiers using evaluate_world_registry_coverage.
        3. Asserts the pass condition: layer 1 coverage strictly exceeds deeper coverage.
        4. Constructs a scatter chart comparing observed coverage against configured range bounds.

    Errors and side effects:
        Raises AssertionError if layer 1 coverage does not exceed deeper coverage.

    Limits:
        Uses an in-process synthetic world run to evaluate emergent reporting dynamics.
    """
    from evaluation.registry_coverage import evaluate_world_registry_coverage
    from world.fast_mode import FastRunSettings
    from world.gang_config import BlockStep, GangBudget, GangConfig
    from world.interactive_records import RunnerSettings
    from world.interactive_runner import run_interactive_world

    steps = (
        BlockStep("acquire", (("count", 8), ("origin_mix", "all_rented"))),
        BlockStep("scam", (("receiving_mules", 2), ("victims_per_receiving_mule", 12))),
        BlockStep("route", (("layers", 2), ("accounts_per_layer", 2))),
    )
    budget = GangBudget(money_paise=50_000_000, time_minutes=500_000, max_risk=0.99)
    gang = GangConfig("g_eval", 1, budget, None, steps, 1440)
    fast_settings = FastRunSettings(
        seed=102,
        n_people=60,
        n_shops=12,
        n_days=61,
        range_name="training",
    )
    runner_settings = RunnerSettings(fraud_enabled=True)
    output = run_interactive_world(fast_settings, runner_settings, gang_configs=(gang,))

    result = evaluate_world_registry_coverage(output)
    assert result.passed is True, (
        f"Layer-1 coverage {result.layer1_coverage:.4f} must exceed deeper coverage {result.deeper_coverage:.4f}"
    )

    m1 = result.by_depth["1"]
    m2 = result.by_depth["2"]
    m3 = result.by_depth["3+"]
    m_deep = result.by_depth["deeper"]

    chart = {
        "type": "scatter",
        "title": "Registry coverage by mule depth vs configured bounds",
        "x_label": "Mule depth layer",
        "y_label": "Coverage fraction",
        "series": [
            {
                "name": "Observed coverage",
                "points": [
                    [1, m1.coverage],
                    [2, m2.coverage],
                    [3, m3.coverage],
                ],
            },
            {
                "name": "Configured range (low)",
                "points": [
                    [1, result.layer1_range[0]],
                    [2, result.deeper_range[0]],
                    [3, result.deeper_range[0]],
                ],
            },
            {
                "name": "Configured range (high)",
                "points": [
                    [1, result.layer1_range[1]],
                    [2, result.deeper_range[1]],
                    [3, result.deeper_range[1]],
                ],
            },
        ],
    }

    key_number = (
        f"layer 1: {m1.coverage * 100:.1f}% ({m1.reported_mules}/{m1.total_mules}), "
        f"deeper: {m_deep.coverage * 100:.1f}% ({m_deep.reported_mules}/{m_deep.total_mules})"
    )

    return CheckOutcome(
        key_number=key_number,
        evidence={
            "layer1_coverage": m1.coverage,
            "layer1_range": list(result.layer1_range),
            "deeper_coverage": m_deep.coverage,
            "deeper_range": list(result.deeper_range),
            "table_text": result.table_text,
        },
        chart=chart,
    )


async def check_9_unlisted_bank_refused_and_shared_dispute(workspace: Any) -> CheckOutcome:
    """Proves that an unlisted bank is refused with HTTP 403, and shared-household disputes remain contested.

    What it is for:
        Defense-in-depth access controls must reject completely unlisted institutions, while
        shared-household phone numbers linked to ongoing illicit activity must not be automatically
        resolved by customer claims alone.

    Inputs:
        workspace: Unused placeholder provided by the showcase runner framework.

    What it returns:
        CheckOutcome recording 403 refusal for unlisted banks and contested status retention.

    How it works:
        1. Set up app with bank_0 configured.
        2. Call POST /screen with an unlisted bank ('bank_unlisted_99'); verify HTTP 403 is returned.
        3. Ingest a fraud report on a shared household phone where report minute is within the current era.
        4. Submit POST /dispute for the shared number; verify response status is 'contested'
           and minutes_to_resolve is None.
        5. Advance time by 100 minutes; verify screening remains match=True and status 'contested'.

    Errors and side effects:
        Raises AssertionError if unlisted bank screening succeeds or shared dispute is resolved.

    Limits:
        Evaluates negative security boundaries in-process.
    """
    sim_time = datetime(2026, 9, 1, 10, 0, tzinfo=IST)
    clock = ManualClock(sim_time)
    test_key = "showcase_test_secret_key"
    app = create_app(
        clock=clock,
        key_provider=lambda: test_key,
        settings={
            "listing_delay_minutes": 15,
            "dispute_resolution_minutes": 60,
            "banks": {"bank_0": {"dpip_participation": True}},
        },
    )

    phone_num = "sim-0000000999"
    run_id = "showcase_run_neg"
    phone_hash = compute_identifier_hash(test_key, run_id, IDENTIFIER_TYPE_PHONE, phone_num)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Unlisted bank attempted screening -> 403
        res_unlisted = await client.post(
            "/screen",
            json={
                "requesting_bank_id": "bank_unlisted_99",
                "identifier_hash": phone_hash,
                "identifier_type": IDENTIFIER_TYPE_PHONE,
                "run_id": run_id,
            },
        )
        assert res_unlisted.status_code == 403
        assert "not a participating DPIP member" in res_unlisted.json()["detail"]

        # 2. Ingest active fraud report on shared household phone
        res_ingest = await client.post(
            "/ingest",
            json={
                "bank_id": "bank_0",
                "report_id": "R-SHARED-ACTIVE",
                "identifier_hash": phone_hash,
                "identifier_type": IDENTIFIER_TYPE_PHONE,
                "reason_code": DEFAULT_FRAUD_REASON_CODE,
                "reported_at": sim_time.isoformat(),
                "run_id": run_id,
            },
        )
        assert res_ingest.status_code == 200
        clock.advance(timedelta(minutes=40))

        # 3. Customer disputes the shared household listing
        res_disp = await client.post(
            "/dispute",
            json={
                "bank_id": "bank_0",
                "identifier_hash": phone_hash,
                "reason_code": "shared_household_dispute",
                "run_id": run_id,
                "filed_by": "customer_via_bank",
            },
        )
        disp_data = res_disp.json()
        assert disp_data["status"] == "contested"
        assert disp_data["minutes_to_resolve"] is None

        # 4. Advance time by 100 minutes (past normal resolution delay) -> remains contested
        clock.advance(timedelta(minutes=100))
        res_screen = await client.post(
            "/screen",
            json={
                "requesting_bank_id": "bank_0",
                "identifier_hash": phone_hash,
                "identifier_type": IDENTIFIER_TYPE_PHONE,
                "run_id": run_id,
            },
        )
        screen_data = res_screen.json()
        assert screen_data["match"] is True
        assert screen_data["status"] == "contested"

    key_number = f"unlisted bank refused 403; shared household dispute status '{screen_data['status']}' after 100m"
    return CheckOutcome(
        key_number=key_number,
        evidence={
            "unlisted_status_code": res_unlisted.status_code,
            "dispute_status": disp_data["status"],
            "screen_match_after_100m": screen_data["match"],
            "screen_status_after_100m": screen_data["status"],
        },
    )


def get_checks(quick: bool = False) -> list[CheckDefinition]:
    """Construct the ordered list of Level 13 showcase verification checks.

    What it is for:
        Supplies the showcase runner framework with the suite of checks evaluating Level 13 (parts a and b).
        All checks run in-process and satisfy the coverage requirement across normal, edge, and negative checks.

    Inputs:
        quick: Boolean flag indicating if quick mode is requested. All Level 13 checks are fast.

    What it returns:
        List of 9 CheckDefinition containers tagged with their respective part.

    How it works:
        1. Construct CheckDefinitions for checks 1 through 5 (Part 13a).
        2. Construct CheckDefinitions for checks 6 through 9 (Part 13b).
        3. Return all 9 checks in order.

    Errors and side effects:
        None. Pure constructor returning check definitions.
    """
    check_1 = CheckDefinition(
        name="1. report then screen: no match before the delay, match after",
        kind="normal",
        proves=(
            "A bank fraud report submitted into the mock national registry remains invisible "
            "during the 15-minute listing delay and becomes visible immediately after."
        ),
        run=check_1_report_then_screen,
        likely_cause="Listing delay calculation failed or screening matched before visible_at elapsed.",
        where_to_look="nationalsim/registry.py, nationalsim/app.py, config/nationalsim.yaml",
        part="a",
    )
    check_2 = CheckDefinition(
        name="2. revoke removes the match; dispute marks it contested",
        kind="edge",
        proves=(
            "Filing a customer dispute changes a listed identifier to contested, and revoking a report "
            "by the reporting bank removes the match completely."
        ),
        run=check_2_revoke_and_dispute,
        likely_cause="Dispute status precedence failed or revocation did not remove matching entries.",
        where_to_look="nationalsim/registry.py, nationalsim/app.py",
        part="a",
    )
    check_3 = CheckDefinition(
        name="3. the same scenario at 1x and 100x gives identical simulated match times",
        kind="consistency",
        proves=(
            "Simulating the exact same event timeline under 1x speed and 100x speed produces identical "
            "event-time screening answers and match timestamps."
        ),
        run=check_3_clock_consistency,
        likely_cause="ReplayClock time advancement drifted or wall clock leaked into simulated time.",
        where_to_look="core/event_clock.py, nationalsim/registry.py",
        part="a",
    )
    check_4 = CheckDefinition(
        name="4. a bank without the key / not participating is refused, and a wrong-key hash never matches",
        kind="negative",
        proves=(
            "A bank not enrolled in DPIP is denied screening with HTTP 403, and an identifier hashed "
            "with the wrong secret key never matches an existing listing."
        ),
        run=check_4_non_participant_and_wrong_key,
        likely_cause="Non-participant guard was bypassed or HMAC hashing failed to isolate different keys.",
        where_to_look="nationalsim/app.py, nationalsim/registry.py, nationalsim/keyed_hash.py",
        part="a",
    )
    check_5 = CheckDefinition(
        name="5. screening exactly at the boundary minute (visible_at - 1 gives no match, visible_at gives a match)",
        kind="edge",
        proves=(
            "Screening an identifier exactly 60 seconds before visible_at yields no match, while "
            "screening at the exact visible_at minute yields an active match."
        ),
        run=check_5_boundary_minute,
        likely_cause="Strict inequality vs less-than-or-equal comparison boundary error in registry visibility check.",
        where_to_look="nationalsim/registry.py, nationalsim/app.py",
        part="a",
    )
    check_6 = CheckDefinition(
        name="6. recycled number wrongly matches genuine person, dispute resolves it",
        kind="edge",
        proves=(
            "An innocent customer holding a recycled phone number matching a prior mule listing "
            "has the match resolved after the dispute resolution delay, logging minutes_to_resolve for N4."
        ),
        run=check_6_recycled_phone_dispute,
        likely_cause="Dispute resolution logic failed to verify pre-reassignment timeline or delay calculation.",
        where_to_look="nationalsim/registry.py, nationalsim/app.py, config/nationalsim.yaml",
        part="b",
    )
    check_7 = CheckDefinition(
        name="7. non-participating bank gets no DPIP match but still sees I4C feed",
        kind="edge",
        proves=(
            "A licensed bank not participating in DPIP is denied DPIP screening with HTTP 403, "
            "but successfully pulls unencrypted suspect listings from the I4C feed."
        ),
        run=check_7_non_participant_i4c_feed,
        likely_cause="DPIP enrollment was checked on I4C feed endpoint or I4C route was blocked.",
        where_to_look="nationalsim/app.py, nationalsim/i4c_feed.py",
        part="b",
    )
    check_8 = CheckDefinition(
        name="8. registry coverage by mule depth, with configured ranges shown beside it",
        kind="normal",
        proves=(
            "National registry coverage is emergent and strictly higher on layer 1 receiver mules "
            "than deeper routing mules, matching empirical reference intervals."
        ),
        run=check_8_registry_coverage_by_mule_depth,
        likely_cause="Layer 1 coverage failed to exceed deeper coverage or world runner failed to generate complaints.",
        where_to_look="evaluation/registry_coverage.py, config/world_ranges.yaml",
        part="b",
    )
    check_9 = CheckDefinition(
        name="9. unlisted bank is refused, and shared-household dispute is not wrongly upheld",
        kind="negative",
        proves=(
            "An unlisted institution is denied DPIP screening with HTTP 403, and a dispute on a "
            "shared household phone with active fraud is not auto-resolved and stays contested."
        ),
        run=check_9_unlisted_bank_refused_and_shared_dispute,
        likely_cause="Unlisted bank guard allowed screening or shared-household dispute was erroneously resolved.",
        where_to_look="nationalsim/app.py, nationalsim/registry.py",
        part="b",
    )
    return [
        check_1,
        check_2,
        check_3,
        check_4,
        check_5,
        check_6,
        check_7,
        check_8,
        check_9,
    ]
