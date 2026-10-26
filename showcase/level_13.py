"""
showcase/level_13.py

Showcase verification suite for Level 13: Mock national systems and event clock (Part 13a).
Runs all verification checks in-process against the real FastAPI application over httpx.ASGITransport.
The checks evaluate report listing delays, customer disputes, revocation authorization,
time replay consistency between 1x and 100x speeds, non-participant refusal, wrong-key isolation,
and exact single-minute visibility boundaries.
"""

import asyncio
import logging
import statistics
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import asyncpg
import httpx
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from api.services.outbox_relay import start_producer
from core.config import config
from core.event_clock import IST, ManualClock, ReplayClock, WallClock
from core.logging import configure_logging, get_logger
from core.raw_identifier_scan import scan_all_five_places, scan_nationalsim_store
from nationalsim.adapter import (
    DEFAULT_FRAUD_REASON_CODE,
    IDENTIFIER_TYPE_ACCOUNT,
    IDENTIFIER_TYPE_PHONE,
)
from nationalsim.app import create_app
from nationalsim.bank_client import NationalsimBankClient
from nationalsim.hold_flow import (
    HoldRequest,
    OutgoingTransfer,
    build_hold_payload,
    classify_hold_outcome,
    decide_one_hop_forwarding,
)
from nationalsim.keyed_hash import compute_identifier_hash
from nationalsim.national_outbox import relay_national_outbox, write_event
from nationalsim.registry import RegistryCore
from showcase.framework import CheckDefinition, CheckOutcome, NotRun
from showcase.isolation import get_showcase_database_url
from showcase.level_12_support_b import get_showcase_b_settings
from world.fast_mode import FastRunSettings
from world.gang_config import BlockStep, GangBudget, GangConfig
from world.hold_replay import (
    measure_hold_latencies_by_window,
    reconstruct_account_balance,
    run_two_pass_replay,
)
from world.interactive_records import RunnerSettings
from world.ledger import Ledger
from world.phone_book import build_phone_book
from world.world_configs import read_world_configs

LEVEL_NUMBER = 13
LEVEL_TITLE = "Mock national systems and event clock (built: parts 13a, 13b, 13c)"
NEEDS_DATABASE = False

WHAT_YOU_SAW = (
    "Level 13 introduces the national systems simulator (nationalsim) and an injectable event clock. "
    "Part 13a adds the DPIP keyed registry mock, a 15-minute listing delay, customer disputes, and speed-invariant "
    "clock replays. Part 13b introduces registry intelligence: synthetic telecom recycling with 90-day statutory "
    "quarantines, dispute resolution timelines logged for N4 metrics, independent I4C intelligence feeds for all "
    "licensed banks, and emergent registry coverage across mule routing depths. "
    "Part 13c completes the 1930 hold flow: composite diurnal latency modeling, two-pass replay with lien injection, "
    "point-in-time balance reconstruction, naive single-hop inter-bank forwarding, resilient bank outage tracking, "
    "transactional outbox with bank-partitioned Kafka emission, and full-spectrum raw identifier redaction "
    "and auditing."
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
    REPO_ROOT / "nationalsim" / "hold_flow.py",
    REPO_ROOT / "world" / "hold_replay.py",
    REPO_ROOT / "nationalsim" / "bank_client.py",
    REPO_ROOT / "core" / "redaction.py",
    REPO_ROOT / "core" / "raw_identifier_scan.py",
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


class RecoveryNationalsimHandler(BaseHTTPRequestHandler):
    """Minimal in-process HTTP handler simulating recovery of nationalsim service."""

    def do_POST(self) -> None:
        if self.path == "/screen":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"match": true, "status": "listed", "risk_score": 0.85}')
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        """Silences HTTP server output during test execution."""
        pass


async def check_10_hold_forwarding_one_hop(workspace: Any) -> CheckOutcome:
    """Proves that a hold request arriving after money departure is forwarded across banks for exactly one hop.

    What it is for:
        When a primary hold lands at Bank 0 after funds have departed to Bank 1, Bank 0
        records status 'too_late' and emits a single-hop forward to Bank 1. Bank 1 evaluates
        the forward against its ledger, recording 'applied'. Further forwarding is blocked.

    Inputs:
        workspace: Unused placeholder provided by the showcase runner framework.

    What it returns:
        CheckOutcome showing outcomes for both banks and confirming second-hop termination.

    How it works:
        1. Create primary HoldRequest targeting Bank 0 with 5,000 INR.
        2. Evaluate against Bank 0 with 0 available balance -> outcome is 'too_late'.
        3. Bank 0 evaluates OutgoingTransfer to Bank 1 and generates a forwarded HoldRequest.
        4. Bank 1 evaluates forwarded HoldRequest with 5,000 INR available balance -> outcome is 'applied'.
        5. Attempting to forward from Bank 1 returns None, proving strict one-hop limitation.

    Errors and side effects:
        Raises AssertionError if outcomes or one-hop limits fail.

    Limits:
        Evaluates two-bank single-hop traceback without recursive chaining.
    """
    primary_hold = HoldRequest(
        complaint_id="CMP-SHOW-10",
        payment_id="TX-PRIMARY-01",
        receiving_bank_index=0,
        account_id="acc_mule_layer1",
        amount_paise=500_000,
        landing_minute=120,
        is_forwarded=False,
    )
    # At Bank 0, money departed at minute 100 via OutgoingTransfer to Bank 1
    outgoing = OutgoingTransfer(
        transfer_id="TX-DEPART-02",
        from_account="acc_mule_layer1",
        to_account="acc_mule_layer2",
        to_bank_index=1,
        amount_paise=500_000,
        minute=100,
    )
    outcome_1 = classify_hold_outcome(
        available_balance_paise=0,
        requested_amount_paise=500_000,
        account_id="acc_mule_layer1",
        landing_minute=120,
        missed=False,
    )
    assert outcome_1.status == "too_late"
    assert outcome_1.held_amount_paise == 0

    fwd_hold = decide_one_hop_forwarding(primary_hold, outcome_1, [outgoing])
    assert fwd_hold is not None
    assert fwd_hold.is_forwarded is True
    assert fwd_hold.receiving_bank_index == 1
    assert fwd_hold.account_id == "acc_mule_layer2"
    assert fwd_hold.amount_paise == 500_000

    # At Bank 1, money is available in acc_mule_layer2
    outcome_2 = classify_hold_outcome(
        available_balance_paise=500_000,
        requested_amount_paise=fwd_hold.amount_paise,
        account_id=fwd_hold.account_id,
        landing_minute=135,
        missed=False,
    )
    assert outcome_2.status == "applied"
    assert outcome_2.held_amount_paise == 500_000
    assert outcome_2.shortfall_paise == 0

    # Verify second-hop forward attempt is strictly rejected
    second_hop = decide_one_hop_forwarding(fwd_hold, outcome_2, [])
    assert second_hop is None

    key_number = (
        f"bank 0 (primary): {outcome_1.status} (held {outcome_1.held_amount_paise / 100:.2f} INR), "
        f"bank 1 (hop 1): {outcome_2.status} (held {outcome_2.held_amount_paise / 100:.2f} INR)"
    )
    return CheckOutcome(
        key_number=key_number,
        evidence={
            "bank_0_outcome": outcome_1.status,
            "bank_0_held_paise": outcome_1.held_amount_paise,
            "bank_0_shortfall_paise": outcome_1.shortfall_paise,
            "forwarded_to_bank_index": fwd_hold.receiving_bank_index,
            "forwarded_account_id": fwd_hold.account_id,
            "bank_1_outcome": outcome_2.status,
            "bank_1_held_paise": outcome_2.held_amount_paise,
            "second_hop_blocked": second_hop is None,
        },
    )


class InMemoryLogHandler(logging.Handler):
    """Captures structured JSON log records emitted during a simulation run."""

    def __init__(self) -> None:
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(record.getMessage())


def make_check_11(quick: bool) -> Any:
    """Build check 11 for quick (CI mode) or full (whole live stack) mode.

    Precedent from Level 8 (showcase/level_08.py):
        - Lines 54-56: "Quick mode (used in CI) restores generated files instead of the real artifacts,
          watches only Postgres, Redis and Redpanda for 20 seconds, and runs a smaller set of tests for
          the first check; full mode uses the real artifacts and the whole stack."
        - Lines 93-98: require_full_mode_stack() raises NotRun("the stack is not running (missing: ...)
          - run: docker compose up -d --wait") when full stack components are missing.
        - Lines 252-253: if not quick: require_full_mode_stack() gates full-stack auditing.

    Under this precedent:
        In quick mode (quick=True), Postgres and Redpanda are active in CI; places 4 and 5 audit the
        in-process local JSON logs and local spans emitted during the real run ("local_logs" and "local_spans"),
        with key_number explicitly noting that Loki and Tempo were not queried.
        In full mode (quick=False), live Loki (http://localhost:3100) and live Tempo (http://localhost:3200)
        are queried live by run_id; if either is unreachable or returns 0 items for the run, NotRun is raised.
    """

    async def run_check(workspace: Any) -> CheckOutcome:
        """Proves that unredacted raw identifiers are absent across all five data planes for a real run.

        What it is for:
            Executes a real simulation run on the tiny fixture world and verifies that raw, unhashed
            account identifiers, hardware device fingerprints, and customer telephone numbers never leak
            into nationalsim store exports, Postgres outbox payloads, Kafka topics, logs, or traces.

        Inputs:
            workspace: Unused placeholder provided by the showcase runner framework.

        What it returns:
            CheckOutcome displaying per-plane item counts (>0) and 0 raw hits.

        How it works:
            1. Executes the real 13c two-pass hold replay on the tiny fixture world with scam syndicate.
            2. Collects all real account IDs, device fingerprints, and phone numbers involved in the run's
               complaints, holds, and registry reports.
            3. Verifies scanner efficacy by asserting that a deliberately planted raw identifier in a throwaway
               data structure is detected.
            4. Stages real hold requests and registry events in Postgres national_outbox and relays them to Kafka.
            5. Ingests registry reports into nationalsim and captures emitted logs and exported spans.
            6. In quick mode (CI precedent from Level 8 lines 54-56 and 93-98), audits local logs and spans.
               In full mode, pushes logs to Loki, flushes spans to Tempo, and queries live Loki and Tempo by run_id.
            7. Audits all five places, asserting items_checked > 0 and raw_hits == 0 for every place.

        Errors and side effects:
            Raises NotRun if full-stack services are unreachable or return 0 items.
            Raises AssertionError if scanner misses planted leak or detects raw leak.

        Limits:
            Audits the synthetic entities generated during the isolated simulation run.
        """
        run_id = f"showcase-13c-{uuid.uuid4().hex[:8]}"
        i4c_key = "local-dev-i4c-key-001"
        dpip_key = "local-dev-dpip-key-001"
        logger = get_logger("showcase.level_13.check_11")

        # 1. Setup logging capture
        configure_logging("INFO")
        log_handler = InMemoryLogHandler()
        root_logger = logging.getLogger()
        root_logger.addHandler(log_handler)

        # 2. Setup tracer provider and span exporters
        in_memory_exporter = InMemorySpanExporter()
        provider = trace.get_tracer_provider()
        if not isinstance(provider, TracerProvider):
            provider = TracerProvider(resource=Resource.create({"service.name": "ashfall-showcase"}))
            trace.set_tracer_provider(provider)
        in_mem_processor = SimpleSpanProcessor(in_memory_exporter)
        provider.add_span_processor(in_mem_processor)

        tempo_processor = None
        if not quick:
            otlp_exporter = OTLPSpanExporter(endpoint="http://localhost:4318/v1/traces")
            tempo_processor = BatchSpanProcessor(otlp_exporter)
            provider.add_span_processor(tempo_processor)

        tracer = trace.get_tracer("nationalsim.showcase")

        try:
            # 3. Run real two-pass replay
            with tracer.start_as_current_span("two_pass_hold_replay") as span:
                span.set_attribute("run_id", run_id)
                logger.info("starting_two_pass_replay", run_id=run_id)

                budget = GangBudget(money_paise=50_000_000, time_minutes=500_000, max_risk=0.99)
                steps = (
                    BlockStep("acquire", (("count", 6),)),
                    BlockStep("scam", (("receiving_mules", 2), ("victims_per_receiving_mule", 12))),
                )
                gang = GangConfig("g_show_13c", 1, budget, None, steps, 1440)
                configs = read_world_configs()
                configs.world_ranges["parameters"]["victim_complaint_share"]["training"] = [1.0, 1.0]
                base_settings, code_version = get_showcase_b_settings()
                settings = FastRunSettings(
                    102, base_settings.n_people, base_settings.n_shops, base_settings.n_days, base_settings.range_name
                )

                replay = run_two_pass_replay(
                    fast_settings=settings,
                    runner_settings=RunnerSettings(fraud_enabled=True),
                    gang_configs=(gang,),
                    code_version=code_version,
                    configs=configs,
                )
                span.set_attribute("primary_holds_count", len(replay.primary_holds))
                logger.info("two_pass_replay_completed", run_id=run_id, primary_holds=len(replay.primary_holds))

            assert len(replay.primary_holds) > 0, "Simulation replay produced 0 holds"

            # 4. Generate phone book and collect known raw identifiers
            phone_book = build_phone_book(replay.pass2_output, run_seed=102)
            hold_accounts = {h.account_id for h in replay.primary_holds}
            event_dict = dict(zip(replay.pass2_output.result.payment_ids, replay.pass2_output.result.run_result.events))
            victim_accounts = {
                event_dict[h.payment_id].from_account
                for h in replay.primary_holds
                if h.payment_id in event_dict
            }
            involved_accounts = hold_accounts | victim_accounts

            involved_fps = {
                p.device.fingerprint
                for p in replay.pass2_output.world.people
                if p.account_id in involved_accounts and hasattr(p, "device") and p.device
            }
            involved_phones = {
                rec.phone_number
                for acc in involved_accounts
                if (rec := phone_book.get_record(acc)) is not None
            }
            known_ids = involved_accounts | involved_fps | involved_phones
            assert len(known_ids) > 0, "Known raw identifiers set is empty"

            # 5. Planted leak test: prove scanner detects real raw identifier
            planted_account = next(iter(hold_accounts))
            throwaway_store = {
                "entries": [{"planted_field": planted_account}],
                "feedback": [],
                "resolutions": [],
            }
            planted_res = scan_nationalsim_store(throwaway_store, known_ids)
            assert planted_res.raw_hits > 0, "Scanner failed to detect deliberately planted raw identifier leak!"
            assert planted_account in planted_res.details

            # 6. Stage hold requests and registry events in national_outbox
            db_url = get_showcase_database_url()
            conn = await asyncpg.connect(db_url)
            try:
                with tracer.start_as_current_span("stage_national_outbox") as span:
                    span.set_attribute("run_id", run_id)
                    # Write real hold requests
                    for h in replay.primary_holds:
                        hold_payload = build_hold_payload(
                            complaint_id=h.complaint_id,
                            payment_id=h.payment_id,
                            amount_paise=h.amount_paise,
                            account_id=h.account_id,
                            receiving_bank_index=h.receiving_bank_index,
                            run_id=run_id,
                            i4c_key=i4c_key,
                        )
                        await write_event(conn, "hold.requests", run_id, h.receiving_bank_index, hold_payload)
                        logger.info(
                            "staged_hold_request",
                            run_id=run_id,
                            complaint_id=h.complaint_id,
                            account_ref=hold_payload["account_ref"],
                        )

                    # Write registry events for reported accounts
                    for h in replay.primary_holds[:4]:
                        mule_hash = compute_identifier_hash(
                            key=dpip_key,
                            run_id=run_id,
                            identifier_type="account",
                            raw_value=h.account_id,
                            bank_index=h.receiving_bank_index,
                        )
                        reg_payload = {
                            "report_id": f"rep-{uuid.uuid4().hex[:8]}",
                            "identifier_hash": mule_hash,
                            "identifier_type": "account",
                            "reason_code": "fraud_mule",
                            "bank_id": f"bank_{h.receiving_bank_index}",
                        }
                        await write_event(conn, "registry.events", run_id, h.receiving_bank_index, reg_payload)
                        logger.info(
                            "staged_registry_event",
                            run_id=run_id,
                            report_id=reg_payload["report_id"],
                            identifier_hash=mule_hash,
                        )

                # 7. Relay publish to Kafka
                with tracer.start_as_current_span("relay_national_outbox") as span:
                    span.set_attribute("run_id", run_id)
                    producer = await start_producer()
                    try:
                        published = await relay_national_outbox(conn, producer)
                        logger.info("relayed_national_outbox", run_id=run_id, published=published)
                        span.set_attribute("published", published)
                    finally:
                        await producer.stop()
            finally:
                await conn.close()

            # 8. Ingest reports into nationalsim store
            with tracer.start_as_current_span("nationalsim_ingest") as span:
                span.set_attribute("run_id", run_id)
                registry = RegistryCore(clock=WallClock(), listing_delay=timedelta(minutes=15))
                for h in replay.primary_holds[:4]:
                    mule_hash = compute_identifier_hash(
                        key=dpip_key,
                        run_id=run_id,
                        identifier_type="account",
                        raw_value=h.account_id,
                        bank_index=h.receiving_bank_index,
                    )
                    registry.ingest(
                        bank_id=f"bank_{h.receiving_bank_index}",
                        report_id=f"rep-{uuid.uuid4().hex[:8]}",
                        identifier_hash=mule_hash,
                        identifier_type="account",
                        reason_code="fraud_mule",
                        reported_at=datetime.now(timezone.utc),
                        run_id=run_id,
                    )
                    logger.info("registry_report_ingested", run_id=run_id, bank_id=f"bank_{h.receiving_bank_index}")
                store_export = registry.export_store()

            # 9. Harvest emitted logs and spans
            captured_logs = [line for line in log_handler.lines if run_id in line]
            assert len(captured_logs) > 0, "No log lines captured for run_id"

            finished_spans = in_memory_exporter.get_finished_spans()
            local_spans = [
                {"name": s.name, "attributes": dict(s.attributes or {})}
                for s in finished_spans
                if (s.attributes or {}).get("run_id") == run_id
            ]
            assert len(local_spans) > 0, "No spans captured for run_id"

            # 10. In full mode, push logs to Loki and flush spans to Tempo
            if not quick:
                now_ns = str(time.time_ns())
                loki_payload = {
                    "streams": [
                        {
                            "stream": {"environment": "development", "job": "showcase_13", "run_id": run_id},
                            "values": [
                                [str(int(now_ns) + i * 1000), line] for i, line in enumerate(captured_logs)
                            ],
                        }
                    ]
                }
                async with httpx.AsyncClient(timeout=3.0) as http_client:
                    try:
                        push_res = await http_client.post("http://localhost:3100/loki/api/v1/push", json=loki_payload)
                        if push_res.status_code != 204:
                            raise NotRun(f"Loki push failed with HTTP {push_res.status_code}")
                    except (httpx.ConnectError, OSError) as err:
                        raise NotRun(f"Loki unreachable at http://localhost:3100 ({err})")

                if tempo_processor:
                    tempo_processor.force_flush()

                # Poll Tempo search until traces are indexed (up to 5 seconds)
                async with httpx.AsyncClient(timeout=3.0) as tempo_check_client:
                    for _ in range(10):
                        try:
                            t_resp = await tempo_check_client.get(
                                "http://localhost:3200/api/search",
                                params={"tags": f"run_id={run_id}"},
                            )
                            if t_resp.status_code == 200 and len(t_resp.json().get("traces", [])) > 0:
                                break
                        except (httpx.ConnectError, OSError):
                            pass
                        await asyncio.sleep(0.5)

            # 11. Run scan across all 5 places
            report = await scan_all_five_places(
                run_id=run_id,
                known_raw_identifiers=known_ids,
                store_export=store_export,
                local_log_lines=captured_logs if quick else None,
                local_spans=local_spans if quick else None,
                database_url=db_url,
                kafka_servers=config.kafka_bootstrap_servers,
                mode="quick" if quick else "full",
            )

            for name, p in report.places.items():
                if p.status == "NOT CHECKED":
                    raise NotRun(f"{name} is NOT CHECKED ({p.reason}) - run the full stack or use --allow-not-run")
                if p.status == "EMPTY":
                    if not quick:
                        raise NotRun(f"{name} returned 0 items in full mode ({p.reason})")
                    raise AssertionError(f"{name} contains 0 items ({p.reason})")
                assert p.items_checked > 0, f"{name} reported 0 items checked"
                assert p.raw_hits == 0, f"{name} detected {p.raw_hits} raw identifier leaks: {p.details}"

            assert report.all_clean is True, f"Raw identifier scan failed: {report.total_raw_hits} raw hits detected"
            assert report.total_raw_hits == 0

            place_counts = [f"{n}: {p.items_checked} checked, {p.raw_hits} raw" for n, p in report.places.items()]
            summary_str = ", ".join(place_counts)
            if quick:
                key_number = f"{summary_str} (Loki and Tempo not queried in quick mode)"
            else:
                key_number = summary_str

            return CheckOutcome(
                key_number=key_number,
                evidence={
                    "run_id": report.run_id,
                    "total_items_checked": report.total_items_checked,
                    "total_raw_hits": report.total_raw_hits,
                    "all_clean": report.all_clean,
                    "known_identifiers_count": len(known_ids),
                    "planted_leak_proof": {
                        "planted_account": planted_account,
                        "detected": True,
                        "hits": planted_res.raw_hits,
                    },
                    "places": {
                        name: {
                            "status": p.status,
                            "items_checked": p.items_checked,
                            "raw_hits": p.raw_hits,
                            "reason": p.reason,
                        }
                        for name, p in report.places.items()
                    },
                },
            )

        finally:
            root_logger.removeHandler(log_handler)

    return run_check


async def check_12_nationalsim_unreachable_outage_tracking(workspace: Any) -> CheckOutcome:
    """Proves that banks withstand nationalsim outages, recording start and end timestamps.

    What it is for:
        When the central registry is down, bank payment operations must continue without
        failing payment workflows, record exact outage windows, and close outages when service restores.

    Inputs:
        workspace: Unused placeholder provided by the showcase runner framework.

    What it returns:
        CheckOutcome detailing outage started_at, ended_at, duration, and failed call count.

    How it works:
        1. Create a single NationalsimBankClient pointing to an offline port with ManualClock.
        2. Call screen(): verify result is unavailable and match is None (never False).
        3. Verify active_outage is opened with started_at set and ended_at None.
        4. Advance clock by 15 minutes, spin up an in-process HTTP server, point client.base_url to it.
        5. Call screen(): verify success, active_outage is None, and outage ended_at is set.

    Errors and side effects:
        Raises AssertionError if outage fails to open, reports match=False, or fails to close.

    Limits:
        Uses a single bank client instance across the offline-to-online transition.
    """
    start_time = datetime(2026, 9, 1, 10, 0, tzinfo=IST)
    clock = ManualClock(start_time)
    client = NationalsimBankClient(
        base_url="http://127.0.0.1:59998",
        bank_id="bank_0",
        run_id="showcase_run_outage_12",
        clock=clock,
        timeout_seconds=0.05,
        retry_backoff_seconds=0.01,
    )

    test_hash = "c" * 64
    first_res = await client.screen(identifier_hash=test_hash)
    assert first_res.status == "unavailable"
    assert first_res.is_unavailable is True
    assert first_res.match is None
    assert first_res.match is not False
    assert len(client.outages) == 1
    assert client.active_outage is not None
    assert client.active_outage.started_at == start_time
    assert client.active_outage.ended_at is None
    assert client.active_outage.calls_failed == 1

    # Spin up recovery server on loopback
    server = HTTPServer(("127.0.0.1", 0), RecoveryNationalsimHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    port = server.server_address[1]

    try:
        # Advance time by 15 minutes and point client to recovered server
        end_time = datetime(2026, 9, 1, 10, 15, tzinfo=IST)
        clock.set_time(end_time)
        client.base_url = f"http://127.0.0.1:{port}"

        second_res = await client.screen(identifier_hash=test_hash)
        assert second_res.status == "match"
        assert second_res.match is True
        assert client.active_outage is None
        assert client.outages[0].ended_at == end_time
    finally:
        server.shutdown()
        server.server_close()

    outage = client.outages[0]
    duration_min = int((outage.ended_at - outage.started_at).total_seconds() / 60)
    key_number = (
        f"outage: started {outage.started_at.strftime('%H:%M')} IST, "
        f"ended {outage.ended_at.strftime('%H:%M')} IST ({duration_min}m), "
        f"{outage.calls_failed} failed call(s)"
    )
    return CheckOutcome(
        key_number=key_number,
        evidence={
            "bank_id": outage.bank_id,
            "started_at": outage.started_at.isoformat(),
            "ended_at": outage.ended_at.isoformat() if outage.ended_at else None,
            "calls_failed": outage.calls_failed,
            "duration_minutes": duration_min,
            "first_screen_status": first_res.status,
            "first_screen_match": first_res.match,
            "second_screen_status": second_res.status,
            "second_screen_match": second_res.match,
        },
    )


async def check_13_hold_latency_by_time_of_day_chart(workspace: Any) -> CheckOutcome:
    """Proves hold latency diurnal variation from a replay and renders a scatter chart.

    What it is for:
        Visualizes the operational latency scaling across day vs night hours from a two-pass
        hold replay under Amendment A-029.

    Inputs:
        workspace: Unused placeholder provided by the showcase runner framework.

    What it returns:
        CheckOutcome with scatter chart of hold latencies by time of day.

    How it works:
        1. Execute run_two_pass_replay on standard showcase settings with scam gang.
        2. Group hold latencies by window with measure_hold_latencies_by_window.
        3. Compute median latencies for day and night.
        4. Construct scatter chart with Day and Night series.

    Errors and side effects:
        Raises AssertionError if replay yields no holds.

    Limits:
        Derives latencies from simulation replay primary holds.
    """
    from world.fast_mode import FastRunSettings
    from world.gang_config import BlockStep, GangBudget, GangConfig
    from world.world_configs import read_world_configs

    budget = GangBudget(money_paise=50_000_000, time_minutes=500_000, max_risk=0.99)
    steps = (
        BlockStep("acquire", (("count", 6),)),
        BlockStep("scam", (("receiving_mules", 2), ("victims_per_receiving_mule", 12))),
    )
    gang = GangConfig("g_show_13", 1, budget, None, steps, 1440)

    configs = read_world_configs()
    configs.world_ranges["parameters"]["victim_complaint_share"]["training"] = [1.0, 1.0]

    base_settings, code_version = get_showcase_b_settings()
    settings = FastRunSettings(
        102, base_settings.n_people, base_settings.n_shops, base_settings.n_days, base_settings.range_name
    )

    replay = run_two_pass_replay(
        fast_settings=settings,
        runner_settings=RunnerSettings(fraud_enabled=True),
        gang_configs=(gang,),
        code_version=code_version,
        configs=configs,
    )
    windows = measure_hold_latencies_by_window(replay)
    assert windows["day"] or windows["night"], "Replay must produce at least one hold"

    med_day = float(statistics.median(windows["day"])) if windows["day"] else 0.0
    med_night = float(statistics.median(windows["night"])) if windows["night"] else 0.0

    day_points = [[i + 1, lat] for i, lat in enumerate(windows["day"])]
    night_points = [[i + 1, lat] for i, lat in enumerate(windows["night"])]

    chart = {
        "type": "scatter",
        "title": "Hold operational latency by time of day (day vs night)",
        "x_label": "Observation index",
        "y_label": "Hold latency (minutes)",
        "series": [
            {"name": "Day holds", "points": day_points},
            {"name": "Night holds", "points": night_points},
        ],
    }
    key_number = (
        f"median day: {med_day:.1f}m ({len(windows['day'])} holds), "
        f"night: {med_night:.1f}m ({len(windows['night'])} holds)"
    )
    return CheckOutcome(
        key_number=key_number,
        evidence={
            "median_day_minutes": med_day,
            "median_night_minutes": med_night,
            "day_hold_count": len(windows["day"]),
            "night_hold_count": len(windows["night"]),
            "weekday_hold_count": len(windows["weekday"]),
            "weekend_hold_count": len(windows["weekend"]),
        },
        chart=chart,
    )


async def check_14_hold_lands_after_money_left_too_late(workspace: Any) -> CheckOutcome:
    """Proves that an administrative hold arriving after funds departed is recorded as too_late.

    What it is for:
        Verifies that when funds have departed an account prior to the hold landing minute,
        the bank reconstructs a depleted balance and classifies the hold strictly as 'too_late'.

    Inputs:
        workspace: Unused placeholder provided by the showcase runner framework.

    What it returns:
        CheckOutcome recording too_late status, zero held amount, and full shortfall.

    How it works:
        1. Create bank Ledger with account acc_mule_departed funded with 5,000 INR at minute 0.
        2. Post debit of 5,000 INR at minute 40, draining the balance to 0.
        3. Reconstruct account balance at landing minute 60; verify balance is 0.
        4. Classify 5,000 INR hold landing at minute 60; assert status is 'too_late'.

    Errors and side effects:
        Raises AssertionError if status is not 'too_late' or held funds are non-zero.

    Limits:
        Evaluates single-account point-in-time balance reconstruction.
    """
    ledger = Ledger(bank_index=1, upi_limit_paise=10_000_000)
    target_account = "acc_mule_departed"
    ledger.add_account(target_account, opening_balance_paise=500_000, minute=0)

    # Funds depart at minute 40 via outgoing debit
    ledger.post_debit(target_account, amount_paise=500_000, minute=40, transfer_id="tx_depart_40")

    # Hold lands at minute 60
    landing_minute = 60
    reconstructed_balance = reconstruct_account_balance(ledger, target_account, landing_minute)
    assert reconstructed_balance <= 0, f"Expected depleted balance <= 0, got {reconstructed_balance}"

    outcome = classify_hold_outcome(
        available_balance_paise=reconstructed_balance,
        requested_amount_paise=500_000,
        account_id=target_account,
        landing_minute=landing_minute,
        missed=False,
    )
    assert outcome.status == "too_late"
    assert outcome.held_amount_paise == 0
    assert outcome.shortfall_paise == 500_000

    key_number = "status: too_late, requested: 5000.00 INR, held: 0.00 INR, shortfall: 5000.00 INR"
    return CheckOutcome(
        key_number=key_number,
        evidence={
            "status": outcome.status,
            "requested_paise": outcome.requested_amount_paise,
            "held_paise": outcome.held_amount_paise,
            "shortfall_paise": outcome.shortfall_paise,
            "landing_minute": landing_minute,
            "reconstructed_balance_paise": reconstructed_balance,
        },
    )


def get_checks(quick: bool = False) -> list[CheckDefinition]:
    """Construct the ordered list of Level 13 showcase verification checks.

    What it is for:
        Supplies the showcase runner framework with the suite of checks evaluating Level 13 (parts a, b, and c).
        All checks satisfy the coverage requirement across normal, edge, and negative checks.

    Inputs:
        quick: Boolean flag indicating if quick mode is requested. All Level 13 checks are fast.

    What it returns:
        List of 14 CheckDefinition containers tagged with their respective part.

    How it works:
        1. Construct CheckDefinitions for checks 1 through 5 (Part 13a).
        2. Construct CheckDefinitions for checks 6 through 9 (Part 13b).
        3. Construct CheckDefinitions for checks 10 through 14 (Part 13c).
        4. Return all 14 checks in order.

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
            "but successfully pulls suspect listings hashed with the I4C key from the I4C feed."
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
    check_10 = CheckDefinition(
        name="10. a hold request is forwarded across two banks (one hop), with both outcomes shown",
        kind="edge",
        proves=(
            "A hold request landing after money departed an initial bank is forwarded to the receiving "
            "institution (one hop), where it is evaluated against available funds."
        ),
        run=check_10_hold_forwarding_one_hop,
        likely_cause="One-hop traceback failed to identify outgoing transfer or failed single-hop termination.",
        where_to_look="nationalsim/hold_flow.py, world/hold_replay.py",
        part="c",
    )
    check_11 = CheckDefinition(
        name="11. the five places are scanned for raw identifiers, and none are found",
        kind="negative",
        proves=(
            "An exhaustive audit across five national data planes (store export, Postgres outbox, "
            "Kafka topics, logs/Loki, and Tempo traces) detects 0 raw identifiers."
        ),
        run=make_check_11(quick),
        likely_cause="Unredacted account, phone, or device fingerprint leaked into a national data plane.",
        where_to_look="core/redaction.py, core/raw_identifier_scan.py",
        part="c",
    )
    check_12 = CheckDefinition(
        name="12. nationalsim is unreachable; banks keep running and record the outage, with started_at/ended_at",
        kind="negative",
        proves=(
            "Banks maintain operational continuity during central registry outages, recording exact "
            "outage start/end timestamps and closing the outage upon recovery."
        ),
        run=check_12_nationalsim_unreachable_outage_tracking,
        likely_cause="Bank client raised network exception during outage or failed to record started_at/ended_at.",
        where_to_look="nationalsim/bank_client.py",
        part="c",
    )
    check_13 = CheckDefinition(
        name="13. hold latency by time of day, as a chart",
        kind="normal",
        proves=(
            "Administrative hold response times follow diurnal scaling with higher operational "
            "latencies during night hours, displayed as a scatter chart."
        ),
        run=check_13_hold_latency_by_time_of_day_chart,
        likely_cause="Hold replay produced no holds or diurnal latency scaling was not applied.",
        where_to_look="nationalsim/hold_flow.py, world/hold_replay.py",
        part="c",
    )
    check_14 = CheckDefinition(
        name="14. a hold lands after the money left and is recorded as too_late",
        kind="edge",
        proves=(
            "When funds have departed an account prior to the hold landing minute, the hold outcome "
            "is classified strictly as too_late with zero held funds."
        ),
        run=check_14_hold_lands_after_money_left_too_late,
        likely_cause="Balance reconstruction failed to detect prior debit or outcome classification was not too_late.",
        where_to_look="nationalsim/hold_flow.py, world/hold_replay.py",
        part="c",
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
        check_10,
        check_11,
        check_12,
        check_13,
        check_14,
    ]
