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
from nationalsim.adapter import DEFAULT_FRAUD_REASON_CODE, IDENTIFIER_TYPE_ACCOUNT
from nationalsim.app import create_app
from nationalsim.keyed_hash import compute_identifier_hash
from showcase.framework import CheckDefinition, CheckOutcome

LEVEL_NUMBER = 13
LEVEL_TITLE = "Mock national systems and event clock (built: part 13a)"
NEEDS_DATABASE = False

WHAT_YOU_SAW = (
    "Level 13a introduces the national systems simulator (nationalsim) and an injectable event clock. "
    "The event clock lets tests advance simulated time without sleeping and guarantees that 1x and 100x "
    "replays produce identical decision timestamps. "
    "The registry mock stores fraud reports using HMAC hashes so raw bank account numbers and device "
    "fingerprints are never held in the national database. "
    "Banks screening an account must wait through a 15-minute listing delay before new reports match. "
    "Banks outside the DPIP network are denied screening access, and customer disputes flag listings "
    "as contested."
)

REPO_ROOT = Path(__file__).resolve().parent.parent
REQUIRED_FILES = [
    REPO_ROOT / "config" / "nationalsim.yaml",
    REPO_ROOT / "core" / "event_clock.py",
    REPO_ROOT / "nationalsim" / "app.py",
    REPO_ROOT / "nationalsim" / "keyed_hash.py",
    REPO_ROOT / "nationalsim" / "registry.py",
    REPO_ROOT / "nationalsim" / "adapter.py",
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


def get_checks(quick: bool = False) -> list[CheckDefinition]:
    """Construct the ordered list of Level 13 showcase verification checks.

    What it is for:
        Supplies the showcase runner framework with the suite of checks evaluating Level 13a.
        All checks run in-process and satisfy the coverage requirement (1 normal, 2 edge, 1 negative).

    Inputs:
        quick: Boolean flag indicating if quick mode is requested. All Level 13 checks are fast.

    What it returns:
        List of 5 CheckDefinition containers tagged with part='a'.

    How it works:
        1. Construct CheckDefinition for check 1 (normal: report then screen).
        2. Construct CheckDefinition for check 2 (edge: revoke and dispute).
        3. Construct CheckDefinition for check 3 (consistency: 1x vs 100x clock replay).
        4. Construct CheckDefinition for check 4 (negative: outsider refused and wrong key).
        5. Construct CheckDefinition for check 5 (edge: boundary minute check).
        6. Return all 5 checks in order.

    Errors and side effects:
        None. Pure constructor returning check definitions.

    Limits:
        Part 13a only. Later parts (13b, 13c) will append subsequent checks.
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
    return [check_1, check_2, check_3, check_4, check_5]
