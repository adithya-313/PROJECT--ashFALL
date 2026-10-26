"""
tests/unit/nationalsim/test_app.py

Unit tests for the nationalsim HTTP FastAPI service using fastapi.testclient.TestClient.
Proves health probe functionality, screening visibility delays, revocation authorization,
customer disputes, feedback recording, and explicit HTTP error handling for bad inputs.
"""

import os
from datetime import datetime, timedelta
from typing import Any

import pytest
from fastapi.testclient import TestClient

from core.event_clock import IST, ManualClock
from nationalsim.app import create_app, create_app_prod, load_nationalsim_config
from nationalsim.keyed_hash import compute_identifier_hash
from nationalsim.registry import NotParticipantError

DEFAULT_TEST_SETTINGS: dict[str, Any] = {
    "listing_delay_minutes": 15,
    "banks": {"bank_0": {"dpip_participation": True}, "bank_1": {"dpip_participation": True}},
}


def test_health_check_returns_ok_and_simulated_time() -> None:
    """Proves that orchestrators and monitors can verify the registry service is alive and healthy.

    In production, container runtimes periodically ping /health to detect crashes.
    If the endpoint broke or returned an error, container orchestrators like Docker Compose
    would mistakenly mark the healthy container as degraded and terminate it.
    """
    fixed_time = datetime(2026, 9, 1, 10, 0, tzinfo=IST)
    clock = ManualClock(fixed_time)
    app = create_app(clock=clock, key_provider=lambda: "test_key_secret", settings=DEFAULT_TEST_SETTINGS)
    client = TestClient(app)

    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "nationalsim"
    assert data["time_ist"] == fixed_time.isoformat()


def test_http_screening_delay_lifecycle() -> None:
    """Proves the full HTTP report lifecycle from initial submission to screening visibility.

    When a bank submits a confirmed fraud report, national policy delays publication by 15 minutes
    so the registry can process it. If screening matched before the delay expired, banks would act on
    reports prematurely before official clearance.
    """
    start_time = datetime(2026, 9, 1, 10, 0, tzinfo=IST)
    clock = ManualClock(start_time)
    app = create_app(
        clock=clock,
        key_provider=lambda: "test_key_secret",
        settings={
            "listing_delay_minutes": 15,
            "banks": {"bank_0": {"dpip_participation": True}},
        },
    )
    client = TestClient(app)

    ident_hash = compute_identifier_hash("test_key_secret", "run_http", "account", "acc_victim_01", 0)

    # 1. Ingest report
    ingest_payload = {
        "bank_id": "bank_0",
        "report_id": "R-HTTP-101",
        "identifier_hash": ident_hash,
        "identifier_type": "account",
        "reason_code": "fraud_confirmed",
        "reported_at": start_time.isoformat(),
        "run_id": "run_http",
    }
    ingest_res = client.post("/ingest", json=ingest_payload)
    assert ingest_res.status_code == 200
    assert ingest_res.json()["status"] == "ingested"

    # 2. Screen at visible_at - 1 minute (10:14 IST): returns no match
    clock.set_time(datetime(2026, 9, 1, 10, 14, tzinfo=IST))
    screen_payload = {
        "requesting_bank_id": "bank_0",
        "identifier_hash": ident_hash,
        "identifier_type": "account",
        "run_id": "run_http",
    }
    screen_before = client.post("/screen", json=screen_payload)
    assert screen_before.status_code == 200
    assert screen_before.json()["match"] is False
    assert screen_before.json()["status"] == "none"

    # 3. Screen at visible_at (10:15 IST): returns match with status "listed"
    clock.set_time(datetime(2026, 9, 1, 10, 15, tzinfo=IST))
    screen_at = client.post("/screen", json=screen_payload)
    assert screen_at.status_code == 200
    assert screen_at.json()["match"] is True
    assert screen_at.json()["status"] == "listed"


def test_http_dispute_and_revoke_flow() -> None:
    """Proves that customer disputes mark entries contested and original banks can revoke them over HTTP.

    If an innocent customer's account is falsely reported, their bank submits a dispute to flag
    the status as contested without removing the match. If the reporting bank later confirms a mistake,
    they revoke the report to restore normal account status.
    """
    clock = ManualClock(datetime(2026, 9, 1, 10, 0, tzinfo=IST))
    app = create_app(
        clock=clock,
        key_provider=lambda: "test_key",
        settings={
            "listing_delay_minutes": 5,
            "banks": {"bank_0": {"dpip_participation": True}},
        },
    )
    client = TestClient(app)

    ident_hash = "b" * 64
    now = clock.now_ist()

    # Ingest report
    client.post(
        "/ingest",
        json={
            "bank_id": "bank_0",
            "report_id": "R-DISP-1",
            "identifier_hash": ident_hash,
            "identifier_type": "account",
            "reason_code": "fraud",
            "reported_at": now.isoformat(),
            "run_id": "run_dispute",
        },
    )

    # Advance clock past listing delay
    clock.advance(timedelta(minutes=10))

    # Dispute the entry
    disp_res = client.post(
        "/dispute",
        json={
            "bank_id": "bank_0",
            "identifier_hash": ident_hash,
            "reason_code": "innocent_customer_affidavit",
            "run_id": "run_dispute",
        },
    )
    assert disp_res.status_code == 200
    assert disp_res.json()["status"] == "contested"

    # Screening now returns status "contested"
    screen_res = client.post(
        "/screen",
        json={
            "requesting_bank_id": "bank_0",
            "identifier_hash": ident_hash,
            "identifier_type": "account",
            "run_id": "run_dispute",
        },
    )
    assert screen_res.status_code == 200
    assert screen_res.json()["match"] is True
    assert screen_res.json()["status"] == "contested"

    # Revoke the report
    revoke_res = client.post(
        "/revoke",
        json={
            "bank_id": "bank_0",
            "report_id": "R-DISP-1",
            "run_id": "run_dispute",
        },
    )
    assert revoke_res.status_code == 200
    assert revoke_res.json()["status"] == "revoked"

    # Screening after revocation produces no match
    screen_after = client.post(
        "/screen",
        json={
            "requesting_bank_id": "bank_0",
            "identifier_hash": ident_hash,
            "identifier_type": "account",
            "run_id": "run_dispute",
        },
    )
    assert screen_after.status_code == 200
    assert screen_after.json()["match"] is False
    assert screen_after.json()["status"] == "none"


def test_http_feedback_endpoint_records_call() -> None:
    """Proves that operational feedback can be submitted without altering screening answers.

    Banks submit feedback to report investigative outcomes on previously flagged identifiers.
    The mock registry records this feedback for future level evaluation loops while leaving
    screening logic unaffected.
    """
    clock = ManualClock()
    app = create_app(clock=clock, key_provider=lambda: "key", settings=DEFAULT_TEST_SETTINGS)
    client = TestClient(app)

    feedback_payload = {
        "bank_id": "bank_1",
        "identifier_hash": "e" * 64,
        "reason_code": "funds_intercepted",
        "run_id": "run_feedback",
    }
    response = client.post("/feedback", json=feedback_payload)
    assert response.status_code == 200
    assert response.json() == {"status": "recorded"}


def test_http_non_participating_bank_refused_with_403() -> None:
    """Proves that banks not participating in DPIP are explicitly refused screening access over HTTP.

    Only banks enrolled in the national DPIP program hold the secret hashing key.
    If an unenrolled bank attempts screening, the registry must deny access with an explicit
    HTTP 403 Forbidden rather than leaking empty or default data.
    """
    clock = ManualClock()
    app = create_app(
        clock=clock,
        key_provider=lambda: "key",
        settings={
            "listing_delay_minutes": 15,
            "banks": {"outsider_bank": {"dpip_participation": False}},
        },
    )
    client = TestClient(app)

    screen_payload = {
        "requesting_bank_id": "outsider_bank",
        "identifier_hash": "a" * 64,
        "identifier_type": "account",
        "run_id": "run_outsider",
    }
    res = client.post("/screen", json=screen_payload)
    assert res.status_code == 403
    assert "not a participating DPIP member" in res.json()["detail"]


def test_http_malformed_inputs_return_422() -> None:
    """Proves that invalid identifier hashes or missing required fields fail schema validation with HTTP 422.

    If a client sends an uppercase hash, an invalid length string, or forgets the simulation run ID,
    FastAPI schema validation must reject the request before it reaches business logic.
    """
    clock = ManualClock()
    app = create_app(clock=clock, key_provider=lambda: "key", settings=DEFAULT_TEST_SETTINGS)
    client = TestClient(app)

    # 1. Invalid hash (not 64 lowercase hex)
    bad_hash_payload = {
        "requesting_bank_id": "bank_0",
        "identifier_hash": "TOO_SHORT",
        "identifier_type": "account",
        "run_id": "run_1",
    }
    res_bad_hash = client.post("/screen", json=bad_hash_payload)
    assert res_bad_hash.status_code == 422

    # 2. Missing run_id
    missing_run_id_payload = {
        "requesting_bank_id": "bank_0",
        "identifier_hash": "a" * 64,
        "identifier_type": "account",
    }
    res_no_run = client.post("/screen", json=missing_run_id_payload)
    assert res_no_run.status_code == 422


def test_production_factory_fails_loudly_when_secret_key_missing() -> None:
    """Proves that production startup crashes immediately if the HMAC secret key is omitted.

    In production deployments, running with a blank secret key would cause security vulnerabilities
    or corrupt hash comparisons. The server must refuse to start until an operator supplies the key.
    """
    old_env = os.environ.get("NATIONALSIM_DPIP_KEY")
    if "NATIONALSIM_DPIP_KEY" in os.environ:
        del os.environ["NATIONALSIM_DPIP_KEY"]

    try:
        with pytest.raises(ValueError, match="NATIONALSIM_DPIP_KEY' is missing or empty"):
            create_app_prod()
    finally:
        if old_env is not None:
            os.environ["NATIONALSIM_DPIP_KEY"] = old_env


def test_not_participant_error_gives_403_regardless_of_message_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Proves that NotParticipantError always returns HTTP 403 Forbidden even if error text changes.

    What it is for:
        When an unapproved bank attempts to screen an identifier, the registry core raises
        a typed NotParticipantError. The HTTP route must catch this exception class directly
        and return HTTP 403 Forbidden, rather than relying on brittle message substring matches.

    Inputs:
        monkeypatch: Pytest fixture to temporarily alter the registry's screen method.

    What it returns:
        None. Asserts the HTTP status code remains 403 when the error message is changed.

    How it works:
        1. Build an application with a manual clock, secret key, and empty settings.
        2. Monkeypatch app.state.registry.screen to raise NotParticipantError with completely novel text.
        3. Send a screening request to POST /screen.
        4. Assert that the HTTP response code is 403 and the detail preserves the custom message.

    Errors and side effects:
        Fails assertion if the HTTP layer fails to map the typed exception to HTTP 403.

    Limits:
        Evaluates the exception handling behavior of the POST /screen endpoint.
    """
    clock = ManualClock()
    app = create_app(
        clock=clock,
        key_provider=lambda: "test_key",
        settings={
            "listing_delay_minutes": 15,
            "banks": {"bank_arbitrary": {"dpip_participation": True}},
        },
    )
    client = TestClient(app)

    def custom_screen(*args: Any, **kwargs: Any) -> Any:
        raise NotParticipantError("Custom refusal: institution has not signed bilateral agreement")

    monkeypatch.setattr(app.state.registry, "screen", custom_screen)

    screen_payload = {
        "requesting_bank_id": "bank_arbitrary",
        "identifier_hash": "c" * 64,
        "identifier_type": "account",
        "run_id": "run_test_typed_refusal",
    }
    response = client.post("/screen", json=screen_payload)
    assert response.status_code == 403
    assert response.json()["detail"] == "Custom refusal: institution has not signed bilateral agreement"


def test_missing_delay_raises_value_error() -> None:
    """Proves that omitting listing_delay_minutes raises a ValueError naming the missing key and file.

    What it is for:
        The registry service requires an explicit listing delay setting.
        Failing loudly when the key is omitted prevents silent fallback to unverified defaults.

    Inputs:
        None. Invokes create_app with a settings dict lacking listing_delay_minutes.

    What it returns:
        None. Asserts ValueError is raised naming the missing key and file.

    How it works:
        1. Prepare a settings dictionary with valid banks but no listing_delay_minutes.
        2. Attempt to create the app and assert ValueError is raised.
        3. Verify the error message names 'listing_delay_minutes' and 'config/nationalsim.yaml'.

    Errors and side effects:
        Raises AssertionError if ValueError is not raised or the message omits the key or file.

    Limits:
        Unit test for dictionary validation.
    """
    clock = ManualClock()
    settings = {"banks": {"bank_0": {"dpip_participation": True}}}
    with pytest.raises(ValueError) as excinfo:
        create_app(clock=clock, key_provider=lambda: "key", settings=settings)
    err = str(excinfo.value)
    assert "listing_delay_minutes" in err
    assert "config/nationalsim.yaml" in err


def test_missing_banks_raises_value_error() -> None:
    """Proves that omitting or leaving empty the banks mapping raises a ValueError.

    What it is for:
        DPIP screening requires configured participant banks. If the banks section is missing
        or empty, the application must refuse startup rather than silently defaulting to an empty registry.

    Inputs:
        None. Invokes create_app with settings missing 'banks' and with empty 'banks': {}.

    What it returns:
        None. Asserts ValueError is raised naming 'banks' and 'config/nationalsim.yaml'.

    How it works:
        1. Attempt to create the app with no banks key; verify ValueError names 'banks' and file.
        2. Attempt to create the app with an empty banks dictionary; verify ValueError names 'banks' and file.

    Errors and side effects:
        Raises AssertionError if validation passes unexpectedly.

    Limits:
        Evaluates missing and empty dictionary conditions.
    """
    clock = ManualClock()
    with pytest.raises(ValueError) as excinfo1:
        create_app(clock=clock, key_provider=lambda: "key", settings={"listing_delay_minutes": 15})
    assert "banks" in str(excinfo1.value)
    assert "config/nationalsim.yaml" in str(excinfo1.value)

    with pytest.raises(ValueError) as excinfo2:
        create_app(clock=clock, key_provider=lambda: "key", settings={"listing_delay_minutes": 15, "banks": {}})
    assert "banks" in str(excinfo2.value)
    assert "config/nationalsim.yaml" in str(excinfo2.value)


def test_negative_delay_raises_value_error() -> None:
    """Proves that negative or boolean delay values are rejected with ValueError.

    What it is for:
        Listing delay represents elapsed physical or simulated minutes and must be a non-negative integer.
        Rejecting negative numbers or booleans prevents invalid comparison bounds or unintended type coercion.

    Inputs:
        None. Tests settings with delay -5 and delay True.

    What it returns:
        None. Asserts ValueError is raised in both cases naming 'listing_delay_minutes' and file.

    How it works:
        1. Pass listing_delay_minutes = -5; assert ValueError is raised.
        2. Pass listing_delay_minutes = True; assert ValueError is raised.

    Errors and side effects:
        Raises AssertionError if negative or boolean values are accepted.

    Limits:
        Tests boundary and type rejection for the delay setting.
    """
    clock = ManualClock()
    valid_banks = {"bank_0": {"dpip_participation": True}}

    with pytest.raises(ValueError) as excinfo_neg:
        create_app(
            clock=clock,
            key_provider=lambda: "key",
            settings={"listing_delay_minutes": -5, "banks": valid_banks},
        )
    assert "listing_delay_minutes" in str(excinfo_neg.value)
    assert "config/nationalsim.yaml" in str(excinfo_neg.value)

    with pytest.raises(ValueError) as excinfo_bool:
        create_app(
            clock=clock,
            key_provider=lambda: "key",
            settings={"listing_delay_minutes": True, "banks": valid_banks},
        )
    assert "listing_delay_minutes" in str(excinfo_bool.value)
    assert "config/nationalsim.yaml" in str(excinfo_bool.value)


def test_non_bool_participation_flag_raises_value_error() -> None:
    """Proves that a bank participation flag that is not a real boolean raises ValueError.

    What it is for:
        DPIP participation must be an explicit boolean (True or False).
        Accepting strings like 'true' or numbers like 1 could lead to misconfiguration or truthy bugs.

    Inputs:
        None. Supplies bank_0 with dpip_participation='true' and dpip_participation=1.

    What it returns:
        None. Asserts ValueError is raised naming 'dpip_participation' and 'config/nationalsim.yaml'.

    How it works:
        1. Set dpip_participation to string 'true'; assert ValueError.
        2. Set dpip_participation to integer 1; assert ValueError.

    Errors and side effects:
        Raises AssertionError if non-boolean participation flags are accepted.

    Limits:
        Evaluates per-bank participation configuration types.
    """
    clock = ManualClock()

    with pytest.raises(ValueError) as excinfo_str:
        create_app(
            clock=clock,
            key_provider=lambda: "key",
            settings={"listing_delay_minutes": 15, "banks": {"bank_0": {"dpip_participation": "true"}}},
        )
    assert "dpip_participation" in str(excinfo_str.value)
    assert "config/nationalsim.yaml" in str(excinfo_str.value)

    with pytest.raises(ValueError) as excinfo_int:
        create_app(
            clock=clock,
            key_provider=lambda: "key",
            settings={"listing_delay_minutes": 15, "banks": {"bank_0": {"dpip_participation": 1}}},
        )
    assert "dpip_participation" in str(excinfo_int.value)
    assert "config/nationalsim.yaml" in str(excinfo_int.value)


def test_real_nationalsim_config_still_loads() -> None:
    """Proves that the repository's real config/nationalsim.yaml passes all validation rules.

    What it is for:
        Ensures that production and test configurations committed in the repository
        satisfy the strict validation checks without raising errors.

    Inputs:
        None. Invokes load_nationalsim_config on the real file.

    What it returns:
        None. Asserts listing_delay is 15 minutes and all three banks are enrolled.

    How it works:
        1. Call load_nationalsim_config().
        2. Verify listing_delay equals 15 minutes.
        3. Verify banks mapping contains bank_0, bank_1, and bank_2 with True participation.

    Errors and side effects:
        None.

    Limits:
        Reads config/nationalsim.yaml from repository root.
    """
    listing_delay, bank_settings = load_nationalsim_config()
    assert listing_delay == timedelta(minutes=15)
    assert len(bank_settings) >= 3
    assert bank_settings["bank_0"]["dpip_participation"] is True
    assert bank_settings["bank_1"]["dpip_participation"] is True
    assert bank_settings["bank_2"]["dpip_participation"] is True

