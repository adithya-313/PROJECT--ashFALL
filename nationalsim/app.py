"""
nationalsim/app.py

FastAPI HTTP service for the mock national fraud registry and DPIP intelligence network.

An ASGI application is a Python program that can receive and respond to asynchronous web requests.
A healthcheck is a simple probe that lets orchestration systems know if a service is running and ready.
HMAC is a way to calculate a secret code over data using a shared password, proving the data was not tampered with.
Namespaced means records are tagged with a specific run identifier so separate simulations do not mix their data.
Idempotent means sending the exact same request twice has the same result as sending it once without duplicates.
"""

import os
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from core.event_clock import Clock, WallClock
from nationalsim.i4c_feed import I4CFeedCore
from nationalsim.registry import NotParticipantError, RegistryCore


class IngestRequest(BaseModel):
    """Payload for submitting a bank fraud report into the registry."""

    bank_id: str
    report_id: str
    identifier_hash: str = Field(
        ...,
        pattern=r"^[0-9a-f]{64}$",
        description="Scrambled identifier, exactly 64 lowercase hexadecimal characters.",
    )
    identifier_type: str = Field(
        ...,
        description="Type of identifier: 'account', 'device', or 'phone'.",
    )
    reason_code: str
    reported_at: datetime
    run_id: str
    linked_account_hash: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
        description="Optional hash of the associated account when reporting a device.",
    )


class ScreenRequest(BaseModel):
    """Payload for asking the registry whether an identifier was reported for fraud."""

    requesting_bank_id: str
    identifier_hash: str = Field(
        ...,
        pattern=r"^[0-9a-f]{64}$",
        description="Scrambled identifier, exactly 64 lowercase hexadecimal characters.",
    )
    identifier_type: str = Field(
        ...,
        description="Type of identifier, either 'account' or 'device'.",
    )
    run_id: str


class ScreenResponse(BaseModel):
    """Registry response indicating whether a screened identifier is listed or contested."""

    match: bool
    status: str
    distinct_accounts_linked: int
    risk_score: float


class RevokeRequest(BaseModel):
    """Payload for revoking a previously submitted fraud report."""

    bank_id: str
    report_id: str
    run_id: str


class DisputeRequest(BaseModel):
    """Payload for disputing an active fraud report on behalf of an innocent customer."""

    bank_id: str
    identifier_hash: str = Field(
        ...,
        pattern=r"^[0-9a-f]{64}$",
        description="Scrambled identifier, exactly 64 lowercase hexadecimal characters.",
    )
    reason_code: str
    run_id: str


class FeedbackRequest(BaseModel):
    """Payload for recording operational feedback on an identifier."""

    bank_id: str
    identifier_hash: str = Field(
        ...,
        pattern=r"^[0-9a-f]{64}$",
        description="Scrambled identifier, exactly 64 lowercase hexadecimal characters.",
    )
    reason_code: str
    run_id: str
    reported_at: datetime | None = None


def validate_nationalsim_settings(
    settings: dict[str, Any],
    file_name: str = "config/nationalsim.yaml",
) -> tuple[int, dict[str, Any]]:
    """Validate listing delay and bank participation dictionary settings.

    What it is for:
        Enforces that registry parameters are strictly validated without silent fallbacks.
        Guarantees that listing delay is a non-negative integer and the bank registry is non-empty.

    Inputs:
        settings: Dictionary containing listing_delay_minutes and banks mappings.
            For example, {"listing_delay_minutes": 15, "banks": {"bank_0": {"dpip_participation": True}}}.
        file_name: Name of the configuration file to include in error messages.
            Defaults to 'config/nationalsim.yaml'.

    What it returns:
        A tuple of (delay_minutes, banks_dict) with validated types.

    How it works:
        1. Check that 'listing_delay_minutes' key is present in settings.
        2. Unpack integer value if structured as a dictionary.
        3. Assert that the delay is an int, not a bool, and >= 0.
        4. Check that 'banks' key is present and is a non-empty mapping.
        5. Verify that each bank entry defines 'dpip_participation' as a real boolean.

    Errors and side effects:
        Raises ValueError naming the missing or invalid key and the file name.

    Limits:
        Validates in-memory dictionary representation.
    """
    if "listing_delay_minutes" not in settings:
        raise ValueError(f"Missing key 'listing_delay_minutes' in {file_name}.")

    raw_delay = settings["listing_delay_minutes"]
    if isinstance(raw_delay, dict):
        if "value" not in raw_delay:
            raise ValueError(f"Missing key 'listing_delay_minutes.value' in {file_name}.")
        raw_delay = raw_delay["value"]

    # In Python, bool is a subclass of int (isinstance(True, int) is True), so reject bool explicitly
    if isinstance(raw_delay, bool) or not isinstance(raw_delay, int) or raw_delay < 0:
        raise ValueError(
            f"Invalid 'listing_delay_minutes' in {file_name}: must be a non-negative integer, got {raw_delay!r}."
        )

    if "banks" not in settings:
        raise ValueError(f"Missing key 'banks' in {file_name}.")

    banks_dict = settings["banks"]
    if not isinstance(banks_dict, dict) or not banks_dict:
        raise ValueError(f"Missing or empty 'banks' section in {file_name}.")

    for bank_id, bank_cfg in banks_dict.items():
        if not isinstance(bank_cfg, dict) or "dpip_participation" not in bank_cfg:
            raise ValueError(f"Missing key 'dpip_participation' for bank '{bank_id}' in {file_name}.")
        flag = bank_cfg["dpip_participation"]
        if isinstance(flag, bool) is False:
            raise ValueError(
                f"Invalid 'dpip_participation' for bank '{bank_id}' in {file_name}: must be a boolean, got {flag!r}."
            )

    return raw_delay, banks_dict


def load_nationalsim_config(
    config_path: Path | str = "config/nationalsim.yaml",
) -> tuple[timedelta, dict[str, Any]]:
    """Load and validate national simulation parameters from the YAML configuration file.

    What it is for:
        Reads the configuration file for the mock national systems from disk,
        extracting the mandatory listing delay and bank enrollment settings without silent fallbacks.

    Inputs:
        config_path: Path or string pointing to the YAML configuration file.
            Defaults to 'config/nationalsim.yaml'.

    What it returns:
        A tuple of (listing_delay, bank_settings) where listing_delay is a timedelta
        and bank_settings is a dictionary mapping bank IDs to participation configs.

    How it works:
        1. Locate and read the YAML file from disk.
        2. Ensure the top-level 'nationalsim' key exists.
        3. Validate listing delay and bank participation mappings using validate_nationalsim_settings.
        4. Return the delay as a timedelta and the bank settings dictionary.

    Errors and side effects:
        Raises ValueError if the file is missing or if listing_delay_minutes or banks
        are missing or invalid, naming the missing key and file.

    Limits:
        Settings are loaded when called; does not poll or watch the file for live changes.
    """
    path = Path(config_path)
    file_name = str(path)
    if not path.is_file():
        raise ValueError(f"Missing configuration file '{file_name}'.")

    raw_text = path.read_text(encoding="utf-8")
    parsed_yaml = yaml.safe_load(raw_text)
    if not isinstance(parsed_yaml, dict) or "nationalsim" not in parsed_yaml:
        raise ValueError(f"Missing 'nationalsim' key in {file_name}.")

    ns_section = parsed_yaml["nationalsim"]
    if not isinstance(ns_section, dict):
        raise ValueError(f"Section 'nationalsim' in {file_name} must be a mapping.")

    delay_minutes, banks_dict = validate_nationalsim_settings(ns_section, file_name=file_name)
    return timedelta(minutes=delay_minutes), banks_dict


def create_app(
    clock: Clock,
    key_provider: Callable[[], str | bytes],
    settings: dict[str, Any],
    i4c_key_provider: Callable[[], str | bytes] | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application for the national registry mock.

    What it is for:
        Builds the web application exposing DPIP screening, ingestion, dispute, revoke,
        and feedback HTTP routes. Callers must supply the clock, secret key provider,
        and registry settings explicitly so test and production environments never mix.

    Inputs:
        clock: An instance conforming to the Clock protocol, providing timezone-aware IST time.
            For example, ManualClock or ReplayClock in tests, or WallClock in production.
        key_provider: A callable returning the shared secret key used for HMAC verification.
            For example, a lambda returning a fixed test string or reading from the environment.
        settings: Dictionary holding listing_delay_minutes (int) and banks (dict).
            For example, {"listing_delay_minutes": 15, "banks": {"bank_0": {"dpip_participation": True}}}.

    What it returns:
        A fully configured FastAPI application ready to serve HTTP traffic.

    How it works:
        1. Evaluate key_provider to verify that the secret key is present and non-empty.
        2. Validate listing delay and bank participation settings from the provided dictionary.
        3. Instantiate the core RegistryCore state container with the clock and settings.
        4. Register HTTP routes for /health, /ingest, /screen, /dispute, /revoke, and /feedback.
        5. Attach error handlers so NotParticipantError returns 403 and ValueError returns 400.

    Errors and side effects:
        Raises ValueError if key_provider returns an empty key, or if settings are missing or invalid.
        In-memory state is held inside the RegistryCore instance for the lifetime of the process.

    Limits:
        Stores registry data in memory. Restarting the server clears all ingested reports.
    """
    resolved_key = key_provider()
    if not resolved_key or (isinstance(resolved_key, str) and not resolved_key.strip()):
        raise ValueError("Key provider returned an empty or whitespace key.")

    delay_minutes, bank_settings = validate_nationalsim_settings(settings, file_name="config/nationalsim.yaml")
    listing_delay = timedelta(minutes=delay_minutes)

    registry = RegistryCore(
        clock=clock,
        listing_delay=listing_delay,
        bank_settings=bank_settings,
    )
    i4c_feed = I4CFeedCore(
        clock=clock,
        listing_delay=listing_delay,
    )

    app = FastAPI(
        title="National Systems Mock (DPIP / I4C)",
        description="Mock HTTP service for national fraud registry (DPIP) and I4C shared intelligence feed.",
        version="0.1.0",
    )

    # Store references on app state so tests or inspection tools can examine them directly
    app.state.registry = registry
    app.state.i4c_feed = i4c_feed
    app.state.clock = clock
    app.state.key = resolved_key
    app.state.i4c_key = i4c_key_provider() if i4c_key_provider is not None else None

    @app.get("/health")
    def health_check() -> dict[str, str]:
        """Return service health status and current simulated time in IST."""
        current_time = clock.now_ist().isoformat()
        return {
            "status": "ok",
            "service": "nationalsim",
            "time_ist": current_time,
        }

    @app.post("/ingest", status_code=status.HTTP_200_OK)
    def ingest_report(payload: IngestRequest) -> dict[str, Any]:
        """Receive a fraud report from a bank and schedule it for registry listing."""
        try:
            entry = registry.ingest(
                bank_id=payload.bank_id,
                report_id=payload.report_id,
                identifier_hash=payload.identifier_hash,
                identifier_type=payload.identifier_type,
                reason_code=payload.reason_code,
                reported_at=payload.reported_at,
                run_id=payload.run_id,
                linked_account_hash=payload.linked_account_hash,
            )
            # When identifier is an account or phone, also ingest into I4C feed
            if payload.identifier_type in ("account", "phone"):
                i4c_feed.ingest(
                    bank_id=payload.bank_id,
                    report_id=payload.report_id,
                    identifier_hash=payload.identifier_hash,
                    identifier_type=payload.identifier_type,
                    reason_code=payload.reason_code,
                    reported_at=payload.reported_at,
                    run_id=payload.run_id,
                )
            return {
                "status": "ingested",
                "entry": entry.to_dict(),
            }
        except ValueError as exc:
            # Reject future dates or malformed payloads with an explicit 400 client error
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    @app.post("/screen", response_model=ScreenResponse)
    def screen_identifier(payload: ScreenRequest) -> ScreenResponse:
        """Screen an identifier against the registry on behalf of a participating bank."""
        try:
            result = registry.screen(
                requesting_bank_id=payload.requesting_bank_id,
                identifier_hash=payload.identifier_hash,
                identifier_type=payload.identifier_type,
                run_id=payload.run_id,
            )
            return ScreenResponse(
                match=result.match,
                status=result.status,
                distinct_accounts_linked=result.distinct_accounts_linked,
                risk_score=result.risk_score,
            )
        except NotParticipantError as exc:
            # A non-participating bank is refused with an explicit 403 Forbidden
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
        except ValueError as exc:
            # Invalid input or badly shaped hash returns 400 Bad Request
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    @app.post("/revoke")
    def revoke_report(payload: RevokeRequest) -> dict[str, Any]:
        """Revoke a previously ingested report on request of the original reporting bank."""
        try:
            revoked_entries = registry.revoke(
                bank_id=payload.bank_id,
                report_id=payload.report_id,
                run_id=payload.run_id,
            )
            return {
                "status": "revoked",
                "revoked_count": len(revoked_entries),
            }
        except ValueError as exc:
            # Unauthorized revocation by another bank or missing report raises 400
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    @app.post("/dispute")
    def dispute_identifier(payload: DisputeRequest) -> dict[str, Any]:
        """Mark all active entries for an identifier hash as contested following a customer dispute."""
        try:
            contested_entries = registry.dispute(
                bank_id=payload.bank_id,
                identifier_hash=payload.identifier_hash,
                reason_code=payload.reason_code,
                run_id=payload.run_id,
            )
            return {
                "status": "contested",
                "contested_count": len(contested_entries),
            }
        except ValueError as exc:
            # Missing entries or invalid hash raises 400
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    @app.post("/feedback")
    def record_feedback(payload: FeedbackRequest) -> dict[str, str]:
        """Record operational feedback from a bank on an identifier."""
        try:
            registry.feedback(
                bank_id=payload.bank_id,
                identifier_hash=payload.identifier_hash,
                reason_code=payload.reason_code,
                run_id=payload.run_id,
                reported_at=payload.reported_at,
            )
            return {"status": "recorded"}
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    @app.get("/i4c/feed")
    def get_i4c_feed(bank_id: str, run_id: str) -> dict[str, Any]:
        """Return visible I4C intelligence feed entries for a requesting bank.

        What it is for:
            Delivers shared fraud intelligence to institutions under the I4C program.
            Returns the bank's own reported accounts plus all publicly visible shared
            phone numbers. Unenrolled or non-participating banks can query the I4C feed.

        Inputs:
            bank_id: Identifier of the requesting bank (e.g. 'bank_0').
            run_id: Simulation namespace identifier (e.g. 'run1').

        What it returns:
            JSON dictionary containing 'status': 'ok', 'bank_id', 'run_id',
            'entries_count', and 'entries' list.

        How it works:
            1. Validate that bank_id and run_id query parameters are non-empty.
            2. Retrieve visible entries from I4CFeedCore where visible_at <= clock.now_ist().
            3. Return serialized entry dicts.

        Errors and side effects:
            Raises HTTPException 400 if bank_id or run_id is missing or empty.

        Limits:
            Returns only active, non-revoked entries that have passed the listing delay.
        """
        if not bank_id or not bank_id.strip():
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing 'bank_id' parameter.")
        if not run_id or not run_id.strip():
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing 'run_id' parameter.")

        feed_entries = app.state.i4c_feed.get_feed(bank_id=bank_id, run_id=run_id)
        return {
            "status": "ok",
            "bank_id": bank_id,
            "run_id": run_id,
            "entries_count": len(feed_entries),
            "entries": [entry.to_dict() for entry in feed_entries],
        }

    return app


def create_app_prod() -> FastAPI:
    """Create the production application reading settings and secret key from the environment.

    What it is for:
        Uvicorn or Docker Compose calls this factory function when booting the real service.
        It wires the app to real system time and loads the shared HMAC secret from the environment.

    Inputs:
        None. It reads configuration directly from environment variables.

    What it returns:
        A ready-to-run FastAPI instance configured for production serving.

    How it works:
        1. Check the NATIONALSIM_DPIP_KEY environment variable.
        2. Raise a clear ValueError immediately if the key is missing or blank.
        3. Pass a WallClock and key provider to create_app.

    Errors and side effects:
        Raises ValueError if NATIONALSIM_DPIP_KEY is missing. It reads from the environment only.

    Limits:
        Uses the machine's real clock, so this factory is meant for live network deployments,
        not for controllable simulation replays.
    """
    dpip_key = os.environ.get("NATIONALSIM_DPIP_KEY")
    if not dpip_key or not dpip_key.strip():
        raise ValueError(
            "Environment variable 'NATIONALSIM_DPIP_KEY' is missing or empty. "
            "The national simulation service requires this secret key to initialize."
        )
    i4c_key = os.environ.get("NATIONALSIM_I4C_KEY")
    if not i4c_key or not i4c_key.strip():
        raise ValueError(
            "Environment variable 'NATIONALSIM_I4C_KEY' is missing or empty. "
            "The national simulation service requires this secret key to initialize."
        )
    listing_delay, bank_settings = load_nationalsim_config()
    prod_settings: dict[str, Any] = {
        "listing_delay_minutes": int(listing_delay.total_seconds() // 60),
        "banks": bank_settings,
    }
    return create_app(
        clock=WallClock(),
        key_provider=lambda: dpip_key,
        settings=prod_settings,
        i4c_key_provider=lambda: i4c_key,
    )
