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
from nationalsim.registry import RegistryCore


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
        description="Type of identifier, either 'account' or 'device'.",
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


def load_nationalsim_config() -> tuple[timedelta, dict[str, Any]]:
    """Read registry configuration and bank participation settings from YAML.

    What it is for:
        When the application starts up, it reads default parameters such as the listing delay
        and bank participation flags from config/nationalsim.yaml.

    Inputs:
        None. The function reads the file at config/nationalsim.yaml.

    What it returns:
        A tuple of (listing_delay, bank_settings).
        listing_delay is a timedelta representing how long reports take to become visible.
        bank_settings is a dictionary of bank participation flags.

    How it works:
        1. Locate config/nationalsim.yaml relative to the repository root.
        2. If the file exists, parse the YAML structure.
        3. Read listing_delay_minutes.value and turn it into a timedelta.
        4. Read the bank participation dictionary under banks.
        5. If the file is missing, return a default 15-minute delay and empty settings.

    Errors and side effects:
        Reads config/nationalsim.yaml from the filesystem. Does not modify any files.

    Limits:
        Does not watch the file for changes. Settings are loaded once when the app is created.
    """
    config_path = Path("config/nationalsim.yaml")
    if not config_path.is_file():
        # Keep a safe fallback so the service can still boot in isolated testing directories
        return timedelta(minutes=15), {}

    raw_text = config_path.read_text(encoding="utf-8")
    parsed_yaml = yaml.safe_load(raw_text)
    if not isinstance(parsed_yaml, dict):
        return timedelta(minutes=15), {}

    ns_section = parsed_yaml.get("nationalsim", {})
    delay_dict = ns_section.get("listing_delay_minutes", {})
    delay_minutes = int(delay_dict.get("value", 15))
    banks_dict = ns_section.get("banks", {})

    return timedelta(minutes=delay_minutes), banks_dict


def create_app(
    clock: Clock | None = None,
    key_provider: Callable[[], str | bytes] | None = None,
    settings: dict[str, Any] | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application for the national registry mock.

    What it is for:
        Builds the web application exposing DPIP screening, ingestion, dispute, revoke,
        and feedback HTTP routes. Tests call this factory to inject fake clocks and test keys,
        while production calls it with system clocks and environment secrets.

    Inputs:
        clock: An instance conforming to the Clock protocol, providing timezone-aware IST time.
            If None, defaults to a WallClock using the computer's real clock.
        key_provider: A callable returning the shared secret key used for HMAC verification.
            If None, the factory reads NATIONALSIM_DPIP_KEY from the environment and fails loudly if absent.
        settings: Optional dictionary holding listing_delay_minutes and bank participation settings.
            If None, settings are loaded from config/nationalsim.yaml.

    What it returns:
        A fully configured FastAPI application ready to serve HTTP traffic.

    How it works:
        1. Resolve the time clock, defaulting to WallClock for live deployments.
        2. Resolve the secret key, ensuring the key is present and refusing to start without it.
        3. Resolve registry settings from the argument or from config/nationalsim.yaml.
        4. Instantiate the core RegistryCore state container.
        5. Register HTTP routes for /health, /ingest, /screen, /dispute, /revoke, and /feedback.
        6. Attach error handlers so business rule rejections return explicit HTTP 4xx error codes.

    Errors and side effects:
        Raises ValueError if key_provider is None and NATIONALSIM_DPIP_KEY is missing from the environment.
        In-memory state is held inside the RegistryCore instance for the lifetime of the process.

    Limits:
        Stores registry data in memory. Restarting the server clears all ingested reports.
    """
    resolved_clock: Clock = clock if clock is not None else WallClock()

    # Verify that a valid secret key is available before starting
    if key_provider is not None:
        resolved_key = key_provider()
    else:
        env_key = os.environ.get("NATIONALSIM_DPIP_KEY")
        if not env_key or not env_key.strip():
            raise ValueError(
                "Environment variable 'NATIONALSIM_DPIP_KEY' is missing or empty. "
                "The national simulation service requires this secret key to initialize."
            )
        resolved_key = env_key

    # Resolve listing delay and bank participation settings
    if settings is not None:
        delay_minutes = int(settings.get("listing_delay_minutes", 15))
        listing_delay = timedelta(minutes=delay_minutes)
        bank_settings = settings.get("banks", {})
    else:
        listing_delay, bank_settings = load_nationalsim_config()

    registry = RegistryCore(
        clock=resolved_clock,
        listing_delay=listing_delay,
        bank_settings=bank_settings,
    )

    app = FastAPI(
        title="National Systems Mock (DPIP)",
        description="Mock HTTP service for national fraud registry and DPIP intelligence screening.",
        version="0.1.0",
    )

    # Store references on app state so tests or inspection tools can examine them directly
    app.state.registry = registry
    app.state.clock = resolved_clock
    app.state.key = resolved_key

    @app.get("/health")
    def health_check() -> dict[str, str]:
        """Return service health status and current simulated time in IST."""
        current_time = resolved_clock.now_ist().isoformat()
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
        except ValueError as exc:
            # A non-participating bank or invalid hash must return an explicit 4xx error
            err_msg = str(exc)
            if "not a participating DPIP member" in err_msg:
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=err_msg) from exc
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=err_msg) from exc

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
    key = os.environ.get("NATIONALSIM_DPIP_KEY")
    if not key or not key.strip():
        raise ValueError(
            "Environment variable 'NATIONALSIM_DPIP_KEY' is missing or empty. "
            "The national simulation service requires this secret key to initialize."
        )
    return create_app(clock=WallClock(), key_provider=lambda: key)
