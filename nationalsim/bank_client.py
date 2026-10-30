"""
nationalsim/bank_client.py

Bank-side HTTP client for the nationalsim service (/screen and /i4c/feed).
Enforces explicit configuration timeouts and non-storming single retries.
When nationalsim is down or unreachable, returns 'unavailable' and records
outage intervals (started_at, ended_at, calls_failed) rather than reporting
false 'no match' screening results.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import yaml

from core.event_clock import Clock, WallClock
from core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class OutageEvent:
    """Audit record capturing an outage window when nationalsim was unreachable.

    What it is for:
        Records downtime metrics for external dependency monitoring. Tracks when an outage
        began, when service was restored, and how many calls failed during the outage.

    Inputs:
        bank_id: Bank identifier experiencing the outage.
        run_id: Simulation run identifier.
        started_at: Timezone-aware timestamp when the first failure occurred.
        ended_at: Timezone-aware timestamp when connectivity was restored (None while ongoing).
        calls_failed: Total number of failed calls during this outage window.

    What it returns:
        Mutable OutageEvent dataclass instance.

    How it works:
        Updated in-place by NationalsimBankClient as calls fail and succeed.

    Errors and side effects:
        None.

    Limits:
        Scoped to client process memory.
    """

    bank_id: str
    run_id: str
    started_at: datetime
    ended_at: datetime | None
    calls_failed: int


@dataclass(frozen=True)
class ScreenResult:
    """Result of an identifier screening request against the national registry.

    What it is for:
        Encapsulates the screening verdict. When the service is down, status is
        'unavailable' and match is None. Match is NEVER False during an outage.

    Inputs:
        status: 'match', 'no_match', or 'unavailable'.
        match: True, False, or None (when unavailable).
        risk_score: Optional float risk score from registry.
        distinct_accounts_linked: Count of distinct accounts linked to the identifier.
        is_unavailable: Boolean flag indicating registry unreachable.
        first_visible_at: Optional timestamp of the earliest matching entry's initial visibility.

    What it returns:
        Immutable ScreenResult instance.

    How it works:
        Holds verdict fields for downstream authorization logic.

    Errors and side effects:
        None.

    Limits:
        Point-in-time screening evaluation.
    """

    status: str
    match: bool | None
    risk_score: float | None = None
    distinct_accounts_linked: int | None = None
    is_unavailable: bool = False
    first_visible_at: datetime | None = None


@dataclass(frozen=True)
class I4CFeedResult:
    """Result of an I4C intelligence feed query.

    What it is for:
        Encapsulates feed entries delivered by nationalsim. When unreachable,
        status is 'unavailable' and entries is empty.

    Inputs:
        status: 'ok' or 'unavailable'.
        entries: List of raw feed entry dictionaries.
        entries_count: Number of feed entries returned.
        is_unavailable: Boolean flag indicating service unreachable.

    What it returns:
        Immutable I4CFeedResult instance.

    How it works:
        Holds I4C intelligence records.

    Errors and side effects:
        None.

    Limits:
        Bounded by visible entries at query time.
    """

    status: str
    entries: list[dict[str, Any]]
    entries_count: int
    is_unavailable: bool = False


def load_bank_client_config(
    config_path: Path | str = "config/nationalsim.yaml",
) -> tuple[float, float]:
    """Loads client timeout and retry backoff settings from configuration.

    What it is for:
        Reads explicit banking client timeout and retry backoff parameters from
        config/nationalsim.yaml without hardcoding or silent fallbacks.

    Inputs:
        config_path: Path or string to the YAML configuration file.

    What it returns:
        Tuple of (timeout_seconds, retry_backoff_seconds).

    How it works:
        1. Reads and parses config/nationalsim.yaml.
        2. Validates 'client_timeout_seconds' and 'client_retry_backoff_seconds' presence.
        3. Returns numeric values as floats.

    Errors and side effects:
        Raises ValueError if configuration file or mandatory keys are missing or invalid.

    Limits:
        Evaluates at load time.
    """
    path = Path(config_path)
    file_name = str(path)
    if not path.is_file():
        raise ValueError(f"Missing configuration file '{file_name}'.")

    raw_text = path.read_text(encoding="utf-8")
    parsed_yaml = yaml.safe_load(raw_text)
    if not isinstance(parsed_yaml, dict) or "nationalsim" not in parsed_yaml:
        raise ValueError(f"Missing 'nationalsim' key in {file_name}.")

    section = parsed_yaml["nationalsim"]
    if not isinstance(section, dict):
        raise ValueError(f"Section 'nationalsim' in {file_name} must be a mapping.")

    # 1. client_timeout_seconds
    if "client_timeout_seconds" not in section:
        raise ValueError(f"Missing key 'client_timeout_seconds' in {file_name}.")
    raw_timeout = section["client_timeout_seconds"]
    if isinstance(raw_timeout, dict):
        if "value" not in raw_timeout:
            raise ValueError(f"Missing key 'client_timeout_seconds.value' in {file_name}.")
        raw_timeout = raw_timeout["value"]
    if isinstance(raw_timeout, bool) or not isinstance(raw_timeout, (int, float)) or raw_timeout <= 0:
        raise ValueError(f"Invalid 'client_timeout_seconds' in {file_name}: must be a positive number.")

    # 2. client_retry_backoff_seconds
    if "client_retry_backoff_seconds" not in section:
        raise ValueError(f"Missing key 'client_retry_backoff_seconds' in {file_name}.")
    raw_backoff = section["client_retry_backoff_seconds"]
    if isinstance(raw_backoff, dict):
        if "value" not in raw_backoff:
            raise ValueError(f"Missing key 'client_retry_backoff_seconds.value' in {file_name}.")
        raw_backoff = raw_backoff["value"]
    if isinstance(raw_backoff, bool) or not isinstance(raw_backoff, (int, float)) or raw_backoff <= 0:
        raise ValueError(f"Invalid 'client_retry_backoff_seconds' in {file_name}: must be a positive number.")

    return float(raw_timeout), float(raw_backoff)


class NationalsimBankClient:
    """Bank-side HTTP client for nationalsim /screen and /i4c/feed with outage recording.

    What it is for:
        Provides banking institutions with an HTTP interface to the mock national fraud registry
        and I4C feed. Ensures resilience during registry outages: connection failures or timeouts
        never return a fake 'no match' and instead return 'unavailable' while maintaining an audit
        log of outage events.

    Inputs:
        base_url: Base URL string of the nationalsim service (e.g. 'http://localhost:8080').
        bank_id: Bank identifier string (e.g. 'bank_0').
        run_id: Simulation run identifier string.
        config_path: Optional path to YAML configuration file; defaults to 'config/nationalsim.yaml'.
        clock: Optional Clock instance for IST timestamps; defaults to WallClock().
        timeout_seconds: Optional timeout override in seconds.
        retry_backoff_seconds: Optional backoff override in seconds.

    What it returns:
        Configured NationalsimBankClient instance.

    How it works:
        1. Resolves timeout and retry backoff from configuration.
        2. Tracks active outage state and completed outage history.
        3. Executes requests with at most one retry on connection or timeout error.
        4. If requests fail, updates or opens an OutageEvent and returns an 'unavailable' result.
        5. When a call succeeds following an outage, closes the outage with ended_at timestamp.

    Errors and side effects:
        Records outage events in client state.
        Never throws network exceptions to caller on outages; returns unavailable results.

    Limits:
        Client-side outage tracking scoped to this client instance.
    """

    def __init__(
        self,
        base_url: str,
        bank_id: str,
        run_id: str,
        config_path: Path | str = "config/nationalsim.yaml",
        clock: Clock | None = None,
        timeout_seconds: float | None = None,
        retry_backoff_seconds: float | None = None,
    ) -> None:
        cfg_timeout, cfg_backoff = load_bank_client_config(config_path)
        self.timeout_seconds: float = timeout_seconds if timeout_seconds is not None else cfg_timeout
        self.retry_backoff_seconds: float = (
            retry_backoff_seconds if retry_backoff_seconds is not None else cfg_backoff
        )
        self.base_url: str = base_url.rstrip("/")
        self.bank_id: str = bank_id
        self.run_id: str = run_id
        self.clock: Clock = clock if clock is not None else WallClock()

        self.active_outage: OutageEvent | None = None
        self.outages: list[OutageEvent] = []

    def _record_success(self) -> None:
        """Closes any ongoing outage when a call succeeds."""
        if self.active_outage is not None:
            self.active_outage.ended_at = self.clock.now_ist()
            logger.info(
                "Nationalsim outage resolved",
                bank_id=self.bank_id,
                run_id=self.run_id,
                started_at=self.active_outage.started_at.isoformat(),
                ended_at=self.active_outage.ended_at.isoformat(),
                calls_failed=self.active_outage.calls_failed,
            )
            self.active_outage = None

    def _record_failure(self) -> None:
        """Opens or updates an outage event when a call fails."""
        now = self.clock.now_ist()
        if self.active_outage is None:
            self.active_outage = OutageEvent(
                bank_id=self.bank_id,
                run_id=self.run_id,
                started_at=now,
                ended_at=None,
                calls_failed=1,
            )
            self.outages.append(self.active_outage)
            logger.warning(
                "Nationalsim outage started",
                bank_id=self.bank_id,
                run_id=self.run_id,
                started_at=now.isoformat(),
            )
        else:
            self.active_outage.calls_failed += 1
            logger.warning(
                "Nationalsim outage continuing",
                bank_id=self.bank_id,
                run_id=self.run_id,
                calls_failed=self.active_outage.calls_failed,
            )

    async def _execute_with_single_retry(
        self,
        method: str,
        path: str,
        json_data: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> tuple[bool, dict[str, Any] | None]:
        """Executes an HTTP request with at most one retry and outage recording."""
        url = f"{self.base_url}{path}"
        network_errors = (
            httpx.ConnectError,
            httpx.TimeoutException,
            httpx.NetworkError,
            httpx.RemoteProtocolError,
        )

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            # Attempt 1
            try:
                resp = await client.request(method, url, json=json_data, params=params)
                if resp.status_code == 403:
                    # Non-participant refused by registry (Rule 3)
                    self._record_failure()
                    return False, None
                resp.raise_for_status()
                self._record_success()
                return True, resp.json()
            except network_errors as err1:
                logger.warning(
                    "Nationalsim request attempt 1 failed, retrying once",
                    url=url,
                    error_type=type(err1).__name__,
                )
            except httpx.HTTPStatusError as status_err1:
                if status_err1.response.status_code == 403:
                    self._record_failure()
                    return False, None
                raise

            # Short backoff before single retry (avoiding retry storm)
            await asyncio.sleep(self.retry_backoff_seconds)

            # Attempt 2 (the single allowed retry)
            try:
                resp = await client.request(method, url, json=json_data, params=params)
                if resp.status_code == 403:
                    # Non-participant refused by registry (Rule 3)
                    self._record_failure()
                    return False, None
                resp.raise_for_status()
                self._record_success()
                return True, resp.json()
            except network_errors as err2:
                logger.error(
                    "Nationalsim request retry failed; marking outage",
                    url=url,
                    error_type=type(err2).__name__,
                )
                self._record_failure()
                return False, None
            except httpx.HTTPStatusError as status_err2:
                if status_err2.response.status_code == 403:
                    self._record_failure()
                    return False, None
                raise

    async def screen(
        self,
        identifier_hash: str,
        identifier_type: str = "account",
        as_of: datetime | None = None,
    ) -> ScreenResult:
        """Screens an identifier against the registry with resilient outage handling.

        What it is for:
            Allows a bank to screen account, device, or phone hashes prior to authorization.
            Guarantees that an offline registry returns 'unavailable' rather than a false negative.

        Inputs:
            identifier_hash: 64-character hexadecimal HMAC digest of the identifier.
            identifier_type: 'account', 'device', or 'phone'. Defaults to 'account'.
            as_of: Optional evaluation timestamp (testbed replay parameter, not part of real DPIP).

        What it returns:
            ScreenResult containing status ('match', 'no_match', or 'unavailable'),
            match boolean (None if unavailable, NEVER False on outage), risk attributes, and first_visible_at.

        How it works:
            Dispatches POST /screen to nationalsim. On outage or 403 non-participant, logs failure
            and returns ScreenResult(status='unavailable', match=None, is_unavailable=True).

        Errors and side effects:
            Records outage events in client history when service is unreachable or bank is refused.

        Limits:
            Does not hash identifiers internally; caller must provide hashed identifiers.
        """
        payload: dict[str, Any] = {
            "requesting_bank_id": self.bank_id,
            "identifier_hash": identifier_hash,
            "identifier_type": identifier_type,
            "run_id": self.run_id,
        }
        if as_of is not None:
            payload["as_of"] = as_of.isoformat()

        success, data = await self._execute_with_single_retry("POST", "/screen", json_data=payload)
        if not success or data is None:
            return ScreenResult(
                status="unavailable",
                match=None,
                risk_score=None,
                distinct_accounts_linked=None,
                is_unavailable=True,
                first_visible_at=None,
            )

        match_bool = bool(data["match"])
        status_str = "match" if match_bool else "no_match"
        raw_first_visible = data.get("first_visible_at")
        first_visible_at: datetime | None = None
        if raw_first_visible is not None:
            first_visible_at = datetime.fromisoformat(raw_first_visible)

        return ScreenResult(
            status=status_str,
            match=match_bool,
            risk_score=data.get("risk_score"),
            distinct_accounts_linked=data.get("distinct_accounts_linked"),
            is_unavailable=False,
            first_visible_at=first_visible_at,
        )

    async def get_i4c_feed(self) -> I4CFeedResult:
        """Retrieves visible entries from the I4C intelligence feed with outage handling.

        What it is for:
            Fetches shared fraud intelligence for the requesting bank.
            On registry failure, returns an 'unavailable' result and logs an outage.

        Inputs:
            None. Uses bank_id and run_id configured on the client.

        What it returns:
            I4CFeedResult containing status ('ok' or 'unavailable'), entries list, and count.

        How it works:
            Dispatches GET /i4c/feed?bank_id=...&run_id=... to nationalsim. On outage,
            returns I4CFeedResult(status='unavailable', entries=[], entries_count=0, is_unavailable=True).

        Errors and side effects:
            Records outage events when service is down.

        Limits:
            Returns entries visible at query time.
        """
        params = {
            "bank_id": self.bank_id,
            "run_id": self.run_id,
        }
        success, data = await self._execute_with_single_retry("GET", "/i4c/feed", params=params)
        if not success or data is None:
            return I4CFeedResult(
                status="unavailable",
                entries=[],
                entries_count=0,
                is_unavailable=True,
            )

        entries = data.get("entries", [])
        return I4CFeedResult(
            status="ok",
            entries=entries,
            entries_count=len(entries),
            is_unavailable=False,
        )

    def screen_sync(
        self,
        identifier_hash: str,
        identifier_type: str = "account",
        as_of: datetime | None = None,
    ) -> ScreenResult:
        """Synchronous wrapper for screen()."""
        return asyncio.run(self.screen(identifier_hash, identifier_type, as_of=as_of))

    def get_i4c_feed_sync(self) -> I4CFeedResult:
        """Synchronous wrapper for get_i4c_feed()."""
        return asyncio.run(self.get_i4c_feed())
