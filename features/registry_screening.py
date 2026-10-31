# File: features/registry_screening.py
# Purpose: Protocol, client adapter, and fake test double for DPIP national registry screening.
# Issue: Level 15b intelligence features (plan Task 1, architect rules 1 and 3, decision 9c).
"""
WHAT: Provides the RegistryScreener Protocol abstraction, the production NationalsimRegistryScreener
adapter wrapping NationalsimBankClient, and the in-memory FakeRegistryScreener test double.

WHY:
  - Feature calculations must remain decoupled from HTTP network clients (dependency injection).
  - Keeps screening outcomes strictly typed as ScreenOutcome with status in ('match', 'no_match', 'unknown').
  - Translates NationalsimBankClient 'unavailable' (from outages or non-participant 403s) directly to 'unknown',
    ensuring downstream intelligence features never mistake an outage for a negative verdict.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol

from core.event_clock import Clock
from nationalsim.bank_client import NationalsimBankClient

__all__ = [
    "ScreenOutcome",
    "RegistryScreener",
    "NationalsimRegistryScreener",
    "FakeRegistryScreener",
    "LiveRegistryScreener",
    "RecordingRegistryScreener",
]


@dataclass(frozen=True)
class ScreenOutcome:
    """Standardized verdict returned by a RegistryScreener."""

    status: str  # 'match', 'no_match', or 'unknown'
    first_visible_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.status not in ("match", "no_match", "unknown"):
            raise ValueError(
                f"Invalid ScreenOutcome status {self.status!r}; expected 'match', 'no_match', or 'unknown'"
            )


class RegistryScreener(Protocol):
    """Protocol for screening identifier digests as of a historical decision time."""

    def screen(
        self,
        identifier_hash: str,
        identifier_type: str,
        as_of: datetime | None = None,
    ) -> ScreenOutcome:
        """Screen an identifier hash as of a specific evaluation timestamp.

        Args:
            identifier_hash: 64-character lowercase hex digest.
            identifier_type: 'account' or 'device'.
            as_of: Optional timezone-aware decision timestamp.

        Returns:
            ScreenOutcome with status 'match', 'no_match', or 'unknown'.
        """
        ...


class NationalsimRegistryScreener:
    """Production screener adapter delegating to NationalsimBankClient."""

    def __init__(self, client: NationalsimBankClient) -> None:
        self._client: NationalsimBankClient = client

    def screen(
        self,
        identifier_hash: str,
        identifier_type: str,
        as_of: datetime | None = None,
    ) -> ScreenOutcome:
        """Execute synchronous screen request via bank client."""
        res = self._client.screen_sync(
            identifier_hash=identifier_hash,
            identifier_type=identifier_type,
            as_of=as_of,
        )
        if res.status == "unavailable" or res.is_unavailable or res.match is None:
            return ScreenOutcome(status="unknown", first_visible_at=None)

        if res.match is True:
            return ScreenOutcome(status="match", first_visible_at=res.first_visible_at)

        return ScreenOutcome(status="no_match", first_visible_at=None)


class FakeRegistryScreener:
    """In-memory fixed table screener for unit tests and deterministic simulation."""

    def __init__(
        self,
        table: dict[tuple[str, str], ScreenOutcome] | None = None,
        default_outcome: ScreenOutcome | None = None,
    ) -> None:
        self._table: dict[tuple[str, str], ScreenOutcome] = dict(table) if table is not None else {}
        self._default_outcome: ScreenOutcome = (
            default_outcome if default_outcome is not None else ScreenOutcome(status="no_match", first_visible_at=None)
        )
        self.calls: list[dict[str, Any]] = []

    def set_outcome(
        self,
        identifier_hash: str,
        identifier_type: str,
        outcome: ScreenOutcome,
    ) -> None:
        """Register a fixed outcome for a specific hash and type."""
        self._table[(identifier_hash, identifier_type)] = outcome

    def screen(
        self,
        identifier_hash: str,
        identifier_type: str,
        as_of: datetime | None = None,
    ) -> ScreenOutcome:
        """Look up fixed outcome from table or return default."""
        self.calls.append(
            {
                "identifier_hash": identifier_hash,
                "identifier_type": identifier_type,
                "as_of": as_of,
            }
        )
        return self._table.get((identifier_hash, identifier_type), self._default_outcome)


class LiveRegistryScreener:
    """
    WHAT: Production live screener adapter ensuring screen requests never pass an as_of timestamp.

    WHY: Live operational decisions must evaluate against currently active registry entries,
    preventing simulation replay timestamps from leaking into live traffic.
    """

    def __init__(self, screener: RegistryScreener) -> None:
        self._screener: RegistryScreener = screener

    def screen(
        self,
        identifier_hash: str,
        identifier_type: str,
        as_of: datetime | None = None,
    ) -> ScreenOutcome:
        """Always forwards screen with as_of=None, ignoring any passed as_of."""
        return self._screener.screen(identifier_hash, identifier_type, as_of=None)


class RecordingRegistryScreener:
    """
    WHAT: Wrapper that records in-memory all screening decisions for telemetry and audit.

    WHY: Enables audit logging, offline validation, and replay comparison.
    """

    def __init__(self, screener: RegistryScreener, clock: Clock | None = None) -> None:
        self._screener: RegistryScreener = screener
        self._clock: Clock | None = clock
        self.answers: list[dict[str, Any]] = []

    def screen(
        self,
        identifier_hash: str,
        identifier_type: str,
        as_of: datetime | None = None,
    ) -> ScreenOutcome:
        """Forwards screen to underlying screener and records request/response parameters."""
        outcome = self._screener.screen(identifier_hash, identifier_type, as_of=as_of)
        received_at = self._clock.now_ist() if self._clock is not None else datetime.now(timezone.utc)
        self.answers.append(
            {
                "identifier_hash": identifier_hash,
                "identifier_type": identifier_type,
                "as_of": as_of,
                "outcome": outcome,
                "received_at": received_at,
            }
        )
        return outcome
