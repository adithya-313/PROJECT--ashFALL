"""
nationalsim/registry.py

In-memory registry core for national shared fraud intelligence systems (DPIP / I4C).
Pure domain implementation decoupled from HTTP transport.
Holds only hashed identifiers, bank IDs, reason codes, timestamps, and entry statuses.
"""

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any

from core.event_clock import IST, Clock
from nationalsim.keyed_hash import validate_identifier_hash


class NotParticipantError(ValueError):
    """Raised when an operation is requested by a bank that does not take part in DPIP.

    What it is for:
        DPIP screening requires banks to hold a shared secret key and agree to national
        intelligence rules. If an unauthorized bank requests screening, the registry refuses
        with this specific error so the HTTP layer can return a 403 Forbidden.

    Inputs:
        message: Plain-text explanation naming the bank that was refused.

    What it returns:
        An instance of NotParticipantError, which is also a ValueError.
    """


@dataclass
class RegistryEntry:
    """
    Stored record of an ingested fraud identifier report.
    """

    run_id: str
    bank_id: str
    report_id: str
    identifier_hash: str
    identifier_type: str
    reason_code: str
    reported_at: datetime
    received_at: datetime
    visible_at: datetime
    status: str  # "listed", "contested", "revoked"
    linked_account_hash: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert entry into JSON-serializable dictionary."""
        data = asdict(self)
        data["reported_at"] = self.reported_at.isoformat()
        data["received_at"] = self.received_at.isoformat()
        data["visible_at"] = self.visible_at.isoformat()
        return data


@dataclass(frozen=True)
class ScreenResult:
    """
    Standardized response returned by DPIP identifier screening.
    """

    match: bool
    status: str  # "listed", "contested", "none"
    distinct_accounts_linked: int
    risk_score: float | None = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert result into JSON-serializable dictionary matching mock schema."""
        return {
            "match": self.match,
            "status": self.status,
            "risk_score": self.risk_score,
            "distinct_accounts_linked": self.distinct_accounts_linked,
        }


@dataclass(frozen=True)
class FeedbackRecord:
    """
    Record of feedback submitted on an identifier.
    """

    bank_id: str
    identifier_hash: str
    reason_code: str
    reported_at: datetime
    received_at: datetime
    run_id: str


class RegistryCore:
    """
    Core in-memory registry maintaining ingested fraud identifiers namespaced by run_id.
    """

    def __init__(
        self,
        clock: Clock,
        listing_delay: timedelta,
        bank_settings: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """Initialize the in-memory registry core.

        What it is for:
            Creates an isolated registry instance for a simulation, holding reported
            fraud identifiers, timestamps, and enrolled bank participant settings.

        Inputs:
            clock: Time provider implementing the Clock protocol (e.g. ManualClock).
            listing_delay: Duration before an ingested report becomes visible for screening
                (e.g. timedelta(minutes=15)).
            bank_settings: Optional mapping of bank identifiers to configuration dictionaries.
                Defaults to an empty dictionary if omitted.

        What it returns:
            A new RegistryCore instance ready to ingest and screen fraud reports.

        How it works:
            1. Store the injected clock and listing delay.
            2. Store bank settings, defaulting to an empty dictionary if None was passed.
            3. Initialize empty storage tables for registry entries and feedback records.

        Errors and side effects:
            None.

        Limits:
            In-memory storage only; all records are lost when the process terminates.
        """
        self._clock: Clock = clock
        self._listing_delay: timedelta = listing_delay
        self._bank_settings: dict[str, dict[str, Any]] = bank_settings if bank_settings is not None else {}

        # Storage collections:
        # _entries: keyed by (run_id, bank_id, report_id, identifier_type) for idempotency
        self._entries: dict[tuple[str, str, str, str], RegistryEntry] = {}
        self._feedback_records: list[FeedbackRecord] = []

    def is_participant(self, bank_id: str) -> bool:
        """Check whether a bank is enrolled as an active DPIP participant.

        What it is for:
            Prevents unapproved banks from querying the shared national fraud registry.
            Only banks that have explicitly joined DPIP with participation enabled can
            screen identifiers.

        Inputs:
            bank_id: Bank identifier string, such as 'bank_0' or '0'.

        What it returns:
            True if the bank is explicitly listed with dpip_participation set to True;
            False if the bank is missing from settings or set to False.

        How it works:
            1. Form lookup keys for both prefixed ('bank_0') and raw numeric ('0') forms.
            2. Look up the bank in the bank settings mapping.
            3. If found, return True if and only if dpip_participation is True.
            4. If the bank is not found in settings, return False.

        Errors and side effects:
            None. This is a read-only configuration lookup.

        Limits:
            Checks only local bank settings supplied at registry initialization.
        """
        lookup_keys = [str(bank_id)]
        if str(bank_id).startswith("bank_"):
            lookup_keys.append(str(bank_id)[5:])
        else:
            lookup_keys.append(f"bank_{bank_id}")

        for key in lookup_keys:
            if key in self._bank_settings:
                return self._bank_settings[key].get("dpip_participation") is True

        return False

    def ingest(
        self,
        bank_id: str,
        report_id: str,
        identifier_hash: str,
        identifier_type: str,
        reason_code: str,
        reported_at: datetime,
        run_id: str,
        linked_account_hash: str | None = None,
    ) -> RegistryEntry:
        """
        Ingest a fraud report for an identifier.

        Args:
            bank_id: ID of the reporting bank.
            report_id: Unique report identifier from the bank.
            identifier_hash: 64-character lowercase hexadecimal hash.
            identifier_type: "account" or "device".
            reason_code: Reported fraud reason indicator.
            reported_at: Timestamp when the reporting bank generated the report.
            run_id: Namespace identifier for simulation isolation.
            linked_account_hash: Optional hash of associated account for device reports.

        Returns:
            The created or existing RegistryEntry.

        Raises:
            ValueError: If identifier_hash is invalid or reported_at is in the future.
        """
        validate_identifier_hash(identifier_hash)
        if linked_account_hash is not None:
            validate_identifier_hash(linked_account_hash)

        # Enforce temporal validity: report cannot originate in the future
        if reported_at.tzinfo is None:
            raise ValueError("reported_at must be timezone-aware")
        reported_at_ist = reported_at.astimezone(IST)
        current_time = self._clock.now_ist()

        if reported_at_ist > current_time:
            raise ValueError(
                f"Report time {reported_at_ist.isoformat()} is in the future "
                f"relative to clock {current_time.isoformat()}"
            )

        # Idempotency: return existing entry if duplicate (run_id, bank_id, report_id, identifier_type)
        entry_key = (run_id, bank_id, report_id, identifier_type)
        if entry_key in self._entries:
            return self._entries[entry_key]

        received_at = current_time
        visible_at = received_at + self._listing_delay

        entry = RegistryEntry(
            run_id=run_id,
            bank_id=bank_id,
            report_id=report_id,
            identifier_hash=identifier_hash,
            identifier_type=identifier_type,
            reason_code=reason_code,
            reported_at=reported_at_ist,
            received_at=received_at,
            visible_at=visible_at,
            status="listed",
            linked_account_hash=linked_account_hash,
        )
        self._entries[entry_key] = entry
        return entry

    def screen(
        self,
        requesting_bank_id: str,
        identifier_hash: str,
        identifier_type: str,
        run_id: str,
    ) -> ScreenResult:
        """
        Screen an identifier hash against the registry.

        Args:
            requesting_bank_id: Identifier of the bank requesting screening.
            identifier_hash: 64-character lowercase hexadecimal hash.
            identifier_type: "account" or "device".
            run_id: Namespace identifier.

        Returns:
            ScreenResult with match, status, distinct_accounts_linked, and risk_score.

        Raises:
            ValueError: If requesting bank is not a DPIP participant or hash is invalid.
        """
        if not self.is_participant(requesting_bank_id):
            raise NotParticipantError(f"Bank '{requesting_bank_id}' is not a participating DPIP member")

        validate_identifier_hash(identifier_hash)

        current_time = self._clock.now_ist()
        active_entries: list[RegistryEntry] = []

        for entry in self._entries.values():
            if (
                entry.run_id == run_id
                and entry.identifier_hash == identifier_hash
                and entry.identifier_type == identifier_type
            ):
                if entry.status != "revoked" and entry.visible_at <= current_time:
                    active_entries.append(entry)

        if not active_entries:
            return ScreenResult(
                match=False,
                status="none",
                distinct_accounts_linked=0,
                risk_score=0.0,
            )

        # Determine composite status: contested takes precedence over listed
        is_contested = any(e.status == "contested" for e in active_entries)
        status = "contested" if is_contested else "listed"

        # Compute distinct accounts linked
        distinct_accounts: set[str] = set()
        for e in active_entries:
            if e.linked_account_hash:
                distinct_accounts.add(e.linked_account_hash)
            elif e.identifier_type == "account":
                distinct_accounts.add(e.identifier_hash)

        distinct_count = len(distinct_accounts) if distinct_accounts else len(active_entries)

        return ScreenResult(
            match=True,
            status=status,
            distinct_accounts_linked=max(1, distinct_count),
            risk_score=0.0,
        )

    def revoke(
        self,
        bank_id: str,
        report_id: str,
        run_id: str,
    ) -> list[RegistryEntry]:
        """
        Revoke an ingested fraud report. Only the original ingesting bank may revoke.

        Args:
            bank_id: Identifier of the bank attempting revocation.
            report_id: Identifier of the report to revoke.
            run_id: Namespace identifier.

        Returns:
            List of revoked RegistryEntry objects.

        Raises:
            ValueError: If report not found or bank_id does not match the ingesting bank.
        """
        matching_entries = [
            e for e in self._entries.values()
            if e.run_id == run_id and e.report_id == report_id
        ]
        if not matching_entries:
            raise ValueError(f"Report '{report_id}' not found in run '{run_id}'")

        for entry in matching_entries:
            if entry.bank_id != bank_id:
                raise ValueError(
                    f"Bank '{bank_id}' cannot revoke report '{report_id}' ingested by bank '{entry.bank_id}'"
                )

        revoked_entries: list[RegistryEntry] = []
        for entry in matching_entries:
            entry.status = "revoked"
            revoked_entries.append(entry)

        return revoked_entries

    def dispute(
        self,
        bank_id: str,
        identifier_hash: str,
        reason_code: str,
        run_id: str,
    ) -> list[RegistryEntry]:
        """
        File a dispute against an identifier hash, marking all matching active entries as contested.

        Args:
            bank_id: Bank filing the dispute on behalf of a customer.
            identifier_hash: 64-character lowercase hexadecimal hash.
            reason_code: Reason code for the dispute.
            run_id: Namespace identifier.

        Returns:
            List of contested RegistryEntry objects.

        Raises:
            ValueError: If hash is invalid or no active entries match.
        """
        validate_identifier_hash(identifier_hash)

        matching_entries = [
            e for e in self._entries.values()
            if e.run_id == run_id and e.identifier_hash == identifier_hash and e.status != "revoked"
        ]
        if not matching_entries:
            raise ValueError(f"No active entries found for hash '{identifier_hash}' in run '{run_id}'")

        contested_entries: list[RegistryEntry] = []
        for entry in matching_entries:
            entry.status = "contested"
            contested_entries.append(entry)

        return contested_entries

    def feedback(
        self,
        bank_id: str,
        identifier_hash: str,
        reason_code: str,
        run_id: str,
        reported_at: datetime | None = None,
    ) -> None:
        """
        Record operational feedback on an identifier. Changes no matches or screening results.

        Args:
            bank_id: Reporting bank.
            identifier_hash: 64-character lowercase hexadecimal hash.
            reason_code: Feedback reason code.
            reported_at: Timestamp of feedback (defaults to current clock time).
            run_id: Namespace identifier.
        """
        validate_identifier_hash(identifier_hash)
        now = self._clock.now_ist()
        timestamp = reported_at.astimezone(IST) if reported_at is not None else now
        self._feedback_records.append(
            FeedbackRecord(
                bank_id=bank_id,
                identifier_hash=identifier_hash,
                reason_code=reason_code,
                reported_at=timestamp,
                received_at=now,
                run_id=run_id,
            )
        )

    def export_store(self) -> dict[str, Any]:
        """
        Export serializable representation of all internal records to verify state and isolation.
        """
        return {
            "entries": [e.to_dict() for e in self._entries.values()],
            "feedback": [
                {
                    "bank_id": f.bank_id,
                    "identifier_hash": f.identifier_hash,
                    "reason_code": f.reason_code,
                    "reported_at": f.reported_at.isoformat(),
                    "received_at": f.received_at.isoformat(),
                    "run_id": f.run_id,
                }
                for f in self._feedback_records
            ],
            "bank_settings": self._bank_settings,
        }
