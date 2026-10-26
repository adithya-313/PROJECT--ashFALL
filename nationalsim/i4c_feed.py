"""
nationalsim/i4c_feed.py

In-memory store and distribution logic for the Indian Cyber Crime Coordination Centre (I4C)
shared fraud intelligence feed. Decoupled from HTTP transport.
Holds only hashed identifiers, bank IDs, reason codes, and timestamps.
"""

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any

from core.event_clock import IST, Clock
from nationalsim.keyed_hash import validate_identifier_hash


def _normalize_bank_id(bank_id: str) -> str:
    """Normalize a bank identifier string to canonical 'bank_N' format."""
    s = str(bank_id).strip()
    return s if s.startswith("bank_") else f"bank_{s}"


@dataclass(frozen=True)
class I4CEntry:
    """Stored intelligence item in the national I4C fraud feed.

    What it is for:
        Represents a single flagged fraud identifier (a compromised bank account or
        illicit telephone number) distributed to financial institutions under the
        national I4C framework.

    Inputs:
        run_id: Simulation namespace identifier string (e.g. 'run_001').
        bank_id: Originating bank identifier string (e.g. 'bank_0').
        report_id: Unique fraud report string (e.g. 'R001').
        identifier_hash: 64-character lowercase hexadecimal HMAC digest.
        identifier_type: Categorical type string ('account' or 'phone').
        reason_code: Standardized reason string (e.g. 'fraud_complaint_confirmed').
        reported_at: Timezone-aware datetime when the reporting bank filed the record.
        received_at: Timezone-aware datetime when the registry received the record.
        visible_at: Timezone-aware datetime when the record becomes publicly visible in the feed.
        status: Lifecycle status string ('listed', 'contested', or 'revoked').

    What it returns:
        An immutable I4CEntry dataclass instance.

    How it works:
        1. Stores hashed identifier and metadata.
        2. Implements to_dict for JSON serialization with ISO-formatted timestamps.

    Errors and side effects:
        None. Dataclass is frozen and immutable.

    Limits:
        Does not store raw phone numbers or account numbers.
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
    status: str = "listed"

    def to_dict(self) -> dict[str, Any]:
        """Convert entry into JSON-serializable dictionary."""
        data = asdict(self)
        data["reported_at"] = self.reported_at.isoformat()
        data["received_at"] = self.received_at.isoformat()
        data["visible_at"] = self.visible_at.isoformat()
        return data


class I4CFeedCore:
    """Core in-memory feed manager for national I4C fraud intelligence.

    What it is for:
        Maintains ingested fraud records and serves them to requesting financial institutions.
        Enforces strict distribution rules: account records are confidential to the bank that
        holds them, while phone entries are broadcast globally to all banks regardless of
        whether they participate in DPIP screening.

    Inputs:
        clock: Injected clock instance implementing the Clock protocol.
        listing_delay: Duration between receipt and public visibility (e.g. timedelta(minutes=15)).

    What it returns:
        A new I4CFeedCore instance.

    How it works:
        1. Stores clock and listing delay configuration.
        2. Maintains an in-memory dictionary of entries keyed by
           (run_id, bank_id, report_id, identifier_type) for idempotent re-ingestion.
        3. Filters entries by run namespace, visibility timestamp, and bank authorization.

    Errors and side effects:
        None upon initialization.

    Limits:
        Stores data strictly in process memory; cleared when the process exits.
    """

    def __init__(self, clock: Clock, listing_delay: timedelta) -> None:
        """Initialize the I4C feed store with an injected clock and listing delay."""
        self._clock: Clock = clock
        self._listing_delay: timedelta = listing_delay
        self._entries: dict[tuple[str, str, str, str], I4CEntry] = {}

    def ingest(
        self,
        bank_id: str,
        report_id: str,
        identifier_hash: str,
        identifier_type: str,
        reason_code: str,
        reported_at: datetime,
        run_id: str,
    ) -> I4CEntry:
        """Ingest a fraud intelligence report into the I4C feed.

        What it is for:
            Records an ingested fraud identifier report originating from a bank or telecom
            source, scheduling its visibility according to the national listing delay.

        Inputs:
            bank_id: Identifier of the originating bank (e.g. 'bank_0').
            report_id: Unique report ID string (e.g. 'R-100').
            identifier_hash: 64-character lowercase hexadecimal hash.
            identifier_type: 'account' or 'phone'.
            reason_code: Reason categorization string (e.g. 'fraud_complaint_confirmed').
            reported_at: Timezone-aware timestamp when report originated.
            run_id: Simulation namespace string (e.g. 'run1').

        What it returns:
            The created or idempotently existing I4CEntry.

        How it works:
            1. Validates that the identifier hash is a valid 64-character lowercase hex string.
            2. Enforces that reported_at is timezone-aware and not in the future of the clock.
            3. Checks for an existing identical entry (run_id, bank_id, report_id, identifier_type)
               and returns it immediately for idempotency.
            4. Calculates visible_at = current_time + listing_delay.
            5. Stores and returns the new I4CEntry.

        Errors and side effects:
            Raises ValueError if identifier_hash is malformed, reported_at is naive, or
            reported_at is in the future.

        Limits:
            In-memory storage only.
        """
        validate_identifier_hash(identifier_hash)

        if reported_at.tzinfo is None:
            raise ValueError("reported_at must be timezone-aware")
        reported_at_ist = reported_at.astimezone(IST)
        current_time = self._clock.now_ist()

        if reported_at_ist > current_time:
            raise ValueError(
                f"Report time {reported_at_ist.isoformat()} is in the future "
                f"relative to clock {current_time.isoformat()}"
            )

        entry_key = (run_id, bank_id, report_id, identifier_type)
        if entry_key in self._entries:
            return self._entries[entry_key]

        received_at = current_time
        visible_at = received_at + self._listing_delay

        entry = I4CEntry(
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
        )
        self._entries[entry_key] = entry
        return entry

    def get_feed(
        self,
        bank_id: str,
        run_id: str,
    ) -> list[I4CEntry]:
        """Retrieve visible I4C intelligence entries for the requesting bank.

        What it is for:
            Delivers shared fraud intelligence to a bank. Under Indian Cyber Crime Coordination
            Centre rules, account data is strictly private to the institution holding the account,
            whereas phone numbers are shared across all banking institutions.

        Inputs:
            bank_id: Requesting bank identifier (e.g. 'bank_0' or '0').
            run_id: Simulation namespace string (e.g. 'run1').

        What it returns:
            List of visible I4CEntry records matching authorization and timeline criteria.

        How it works:
            1. Normalizes requesting bank identifier.
            2. Evaluates current simulation time from injected clock.
            3. Iterates over stored entries matching run_id.
            4. Discards unexpired listing delay records (visible_at > current_time) and revoked records.
            5. For phone entries: includes unconditionally for all requesting banks.
            6. For account entries: includes if and only if originating bank matches requesting bank.

        Errors and side effects:
            None. Read-only query.

        Limits:
            Returns only active, visible entries.
        """
        req_bank = _normalize_bank_id(bank_id)
        current_time = self._clock.now_ist()
        visible_entries: list[I4CEntry] = []

        for entry in self._entries.values():
            if entry.run_id != run_id:
                continue
            if entry.status == "revoked":
                continue
            if entry.visible_at > current_time:
                continue

            # Phone numbers are broadcast to all institutions
            if entry.identifier_type == "phone":
                visible_entries.append(entry)
            # Account entries are delivered exclusively to the institution holding the account
            elif entry.identifier_type == "account":
                if _normalize_bank_id(entry.bank_id) == req_bank:
                    visible_entries.append(entry)

        return visible_entries

    def export_feed(self) -> dict[str, Any]:
        """Export serialized feed state for audit and testing inspection."""
        return {
            "entries": [e.to_dict() for e in self._entries.values()],
        }
