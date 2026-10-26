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
    """Stored record of an ingested fraud identifier report."""

    run_id: str
    bank_id: str
    report_id: str
    identifier_hash: str
    identifier_type: str
    reason_code: str
    reported_at: datetime
    received_at: datetime
    visible_at: datetime
    status: str  # "listed", "contested", "resolved", "revoked"
    linked_account_hash: str | None = None
    reported_minute: int | None = None
    resolved_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert entry into JSON-serializable dictionary."""
        data = asdict(self)
        data["reported_at"] = self.reported_at.isoformat()
        data["received_at"] = self.received_at.isoformat()
        data["visible_at"] = self.visible_at.isoformat()
        if self.resolved_at is not None:
            data["resolved_at"] = self.resolved_at.isoformat()
        return data


@dataclass(frozen=True)
class DisputeResolutionRecord:
    """Audit record capturing the adjudication of a customer dispute.

    What it is for:
        Maintains an immutable record of dispute outcomes and SLA timing for Issue 13-10
        and research metric N4 (time to dispute resolution).

    Inputs:
        bank_id: Bank filing the dispute on behalf of their customer.
        identifier_hash: 64-character lowercase hexadecimal hash of the disputed identifier.
        run_id: Simulation namespace identifier.
        filed_by: Originator entity, e.g. 'customer_via_bank'.
        status: Final adjudication status, 'resolved' if upheld or 'contested' if unresolved.
        filed_at: Timezone-aware datetime when the dispute was submitted.
        resolved_at: Timezone-aware datetime when the dispute resolution takes effect, or None.
        minutes_to_resolve: Elapsed minutes between filing and effective resolution, or None.
    """

    bank_id: str
    identifier_hash: str
    run_id: str
    filed_by: str
    status: str
    filed_at: datetime
    resolved_at: datetime | None
    minutes_to_resolve: int | None

    def to_dict(self) -> dict[str, Any]:
        """Convert resolution record into JSON-serializable dictionary."""
        data = asdict(self)
        data["filed_at"] = self.filed_at.isoformat()
        if self.resolved_at is not None:
            data["resolved_at"] = self.resolved_at.isoformat()
        return data


@dataclass(frozen=True)
class ScreenResult:
    """
    Standardized response returned by DPIP identifier screening.
    """

    match: bool
    status: str  # "listed", "contested", "none"
    distinct_accounts_linked: int
    risk_score: float = 0.0

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
        dispute_resolution_delay: timedelta | None = None,
    ) -> None:
        """Initialize the in-memory registry core.

        What it is for:
            Creates an isolated registry instance for a simulation, holding reported
            fraud identifiers, timestamps, dispute resolution settings, and enrolled bank participant settings.

        Inputs:
            clock: Time provider implementing the Clock protocol (e.g. ManualClock).
            listing_delay: Duration before an ingested report becomes visible for screening
                (e.g. timedelta(minutes=15)).
            bank_settings: Optional mapping of bank identifiers to configuration dictionaries.
                Defaults to an empty dictionary if omitted.
            dispute_resolution_delay: Optional duration before an upheld dispute resolves
                and clears false matches for screening. Defaults to 60 minutes.

        What it returns:
            A new RegistryCore instance ready to ingest and screen fraud reports.

        How it works:
            1. Store the injected clock, listing delay, and dispute resolution delay.
            2. Store bank settings, defaulting to an empty dictionary if None was passed.
            3. Initialize empty storage tables for registry entries, feedback records,
               telecom reassignments, and dispute resolution audit logs.

        Errors and side effects:
            None.

        Limits:
            In-memory storage only; all records are lost when the process terminates.
        """
        self._clock: Clock = clock
        self._listing_delay: timedelta = listing_delay
        self._dispute_resolution_delay: timedelta = (
            dispute_resolution_delay
            if dispute_resolution_delay is not None
            else timedelta(minutes=60)
        )
        self._bank_settings: dict[str, dict[str, Any]] = bank_settings if bank_settings is not None else {}

        # Storage collections:
        # _entries: keyed by (run_id, bank_id, report_id, identifier_type) for idempotency
        self._entries: dict[tuple[str, str, str, str], RegistryEntry] = {}
        self._feedback_records: list[FeedbackRecord] = []
        self._telecom_reassignments: dict[tuple[str, str], int] = {}
        self._resolution_records: list[DisputeResolutionRecord] = []

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
        reported_minute: int | None = None,
    ) -> RegistryEntry:
        """Ingest a fraud report for an identifier.

        What it is for:
            Records a reported fraud identifier submitted by a participating bank,
            computing its public visibility schedule based on national listing delays.

        Inputs:
            bank_id: ID of the reporting bank (e.g. 'bank_0').
            report_id: Unique report identifier from the bank.
            identifier_hash: 64-character lowercase hexadecimal hash.
            identifier_type: 'account', 'device', or 'phone'.
            reason_code: Reported fraud reason indicator.
            reported_at: Timezone-aware timestamp when report originated.
            run_id: Namespace identifier for simulation isolation.
            linked_account_hash: Optional hash of associated account for device/phone reports.
            reported_minute: Optional integer simulation minute of the report.

        What it returns:
            The created or existing RegistryEntry.

        Errors and side effects:
            Raises ValueError if identifier_hash is invalid or reported_at is in the future.

        Limits:
            In-memory storage only.
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
            reported_minute=reported_minute,
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
        """Screen an identifier hash against the registry.

        What it is for:
            Evaluates whether an identifier presented in a transaction has active fraud listings,
            taking into account listing delays, dispute adjudications, and revocations.

        Inputs:
            requesting_bank_id: Identifier of the bank requesting screening.
            identifier_hash: 64-character lowercase hexadecimal hash.
            identifier_type: 'account', 'device', or 'phone'.
            run_id: Namespace identifier.

        What it returns:
            ScreenResult with match boolean, status, distinct_accounts_linked, and risk_score.

        How it works:
            1. Verifies that the requesting bank is an active DPIP participant.
            2. Validates format of identifier_hash.
            3. Finds matching entries, transitioning resolved disputes if dispute_resolution_delay has elapsed.
            4. If all matching listings are resolved, returns match=False, status='resolved'.
            5. If no active entries, returns match=False, status='none'.
            6. If active entries remain, determines composite status ('contested' or 'listed').

        Errors and side effects:
            Raises NotParticipantError if bank is unlisted or not participating.
            Raises ValueError if identifier_hash is invalid.

        Limits:
            Read-only check; does not modify external databases.
        """
        if not self.is_participant(requesting_bank_id):
            raise NotParticipantError(f"Bank '{requesting_bank_id}' is not a participating DPIP member")

        validate_identifier_hash(identifier_hash)

        current_time = self._clock.now_ist()
        active_entries: list[RegistryEntry] = []
        all_resolved = True
        has_matching_entries = False

        for entry in self._entries.values():
            if (
                entry.run_id == run_id
                and entry.identifier_hash == identifier_hash
                and entry.identifier_type == identifier_type
            ):
                if entry.status == "revoked":
                    continue
                has_matching_entries = True

                # Transition contested entry to resolved once dispute resolution delay elapses
                if entry.status == "contested" and entry.resolved_at is not None:
                    if current_time >= entry.resolved_at:
                        entry.status = "resolved"

                if entry.status == "resolved":
                    continue

                all_resolved = False
                if entry.visible_at <= current_time:
                    active_entries.append(entry)

        if not active_entries:
            if has_matching_entries and all_resolved:
                return ScreenResult(
                    match=False,
                    status="resolved",
                    distinct_accounts_linked=0,
                    risk_score=0.0,
                )
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
        """Revoke an ingested fraud report. Only the original ingesting bank may revoke."""
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

    def register_telecom_reassignment(
        self,
        identifier_hash: str,
        reassigned_minute: int,
        run_id: str,
    ) -> None:
        """Record public telecom reassignment timing for an identifier hash.

        What it is for:
            Records the minute when telecom operators reassigned a recycled mobile number
            to a new genuine subscriber using public telecom data, avoiding access to hidden
            truth facts or private customer identities.

        Inputs:
            identifier_hash: 64-character lowercase hexadecimal hash of the telephone number.
            reassigned_minute: Integer simulation minute of reassignment (typically <= 0 for pre-world).
            run_id: Simulation namespace string.

        What it returns:
            None.

        How it works:
            1. Validates that the identifier hash is a valid 64-character lowercase hex string.
            2. Stores reassigned_minute in internal telecom map keyed by (run_id, identifier_hash).

        Errors and side effects:
            Raises ValueError if identifier_hash is invalid.

        Limits:
            Stores public telecom timeline data only.
        """
        validate_identifier_hash(identifier_hash)
        self._telecom_reassignments[(run_id, identifier_hash)] = int(reassigned_minute)

    def dispute(
        self,
        bank_id: str,
        identifier_hash: str,
        reason_code: str,
        run_id: str,
        filed_by: str = "customer_via_bank",
        reassigned_minute: int | None = None,
    ) -> list[RegistryEntry]:
        """Process a customer dispute filed through their bank against an identifier listing.

        What it is for:
            Enables a genuine customer whose assigned phone number was previously used by a fraudster
            or shared in a household to dispute wrongful collateral matches.

        Inputs:
            bank_id: Identifier of the bank submitting the dispute (e.g. 'bank_0').
            identifier_hash: 64-character lowercase hexadecimal hash of the disputed identifier.
            reason_code: Reason code describing the dispute (e.g. 'recycled_number_dispute').
            run_id: Simulation namespace string (e.g. 'run_001').
            filed_by: Submitting party description; defaults to 'customer_via_bank'.
            reassigned_minute: Optional explicit telecom reassignment minute override.

        What it returns:
            List of affected RegistryEntry instances.

        How it works:
            1. Validates that identifier_hash is a 64-character lowercase hex string.
            2. Finds all non-revoked entries for the specified run_id and identifier_hash.
            3. Looks up the public telecom reassignment minute from registry state if not supplied.
            4. Evaluates the resolution rule without using simulation truth:
               - If reassigned_minute is known and EVERY listing on that hash was reported
                 strictly before reassigned_minute, the dispute is eligible to be upheld.
                 Resolution is scheduled after dispute_resolution_delay.
                 A DisputeResolutionRecord is logged with status 'resolved' and minutes_to_resolve.
               - If reassigned_minute is unknown or any listing was reported at or after
                 reassignment (e.g. shared household phone with current fraud activity),
                 the dispute is NOT auto-upheld. The entries remain 'contested' indefinitely.
                 A DisputeResolutionRecord is logged with status 'contested' and minutes_to_resolve=None.
            5. Sets status='contested' on all matching entries (and records resolved_at if eligible).
            6. Returns the list of updated entries.

        Errors and side effects:
            Raises ValueError if identifier_hash is malformed or no active entries match.

        Limits:
            A shared-household dispute is NOT auto-upheld because the fraud listing is current;
            it remains 'contested' requiring secondary manual investigation.
        """
        validate_identifier_hash(identifier_hash)

        matching_entries = [
            e for e in self._entries.values()
            if e.run_id == run_id and e.identifier_hash == identifier_hash and e.status != "revoked"
        ]
        if not matching_entries:
            raise ValueError(f"No active entries found for hash '{identifier_hash}' in run '{run_id}'")

        # Determine telecom reassignment minute from parameter or registered public telecom data
        effective_reassigned_minute = reassigned_minute
        if effective_reassigned_minute is None:
            effective_reassigned_minute = self._telecom_reassignments.get((run_id, identifier_hash))

        current_time = self._clock.now_ist()
        is_eligible = False

        if effective_reassigned_minute is not None:
            # Check if EVERY listing on that hash was reported before reassigned_minute
            all_before = True
            for entry in matching_entries:
                entry_minute = entry.reported_minute
                if entry_minute is None:
                    # Infer minute relative to simulation start date if not explicitly stored
                    sim_start = datetime(2026, 9, 1, 0, 0, tzinfo=IST)
                    entry_minute = int((entry.reported_at - sim_start).total_seconds() // 60)
                if entry_minute >= effective_reassigned_minute:
                    all_before = False
                    break
            is_eligible = all_before

        contested_entries: list[RegistryEntry] = []
        minutes_to_resolve: int | None = None
        resolved_at: datetime | None = None

        if is_eligible:
            resolved_at = current_time + self._dispute_resolution_delay
            minutes_to_resolve = int(self._dispute_resolution_delay.total_seconds() // 60)
            status_label = "resolved"
        else:
            status_label = "contested"

        for entry in matching_entries:
            entry.status = "contested"
            entry.resolved_at = resolved_at
            contested_entries.append(entry)

        # Log resolution audit record for N4 metrics
        self._resolution_records.append(
            DisputeResolutionRecord(
                bank_id=bank_id,
                identifier_hash=identifier_hash,
                run_id=run_id,
                filed_by=filed_by,
                status=status_label,
                filed_at=current_time,
                resolved_at=resolved_at,
                minutes_to_resolve=minutes_to_resolve,
            )
        )

        return contested_entries

    def get_resolutions(self) -> list[DisputeResolutionRecord]:
        """Return a copy of all dispute resolution audit records logged so far."""
        return list(self._resolution_records)

    def feedback(
        self,
        bank_id: str,
        identifier_hash: str,
        reason_code: str,
        run_id: str,
        reported_at: datetime | None = None,
    ) -> None:
        """Record operational feedback on an identifier. Changes no matches or screening results."""
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
        """Export serializable representation of all internal records to verify state and isolation."""
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
            "resolutions": [r.to_dict() for r in self._resolution_records],
            "telecom_reassignments": {
                f"{k[0]}:{k[1]}": v for k, v in self._telecom_reassignments.items()
            },
            "bank_settings": self._bank_settings,
        }


def draw_dpip_participation(
    bank_ids: Any,
    world_ranges: dict[str, Any],
    split: str,
    rng: Any,
) -> dict[str, dict[str, bool]]:
    """Draw per-world DPIP participation enrollment settings for banks.

    What it is for:
        Determines which banking institutions take part in the shared DPIP registry
        for a simulated world, based on the research assumption probability_a_bank_takes_part_in_dpip.

    Inputs:
        bank_ids: Iterable of bank identifiers or indices (e.g. ['bank_0', 'bank_1', 'bank_2']).
        world_ranges: Dictionary containing world ranges read from config/world_ranges.yaml.
        split: Range split name, such as 'training', 'test', or 'stress'.
        rng: Injected random generator providing .uniform() and .random() methods.

    What it returns:
        Mapping of canonical bank keys (e.g. 'bank_0') to participation dictionaries:
        {'bank_0': {'dpip_participation': True}, ...}.

    How it works:
        1. Locate the 'probability_a_bank_takes_part_in_dpip' parameter block in world_ranges.
        2. Extract the [min, max] boundary list for the requested evaluation split.
        3. Draw a single uniform probability between min and max bounds using the injected generator.
        4. For each bank in bank_ids, evaluate a Bernoulli trial (rng.random() < probability).
        5. Return the resulting bank settings dictionary.

    Errors and side effects:
        Raises ValueError if probability_a_bank_takes_part_in_dpip is missing from world_ranges,
        or if the requested split is not present in its bounds mapping.

    Limits:
        Evaluated once per world; results stay fixed throughout the run.
    """
    params = world_ranges.get("parameters", world_ranges)
    if "probability_a_bank_takes_part_in_dpip" not in params:
        raise ValueError("Missing 'probability_a_bank_takes_part_in_dpip' in world_ranges.")
    entry = params["probability_a_bank_takes_part_in_dpip"]
    if split not in entry:
        raise ValueError(f"Split '{split}' not found in 'probability_a_bank_takes_part_in_dpip'.")

    bounds = entry[split]
    if not isinstance(bounds, list) or len(bounds) != 2:
        raise ValueError(f"Invalid bounds for split '{split}': expected [min, max], got {bounds!r}")

    low = float(bounds[0])
    high = float(bounds[1])
    prob = float(rng.uniform(low, high))

    bank_settings: dict[str, dict[str, bool]] = {}
    for raw_id in bank_ids:
        b_str = str(raw_id).strip()
        b_key = b_str if b_str.startswith("bank_") else f"bank_{b_str}"
        takes_part = bool(float(rng.random()) < prob)
        bank_settings[b_key] = {"dpip_participation": takes_part}

    return bank_settings

