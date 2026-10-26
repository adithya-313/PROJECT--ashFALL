"""
nationalsim/adapter.py

Adapter translating simulation registry reports (world/label_records.py) into mock DPIP ingestion payloads.
This is the ONLY module aware of mock field mappings and simulator-specific record formats.
Hashing of account and device identifiers occurs strictly on the reporting bank's side before payloads
reach the national registry.
"""

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from core.event_clock import world_minute_to_ist
from nationalsim.keyed_hash import compute_identifier_hash
from nationalsim.registry import RegistryCore, RegistryEntry

# UNCERTAIN: The actual reason code strings used in live DPIP are not published in public documentation.
# We define "fraud_complaint_confirmed" as the project's standardized mock value for complaint-driven reports.
DEFAULT_FRAUD_REASON_CODE: str = "fraud_complaint_confirmed"

# UNCERTAIN: DPIP documentation names account and device screening at a high level, but exact schema strings
# ("account", "device") are project design conventions.
IDENTIFIER_TYPE_ACCOUNT: str = "account"
IDENTIFIER_TYPE_DEVICE: str = "device"
IDENTIFIER_TYPE_PHONE: str = "phone"


@dataclass(frozen=True)
class IngestionItem:
    """Representation of an ingestion payload ready for dispatch to DPIP or I4C.

    What it is for:
        Encapsulates a single hashed identifier report payload prepared on the bank side,
        prior to submission to national shared fraud intelligence systems.

    Inputs:
        bank_id: Reporting bank ID string (e.g. 'bank_0').
        report_id: Unique fraud report identifier string (e.g. 'R001').
        identifier_hash: 64-character lowercase hexadecimal HMAC digest.
        identifier_type: 'account', 'device', or 'phone'.
        reason_code: Categorical reason code (e.g. 'fraud_complaint_confirmed').
        reported_at: Timezone-aware timestamp when report was created.
        run_id: Simulation namespace string.
        linked_account_hash: Optional account hash linked to device or phone reports.

    What it returns:
        An immutable IngestionItem dataclass instance.

    How it works:
        1. Stores hashed identifier fields and metadata.
        2. Provides to_dict for JSON serialization with ISO-formatted timestamps.

    Errors and side effects:
        None.

    Limits:
        Stores only pre-hashed identifiers; never accepts raw phone numbers or account numbers.
    """

    bank_id: str
    report_id: str
    identifier_hash: str
    identifier_type: str
    reason_code: str
    reported_at: datetime
    run_id: str
    linked_account_hash: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert ingestion item to dictionary suitable for JSON serialization."""
        return {
            "bank_id": self.bank_id,
            "report_id": self.report_id,
            "identifier_hash": self.identifier_hash,
            "identifier_type": self.identifier_type,
            "reason_code": self.reason_code,
            "reported_at": self.reported_at.isoformat(),
            "run_id": self.run_id,
            "linked_account_hash": self.linked_account_hash,
        }


def registry_report_to_ingestions(
    report: Any,
    run_id: str,
    key: str | bytes,
    world_start: date,
    phone_book: Any | None = None,
) -> list[IngestionItem]:
    """Convert a simulated RegistryReport into DPIP ingestion items.

    What it is for:
        Translates a domain fraud report event into pre-hashed DPIP ingestion payloads.
        Always emits an account item. If a device fingerprint is present, emits a device item.
        If a phone book is supplied and the account holder has a phone at reported_minute,
        emits a phone ingestion item hashed with the DPIP secret key.

    Inputs:
        report: RegistryReport instance or dict containing report attributes.
        run_id: Simulation run namespace string (e.g. 'run1').
        key: Shared HMAC key held by DPIP participants.
        world_start: World start calendar date (e.g. date(2026, 9, 1)).
        phone_book: Optional PhoneBook mapping accounts to synthetic phone numbers.

    What it returns:
        List containing 1, 2, or 3 IngestionItem instances.

    How it works:
        1. Extracts report_id, reporting_bank_index, account_id, device_fingerprint,
           and reported_minute from report object or mapping.
        2. Converts reported_minute to timezone-aware IST datetime.
        3. Computes account HMAC hash and appends account IngestionItem.
        4. If device_fingerprint is present, computes device HMAC hash and appends device IngestionItem.
        5. If phone_book is provided, looks up account_id in phone_book; if found and held at
           reported_minute, computes phone HMAC hash and appends phone IngestionItem.

    Errors and side effects:
        Raises ValueError if key is empty or report format is invalid.

    Limits:
        Does not mutate the input report or phone book.
    """
    # 1. Extract report fields supporting both object attributes and dict keys
    if isinstance(report, dict):
        report_id: str = str(report["report_id"])
        reporting_bank_index: int = int(report["reporting_bank_index"])
        account_id: str = str(report["account_id"])
        device_fingerprint: str | None = report.get("device_fingerprint")
        reported_minute: int = int(report["reported_minute"])
    else:
        report_id = str(report.report_id)
        reporting_bank_index = int(report.reporting_bank_index)
        account_id = str(report.account_id)
        device_fingerprint = getattr(report, "device_fingerprint", None)
        reported_minute = int(report.reported_minute)

    # Convert event minute into timezone-aware IST datetime
    reported_at = world_minute_to_ist(reported_minute, world_start)
    bank_id = f"bank_{reporting_bank_index}"

    # 2. Hash account identifier on the bank side
    account_hash = compute_identifier_hash(
        key=key,
        run_id=run_id,
        identifier_type=IDENTIFIER_TYPE_ACCOUNT,
        raw_value=account_id,
        bank_index=reporting_bank_index,
    )

    items: list[IngestionItem] = [
        IngestionItem(
            bank_id=bank_id,
            report_id=report_id,
            identifier_hash=account_hash,
            identifier_type=IDENTIFIER_TYPE_ACCOUNT,
            reason_code=DEFAULT_FRAUD_REASON_CODE,
            reported_at=reported_at,
            run_id=run_id,
            linked_account_hash=account_hash,
        )
    ]

    # 3. If device fingerprint is present, emit a second ingestion for the device
    if device_fingerprint is not None:
        device_hash = compute_identifier_hash(
            key=key,
            run_id=run_id,
            identifier_type=IDENTIFIER_TYPE_DEVICE,
            raw_value=device_fingerprint,
        )
        items.append(
            IngestionItem(
                bank_id=bank_id,
                report_id=report_id,
                identifier_hash=device_hash,
                identifier_type=IDENTIFIER_TYPE_DEVICE,
                reason_code=DEFAULT_FRAUD_REASON_CODE,
                reported_at=reported_at,
                run_id=run_id,
                linked_account_hash=account_hash,
            )
        )

    # 4. If phone book is provided, check if holder has a phone at reported_minute
    if phone_book is not None and hasattr(phone_book, "get_record"):
        phone_record = phone_book.get_record(account_id)
        if phone_record is not None:
            # Check if holder held the phone at reported_minute
            reassigned = phone_record.reassigned_minute
            if reassigned is None or reported_minute >= reassigned:
                phone_hash = compute_identifier_hash(
                    key=key,
                    run_id=run_id,
                    identifier_type=IDENTIFIER_TYPE_PHONE,
                    raw_value=phone_record.phone_number,
                )
                items.append(
                    IngestionItem(
                        bank_id=bank_id,
                        report_id=report_id,
                        identifier_hash=phone_hash,
                        identifier_type=IDENTIFIER_TYPE_PHONE,
                        reason_code=DEFAULT_FRAUD_REASON_CODE,
                        reported_at=reported_at,
                        run_id=run_id,
                        linked_account_hash=account_hash,
                    )
                )

    return items


def registry_report_to_i4c_ingestions(
    report: Any,
    run_id: str,
    account_key: str | bytes,
    i4c_key: str | bytes,
    world_start: date,
    phone_book: Any | None = None,
) -> list[IngestionItem]:
    """Convert a simulated RegistryReport into I4C feed ingestion items.

    What it is for:
        Translates a bank fraud report into I4C intelligence records. The account
        record uses the account key (bank's own data), while the phone entry uses
        the separate national I4C key for shared multi-bank visibility.

    Inputs:
        report: RegistryReport instance or dict containing report fields.
        run_id: Simulation run namespace string.
        account_key: Key for account identifier hashing (e.g. DPIP secret key).
        i4c_key: Dedicated key for I4C phone identifier hashing (NATIONALSIM_I4C_KEY).
        world_start: World start calendar date (e.g. date(2026, 9, 1)).
        phone_book: Optional PhoneBook mapping accounts to synthetic phone numbers.

    What it returns:
        List of IngestionItem instances ready to be ingested into I4CFeedCore.

    How it works:
        1. Extract report_id, reporting_bank_index, account_id, and reported_minute.
        2. Convert reported_minute to IST datetime.
        3. Emit an account IngestionItem hashed with account_key.
        4. If phone_book is provided and the holder has a phone in the phone book at
           reported_minute, emit a phone IngestionItem hashed with i4c_key.

    Errors and side effects:
        Raises ValueError if required keys are missing or invalid.

    Limits:
        Does not emit device fingerprints (I4C focus is account and phone intelligence).
    """
    if isinstance(report, dict):
        report_id = str(report["report_id"])
        reporting_bank_index = int(report["reporting_bank_index"])
        account_id = str(report["account_id"])
        reported_minute = int(report["reported_minute"])
    else:
        report_id = str(report.report_id)
        reporting_bank_index = int(report.reporting_bank_index)
        account_id = str(report.account_id)
        reported_minute = int(report.reported_minute)

    reported_at = world_minute_to_ist(reported_minute, world_start)
    bank_id = f"bank_{reporting_bank_index}"

    account_hash = compute_identifier_hash(
        key=account_key,
        run_id=run_id,
        identifier_type=IDENTIFIER_TYPE_ACCOUNT,
        raw_value=account_id,
        bank_index=reporting_bank_index,
    )

    items: list[IngestionItem] = [
        IngestionItem(
            bank_id=bank_id,
            report_id=report_id,
            identifier_hash=account_hash,
            identifier_type=IDENTIFIER_TYPE_ACCOUNT,
            reason_code=DEFAULT_FRAUD_REASON_CODE,
            reported_at=reported_at,
            run_id=run_id,
            linked_account_hash=account_hash,
        )
    ]

    if phone_book is not None and hasattr(phone_book, "get_record"):
        phone_record = phone_book.get_record(account_id)
        if phone_record is not None:
            reassigned = phone_record.reassigned_minute
            if reassigned is None or reported_minute >= reassigned:
                # Phone entries in I4C are hashed with the dedicated I4C key
                phone_hash = compute_identifier_hash(
                    key=i4c_key,
                    run_id=run_id,
                    identifier_type=IDENTIFIER_TYPE_PHONE,
                    raw_value=phone_record.phone_number,
                )
                items.append(
                    IngestionItem(
                        bank_id=bank_id,
                        report_id=report_id,
                        identifier_hash=phone_hash,
                        identifier_type=IDENTIFIER_TYPE_PHONE,
                        reason_code=DEFAULT_FRAUD_REASON_CODE,
                        reported_at=reported_at,
                        run_id=run_id,
                        linked_account_hash=account_hash,
                    )
                )

    return items


def ingest_report(
    registry: RegistryCore,
    report: Any,
    run_id: str,
    key: str | bytes,
    world_start: date,
    phone_book: Any | None = None,
    i4c_feed: Any | None = None,
    i4c_key: str | bytes | None = None,
) -> list[RegistryEntry]:
    """Convert a report and ingest all resulting items into RegistryCore and optional I4CFeed.

    What it is for:
        Translates a simulated report into pre-hashed ingestion items and saves them
        to the DPIP registry and, when provided, the I4C feed.

    Inputs:
        registry: Target RegistryCore instance.
        report: RegistryReport object or dict.
        run_id: Simulation run namespace string.
        key: HMAC secret key for DPIP.
        world_start: Simulation start calendar date.
        phone_book: Optional PhoneBook instance for telephone intelligence mapping.
        i4c_feed: Optional I4CFeedCore instance for I4C intelligence distribution.
        i4c_key: Optional secret key for I4C phone hashing.

    What it returns:
        List of created or existing RegistryEntry instances stored in RegistryCore.

    How it works:
        1. Converts report into DPIP ingestion items with registry_report_to_ingestions.
        2. Ingests each item into registry.ingest.
        3. If i4c_feed and i4c_key are provided, converts report with registry_report_to_i4c_ingestions
           and ingests each item into i4c_feed.ingest.
        4. Returns list of created or existing registry entries.

    Errors and side effects:
        Raises ValueError if inputs or hashes are invalid.

    Limits:
        Does not bypass listing delays or bank participation rules.
    """
    items = registry_report_to_ingestions(report, run_id, key, world_start, phone_book=phone_book)
    entries: list[RegistryEntry] = []
    for item in items:
        entry = registry.ingest(
            bank_id=item.bank_id,
            report_id=item.report_id,
            identifier_hash=item.identifier_hash,
            identifier_type=item.identifier_type,
            reason_code=item.reason_code,
            reported_at=item.reported_at,
            run_id=item.run_id,
            linked_account_hash=item.linked_account_hash,
        )
        entries.append(entry)

    if i4c_feed is not None and i4c_key is not None:
        i4c_items = registry_report_to_i4c_ingestions(
            report=report,
            run_id=run_id,
            account_key=key,
            i4c_key=i4c_key,
            world_start=world_start,
            phone_book=phone_book,
        )
        for i4c_item in i4c_items:
            i4c_feed.ingest(
                bank_id=i4c_item.bank_id,
                report_id=i4c_item.report_id,
                identifier_hash=i4c_item.identifier_hash,
                identifier_type=i4c_item.identifier_type,
                reason_code=i4c_item.reason_code,
                reported_at=i4c_item.reported_at,
                run_id=i4c_item.run_id,
            )

    return entries

