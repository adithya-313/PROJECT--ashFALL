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


@dataclass(frozen=True)
class IngestionItem:
    """
    Representation of an ingestion payload ready for dispatch to DPIP/registry core.
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
) -> list[IngestionItem]:
    """
    Convert a simulated RegistryReport into one or two DPIP ingestion items.

    Always emits exactly one ingestion for the reported bank account.
    If device_fingerprint is present (not None), emits a second ingestion for the device.
    All identifier hashing is executed here on the bank side before submission to the registry.

    Args:
        report: RegistryReport instance or compatible object/dict containing report fields.
        run_id: Simulation run identifier namespace.
        key: Shared HMAC key held by DPIP participants.
        world_start: World start calendar date (e.g. date(2026, 9, 1)).

    Returns:
        List containing 1 or 2 IngestionItem instances.
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
    # Account raw value is namespaced by bank index because account numbers repeat across banks
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

    return items


def ingest_report(
    registry: RegistryCore,
    report: Any,
    run_id: str,
    key: str | bytes,
    world_start: date,
) -> list[RegistryEntry]:
    """
    Convenience helper converting a report and ingesting all resulting items into a RegistryCore.

    Args:
        registry: Target RegistryCore instance.
        report: RegistryReport object or dict.
        run_id: Namespace string.
        key: HMAC secret key.
        world_start: Simulation start calendar date.

    Returns:
        List of created or existing RegistryEntry instances.
    """
    items = registry_report_to_ingestions(report, run_id, key, world_start)
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
    return entries
