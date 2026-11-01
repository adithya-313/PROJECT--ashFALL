# File: features/label_store.py
# Purpose: Level 16a Round 1 - label record definition, idempotent storage, and maturity reader.
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from features.account_keys import extract_run_id

LABEL_SOURCES: tuple[str, ...] = ("complaint", "registry", "analyst", "police_bulk")
LABEL_VERDICTS: tuple[str, ...] = ("fraud", "not_fraud")


@dataclass(frozen=True)
class LabelRow:
    """
    WHAT: Immutable in-memory representation of a single ground-truth or external fraud label row.

    WHY: Mirrors the PostgreSQL `labels` table columns to ensure strict typed validation
    across ingestion and maturity replay before interacting with the database.

    Attributes:
        run_id: 16-hex simulation run namespace.
        bank_id: Tenant bank identifier owning the labeled account.
        label_id: Identifier of the source event (e.g. complaint_id). Must start with run_id + '-'.
        source: Provenance source in LABEL_SOURCES.
        account_id: Account identifier labeled (must belong to bank_id).
        payment_ids: Sequence of related payment IDs, or empty.
        verdict: Fraud verdict in LABEL_VERDICTS ('fraud', 'not_fraud').
        arrival_time: Timezone-aware IST world time when the bank learned the label (used for maturity).
        region: Integer region ID (>= 0) or None if unknown.
        batch_id: Bulk confirmation batch identifier or None.
        claimed_type: Raw claim string reported by complainant or None.
        recorded_at: Real database clock (clock_timestamp()) assigned on write, or None when writing.
        via: Optional routing mechanism/batch channel (e.g. 'hold_request') or None.
    """

    run_id: str
    bank_id: str
    label_id: str
    source: str
    account_id: str
    payment_ids: tuple[str, ...]
    verdict: str
    arrival_time: datetime
    region: int | None = None
    batch_id: str | None = None
    claimed_type: str | None = None
    recorded_at: datetime | None = None
    via: str | None = None

    def __post_init__(self) -> None:
        if not self.bank_id or not isinstance(self.bank_id, str) or not self.bank_id.strip():
            raise ValueError("bank_id cannot be empty")
        if not self.account_id or not isinstance(self.account_id, str) or not self.account_id.strip():
            raise ValueError("account_id cannot be empty")
        if self.source not in LABEL_SOURCES:
            raise ValueError(f"Invalid source {self.source!r}; must be one of {LABEL_SOURCES}")
        if self.verdict not in LABEL_VERDICTS:
            raise ValueError(f"Invalid verdict {self.verdict!r}; must be one of {LABEL_VERDICTS}")
        if (
            not isinstance(self.arrival_time, datetime)
            or self.arrival_time.tzinfo is None
            or self.arrival_time.utcoffset() is None
        ):
            raise ValueError("arrival_time must be a timezone-aware datetime")
        extracted_run = extract_run_id(self.label_id)
        if extracted_run != self.run_id:
            raise ValueError(
                f"label_id run prefix {extracted_run!r} does not match run_id {self.run_id!r}"
            )
        if self.region is not None and self.region < 0:
            raise ValueError(f"region must be None or >= 0, got {self.region}")
        if self.via is not None:
            if not isinstance(self.via, str) or not self.via.strip():
                raise ValueError(f"via must be None or non-blank string, got {self.via!r}")
        if not isinstance(self.payment_ids, tuple):
            object.__setattr__(self, "payment_ids", tuple(self.payment_ids))


async def insert_labels(conn: Any, rows: Sequence[LabelRow]) -> int:
    """
    WHAT: Idempotently inserts a sequence of LabelRow records into the `labels` table.

    WHY: Uses ON CONFLICT (bank_id, source, label_id, account_id) DO NOTHING so duplicate deliveries
    or replay re-runs do not duplicate labels, and returns the honest count of rows actually inserted.

    Args:
        conn: asyncpg database connection.
        rows: Sequence of LabelRow instances to insert.

    Returns:
        The number of rows actually inserted into the database.
    """
    if not rows:
        return 0

    query = """
    INSERT INTO labels (
        run_id, bank_id, label_id, source, account_id, payment_ids,
        verdict, arrival_time, region, batch_id, claimed_type, recorded_at, via
    ) VALUES (
        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11,
        COALESCE($12, clock_timestamp()), $13
    )
    ON CONFLICT (bank_id, source, label_id, account_id) DO NOTHING
    """
    inserted_count = 0
    async with conn.transaction():
        for row in rows:
            res = await conn.execute(
                query,
                row.run_id,
                row.bank_id,
                row.label_id,
                row.source,
                row.account_id,
                list(row.payment_ids),
                row.verdict,
                row.arrival_time,
                row.region,
                row.batch_id,
                row.claimed_type,
                row.recorded_at,
                row.via,
            )
            parts = res.split()
            if len(parts) == 3 and parts[0] == "INSERT":
                inserted_count += int(parts[2])

    return inserted_count


async def read_labels_known_by(
    conn: Any,
    run_id: str,
    bank_id: str,
    cutoff: datetime,
) -> list[LabelRow]:
    """
    WHAT: Reads all labels known by a bank for a run up to a maturity cutoff timestamp.

    WHY: Enforces Ground Rule 30 and label maturity: features and training sets replayed as of
    `cutoff` can only access labels whose arrival_time <= cutoff. Deterministically ordered by
    (arrival_time, source, label_id, account_id).

    Args:
        conn: asyncpg database connection.
        run_id: 16-hex simulation run namespace.
        bank_id: Tenant bank identifier.
        cutoff: Timezone-aware datetime representing the replay/decision cutoff.

    Returns:
        List of LabelRow instances arriving on or before cutoff.

    Raises:
        ValueError: If cutoff is not timezone-aware.
    """
    if not isinstance(cutoff, datetime) or cutoff.tzinfo is None or cutoff.utcoffset() is None:
        raise ValueError("cutoff must be a timezone-aware datetime")

    query = """
    SELECT
        run_id,
        bank_id,
        label_id,
        source,
        account_id,
        payment_ids,
        verdict,
        arrival_time,
        region,
        batch_id,
        claimed_type,
        recorded_at,
        via
    FROM labels
    WHERE run_id = $1 AND bank_id = $2 AND arrival_time <= $3
    ORDER BY arrival_time, source, label_id, account_id
    """
    records = await conn.fetch(query, run_id, bank_id, cutoff)
    return [
        LabelRow(
            run_id=r["run_id"],
            bank_id=r["bank_id"],
            label_id=r["label_id"],
            source=r["source"],
            account_id=r["account_id"],
            payment_ids=tuple(r["payment_ids"] or ()),
            verdict=r["verdict"],
            arrival_time=r["arrival_time"],
            region=r["region"],
            batch_id=r["batch_id"],
            claimed_type=r["claimed_type"],
            recorded_at=r["recorded_at"],
            via=r["via"],
        )
        for r in records
    ]
