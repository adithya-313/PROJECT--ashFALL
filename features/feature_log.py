# File: features/feature_log.py
# Purpose: Level 15c feature logging, canonical schema ordering, and partitioned database storage.
# Issue: Level 15c watermark and feature log (plan Task 1, D2, S4).
"""
WHAT: Provides canonical M1 feature ordering, serialization to JSON arrays,
and database operations for the partitioned feature_log table.

WHY: High-throughput shadow feature logging enables point-in-time model evaluation and
offline training in Level 16 without data drift. Daily range partitioning ensures scalable
table maintenance while maintaining tenant isolation via PostgreSQL Row Level Security.
"""

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

import asyncpg
import orjson

from features.feature_set import FEATURE_SET_VERSION, load_feature_set_m1

__all__ = [
    "ALLOWED_STATUSES",
    "FeatureLogRow",
    "canonical_feature_names",
    "canonical_feature_names_sha256",
    "to_ordered_values",
    "to_feature_values_array",
    "insert_feature_log_row",
    "read_feature_log_row",
]

ALLOWED_STATUSES: frozenset[str] = frozenset({"ok", "incomplete_request", "no_watermark", "error"})


@lru_cache(maxsize=1)
def canonical_feature_names() -> tuple[str, ...]:
    """
    WHAT: Returns canonical M1 feature names in YAML definition order.

    WHY: Enforces strict schema ordering for dense vector training serialization in Level 16.
    """
    feature_set = load_feature_set_m1()
    return tuple(f.name for f in feature_set.features)


@lru_cache(maxsize=1)
def canonical_feature_names_sha256() -> str:
    """Returns SHA256 hex digest of canonical feature names joined by newline."""
    names = canonical_feature_names()
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()


def to_ordered_values(features_dict: dict[str, Any]) -> list[Any]:
    """
    WHAT: Serializes a dictionary of feature values into a JSON-compatible list matching canonical_feature_names order.

    WHY: Point-in-time logging stores feature values as a JSON array in canonical order,
    preserving exact types (integers, strings, booleans, floats, nulls).
    Raises ValueError if keys in features_dict do not match the canonical set exactly.
    """
    names = canonical_feature_names()
    expected = set(names)
    actual = set(features_dict.keys())
    if actual != expected:
        raise ValueError(
            "Feature dict keys do not match canonical schema. "
            f"Missing: {expected - actual}, Unexpected: {actual - expected}"
        )
    return [features_dict[name] for name in names]


to_feature_values_array = to_ordered_values


@dataclass(frozen=True)
class FeatureLogRow:
    """
    Represents a single decision feature log record in feature_log.

    Attributes:
        bank_id: Tenant commercial bank identifier.
        payment_id: Global UPI payment transaction identifier.
        run_id: 16-hex simulation run identifier.
        account_id: Customer account identifier.
        decision_time: Timezone-aware timestamp of the decision evaluation (request.event_time).
        decided_at: Timezone-aware timestamp when decision was computed (clock.now_ist()).
        arrival_cutoff: Point-in-time ingestion watermark cutoff (None only when status='no_watermark').
        watermark_lag_ms: Difference in ms between decided_at and bank watermark (None if unwatermarked).
        status: Evaluation status ('ok', 'incomplete_request', 'no_watermark', 'error').
        feature_values: List holding evaluated feature values in canonical schema order.
        feature_set_version: Version identifier for the feature set ('m1').
        feature_names_sha256: SHA-256 hash digest of canonical feature names.
        error_type: Exception class name on error status, else None.
        error_message: Error string description on error status, else None.
        missing_request_fields: List of missing field names if incomplete_request (empty list, never None).
        registry_answers: List of registry screening calls and responses (empty list, never None).
        created_at: Database insertion timestamp.
    """

    bank_id: str
    payment_id: str
    run_id: str
    account_id: str
    decision_time: datetime
    decided_at: datetime
    arrival_cutoff: datetime | None
    watermark_lag_ms: int | None
    status: str
    feature_values: list[Any] | dict[str, Any]
    feature_set_version: str = FEATURE_SET_VERSION
    feature_names_sha256: str = ""
    error_type: str | None = None
    error_message: str | None = None
    missing_request_fields: list[str] = field(default_factory=list)
    registry_answers: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.status not in ALLOWED_STATUSES:
            raise ValueError(f"Invalid status: {self.status!r}. Must be one of {sorted(ALLOWED_STATUSES)}")
        if not isinstance(self.bank_id, str) or not self.bank_id:
            raise ValueError(f"Invalid bank_id: {self.bank_id!r}; must be non-empty string")
        if not isinstance(self.payment_id, str) or not self.payment_id:
            raise ValueError(f"Invalid payment_id: {self.payment_id!r}; must be non-empty string")
        if not isinstance(self.run_id, str) or not self.run_id:
            raise ValueError(f"Invalid run_id: {self.run_id!r}; must be non-empty string")
        if not isinstance(self.account_id, str) or not self.account_id:
            raise ValueError(f"Invalid account_id: {self.account_id!r}; must be non-empty string")
        if self.decision_time.tzinfo is None:
            raise ValueError("decision_time must be timezone-aware")
        if self.decided_at.tzinfo is None:
            raise ValueError("decided_at must be timezone-aware")
        if self.created_at is not None and self.created_at.tzinfo is None:
            raise ValueError("created_at must be timezone-aware")

        # G2 CHECK 1: arrival_cutoff IS NOT NULL OR status = 'no_watermark'
        if self.arrival_cutoff is None and self.status != "no_watermark":
            raise ValueError("arrival_cutoff cannot be None unless status is 'no_watermark'")
        if self.arrival_cutoff is not None and self.arrival_cutoff.tzinfo is None:
            raise ValueError("arrival_cutoff must be timezone-aware")

        # G2 CHECK 2: both NOT NULL when status='error', both NULL otherwise
        if self.status == "error":
            if self.error_type is None or self.error_message is None:
                raise ValueError("Both error_type and error_message must be non-None when status is 'error'")
        else:
            if self.error_type is not None or self.error_message is not None:
                raise ValueError("Both error_type and error_message must be None when status is not 'error'")

        # missing_request_fields and registry_answers are empty list, never None
        if self.missing_request_fields is None:
            raise ValueError("missing_request_fields must be a list (never None)")
        if not isinstance(self.missing_request_fields, list):
            raise ValueError("missing_request_fields must be a list")

        if self.registry_answers is None:
            raise ValueError("registry_answers must be a list (never None)")
        if not isinstance(self.registry_answers, list):
            raise ValueError("registry_answers must be a list")

        if not self.feature_names_sha256:
            object.__setattr__(self, "feature_names_sha256", canonical_feature_names_sha256())
        if not self.feature_set_version:
            object.__setattr__(self, "feature_set_version", FEATURE_SET_VERSION)

        if isinstance(self.feature_values, dict):
            object.__setattr__(self, "feature_values", to_feature_values_array(self.feature_values))


async def insert_feature_log_row(conn: Any, row: FeatureLogRow) -> None:
    """
    WHAT: Inserts a FeatureLogRow into feature_log, automatically creating the daily partition if missing.

    WHY: Range-partitioned table by decision_time requires the target partition to exist.
    On check_violation (SQLSTATE 23514), creates the partition via create_feature_log_partition and retries once.
    """
    feature_values_json = orjson.dumps(row.feature_values).decode("utf-8")
    registry_answers_json = orjson.dumps(row.registry_answers).decode("utf-8")
    created_at = row.created_at if row.created_at is not None else datetime.now(timezone.utc)

    query = """
    INSERT INTO feature_log (
        bank_id,
        payment_id,
        run_id,
        account_id,
        decision_time,
        decided_at,
        arrival_cutoff,
        watermark_lag_ms,
        status,
        feature_set_version,
        feature_names_sha256,
        feature_values,
        error_type,
        error_message,
        missing_request_fields,
        registry_answers,
        created_at
    ) VALUES (
        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12::jsonb, $13, $14, $15, $16::jsonb, $17
    )
    """

    async def _execute() -> None:
        await conn.execute(
            query,
            row.bank_id,
            row.payment_id,
            row.run_id,
            row.account_id,
            row.decision_time,
            row.decided_at,
            row.arrival_cutoff,
            row.watermark_lag_ms,
            row.status,
            row.feature_set_version,
            row.feature_names_sha256,
            feature_values_json,
            row.error_type,
            row.error_message,
            list(row.missing_request_fields),
            registry_answers_json,
            created_at,
        )

    if hasattr(conn, "transaction"):
        try:
            async with conn.transaction():
                await _execute()
        except asyncpg.CheckViolationError:
            await conn.execute(
                "SELECT create_feature_log_partition($1::date)",
                row.decision_time.date(),
            )
            await _execute()
    else:
        try:
            await _execute()
        except asyncpg.CheckViolationError:
            await conn.execute(
                "SELECT create_feature_log_partition($1::date)",
                row.decision_time.date(),
            )
            await _execute()


async def read_feature_log_row(
    conn: Any, bank_id: str, payment_id: str, decision_time: datetime | None = None
) -> FeatureLogRow | None:
    """
    WHAT: Queries the feature log for a specific bank_id, payment_id, and optional decision_time.

    Returns:
        FeatureLogRow if found, else None.
    """
    if decision_time is not None:
        row = await conn.fetchrow(
            """
            SELECT bank_id, payment_id, run_id, account_id, decision_time, decided_at,
                   arrival_cutoff, watermark_lag_ms, status, feature_set_version,
                   feature_names_sha256, feature_values, error_type, error_message,
                   missing_request_fields, registry_answers, created_at
            FROM feature_log
            WHERE bank_id = $1 AND payment_id = $2 AND decision_time = $3
            LIMIT 1
            """,
            bank_id,
            payment_id,
            decision_time,
        )
    else:
        row = await conn.fetchrow(
            """
            SELECT bank_id, payment_id, run_id, account_id, decision_time, decided_at,
                   arrival_cutoff, watermark_lag_ms, status, feature_set_version,
                   feature_names_sha256, feature_values, error_type, error_message,
                   missing_request_fields, registry_answers, created_at
            FROM feature_log
            WHERE bank_id = $1 AND payment_id = $2
            ORDER BY decision_time DESC
            LIMIT 1
            """,
            bank_id,
            payment_id,
        )

    if row is None:
        return None

    fv = row["feature_values"]
    if isinstance(fv, (str, bytes)):
        feature_values = orjson.loads(fv)
    elif isinstance(fv, list):
        feature_values = fv
    elif isinstance(fv, dict):
        feature_values = to_feature_values_array(fv)
    else:
        raise ValueError(f"Unexpected feature_values type: {type(fv)}")

    ra = row["registry_answers"]
    if ra is not None:
        if isinstance(ra, (str, bytes)):
            registry_answers = orjson.loads(ra)
        elif isinstance(ra, list):
            registry_answers = ra
        else:
            raise ValueError(f"Unexpected registry_answers type: {type(ra)}")
    else:
        registry_answers = []

    missing = row["missing_request_fields"]
    missing_fields = list(missing) if missing is not None else []

    return FeatureLogRow(
        bank_id=row["bank_id"],
        payment_id=row["payment_id"],
        run_id=row["run_id"],
        account_id=row["account_id"],
        decision_time=row["decision_time"],
        decided_at=row["decided_at"],
        arrival_cutoff=row["arrival_cutoff"],
        watermark_lag_ms=row["watermark_lag_ms"],
        status=row["status"],
        feature_set_version=row["feature_set_version"],
        feature_names_sha256=row["feature_names_sha256"],
        feature_values=feature_values,
        error_type=row["error_type"],
        error_message=row["error_message"],
        missing_request_fields=missing_fields,
        registry_answers=registry_answers,
        created_at=row["created_at"],
    )
