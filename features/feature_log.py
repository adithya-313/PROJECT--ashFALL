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

from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

import asyncpg
import orjson

from features.feature_set import load_feature_set_m1

__all__ = [
    "ALLOWED_STATUSES",
    "FeatureLogRow",
    "canonical_feature_names",
    "to_feature_values_array",
    "insert_feature_log_row",
    "read_feature_log_row",
]

ALLOWED_STATUSES: frozenset[str] = frozenset(
    {"ok", "incomplete_request", "no_watermark", "error"}
)


@lru_cache(maxsize=1)
def canonical_feature_names() -> tuple[str, ...]:
    """
    WHAT: Returns canonical M1 feature names in YAML definition order.

    WHY: Enforces strict schema ordering for dense vector training serialization in Level 16.
    """
    feature_set = load_feature_set_m1()
    return tuple(f.name for f in feature_set.features)


def to_feature_values_array(features_dict: dict[str, Any]) -> list[Any]:
    """
    WHAT: Serializes a dictionary of feature values into a JSON-compatible list matching canonical_feature_names order.

    WHY: Point-in-time logging stores feature values as a JSON array in canonical order,
    preserving exact types (integers, strings, booleans, floats, nulls). Missing keys map to None.
    """
    names = canonical_feature_names()
    return [features_dict.get(name) for name in names]


@dataclass(frozen=True)
class FeatureLogRow:
    """
    Represents a single decision feature log record in feature_log.

    Attributes:
        bank_id: Tenant commercial bank identifier.
        payment_id: Global UPI payment transaction identifier.
        decision_time: Timezone-aware timestamp of the decision evaluation (request.event_time).
        decided_at: Timezone-aware timestamp when decision was computed (clock.now_ist()).
        watermark_lag_ms: Difference in ms between decision_time and bank watermark (None if unwatermarked).
        status: Evaluation status ('ok', 'incomplete_request', 'no_watermark', 'error').
        feature_values: List holding evaluated feature values in canonical schema order.
        missing_request_fields: Optional list of missing field names if incomplete_request.
        registry_answers: Optional list of registry screening calls and responses.
        created_at: Database insertion timestamp.
    """

    bank_id: str
    payment_id: str
    decision_time: datetime
    decided_at: datetime
    watermark_lag_ms: int | None
    status: str
    feature_values: list[Any] | dict[str, Any]
    missing_request_fields: list[str] | None = None
    registry_answers: list[dict[str, Any]] | None = None
    created_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.status not in ALLOWED_STATUSES:
            raise ValueError(
                f"Invalid status: {self.status!r}. Must be one of {sorted(ALLOWED_STATUSES)}"
            )
        if not isinstance(self.bank_id, str) or not self.bank_id:
            raise ValueError(f"Invalid bank_id: {self.bank_id!r}; must be non-empty string")
        if not isinstance(self.payment_id, str) or not self.payment_id:
            raise ValueError(f"Invalid payment_id: {self.payment_id!r}; must be non-empty string")
        if self.decision_time.tzinfo is None:
            raise ValueError("decision_time must be timezone-aware")
        if self.decided_at.tzinfo is None:
            raise ValueError("decided_at must be timezone-aware")
        if self.created_at is not None and self.created_at.tzinfo is None:
            raise ValueError("created_at must be timezone-aware")

        if isinstance(self.feature_values, dict):
            object.__setattr__(
                self, "feature_values", to_feature_values_array(self.feature_values)
            )


async def insert_feature_log_row(conn: Any, row: FeatureLogRow) -> None:
    """
    WHAT: Inserts a FeatureLogRow into feature_log, automatically creating the daily partition if missing.

    WHY: Range-partitioned table by decision_time requires the target partition to exist.
    On check_violation (SQLSTATE 23514), creates the partition via create_feature_log_partition and retries once.
    """
    feature_values_json = orjson.dumps(row.feature_values).decode("utf-8")
    registry_answers_json = (
        orjson.dumps(row.registry_answers).decode("utf-8")
        if row.registry_answers is not None
        else None
    )
    created_at = (
        row.created_at
        if row.created_at is not None
        else datetime.now(timezone.utc)
    )

    query = """
    INSERT INTO feature_log (
        bank_id,
        payment_id,
        decision_time,
        decided_at,
        watermark_lag_ms,
        status,
        feature_values,
        missing_request_fields,
        registry_answers,
        created_at
    ) VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8, $9::jsonb, $10)
    """

    async def _execute() -> None:
        await conn.execute(
            query,
            row.bank_id,
            row.payment_id,
            row.decision_time,
            row.decided_at,
            row.watermark_lag_ms,
            row.status,
            feature_values_json,
            row.missing_request_fields,
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
            SELECT bank_id, payment_id, decision_time, decided_at,
                   watermark_lag_ms, status, feature_values,
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
            SELECT bank_id, payment_id, decision_time, decided_at,
                   watermark_lag_ms, status, feature_values,
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
        registry_answers = None

    missing = row["missing_request_fields"]
    missing_fields = list(missing) if missing is not None else None

    return FeatureLogRow(
        bank_id=row["bank_id"],
        payment_id=row["payment_id"],
        decision_time=row["decision_time"],
        decided_at=row["decided_at"],
        watermark_lag_ms=row["watermark_lag_ms"],
        status=row["status"],
        feature_values=feature_values,
        missing_request_fields=missing_fields,
        registry_answers=registry_answers,
        created_at=row["created_at"],
    )
