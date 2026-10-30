# File: features/feature_log.py
# Purpose: Level 15c feature logging, canonical schema ordering, and partitioned database storage.
# Issue: Level 15c watermark and feature log (plan Task 1, D2, S4).
"""
WHAT: Provides canonical M1 feature ordering, serialization to dense numeric arrays,
and database operations for the partitioned bank_feature_log table.

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
    "to_ordered_values",
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


def to_ordered_values(features_dict: dict[str, Any]) -> list[float | None]:
    """
    WHAT: Serializes a dictionary of feature values into a dense list matching canonical_feature_names order.

    WHY: Training and evaluation in Level 16 requires a deterministic, fixed-order float array.
    Missing keys or None values map to None. Floats and ints are cast to float; booleans are cast to 1.0 / 0.0.
    """
    names = canonical_feature_names()
    ordered: list[float | None] = []
    for name in names:
        val = features_dict.get(name)
        if val is None:
            ordered.append(None)
        elif isinstance(val, bool):
            ordered.append(1.0 if val else 0.0)
        elif isinstance(val, (int, float)):
            ordered.append(float(val))
        else:
            raise ValueError(f"Feature {name!r} has invalid non-numeric value: {val!r}")
    return ordered


@dataclass(frozen=True)
class FeatureLogRow:
    """
    Represents a single decision feature log record in bank_feature_log.

    Attributes:
        bank_id: Tenant commercial bank identifier.
        payment_id: Global UPI payment transaction identifier.
        decision_id: Unique decision evaluation identifier.
        decision_time: Timezone-aware timestamp of the decision evaluation.
        arrival_watermark: Ingestion watermark at decision evaluation time (None if unwatermarked).
        status: Evaluation status ('ok', 'incomplete_request', 'no_watermark', 'error').
        feature_values: Dictionary holding evaluated feature values.
        feature_values_ordered: Optional ordered list of floats matching canonical schema order.
        computed_at: Timezone-aware timestamp when feature computation finished.
        id: Optional surrogate database row identifier.
    """

    bank_id: str
    payment_id: str
    decision_id: str
    decision_time: datetime
    arrival_watermark: datetime | None
    status: str
    feature_values: dict[str, Any]
    feature_values_ordered: list[float | None] | None = None
    computed_at: datetime | None = None
    id: int | None = None

    def __post_init__(self) -> None:
        if self.status not in ALLOWED_STATUSES:
            raise ValueError(
                f"Invalid status: {self.status!r}. Must be one of {sorted(ALLOWED_STATUSES)}"
            )
        if self.decision_time.tzinfo is None:
            raise ValueError("decision_time must be timezone-aware")
        if self.arrival_watermark is not None and self.arrival_watermark.tzinfo is None:
            raise ValueError("arrival_watermark must be timezone-aware")
        if self.computed_at is not None and self.computed_at.tzinfo is None:
            raise ValueError("computed_at must be timezone-aware")

        if self.feature_values_ordered is None and self.status == "ok":
            object.__setattr__(
                self, "feature_values_ordered", to_ordered_values(self.feature_values)
            )


async def insert_feature_log_row(conn: Any, row: FeatureLogRow) -> int:
    """
    WHAT: Inserts a FeatureLogRow into bank_feature_log, automatically creating the daily partition if missing.

    WHY: Range-partitioned table by decision_time requires the target partition to exist.
    On check_violation (SQLSTATE 23514), creates the partition via create_feature_log_partition and retries once.

    Returns:
        Integer primary key id of the inserted row.
    """
    feature_values_json = orjson.dumps(row.feature_values).decode("utf-8")
    computed_at = (
        row.computed_at
        if row.computed_at is not None
        else datetime.now(timezone.utc)
    )

    ordered = row.feature_values_ordered
    if ordered is None and row.status == "ok":
        ordered = to_ordered_values(row.feature_values)

    query = """
    INSERT INTO bank_feature_log (
        bank_id,
        payment_id,
        decision_id,
        decision_time,
        arrival_watermark,
        status,
        feature_values,
        feature_values_ordered,
        computed_at
    ) VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8, $9)
    RETURNING id
    """

    async def _execute() -> int:
        res = await conn.fetchval(
            query,
            row.bank_id,
            row.payment_id,
            row.decision_id,
            row.decision_time,
            row.arrival_watermark,
            row.status,
            feature_values_json,
            ordered,
            computed_at,
        )
        return int(res)

    if hasattr(conn, "transaction"):
        try:
            async with conn.transaction():
                return await _execute()
        except asyncpg.CheckViolationError:
            await conn.execute(
                "SELECT create_feature_log_partition($1::date)",
                row.decision_time.date(),
            )
            return await _execute()
    else:
        try:
            return await _execute()
        except asyncpg.CheckViolationError:
            await conn.execute(
                "SELECT create_feature_log_partition($1::date)",
                row.decision_time.date(),
            )
            return await _execute()


async def read_feature_log_row(
    conn: Any, bank_id: str, decision_id: str
) -> FeatureLogRow | None:
    """
    WHAT: Queries the feature log for a specific bank_id and decision_id.

    Returns:
        FeatureLogRow if found, else None.
    """
    row = await conn.fetchrow(
        """
        SELECT id, bank_id, payment_id, decision_id, decision_time,
               arrival_watermark, status, feature_values,
               feature_values_ordered, computed_at
        FROM bank_feature_log
        WHERE bank_id = $1 AND decision_id = $2
        ORDER BY decision_time DESC
        LIMIT 1
        """,
        bank_id,
        decision_id,
    )
    if row is None:
        return None

    fv = row["feature_values"]
    if isinstance(fv, (str, bytes)):
        feature_values = orjson.loads(fv)
    elif isinstance(fv, dict):
        feature_values = fv
    else:
        raise ValueError(f"Unexpected feature_values type: {type(fv)}")

    fvo = row["feature_values_ordered"]
    if fvo is not None:
        ordered_list: list[float | None] | None = [
            float(x) if x is not None else None for x in fvo
        ]
    else:
        ordered_list = None

    return FeatureLogRow(
        id=int(row["id"]),
        bank_id=row["bank_id"],
        payment_id=row["payment_id"],
        decision_id=row["decision_id"],
        decision_time=row["decision_time"],
        arrival_watermark=row["arrival_watermark"],
        status=row["status"],
        feature_values=feature_values,
        feature_values_ordered=ordered_list,
        computed_at=row["computed_at"],
    )
