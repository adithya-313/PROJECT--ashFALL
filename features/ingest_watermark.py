# File: features/ingest_watermark.py
# Purpose: Level 15c per-source ingestion watermark tracking across Kafka and Postgres pollers.
# Issue: Level 15c watermark and feature log (plan Task 1, D1, S1, S2).
"""
WHAT: Provides recording and querying of per-source ingestion watermarks for Level 15 account features.
Watermarks track progress across 5 sources per bank:
  1. "payments": Kafka upi.payments (per-partition watermarks).
  2. "events": Postgres bank_account_events.
  3. "withdrawals": Postgres bank_withdrawals.
  4. "feedback": Postgres bank_feedback.
  5. "complaints": Postgres bank_complaints.

WHY: Enforces Section 22 evaluation correctness: an account feature snapshot or decision is valid
at decision_time T only if all data ingestion sources have processed up to at least T. Watermark
writes are strictly monotonic via an atomic Redis Lua script so late-arriving batches cannot move
watermarks backwards.
"""

from datetime import datetime, timezone
from typing import Any

from features.account_keys import payment_watermark_key, payments_partitions_key, poller_watermark_key
from features.account_summaries import to_unix_seconds

__all__ = [
    "SET_WATERMARK_LUA",
    "POLLER_SOURCES",
    "to_unix_seconds",
    "to_watermark_seconds",
    "record_payment_watermark",
    "record_poller_watermark",
    "get_payment_watermark",
    "get_poller_watermark",
    "get_bank_watermark",
]

SET_WATERMARK_LUA = """
if redis.call('get', KEYS[1]) then
  if tonumber(ARGV[1]) > tonumber(redis.call('get', KEYS[1])) then
    redis.call('set', KEYS[1], ARGV[1])
  end
else
  redis.call('set', KEYS[1], ARGV[1])
end
"""

POLLER_SOURCES: frozenset[str] = frozenset({"events", "withdrawals", "feedback", "complaints"})


def to_watermark_seconds(ts: datetime | int | float) -> int:
    """Convert a UTC or timezone-aware datetime or int/float timestamp to integer Unix seconds."""
    if isinstance(ts, (int, float)):
        return int(ts)
    if isinstance(ts, datetime):
        return int(to_unix_seconds(ts))
    raise ValueError(f"Expected datetime or int timestamp, got {type(ts)}")


def record_payment_watermark(client: Any, partition: int, timestamp: datetime | int | float) -> None:
    """
    WHAT: Monotonically records the watermark timestamp for a Kafka payments partition and registers
    the partition in the known partitions set.

    WHY: Tracks real-time Kafka consumer progress per-partition without moving backwards.
    """
    key = payment_watermark_key(partition)
    ts = to_watermark_seconds(timestamp)
    client.eval(SET_WATERMARK_LUA, 1, key, ts)
    client.sadd(payments_partitions_key(), partition)


def record_poller_watermark(
    client: Any,
    bank_id: str,
    source: str,
    timestamp: datetime | int | float,
) -> None:
    """
    WHAT: Monotonically records the watermark timestamp for a bank-scoped Postgres poller.

    WHY: Tracks table poller progress for lifecycle events, withdrawals, feedback, and complaints.
    """
    if source not in POLLER_SOURCES:
        raise ValueError(f"Unknown poller source: {source!r}. Must be one of {sorted(POLLER_SOURCES)}")
    key = poller_watermark_key(bank_id, source)
    ts = to_watermark_seconds(timestamp)
    client.eval(SET_WATERMARK_LUA, 1, key, ts)


def get_payment_watermark(client: Any, partition: int) -> datetime | None:
    """
    WHAT: Reads the watermark timestamp for a Kafka payments partition.

    Returns:
        Timezone-aware UTC datetime, or None if no watermark recorded.
    """
    key = payment_watermark_key(partition)
    val = client.get(key)
    if val is None:
        return None
    return datetime.fromtimestamp(int(val), tz=timezone.utc)


def get_poller_watermark(client: Any, bank_id: str, source: str) -> datetime | None:
    """
    WHAT: Reads the watermark timestamp for a bank-scoped Postgres poller.

    Returns:
        Timezone-aware UTC datetime, or None if no watermark recorded.
    """
    if source not in POLLER_SOURCES:
        raise ValueError(f"Unknown poller source: {source!r}. Must be one of {sorted(POLLER_SOURCES)}")
    key = poller_watermark_key(bank_id, source)
    val = client.get(key)
    if val is None:
        return None
    return datetime.fromtimestamp(int(val), tz=timezone.utc)


def get_bank_watermark(
    client: Any,
    bank_id: str,
) -> datetime | None:
    """
    WHAT: Computes the aggregate ingestion watermark for bank_id across all sources.

    WHY: Section 22 correctness: an account feature snapshot or decision is valid only if
    all ingestion sources (Kafka payments and 4 Postgres pollers) have progressed to at least T.
    If the known partitions set is empty, or ANY source has no recorded watermark, returns None.
    Otherwise returns MIN across all sources as a timezone-aware UTC datetime.
    """
    raw_partitions = client.smembers(payments_partitions_key())
    if not raw_partitions:
        return None

    partition_ts_list: list[int] = []
    for raw_p in raw_partitions:
        p = int(raw_p.decode("utf-8") if isinstance(raw_p, bytes) else raw_p)
        val = client.get(payment_watermark_key(p))
        if val is None:
            return None
        partition_ts_list.append(int(val))

    payments_ts = min(partition_ts_list)

    poller_ts_list: list[int] = []
    for src in sorted(POLLER_SOURCES):
        val = client.get(poller_watermark_key(bank_id, src))
        if val is None:
            return None
        poller_ts_list.append(int(val))

    all_ts = [payments_ts] + poller_ts_list
    min_ts = min(all_ts)
    return datetime.fromtimestamp(min_ts, tz=timezone.utc)

