# File: features/velocity.py
# Purpose: Redis-backed sliding-window transaction velocity features and per-account state updates.
from datetime import datetime, timezone

import redis

from features.entropy import parse_device_id

WINDOW_1H_SECONDS = 3600
WINDOW_6H_SECONDS = 21600
WINDOW_24H_SECONDS = 86400
ALL_WINDOWS = [WINDOW_1H_SECONDS, WINDOW_6H_SECONDS, WINDOW_24H_SECONDS]

VELOCITY_TTL_SECONDS = 86400
MERCHANT_HISTORY_MAX = 100
MERCHANT_HISTORY_TTL_SECONDS = 86400
DEVICE_HISTORY_MAX = 500
DEVICE_HISTORY_TTL_SECONDS = 604800


def to_unix_seconds(timestamp: datetime) -> float:
    """Convert a datetime to Unix seconds, treating naive datetimes as UTC (the endpoint uses utcnow())."""
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.timestamp()


def velocity_key(account_id: str, window_seconds: int) -> str:
    """Build the sorted-set key for an account's velocity window (schema: vel:{account_id}:{window_seconds})."""
    return f"vel:{account_id}:{window_seconds}"


def record_transaction(client: redis.Redis, account_id: str, txn_id: str, timestamp: datetime) -> None:
    """Add a transaction to each velocity window's sorted set and trim expired entries."""
    now_unix = to_unix_seconds(timestamp)
    for window_seconds in ALL_WINDOWS:
        key = velocity_key(account_id, window_seconds)
        # A pipeline sends all three commands in one network round-trip instead of three.
        pipe = client.pipeline()
        pipe.zadd(key, {txn_id: now_unix})
        # Drop anything older than the window so the set never grows without bound.
        pipe.zremrangebyscore(key, "-inf", now_unix - window_seconds)
        pipe.expire(key, VELOCITY_TTL_SECONDS)
        pipe.execute()


def compute_velocity(client: redis.Redis, account_id: str, window_seconds: int, now: datetime) -> int:
    """Count the account's transactions within [now - window_seconds, now]."""
    now_unix = to_unix_seconds(now)
    key = velocity_key(account_id, window_seconds)
    return int(client.zcount(key, now_unix - window_seconds, now_unix))


def compute_all_velocities(client: redis.Redis, account_id: str, now: datetime) -> dict[str, int]:
    """Return the 1h, 6h and 24h transaction counts for the account."""
    return {
        "velocity_1h": compute_velocity(client, account_id, WINDOW_1H_SECONDS, now),
        "velocity_6h": compute_velocity(client, account_id, WINDOW_6H_SECONDS, now),
        "velocity_24h": compute_velocity(client, account_id, WINDOW_24H_SECONDS, now),
    }


def update_account_state(
    client: redis.Redis,
    account_id: str,
    txn_id: str,
    geo_coordinates: str,
    merchant_id: str,
    device_fingerprint: str,
    timestamp: datetime,
) -> None:
    """Update every Redis key in the cache schema for this account after a transaction."""
    iso_timestamp = timestamp.isoformat()

    # First-seen time: written only once (NX = "only if not exists") so account age is measured from the first txn.
    client.set(f"acct:first_seen:{account_id}", iso_timestamp, nx=True)

    # Last location: always overwritten so the next transaction can compute geo-velocity from here.
    client.set(f"acct:last_loc:{account_id}", geo_coordinates)

    # Last transaction time: always overwritten so the next transaction can compute time-since-last.
    client.set(f"acct:last_txn_time:{account_id}", iso_timestamp)

    # Velocity windows: record this transaction in the 1h/6h/24h sorted sets.
    record_transaction(client, account_id, txn_id, timestamp)

    # Merchant history: newest first, capped at 100 entries, used later for merchant entropy.
    merchant_key = f"merch:history:{account_id}"
    client.lpush(merchant_key, merchant_id)
    client.ltrim(merchant_key, 0, MERCHANT_HISTORY_MAX - 1)
    client.expire(merchant_key, MERCHANT_HISTORY_TTL_SECONDS)

    # Device history: only the dev_id part of the fingerprint is stored, capped at 500, used for device ratio.
    device_key = f"dev:history:{account_id}"
    client.lpush(device_key, parse_device_id(device_fingerprint))
    client.ltrim(device_key, 0, DEVICE_HISTORY_MAX - 1)
    client.expire(device_key, DEVICE_HISTORY_TTL_SECONDS)
