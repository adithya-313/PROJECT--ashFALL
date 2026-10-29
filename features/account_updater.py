# File: features/account_updater.py
# Purpose: Level 15a atomic, idempotent account state updating and trimming in Redis.
# Issue: Level 15a account feature updater (plan Task 1, C4, K2).
"""
WHAT: Provides atomic and idempotent update functions (apply_payment_side and apply_account_event)
for commercial bank transaction sides and account lifecycle events in Redis.

WHY: High-throughput fraud detection requires consistent sliding-window metrics per customer account.
All writes for a record execute in a single MULTI/EXEC Redis pipeline, guaranteeing that a failure
never leaves partial state. Trimming older records is safe under single-writer-per-account partitioning
by bank_id:account_id from Kafka.

IDEMPOTENCY PROOF:
Every operation performed on Redis by this module is inherently idempotent:
  1. ZADD is a set-insert with score replacement: inserting the same member with the same
     event timestamp multiple times preserves identical set membership and score.
  2. HSET is key-value replacement: storing the serialized JSON record at the same field key
     ('payment_id:side' or event_id) is idempotent.
  3. Watermarks in the marks ZSET use conditional modifiers: 'first_event' uses LT (minimum),
     'last_event' uses GT (maximum), and 'arrival_watermark' uses GT (maximum). Re-applying
     the same or earlier records leaves watermark scores strictly unchanged.
  4. Trimming deletes keys strictly older than (last_event - (retention + allowed_lateness))
     atomically via registered Lua script.
Therefore, duplicate delivery of payment sides or account events causes zero state divergence.
Kafka gives no per-account ordering here; state is correct in any order because every write is
idempotent (see the shuffle test).
"""

import logging
from typing import Any

from features.account_keys import (
    AccountFeaturesSettings,
    event_records_key,
    event_times_key,
    get_default_settings,
    marks_key,
    side_records_key,
    side_times_key,
)
from features.account_records import (
    AccountEventRecord,
    PaymentSideRecord,
    record_member,
    serialize_record,
)
from features.velocity import to_unix_seconds

logger = logging.getLogger(__name__)

# Atomic trim script: fetches expired members from ZSET, deletes from HASH, prunes from ZSET.
TRIM_LUA_SCRIPT: str = """local expired = redis.call('ZRANGEBYSCORE', KEYS[1], '-inf', ARGV[1])
if #expired > 0 then
    redis.call('HDEL', KEYS[2], unpack(expired))
    redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', ARGV[1])
end
return #expired"""


def _trim_expired(
    client: Any,
    times_key: str,
    records_key: str,
    cutoff_arg: str,
) -> int:
    """
    PURPOSE:
        Atomically trims expired member records from both the sorted set and hash storage in Redis.

    INPUTS:
        client: Redis or FakeRedis client instance.
        times_key: Sorted set key (KEYS[1]) containing event members scored by timestamp.
        records_key: Hash key (KEYS[2]) storing serialized record payloads by member name.
        cutoff_arg: Upper score bound string for ZRANGEBYSCORE (e.g. exclusive bound '(<cutoff>').

    WHAT THE LUA SCRIPT DOES LINE BY LINE:
        Line 1 (local expired = redis.call('ZRANGEBYSCORE', KEYS[1], '-inf', ARGV[1])):
            Queries all member names from the sorted set scored up to the cutoff threshold.
        Line 2 (if #expired > 0 then):
            Checks whether any expired members were found.
        Line 3 (redis.call('HDEL', KEYS[2], unpack(expired))):
            Deletes all expired member fields from the record hash storage.
        Line 4 (redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', ARGV[1])):
            Removes all expired members from the sorted set.
        Line 5 (end):
            Concludes the conditional branch.
        Line 6 (return #expired):
            Returns the count of pruned records.

    RETURN VALUE:
        int: The number of expired records pruned from both keys.

    ERRORS:
        Any Redis error propagates directly to the caller without suppression or fallback.

    CONSTRAINT:
        One call is atomic on the Redis server, guaranteeing no orphan entries remain in the hash.
    """
    # redis-py registered scripts already use EVALSHA with an automatic reload, so no caching is needed.
    script = client.register_script(TRIM_LUA_SCRIPT)
    count = script(keys=[times_key, records_key], args=[cutoff_arg], client=client)
    return int(count) if count is not None else 0


def apply_payment_side(
    client: Any,
    record: PaymentSideRecord,
    settings: AccountFeaturesSettings | None = None,
) -> str:
    """
    WHAT: Atomically writes a PaymentSideRecord to Redis and prunes records older than retention.

    WHY: Updates the account's transaction time-series and watermark marks in a single atomic
    pipeline. If the record's timestamp falls behind the retention horizon (last_event - retention),
    it is written and trimmed in the same call and reported as 'older_than_retention'.

    Args:
        client: Redis or FakeRedis client instance.
        record: Validated PaymentSideRecord to apply.
        settings: Optional AccountFeaturesSettings; defaults to repository configuration.

    Returns:
        'stored' if the record remains within the retention horizon, or
        'older_than_retention' if the record is older than (last_event - retention).
    """
    cfg = settings if settings is not None else get_default_settings()

    st_key = side_times_key(record.run_id, record.bank_id, record.account_id)
    sr_key = side_records_key(record.run_id, record.bank_id, record.account_id)
    m_key = marks_key(record.run_id, record.bank_id, record.account_id)

    member = record_member(record)
    event_unix = to_unix_seconds(record.event_time)
    arrival_unix = to_unix_seconds(record.arrival_time)
    payload_bytes = serialize_record(record)
    ttl_seconds = cfg.expiry_seconds

    # Atomic write pipeline (MULTI/EXEC)
    pipeline = client.pipeline(transaction=True)
    pipeline.zadd(st_key, {member: event_unix})
    pipeline.hset(sr_key, member, payload_bytes)
    pipeline.zadd(m_key, {"first_event": event_unix}, lt=True)
    pipeline.zadd(m_key, {"last_event": event_unix}, gt=True)
    pipeline.zadd(m_key, {"arrival_watermark": arrival_unix}, gt=True)
    pipeline.expire(st_key, ttl_seconds)
    pipeline.expire(sr_key, ttl_seconds)
    pipeline.expire(m_key, ttl_seconds)
    pipeline.execute()

    # Trim records older than (last_event - (retention + allowed_lateness))
    last_event_unix = client.zscore(m_key, "last_event")
    if last_event_unix is not None:
        cutoff = float(last_event_unix) - (cfg.retention_seconds + cfg.allowed_lateness_seconds)
        _trim_expired(client, st_key, sr_key, f"({cutoff}")

        # Check if this incoming record is older than retention horizon
        if event_unix < cutoff:
            logger.info(
                "Payment side %s for account %s is older than retention horizon (event=%s, cutoff=%s)",
                record.payment_id,
                record.account_id,
                event_unix,
                cutoff,
            )
            return "older_than_retention"

    return "stored"


def apply_account_event(
    client: Any,
    record: AccountEventRecord,
    settings: AccountFeaturesSettings | None = None,
) -> str:
    """
    WHAT: Atomically writes an AccountEventRecord to Redis and prunes events older than retention.

    WHY: Updates the account's lifecycle event history and watermark marks in a single atomic
    pipeline, ensuring lifecycle events (e.g. SIM swaps, device changes) are available for
    mule scoring alongside payment sides.

    Args:
        client: Redis or FakeRedis client instance.
        record: Validated AccountEventRecord to apply.
        settings: Optional AccountFeaturesSettings; defaults to repository configuration.

    Returns:
        'stored' if the record remains within the retention horizon, or
        'older_than_retention' if the record is older than (last_event - retention).
    """
    cfg = settings if settings is not None else get_default_settings()

    et_key = event_times_key(record.run_id, record.bank_id, record.account_id)
    er_key = event_records_key(record.run_id, record.bank_id, record.account_id)
    m_key = marks_key(record.run_id, record.bank_id, record.account_id)

    member = record_member(record)
    event_unix = to_unix_seconds(record.event_time)
    arrival_unix = to_unix_seconds(record.arrival_time)
    payload_bytes = serialize_record(record)
    ttl_seconds = cfg.expiry_seconds

    # Atomic write pipeline (MULTI/EXEC)
    pipeline = client.pipeline(transaction=True)
    pipeline.zadd(et_key, {member: event_unix})
    pipeline.hset(er_key, member, payload_bytes)
    pipeline.zadd(m_key, {"first_event": event_unix}, lt=True)
    pipeline.zadd(m_key, {"last_event": event_unix}, gt=True)
    pipeline.zadd(m_key, {"arrival_watermark": arrival_unix}, gt=True)
    pipeline.expire(et_key, ttl_seconds)
    pipeline.expire(er_key, ttl_seconds)
    pipeline.expire(m_key, ttl_seconds)
    pipeline.execute()

    # Trim events older than (last_event - (retention + allowed_lateness))
    last_event_unix = client.zscore(m_key, "last_event")
    if last_event_unix is not None:
        cutoff = float(last_event_unix) - (cfg.retention_seconds + cfg.allowed_lateness_seconds)
        _trim_expired(client, et_key, er_key, f"({cutoff}")

        if event_unix < cutoff:
            logger.info(
                "Account event %s for account %s is older than retention horizon (event=%s, cutoff=%s)",
                record.event_id,
                record.account_id,
                event_unix,
                cutoff,
            )
            return "older_than_retention"

    return "stored"
