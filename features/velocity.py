# File: features/velocity.py
# Purpose: Redis-backed sliding-window transaction velocity features and per-account state updates.
#
# What "velocity" means: how many transactions an account made in the last hour, 6 hours and 24 hours. A sudden
# spike in that count is a strong indicator of account compromise or automated fraud: a stolen card being drained,
# a script testing card numbers, or a money-mule account receiving many small payments before one large payout.
# Normal customers transact at a fairly steady, low rate, so unusual counts stand out.
#
# Why Redis: the counts must be available in milliseconds while a payment is being scored, and they must be shared
# across all API workers. An in-process dictionary would be lost on restart and would differ between workers.
from datetime import datetime, timezone

import redis

from features.entropy import parse_device_id

# The three look-back windows, in seconds.
#   1 hour  : catches rapid bursts (card testing, scripted attacks).
#   6 hours : catches a sustained session of suspicious activity, such as an account takeover in progress.
#   24 hours: catches slower patterns such as a mule account collecting many small deposits over a day.
WINDOW_1H_SECONDS = 3600
WINDOW_6H_SECONDS = 21600
WINDOW_24H_SECONDS = 86400
ALL_WINDOWS = [WINDOW_1H_SECONDS, WINDOW_6H_SECONDS, WINDOW_24H_SECONDS]

# Velocity keys expire after 24h because the largest window is 24h; anything older can never be counted again.
VELOCITY_TTL_SECONDS = 86400

# Rolling history sizes. We only need recent behaviour to judge "variety", not an account's entire lifetime, so the
# lists are capped. This bounds Redis memory per account and keeps the entropy / device-ratio calculations fast.
MERCHANT_HISTORY_MAX = 100
MERCHANT_HISTORY_TTL_SECONDS = 86400
DEVICE_HISTORY_MAX = 500
DEVICE_HISTORY_TTL_SECONDS = 604800


def to_unix_seconds(timestamp: datetime) -> float:
    """
    Convert a datetime into Unix seconds (seconds since 1970-01-01 UTC).

    Why this function exists: Redis sorted sets order members by a numeric score, so we store each transaction
    time as a plain number. Sorting and range-querying numbers is what makes time-window counts fast.

    Args:
        timestamp: a datetime. Per the DATETIME CONTRACT in features/__init__.py it is normally naive UTC (from
            datetime.utcnow() at the scoring endpoint).

    Returns:
        Seconds since the Unix epoch as a float (fractions of a second are kept).

    Edge cases:
        Naive datetimes (no timezone attached) are treated as UTC. Python's own .timestamp() would otherwise
        assume the machine's LOCAL time zone, which would silently shift every score by the server's UTC offset and
        break the windows. Timezone-aware datetimes are converted correctly as-is.
    """
    # A naive datetime has no timezone, so label it as UTC explicitly. Without this, .timestamp() would guess the
    # local time zone of whatever machine runs the code.
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.timestamp()


def velocity_key(account_id: str, window_seconds: int) -> str:
    """
    Build the Redis key name for one account's velocity window.

    Why this function exists: the key format is a locked contract (data/schemas/feature_cache_schema.yaml) shared
    with the Level 10.5 agent tools. Building it in one place means the writer and the reader can never disagree.

    Args:
        account_id: the account the transactions belong to.
        window_seconds: the window length in seconds, one of ALL_WINDOWS.

    Returns:
        A key of the form "vel:{account_id}:{window_seconds}", for example "vel:acct_1:3600".

    Edge cases:
        None. No validation is done; callers pass known window constants.
    """
    return f"vel:{account_id}:{window_seconds}"


def record_transaction(client: redis.Redis, account_id: str, txn_id: str, timestamp: datetime) -> None:
    """
    Record one transaction in each of the account's velocity windows and trim expired entries.

    Why this function exists: velocity can only be computed if every transaction is remembered somewhere. This is
    the write side; compute_velocity is the read side. We store the transactions in a Redis SORTED SET rather than a
    list or a simple counter:
      - A plain counter cannot "forget" old transactions, so it cannot represent a sliding window.
      - A list has no efficient way to ask "how many entries are newer than time X?".
      - A sorted set uses a numeric score per member. We use the transaction time as the score, so Redis keeps
        members ordered by time and can count or delete a time range efficiently (ZCOUNT / ZREMRANGEBYSCORE).

    Args:
        client: an injected Redis connection (this function never creates its own, which keeps it testable).
        account_id: the account that made the transaction.
        txn_id: the unique transaction ID. It is the sorted-set member, so recording the same txn_id twice does not
            double count it.
        timestamp: when the transaction happened (naive UTC). This is also what old entries are measured against;
            the wall clock is never read inside this function.

    Returns:
        None. The effect is the change in Redis.

    Edge cases:
        - Re-recording the same txn_id for the same account just updates its score instead of adding a duplicate.
        - Each key's TTL is refreshed on every write, so an active account's windows never expire mid-use.
    """
    now_unix = to_unix_seconds(timestamp)

    for window_seconds in ALL_WINDOWS:
        key = velocity_key(account_id, window_seconds)

        # Use a pipeline: queue the three commands below and send them to Redis together. Each separate command
        # would otherwise cost one network round-trip (the request goes out and we wait for the reply). Over a
        # network, round-trips dominate latency, so one batched trip instead of three keeps scoring fast.
        pipe = client.pipeline()

        # Add this transaction: member = txn_id, score = its Unix time. The score is what lets Redis answer
        # "how many transactions fall between time A and time B" without scanning everything.
        pipe.zadd(key, {txn_id: now_unix})

        # Remove any transactions that are older than the window boundary.
        # We do this on every write so the sorted set never grows unbounded: Redis would otherwise keep every
        # transaction forever, wasting memory. After this, only transactions within
        # [now - window_seconds, now] remain. The boundary uses the transaction's own timestamp, not the wall
        # clock, so replaying historical data gives the same answer as live traffic.
        pipe.zremrangebyscore(key, "-inf", now_unix - window_seconds)

        # Reset the key's expiry. If an account goes quiet for a full day, Redis deletes the key by itself, so
        # inactive accounts do not use memory forever.
        pipe.expire(key, VELOCITY_TTL_SECONDS)

        # Send all three queued commands to Redis in one round-trip.
        pipe.execute()


def compute_velocity(client: redis.Redis, account_id: str, window_seconds: int, now: datetime) -> int:
    """
    Count how many transactions the account made inside one look-back window ending at "now".

    Why this function exists: this is the velocity fraud signal itself. A jump in this count (for example 12
    transactions in 24 hours for an account that normally makes one) suggests account compromise, scripted card
    testing or a money mule, and the model uses it to raise the risk score.

    Args:
        client: an injected Redis connection.
        account_id: the account to count for.
        window_seconds: the window length in seconds (normally one of ALL_WINDOWS).
        now: the end of the window, usually the arrival time of the transaction being scored.

    Returns:
        An integer count of transactions with a time in [now - window_seconds, now]. Both ends are included.

    Edge cases:
        - An account with no recorded transactions returns 0 (the key does not exist, and ZCOUNT of a missing key
          is 0).
        - Transactions recorded with a time after "now" are not counted.
    """
    now_unix = to_unix_seconds(now)
    key = velocity_key(account_id, window_seconds)

    # Ask Redis to count members whose score (transaction time) lies between the start and end of the window.
    # Redis does this directly on its sorted index, so we never download the transactions just to count them.
    return int(client.zcount(key, now_unix - window_seconds, now_unix))


def compute_all_velocities(client: redis.Redis, account_id: str, now: datetime) -> dict[str, int]:
    """
    Return the 1-hour, 6-hour and 24-hour transaction counts for the account in one dictionary.

    Why this function exists: the model wants all three windows together. Comparing them is informative: many
    transactions in the last hour but few in 24h means a sudden burst, while an even spread is more normal.

    Args:
        client: an injected Redis connection.
        account_id: the account to count for.
        now: the end of every window, usually the arrival time of the transaction being scored.

    Returns:
        A dict with exactly the keys "velocity_1h", "velocity_6h" and "velocity_24h", each an integer count.

    Edge cases:
        Unknown accounts return 0 for all three. The windows are nested, so velocity_1h <= velocity_6h <=
        velocity_24h always holds.
    """
    # Three simple reads. A pipeline is not used here because reads are cheap and the code is easier to follow;
    # it can be added later if profiling shows the extra round-trips matter.
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
    """
    Update every piece of per-account state in Redis after a transaction has been scored.

    Why this function exists: there is no Account database table, so Redis is where we remember what each account
    did last. The NEXT transaction reads this state to compute its features (account age, time since last
    transaction, geo-velocity, merchant entropy, device ratio, velocity). The keys written here are exactly those in
    data/schemas/feature_cache_schema.yaml.

    Args:
        client: an injected Redis connection.
        account_id: the account that made the transaction.
        txn_id: the unique transaction ID, stored in the velocity windows.
        geo_coordinates: where the transaction happened, as a "lat,lon" string.
        merchant_id: the merchant. Used as a stand-in for merchant category, which the schema does not have.
        device_fingerprint: the raw string "dev_id=...|os=...|browser=...". Only the dev_id part is stored.
        timestamp: when the transaction happened (naive UTC).

    Returns:
        None. The effect is the change in Redis.

    Edge cases:
        - A device_fingerprint with no "dev_id=" segment makes parse_device_id raise ValueError. By then the other
          keys have already been written, so callers should validate the fingerprint before calling.
        - The first-seen key is never overwritten, even if a later call carries an earlier timestamp.
    """
    # Store the time as ISO 8601 text (for example "2024-01-15T10:00:00"). It is human readable and easy to parse
    # back into a datetime when the next transaction needs it.
    iso_timestamp = timestamp.isoformat()

    # Remember when we FIRST saw this account. SET with nx=True means "only set if the key does not already exist".
    # We want the very first time we ever saw the account, so NX stops every later transaction from overwriting it.
    # This value feeds the account-age feature; a moving "first seen" would make every account look brand new.
    client.set(f"acct:first_seen:{account_id}", iso_timestamp, nx=True)

    # Remember the location of the latest transaction. It is ALWAYS overwritten, because the next transaction needs
    # the most recent location to work out how far the user appears to have travelled (geo-velocity).
    client.set(f"acct:last_loc:{account_id}", geo_coordinates)

    # Remember the time of the latest transaction, always overwritten. The next transaction uses it to calculate
    # the gap since the last one, and as the elapsed time when computing geo-velocity.
    client.set(f"acct:last_txn_time:{account_id}", iso_timestamp)

    # Add this transaction to the 1h / 6h / 24h sorted sets so the velocity counts include it from now on.
    record_transaction(client, account_id, txn_id, timestamp)

    # Keep a rolling list of recent merchants, newest first (LPUSH puts the new value at the head). The list is
    # trimmed to the latest 100 entries (LTRIM keeps positions 0..99) because we only need recent behaviour to
    # measure how varied the merchants are, not the whole lifetime. The cap bounds memory per account. The 24h
    # expiry means a dormant account's list disappears on its own.
    merchant_key = f"merch:history:{account_id}"
    client.lpush(merchant_key, merchant_id)
    client.ltrim(merchant_key, 0, MERCHANT_HISTORY_MAX - 1)
    client.expire(merchant_key, MERCHANT_HISTORY_TTL_SECONDS)

    # Keep a rolling list of recent device IDs, newest first. Only the dev_id part of the fingerprint is stored
    # (OS and browser change on updates and would make the same device look like many). The list is capped at 500
    # entries and kept for 7 days: devices change slowly, so a longer, larger window is needed to tell "same
    # device as always" apart from "many different devices", yet we still do not need the full lifetime.
    device_key = f"dev:history:{account_id}"
    client.lpush(device_key, parse_device_id(device_fingerprint))
    client.ltrim(device_key, 0, DEVICE_HISTORY_MAX - 1)
    client.expire(device_key, DEVICE_HISTORY_TTL_SECONDS)
