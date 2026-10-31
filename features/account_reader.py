# File: features/account_reader.py
# Purpose: Level 15a point-in-time account feature reading, snapshot extraction, and window counts.
# Issue: Level 15a account feature updater (plan Task 1, C4, K2).
"""
WHAT: Provides point-in-time inspection of customer account transaction sides and lifecycle events
from Redis, producing immutable AccountSnapshot objects filtered against decision timestamps and
arrival watermarks.

WHY: Prevents future-leakage in feature engineering (Section 22 ground rule). Only events whose
event_time is within the sliding look-back window (decision_time - retention <= event_time <= decision_time)
AND whose ingestion arrival_time <= arrival_cutoff are admitted into the feature snapshot.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from features.account_keys import (
    AccountFeaturesSettings,
    event_records_key,
    event_times_key,
    get_default_settings,
    side_records_key,
    side_times_key,
    withdrawal_records_key,
    withdrawal_times_key,
)
from features.account_records import (
    AccountEventRecord,
    PaymentSideRecord,
    WithdrawalRecord,
    deserialize_account_event_record,
    deserialize_payment_side_record,
    deserialize_withdrawal_record,
)
from features.velocity import to_unix_seconds


@dataclass(frozen=True)
class AccountSnapshot:
    """
    Point-in-time snapshot of an account's transaction sides and lifecycle events.

    Attributes:
        has_history: True if at least one visible event exists within the decision horizon.
        side_records: Tuple of PaymentSideRecord instances ordered chronologically (oldest first).
        account_events: Tuple of AccountEventRecord instances ordered chronologically (oldest first).
        first_event_time: Earliest event timestamp observed in the visible snapshot.
        last_event_time: Latest event timestamp observed in the visible snapshot.
        arrival_watermark: Latest arrival/storage timestamp observed in the visible snapshot.
    """

    has_history: bool
    side_records: tuple[PaymentSideRecord, ...]
    account_events: tuple[AccountEventRecord, ...]
    withdrawals: tuple[WithdrawalRecord, ...]
    first_event_time: datetime | None
    last_event_time: datetime | None
    arrival_watermark: datetime | None


def empty_snapshot() -> AccountSnapshot:
    """
    WHAT: Returns the canonical default snapshot for an account with no visible history.
    WHY: Unknown or quiet accounts return defined defaults rather than raising errors (plan Task 2).
    """
    return AccountSnapshot(
        has_history=False,
        side_records=(),
        account_events=(),
        withdrawals=(),
        first_event_time=None,
        last_event_time=None,
        arrival_watermark=None,
    )


def _decode_members(scored_items: list[Any]) -> list[str]:
    """Decode the members of a zrangebyscore(..., withscores=True) result to strings."""
    return [m[0].decode("utf-8") if isinstance(m[0], bytes) else str(m[0]) for m in scored_items]


def _visible_side_records(
    raw_records: list[Any],
    decision_time: datetime,
    arrival_cutoff: datetime,
    min_allowed_time: datetime,
) -> list[PaymentSideRecord]:
    """
    WHAT: Deserialises the stored side records and keeps those inside the look-back window that had arrived
    by the arrival cutoff, in the order given.

    WHY: The one copy of the side-record visibility rule, shared by the single-account snapshot read and the
    many-account read so the two can never disagree.
    """
    visible: list[PaymentSideRecord] = []
    for raw in raw_records:
        if raw is None:
            continue
        rec = deserialize_payment_side_record(raw)
        if min_allowed_time <= rec.event_time <= decision_time and rec.arrival_time <= arrival_cutoff:
            visible.append(rec)
    return visible


def read_side_records_many(
    client: Any,
    run_id: str,
    bank_id: str,
    account_ids: Sequence[str],
    decision_time: datetime,
    arrival_cutoff: datetime,
    settings: AccountFeaturesSettings | None = None,
) -> dict[str, tuple[PaymentSideRecord, ...]]:
    """
    WHAT: Returns, for each account, exactly read_account_snapshot(...).side_records, using two Redis round
    trips in total (one pipeline of zrangebyscore calls, one pipeline of hmget calls) however many accounts there are.

    WHY: The network features read the side records of many accounts per decision; one round trip per account
    dominated their latency. Time window, arrival filter and order are identical to the single-account read.
    """
    cfg = settings if settings is not None else get_default_settings()
    unique_accounts = list(dict.fromkeys(account_ids))
    if not unique_accounts:
        return {}

    decision_unix = to_unix_seconds(decision_time)
    min_unix = decision_unix - cfg.retention_seconds
    min_allowed_time = decision_time - timedelta(seconds=cfg.retention_seconds)

    times_pipe = client.pipeline(transaction=False)
    for account_id in unique_accounts:
        times_pipe.zrangebyscore(side_times_key(run_id, bank_id, account_id), min_unix, decision_unix, withscores=True)
    scored_by_account = dict(zip(unique_accounts, times_pipe.execute()))

    records_by_account: dict[str, tuple[PaymentSideRecord, ...]] = {account_id: () for account_id in unique_accounts}
    with_items = [account_id for account_id in unique_accounts if scored_by_account[account_id]]
    if not with_items:
        return records_by_account

    records_pipe = client.pipeline(transaction=False)
    for account_id in with_items:
        records_pipe.hmget(
            side_records_key(run_id, bank_id, account_id), _decode_members(scored_by_account[account_id])
        )
    for account_id, raw_records in zip(with_items, records_pipe.execute()):
        records_by_account[account_id] = tuple(
            _visible_side_records(raw_records, decision_time, arrival_cutoff, min_allowed_time)
        )
    return records_by_account


def read_account_snapshot(
    client: Any,
    run_id: str,
    bank_id: str,
    account_id: str,
    decision_time: datetime,
    arrival_cutoff: datetime,
    settings: AccountFeaturesSettings | None = None,
) -> AccountSnapshot:
    """
    WHAT: Queries Redis to build an AccountSnapshot filtered against decision time and arrival cutoff.

    WHY: Enforces the Section 22 future-leak rule. A transaction or event that occurred before
    decision_time is excluded if its storage timestamp (arrival_time) is later than arrival_cutoff.
    Similarly, any record older than (decision_time - retention) is excluded from the snapshot.

    Args:
        client: Redis or FakeRedis client instance.
        run_id: Simulation run identifier.
        bank_id: Tenant commercial bank identifier.
        account_id: Customer account identifier.
        decision_time: Upper boundary for event_time (scoring moment).
        arrival_cutoff: Upper boundary for arrival_time (watermark cutoff).
        settings: Optional AccountFeaturesSettings; defaults to repository configuration.

    Returns:
        Immutable AccountSnapshot instance (or empty_snapshot if no visible events match).
    """
    cfg = settings if settings is not None else get_default_settings()
    retention_seconds = cfg.retention_seconds

    decision_unix = to_unix_seconds(decision_time)
    min_unix = decision_unix - retention_seconds
    max_unix = decision_unix
    min_allowed_time = decision_time - timedelta(seconds=retention_seconds)

    # 1. Query transaction sides within [decision_time - retention, decision_time]
    st_key = side_times_key(run_id, bank_id, account_id)
    sr_key = side_records_key(run_id, bank_id, account_id)

    side_items = client.zrangebyscore(st_key, min_unix, max_unix, withscores=True)
    filtered_sides: list[PaymentSideRecord] = []
    if side_items:
        raw_sides = client.hmget(sr_key, _decode_members(side_items))
        filtered_sides = _visible_side_records(raw_sides, decision_time, arrival_cutoff, min_allowed_time)

    # 2. Query account lifecycle events within [decision_time - retention, decision_time]
    et_key = event_times_key(run_id, bank_id, account_id)
    er_key = event_records_key(run_id, bank_id, account_id)

    ev_items = client.zrangebyscore(et_key, min_unix, max_unix, withscores=True)
    filtered_events: list[AccountEventRecord] = []
    if ev_items:
        ev_members = [
            m[0].decode("utf-8") if isinstance(m[0], bytes) else str(m[0])
            for m in ev_items
        ]
        raw_evs = client.hmget(er_key, ev_members)
        for raw in raw_evs:
            if raw is None:
                continue
            ev = deserialize_account_event_record(raw)
            if min_allowed_time <= ev.event_time <= decision_time and ev.arrival_time <= arrival_cutoff:
                filtered_events.append(ev)

    # 3. Query cash withdrawals within [decision_time - retention, decision_time]
    wt_key = withdrawal_times_key(run_id, bank_id, account_id)
    wr_key = withdrawal_records_key(run_id, bank_id, account_id)

    w_items = client.zrangebyscore(wt_key, min_unix, max_unix, withscores=True)
    filtered_withdrawals: list[WithdrawalRecord] = []
    if w_items:
        w_members = [
            m[0].decode("utf-8") if isinstance(m[0], bytes) else str(m[0])
            for m in w_items
        ]
        raw_ws = client.hmget(wr_key, w_members)
        for raw in raw_ws:
            if raw is None:
                continue
            w_rec = deserialize_withdrawal_record(raw)
            if min_allowed_time <= w_rec.event_time <= decision_time and w_rec.arrival_time <= arrival_cutoff:
                filtered_withdrawals.append(w_rec)

    all_visible = list(filtered_sides) + list(filtered_events) + list(filtered_withdrawals)
    if not all_visible:
        return empty_snapshot()

    first_event_time = min(r.event_time for r in all_visible)
    last_event_time = max(r.event_time for r in all_visible)
    arrival_watermark = max(r.arrival_time for r in all_visible)

    return AccountSnapshot(
        has_history=True,
        side_records=tuple(filtered_sides),
        account_events=tuple(filtered_events),
        withdrawals=tuple(filtered_withdrawals),
        first_event_time=first_event_time,
        last_event_time=last_event_time,
        arrival_watermark=arrival_watermark,
    )


def count_sides_in_window(
    snapshot: AccountSnapshot,
    side: str,
    decision_time: datetime,
    window_seconds: int,
    settings: AccountFeaturesSettings | None = None,
) -> int:
    """
    WHAT: Counts transaction sides within the look-back window (decision_time - window_seconds, decision_time].

    WHY: Evaluates velocity counts for incoming payment scoring. Enforces strict boundary rules:
    events exactly at decision_time are included; events exactly at decision_time - window_seconds
    are excluded (half-open interval).

    Args:
        snapshot: AccountSnapshot containing visible PaymentSideRecords.
        side: Transaction side to count ('debit' or 'credit').
        decision_time: Upper boundary timestamp.
        window_seconds: Look-back duration in seconds. Must be in settings.windows_seconds.
        settings: Optional AccountFeaturesSettings; defaults to repository configuration.

    Returns:
        Integer count of matching transaction sides.

    Raises:
        ValueError: If window_seconds is not in configured windows_seconds.
    """
    cfg = settings if settings is not None else get_default_settings()
    if window_seconds not in cfg.windows_seconds:
        raise ValueError(
            f"Invalid window_seconds {window_seconds}; must be one of {cfg.windows_seconds}"
        )

    window_start = decision_time - timedelta(seconds=window_seconds)
    count = 0
    for r in snapshot.side_records:
        if r.side == side and window_start < r.event_time <= decision_time:
            count += 1
    return count
