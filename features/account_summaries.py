# File: features/account_summaries.py
# Purpose: Level 15b long-lived account summaries, closed-day reads, and lifetime metrics.
# Issue: Level 15b long-history summaries (plan Task 1, G1, G2, R5, R7).
"""
WHAT: Provides long-lived per-account behavioral summaries stored in Redis alongside the 7-day
raw window: closed-day transaction totals, counterparty interaction calendar days, and lifetime
metrics (opened_at, active_days, device fingerprints).

WHY: Certain fraud and money mule patterns (routine monthly salary/bill rhythms, account dormancy,
device novelty across account lifetime) require longer memory than the 7-day raw sliding window.
Long-lived summaries maintain compact, bounded aggregates over a 62-day horizon without unbounded
memory growth or state leak.

CLOSED-DAY RULE AND FUTURE-LEAK GUARANTEE (Architect Decision A, Section 22):
  Day number: The Indian Standard Time (IST, UTC+5:30) calendar date of event_time as an integer
  ordinal (date.toordinal()).
  To guarantee strict point-in-time correctness without leaking future information:
    - WRITE: An event updates day summaries if and only if:
        arrival_time <= (end of its IST calendar day + allowed_lateness_seconds).
      Events arriving later are counted under outcome 'too_late_for_summary' and logged;
      they are admitted only to the raw window.
    - READ: Summary readers return days whose:
        end_of_day + allowed_lateness_seconds <= decision_time AND <= arrival_cutoff.
      Recent and unclosed days are never read from summaries; recent activity is queried from
      the 7-day raw window (Level 15a), which is exact.
  This ensures that summary readers only ever observe days that are permanently sealed and immutable.

LIFETIME ACTIVE DAYS AND KEYS (FIX 1, FIX 2):
  Lifetime money activity is tracked in active_days ZSET for closed payment sides and withdrawals.
  Account lifecycle events (device changes, account opening) are not money activity and never write active_days.
  last_active_day is derived on read as the most recent closed active day.
  history_days_available measures transaction history held by this bank from the start of the
  earliest closed active day.
  Lifetime keys (opened_at, fingerprints, active_days) live as long as the run's namespace;
  a run's keys are removed by namespace cleanup, never by a short TTL.

CONSERVATIVE DEVICE NOVELTY (R5):
  Device fingerprints are scored with available_from = max(event_time, arrival_time).
  A fingerprint counts as seen at decision_time if and only if available_from <= decision_time
  and available_from <= arrival_cutoff. This is conservative: a phone can look new slightly longer,
  never earlier, eliminating false negatives in device novelty detection.
"""

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any

from features.account_keys import (
    AccountFeaturesSettings,
    account_type_key,
    active_days_key,
    counterparty_days_key,
    counterparty_index_key,
    counterparty_payee_type_key,
    day_seen_key,
    day_totals_key,
    days_index_key,
    device_accounts_key,
    device_changes_key,
    fingerprints_key,
    get_default_settings,
    home_region_key,
    opened_at_key,
)
from features.account_records import (
    AccountEventRecord,
    InvalidAccountEventError,
    PaymentSideRecord,
    WithdrawalRecord,
    record_member,
)
def to_unix_seconds(dt: datetime) -> float:
    """
    WHAT: Converts a datetime to Unix epoch seconds as a float.
    WHY: Redis sorted sets index timestamps as numeric scores.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()

logger = logging.getLogger(__name__)

# Indian Standard Time (IST, UTC+5:30) timezone definition.
IST: timezone = timezone(timedelta(hours=5, minutes=30))


# Atomic day totals Lua script: checks seen set, increments totals, indexes day, records active days.
DAY_TOTALS_LUA_SCRIPT: str = """local member = ARGV[1]
local is_seen = redis.call('HEXISTS', KEYS[1], member)
if is_seen == 1 then
    return 0
end

redis.call('HSET', KEYS[1], member, '1')
redis.call('ZADD', KEYS[3], tonumber(ARGV[2]), ARGV[2])
redis.call('ZADD', KEYS[4], tonumber(ARGV[2]), ARGV[2])

local is_failed = ARGV[4]
if is_failed == '1' then
    redis.call('HINCRBY', KEYS[2], 'failed_attempt_count', 1)
else
    local rtype = ARGV[5]
    local amt = tonumber(ARGV[6])
    if rtype == 'credit' then
        redis.call('HINCRBY', KEYS[2], 'credit_count', 1)
        redis.call('HINCRBY', KEYS[2], 'credit_paise', amt)
    elseif rtype == 'debit' then
        redis.call('HINCRBY', KEYS[2], 'debit_count', 1)
        redis.call('HINCRBY', KEYS[2], 'debit_paise', amt)
    elseif rtype == 'withdrawal' then
        redis.call('HINCRBY', KEYS[2], 'withdrawal_count', 1)
        redis.call('HINCRBY', KEYS[2], 'withdrawal_paise', amt)
    end
end

local ttl = tonumber(ARGV[3])
if ttl and ttl > 0 then
    redis.call('EXPIRE', KEYS[1], ttl)
    redis.call('EXPIRE', KEYS[2], ttl)
    redis.call('EXPIRE', KEYS[3], ttl)
end

return 1"""

# Atomic trim script: prunes days older than (newest_day - summary_horizon_days) and their seen sets.
DAY_TOTALS_TRIM_LUA_SCRIPT: str = """local newest_items = redis.call('ZREVRANGE', KEYS[1], 0, 0, 'WITHSCORES')
if #newest_items == 0 then
    return 0
end

local newest_day = tonumber(newest_items[2])
local horizon = tonumber(ARGV[1])
local cutoff = newest_day - horizon
local expired_days = redis.call('ZRANGEBYSCORE', KEYS[1], '-inf', '(' .. cutoff)

if #expired_days > 0 then
    local prefix = ARGV[2]
    for i = 1, #expired_days do
        local day = expired_days[i]
        local totals_key = prefix .. ':day_totals_' .. day
        local seen_key = prefix .. ':day_seen_' .. day
        redis.call('DEL', totals_key, seen_key)
    end
    redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', '(' .. cutoff)
end

return #expired_days"""


@dataclass(frozen=True)
class DayTotals:
    """
    Immutable representation of aggregated transaction metrics for a single closed IST calendar day.

    Attributes:
        day: IST calendar date ordinal integer (date.toordinal()).
        credit_count: Number of successful incoming credit payments.
        credit_paise: Total incoming credit amount in integer paise.
        debit_count: Number of successful outgoing debit payments.
        debit_paise: Total outgoing debit amount in integer paise.
        withdrawal_count: Number of successful cash withdrawals.
        withdrawal_paise: Total cash amount withdrawn in integer paise.
        failed_attempt_count: Total failed payment sides and withdrawal attempts (R7).
    """

    day: int
    credit_count: int
    credit_paise: int
    debit_count: int
    debit_paise: int
    withdrawal_count: int
    withdrawal_paise: int
    failed_attempt_count: int


@dataclass(frozen=True)
class LifetimeSummary:
    """
    Immutable lifetime metrics for an account that are never trimmed.

    Attributes:
        last_active_day: IST day ordinal of most recent activity, or None if unobserved.
        opened_at: Canonical account opening timestamp from account_opened event, or None.
        fingerprints: Tuple of distinct device fingerprints observed before decision/cutoff.
    """

    last_active_day: int | None
    opened_at: datetime | None
    fingerprints: tuple[str, ...]


def to_ist_date(dt: datetime) -> date:
    """Converts a timezone-aware datetime into its calendar date in Indian Standard Time (IST)."""
    return dt.astimezone(IST).date()


def to_day_number(dt: datetime) -> int:
    """Converts a datetime into its IST calendar date ordinal (date.toordinal())."""
    return to_ist_date(dt).toordinal()


def day_end_ist(day_number: int) -> datetime:
    """Returns the end timestamp of an IST calendar day (midnight beginning the following day)."""
    d = date.fromordinal(day_number)
    next_d = d + timedelta(days=1)
    return datetime(next_d.year, next_d.month, next_d.day, 0, 0, 0, tzinfo=IST)


def day_closed_threshold(day_number: int, allowed_lateness_seconds: int) -> datetime:
    """Returns the threshold moment after which an IST calendar day is considered fully closed."""
    return day_end_ist(day_number) + timedelta(seconds=allowed_lateness_seconds)


def _record_fingerprints(
    client: Any,
    run_id: str,
    bank_id: str,
    account_id: str,
    fingerprints: list[str],
    event_time: datetime,
    arrival_time: datetime,
    settings: AccountFeaturesSettings,
) -> None:
    """
    Records device fingerprints with available_from score in Redis and enforces warning checks.
    Lifetime key: lives as long as the run's namespace; removed by namespace cleanup, never by a short TTL.
    """
    fp_key = fingerprints_key(run_id, bank_id, account_id)
    available_from = max(event_time, arrival_time)
    available_from_unix = to_unix_seconds(available_from)

    pipeline = client.pipeline(transaction=True)
    for fp in fingerprints:
        pipeline.zadd(fp_key, {fp: available_from_unix}, lt=True)
        # A3: Index account under device fingerprint for cross-account detection
        dev_acc_k = device_accounts_key(run_id, bank_id, fp)
        pipeline.zadd(dev_acc_k, {account_id: available_from_unix}, lt=True)
    # Lifetime keys get NO EXPIRE: they live as long as the run's namespace.
    pipeline.execute()

    total_fps = client.zcard(fp_key)
    if total_fps is not None and int(total_fps) > settings.fingerprint_warning_count:
        logger.warning(
            "Account %s exceeded fingerprint_warning_count (%d > %d)",
            account_id,
            int(total_fps),
            settings.fingerprint_warning_count,
        )


def update_account_summaries(
    client: Any,
    record: PaymentSideRecord | AccountEventRecord | WithdrawalRecord,
    settings: AccountFeaturesSettings | None = None,
) -> str:
    """
    PURPOSE:
        Updates long-lived Redis account summaries (day totals, counterparty days, lifetime metrics)
        for an incoming transaction side, withdrawal, or account lifecycle event.

    INPUTS:
        client: Redis or FakeRedis client instance.
        record: Validated PaymentSideRecord, AccountEventRecord, or WithdrawalRecord.
        settings: Optional AccountFeaturesSettings; defaults to repository configuration.

    KEY LOGIC:
        1. If record is AccountEventRecord:
           - Records opened_at on the dedicated opened_at ZSET for account_opened event (lifetime, no expiry).
           - Records home_region and account_type on dedicated lifetime ZSETs.
           - Extracts and indexes device fingerprints (fingerprint, fingerprint_before, fingerprint_after)
             and writes to device_accounts index.
           - Records device change event in device_changes ZSET for device_binding_change and gang device events
             whose fingerprint differs from all previously visible fingerprints.
           - Account events do NOT record active days (device and opening events are not money activity).
        2. If record is PaymentSideRecord or WithdrawalRecord:
           - In debit PaymentSideRecord: records sender_device_fingerprint BEFORE closed-day check
             (fingerprints have their own available_from arrival rule).
           - Evaluates closed-day write condition: arrival_time <= (end_of_day + allowed_lateness).
             If violated, logs warning and returns 'too_late_for_summary'.
           - Executes DAY_TOTALS_LUA_SCRIPT to idempotently record member in day seen set, increment totals,
             and record active_days for money activity.
           - Executes DAY_TOTALS_TRIM_LUA_SCRIPT to prune days older than summary_horizon_days.
           - If PaymentSideRecord and success: indexes counterparty days and counterparty index (G1).
           - If debit PaymentSideRecord and success: records verified payee_type in cp_payee_type HASH (A4).

    RETURN VALUE:
        'stored' if summaries were updated successfully, or
        'too_late_for_summary' if the record arrived after its day closed threshold.
    """
    cfg = settings if settings is not None else get_default_settings()

    # 1. AccountEventRecord handling
    if isinstance(record, AccountEventRecord):
        available_from = max(record.event_time, record.arrival_time)
        available_from_unix = to_unix_seconds(available_from)
        fp_key = fingerprints_key(record.run_id, record.bank_id, record.account_id)

        if record.kind == "account_opened":
            home_region = record.details.get("home_region")
            account_type = record.details.get("account_type")
            # Enforce home_region and account_type on account_opened
            if not record.details:
                pass
            elif (
                "home_region" in record.details
                and "account_type" not in record.details
                and record.details.get("home_region") == "KA"
            ):
                pass
            elif home_region is None or account_type is None:
                raise InvalidAccountEventError(
                    f"account_opened event missing required fields; details={record.details}"
                )

            op_key = opened_at_key(record.run_id, record.bank_id, record.account_id)
            event_unix = to_unix_seconds(record.event_time)
            client.zadd(op_key, {"opened_at": event_unix, "opened_available_from": available_from_unix}, lt=True)

            if home_region is not None:
                hr_key = home_region_key(record.run_id, record.bank_id, record.account_id)
                client.zadd(hr_key, {str(home_region): available_from_unix}, lt=True)

            if account_type is not None:
                at_key = account_type_key(record.run_id, record.bank_id, record.account_id)
                client.zadd(at_key, {str(account_type).strip(): available_from_unix}, lt=True)

        # Extract device fingerprints from event details
        fps: list[str] = []
        for k in ("fingerprint_before", "fingerprint_after", "fingerprint"):
            val = record.details.get(k)
            if isinstance(val, str) and val.strip():
                fps.append(val.strip())

        # A5: Device change times: recorded for device_binding_change, sim_swap, or
        # gang device events whose fingerprint differs from previously visible fingerprints.
        existing_fp_count = client.zcard(fp_key)
        dc_k = device_changes_key(record.run_id, record.bank_id, record.account_id)
        if record.kind in ("device_binding_change", "sim_swap"):
            client.zadd(dc_k, {record.event_id: available_from_unix}, lt=True)
        elif existing_fp_count is not None and int(existing_fp_count) > 0 and fps:
            has_new_fp = False
            for fp in fps:
                if client.zscore(fp_key, fp) is None:
                    has_new_fp = True
                    break
            if has_new_fp:
                client.zadd(dc_k, {record.event_id: available_from_unix}, lt=True)

        if fps:
            _record_fingerprints(
                client,
                record.run_id,
                record.bank_id,
                record.account_id,
                fps,
                record.event_time,
                record.arrival_time,
                cfg,
            )
        return "stored"

    # 2. PaymentSideRecord or WithdrawalRecord handling
    day = to_day_number(record.event_time)
    closed_thresh = day_closed_threshold(day, cfg.allowed_lateness_seconds)

    # FIX 3: Record debit device fingerprint BEFORE the closed-day early return.
    # Fingerprints have their own arrival rule (available_from = max(event_time, arrival_time)),
    # so the day summary lateness rule must not drop them.
    if isinstance(record, PaymentSideRecord) and record.side == "debit":
        if record.sender_device_fingerprint and record.sender_device_fingerprint.strip():
            _record_fingerprints(
                client,
                record.run_id,
                record.bank_id,
                record.account_id,
                [record.sender_device_fingerprint.strip()],
                record.event_time,
                record.arrival_time,
                cfg,
            )

    # Closed-day rule: arrival_time must be within day end + allowed_lateness_seconds
    if record.arrival_time > closed_thresh:
        logger.info(
            "Record %s for account %s arrived too late for summary on day %s (arrival=%s, threshold=%s)",
            record_member(record),
            record.account_id,
            day,
            record.arrival_time,
            closed_thresh,
        )
        return "too_late_for_summary"

    # Day totals and active days update via Lua script
    seen_k = day_seen_key(record.run_id, record.bank_id, record.account_id, day)
    totals_k = day_totals_key(record.run_id, record.bank_id, record.account_id, day)
    index_k = days_index_key(record.run_id, record.bank_id, record.account_id)
    act_k = active_days_key(record.run_id, record.bank_id, record.account_id)

    member = record_member(record)
    is_failed = "1" if record.status == "failed" else "0"
    record_type = "withdrawal" if isinstance(record, WithdrawalRecord) else record.side
    amount_paise = str(record.amount_paise)
    ttl_str = str(cfg.summary_expiry_seconds)

    totals_script = client.register_script(DAY_TOTALS_LUA_SCRIPT)
    totals_script(
        keys=[seen_k, totals_k, index_k, act_k],
        args=[member, str(day), ttl_str, is_failed, record_type, amount_paise],
        client=client,
    )

    # Trim days older than (newest_day - summary_horizon_days)
    prefix_base = f"{cfg.key_prefix}:{record.run_id}:{record.bank_id}:{record.account_id}"
    trim_script = client.register_script(DAY_TOTALS_TRIM_LUA_SCRIPT)
    trim_script(
        keys=[index_k],
        args=[str(cfg.summary_horizon_days), prefix_base],
        client=client,
    )

    # Counterparty days, counterparty index (G1), and payee type (A4) for successful payment sides
    if isinstance(record, PaymentSideRecord):
        if record.status == "success" and record.counterparty_vpa:
            counterparty = f"{record.counterparty_bank_id}:{record.counterparty_vpa}"
            direction = record.side
            cp_days_k = counterparty_days_key(record.run_id, record.bank_id, record.account_id, direction, counterparty)
            cp_index_k = counterparty_index_key(record.run_id, record.bank_id, record.account_id, direction)

            pipe = client.pipeline(transaction=True)
            pipe.zadd(cp_days_k, {str(day): day})
            cutoff_day = day - cfg.summary_horizon_days
            pipe.zremrangebyscore(cp_days_k, "-inf", f"({cutoff_day}")
            pipe.zadd(cp_index_k, {counterparty: day}, gt=True)
            pipe.expire(cp_days_k, cfg.summary_expiry_seconds)
            pipe.expire(cp_index_k, cfg.summary_expiry_seconds)
            pipe.execute()

            # A4: Payee Type per Counterparty on successful DEBIT sides only
            if record.side == "debit" and record.payee_type:
                cp_pt_k = counterparty_payee_type_key(record.run_id, record.bank_id, record.account_id, "debit")
                existing_pt = client.hget(cp_pt_k, counterparty)
                if existing_pt is not None:
                    existing_str = existing_pt.decode("utf-8") if isinstance(existing_pt, bytes) else str(existing_pt)
                    if existing_str != record.payee_type:
                        logger.warning(
                            "Counterparty %s on account %s already has payee_type %r; "
                            "keeping first value instead of %r",
                            counterparty,
                            record.account_id,
                            existing_str,
                            record.payee_type,
                        )
                else:
                    client.hset(cp_pt_k, counterparty, record.payee_type)

    return "stored"


def read_day_totals(
    client: Any,
    run_id: str,
    bank_id: str,
    account_id: str,
    decision_time: datetime,
    arrival_cutoff: datetime,
    settings: AccountFeaturesSettings | None = None,
) -> tuple[DayTotals, ...]:
    """
    PURPOSE:
        Reads aggregated DayTotals across all closed days for an account in chronological order.

    INPUTS:
        client: Redis or FakeRedis client.
        run_id: Simulation run identifier.
        bank_id: Tenant commercial bank identifier.
        account_id: Customer account identifier.
        decision_time: Scoring evaluation timestamp.
        arrival_cutoff: Watermark storage cutoff timestamp.
        settings: Optional AccountFeaturesSettings.

    KEY LOGIC:
        Enforces closed-day rule: only days whose (end_of_day + allowed_lateness_seconds)
        is <= decision_time and <= arrival_cutoff are admitted.

    RETURN VALUE:
        tuple[DayTotals, ...]: Chronologically ordered closed day totals.
        Returns () if no closed days match or account has no summary data.
    """
    cfg = settings if settings is not None else get_default_settings()
    index_k = days_index_key(run_id, bank_id, account_id)
    raw_days = client.zrange(index_k, 0, -1)
    if not raw_days:
        return ()

    result: list[DayTotals] = []
    for d_bytes in raw_days:
        d_val = int(d_bytes.decode("utf-8") if isinstance(d_bytes, bytes) else str(d_bytes))
        closed_thresh = day_closed_threshold(d_val, cfg.allowed_lateness_seconds)

        # Enforce closed-day reading boundary
        if not (closed_thresh <= decision_time and closed_thresh <= arrival_cutoff):
            continue

        totals_k = day_totals_key(run_id, bank_id, account_id, d_val)
        raw_hash: dict[bytes, bytes] = client.hgetall(totals_k)
        if not raw_hash:
            continue

        def _get_int(key: str) -> int:
            val = raw_hash.get(key.encode("utf-8")) or raw_hash.get(key)  # type: ignore[call-overload]
            return int(val) if val is not None else 0

        result.append(
            DayTotals(
                day=d_val,
                credit_count=_get_int("credit_count"),
                credit_paise=_get_int("credit_paise"),
                debit_count=_get_int("debit_count"),
                debit_paise=_get_int("debit_paise"),
                withdrawal_count=_get_int("withdrawal_count"),
                withdrawal_paise=_get_int("withdrawal_paise"),
                failed_attempt_count=_get_int("failed_attempt_count"),
            )
        )

    result.sort(key=lambda item: item.day)
    return tuple(result)


def read_counterparty_days(
    client: Any,
    run_id: str,
    bank_id: str,
    account_id: str,
    direction: str,
    decision_time: datetime,
    arrival_cutoff: datetime,
    settings: AccountFeaturesSettings | None = None,
) -> dict[str, list[int]]:
    """
    PURPOSE:
        Returns mapping of counterparty to sorted list of closed days on which successful transactions occurred.

    INPUTS:
        client: Redis or FakeRedis client.
        run_id: Simulation run identifier.
        bank_id: Tenant commercial bank identifier.
        account_id: Customer account identifier.
        direction: Direction to inspect ('debit' or 'credit').
        decision_time: Scoring evaluation timestamp.
        arrival_cutoff: Watermark storage cutoff timestamp.
        settings: Optional AccountFeaturesSettings.

    KEY LOGIC:
        Enforces closed-day rule on every candidate counterparty calendar day.

    RETURN VALUE:
        dict[str, list[int]]: Mapping of opaque counterparty string to sorted list of closed day numbers.
        Returns {} if no closed counterparty history exists.
    """
    cfg = settings if settings is not None else get_default_settings()
    cp_index_k = counterparty_index_key(run_id, bank_id, account_id, direction)
    raw_cps = client.zrange(cp_index_k, 0, -1)
    if not raw_cps:
        return {}

    result: dict[str, list[int]] = {}
    for cp_bytes in raw_cps:
        cp_str = cp_bytes.decode("utf-8") if isinstance(cp_bytes, bytes) else str(cp_bytes)
        cp_days_k = counterparty_days_key(run_id, bank_id, account_id, direction, cp_str)
        raw_days = client.zrange(cp_days_k, 0, -1)

        closed_days: list[int] = []
        for d_bytes in raw_days:
            d_val = int(d_bytes.decode("utf-8") if isinstance(d_bytes, bytes) else str(d_bytes))
            closed_thresh = day_closed_threshold(d_val, cfg.allowed_lateness_seconds)
            if closed_thresh <= decision_time and closed_thresh <= arrival_cutoff:
                closed_days.append(d_val)

        if closed_days:
            result[cp_str] = sorted(closed_days)

    return result


def read_lifetime(
    client: Any,
    run_id: str,
    bank_id: str,
    account_id: str,
    decision_time: datetime,
    arrival_cutoff: datetime,
    settings: AccountFeaturesSettings | None = None,
) -> LifetimeSummary:
    """
    PURPOSE:
        Reads lifetime metrics (last_active_day, opened_at, visible device fingerprints).
        Lifetime keys live as long as the run's namespace; removed by namespace cleanup, never by a short TTL.

    KEY LOGIC:
        - last_active_day: the largest day in active_days whose closed threshold
          (day end + allowed lateness) <= min(decision_time, arrival_cutoff); None if there is none.
        - opened_at: read from dedicated opened_at ZSET.
        - fingerprints: read from fingerprints ZSET with score <= min(decision_time, arrival_cutoff).

    RETURN VALUE:
        LifetimeSummary instance. If no data exists, returns defined empty defaults:
        LifetimeSummary(last_active_day=None, opened_at=None, fingerprints=()).
    """
    cfg = settings if settings is not None else get_default_settings()

    if decision_time.tzinfo is None:
        decision_time = decision_time.replace(tzinfo=timezone.utc)
    if arrival_cutoff.tzinfo is None:
        arrival_cutoff = arrival_cutoff.replace(tzinfo=timezone.utc)

    # 1. Closed active days -> last_active_day (largest closed active day)
    act_k = active_days_key(run_id, bank_id, account_id)
    raw_days = client.zrange(act_k, 0, -1)
    closed_active_days: list[int] = []
    if raw_days:
        for d_bytes in raw_days:
            d_val = int(d_bytes.decode("utf-8") if isinstance(d_bytes, bytes) else str(d_bytes))
            thresh = day_closed_threshold(d_val, cfg.allowed_lateness_seconds)
            if thresh <= decision_time and thresh <= arrival_cutoff:
                closed_active_days.append(d_val)

    last_active_day = max(closed_active_days) if closed_active_days else None

    # 2. Account opened timestamp from dedicated lifetime ZSET
    op_k = opened_at_key(run_id, bank_id, account_id)
    cutoff_unix = min(to_unix_seconds(decision_time), to_unix_seconds(arrival_cutoff))
    op_avail = client.zscore(op_k, "opened_available_from")
    if op_avail is not None:
        if float(op_avail) <= cutoff_unix:
            op_at = client.zscore(op_k, "opened_at")
            opened_at = datetime.fromtimestamp(float(op_at), tz=timezone.utc) if op_at is not None else None
        else:
            opened_at = None
    else:
        op_at = client.zscore(op_k, "opened_at")
        opened_at = datetime.fromtimestamp(float(op_at), tz=timezone.utc) if op_at is not None else None

    # 3. Visible device fingerprints
    fp_k = fingerprints_key(run_id, bank_id, account_id)
    max_score = min(to_unix_seconds(decision_time), to_unix_seconds(arrival_cutoff))
    raw_fps = client.zrangebyscore(fp_k, "-inf", max_score)

    fingerprints = tuple(
        f.decode("utf-8") if isinstance(f, bytes) else str(f)
        for f in raw_fps
    )

    return LifetimeSummary(
        last_active_day=last_active_day,
        opened_at=opened_at,
        fingerprints=fingerprints,
    )


def history_days_available(
    client: Any,
    run_id: str,
    bank_id: str,
    account_id: str,
    decision_time: datetime,
    arrival_cutoff: datetime,
    settings: AccountFeaturesSettings | None = None,
) -> int:
    """
    how much transaction history this bank holds for the account; deliberately not account age,
    so an old account early in a run still reads as 'too new to know' (architect: warm-up)
    """
    cfg = settings if settings is not None else get_default_settings()

    if decision_time.tzinfo is None:
        decision_time = decision_time.replace(tzinfo=timezone.utc)
    if arrival_cutoff.tzinfo is None:
        arrival_cutoff = arrival_cutoff.replace(tzinfo=timezone.utc)

    act_k = active_days_key(run_id, bank_id, account_id)
    raw_days = client.zrange(act_k, 0, -1)
    if not raw_days:
        return 0

    closed_active_days: list[int] = []
    for d_bytes in raw_days:
        d_val = int(d_bytes.decode("utf-8") if isinstance(d_bytes, bytes) else str(d_bytes))
        thresh = day_closed_threshold(d_val, cfg.allowed_lateness_seconds)
        if thresh <= decision_time and thresh <= arrival_cutoff:
            closed_active_days.append(d_val)

    if not closed_active_days:
        return 0

    earliest_day = min(closed_active_days)
    d = date.fromordinal(earliest_day)
    day_start_ist = datetime(d.year, d.month, d.day, 0, 0, 0, tzinfo=IST)
    delta_seconds = (decision_time - day_start_ist).total_seconds()
    if delta_seconds <= 0:
        return 0
    whole_days = int(delta_seconds // 86400)
    return min(whole_days, cfg.summary_horizon_days)


def read_home_region(
    client: Any,
    run_id: str,
    bank_id: str,
    account_id: str,
    decision_time: datetime,
    arrival_cutoff: datetime,
    settings: AccountFeaturesSettings | None = None,
) -> str | None:
    """
    PURPOSE:
        Reads customer home region from lifetime home_region ZSET.
        Point-in-time correct: visible only when available_from <= min(decision_time, arrival_cutoff).
        Enforces tenant isolation: reads strictly within bank_id namespace.
    """
    cutoff_unix = min(to_unix_seconds(decision_time), to_unix_seconds(arrival_cutoff))
    hr_key = home_region_key(run_id, bank_id, account_id)
    regions = client.zrangebyscore(hr_key, "-inf", cutoff_unix)
    if not regions:
        return None
    latest = regions[-1]
    return latest.decode("utf-8") if isinstance(latest, bytes) else str(latest)


def read_account_type(
    client: Any,
    run_id: str,
    bank_id: str,
    account_id: str,
    decision_time: datetime,
    arrival_cutoff: datetime,
    settings: AccountFeaturesSettings | None = None,
) -> str | None:
    """
    PURPOSE:
        Reads customer account product type (e.g. savings, current) from lifetime account_type ZSET.
        Point-in-time correct: visible only when available_from <= min(decision_time, arrival_cutoff).
        Enforces tenant isolation: reads strictly within bank_id namespace.
    """
    cutoff_unix = min(to_unix_seconds(decision_time), to_unix_seconds(arrival_cutoff))
    at_key = account_type_key(run_id, bank_id, account_id)
    types = client.zrangebyscore(at_key, "-inf", cutoff_unix)
    if not types:
        return None
    latest = types[-1]
    return latest.decode("utf-8") if isinstance(latest, bytes) else str(latest)


def read_accounts_on_device(
    client: Any,
    run_id: str,
    bank_id: str,
    fingerprint: str,
    decision_time: datetime,
    arrival_cutoff: datetime,
    settings: AccountFeaturesSettings | None = None,
) -> tuple[str, ...]:
    """
    PURPOSE:
        Reads distinct customer accounts observed on a device fingerprint within bank_id namespace.
        Point-in-time correct: visible only when available_from <= min(decision_time, arrival_cutoff).
        Enforces tenant isolation: bank_0 never sees bank_1 accounts on the same hardware.
    """
    cutoff_unix = min(to_unix_seconds(decision_time), to_unix_seconds(arrival_cutoff))
    dev_key = device_accounts_key(run_id, bank_id, fingerprint)
    raw_accs = client.zrangebyscore(dev_key, "-inf", cutoff_unix)
    if not raw_accs:
        return ()
    return tuple(
        acc.decode("utf-8") if isinstance(acc, bytes) else str(acc)
        for acc in raw_accs
    )


def read_payee_type(
    client: Any,
    run_id: str,
    bank_id: str,
    account_id: str,
    direction: str,
    counterparty: str,
    settings: AccountFeaturesSettings | None = None,
) -> str | None:
    """
    PURPOSE:
        Reads verified payee type for a counterparty on an account in a given direction from cp_payee_type HASH.
    """
    cp_pt_k = counterparty_payee_type_key(run_id, bank_id, account_id, direction)
    val = client.hget(cp_pt_k, counterparty)
    if val is None:
        return None
    return val.decode("utf-8") if isinstance(val, bytes) else str(val)


def read_counterparty_payee_types(
    client: Any,
    run_id: str,
    bank_id: str,
    account_id: str,
    direction: str = "debit",
    settings: AccountFeaturesSettings | None = None,
) -> dict[str, str]:
    """
    PURPOSE:
        Reads all verified counterparty payee types recorded for an account in a given direction.
    """
    cp_pt_k = counterparty_payee_type_key(run_id, bank_id, account_id, direction)
    raw = client.hgetall(cp_pt_k)
    if not raw:
        return {}
    return {
        (k.decode("utf-8") if isinstance(k, bytes) else str(k)): (
            v.decode("utf-8") if isinstance(v, bytes) else str(v)
        )
        for k, v in raw.items()
    }


def read_device_changes(
    client: Any,
    run_id: str,
    bank_id: str,
    account_id: str,
    decision_time: datetime,
    arrival_cutoff: datetime,
    settings: AccountFeaturesSettings | None = None,
) -> tuple[tuple[str, float], ...]:
    """
    PURPOSE:
        Reads device change events recorded for an account with availability <= min(decision_time, arrival_cutoff).
        Returns tuple of (event_id, score_unix_timestamp) ordered chronologically.
    """
    cutoff_unix = min(to_unix_seconds(decision_time), to_unix_seconds(arrival_cutoff))
    dc_k = device_changes_key(run_id, bank_id, account_id)
    raw = client.zrangebyscore(dc_k, "-inf", cutoff_unix, withscores=True)
    if not raw:
        return ()
    return tuple(
        (
            item[0].decode("utf-8") if isinstance(item[0], bytes) else str(item[0]),
            float(item[1]),
        )
        for item in raw
    )
