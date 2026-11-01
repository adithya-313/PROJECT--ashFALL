# File: features/account_labels.py
# Purpose: Level 15b ground truth fraud labels, complaints polling, and same-bank counterparty graph edges.
# Issue: Level 15b M1 network features (plan Task 1, G1, R8, architect decision 9b).
"""
WHAT: Provides PostgreSQL polling adapters for bank_feedback and bank_complaints,
Redis lifetime label/complaint state management, and same-bank counterparty edge extraction.

WHY:
  - Fraud detection models require confirmed ground-truth labels (confirmed_fraud) to compute network
    proximity features (hops_to_confirmed_label, shared_counterparties_with_flagged_7d).
  - Feedback arrives asynchronously from fraud operations; confirmed_fraud marks the receiving credit-side
    account (mule) because the sending debit-side account is typically the victim.
  - Complaints provide a weaker, unverified signal marking reported accounts belonging to this bank.
  - All label and complaint entries in Redis are stored on lifetime ZSETs with available_from scores
    and no expiry, ensuring point-in-time Section 22 evaluation without future leakage.
  - Counterparty graph edges only ever include this bank's accounts, strictly respecting tenant isolation.
"""

import logging
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from core.event_clock import Clock
from features.account_keys import (
    AccountFeaturesSettings,
    complained_accounts_key,
    extract_run_id,
    get_default_settings,
    labelled_accounts_key,
)
from features.account_reader import read_side_records_many
from features.account_records import arrival_from_row
from features.account_summaries import read_counterparty_days_many
from features.ingest_watermark import record_poller_watermark

logger = logging.getLogger(__name__)


def to_unix_seconds(dt: datetime) -> int:
    """Convert a UTC or timezone-aware datetime to integer Unix seconds."""
    if dt.tzinfo is None:
        return int(dt.replace(tzinfo=timezone.utc).timestamp())
    return int(dt.timestamp())


def arrival_of(row: Mapping[str, Any]) -> datetime:
    """
    WHAT: Resolves the effective arrival timestamp for a complaint or feedback row.

    WHY: Delegates to the shared arrival_from_row helper with fallback column 'created_at'.
    Domain visibility, cutoffs, and watermarks are evaluated against arrival_time (from the
    injected ingestion clock). When arrival_time is None (real-clock legacy mode or historical
    rows), falls back to created_at per Ground Rule 13 and Ground Rule 30.

    Args:
        row: Mapping containing row data (must contain 'created_at' and optionally 'arrival_time').

    Returns:
        A timezone-aware datetime representing the arrival time.

    Raises:
        ValueError: If the resolved timestamp is naive or invalid.
    """
    return arrival_from_row(row, "created_at")


async def apply_feedback_row(
    connection: Any,
    redis_client: Any,
    row: Mapping[str, Any],
    settings: AccountFeaturesSettings | None = None,
) -> None:
    """
    WHAT: Applies a single bank_feedback row to Redis fraud labels.

    WHY: Shared row-level handler used by live polling and history replay. Confirmed fraud
    labels the credit-side account (mule) on the lifetime ZSET with available_from score.

    Args:
        connection: PostgreSQL connection for transaction lookups.
        redis_client: Redis client instance.
        row: Mapping containing bank_feedback row columns.
        settings: Optional account feature settings.
    """
    outcome = row["outcome"]
    # Architect decision 9b: confirmed_fraud feedback only; not_fraud and customer_dispute are ignored here.
    if outcome != "confirmed_fraud":
        logger.info(
            "Feedback %s on payment %s outcome is %r; not confirmed_fraud, ignoring",
            row["feedback_id"],
            row["payment_id"],
            outcome,
        )
        return

    payment_id = row["payment_id"]
    run_id = extract_run_id(payment_id)
    bank_id = row["bank_id"]

    # Look up THIS bank's transactions rows for that payment_id
    txn_rows = await connection.fetch(
        """
        SELECT side, account_id
        FROM transactions
        WHERE bank_id = $1 AND payment_id = $2
        """,
        bank_id,
        payment_id,
    )

    credit_accounts = [r["account_id"] for r in txn_rows if r["side"] == "credit"]
    debit_accounts = [r["account_id"] for r in txn_rows if r["side"] == "debit"]

    if credit_accounts:
        # Score is available_from: max(feedback event_time, arrival_of(row))
        available_from = max(row["event_time"], arrival_of(row))
        score = to_unix_seconds(available_from)
        key = labelled_accounts_key(run_id, bank_id)
        for acc_id in credit_accounts:
            redis_client.zadd(key, {acc_id: score}, lt=True)
    elif debit_accounts:
        # Only debit side exists; sender is victim, do not label
        logger.info(
            "Feedback %s on payment %s for bank %s has only debit side; labelling nothing",
            row["feedback_id"],
            payment_id,
            bank_id,
        )
    else:
        logger.info(
            "Feedback %s on payment %s for bank %s has no transaction records; labelling nothing",
            row["feedback_id"],
            payment_id,
            bank_id,
        )


async def apply_feedback_since(
    connection: Any,
    redis_client: Any,
    settings: AccountFeaturesSettings | None = None,
    bank_id: str = "bank_0",
    after_id: int = 0,
    clock: Clock | None = None,
    run_id: str | None = None,
) -> int:
    """
    WHAT: Queries bank_feedback for rows with id > after_id (with overlap re-read) and updates Redis labelled_accounts.

    WHY: Architect decision 9b: confirmed labels are driven by confirmed_fraud feedback only.
    For each confirmed_fraud row, this bank's transactions table is queried for the payment_id.
    If a credit side exists, its account_id (the recipient/mule) is labelled in Redis. If only a debit
    side exists, the account is a victim, so nothing is labelled.

    NOTE: not_fraud and customer_dispute outcomes are ignored here as they do not constitute confirmed fraud.

    Args:
        connection: Active database connection or connection-like object (asyncpg.Connection).
        redis_client: Redis client instance.
        settings: Optional AccountFeaturesSettings; defaults to repository configuration.
        bank_id: Commercial bank identifier.
        after_id: Cursor identifier; rows with id > (after_id - overlap) are selected.
        clock: Optional domain ingestion clock. When provided, poll_start uses clock.now_ist()
            without querying DB now(); when None, falls back to fetchval("SELECT now()").
        run_id: Optional simulation run identifier. When clock is provided, run_id must also be
            provided to write run-scoped watermarks ('acctwm:{run_id}:{bank_id}:feedback').
            Run-scoped keys belong only to world-clock runs; if clock is None and run_id is given,
            raises ValueError.

    Returns:
        The highest row ID processed, or after_id if no rows were found.
    """
    if clock is not None and run_id is None:
        raise ValueError("When clock is provided, run_id must also be provided to write run-scoped watermarks")
    if clock is None and run_id is not None:
        raise ValueError(
            "run_id cannot be provided when clock is None; run-scoped keys belong only to world-clock runs"
        )

    cfg = settings if settings is not None else get_default_settings()
    if clock is not None:
        poll_start_time = clock.now_ist()
    else:
        poll_start_time = await connection.fetchval("SELECT now()")

    batch_size = cfg.account_event_batch_size
    overlap = cfg.account_event_overlap_rows
    start_id = max(0, after_id - overlap)

    rows = await connection.fetch(
        """
        SELECT id, bank_id, feedback_id, payment_id, outcome, event_time, created_at, arrival_time
        FROM bank_feedback
        WHERE bank_id = $1 AND id > $2
        ORDER BY id
        LIMIT $3
        """,
        bank_id,
        start_id,
        batch_size,
    )

    last_id = after_id
    if rows:
        for row in rows:
            row_id = int(row["id"])
            if row_id > last_id:
                last_id = row_id
            await apply_feedback_row(connection, redis_client, row, cfg)

    watermark_ts = int(to_unix_seconds(poll_start_time)) - cfg.commit_grace
    record_poller_watermark(redis_client, bank_id, "feedback", watermark_ts, run_id=run_id)

    return last_id


def apply_complaint_row(
    redis_client: Any,
    row: Mapping[str, Any],
    settings: AccountFeaturesSettings | None = None,
) -> None:
    """
    WHAT: Applies a single bank_complaints row to Redis complaint labels.

    WHY: Shared row-level handler used by live polling and history replay. Marks reported
    accounts belonging to this bank on the lifetime ZSET with available_from score.

    Args:
        redis_client: Redis client instance.
        row: Mapping containing bank_complaints row columns.
        settings: Optional account feature settings.
    """
    bank_id = row["bank_id"]
    # Mark reported_account ONLY when reported_bank_id equals this bank
    if row["reported_bank_id"] != bank_id:
        logger.info(
            "Complaint %s reported_bank_id %r does not match this bank %r; ignoring",
            row["complaint_id"],
            row["reported_bank_id"],
            bank_id,
        )
        return

    complaint_id = row["complaint_id"]
    run_id = extract_run_id(complaint_id)
    reported_account = row["reported_account"]

    available_from = max(row["received_at"], arrival_of(row))
    score = to_unix_seconds(available_from)
    key = complained_accounts_key(run_id, bank_id)
    redis_client.zadd(key, {reported_account: score}, lt=True)


async def apply_complaints_since(
    connection: Any,
    redis_client: Any,
    settings: AccountFeaturesSettings | None = None,
    bank_id: str = "bank_0",
    after_id: int = 0,
    clock: Clock | None = None,
    run_id: str | None = None,
) -> int:
    """
    WHAT: Queries bank_complaints for rows with id > after_id and records complained accounts for this bank.

    WHY: Complaints provide a weaker suspicion signal. Each complaint row marks reported_account
    ONLY when reported_bank_id equals this bank.

    Args:
        connection: Active database connection or connection-like object (asyncpg.Connection).
        redis_client: Redis client instance.
        settings: Optional AccountFeaturesSettings; defaults to repository configuration.
        bank_id: Commercial bank identifier.
        after_id: Cursor identifier; rows with id > (after_id - overlap) are selected.
        clock: Optional domain ingestion clock. When provided, poll_start uses clock.now_ist()
            without querying DB now(); when None, falls back to fetchval("SELECT now()").
        run_id: Optional simulation run identifier. When clock is provided, run_id must also be
            provided to write run-scoped watermarks ('acctwm:{run_id}:{bank_id}:complaints').
            Run-scoped keys belong only to world-clock runs; if clock is None and run_id is given,
            raises ValueError.

    Returns:
        The highest row ID processed, or after_id if no rows were found.
    """
    if clock is not None and run_id is None:
        raise ValueError("When clock is provided, run_id must also be provided to write run-scoped watermarks")
    if clock is None and run_id is not None:
        raise ValueError(
            "run_id cannot be provided when clock is None; run-scoped keys belong only to world-clock runs"
        )

    cfg = settings if settings is not None else get_default_settings()
    if clock is not None:
        poll_start_time = clock.now_ist()
    else:
        poll_start_time = await connection.fetchval("SELECT now()")

    batch_size = cfg.account_event_batch_size
    overlap = cfg.account_event_overlap_rows
    start_id = max(0, after_id - overlap)

    rows = await connection.fetch(
        """
        SELECT id, bank_id, complaint_id, complainant_account, reported_account, reported_bank_id,
               payment_ids, claimed_type, received_at, created_at, arrival_time
        FROM bank_complaints
        WHERE bank_id = $1 AND id > $2
        ORDER BY id
        LIMIT $3
        """,
        bank_id,
        start_id,
        batch_size,
    )

    last_id = after_id
    if rows:
        for row in rows:
            row_id = int(row["id"])
            if row_id > last_id:
                last_id = row_id
            apply_complaint_row(redis_client, row, cfg)

    watermark_ts = int(to_unix_seconds(poll_start_time)) - cfg.commit_grace
    record_poller_watermark(redis_client, bank_id, "complaints", watermark_ts, run_id=run_id)

    return last_id


def labelled_accounts_visible(
    client: Any,
    run_id: str,
    bank_id: str,
    decision_time: datetime,
    arrival_cutoff: datetime,
) -> frozenset[str]:
    """
    WHAT: Returns the set of confirmed fraud labelled accounts visible at decision time.

    WHY: Evaluates point-in-time visibility: an account is visible only if available_from <= min(decision, cutoff).
    """
    cutoff = min(decision_time, arrival_cutoff)
    cutoff_unix = to_unix_seconds(cutoff)
    key = labelled_accounts_key(run_id, bank_id)
    raw_members = client.zrangebyscore(key, "-inf", cutoff_unix)
    if not raw_members:
        return frozenset()
    return frozenset(m.decode("utf-8") if isinstance(m, bytes) else str(m) for m in raw_members)


def complained_accounts_visible(
    client: Any,
    run_id: str,
    bank_id: str,
    decision_time: datetime,
    arrival_cutoff: datetime,
) -> frozenset[str]:
    """
    WHAT: Returns the set of complained accounts visible at decision time.

    WHY: Evaluates point-in-time visibility: an account is visible only if available_from <= min(decision, cutoff).
    """
    cutoff = min(decision_time, arrival_cutoff)
    cutoff_unix = to_unix_seconds(cutoff)
    key = complained_accounts_key(run_id, bank_id)
    raw_members = client.zrangebyscore(key, "-inf", cutoff_unix)
    if not raw_members:
        return frozenset()
    return frozenset(m.decode("utf-8") if isinstance(m, bytes) else str(m) for m in raw_members)


def same_bank_counterparties(
    client: Any,
    run_id: str,
    bank_id: str,
    account_id: str,
    decision_time: datetime,
    arrival_cutoff: datetime,
    settings: AccountFeaturesSettings | None = None,
) -> frozenset[str]:
    """
    PURPOSE:
        Returns the set of account IDs at this bank that this account paid or was paid by,
        sourced from closed counterparty days (both directions) plus the visible 7-day raw window.

    CONSTRAINTS:
        - Only ever contains this bank's accounts (counterparty_bank_id == bank_id).
        - Cross-bank counterparties are strictly excluded (tenant isolation).
        - Self-loops (account_id paying account_id) are discarded.
        - This forms the payment graph edge set for local network features.
    """
    return same_bank_counterparties_many(
        client, run_id, bank_id, [account_id], decision_time, arrival_cutoff, settings
    )[account_id]


def same_bank_counterparties_many(
    client: Any,
    run_id: str,
    bank_id: str,
    account_ids: Sequence[str],
    decision_time: datetime,
    arrival_cutoff: datetime,
    settings: AccountFeaturesSettings | None = None,
) -> dict[str, frozenset[str]]:
    """
    PURPOSE:
        same_bank_counterparties for several accounts at once, using four Redis round trips in total
        (two pipelines for the closed counterparty days, two for the visible 7-day side records).

    CONSTRAINTS:
        Each account's set is exactly what a separate same_bank_counterparties call returns.
    """
    cfg = settings if settings is not None else get_default_settings()
    unique_accounts = list(dict.fromkeys(account_ids))
    closed_days = read_counterparty_days_many(
        client, run_id, bank_id, unique_accounts, ("debit", "credit"), decision_time, arrival_cutoff, cfg
    )
    side_records = read_side_records_many(
        client, run_id, bank_id, unique_accounts, decision_time, arrival_cutoff, cfg
    )

    result: dict[str, frozenset[str]] = {}
    for account_id in unique_accounts:
        accs: set[str] = set()

        # 1. Closed counterparty days in both directions (debit and credit)
        for direction in ("debit", "credit"):
            for cp_str, days in closed_days[(account_id, direction)].items():
                if not days:
                    continue
                if ":" in cp_str:
                    cp_bank, cp_vpa = cp_str.split(":", 1)
                    if cp_bank == bank_id and cp_vpa:
                        accs.add(cp_vpa)

        # 2. Visible 7-day raw window records
        for rec in side_records[account_id]:
            if rec.status == "success" and rec.counterparty_bank_id == bank_id and rec.counterparty_vpa:
                accs.add(rec.counterparty_vpa)

        accs.discard(account_id)
        result[account_id] = frozenset(accs)
    return result
