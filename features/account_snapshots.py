# File: features/account_snapshots.py
# Purpose: Level 15c Round 2b day-partitioned account feature snapshots with watermark cutoff.
# Issue: Level 15c account snapshots and size estimate (D1-D3).
"""
WHAT: Provides scheduled point-in-time account feature snapshot generation and persistence,
partitioned by snapshot_time, capturing active account profiles daily and all accounts weekly.

WHY: Machine learning models for live mule detection require dense point-in-time feature rows
for both active and dormant accounts. Taking snapshots with the ingestion watermark as the arrival
cutoff prevents future-leakage, ensures point-in-time consistency, and avoids online recomputation
latency during training data preparation.
"""

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import orjson
import yaml

from features.account_keys import (
    AccountFeaturesSettings,
    extract_run_id,
    get_default_settings,
)
from features.feature_set import FEATURE_SET_VERSION, load_feature_set_m1
from features.ingest_watermark import get_bank_watermark
from features.m1_features import SUPPORTED_GROUPS, compute_account_features

__all__ = [
    "NoWatermarkError",
    "SnapshotSettings",
    "account_feature_names",
    "account_feature_names_sha256",
    "full_snapshot_due",
    "load_snapshot_settings",
    "select_all_accounts",
    "select_daily_snapshot_accounts",
    "to_account_ordered_values",
    "write_account_snapshots",
]


class NoWatermarkError(Exception):
    """Raised when an account snapshot is attempted but no ingestion watermark is available."""


@dataclass(frozen=True)
class SnapshotSettings:
    """
    Configuration parameters for account feature snapshots.

    Attributes:
        full_interval_days: Interval in days between full snapshots across all accounts.
    """

    full_interval_days: int


def load_snapshot_settings(config_path: Path | None = None) -> SnapshotSettings:
    """
    WHAT: Loads account snapshot settings from config/account_features.yaml.

    WHY: Reads operational cadence parameters from configuration rather than typed literals.

    Args:
        config_path: Optional explicit Path to configuration file.

    Returns:
        SnapshotSettings instance.

    Raises:
        FileNotFoundError: If configuration file is missing.
        ValueError: If configuration values fail schema or type validation.
    """
    root_dir = Path(__file__).resolve().parent.parent
    if config_path is None:
        config_path = root_dir / "config" / "account_features.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)

    if not isinstance(raw_cfg, dict) or "account_snapshots" not in raw_cfg:
        raise ValueError(f"Missing 'account_snapshots' root block in {config_path}")

    block = raw_cfg["account_snapshots"]
    if not isinstance(block, dict) or "full_interval_days" not in block:
        raise ValueError(f"Missing 'full_interval_days' in account_snapshots block in {config_path}")

    full_int_block = block["full_interval_days"]
    if isinstance(full_int_block, dict):
        if "value" not in full_int_block:
            raise ValueError(f"Missing 'value' in full_interval_days in {config_path}")
        full_interval_days = int(full_int_block["value"])
    elif isinstance(full_int_block, int):
        full_interval_days = full_int_block
    else:
        raise ValueError(f"Invalid full_interval_days in {config_path}")

    if full_interval_days <= 0:
        raise ValueError(f"full_interval_days must be positive; got {full_interval_days}")

    return SnapshotSettings(full_interval_days=full_interval_days)


@lru_cache(maxsize=1)
def account_feature_names() -> tuple[str, ...]:
    """
    WHAT: Returns canonical M1 feature names excluding payment-level features, in schema catalog order.

    WHY: Account snapshots persist account-scoped state independently of specific payment transactions.
    Computed programmatically from features/feature_set.py functions, never retyped.
    """
    feature_set = load_feature_set_m1()
    payment_names = {f.name for f in feature_set.features_by_group("payment_level")}
    return tuple(f.name for f in feature_set.features if f.name not in payment_names)


@lru_cache(maxsize=1)
def account_feature_names_sha256() -> str:
    """Returns SHA256 hex digest of account feature names joined by newline."""
    names = account_feature_names()
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()


def to_account_ordered_values(features_dict: dict[str, Any]) -> list[Any]:
    """
    WHAT: Serializes a dictionary of account features into a list matching account_feature_names() order.

    WHY: Validates schema keys against catalog and preserves deterministic ordering for JSONB arrays.

    Raises:
        ValueError: If keys in features_dict do not match account_feature_names() exactly.
    """
    names = account_feature_names()
    expected = set(names)
    actual = set(features_dict.keys())
    if actual != expected:
        raise ValueError(
            "Account feature dict keys do not match account schema. "
            f"Missing: {expected - actual}, Unexpected: {actual - expected}"
        )
    return [features_dict[name] for name in names]


def full_snapshot_due(day_number: int, settings: SnapshotSettings) -> bool:
    """
    WHAT: Evaluates whether a full snapshot across all customer accounts is due on day_number.

    WHY: Balances daily active-account capture with weekly complete-ledger state coverage.
    """
    return day_number % settings.full_interval_days == 0


def _ensure_utc(dt: datetime) -> datetime:
    """Ensures datetime is timezone-aware and normalized to UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


async def select_daily_snapshot_accounts(
    pg: Any,
    run_id: str,
    bank_id: str,
    day_start: datetime,
    day_end: datetime,
) -> list[str]:
    """
    WHAT: Selects unique, sorted account IDs belonging to bank_id and run_id that had ANY event
    arrive in [day_start, day_end).

    WHY: Captures active customer accounts for daily snapshot training sets across all 5 operational feeds:
      1. transactions (debit and credit legs where bank_id = bank_id and created_at in window)
      2. bank_account_events (account_id where bank_id = bank_id and created_at in window)
      3. bank_withdrawals (account_id where bank_id = bank_id and recorded_at in window)
      4. bank_feedback (confirmed_fraud feedback arriving in window -> credit-side mule account)
      5. bank_complaints (reported_account where reported_bank_id = bank_id and created_at in window)

    Filters run_id using extract_run_id(id_col) == run_id to enforce multi-run isolation.
    """
    t_start = _ensure_utc(day_start)
    t_end = _ensure_utc(day_end)
    active_accounts: set[str] = set()

    # 1. Transactions: debit and credit legs where created_at is within [day_start, day_end)
    txn_rows = await pg.fetch(
        """
        SELECT account_id, payment_id
        FROM transactions
        WHERE bank_id = $1 AND created_at >= $2 AND created_at < $3
        """,
        bank_id,
        t_start,
        t_end,
    )
    for r in txn_rows:
        pid = str(r["payment_id"])
        try:
            if extract_run_id(pid) == run_id:
                active_accounts.add(str(r["account_id"]))
        except ValueError:
            pass

    # 2. Bank account events: account lifecycle changes arriving in window
    event_rows = await pg.fetch(
        """
        SELECT account_id, event_id
        FROM bank_account_events
        WHERE bank_id = $1 AND created_at >= $2 AND created_at < $3
        """,
        bank_id,
        t_start,
        t_end,
    )
    for r in event_rows:
        eid = str(r["event_id"])
        try:
            if extract_run_id(eid) == run_id:
                active_accounts.add(str(r["account_id"]))
        except ValueError:
            pass

    # 3. Bank withdrawals: cash withdrawals arriving in window (recorded_at)
    withdrawal_rows = await pg.fetch(
        """
        SELECT account_id, withdrawal_id
        FROM bank_withdrawals
        WHERE bank_id = $1 AND recorded_at >= $2 AND recorded_at < $3
        """,
        bank_id,
        t_start,
        t_end,
    )
    for r in withdrawal_rows:
        wid = str(r["withdrawal_id"])
        try:
            if extract_run_id(wid) == run_id:
                active_accounts.add(str(r["account_id"]))
        except ValueError:
            pass

    # 4. Bank feedback: confirmed_fraud feedback arriving in window -> look up credit-side (mule) account
    feedback_rows = await pg.fetch(
        """
        SELECT payment_id, feedback_id
        FROM bank_feedback
        WHERE bank_id = $1 AND outcome = 'confirmed_fraud' AND created_at >= $2 AND created_at < $3
        """,
        bank_id,
        t_start,
        t_end,
    )
    for r in feedback_rows:
        pid = str(r["payment_id"])
        try:
            if extract_run_id(pid) == run_id:
                mule_rows = await pg.fetch(
                    """
                    SELECT account_id
                    FROM transactions
                    WHERE bank_id = $1 AND payment_id = $2 AND side = 'credit'
                    """,
                    bank_id,
                    pid,
                )
                for mr in mule_rows:
                    active_accounts.add(str(mr["account_id"]))
        except ValueError:
            pass

    # 5. Bank complaints: reported_account where reported_bank_id = bank_id and created_at in window
    complaint_rows = await pg.fetch(
        """
        SELECT reported_account, complaint_id
        FROM bank_complaints
        WHERE reported_bank_id = $1 AND created_at >= $2 AND created_at < $3
        """,
        bank_id,
        t_start,
        t_end,
    )
    for r in complaint_rows:
        cid = str(r["complaint_id"])
        try:
            if extract_run_id(cid) == run_id:
                active_accounts.add(str(r["reported_account"]))
        except ValueError:
            pass

    return sorted(list(active_accounts))


async def select_all_accounts(
    pg: Any,
    run_id: str,
    bank_id: str,
    arrival_cutoff: datetime,
) -> list[str]:
    """
    WHAT: Selects sorted list of every account with an account_opened event in bank_account_events
    arrived by arrival_cutoff, belonging to bank_id and run_id.

    WHY: Populates weekly full snapshot covering quiet and dormant accounts without transaction activity.
    """
    cutoff = _ensure_utc(arrival_cutoff)
    accounts: set[str] = set()

    event_rows = await pg.fetch(
        """
        SELECT account_id, event_id
        FROM bank_account_events
        WHERE bank_id = $1 AND kind = 'account_opened' AND created_at <= $2
        """,
        bank_id,
        cutoff,
    )
    for r in event_rows:
        eid = str(r["event_id"])
        try:
            if extract_run_id(eid) == run_id:
                accounts.add(str(r["account_id"]))
        except ValueError:
            pass

    return sorted(list(accounts))


async def write_account_snapshots(
    pg: Any,
    redis: Any,
    run_id: str,
    bank_id: str,
    snapshot_time: datetime,
    snapshot_kind: str,
    account_ids: Sequence[str],
    settings: Any = None,
) -> int:
    """
    WHAT: Computes point-in-time account features using live Redis state and writes snapshot rows
    to PostgreSQL table account_snapshots.

    WHY: Records behavioral feature vectors for training data generation with the ingestion watermark
    as arrival cutoff, preventing future-leakage and guaranteeing reproducibility.

    Args:
        pg: PostgreSQL asyncpg connection or pool.
        redis: Redis or FakeRedis client containing live account state.
        run_id: 16-hex simulation run identifier.
        bank_id: Tenant commercial bank identifier.
        snapshot_time: Timestamp identifying the snapshot instant.
        snapshot_kind: Cadence label ('daily' or 'full').
        account_ids: Sequence of customer account identifiers to snapshot.
        settings: Optional AccountFeaturesSettings or SnapshotSettings.

    Returns:
        Count of new snapshot rows inserted (0 on idempotent conflict).

    Raises:
        ValueError: If snapshot_kind is invalid.
        NoWatermarkError: If no watermark is available for bank_id.
    """
    if snapshot_kind not in ("daily", "full"):
        raise ValueError(f"Invalid snapshot_kind {snapshot_kind!r}; must be 'daily' or 'full'")

    arrival_cutoff = get_bank_watermark(redis, bank_id)
    if arrival_cutoff is None:
        raise NoWatermarkError(f"No watermark available for bank {bank_id!r}")

    acct_settings: AccountFeaturesSettings
    if isinstance(settings, AccountFeaturesSettings):
        acct_settings = settings
    else:
        acct_settings = get_default_settings()

    account_groups = tuple(SUPPORTED_GROUPS)
    snap_time = _ensure_utc(snapshot_time)
    feature_set_version = FEATURE_SET_VERSION
    feature_names_sha = account_feature_names_sha256()
    names = account_feature_names()

    query = """
    INSERT INTO account_snapshots (
        bank_id,
        run_id,
        account_id,
        snapshot_time,
        snapshot_kind,
        arrival_cutoff,
        feature_set_version,
        feature_names_sha256,
        feature_values,
        logged_at
    ) VALUES (
        $1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb, $10
    )
    ON CONFLICT (bank_id, run_id, account_id, snapshot_time, snapshot_kind)
    DO NOTHING
    """

    inserted_count = 0
    now_utc = datetime.now(timezone.utc)

    for acc_id in account_ids:
        feats = compute_account_features(
            client=redis,
            run_id=run_id,
            bank_id=bank_id,
            account_id=acc_id,
            decision_time=snap_time,
            arrival_cutoff=arrival_cutoff,
            settings=acct_settings,
            groups=account_groups,
            screener=None,
        )
        ordered_vals = [feats[name] for name in names]
        vals_json = orjson.dumps(ordered_vals).decode("utf-8")

        res = await pg.execute(
            query,
            bank_id,
            run_id,
            acc_id,
            snap_time,
            snapshot_kind,
            arrival_cutoff,
            feature_set_version,
            feature_names_sha,
            vals_json,
            now_utc,
        )
        parts = res.split()
        if len(parts) == 3 and parts[0] == "INSERT":
            inserted_count += int(parts[2])

    return inserted_count
