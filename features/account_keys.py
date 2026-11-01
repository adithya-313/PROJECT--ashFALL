# File: features/account_keys.py
# Purpose: Level 15a namespaced Redis key builders, run-ID validation, and configuration loading.
# Issue: Level 15a account feature updater (plan Task 1, C4, K2).
"""
WHAT: Provides key generation routines for Level 15 account feature structures in Redis,
run identifier extraction and validation from payment/event IDs, and settings loading from
config/account_features.yaml.

WHY: Enforces strict tenant and run isolation across concurrent simulation runs. Namespacing
every Redis key under {prefix}:{run_id}:{bank_id}:{account_id}:<suffix> ensures zero state bleed
between runs, banks, or customer accounts, and prevents key collision with Level 4 velocity keys.
"""

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

# Hexadecimal character set for run ID validation.
HEX_DIGITS: frozenset[str] = frozenset("0123456789abcdef")


@dataclass(frozen=True)
class AccountFeaturesSettings:
    """
    Configuration parameters for account feature updating, Redis key lifecycle, and polling.

    Attributes:
        windows_seconds: Sliding look-back window intervals (e.g. 1h, 24h, 7d).
        retention_seconds: Maximum look-back horizon; older records are pruned.
        allowed_lateness_seconds: Maximum duration events may arrive late before being pruned.
        key_expiry_margin_seconds: Additional TTL margin added to retention for key expiry.
        key_prefix: Root Redis namespace prefix for account feature keys.
        run_id_hex_length: Required character length of simulation run identifiers.
        account_event_batch_size: Maximum rows read per table poll of bank_account_events.
        account_event_overlap_rows: Tail overlap rows re-read on every poll to prevent commit skips.
    """

    windows_seconds: tuple[int, ...]
    retention_seconds: int
    allowed_lateness_seconds: int
    key_expiry_margin_seconds: int
    key_prefix: str
    run_id_hex_length: int
    account_event_batch_size: int
    account_event_overlap_rows: int
    summary_horizon_days: int = 62
    routine_life_warm_up_days: int = 62
    fingerprint_warning_count: int = 10
    routine_cycle_days: int = 30
    routine_rhythm_tolerance_days: int = 5
    routine_min_history_days: int = 35
    machine_timing_min_payments: int = 5
    device_change_window_seconds: int = 604800
    network_max_hops: int = 3
    network_max_nodes_visited: int = 5000
    commit_grace: int = 5

    @property
    def expiry_seconds(self) -> int:
        """Total Redis TTL in seconds: retention horizon + allowed lateness + housekeeping margin."""
        return self.retention_seconds + self.allowed_lateness_seconds + self.key_expiry_margin_seconds

    @property
    def summary_expiry_seconds(self) -> int:
        """Total Redis TTL for long-lived summary structures: horizon + lateness + housekeeping margin."""
        return (self.summary_horizon_days * 86400) + self.allowed_lateness_seconds + self.key_expiry_margin_seconds


def load_account_features_settings(config_path: Path | None = None) -> AccountFeaturesSettings:
    """
    WHAT: Loads account feature settings from config/account_features.yaml and resolves
    run_id_hex_length from config/interactive_runner.yaml.

    WHY: Prevents hardcoded magic numbers by reading operational thresholds directly from
    declarative, validated configuration files.

    Args:
        config_path: Optional explicit Path to config/account_features.yaml.

    Returns:
        Immutable AccountFeaturesSettings instance.

    Raises:
        FileNotFoundError: If configuration files cannot be located on disk.
        ValueError: If configuration values fail schema or type validation.
    """
    root_dir = Path(__file__).resolve().parent.parent
    if config_path is None:
        config_path = root_dir / "config" / "account_features.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)

    if not isinstance(raw_cfg, dict) or "account_features" not in raw_cfg:
        raise ValueError(f"Missing 'account_features' root block in {config_path}")

    block = raw_cfg["account_features"]

    # Windows seconds list
    win_block = block["windows_seconds"]
    raw_windows = win_block.get("value") or win_block.get("values")
    if not isinstance(raw_windows, list) or not raw_windows:
        raise ValueError("windows_seconds must be a non-empty list of integers")
    windows_seconds = tuple(int(w) for w in raw_windows)

    # Retention seconds
    retention_seconds = int(block["retention_seconds"]["value"])

    # Allowed lateness seconds
    allowed_lateness_seconds = int(block.get("allowed_lateness_seconds", {}).get("value", 86400))

    # Key expiry margin
    key_expiry_margin_seconds = int(block["key_expiry_margin_seconds"]["value"])

    # Key prefix
    key_prefix = str(block["key_prefix"]["value"]).strip()
    if not key_prefix:
        raise ValueError("key_prefix cannot be empty")

    # Run ID length dynamically loaded from interactive_runner.yaml (per design rule)
    run_id_source = block["run_id_source"]
    runner_cfg_rel = run_id_source.get("path", "config/interactive_runner.yaml")
    runner_cfg_path = root_dir / runner_cfg_rel
    with open(runner_cfg_path, "r", encoding="utf-8") as rf:
        runner_raw = yaml.safe_load(rf)
    run_id_hex_length = int(runner_raw["interactive_runner"]["run_id_hex_length"])

    # Account event batch size and overlap
    account_event_batch_size = int(block["account_event_batch_size"]["value"])
    account_event_overlap_rows = int(block.get("account_event_overlap_rows", {}).get("value", 200))

    summary_horizon_days = int(block.get("summary_horizon_days", {}).get("value", 62))
    routine_life_warm_up_days = int(block.get("routine_life_warm_up_days", {}).get("value", 62))
    fingerprint_warning_count = int(block.get("fingerprint_warning_count", {}).get("value", 10))
    routine_cycle_days = int(block.get("routine_cycle_days", {}).get("value", 30))
    routine_rhythm_tolerance_days = int(block.get("routine_rhythm_tolerance_days", {}).get("value", 5))
    routine_min_history_days = int(block.get("routine_min_history_days", {}).get("value", 35))
    machine_timing_min_payments = int(block.get("machine_timing_min_payments", {}).get("value", 5))
    device_change_window_seconds = int(block.get("device_change_window_seconds", {}).get("value", 604800))
    network_max_hops = int(block.get("network_max_hops", {}).get("value", 3))
    network_max_nodes_visited = int(block.get("network_max_nodes_visited", {}).get("value", 5000))
    commit_grace = int(block.get("commit_grace", {}).get("value", 5))

    return AccountFeaturesSettings(
        windows_seconds=windows_seconds,
        retention_seconds=retention_seconds,
        allowed_lateness_seconds=allowed_lateness_seconds,
        key_expiry_margin_seconds=key_expiry_margin_seconds,
        key_prefix=key_prefix,
        run_id_hex_length=run_id_hex_length,
        account_event_batch_size=account_event_batch_size,
        account_event_overlap_rows=account_event_overlap_rows,
        summary_horizon_days=summary_horizon_days,
        routine_life_warm_up_days=routine_life_warm_up_days,
        fingerprint_warning_count=fingerprint_warning_count,
        routine_cycle_days=routine_cycle_days,
        routine_rhythm_tolerance_days=routine_rhythm_tolerance_days,
        routine_min_history_days=routine_min_history_days,
        machine_timing_min_payments=machine_timing_min_payments,
        device_change_window_seconds=device_change_window_seconds,
        network_max_hops=network_max_hops,
        network_max_nodes_visited=network_max_nodes_visited,
        commit_grace=commit_grace,
    )


@lru_cache(maxsize=1)
def get_default_settings() -> AccountFeaturesSettings:
    """Cached loader returning default repository AccountFeaturesSettings."""
    return load_account_features_settings()


def extract_run_id(identifier: str) -> str:
    """
    WHAT: Extracts and verifies the simulation run identifier from a payment ID or account event ID.

    WHY: Every payment and account event emitted by the simulation engine prefixes its identifier
    with the 16-hex run identifier followed by a hyphen '-'. Extracting and validating this prefix
    ensures features never mix state across distinct simulation runs.

    Args:
        identifier: A hyphen-delimited ID string (e.g. '66db9aa439b092ee-000042').

    Returns:
        The extracted lowercase hexadecimal run ID string.

    Raises:
        ValueError: If identifier lacks a hyphen, or the prefix length differs from
            run_id_hex_length, or contains uppercase or non-hexadecimal characters.
    """
    if not isinstance(identifier, str) or "-" not in identifier:
        raise ValueError(f"Identifier {identifier!r} missing '-' delimiter to extract run ID")

    run_id = identifier.split("-", 1)[0]
    expected_length = get_default_settings().run_id_hex_length

    if len(run_id) != expected_length:
        raise ValueError(
            f"Identifier {identifier!r} has invalid run ID length {len(run_id)}; "
            f"expected exactly {expected_length} characters"
        )

    # Must be strictly lowercase hexadecimal characters (0-9, a-f)
    if not all(c in HEX_DIGITS for c in run_id):
        raise ValueError(
            f"Identifier {identifier!r} has invalid run ID {run_id!r}; "
            "must contain only lowercase hexadecimal digits (0-9, a-f)"
        )

    return run_id


def _validate_key_component(name: str, value: str) -> None:
    """Validates that a key segment is a non-empty string without colon delimiters."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"Key component {name!r} cannot be empty; got {value!r}")
    if ":" in value:
        raise ValueError(f"Key component {name!r} cannot contain ':' separator; got {value!r}")


def _build_key(run_id: str, bank_id: str, account_id: str, suffix: str) -> str:
    """
    Internal helper to construct a fully qualified Redis key under the configured prefix.
    Format: {key_prefix}:{run_id}:{bank_id}:{account_id}:{suffix}
    """
    _validate_key_component("run_id", run_id)
    _validate_key_component("bank_id", bank_id)
    _validate_key_component("account_id", account_id)
    _validate_key_component("suffix", suffix)

    prefix = get_default_settings().key_prefix
    _validate_key_component("key_prefix", prefix)
    return f"{prefix}:{run_id}:{bank_id}:{account_id}:{suffix}"


def side_times_key(run_id: str, bank_id: str, account_id: str) -> str:
    """
    Redis key for the sorted set (ZSET) tracking transaction side event timestamps.
    Score: Unix epoch seconds; Member: 'payment_id:side'.
    """
    return _build_key(run_id, bank_id, account_id, "side_times")


def side_records_key(run_id: str, bank_id: str, account_id: str) -> str:
    """
    Redis key for the hash (HASH) storing serialized PaymentSideRecord payloads.
    Field: 'payment_id:side'; Value: JSON payload bytes.
    """
    return _build_key(run_id, bank_id, account_id, "side_records")


def event_times_key(run_id: str, bank_id: str, account_id: str) -> str:
    """
    Redis key for the sorted set (ZSET) tracking bank account lifecycle event timestamps.
    Score: Unix epoch seconds; Member: event_id.
    """
    return _build_key(run_id, bank_id, account_id, "event_times")


def event_records_key(run_id: str, bank_id: str, account_id: str) -> str:
    """
    Redis key for the hash (HASH) storing serialized AccountEventRecord payloads.
    Field: event_id; Value: JSON payload bytes.
    """
    return _build_key(run_id, bank_id, account_id, "event_records")


def withdrawal_times_key(run_id: str, bank_id: str, account_id: str) -> str:
    """
    Redis key for the sorted set (ZSET) tracking cash withdrawal event timestamps.
    Score: Unix epoch seconds; Member: withdrawal_id.
    """
    return _build_key(run_id, bank_id, account_id, "withdrawal_times")


def withdrawal_records_key(run_id: str, bank_id: str, account_id: str) -> str:
    """
    Redis key for the hash (HASH) storing serialized WithdrawalRecord payloads.
    Field: withdrawal_id; Value: JSON payload bytes.
    """
    return _build_key(run_id, bank_id, account_id, "withdrawal_records")


def marks_key(run_id: str, bank_id: str, account_id: str) -> str:
    """
    Redis key for the sorted set (ZSET) holding watermark timestamps for the account.
    Members: 'first_event', 'last_event', 'arrival_watermark'.
    """
    return _build_key(run_id, bank_id, account_id, "marks")


def active_days_key(run_id: str, bank_id: str, account_id: str) -> str:
    """
    Redis key for the sorted set (ZSET) tracking lifetime money activity days (payment sides and withdrawals).
    Score: IST day ordinal integer; Member: IST day ordinal integer string.
    Lifetime key: lives as long as the run's namespace; removed by namespace cleanup, never by a short TTL.
    """
    return _build_key(run_id, bank_id, account_id, "active_days")


def opened_at_key(run_id: str, bank_id: str, account_id: str) -> str:
    """
    Redis key for the sorted set (ZSET) tracking the account opening timestamp from account_opened event.
    Score: Unix epoch seconds of event_time; Member: 'opened_at'.
    Lifetime key: lives as long as the run's namespace; removed by namespace cleanup, never by a short TTL.
    """
    return _build_key(run_id, bank_id, account_id, "opened_at")


def days_index_key(run_id: str, bank_id: str, account_id: str) -> str:
    """
    Redis key for the sorted set (ZSET) indexing active summary calendar days for an account.
    Score: IST day ordinal integer; Member: IST day ordinal integer string.
    """
    return _build_key(run_id, bank_id, account_id, "days_index")


def day_totals_key(run_id: str, bank_id: str, account_id: str, day: int) -> str:
    """
    Redis key for the hash (HASH) storing aggregate transaction totals for a single IST calendar day.
    Fields: credit_count, credit_paise, debit_count, debit_paise, withdrawal_count, withdrawal_paise,
    failed_attempt_count.
    """
    return _build_key(run_id, bank_id, account_id, f"day_totals_{day}")


def day_seen_key(run_id: str, bank_id: str, account_id: str, day: int) -> str:
    """
    Redis key for the set (SET) tracking seen member identifiers for a single IST calendar day.
    Members: 'payment_id:side' or 'withdrawal_id'.
    """
    return _build_key(run_id, bank_id, account_id, f"day_seen_{day}")


def counterparty_days_key(run_id: str, bank_id: str, account_id: str, direction: str, counterparty: str) -> str:
    """
    Redis key for the sorted set (ZSET) tracking active days for a specific counterparty in a direction.
    Score: IST day ordinal integer; Member: IST day ordinal integer string.
    """
    _validate_key_component("direction", direction)
    counterparty_token = hashlib.sha256(counterparty.encode("utf-8")).hexdigest()[:16]
    return _build_key(run_id, bank_id, account_id, f"cp_days_{direction}_{counterparty_token}")


def counterparty_index_key(run_id: str, bank_id: str, account_id: str, direction: str) -> str:
    """
    Redis key for the sorted set (ZSET) indexing all counterparties for an account in a given direction.
    Score: latest IST day ordinal integer (ZADD GT); Member: opaque counterparty string ('bank:vpa').
    """
    _validate_key_component("direction", direction)
    return _build_key(run_id, bank_id, account_id, f"cp_index_{direction}")


def fingerprints_key(run_id: str, bank_id: str, account_id: str) -> str:
    """
    Redis key for the sorted set (ZSET) tracking mobile device fingerprints observed on an account.
    Score: Unix epoch seconds of earliest availability max(event_time, arrival_time); Member: fingerprint string.
    Lifetime key: lives as long as the run's namespace; removed by namespace cleanup, never by a short TTL.
    """
    return _build_key(run_id, bank_id, account_id, "fingerprints")


def home_region_key(run_id: str, bank_id: str, account_id: str) -> str:
    """
    Redis key for the sorted set (ZSET) tracking the customer home region from account_opened event.
    Score: Unix epoch seconds of earliest availability max(event_time, arrival_time) (ZADD LT);
    Member: region identifier string.
    Lifetime key: lives as long as the run's namespace; removed by namespace cleanup, never by a short TTL.
    """
    return _build_key(run_id, bank_id, account_id, "home_region")


def account_type_key(run_id: str, bank_id: str, account_id: str) -> str:
    """
    Redis key for the sorted set (ZSET) tracking customer account type (savings, current) from account_opened event.
    Score: Unix epoch seconds of earliest availability max(event_time, arrival_time) (ZADD LT);
    Member: account type string (e.g. 'savings', 'current').
    Lifetime key: lives as long as the run's namespace; removed by namespace cleanup, never by a short TTL.
    """
    return _build_key(run_id, bank_id, account_id, "account_type")


def device_accounts_key(run_id: str, bank_id: str, fingerprint: str) -> str:
    """
    Redis key for the sorted set (ZSET) indexing customer accounts observed on a specific device fingerprint.
    Score: Unix epoch seconds of earliest availability max(event_time, arrival_time) (ZADD LT);
    Member: account_id string.
    Lifetime key: lives as long as the run's namespace; removed by namespace cleanup, never by a short TTL.
    """
    _validate_key_component("run_id", run_id)
    _validate_key_component("bank_id", bank_id)
    prefix = get_default_settings().key_prefix
    _validate_key_component("key_prefix", prefix)
    fp_token = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}:{run_id}:{bank_id}:dev_accs_{fp_token}"


def counterparty_payee_type_key(run_id: str, bank_id: str, account_id: str, direction: str) -> str:
    """
    Redis key for the hash (HASH) tracking verified payee types for counterparties.
    Field: opaque counterparty identifier ('bank:vpa'); Value: payee_type string ('person', 'merchant').
    Lifetime key: lives as long as the run's namespace; removed by namespace cleanup, never by a short TTL.
    """
    _validate_key_component("direction", direction)
    return _build_key(run_id, bank_id, account_id, f"cp_payee_type_{direction}")


def device_changes_key(run_id: str, bank_id: str, account_id: str) -> str:
    """
    Redis key for the sorted set (ZSET) tracking device change event timestamps on an account.
    Score: Unix epoch seconds of availability max(event_time, arrival_time); Member: event_id string.
    Lifetime key: lives as long as the run's namespace; removed by namespace cleanup, never by a short TTL.
    """
    return _build_key(run_id, bank_id, account_id, "device_changes")


def labelled_accounts_key(run_id: str, bank_id: str) -> str:
    """
    Redis key for the sorted set (ZSET) tracking confirmed fraud labelled accounts for a bank.
    Score: Unix epoch seconds of earliest availability max(event_time, arrival_of(row)); Member: account_id.
    Lifetime key: lives as long as the run's namespace; no expiry.
    """
    _validate_key_component("run_id", run_id)
    _validate_key_component("bank_id", bank_id)
    prefix = get_default_settings().key_prefix
    _validate_key_component("key_prefix", prefix)
    return f"{prefix}:{run_id}:{bank_id}:labelled_accounts"


def complained_accounts_key(run_id: str, bank_id: str) -> str:
    """
    Redis key for the sorted set (ZSET) tracking complained accounts for a bank.
    Score: Unix epoch seconds of earliest availability max(received_at, arrival_of(row)); Member: account_id.
    Lifetime key: lives as long as the run's namespace; no expiry.
    """
    _validate_key_component("run_id", run_id)
    _validate_key_component("bank_id", bank_id)
    prefix = get_default_settings().key_prefix
    _validate_key_component("key_prefix", prefix)
    return f"{prefix}:{run_id}:{bank_id}:complained_accounts"


def namespace_pattern(run_id: str, bank_id: str) -> str:
    """
    WHAT: Generates a Redis SCAN pattern matching all keys for a specific run and bank.

    WHY: Enables scoped iteration and cleanup of test and showcase namespaces without
    using the blocking KEYS command or scanning keys belonging to other banks or runs.

    Args:
        run_id: Simulation run identifier.
        bank_id: Commercial bank identifier.

    Returns:
        Pattern string of the form '{key_prefix}:{run_id}:{bank_id}:*'.
    """
    _validate_key_component("run_id", run_id)
    _validate_key_component("bank_id", bank_id)
    prefix = get_default_settings().key_prefix
    _validate_key_component("key_prefix", prefix)
    return f"{prefix}:{run_id}:{bank_id}:*"


WATERMARK_PREFIX = "acctwm:"


def payment_watermark_key(partition: int) -> str:
    """Redis key for Kafka upi.payments partition watermark."""
    return f"{WATERMARK_PREFIX}payments:{partition}"


def payments_partitions_key() -> str:
    """Redis set key holding known Kafka upi.payments partition numbers."""
    return f"{WATERMARK_PREFIX}payments:partitions"


def poller_watermark_key(bank_id: str, source: str) -> str:
    """Redis key for bank-scoped Postgres poller watermark."""
    _validate_key_component("bank_id", bank_id)
    return f"{WATERMARK_PREFIX}{bank_id}:{source}"

