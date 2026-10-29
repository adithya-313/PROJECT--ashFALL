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
        upi_key_by_account: Whether transactional outbox relays schema-2 events keyed by account.
    """

    windows_seconds: tuple[int, ...]
    retention_seconds: int
    allowed_lateness_seconds: int
    key_expiry_margin_seconds: int
    key_prefix: str
    run_id_hex_length: int
    account_event_batch_size: int
    account_event_overlap_rows: int
    upi_key_by_account: bool

    @property
    def expiry_seconds(self) -> int:
        """Total Redis TTL in seconds: retention horizon + allowed lateness + housekeeping margin."""
        return self.retention_seconds + self.allowed_lateness_seconds + self.key_expiry_margin_seconds


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

    # UPI key by account switch default
    upi_key_by_account = bool(block.get("upi_key_by_account", {}).get("value", True))

    return AccountFeaturesSettings(
        windows_seconds=windows_seconds,
        retention_seconds=retention_seconds,
        allowed_lateness_seconds=allowed_lateness_seconds,
        key_expiry_margin_seconds=key_expiry_margin_seconds,
        key_prefix=key_prefix,
        run_id_hex_length=run_id_hex_length,
        account_event_batch_size=account_event_batch_size,
        account_event_overlap_rows=account_event_overlap_rows,
        upi_key_by_account=upi_key_by_account,
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


def marks_key(run_id: str, bank_id: str, account_id: str) -> str:
    """
    Redis key for the sorted set (ZSET) holding watermark timestamps for the account.
    Members: 'first_event', 'last_event', 'arrival_watermark'.
    """
    return _build_key(run_id, bank_id, account_id, "marks")


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

