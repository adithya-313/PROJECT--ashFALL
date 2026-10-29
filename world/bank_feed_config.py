# File: world/bank_feed_config.py
# Purpose: Configuration model and strict validator for projecting simulation records to bank feeds.
"""
WHAT: Loads and strictly validates configuration parameters governing the bank feed projection.
It reads bank mappings, account prefix classifications, failure domain visibility tables,
and device event kind mappings from config/bank_feed.yaml and validates them against known banks.

WHY: Ground rule 29 requires decoupling commercial bank API representations from internal simulator
models. This configuration controls how simulator-internal failure domains, account prefixes, and
telemetry hardware events map cleanly into bank-visible data structures without leakage.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.config_validator import load_yaml
from world.failures import FAILURE_DOMAINS

DEFAULT_BANK_FEED_CONFIG_PATH: Path = Path("config/bank_feed.yaml")
DEFAULT_BANK_API_CONFIG_PATH: Path = Path("config/bank_api.yaml")


@dataclass(frozen=True)
class BankFeedConfig:
    """
    WHAT: Immutable container storing verified bank-feed projection configuration settings.
    It encapsulates bank identifiers, payee type prefixes, default channel tokens, failure domain
    visibility tables, device event mapping rules, and supported account event kinds.

    WHY: Freezing configuration parameters prevents accidental runtime mutation during simulation runs
    and guarantees bitwise deterministic projections across repeated invocations.
    """

    bank_ids_by_index: dict[int, str]
    payee_type_by_account_prefix: dict[str, str]
    channel_value: str
    decision_timeout_seconds: float
    failed_payment_visibility: dict[str, dict[str, bool]]
    device_event_kind_map: dict[str, str]
    unmapped_gang_device_kinds: tuple[str, ...]
    allowed_account_event_kinds: tuple[str, ...]
    emit_opening_for_existing_accounts: bool


def load_bank_feed_config(path: Path | str | None = None) -> BankFeedConfig:
    """
    WHAT: Reads config/bank_feed.yaml and returns a verified, immutable BankFeedConfig instance.
    It parses YAML mappings, validates all bank identifiers against config/bank_api.yaml, verifies
    that failure visibility precisely matches FAILURE_DOMAINS, and checks mapped account kinds.

    WHY: Strict validation at configuration load time halts execution before running projections if
    new simulation domains or unregistered banks are introduced without proper feed rules.

    Args:
        path: Optional file path to the bank feed YAML configuration; defaults to config/bank_feed.yaml.

    Returns:
        Validated immutable BankFeedConfig instance populated from verified YAML blocks.

    Raises:
        ValueError: If any required section is absent, an unknown bank is specified, failure domains
            do not exactly match FAILURE_DOMAINS, or an unapproved account-event kind is mapped.
    """
    target_path = Path(path) if path is not None else DEFAULT_BANK_FEED_CONFIG_PATH
    if not target_path.is_file():
        raise ValueError(f"Bank feed configuration file not found: {target_path}")

    raw_data: Any = load_yaml(target_path)
    if not isinstance(raw_data, dict) or "bank_feed" not in raw_data:
        raise ValueError(f"Configuration file '{target_path}' must contain a top-level 'bank_feed' key")

    feed_data = raw_data["bank_feed"]

    # 1. Load known banks from config/bank_api.yaml to enforce bank identifier cross-validation.
    if not DEFAULT_BANK_API_CONFIG_PATH.is_file():
        raise ValueError(f"Bank API configuration file not found: {DEFAULT_BANK_API_CONFIG_PATH}")
    bank_api_raw: Any = load_yaml(DEFAULT_BANK_API_CONFIG_PATH)
    if not isinstance(bank_api_raw, dict) or "bank_api" not in bank_api_raw:
        raise ValueError("Bank API configuration must contain a top-level 'bank_api' key")
    known_banks = set(bank_api_raw["bank_api"].get("banks", []))

    # 2. Extract and validate bank_ids_by_index mapping against known commercial banks.
    raw_bank_mapping = feed_data.get("bank_ids_by_index", {}).get("mapping", {})
    if not isinstance(raw_bank_mapping, dict) or not raw_bank_mapping:
        raise ValueError("bank_ids_by_index must define a non-empty mapping")

    bank_ids_by_index: dict[int, str] = {}
    for idx_key, bank_id_val in raw_bank_mapping.items():
        parsed_idx = int(idx_key)
        parsed_bank_id = str(bank_id_val)
        if parsed_bank_id not in known_banks:
            raise ValueError(
                f"Unknown bank id '{parsed_bank_id}' at index {parsed_idx}; expected one of {sorted(known_banks)}"
            )
        bank_ids_by_index[parsed_idx] = parsed_bank_id

    # 3. Extract and validate payee_type_by_account_prefix mapping.
    raw_prefix_mapping = feed_data.get("payee_type_by_account_prefix", {}).get("mapping", {})
    if not isinstance(raw_prefix_mapping, dict) or not raw_prefix_mapping:
        raise ValueError("payee_type_by_account_prefix must define a non-empty mapping")
    payee_type_by_account_prefix = {str(prefix): str(ptype) for prefix, ptype in raw_prefix_mapping.items()}

    # 4. Extract default channel identifier token and decision timeout.
    channel_token = feed_data.get("channel_value", {}).get("value")
    if not isinstance(channel_token, str) or not channel_token:
        raise ValueError("channel_value.value must be a non-empty string")

    raw_decision_timeout = feed_data.get("decision_timeout_seconds", {}).get("value")
    if not isinstance(raw_decision_timeout, (int, float)) or raw_decision_timeout <= 0:
        raise ValueError("decision_timeout_seconds.value must be a positive number")
    decision_timeout_seconds = float(raw_decision_timeout)

    # 5. Extract and validate failed_payment_visibility against world FAILURE_DOMAINS.
    raw_visibility = feed_data.get("failed_payment_visibility", {}).get("domains", {})
    if not isinstance(raw_visibility, dict):
        raise ValueError("failed_payment_visibility must define a domains dictionary")

    expected_domains = set(FAILURE_DOMAINS)
    configured_domains = set(raw_visibility.keys())
    if configured_domains != expected_domains:
        raise ValueError(
            f"failed_payment_visibility must name exactly {sorted(expected_domains)}, got {sorted(configured_domains)}"
        )

    failed_payment_visibility: dict[str, dict[str, bool]] = {}
    for domain_name, visibility_entry in raw_visibility.items():
        if (
            not isinstance(visibility_entry, dict)
            or "sender" not in visibility_entry
            or "receiver" not in visibility_entry
        ):
            raise ValueError(f"Visibility entry for domain '{domain_name}' must specify 'sender' and 'receiver'")
        if not visibility_entry["sender"]:
            raise ValueError(
                f"failed_payment_visibility for '{domain_name}' has sender=false: "
                "the sending bank always sees its own attempt"
            )
        failed_payment_visibility[domain_name] = {
            "sender": bool(visibility_entry["sender"]),
            "receiver": bool(visibility_entry["receiver"]),
        }

    # 6. Extract allowed account event kinds.
    raw_allowed_kinds = feed_data.get("allowed_account_event_kinds", {}).get("kinds", [])
    if not isinstance(raw_allowed_kinds, list) or len(raw_allowed_kinds) != 6:
        raise ValueError(f"allowed_account_event_kinds must specify exactly 6 kinds, got {len(raw_allowed_kinds)}")
    allowed_account_event_kinds = tuple(str(k) for k in raw_allowed_kinds)

    # 7. Extract and validate device_event_kind_map against approved account kinds.
    raw_kind_map = feed_data.get("device_event_kind_map", {}).get("mapping", {})
    if not isinstance(raw_kind_map, dict) or not raw_kind_map:
        raise ValueError("device_event_kind_map must define a non-empty mapping")

    device_event_kind_map: dict[str, str] = {}
    for source_kind, target_kind in raw_kind_map.items():
        str_target = str(target_kind)
        if str_target not in allowed_account_event_kinds:
            raise ValueError(
                f"Mapped target kind '{str_target}' is not in approved account kinds {allowed_account_event_kinds}"
            )
        device_event_kind_map[str(source_kind)] = str_target

    # 8. Extract unmapped gang device kinds tuple.
    raw_unmapped = feed_data.get("unmapped_gang_device_kinds", {}).get("kinds", [])
    if not isinstance(raw_unmapped, list) or not raw_unmapped:
        raise ValueError("unmapped_gang_device_kinds must list at least one unmapped kind")
    unmapped_gang_device_kinds = tuple(str(k) for k in raw_unmapped)

    # 9. Extract emit_opening_for_existing_accounts policy flag.
    raw_emit_opening = feed_data.get("emit_opening_for_existing_accounts", {}).get("value")
    if not isinstance(raw_emit_opening, bool):
        raise ValueError("emit_opening_for_existing_accounts.value must be a boolean")
    emit_opening_for_existing_accounts = bool(raw_emit_opening)

    return BankFeedConfig(
        bank_ids_by_index=bank_ids_by_index,
        payee_type_by_account_prefix=payee_type_by_account_prefix,
        channel_value=channel_token,
        decision_timeout_seconds=decision_timeout_seconds,
        failed_payment_visibility=failed_payment_visibility,
        device_event_kind_map=device_event_kind_map,
        unmapped_gang_device_kinds=unmapped_gang_device_kinds,
        allowed_account_event_kinds=allowed_account_event_kinds,
        emit_opening_for_existing_accounts=emit_opening_for_existing_accounts,
    )
