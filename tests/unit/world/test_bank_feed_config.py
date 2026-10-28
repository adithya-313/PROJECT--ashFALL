# File: tests/unit/world/test_bank_feed_config.py
# Purpose: Unit tests verifying bank-feed projection configuration parsing and strict validation.
"""
WHAT: Tests the loader and validation logic of world/bank_feed_config.py.
It verifies that valid configuration is loaded cleanly, that missing failure domains and
unknown bank identifiers trigger explicit ValueErrors, and that allowed account event kinds
match the bank API schema definitions in api/v1/models.py.

WHY: Strict validation protects downstream projection pipelines against mismatched failure
domain definitions, unauthorized bank identifiers, and unapproved account event kinds.
"""

from pathlib import Path
from typing import get_args

import pytest
import yaml

from api.v1.models import AccountEventKind
from core.config_validator import load_yaml
from world.bank_feed_config import DEFAULT_BANK_FEED_CONFIG_PATH, BankFeedConfig, load_bank_feed_config
from world.failures import FAILURE_DOMAINS


def test_loads_real_configuration_file() -> None:
    """Verifies that the canonical config/bank_feed.yaml file loads cleanly into a BankFeedConfig record."""
    cfg = load_bank_feed_config()
    assert isinstance(cfg, BankFeedConfig)
    assert cfg.bank_ids_by_index == {0: "bank_0", 1: "bank_1", 2: "bank_2"}
    assert cfg.payee_type_by_account_prefix["acc_person_"] == "person"
    assert cfg.payee_type_by_account_prefix["acc_shop_"] == "merchant"
    assert cfg.channel_value == "upi_app"

    # Verify R7 visibility rules: sender sees all 5 domains, receiver sees only receiver_bank.
    assert set(cfg.failed_payment_visibility.keys()) == set(FAILURE_DOMAINS)
    for domain in FAILURE_DOMAINS:
        assert cfg.failed_payment_visibility[domain]["sender"] is True
        if domain == "receiver_bank":
            assert cfg.failed_payment_visibility[domain]["receiver"] is True
        else:
            assert cfg.failed_payment_visibility[domain]["receiver"] is False

    assert cfg.device_event_kind_map["phone_change"] == "device_binding_change"
    assert cfg.device_event_kind_map["reinstall"] == "app_reinstall"
    assert cfg.device_event_kind_map["os_update"] == "os_update"
    assert cfg.device_event_kind_map["sim_swap"] == "sim_swap"


def test_visibility_table_missing_one_domain_is_rejected(tmp_path: Path) -> None:
    """Ensures that omitting any canonical failure domain from the visibility table raises a ValueError."""
    raw_data = load_yaml(DEFAULT_BANK_FEED_CONFIG_PATH)
    # Remove one required failure domain from the visibility configuration dictionary.
    del raw_data["bank_feed"]["failed_payment_visibility"]["domains"]["customer"]

    corrupted_path = tmp_path / "corrupted_feed.yaml"
    corrupted_path.write_text(yaml.safe_dump(raw_data), encoding="utf-8")

    with pytest.raises(ValueError, match="failed_payment_visibility must name exactly"):
        load_bank_feed_config(corrupted_path)


def test_unknown_bank_in_feed_mapping_is_rejected(tmp_path: Path) -> None:
    """Ensures that assigning an unregistered bank identifier in bank_ids_by_index raises a ValueError."""
    raw_data = load_yaml(DEFAULT_BANK_FEED_CONFIG_PATH)
    # Assign an unknown bank identifier not registered in config/bank_api.yaml.
    raw_data["bank_feed"]["bank_ids_by_index"]["mapping"][0] = "bank_99_unregistered"

    corrupted_path = tmp_path / "unknown_bank_feed.yaml"
    corrupted_path.write_text(yaml.safe_dump(raw_data), encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown bank id 'bank_99_unregistered'"):
        load_bank_feed_config(corrupted_path)


def test_allowed_kind_list_matches_api_models_account_event_kind() -> None:
    """Verifies that allowed account event kinds in config match api.v1.models.AccountEventKind."""
    cfg = load_bank_feed_config()
    api_kinds = set(get_args(AccountEventKind))
    config_kinds = set(cfg.allowed_account_event_kinds)
    assert config_kinds == api_kinds
