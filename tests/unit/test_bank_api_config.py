# File: tests/unit/test_bank_api_config.py
# Purpose: Proves bank API configuration integrity and bank identity validation without DB.
import pytest

from api.v1.bank_scope import UnknownBankError, bank_scoped_transaction, load_known_banks
from core.config_validator import load_yaml


def test_banks_in_bank_api_yaml_match_nationalsim_banks():
    """banks listed in config/bank_api.yaml exactly equal bank keys in config/nationalsim.yaml."""
    bank_api_data = load_yaml("config/bank_api.yaml")
    nationalsim_data = load_yaml("config/nationalsim.yaml")

    bank_api_banks = set(bank_api_data["bank_api"]["banks"])
    nationalsim_banks = set(nationalsim_data["nationalsim"]["banks"].keys())

    assert bank_api_banks == nationalsim_banks
    assert load_known_banks() == sorted(bank_api_banks)


@pytest.mark.anyio
async def test_unknown_bank_raises_unknown_bank_error_without_database():
    """bank_scoped_transaction raises UnknownBankError immediately without touching database."""
    # Pass a None connection object to prove database is never reached
    with pytest.raises(UnknownBankError) as exc_info:
        async with bank_scoped_transaction(None, "bank_rogue_999"):  # type: ignore[arg-type]
            pass

    assert "bank_rogue_999" in str(exc_info.value)
