"""
WHAT: Unit tests for Step 4 (devices, personas, shops, and people).
WHY: Verifies T11 (bank share among people), T12 (persona distribution & look-alikes),
     and T13 (device uniqueness, Android share, and independence from KYC status).
ISSUE IDs: E4 (bank customer mix), E3 (regions).
TERM: look-alike: A genuine customer whose ordinary transaction pattern resembles a mule account.
"""

import numpy as np

from world.banks import build_banks
from world.config_reader import read_world_ranges
from world.devices import create_device
from world.ledger import Ledger
from world.people import build_people
from world.personas import get_persona
from world.placeholders import ANDROID_SHARE, OS_GENERATIONS, PERSONA_MIX
from world.shops import build_shops


def setup_test_ecosystem(n_people: int, n_shops: int, seed: int = 1234):
    """
    WHAT: Builds banks, ledgers, shops, and people for statistical verification.
    WHY: Shared fixture helper to test multi-entity correlations and shares.
    """
    ranges = read_world_ranges()
    rng = np.random.default_rng(seed)
    banks = build_banks(ranges, rng)
    ledgers = [Ledger(bank_index=b.bank_index, upi_limit_paise=10000000) for b in banks]
    n_regions = 4
    shops = build_shops(n_shops, n_regions, banks, ledgers, rng)
    people = build_people(n_people, n_regions, banks, ledgers, shops, ranges, rng)
    return banks, ledgers, shops, people


def test_t13_devices_properties_and_uniqueness() -> None:
    """
    WHAT: Tests that every person gets exactly one unique device within OS bounds.
    WHY: Fulfills first part of T13 ensuring valid device hardware generation.
    """
    rng = np.random.default_rng(42)
    devices = [create_device(f"dev_{i}", rng) for i in range(100)]
    assert len(devices) == 100

    # Fingerprints are unique
    fingerprints = {d.fingerprint for d in devices}
    assert len(fingerprints) == 100

    # Generations strictly in 1..OS_GENERATIONS
    for d in devices:
        assert 1 <= d.os_generation <= OS_GENERATIONS
        assert d.platform in ("android", "ios")


def test_t13_devices_statistical_distribution_and_independence() -> None:
    """
    WHAT: With 3,000 people, checks Android share matches config and is independent of weak_kyc.
    WHY: Fulfills statistical requirements of T13.
    """
    _, _, _, people = setup_test_ecosystem(n_people=3000, n_shops=20, seed=42)

    # Android share within 0.03 of ANDROID_SHARE
    android_count = sum(1 for p in people if p.device.platform == "android")
    measured_share = android_count / 3000.0
    assert abs(measured_share - ANDROID_SHARE) < 0.03

    # Platform and weak_kyc status are statistically unrelated (difference < 0.05)
    android_people = [p for p in people if p.device.platform == "android"]
    ios_people = [p for p in people if p.device.platform == "ios"]
    android_kyc_rate = sum(1 for p in android_people if p.weak_kyc) / len(android_people)
    ios_kyc_rate = sum(1 for p in ios_people if p.weak_kyc) / len(ios_people)
    assert abs(android_kyc_rate - ios_kyc_rate) < 0.05


def test_t12_personas_mix_and_lookalikes() -> None:
    """
    WHAT: With 3,000 people, checks persona shares match PERSONA_MIX within 0.04 and look-alikes exist.
    WHY: Fulfills distribution check of T12 ensuring proper population representation.
    """
    _, _, _, people = setup_test_ecosystem(n_people=3000, n_shops=20, seed=777)

    # Each persona share within 0.04 of PERSONA_MIX
    for p_name, target_share in PERSONA_MIX.items():
        count = sum(1 for p in people if p.persona_name == p_name)
        measured = count / 3000.0
        assert abs(measured - target_share) < 0.04

    # At least one look-alike exists
    lookalikes = [p for p in people if get_persona(p.persona_name).looks_like_mule_in_normal_life]
    assert len(lookalikes) > 0


def test_t11_banks_people_share_matches_customer_share() -> None:
    """
    WHAT: With 3,000 people, checks each bank's actual share of people is within 0.05 of customer_share.
    WHY: Fulfills people allocation portion of T11.
    """
    banks, _, _, people = setup_test_ecosystem(n_people=3000, n_shops=20, seed=888)

    for bank in banks:
        count = sum(1 for p in people if p.bank_index == bank.bank_index)
        measured = count / 3000.0
        assert abs(measured - bank.customer_share) < 0.05


def test_shops_and_people_opening_balances_in_ledger() -> None:
    """
    WHAT: Tests that shops and people have their accounts and opening balances recorded in ledgers.
    WHY: Verifies that entity setup correctly seeds the underlying financial ledgers.
    """
    _, ledgers, shops, people = setup_test_ecosystem(n_people=50, n_shops=10, seed=999)

    for s in shops:
        ledger = ledgers[s.bank_index]
        assert s.account_id in ledger.accounts
        assert ledger.balance_of(s.account_id) > 0

    for p in people:
        ledger = ledgers[p.bank_index]
        assert p.account_id in ledger.accounts
        assert ledger.balance_of(p.account_id) == p.opening_balance_paise
