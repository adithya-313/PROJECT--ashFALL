"""
WHAT: Unit tests for Step 3 (banks, ledgers, atomic payments, and invariant checkers).
WHY: Verifies T6 (ledger limit check), T7 (atomic payments), T8 (cross-bank entries & privacy),
     T9 (conservation of money), T10 (checkers and negative corrupted tests), and T11 (bank properties).
ISSUE IDs: E4 (bank differences), E2 (UPI limit enforcement).
TERM: atomic payment: A financial operation that either executes all debit and credit legs completely or changes nothing.
TERM: per-bank visibility: An information boundary where a bank's ledger only answers for its own accounts.
"""

from dataclasses import dataclass
import numpy as np
import pytest

from world.banks import Bank, build_banks
from world.checks import (
    check_balances_equal_entry_sums,
    check_money_conserved,
    check_no_negative_balances,
)
from world.config_reader import read_world_ranges
from world.ledger import AccountNotInThisBank, Ledger, LedgerEntry
from world.payments import PaymentResult, make_payment


@dataclass
class SimpleWorldStub:
    """
    WHAT: A minimal test container holding banks and ledgers for payment testing.
    WHY: Allows testing make_payment before the full build_world pipeline is constructed in Step 5.
    """

    banks: list[Bank]
    ledgers: list[Ledger]


def create_test_setup() -> SimpleWorldStub:
    """
    WHAT: Creates two ledgers with 100,000 paise limit and two test accounts each.
    WHY: Provides a deterministic fixture for payment and ledger tests.
    """
    limit_paise = 100000
    ledger0 = Ledger(bank_index=0, upi_limit_paise=limit_paise)
    ledger1 = Ledger(bank_index=1, upi_limit_paise=limit_paise)

    ledger0.add_account("acc_0_a", opening_balance_paise=50000)
    ledger0.add_account("acc_0_b", opening_balance_paise=20000)

    ledger1.add_account("acc_1_a", opening_balance_paise=30000)
    ledger1.add_account("acc_1_b", opening_balance_paise=10000)

    banks = [
        Bank(bank_index=0, kind="large_public", customer_share=0.5, weak_kyc_share=0.1, hold_delay_factor=3.0),
        Bank(bank_index=1, kind="private", customer_share=0.5, weak_kyc_share=0.05, hold_delay_factor=1.0),
    ]
    return SimpleWorldStub(banks=banks, ledgers=[ledger0, ledger1])


def test_t6_ledger_refuses_leg_above_limit_directly() -> None:
    """
    WHAT: Tests that Ledger.post_debit and post_credit raise ValueError when called with amount > upi_limit_paise.
    WHY: Fulfills test requirement T6 ensuring the ledger enforces the limit at the storage layer.
    """
    ledger = Ledger(bank_index=0, upi_limit_paise=50000)
    ledger.add_account("acc_test", opening_balance_paise=100000)

    with pytest.raises(ValueError, match="limit"):
        ledger.post_debit("acc_test", amount_paise=50001, minute=1, transfer_id="tx_1")

    with pytest.raises(ValueError, match="limit"):
        ledger.post_credit("acc_test", amount_paise=50001, minute=1, transfer_id="tx_2")


def test_t7_payments_reject_invalid_amount_and_unknown_account() -> None:
    """
    WHAT: Tests that payments with zero amount or non-existent accounts are rejected atomically.
    WHY: Fulfills first half of T7 ensuring input errors cause no state changes.
    """
    world = create_test_setup()
    l0, l1 = world.ledgers[0], world.ledgers[1]
    init_l0, init_l1 = list(l0.entries), list(l1.entries)

    res1 = make_payment(world, "acc_0_a", "acc_1_a", amount_paise=0, minute=1)
    assert not res1.success and res1.reason == "invalid_amount"

    res2 = make_payment(world, "acc_0_a", "unknown_user", amount_paise=1000, minute=2)
    assert not res2.success and res2.reason == "unknown_account"

    assert l0.entries == init_l0 and l1.entries == init_l1


def test_t7_payments_reject_over_limit_and_insufficient_funds() -> None:
    """
    WHAT: Tests that over-limit and overdrawn payments leave balances and entries exactly unchanged.
    WHY: Fulfills second half of T7 ensuring business rule violations do not modify ledgers.
    """
    world = create_test_setup()
    l0, l1 = world.ledgers[0], world.ledgers[1]
    init_l0, init_l1 = list(l0.entries), list(l1.entries)

    res3 = make_payment(world, "acc_0_a", "acc_1_a", amount_paise=150000, minute=3)
    assert not res3.success and res3.reason == "over_upi_limit"

    res4 = make_payment(world, "acc_0_a", "acc_1_a", amount_paise=90000, minute=4)
    assert not res4.success and res4.reason == "insufficient_balance"

    assert l0.entries == init_l0 and l1.entries == init_l1
    assert l0.balance_of("acc_0_a") == 50000 and l1.balance_of("acc_1_a") == 30000


def test_t8_cross_bank_payment_legs_and_privacy() -> None:
    """
    WHAT: Tests cross-bank payment posts 1 debit in sender and 1 credit in receiver, and enforces bank privacy.
    WHY: Fulfills test requirement T8 proving multi-bank ledger separation and AccountNotInThisBank.
    """
    world = create_test_setup()
    l0 = world.ledgers[0]
    l1 = world.ledgers[1]

    res = make_payment(world, "acc_0_a", "acc_1_a", amount_paise=5000, minute=10)
    assert res.success
    assert res.reason == "ok"
    assert l0.balance_of("acc_0_a") == 45000
    assert l1.balance_of("acc_1_a") == 35000

    # Sender has debit leg, receiver has credit leg
    assert l0.entries[-1].kind == "transfer_debit"
    assert l0.entries[-1].amount_paise == -5000
    assert l1.entries[-1].kind == "transfer_credit"
    assert l1.entries[-1].amount_paise == 5000

    # Privacy: asking l0 about l1's account raises AccountNotInThisBank
    with pytest.raises(AccountNotInThisBank):
        l0.balance_of("acc_1_a")
    with pytest.raises(AccountNotInThisBank):
        l1.balance_of("acc_0_a")


def test_t9_conservation_across_random_payments() -> None:
    """
    WHAT: Runs hundreds of accepted and rejected payments and checks total money equals opening balance.
    WHY: Fulfills test requirement T9 proving strict conservation of money across all banks.
    """
    world = create_test_setup()
    opening_total = sum(l.balance_of(acc) for l in world.ledgers for acc in l.accounts)
    rng = np.random.default_rng(777)
    all_accounts = ["acc_0_a", "acc_0_b", "acc_1_a", "acc_1_b"]

    for m in range(500):
        sender = str(rng.choice(all_accounts))
        receiver = str(rng.choice(all_accounts))
        amount = int(rng.integers(1, 150000))
        make_payment(world, sender, receiver, amount, minute=m)

    current_total = sum(l.balance_of(acc) for l in world.ledgers for acc in l.accounts)
    assert current_total == opening_total


def test_t10_checkers_pass_and_detect_corruptions() -> None:
    """
    WHAT: Tests that invariant checkers pass on valid ledgers and fail on corrupted test states.
    WHY: Fulfills test requirement T10 verifying that checkers detect negative balances, drift, and leakage.
    """
    world = create_test_setup()
    total_opening = sum(l.balance_of(acc) for l in world.ledgers for acc in l.accounts)

    assert check_no_negative_balances(world.ledgers) == []
    assert check_balances_equal_entry_sums(world.ledgers) == []
    assert check_money_conserved(world.ledgers, total_opening) == []

    # Corrupt negative balance
    world.ledgers[0].balances["acc_0_a"] = -100
    assert len(check_no_negative_balances(world.ledgers)) > 0

    # Corrupt stored balance vs entries sum
    world.ledgers[0].balances["acc_0_a"] = 50000
    world.ledgers[0].balances["acc_0_b"] = 99999
    assert len(check_balances_equal_entry_sums(world.ledgers)) > 0

    # Corrupt money conservation
    assert len(check_money_conserved(world.ledgers, total_opening)) > 0


def test_t11_banks_configuration_and_ordering() -> None:
    """
    WHAT: Verifies build_banks creates three banks with correct kinds, shares summing to 1, and ordered factors.
    WHY: Fulfills test requirement T11 ensuring multi-bank ecosystem matches empirical assumptions.
    """
    ranges = read_world_ranges()
    rng = np.random.default_rng(2026)
    banks = build_banks(ranges, rng, range_name="training")

    assert len(banks) == 3
    kinds = [b.kind for b in banks]
    assert kinds == ["large_public", "private", "payments_bank"]

    # Shares sum to 1 within 1e-9
    total_share = sum(b.customer_share for b in banks)
    assert abs(total_share - 1.0) < 1e-9

    # Weak KYC ordering: payments_bank > large_public > private
    pub_b = next(b for b in banks if b.kind == "large_public")
    priv_b = next(b for b in banks if b.kind == "private")
    pay_b = next(b for b in banks if b.kind == "payments_bank")
    assert pay_b.weak_kyc_share >= pub_b.weak_kyc_share >= priv_b.weak_kyc_share

    # Hold delay factors: private == 1.0, pub >= 1.0, pay between 1.0 and pub
    assert priv_b.hold_delay_factor == 1.0
    assert pub_b.hold_delay_factor >= 1.0
    assert priv_b.hold_delay_factor <= pay_b.hold_delay_factor <= pub_b.hold_delay_factor
