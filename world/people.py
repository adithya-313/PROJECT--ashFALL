"""
WHAT: Individual person citizen modeling and population generation pipeline.
WHY: Generates citizen demographics, assigns bank accounts, devices, employers, and funds initial balances.
ISSUE IDs: E2 (amount profiles), E3 (regions), E4 (banks and KYC).
TERM: opening balance: Initial funded deposit placed into an account before any simulation transfers run.
"""

from dataclasses import dataclass
from typing import Any
import numpy as np

from world.amounts import AmountProfile, create_amount_profile
from world.banks import Bank
from world.config_reader import get_parameter_entry
from world.devices import Device, create_device
from world.ledger import Ledger
from world.personas import choose_persona_name, get_persona
from world.placeholders import (
    EMPLOYER_SALARY_MONTHS_COVERED,
    OPENING_BALANCE_MULTIPLE,
    SALARY_MULTIPLE,
)
from world.regions import assign_region
from world.shops import Shop, choose_bank_index


@dataclass
class Person:
    """
    WHAT: Represents a simulated private citizen in the town.
    WHY: Combines identity, device, account, employer link, and spending habits into one coherent entity.
    """

    person_id: str
    persona_name: str
    region: int
    bank_index: int
    account_id: str
    weak_kyc: bool
    device: Device
    amount_profile: AmountProfile
    employer_shop_id: str | None
    opening_balance_paise: int


def assign_bank_and_kyc(banks: list[Bank], rng: np.random.Generator) -> tuple[int, bool]:
    """
    WHAT: Selects a bank index and draws the weak KYC flag based on that bank's KYC share.
    WHY: Couples customer bank distribution with each bank's identity verification rigour.
    """
    b_idx = choose_bank_index(banks, rng)
    weak_kyc = float(rng.uniform(0.0, 1.0)) < banks[b_idx].weak_kyc_share
    return (b_idx, weak_kyc)


def create_single_person(
    index: int,
    n_regions: int,
    banks: list[Bank],
    employer_ids: list[str],
    med_e: dict[str, Any],
    sig_e: dict[str, Any],
    rng: np.random.Generator,
    range_name: str,
) -> Person:
    """
    WHAT: Synthesizes a single Person with assigned demographics, device, bank, and employer.
    WHY: Encapsulates per-individual random attribute generation in an isolated helper.
    """
    p_id = f"person_{index:06d}"
    p_name = choose_persona_name(rng)
    persona = get_persona(p_name)
    b_idx, weak_kyc = assign_bank_and_kyc(banks, rng)
    device = create_device(f"dev_{p_id}", rng)
    profile = create_amount_profile(persona.spend_level, med_e, sig_e, rng, range_name)
    opening_paise = OPENING_BALANCE_MULTIPLE * profile.median_paise
    employer_id = str(rng.choice(employer_ids)) if p_name == "salaried_employee" else None

    return Person(
        p_id, p_name, assign_region(n_regions, rng), b_idx, f"acc_{p_id}",
        weak_kyc, device, profile, employer_id, opening_paise
    )


def fund_employer_shops(shops: list[Shop], people: list[Person], ledgers: list[Ledger]) -> None:
    """
    WHAT: Tops up employer shop balances so each can fund all assigned employee salaries for 24 months.
    WHY: Prevents employers from running out of funds during multi-month routine payroll runs.
    """
    employer_shops = {s.shop_id: s for s in shops if s.kind == "employer"}
    for s_id, shop in employer_shops.items():
        employees = [p for p in people if p.employer_shop_id == s_id]
        monthly_payroll = sum(SALARY_MULTIPLE * p.amount_profile.median_paise for p in employees)
        needed_paise = monthly_payroll * EMPLOYER_SALARY_MONTHS_COVERED

        ledger = ledgers[shop.bank_index]
        current_bal = ledger.balances[shop.account_id]
        if needed_paise > current_bal:
            diff = needed_paise - current_bal
            ledger.balances[shop.account_id] += diff
            for entry in ledger.entries:
                if entry.account_id == shop.account_id and entry.kind == "opening":
                    entry.amount_paise += diff
                    break


def build_people(
    n_people: int,
    n_regions: int,
    banks: list[Bank],
    ledgers: list[Ledger],
    shops: list[Shop],
    ranges: dict[str, Any],
    rng: np.random.Generator,
    range_name: str = "training",
) -> list[Person]:
    """
    WHAT: Generates n_people, adds their accounts to ledgers, and funds employers for payroll.
    WHY: Constructs the full human population of the simulated world.
    """
    if n_people <= 0:
        raise ValueError(f"Population count must be positive, got {n_people}")

    employer_ids = [s.shop_id for s in shops if s.kind == "employer"]
    med_e = get_parameter_entry(ranges, "genuine_payment_median_amount_inr")
    sig_e = get_parameter_entry(ranges, "genuine_payment_amount_spread_sigma")

    people: list[Person] = []
    for i in range(n_people):
        p = create_single_person(i, n_regions, banks, employer_ids, med_e, sig_e, rng, range_name)
        ledgers[p.bank_index].add_account(p.account_id, opening_balance_paise=p.opening_balance_paise)
        people.append(p)

    fund_employer_shops(shops, people, ledgers)
    return people
