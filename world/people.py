"""
WHAT: Individual person citizen modeling and population generation pipeline.
WHY: Generates citizen demographics, assigns bank accounts, devices, employers, and funds initial balances.
ISSUE IDs: E2 (amount profiles), E3 (regions), E4 (banks and KYC).
TERM: opening balance: Initial funded deposit placed into an account before any simulation transfers run,
      e.g. depositing 50,000 rupees when opening a checking account so funds are immediately available for spending.
TERM: Know Your Customer (KYC): Mandatory identity verification performed by financial institutions,
      e.g. verifying an Aadhaar card or passport before enabling high-value digital payment transfers.
TERM: weak KYC: Customer accounts opened with minimal verification documentation carrying elevated fraud risk,
      e.g. accounts created with unverified phone numbers or basic self-attested forms.
TERM: employer shop: A commercial business entity whose primary role is paying scheduled wages to assigned workers,
      e.g. a local manufacturing enterprise or IT firm funding regular monthly employee payroll.
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
    WHAT: Represents a simulated private citizen residing in the town.
    It links an identity to a behavioral persona, geographic region, bank account, mobile device, and employer.
    WHY: Combines individual demographic traits, devices, and payment accounts into one cohesive entity.
    This unified model allows simulating routine living expenses, employment payrolls, and retail shopping.
    Args:
        person_id: Unique string identifier for the citizen, e.g. 'person_000000'.
        persona_name: Sociodemographic lifestyle archetype name, e.g. 'gig_worker'.
        region: Integer municipal district index where the person lives and shops.
        bank_index: Integer index of the bank managing this person's account.
        account_id: Unique string account identifier registered in the bank ledger.
        weak_kyc: Boolean indicating whether this account has minimal identity documentation.
        device: Device instance representing the person's mobile phone handset.
        amount_profile: AmountProfile defining lognormal spend scale and variance.
        employer_shop_id: Optional string shop ID for salaried workers, or None.
        opening_balance_paise: Initial deposit amount in integer paise.
    Example:
        person = Person("person_0", "gig_worker", 1, 1, "acc_person_0", False, None, None, None, 1740300)
        assert person.person_id == "person_0" and person.opening_balance_paise == 1740300
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
    It weights the choice of bank by empirical customer market shares.
    It then tests a uniform random draw against the chosen bank's weak KYC rate.
    WHY: Couples customer bank distribution with each bank's identity verification rigour.
    Institutions with lax compliance harbor more accounts vulnerable to fraudulent misuse.
    Args:
        banks: List of Bank instances containing market shares and weak KYC rates.
        rng: NumPy random generator stream ensuring reproducible attribute assignment.
    Returns:
        Tuple of (bank_index, weak_kyc_flag) determining the person's banking profile.
    Example:
        banks = [Bank(0, "large", 0.6, 0.05, 0.01), Bank(1, "regional", 0.4, 0.20, 0.02)]
        b_idx, weak = assign_bank_and_kyc(banks, np.random.default_rng(42))
        assert b_idx == 1 and weak is False
    """
    # Sample a bank index according to the customer market share of each institution.
    b_idx = choose_bank_index(banks, rng)

    # Determine whether the customer account has weak KYC based on the chosen bank's rate.
    weak_kyc = float(rng.uniform(0.0, 1.0)) < banks[b_idx].weak_kyc_share

    # Deliver the selected bank index and the boolean weak KYC status.
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
    It allocates persona archetypes, selects home regions, and configures lognormal spend profiles.
    It also links salaried workers to an employer shop and funds an initial opening balance.
    WHY: Encapsulates per-individual random attribute generation in an isolated helper.
    Ensures every citizen has complete, consistent identity, hardware, and financial attributes.
    Args:
        index: Sequential integer index used to generate zero-padded person identifiers.
        n_regions: Total number of geographic municipal regions in the simulated world.
        banks: List of active Bank institutions available for account creation.
        employer_ids: List of merchant shop identifiers categorized as employers.
        med_e: Parameter configuration dictionary for transaction median amounts.
        sig_e: Parameter configuration dictionary for transaction lognormal spread sigma.
        rng: NumPy random generator stream ensuring reproducible demographic draws.
        range_name: Configuration split name, typically 'training' or 'evaluation'.
    Returns:
        A fully constructed Person instance ready for ledger account registration.
    Example:
        p = create_single_person(0, 2, banks, ["shop_0000"], med_e, sig_e, np.random.default_rng(42), "training")
        assert p.person_id == "person_000000" and p.persona_name == "small_shopkeeper"
        assert p.opening_balance_paise == 5963400 and p.employer_shop_id is None
    """
    # Construct zero-padded unique person identifier string.
    p_id = f"person_{index:06d}"

    # Sample a demographic persona archetype and retrieve its spending tier.
    p_name = choose_persona_name(rng)
    persona = get_persona(p_name)

    # Assign a commercial bank and evaluate customer KYC documentation rigor.
    b_idx, weak_kyc = assign_bank_and_kyc(banks, rng)

    # Provision a mobile handset with an authentic platform and hardware fingerprint.
    device = create_device(f"dev_{p_id}", rng)

    # Establish lognormal spending profile and compute initial funded balance.
    profile = create_amount_profile(persona.spend_level, med_e, sig_e, rng, range_name)
    opening_paise = OPENING_BALANCE_MULTIPLE * profile.median_paise

    # If the persona is a salaried employee, randomly assign an employer shop.
    employer_id = str(rng.choice(employer_ids)) if p_name == "salaried_employee" else None

    # Assemble all demographic and banking components into the Person instance.
    return Person(
        p_id, p_name, assign_region(n_regions, rng), b_idx, f"acc_{p_id}",
        weak_kyc, device, profile, employer_id, opening_paise
    )


def fund_employer_shops(shops: list[Shop], people: list[Person], ledgers: list[Ledger]) -> None:
    """
    WHAT: Tops up employer shop balances so each can fund all assigned employee salaries for 24 months.
    It calculates the total monthly payroll obligation for each employer's assigned staff.
    If the employer's current ledger balance is insufficient, it deposits the required difference.
    WHY: Prevents employers from running out of funds during multi-month routine payroll runs.
    Guarantees that employee salary deposits never fail due to employer insolvency.
    Args:
        shops: Complete list of Shop instances including employer establishments.
        people: List of Person instances whose assigned employers require funding.
        ledgers: List of Ledger accounting books storing bank balances and journal entries.
    Returns:
        None; updates ledger balances and opening entries in-place.
    Example:
        fund_employer_shops(shops, people, ledgers)
        assert ledgers[shops[0].bank_index].balances[shops[0].account_id] >= 0
    """
    # Filter commercial shops to identify establishments operating as employers.
    employer_shops = {s.shop_id: s for s in shops if s.kind == "employer"}

    # Evaluate payroll requirements for each employer establishment.
    for s_id, shop in employer_shops.items():
        # Identify all salaried employees assigned to this specific employer.
        employees = [p for p in people if p.employer_shop_id == s_id]

        # Calculate monthly payroll and multiply by 24 months of reserve coverage.
        monthly_payroll = sum(SALARY_MULTIPLE * p.amount_profile.median_paise for p in employees)
        needed_paise = monthly_payroll * EMPLOYER_SALARY_MONTHS_COVERED

        # Check existing shop ledger balance and top up if reserve is insufficient.
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
    It iteratively creates each individual with demographic attributes, accounts, and devices.
    Finally, it calls fund_employer_shops to guarantee employers can disburse monthly salaries.
    WHY: Constructs the full human population of the simulated world.
    Ensures every citizen is properly registered in bank ledgers with an opening balance deposit.
    Args:
        n_people: Target number of citizen individuals to create in the simulation.
        n_regions: Number of municipal districts across which people are distributed.
        banks: List of Bank institutions providing financial services.
        ledgers: List of Bank ledger accounting books where accounts are opened.
        shops: List of commercial Shop entities including employers and retail merchants.
        ranges: Configuration dictionary defining transaction distribution parameters.
        rng: NumPy pseudo-random number generator for deterministic population draws.
        range_name: Configuration split name, defaults to 'training'.
    Returns:
        List of created Person instances representing the complete town population.
    Example:
        people = build_people(3, 2, banks, ledgers, shops, ranges, np.random.default_rng(42), "training")
        assert len(people) == 3 and people[0].person_id == "person_000000"
        assert people[0].opening_balance_paise == 5890100
    """
    # Enforce strictly positive population size.
    if n_people <= 0:
        raise ValueError(f"Population count must be positive, got {n_people}")

    # Gather employer identifiers and extract payment parameter distribution specifications.
    employer_ids = [s.shop_id for s in shops if s.kind == "employer"]
    med_e = get_parameter_entry(ranges, "genuine_payment_median_amount_inr")
    sig_e = get_parameter_entry(ranges, "genuine_payment_amount_spread_sigma")

    # Generate each citizen and register their opening balance in the bank ledger.
    people: list[Person] = []
    for i in range(n_people):
        p = create_single_person(i, n_regions, banks, employer_ids, med_e, sig_e, rng, range_name)
        ledgers[p.bank_index].add_account(p.account_id, opening_balance_paise=p.opening_balance_paise)
        people.append(p)

    # Fund employer accounts with sufficient reserves to pay employee wages.
    fund_employer_shops(shops, people, ledgers)

    # Provide the fully populated list of citizen entities.
    return people
