"""
WHAT: Commercial merchant shop entities and catalog generation across municipal regions.
WHY: Groundwork for Issue A2; provides business destinations for salaries, rents, utilities, and daily purchases.
ISSUE IDs: A2 (genuine shops downstream of mules), E3 (regions), E4 (banks).
TERM: shop: A registered merchant commercial account that receives consumer purchases or disburses employee salaries,
      e.g. a neighborhood grocery store taking QR code payments or an electricity company collecting utility bills.
TERM: mandatory shop kinds: The foundational commercial categories required to sustain citizen routine life,
      e.g. employer, utility, landlord, and grocery store guaranteeing essential payment destinations exist.
TERM: bank market share: The proportion of total municipal accounts held by a specific financial institution,
      e.g. a leading regional bank holding 60% of commercial shop accounts.
"""

from dataclasses import dataclass

import numpy as np

from world.banks import Bank
from world.ledger import Ledger
from world.money import rupees_to_paise
from world.placeholders import SHOP_KIND_WEIGHTS, SHOP_OPENING_BALANCE_RUPEES
from world.regions import assign_region


@dataclass
class Shop:
    """
    WHAT: Represents a registered commercial business establishment in the town.
    It encapsulates the unique merchant identifier, shop kind category, geographic region, bank index, and account ID.
    WHY: Holds banking, location, and merchant category details for payment routing.
    Enables realistic merchant payment flows like employee salaries, household bills, and daily living expenditures.
    Args:
        shop_id: Unique string identifier for the merchant establishment, e.g. 'shop_0000'.
        kind: Commercial category string such as 'employer', 'utility', 'landlord', or 'grocery'.
        region: Integer municipal district index where the commercial store is physically located.
        bank_index: Integer index of the bank holding and managing the merchant's commercial account.
        account_id: Unique string account identifier registered in the bank ledger, e.g. 'acc_shop_0000'.
    Example:
        shop = Shop("shop_0000", "employer", 0, 0, "acc_shop_0000")
        assert shop.kind == "employer" and shop.bank_index == 0
    """

    shop_id: str
    kind: str
    region: int
    bank_index: int
    account_id: str


def choose_bank_index(banks: list[Bank], rng: np.random.Generator) -> int:
    """WHAT: Samples a bank index weighted by the customer shares of the available banks.
    It extracts bank indices and their corresponding customer market share proportions.
    It then draws an integer index using NumPy's weighted random choice mechanism.
    WHY: Distributes institutional accounts according to empirical bank market shares.
    Ensures realistic customer concentration across major versus smaller institutions.
    Args: banks: List of Bank instances containing market shares; rng: Random generator stream.
    Returns: Integer index of the chosen bank institution, such as 0 or 1.
    Example:
        banks = [Bank(0, "large", 0.6, 0.05, 0.01), Bank(1, "regional", 0.4, 0.20, 0.02)]
        b_idx = choose_bank_index(banks, np.random.default_rng(42))
        assert b_idx == 1
    """
    # Extract the integer bank indices and their proportional customer market shares.
    indices = [b.bank_index for b in banks]
    shares = [b.customer_share for b in banks]

    # Sample a bank index according to the empirical market share distribution.
    return int(rng.choice(indices, p=shares))


def determine_shop_kind(index: int, rng: np.random.Generator) -> str:
    """WHAT: Determines the commercial category of a shop based on its sequence index.
    It reserves the first four shop slots for mandatory categories: employer, utility, landlord, and grocery.
    For any subsequent shop indices, it samples categories according to configured market weights.
    WHY: Guarantees that every simulated town has the essential merchant destinations for routine life.
    Subsequent retail diversity mirrors realistic municipal retail distributions.
    Args: index: Sequential integer index of the shop being constructed; rng: Random generator stream.
    Returns: Category string such as 'employer', 'utility', 'landlord', 'grocery', or 'restaurant'.
    Example:
        k0 = determine_shop_kind(0, np.random.default_rng(42))
        k4 = determine_shop_kind(4, np.random.default_rng(42))
        assert k0 == "employer" and k4 == "restaurant"
    """
    # Define the mandatory commercial categories required for basic economic simulation.
    mandatory_kinds = ["employer", "utility", "landlord", "grocery"]
    if index < len(mandatory_kinds):
        return mandatory_kinds[index]

    # Extract non-mandatory shop categories and their configured probability weights.
    kinds = list(SHOP_KIND_WEIGHTS.keys())
    weights = [SHOP_KIND_WEIGHTS[k] for k in kinds]

    # Sample a commercial category according to the empirical retail distribution.
    return str(rng.choice(kinds, p=weights))


def build_shops(
    n_shops: int,
    n_regions: int,
    banks: list[Bank],
    ledgers: list[Ledger],
    rng: np.random.Generator,
) -> list[Shop]:
    """
    WHAT: Builds n_shops commercial shops, registers their accounts, and seeds their opening balances.
    It guarantees the mandatory merchant types are established and places each store in a region.
    It also opens an account in the chosen bank ledger and deposits the standard merchant opening balance.
    WHY: Creates the business network required for citizen economic life.
    Provides destination accounts to receive routine bill payments, rent, and grocery purchases.
    Args:
        n_shops: Total number of commercial merchant establishments to build (minimum 4).
        n_regions: Number of municipal districts across which shops are distributed.
        banks: List of Bank institutions providing commercial banking services.
        ledgers: List of Ledger accounting books where merchant accounts are registered.
        rng: NumPy pseudo-random number generator ensuring reproducible shop creation.
    Returns:
        List of constructed Shop instances representing all town merchant establishments.
    Example:
        banks = [Bank(0, "large", 0.6, 0.05, 0.01), Bank(1, "regional", 0.4, 0.20, 0.02)]
        ledgers = [Ledger(0, 10000000), Ledger(1, 10000000)]
        shops = build_shops(4, 2, banks, ledgers, np.random.default_rng(42))
        assert len(shops) == 4 and shops[0].kind == "employer" and shops[0].shop_id == "shop_0000"
    """
    # Enforce minimum shop count to guarantee all mandatory categories are present.
    if n_shops < 4:
        raise ValueError(f"Must have at least 4 shops for routine life, requested {n_shops}")

    # Initialize container for created shops and compute initial merchant deposit in paise.
    shops: list[Shop] = []
    base_opening_paise = rupees_to_paise(SHOP_OPENING_BALANCE_RUPEES)

    # Iteratively generate each commercial shop establishment.
    for i in range(n_shops):
        s_id = f"shop_{i:04d}"
        kind = determine_shop_kind(i, rng)
        region = assign_region(n_regions, rng)
        b_idx = choose_bank_index(banks, rng)
        acc_id = f"acc_{s_id}"

        # Register the merchant account in the chosen bank ledger with opening deposit.
        ledgers[b_idx].add_account(acc_id, opening_balance_paise=base_opening_paise)
        shops.append(Shop(shop_id=s_id, kind=kind, region=region, bank_index=b_idx, account_id=acc_id))

    # Deliver the complete list of registered merchant establishments.
    return shops
