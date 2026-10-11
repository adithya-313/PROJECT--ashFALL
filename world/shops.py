"""
WHAT: Commercial merchant shop entities and catalog generation across municipal regions.
WHY: Groundwork for Issue A2; provides business destinations for salaries, rents, utilities, and daily purchases.
ISSUE IDs: A2 (genuine shops downstream of mules), E3 (regions), E4 (banks).
TERM: shop: A merchant account receiving everyday customer purchases or paying scheduled employment wages.
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
    WHY: Holds banking, location, and merchant category details for payment routing.
    """

    shop_id: str
    kind: str
    region: int
    bank_index: int
    account_id: str


def choose_bank_index(banks: list[Bank], rng: np.random.Generator) -> int:
    """
    WHAT: Samples a bank index weighted by the customer shares of the available banks.
    WHY: Distributes institutional accounts according to empirical bank market shares.
    """
    indices = [b.bank_index for b in banks]
    shares = [b.customer_share for b in banks]
    return int(rng.choice(indices, p=shares))


def determine_shop_kind(index: int, rng: np.random.Generator) -> str:
    """
    WHAT: Determines the commercial category of a shop by index.
    WHY: Forces the first four shops to be employer, utility, landlord, and grocery so routine life
         always has required destination merchants, while subsequent shops follow category weights.
    """
    mandatory_kinds = ["employer", "utility", "landlord", "grocery"]
    if index < len(mandatory_kinds):
        return mandatory_kinds[index]

    kinds = list(SHOP_KIND_WEIGHTS.keys())
    weights = [SHOP_KIND_WEIGHTS[k] for k in kinds]
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
    WHY: Creates the business network required for citizen economic life.
    """
    if n_shops < 4:
        raise ValueError(f"Must have at least 4 shops for routine life, requested {n_shops}")

    shops: list[Shop] = []
    base_opening_paise = rupees_to_paise(SHOP_OPENING_BALANCE_RUPEES)

    for i in range(n_shops):
        s_id = f"shop_{i:04d}"
        kind = determine_shop_kind(i, rng)
        region = assign_region(n_regions, rng)
        b_idx = choose_bank_index(banks, rng)
        acc_id = f"acc_{s_id}"

        ledgers[b_idx].add_account(acc_id, opening_balance_paise=base_opening_paise)
        shops.append(Shop(shop_id=s_id, kind=kind, region=region, bank_index=b_idx, account_id=acc_id))

    return shops
