"""
WHAT: Master world builder orchestration and cryptographic deterministic state fingerprinting.
WHY: Constructs the full town simulation in a reproducible, seed-driven sequence and validates identity via SHA-256.
TERM: world fingerprint: A unique SHA-256 digest calculated over the serialized state of all entities in a world.
"""

from dataclasses import dataclass
import hashlib
from typing import Any
import numpy as np

from world.banks import Bank, build_banks
from world.config_reader import draw_in_range, get_parameter_entry, read_world_ranges
from world.ledger import Ledger
from world.money import rupees_to_paise
from world.people import Person, build_people
from world.regions import draw_number_of_regions
from world.shops import Shop, build_shops


@dataclass
class World:
    """
    WHAT: Root container holding all banks, ledgers, shops, people, and simulation parameters.
    WHY: Represents a complete independent world instance generated from a specific seed.
    """

    seed: int
    banks: list[Bank]
    ledgers: list[Ledger]
    shops: list[Shop]
    people: list[Person]
    parameters_drawn: dict[str, Any]


def validate_world_sizes(n_people: int, n_shops: int) -> None:
    """
    WHAT: Validates that population and shop count parameters are positive whole numbers.
    WHY: Defends the builder against corrupt or nonsensical scale arguments.
    """
    if not isinstance(n_people, int) or n_people <= 0:
        raise ValueError(f"Invalid people count: expected positive integer, got {n_people}")
    if not isinstance(n_shops, int) or n_shops <= 0:
        raise ValueError(f"Invalid shops count: expected positive integer, got {n_shops}")


def build_world(
    seed: int,
    n_people: int,
    n_shops: int,
    range_name: str = "training",
) -> World:
    """
    WHAT: Builds a complete simulated world deterministically from a single seed.
    WHY: The primary entry point for generating Level 11 town environments.
    """
    validate_world_sizes(n_people, n_shops)
    rng = np.random.default_rng(seed)
    ranges = read_world_ranges()

    n_reg_entry = get_parameter_entry(ranges, "number_of_regions")
    n_regions = draw_number_of_regions(n_reg_entry, rng, range_name=range_name)
    banks = build_banks(ranges, rng, range_name=range_name)

    upi_entry = get_parameter_entry(ranges, "upi_per_transaction_limit_inr")
    upi_rupees = draw_in_range(upi_entry, rng, range_name=range_name)
    upi_limit_paise = rupees_to_paise(upi_rupees)
    ledgers = [Ledger(bank_index=b.bank_index, upi_limit_paise=upi_limit_paise) for b in banks]

    shops = build_shops(n_shops, n_regions, banks, ledgers, rng)
    people = build_people(n_people, n_regions, banks, ledgers, shops, ranges, rng, range_name=range_name)
    drawn = {"n_regions": n_regions, "upi_limit_paise": upi_limit_paise, "range_name": range_name}

    return World(seed, banks, ledgers, shops, people, drawn)


def serialize_world_entities(world: World) -> list[str]:
    """
    WHAT: Converts banks, shops, people, and ledger entries into canonical string lines.
    WHY: Produces an unambiguous string serialization for cryptographic digest generation.
    """
    lines: list[str] = [f"seed:{world.seed}"]
    for b in world.banks:
        lines.append(f"b:{b.bank_index}:{b.kind}:{b.customer_share:.8f}:{b.weak_kyc_share:.8f}:{b.hold_delay_factor:.8f}")
    for s in world.shops:
        lines.append(f"s:{s.shop_id}:{s.kind}:{s.region}:{s.bank_index}:{s.account_id}")
    for p in world.people:
        lines.append(
            f"p:{p.person_id}:{p.persona_name}:{p.region}:{p.bank_index}:{p.account_id}:"
            f"{p.weak_kyc}:{p.device.fingerprint}:{p.amount_profile.median_paise}:"
            f"{p.amount_profile.sigma:.6f}:{p.employer_shop_id}:{p.opening_balance_paise}"
        )
    for l in world.ledgers:
        for acc in sorted(l.accounts):
            lines.append(f"l:{l.bank_index}:acc:{acc}:{l.balance_of(acc)}")
        for e in l.entries:
            lines.append(f"l:{l.bank_index}:e:{e.entry_id}:{e.minute}:{e.account_id}:{e.amount_paise}:{e.kind}:{e.transfer_id}")
    return lines


def world_fingerprint(world: World) -> str:
    """
    WHAT: Computes the SHA-256 hash of the fully serialized world state.
    WHY: Proves complete bitwise reproducibility across runs and Python processes.
    """
    lines = serialize_world_entities(world)
    payload = "\n".join(lines).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
