"""
WHAT: Master world builder orchestration and cryptographic deterministic state fingerprinting.
It instantiates the complete simulated municipal economy, including banks, ledgers, shops, and citizens.
It also provides canonical string serialization and SHA-256 fingerprinting to guarantee bitwise reproducibility.

WHY: Constructs the full town simulation in a reproducible, seed-driven sequence and validates identity via SHA-256.
Deterministic environment construction ensures that experiments and benchmarks can be precisely replicated.
Cryptographic fingerprinting detects accidental statistical drift or unintended regressions during refactoring.

TERM: world fingerprint: A unique SHA-256 cryptographic digest calculated over the serialized state of all entities in a world. For example, a 64-character hex hash guaranteeing exact bitwise reproducibility of a generated town.
TERM: deterministic world generation: Constructing an entire simulated economy from a single integer seed so that every entity and balance is reproducible. For example, generating identical towns whenever seed=7 is passed.
TERM: entity serialization: Converting structured objects (banks, shops, citizens, accounts) into standardized text lines for hashing. For example, formatting a person into a colon-delimited string record.
"""

import hashlib
from dataclasses import dataclass
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
    It encapsulates the full state of a simulated town generated from a specific seed.
    Downstream simulation and routine execution modules pass this object to access world entities.

    WHY: Represents a complete independent world instance generated from a specific seed.
    Grouping all town entities into a single cohesive structure avoids global variables.
    It provides an isolated sandbox that can be serialized, evaluated, or run independently.

    Args:
        seed: Master integer random seed used to generate all entities and parameters in this world.
        banks: List of the three Bank instances operating within the simulated town.
        ledgers: List of Ledger instances maintaining double-entry accounts and balances for each bank.
        shops: List of Shop instances representing local commercial establishments and employers.
        people: List of Person instances representing synthetic citizens and account holders.
        parameters_drawn: Dictionary recording all continuous configuration parameter values drawn.

    Returns:
        Initialized World instance containing all simulated town entities.

    Example:
        # Build a 10-person world and verify root container properties:
        # w = build_world(seed=7, n_people=10, n_shops=5)
        # assert w.seed == 7 and len(w.people) == 10
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
    It checks both type and positive value constraints for n_people and n_shops.
    If either argument is non-integer, zero, or negative, it raises a descriptive ValueError.

    WHY: Defends the builder against corrupt or nonsensical scale arguments.
    Passing non-positive entity counts causes downstream stochastic generators to fail with confusing errors.
    Enforcing structural preconditions upfront ensures clean failure reporting during configuration errors.

    Args:
        n_people: Proposed count of citizens to instantiate in the simulated world.
        n_shops: Proposed count of commercial shops to instantiate in the simulated world.

    Returns:
        None.

    Example:
        # Validating acceptable world sizes:
        # validate_world_sizes(10, 5)
    """
    # Validate that people count is a strictly positive integer.
    if not isinstance(n_people, int) or n_people <= 0:
        raise ValueError(f"Invalid people count: expected positive integer, got {n_people}")
    # Validate that shop count is a strictly positive integer.
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
    It initializes random number generators, loads configuration ranges, and samples town dimensions.
    It instantiates banks, ledgers, shops, and citizens, returning the populated World container.

    WHY: The primary entry point for generating Level 11 town environments.
    Orchestrating entity creation in a strict dependency sequence ensures bitwise reproducibility.
    It establishes complete economic infrastructure including accounts and opening balances.

    Args:
        seed: Integer random seed governing all pseudo-random draws during world generation.
        n_people: Number of citizen agents to generate within the simulated town.
        n_shops: Number of commercial retail and employer shops to generate.
        range_name: Configuration interval split to sample parameter bounds from, defaulting to 'training'.

    Returns:
        Fully initialized World instance ready for routine execution and fraud modeling.

    Example:
        # Build a small world with seed 7:
        # w = build_world(seed=7, n_people=10, n_shops=5)
        # assert len(w.people) == 10 and len(w.shops) == 5
    """
    # Validate requested population and commercial scale parameters.
    validate_world_sizes(n_people, n_shops)
    # Initialize deterministic random generator and load master parameter ranges.
    rng = np.random.default_rng(seed)
    ranges = read_world_ranges()

    # Draw the municipal region partition count and construct the three banks.
    n_reg_entry = get_parameter_entry(ranges, "number_of_regions")
    n_regions = draw_number_of_regions(n_reg_entry, rng, range_name=range_name)
    banks = build_banks(ranges, rng, range_name=range_name)

    # Determine the regulatory UPI transaction ceiling and initialize bank ledgers.
    upi_entry = get_parameter_entry(ranges, "upi_per_transaction_limit_inr")
    upi_rupees = draw_in_range(upi_entry, rng, range_name=range_name)
    upi_limit_paise = rupees_to_paise(upi_rupees)
    ledgers = [Ledger(bank_index=b.bank_index, upi_limit_paise=upi_limit_paise) for b in banks]

    # Generate commercial shops and seed citizen agents with accounts and balances.
    shops = build_shops(n_shops, n_regions, banks, ledgers, rng)
    people = build_people(n_people, n_regions, banks, ledgers, shops, ranges, rng, range_name=range_name)
    drawn = {"n_regions": n_regions, "upi_limit_paise": upi_limit_paise, "range_name": range_name}

    return World(seed, banks, ledgers, shops, people, drawn)


def serialize_world_entities(world: World) -> list[str]:
    """
    WHAT: Converts banks, shops, people, and ledger entries into canonical string lines.
    It loops through each entity category in sorted, deterministic order and writes standardized text fields.
    The returned list of strings represents the complete serialized state of the world economy.

    WHY: Produces an unambiguous string serialization for cryptographic digest generation.
    Hashing complex nested objects requires a canonical byte representation immune to dict ordering variations.
    It enables exact state comparisons and reproducibility auditing across different machines and platforms.

    Args:
        world: World instance whose entities, accounts, and ledger entries are to be serialized.

    Returns:
        Ordered list of canonical string records representing the complete world state.

    Example:
        # Serialize a world of 10 people and 5 shops:
        # w = build_world(seed=7, n_people=10, n_shops=5)
        # lines = serialize_world_entities(w)
        # assert len(lines) == 49
    """
    # Serialize world master seed.
    lines: list[str] = [f"seed:{world.seed}"]
    # Serialize bank institutions with operational parameters.
    for b in world.banks:
        lines.append(
            f"b:{b.bank_index}:{b.kind}:{b.customer_share:.8f}:{b.weak_kyc_share:.8f}:{b.hold_delay_factor:.8f}"
        )
    # Serialize commercial shops and their territorial assignments.
    for s in world.shops:
        lines.append(f"s:{s.shop_id}:{s.kind}:{s.region}:{s.bank_index}:{s.account_id}")
    # Serialize citizen demographic attributes, devices, and financial profiles.
    for p in world.people:
        lines.append(
            f"p:{p.person_id}:{p.persona_name}:{p.region}:{p.bank_index}:{p.account_id}:"
            f"{p.weak_kyc}:{p.device.fingerprint}:{p.amount_profile.median_paise}:"
            f"{p.amount_profile.sigma:.6f}:{p.employer_shop_id}:{p.opening_balance_paise}"
        )
    # Serialize bank ledger account balances and immutable entry histories.
    for ldg in world.ledgers:
        for acc in sorted(ldg.accounts):
            lines.append(f"l:{ldg.bank_index}:acc:{acc}:{ldg.balance_of(acc)}")
        for e in ldg.entries:
            lines.append(
                f"l:{ldg.bank_index}:e:{e.entry_id}:{e.minute}:{e.account_id}:{e.amount_paise}:{e.kind}:{e.transfer_id}"
            )
    return lines


def world_fingerprint(world: World) -> str:
    """
    WHAT: Computes the SHA-256 hash of the fully serialized world state.
    It serializes all entities into canonical text lines, encodes them to UTF-8, and calculates the digest.
    The returned 64-character hexadecimal string uniquely identifies the generated world state.

    WHY: Proves complete bitwise reproducibility across runs and Python processes.
    Comparing fingerprints guarantees that changes in code do not inadvertently alter world generation seeds.
    It serves as the golden standard check verifying environment generation determinism.

    Args:
        world: World instance whose serialized state will be cryptographically digested.

    Returns:
        64-character hexadecimal SHA-256 hash string representing the world fingerprint.

    Example:
        # Compute fingerprint for a 10-person, 5-shop world:
        # w = build_world(seed=7, n_people=10, n_shops=5)
        # fp = world_fingerprint(w)
        # assert fp == "2bc0abe533ff03b91f00e4336b5fe8f34326bc1ab1f6dd0e7bff7a4603826c81"
    """
    # Generate canonical string representations of all world entities.
    lines = serialize_world_entities(world)
    # Join lines with newlines and compute the cryptographic SHA-256 digest.
    payload = "\n".join(lines).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

