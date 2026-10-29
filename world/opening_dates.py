"""
WHAT: Draws realistic opening dates (in days before run start) for accounts that exist when the world begins.
Accounts belonging to starting people and shops draw an integer age from their configured ranges on child 23.

WHY (architect decision):
Accounts that exist when the world starts have no opening date; round 1 stamps their account_opened at the run's
first minute. That would let a model learn "opened at the start = genuine", which is a fact about our simulator,
not about mules (most genuine customers exist from minute 0; many mule accounts are opened mid-run by gangs).
Real banks know every account's true opening date. So the world draws a realistic opening date for every starting
account, and the bank feed uses it. Accounts opened mid-run keep their true opening minute.

TERM: opening age in days: How many whole 24-hour days before the simulation start minute an account was opened.
TERM: child stream: An isolated random number generator spawned from the master seed sequence by index.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np

from world.config_reader import get_parameter_entry
from world.people import Person
from world.run_identity import load_runner_config
from world.shops import Shop

STREAM_NAME: str = "account_opening_dates"


def draw_opening_days(
    people: Sequence[Person],
    shops: Sequence[Shop],
    seed: int,
    ranges: dict[str, Any],
    range_name: str = "training",
) -> dict[str, int]:
    """
    WHAT: Draws the initial age (in days) for every account belonging to starting people and shops.
    Each person's account age is sampled uniformly from their persona's configured day range.
    Each shop's account age is sampled uniformly from the shop day range. Both endpoints are included.

    WHY: Prevents detection models from falsely learning that an account existing at minute 0 is guaranteed
    to be legitimate, grounding account tenures in realistic multi-year banking history.

    ORDER DETERMINISM:
    Accounts are processed in a fixed, strictly deterministic sequence: people sorted alphabetically by account_id,
    followed by shops sorted alphabetically by account_id. This fixed ordering guarantees that given the same
    random seed and entity set, the sequence of pseudo-random numbers consumed from child stream 23 remains
    identical and reproducible across runs.

    Args:
        people: Sequence of Person instances created during world building.
        shops: Sequence of Shop instances created during world building.
        seed: Master random seed for initializing the child random stream.
        ranges: Master world ranges configuration dictionary containing account_opening_age_days.
        range_name: Evaluation split name ('training', 'test', or 'stress').

    Returns:
        Mapping from account_id string to drawn integer days open at simulation start.

    Raises:
        ValueError: If an unrecognized persona name is encountered on any person.
    """
    # Read child stream index from runner config (never typed as a literal in source code).
    runner_config = load_runner_config()
    child = int(runner_config["random_stream_children"][STREAM_NAME])
    # Spawn the isolated single-child generator using the SeedSequence spawn pattern.
    rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(child + 1)[child])

    # Retrieve the configured age bounds block from world ranges.
    entry = get_parameter_entry(ranges, "account_opening_age_days")
    per_persona: dict[str, list[int]] = entry.get("per_persona", {})
    shop_bounds: list[int] = entry["shop"]

    opening_days: dict[str, int] = {}

    # 1. Walk people accounts sorted by account_id for deterministic stream consumption.
    sorted_people = sorted(people, key=lambda p: p.account_id)
    for person in sorted_people:
        persona = person.persona_name
        if persona not in per_persona:
            raise ValueError(f"Unknown persona name {persona!r} for person {person.person_id}")
        bounds = per_persona[persona]
        # Draw uniform integer age with both endpoints included.
        days = int(rng.integers(int(bounds[0]), int(bounds[1]), endpoint=True))
        opening_days[person.account_id] = days

    # 2. Walk shop accounts sorted by account_id for deterministic stream consumption.
    sorted_shops = sorted(shops, key=lambda s: s.account_id)
    for shop in sorted_shops:
        # Draw uniform integer age with both endpoints included.
        days = int(rng.integers(int(shop_bounds[0]), int(shop_bounds[1]), endpoint=True))
        opening_days[shop.account_id] = days

    return opening_days
