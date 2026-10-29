"""
WHAT: Unit tests for realistic account opening date draws (child stream 23).
WHY: Verifies reproducibility across seeds, conformance with configured day intervals,
error handling for unknown personas, and pinned cryptographic golden digest consistency.
"""

import hashlib

import pytest

from world.build_world import build_world
from world.config_reader import get_parameter_entry, read_world_ranges
from world.opening_dates import draw_opening_days
from world.people import Person

pytestmark = pytest.mark.anyio


def test_same_seed_gives_identical_dates_different_seed_different() -> None:
    """
    WHAT: Tests that drawing opening dates with the same seed yields identical results,
    while drawing with a different seed produces different dates.
    WHY: Enforces deterministic pseudo-random stream properties required for reproducibility.
    """
    ranges = read_world_ranges()
    world_a = build_world(seed=101, n_people=30, n_shops=5)

    dates_run1 = draw_opening_days(world_a.people, world_a.shops, seed=101, ranges=ranges, range_name="training")
    dates_run2 = draw_opening_days(world_a.people, world_a.shops, seed=101, ranges=ranges, range_name="training")
    dates_diff_seed = draw_opening_days(world_a.people, world_a.shops, seed=202, ranges=ranges, range_name="training")

    # Prove non-empty mapping was generated for all people and shops.
    expected_count = len(world_a.people) + len(world_a.shops)
    assert len(dates_run1) == expected_count
    assert expected_count > 0

    # Same seed must yield identical mappings.
    assert dates_run1 == dates_run2

    # Different seeds must produce distinct draws.
    assert dates_run1 != dates_diff_seed


def test_every_drawn_date_lies_within_owner_and_persona_range() -> None:
    """
    WHAT: Tests that every drawn opening age falls strictly within the configured [min_days, max_days]
    interval for its owner persona or shop category, endpoints included.
    WHY: Guarantees that sampling respects empirical boundaries configured in world_ranges.yaml.
    """
    ranges = read_world_ranges()
    world = build_world(seed=101, n_people=50, n_shops=10)
    dates = draw_opening_days(world.people, world.shops, seed=101, ranges=ranges, range_name="training")

    entry = get_parameter_entry(ranges, "account_opening_age_days")
    per_persona: dict[str, list[int]] = entry["per_persona"]
    shop_bounds: list[int] = entry["shop"]

    # Verify each person's account age is within its persona's configured range.
    for person in world.people:
        age_days = dates[person.account_id]
        bounds = per_persona[person.persona_name]
        assert bounds[0] <= age_days <= bounds[1], (
            f"Person {person.person_id} with persona {person.persona_name} got age {age_days}, "
            f"outside [{bounds[0]}, {bounds[1]}]"
        )

    # Verify each shop's account age is within the shop configured range.
    for shop in world.shops:
        age_days = dates[shop.account_id]
        assert shop_bounds[0] <= age_days <= shop_bounds[1], (
            f"Shop {shop.shop_id} got age {age_days}, outside [{shop_bounds[0]}, {shop_bounds[1]}]"
        )


def test_unknown_persona_raises_value_error() -> None:
    """
    WHAT: Tests that an unrecognized persona name encountered on a Person raises ValueError.
    WHY: Enforces strict fail-fast validation against configuration drift or corrupted agent data.
    """
    ranges = read_world_ranges()
    invalid_person = Person(
        person_id="person_bad_01",
        persona_name="unknown_fictional_persona",
        region=1,
        bank_index=0,
        account_id="acc_bad_01",
        weak_kyc=False,
        device=None,
        amount_profile=None,
        employer_shop_id=None,
        opening_balance_paise=100000,
    )

    with pytest.raises(ValueError, match="unknown_fictional_persona"):
        draw_opening_days([invalid_person], [], seed=101, ranges=ranges, range_name="training")


def test_pinned_golden_hash_for_fixed_small_world() -> None:
    """
    WHAT: Tests that drawn opening dates for the standard tiny world (seed 101, 60 people, 12 shops)
    match the pinned cryptographic SHA-256 digest.
    WHY: Acts as a golden regression anchor ensuring that future generator alterations do not silently shift draws.
    """
    # Generated from: build_world(seed=101, n_people=60, n_shops=12), child 23 uniform draws,
    # serialized as newline-delimited sorted 'account_id:days' pairs and hashed with SHA-256.
    PINNED_OPENING_DATES_HASH = "ab56ce38279918c90f4a142a759cbb611011f04b31dbe4ab4002477b2d3c7d33"

    ranges = read_world_ranges()
    world = build_world(seed=101, n_people=60, n_shops=12)
    dates = draw_opening_days(world.people, world.shops, seed=101, ranges=ranges, range_name="training")

    assert len(dates) == 72

    payload = "\n".join(f"{account_id}:{days}" for account_id, days in sorted(dates.items())).encode("utf-8")
    actual_hash = hashlib.sha256(payload).hexdigest()

    assert actual_hash == PINNED_OPENING_DATES_HASH
