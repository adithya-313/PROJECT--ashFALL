"""
WHAT: Unit tests for realistic account opening date draws (child stream 23) and bank feed integration.
WHY: Verifies reproducibility across seeds, conformance with configured day intervals,
cryptographic golden digest consistency, stream isolation against prior components, statistical
overlap ensuring account age does not leak mule identity, and accurate bank feed projection timing.
"""

import hashlib
import math
from dataclasses import replace
from datetime import timedelta

import pytest

from core.event_clock import world_minute_to_ist
from world.bank_feed import project_run
from world.build_world import build_world
from world.config_reader import get_parameter_entry, read_world_ranges
from world.gang_config import BlockStep, GangBudget, GangConfig
from world.interactive_runner import FastRunSettings, RunnerSettings, run_interactive_world
from world.opening_dates import draw_opening_days
from world.people import Person
from world.run_hashes import hash_ledger_entries, hash_payment_events

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


def test_stream_isolation_between_opening_dates_and_world_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    WHAT: Proves that drawing realistic opening dates on child stream 23 does not shift or alter
    any payments, ledger balances, or device events produced by existing simulation components.
    WHY: Protects stream isolation invariants: adding new feature randomness must never perturb old results.
    """
    tiny_settings = FastRunSettings(101, 60, 12, 61, "training")
    runner_settings = RunnerSettings(fraud_enabled=True)

    # 1. Run simulation with standard opening date drawing active on child 23.
    output_with_dates = run_interactive_world(tiny_settings, runner_settings=runner_settings)

    # 2. Run simulation with opening dates bypassed (as in earlier rounds before child 23 was active).
    monkeypatch.setattr("world.interactive_runner.draw_opening_days", lambda *args, **kwargs: {})
    output_without_dates = run_interactive_world(tiny_settings, runner_settings=runner_settings)

    # 3. Assert cryptographic hashes of payment events, double-entry ledgers, and device schedules match exactly.
    assert hash_payment_events(output_with_dates.result.run_result.events) == hash_payment_events(
        output_without_dates.result.run_result.events
    )
    assert hash_ledger_entries(output_with_dates.world.ledgers) == hash_ledger_entries(
        output_without_dates.world.ledgers
    )
    assert output_with_dates.device_events == output_without_dates.device_events


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


def test_mule_and_genuine_opening_age_overlap() -> None:
    """
    WHAT: Tests that the age distribution of rented mule starting accounts statistically overlaps with genuine
    starting accounts, satisfying the Mann-Whitney AUC |AUC - 0.5| <= 4 * sigma_AUC bound.
    WHY: Architect rule: rented mules must share the same age distribution as ordinary accounts so age alone
    cannot separate genuine customers from money mules.
    """
    # Configure a world with 200 people, 20 shops, and a gang acquiring 30 rented accounts across 5 layers.
    # This guarantees at least 20 rented mule accounts that existed at run start.
    budget = GangBudget(money_paise=50_000_000, time_minutes=500_000, max_risk=0.99)
    steps = (
        BlockStep("acquire", (("count", 30), ("origin_mix", "all_rented"))),
        BlockStep("scam", (("receiving_mules", 5), ("victims_per_receiving_mule", 2))),
        BlockStep("route", (("layers", 5), ("accounts_per_layer", 5))),
    )
    gang = GangConfig("g_overlap", 1, budget, None, steps, 1440)
    settings = FastRunSettings(101, 200, 20, 61, "training")

    output = run_interactive_world(settings, runner_settings=RunnerSettings(fraud_enabled=True), gang_configs=(gang,))

    # Extract rented mule starting accounts using ground truth roles.
    rented_mule_ids = {
        role.account_id
        for role in output.result.truth.roles
        if role.role == "mule" and role.origin == "rented" and role.account_id in output.account_opening_days
    }
    # Extract genuine starting accounts (starting accounts that were never assigned a mule role).
    genuine_ids = [acc for acc in output.account_opening_days if acc not in rented_mule_ids]

    n1 = len(rented_mule_ids)
    n2 = len(genuine_ids)

    # Prove scenario: test requires >= 20 rented mule accounts existing at the start.
    assert n1 >= 20, f"Expected at least 20 rented starting mules, got {n1}"
    assert n2 > 0

    x = [output.account_opening_days[acc] for acc in rented_mule_ids]
    y = [output.account_opening_days[acc] for acc in genuine_ids]

    # Calculate Mann-Whitney U and AUC for age as a score distinguishing mules from genuine.
    u1 = sum(1.0 if xi > yj else (0.5 if xi == yj else 0.0) for xi in x for yj in y)
    auc = u1 / (n1 * n2)

    # Standard error of AUC under the null hypothesis of no distributional difference.
    standard_error = math.sqrt((n1 + n2 + 1) / (12.0 * n1 * n2))
    bound = 4.0 * standard_error

    # Print the required single MEASURED line containing sample sizes, AUC, and tolerance bound.
    print(f"MEASURED: n1={n1}, n2={n2}, AUC={auc:.4f}, bound={bound:.4f}")

    assert abs(auc - 0.5) <= bound, f"|AUC - 0.5| = {abs(auc - 0.5):.4f} exceeded bound {bound:.4f}"


def test_bank_feed_starting_account_event_time_and_missing_account_error() -> None:
    """
    WHAT: Tests that projected account_opened events for starting accounts have an event_time matching
    the simulation start minute minus their drawn opening days, and that a missing account raises ValueError.
    WHY: Verifies correct bank feed projection timing and strict fail-fast validation against incomplete date maps.
    """
    tiny_settings = FastRunSettings(101, 60, 12, 61, "training")
    output = run_interactive_world(tiny_settings, runner_settings=RunnerSettings(fraud_enabled=True))

    feed = project_run(output)
    all_events = [ev for events in feed.account_events_by_bank.values() for ev in events]
    open_events = [ev for ev in all_events if ev.kind == "account_opened"]

    # 1. Verify every starting account's account_opened event_time equals minute 0 minus drawn opening days.
    start_time = world_minute_to_ist(0, output.plan.clock.start_date)
    assert len(open_events) == len(output.account_opening_days)
    assert len(open_events) > 0

    for ev in open_events:
        assert ev.account_id in output.account_opening_days
        days = output.account_opening_days[ev.account_id]
        expected_time = start_time - timedelta(days=days)
        assert ev.event_time == expected_time

    # 2. Verify that projecting a run with an account missing from account_opening_days raises ValueError.
    corrupted_days = dict(output.account_opening_days)
    missing_account = list(corrupted_days.keys())[0]
    corrupted_days.pop(missing_account)

    corrupted_output = replace(output, account_opening_days=corrupted_days)

    with pytest.raises(ValueError, match=rf"Account {missing_account!r} missing from account_opening_days"):
        project_run(corrupted_output)
