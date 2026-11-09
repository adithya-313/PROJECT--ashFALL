"""
Tests for the replay's registry plan (replay/registry_replay.py and replay/registry_driver.py): the pre-world
seed, the visible minute, the new replay step kinds and the switches.

WHY: The registry seed and world reports must be reproducible, must not move any other random draw, and must follow
deactivation < listing < reassignment with nothing visible before day 0.

TERM: seed listing: a registry entry for the phone of a previous (mule) holder, made before day 0.
TERM: visible minute: ingest time plus the listing delay, never before day 0.
"""

from datetime import date, datetime, timedelta

import numpy as np

from core.event_clock import IST, ManualClock
from replay.ordering import KIND_ORDER
from replay.registry_driver import build_registry_driver
from replay.registry_replay import (
    REGISTRY_INGEST,
    REGISTRY_VISIBLE,
    RegistryReplaySettings,
    load_registry_replay_settings,
    plan_registry_seed,
    replay_dpip_key,
    visible_world_time,
)
from replay.runner import load_default_world_registry_reporting
from tests.unit.world.test_phone_book_recording import record
from world.interactive_loop import build_component_streams
from world.phone_book import PhoneBook, PhoneRecord
from world.run_identity import load_runner_config

START = date(2026, 9, 1)
MIDNIGHT = datetime(2026, 9, 1, tzinfo=IST)
DAY = 1440


def settings(share: tuple[float, float] = (1.0, 1.0)) -> RegistryReplaySettings:
    """Three DPIP participants, 15 minute delay, and the given share range."""
    banks = {f"bank_{i}": {"dpip_participation": True} for i in range(3)}
    reporter_choice = "uniform_over_dpip_participants"
    return RegistryReplaySettings(timedelta(minutes=15), timedelta(minutes=60), banks, share, reporter_choice)


def recycled_book(n: int = 4) -> PhoneBook:
    """A phone book whose first n accounts hold recycled numbers (deactivated 100+ days, reassigned later)."""
    records = {}
    for i in range(n):
        records[f"acc_{i}"] = PhoneRecord(
            f"acc_{i}", f"sim-{i:010d}", True, -(100 + i) * DAY, -(100 + i) * DAY + 91 * DAY, False, None
        )
    records["acc_plain"] = PhoneRecord("acc_plain", "sim-9999999999", False, None, None, False, None)
    by_phone = {r.phone_number: [r] for r in records.values()}
    return PhoneBook(records, by_phone, 1, n, 0)


def test_same_seed_gives_the_same_listings() -> None:
    """Checks the seed plan is a pure function of its stream."""
    key = replay_dpip_key("r1")
    first = plan_registry_seed(recycled_book(), settings(), np.random.default_rng(5), "r1", key, START)
    second = plan_registry_seed(recycled_book(), settings(), np.random.default_rng(5), "r1", key, START)
    assert first == second
    assert len(first) == 4


def test_listing_order_and_visible_minute_are_pre_world_then_day_zero() -> None:
    """Checks deactivation < listing < reassignment, and visible time is max(listing, day 0)."""
    book = recycled_book()
    listings = plan_registry_seed(book, settings(), np.random.default_rng(5), "r1", replay_dpip_key("r1"), START)
    for listing in listings:
        record_ = next(
            r for r in book.records.values() if r.previous_holder_deactivated_minute == listing.item.reported_minute
        )
        deactivated = MIDNIGHT + timedelta(minutes=record_.previous_holder_deactivated_minute or 0)
        listed = deactivated + timedelta(minutes=15)
        reassigned = MIDNIGHT + timedelta(minutes=record_.reassigned_minute or 0)
        assert deactivated < listed < reassigned < MIDNIGHT
        assert listing.visible_time == MIDNIGHT
        assert listing.item.reported_at == deactivated
        assert listing.item.bank_id in {"bank_0", "bank_1", "bank_2"}
    ingest = MIDNIGHT + timedelta(minutes=10)
    assert visible_world_time(ingest, timedelta(minutes=15), START) == MIDNIGHT + timedelta(minutes=25)


def test_share_zero_lists_nothing_but_draws_the_same_count() -> None:
    """Checks a zero share lists nothing, and the stream is left where a full share leaves it."""
    key = replay_dpip_key("r1")
    none_rng, all_rng = np.random.default_rng(5), np.random.default_rng(5)
    assert plan_registry_seed(recycled_book(), settings((0.0, 0.0)), none_rng, "r1", key, START) == []
    plan_registry_seed(recycled_book(), settings((1.0, 1.0)), all_rng, "r1", key, START)
    assert none_rng.uniform() == all_rng.uniform()


def test_the_seed_stream_is_its_own_child_and_moves_no_other_draw() -> None:
    """Checks registry_seed has its own child number and adding it leaves every other child as it was."""
    children = load_runner_config()["random_stream_children"]
    assert list(children.values()).count(children["registry_seed"]) == 1
    others = {name: n for name, n in children.items() if name != "registry_seed"}
    before = build_component_streams(101, others)
    after = build_component_streams(101, children)
    for name in others:
        assert before[name].uniform() == after[name].uniform()


def test_kind_order_places_registry_after_complaints_and_ingest_before_visible() -> None:
    """Checks the tie order: complaint 3, registry_ingest 4, registry_visible 5."""
    assert KIND_ORDER["complaint"] < KIND_ORDER[REGISTRY_INGEST] < KIND_ORDER[REGISTRY_VISIBLE]
    assert (KIND_ORDER[REGISTRY_INGEST], KIND_ORDER[REGISTRY_VISIBLE]) == (4, 5)


def test_switch_defaults_replay_on_and_live_off() -> None:
    """Checks replay.yaml turns world_registry_reporting on and config/nationalsim.yaml keeps it off."""
    from core.config_validator import load_yaml
    from replay.registry_replay import NATIONALSIM_PATH

    assert load_default_world_registry_reporting() is True
    assert load_yaml(NATIONALSIM_PATH)["nationalsim"]["world_registry_reporting"]["value"] is False


def test_real_world_plan_is_reproducible_and_holds_hashes_only() -> None:
    """Checks the driver for the small recorded world is identical twice and its payloads carry no raw identifier."""
    first = build_registry_driver(record(), ManualClock(MIDNIGHT), load_registry_replay_settings("training"))
    second = build_registry_driver(record(), ManualClock(MIDNIGHT), load_registry_replay_settings("training"))
    assert [(x.item, x.visible_time) for x in first.seed] == [(x.item, x.visible_time) for x in second.seed]
    assert [(x.item, x.visible_time) for x in first.world] == [(x.item, x.visible_time) for x in second.world]
    for listing in [*first.seed, *first.world]:
        assert len(listing.item.identifier_hash) == 64
        assert listing.visible_time >= MIDNIGHT
    steps = first.steps()
    assert {s.kind for s in steps} <= {REGISTRY_INGEST, REGISTRY_VISIBLE}
    assert sum(s.kind == REGISTRY_VISIBLE for s in steps) == len(first.seed) + len(first.world)
