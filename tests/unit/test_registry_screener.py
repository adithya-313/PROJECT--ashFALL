"""
Tests for the replay registry screener (replay/registry_screener.py): it asks the in-process registry as of the
decision time, draws one delay per question (at least 1 ms, own stream), and answers "unknown" over the timeout.

WHY: Replay features must only see listings visible at the decision, never the latest state, and a slow registry must
degrade to "unknown", never to "no match".

TERM: as_of: the decision time the registry is asked about.
TERM: unknown: no answer within the registry timeout.
"""

import dataclasses
from datetime import datetime, timedelta

import numpy as np

from core.event_clock import IST, ManualClock
from nationalsim.keyed_hash import compute_identifier_hash
from nationalsim.registry import RegistryCore
from replay.registry_driver import build_registry_driver
from replay.registry_replay import load_registry_replay_settings
from replay.registry_screener import CoreRegistryScreener, ScreenStats
from tests.unit.world.test_phone_book_recording import record
from world.interactive_loop import build_component_streams
from world.run_identity import load_runner_config

RUN = "0123456789abcdef"
KEY = "unit-test-key"
T0 = datetime(2026, 9, 1, 10, 0, tzinfo=IST)
BANKS = {"bank_0": {"dpip_participation": True}, "bank_1": {"dpip_participation": False}}


def listed_core(clock: ManualClock) -> tuple[RegistryCore, str]:
    """A registry holding one phone entry reported by bank_0 at T0, visible 15 minutes later."""
    core = RegistryCore(clock, timedelta(minutes=15), BANKS)
    phone_hash = compute_identifier_hash(KEY, RUN, "phone", "sim-1")
    clock.set_time(T0)
    core.ingest("bank_0", f"{RUN}-R000001", phone_hash, "phone", "fraud", T0, RUN)
    return core, phone_hash


def screener(
    core: RegistryCore, clock: ManualClock, delay_ms: tuple[float, float] = (2.0, 15.0), timeout_ms: float = 33.19,
    seed: int = 7, stats: ScreenStats | None = None,
) -> CoreRegistryScreener:
    """A bank_0 screener over the core."""
    return CoreRegistryScreener(
        core, "bank_0", RUN, clock, np.random.default_rng(seed), delay_ms, timeout_ms, stats or ScreenStats()
    )


def test_entry_is_seen_only_once_visible_as_of_the_decision() -> None:
    """Checks a decision before visible_at sees no match though the registry clock is later; one after sees it."""
    clock = ManualClock(T0)
    core, phone_hash = listed_core(clock)
    clock.set_time(T0 + timedelta(hours=2))
    s = screener(core, clock)
    assert s.screen(phone_hash, "phone", as_of=T0 + timedelta(minutes=10)).status == "no_match"
    assert s.screen(phone_hash, "phone", as_of=T0 + timedelta(minutes=15)).status == "match"
    assert s.screen(phone_hash, "phone").status == "match"  # no as_of: the world clock now


def test_same_seed_gives_the_same_delays_and_answers() -> None:
    """Checks two screeners on the same seed and range count the same unknowns."""
    clock = ManualClock(T0)
    core, phone_hash = listed_core(clock)
    outcomes = []
    for _ in range(2):
        stats = ScreenStats()
        s = screener(core, clock, delay_ms=(20.0, 50.0), seed=11, stats=stats)
        outcomes.append([s.screen(phone_hash, "phone").status for _ in range(20)] + [stats.unknown])
    assert outcomes[0] == outcomes[1]
    assert 0 < outcomes[0][-1] < 20


def test_a_forced_slow_range_gives_unknown_never_no_match() -> None:
    """Checks a range wholly over the timeout answers unknown every time and counts it."""
    clock = ManualClock(T0)
    core, phone_hash = listed_core(clock)
    stats = ScreenStats()
    s = screener(core, clock, delay_ms=(50.0, 60.0), stats=stats)
    assert {s.screen(phone_hash, "phone").status for _ in range(5)} == {"unknown"}
    assert (stats.screens, stats.unknown) == (5, 5)


def test_the_delay_is_never_zero() -> None:
    """Checks a (0, 0) range still counts as 1 ms, so a timeout under 1 ms gives unknown."""
    clock = ManualClock(T0)
    core, phone_hash = listed_core(clock)
    clock.set_time(T0 + timedelta(hours=1))
    assert screener(core, clock, delay_ms=(0.0, 0.0), timeout_ms=0.5).screen(phone_hash, "phone").status == "unknown"
    assert screener(core, clock, delay_ms=(0.0, 0.0), timeout_ms=5.0).screen(phone_hash, "phone").status == "match"


def test_the_delay_stream_is_its_own_child_and_moves_no_other_draw() -> None:
    """Checks registry_screen_delay has its own child number and adding it leaves the other children unchanged."""
    children = load_runner_config()["random_stream_children"]
    assert list(children.values()).count(children["registry_screen_delay"]) == 1
    others = {name: n for name, n in children.items() if name != "registry_screen_delay"}
    before = build_component_streams(101, others)
    after = build_component_streams(101, children)
    for name in others:
        assert before[name].uniform() == after[name].uniform()


def test_non_participant_banks_get_no_screener() -> None:
    """Checks the driver gives a screener to DPIP participants only."""
    reg = load_registry_replay_settings("training")
    banks = {b: dict(s) for b, s in reg.bank_settings.items()}
    banks["bank_2"]["dpip_participation"] = False
    driver = build_registry_driver(record(), ManualClock(T0), dataclasses.replace(reg, bank_settings=banks))
    assert driver.screener_for("bank_0") is not None
    assert driver.screener_for("bank_2") is None
