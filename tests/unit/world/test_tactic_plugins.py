"""
WHAT: Unit tests for local tactic plugins: the lint refuses unsafe files without importing them, paths outside the
plugin folder are refused, the cool_off example plans inside a gang, plans repeat exactly, the registry is cleared
after use, plugin steps change the gang's plain form only when a plugin is used, and bad plugin results are refused.
Expected values come from the test itself (an identical generator, a literal dictionary, hashlib), never from the
result being checked.

WHY: A plugin is local code. If the lint missed a random source or the registry leaked, plans would differ between
runs or tests without any error.

TERM: lint: reading a file's syntax tree for forbidden names without running it.
"""

import dataclasses
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from world.fast_mode import FastRunSettings
from world.gang_config import gang_config_from_plain, gang_config_to_plain, gang_start_minute
from world.gang_planner import GangPlan, _plan_acquire_step, plan_gang
from world.interactive_runner import prepare_world
from world.tactic_plugins import (
    LoadedPlugin,
    PluginContext,
    PluginStepResult,
    TacticPlugin,
    get_registered_plugin,
    load_plugin,
    plugins_registered,
    run_plugin_step,
)
from world.world_sizing import resolve_settings

REPO_ROOT = Path(__file__).resolve().parents[3]
COOL_OFF = REPO_ROOT / "patterns" / "plugins" / "cool_off.py"
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "plugins"
MINUTES_PER_DAY = 1440
PREVIEW_SEED = 1000000


def _gang_plain(with_cool_off: bool) -> dict[str, Any]:
    """Returns a plain gang dictionary: acquire, optionally cool_off, then scam and route."""
    steps: list[dict[str, Any]] = [{"block": "acquire", "params": {"count": 4}}]
    if with_cool_off:
        steps.append({"block": "cool_off", "params": {}})
    steps.append({"block": "scam", "params": {"receiving_mules": 2, "victims_per_receiving_mule": 8}})
    steps.append({"block": "route", "params": {"layers": 2, "accounts_per_layer": 2}})
    return {
        "gang_id": "plugin_test_gang",
        "strength": 1,
        "budget": {"money_paise": 50000000, "time_minutes": 500000, "max_risk": 0.99},
        "start_minute": 1440,
        "steps": steps,
    }


def _settings() -> FastRunSettings:
    """Returns the tiny training settings with the first training seed."""
    return resolve_settings(seed=PREVIEW_SEED, preset="tiny", range_name="training")


def _rngs() -> tuple[np.random.Generator, np.random.Generator]:
    """Returns the device and planning generators exactly as the interactive runner builds them (children 6 and 7)."""
    children = np.random.SeedSequence(PREVIEW_SEED).spawn(10)
    return np.random.default_rng(children[6]), np.random.default_rng(children[7])


@pytest.fixture(scope="module")
def world() -> Any:
    """Builds the small training world once for the module."""
    return prepare_world(_settings(), None).world


@pytest.fixture(scope="module")
def planned(world: Any) -> tuple[GangPlan, int]:
    """Plans the cool_off gang once inside the context manager; returns the plan and the days a twin generator draws."""
    with plugins_registered([COOL_OFF]):
        config = gang_config_from_plain(_gang_plain(True))
        dev_rng, plan_rng = _rngs()
        plan = plan_gang(config, world, plan_rng, reserved_accounts=set(), gang_devices_rng=dev_rng)
        # An identical generator: replay the acquire step, then draw as the plugin does.
        _, twin = _rngs()
        _plan_acquire_step(world, config, config.steps[0], gang_start_minute(config), set(), twin)
        expected_days = int(twin.integers(1, 7, endpoint=True))
    return plan, expected_days


def test_global_random_is_refused_and_never_imported() -> None:
    """A plugin importing random is refused by name; its top-level raise is never reached."""
    with pytest.raises(ValueError, match="imports 'random'"):
        load_plugin(FIXTURES / "bad_global_random.py", allowed_root=FIXTURES)


def test_numpy_global_calls_are_refused_through_an_alias() -> None:
    """np.random.seed and np.random.default_rng, reached through an alias, are both named; nothing is imported."""
    with pytest.raises(ValueError, match="numpy.random.seed") as problem:
        load_plugin(FIXTURES / "bad_numpy_global.py", allowed_root=FIXTURES)
    assert "numpy.random.default_rng" in str(problem.value)


def test_name_clash_with_a_built_in_block_is_refused() -> None:
    """A clean plugin named 'route' must be refused because route is a built-in block."""
    with pytest.raises(ValueError, match="clashes with a built-in block name"):
        load_plugin(FIXTURES / "bad_name_clash.py", allowed_root=FIXTURES)


def test_path_outside_the_plugin_folder_is_refused() -> None:
    """Without an explicit root, a file outside patterns/plugins must be refused before it is read."""
    with pytest.raises(ValueError, match="outside"):
        load_plugin(FIXTURES / "bad_name_clash.py")


def test_cool_off_loads_with_the_hash_of_its_file() -> None:
    """The example loads, and its recorded hash equals hashlib's hash of the file bytes."""
    loaded = load_plugin(COOL_OFF)
    assert loaded.plugin.name == "cool_off"
    assert loaded.sha256 == hashlib.sha256(COOL_OFF.read_bytes()).hexdigest()


def test_gang_with_cool_off_plans_and_the_clock_advances(planned: tuple[GangPlan, int]) -> None:
    """The cool_off step costs exactly the twin generator's days times 1440 minutes, and the scam starts after it."""
    plan, expected_days = planned
    assert 1 <= expected_days <= 7
    assert plan.costs[1].time_minutes == expected_days * MINUTES_PER_DAY
    assert plan.costs[1].money_paise == 0
    first_scam_minute = min(
        payment.minute for payment, story in zip(plan.payments, plan.truth.payment_stories)
        if story.scam_step == "victim_payment"
    )
    assert first_scam_minute >= 1440 + expected_days * MINUTES_PER_DAY


def test_same_seed_plans_identically(world: Any, planned: tuple[GangPlan, int]) -> None:
    """Planning the same gang again from the same seed gives the same payments and totals."""
    first, _ = planned
    with plugins_registered([COOL_OFF]):
        config = gang_config_from_plain(_gang_plain(True))
        dev_rng, plan_rng = _rngs()
        second = plan_gang(config, world, plan_rng, reserved_accounts=set(), gang_devices_rng=dev_rng)
    assert first.payments == second.payments
    assert first.totals == second.totals


def test_registry_is_cleared_after_the_context_manager() -> None:
    """Outside plugins_registered the plugin is gone and a gang naming it is refused as an unknown block."""
    with plugins_registered([COOL_OFF]):
        assert get_registered_plugin("cool_off") is not None
    assert get_registered_plugin("cool_off") is None
    with pytest.raises(ValueError, match="unknown block 'cool_off'"):
        gang_config_from_plain(_gang_plain(True))


def test_plain_form_is_unchanged_without_plugins_and_carries_a_hash_with_one() -> None:
    """A built-in-only gang serialises as a literal dictionary; a cool_off step adds plugin_sha256 to that step only."""
    plain = gang_config_to_plain(gang_config_from_plain(_gang_plain(False)))
    assert plain == {
        "gang_id": "plugin_test_gang",
        "strength": 1,
        "budget": {"money_paise": 50000000, "time_minutes": 500000, "max_risk": 0.99},
        "start_minute": 1440,
        "steps": [
            {"block": "acquire", "params": [("count", 4)]},
            {"block": "scam", "params": [("receiving_mules", 2), ("victims_per_receiving_mule", 8)]},
            {"block": "route", "params": [("layers", 2), ("accounts_per_layer", 2)]},
        ],
    }
    with plugins_registered([COOL_OFF]):
        with_plugin = gang_config_to_plain(gang_config_from_plain(_gang_plain(True)))
    file_hash = hashlib.sha256(COOL_OFF.read_bytes()).hexdigest()
    assert with_plugin["steps"][1] == {"block": "cool_off", "params": [], "plugin_sha256": file_hash}
    for index in (0, 2, 3):
        assert "plugin_sha256" not in with_plugin["steps"][index]


def test_plugin_result_cannot_carry_device_events() -> None:
    """A plugin returns only minutes and money, and sees no world, ledger, truth or accounts (12-8 structurally)."""
    result_fields = {field.name for field in dataclasses.fields(PluginStepResult)}
    assert result_fields == {"minutes_advanced", "money_paise"}
    context_fields = {field.name for field in dataclasses.fields(PluginContext)}
    assert context_fields == {"params", "clock_minute", "strength", "rng"}
    for forbidden in ("world", "ledger", "ledgers", "truth", "accounts"):
        assert forbidden not in context_fields


def test_bad_plugin_results_and_parameters_are_refused() -> None:
    """Negative minutes and a parameter outside its range are both refused by run_plugin_step."""
    rng = np.random.default_rng(1)

    def negative(context: PluginContext) -> PluginStepResult:
        """Returns negative minutes, which a plugin must never do."""
        return PluginStepResult(-1, 0)

    bad = TacticPlugin("negative_wait", "x", (1,), {}, (0.0, 0.0), "assumption", "A test plugin.", negative)
    with pytest.raises(ValueError, match="minutes_advanced must be a whole number of at least 0"):
        run_plugin_step(LoadedPlugin(bad, "0" * 64, COOL_OFF), {}, 0, 1, rng)
    with plugins_registered([COOL_OFF]) as loaded_list:
        with pytest.raises(ValueError, match="outside its range"):
            run_plugin_step(loaded_list[0], {"days": 8}, 0, 1, rng)
