"""
WHAT: Unit tests for Level 12g device_rotation tactical option in take_control planning and truth tracking.

WHY: Level 12g device_rotation replaces shared device clusters with dedicated single-mule handsets that rotate
on a strict calendar interval, eliminating shared device signatures from forensic clustering. These tests verify
generational device provisioning, chronological handover intervals, single-account hardware exclusivity,
and split-level authorization guardrails.

ISSUE IDs: 12-8
"""

import numpy as np
import pytest

from world.build_world import build_world
from world.fast_mode import FastRunSettings
from world.gang_config import BlockStep, FixtureSource, GangBudget, GangConfig, validate_gang_config
from world.gang_planner import plan_gang

TEST_BUDGET = GangBudget(money_paise=50_000_000, time_minutes=500_000, max_risk=0.99)
TEST_SOURCE = FixtureSource("acc_gang_src_01", 0, 10_000_000, 1440)
TEST_SETTINGS = FastRunSettings(seed=42, n_people=60, n_shops=10, n_days=30, range_name="test")


def _make_rotation_gang(
    rotate_every_days: int = 7,
    rotations: int = 2,
    sim_swap: bool = True,
    emulator: bool = False,
    devices: int | None = None,
) -> GangConfig:
    """Helper to construct a rotation gang config for unit tests."""
    take_control_params: list[tuple[str, int | bool]] = [
        ("rotate_every_days", rotate_every_days),
        ("rotations", rotations),
        ("sim_swap", sim_swap),
        ("emulator", emulator),
    ]
    if devices is not None:
        take_control_params.append(("devices", devices))

    steps = (
        BlockStep("acquire", (("count", 3),)),
        BlockStep("take_control", tuple(take_control_params)),
        BlockStep(
            "route",
            (
                ("layers", 3),
                ("accounts_per_layer", 1),
                ("amount_inr", 5000),
                ("dwell_minutes", 60),
            ),
        ),
    )
    return GangConfig("gang_rotation_test", 1, TEST_BUDGET, TEST_SOURCE, steps, None)


def test_rotation_spans_and_devices_structure() -> None:
    """Verifies that device rotation creates distinct handsets per mule generation with continuous valid spans."""
    world = build_world(TEST_SETTINGS.seed, TEST_SETTINGS.n_people, TEST_SETTINGS.n_shops)
    rng = np.random.default_rng(123)

    mules_count = 3
    rotate_days = 7
    rotations = 2
    generations = rotations + 1  # 3 generations
    period_minutes = rotate_days * 1440  # 10080 minutes
    t0 = 1440

    gang = _make_rotation_gang(rotate_every_days=rotate_days, rotations=rotations, sim_swap=True, emulator=False)
    plan = plan_gang(gang, world, rng, world_split="test")

    spans = plan.truth.device_control
    # 1. Total devices must equal mules_count * generations = 3 * 3 = 9 distinct devices.
    expected_device_count = mules_count * generations
    distinct_devices = {s.device_id for s in spans}
    assert len(distinct_devices) == expected_device_count

    # 2. Each account has 3 spans with valid_from at t0, t0+10080, t0+20160.
    acquired_mules = [r.account_id for r in plan.truth.roles if r.role == "mule"]
    assert len(acquired_mules) == mules_count

    for acc in acquired_mules:
        acc_spans = [s for s in spans if s.account_id == acc]
        assert len(acc_spans) == generations

        expected_froms = [t0 + g * period_minutes for g in range(generations)]
        actual_froms = [s.valid_from for s in acc_spans]
        assert actual_froms == expected_froms

        # Earlier span's valid_to equals next span's valid_from.
        assert acc_spans[0].valid_to == acc_spans[1].valid_from
        assert acc_spans[1].valid_to == acc_spans[2].valid_from
        # Last generation's valid_to is None.
        assert acc_spans[2].valid_to is None

        # Verify compromise flags carried through across every generation.
        for s in acc_spans:
            assert s.sim_swap is True
            assert s.emulator is False

    # 3. Exclusivity: At any minute, no device controls more than one account, and no two accounts share a device.
    device_to_accounts: dict[str, set[str]] = {}
    for s in spans:
        device_to_accounts.setdefault(s.device_id, set()).add(s.account_id)
    assert all(len(accs) == 1 for accs in device_to_accounts.values())

    # 4. Telemetry events: exactly 9 device_acquired events, each positioned at its generation start.
    dev_acquired_events = [e for e in plan.gang_device_events if e.kind == "device_acquired"]
    assert len(dev_acquired_events) == expected_device_count
    assert all(e.account_id is None for e in dev_acquired_events)

    gen_0_events = [e for e in dev_acquired_events if e.minute == t0]
    gen_1_events = [e for e in dev_acquired_events if e.minute == t0 + period_minutes]
    gen_2_events = [e for e in dev_acquired_events if e.minute == t0 + 2 * period_minutes]
    assert len(gen_0_events) == mules_count
    assert len(gen_1_events) == mules_count
    assert len(gen_2_events) == mules_count


def test_take_control_refuses_devices_with_rotation() -> None:
    """Verifies that configuring devices alongside rotate_every_days raises ValueError during config validation."""
    gang = _make_rotation_gang(devices=2)
    with pytest.raises(ValueError, match=r"cannot specify devices together with rotation options"):
        validate_gang_config(gang)


def test_sim_swap_events_generated_for_each_account_generation() -> None:
    """Verifies that with sim_swap True, every generation's device produces a sim_swap event for its account."""
    world = build_world(TEST_SETTINGS.seed, TEST_SETTINGS.n_people, TEST_SETTINGS.n_shops)
    rng = np.random.default_rng(123)

    mules_count = 3
    rotate_days = 7
    rotations = 2
    generations = rotations + 1  # 3 generations
    period_minutes = rotate_days * 1440
    t0 = 1440

    gang = _make_rotation_gang(rotate_every_days=rotate_days, rotations=rotations, sim_swap=True, emulator=False)
    plan = plan_gang(gang, world, rng, world_split="test")

    sim_swaps = [e for e in plan.gang_device_events if e.kind == "sim_swap"]
    # 3 accounts * 3 generations = 9 sim_swap events
    assert len(sim_swaps) == mules_count * generations

    acquired_mules = [r.account_id for r in plan.truth.roles if r.role == "mule"]
    assert len(acquired_mules) == mules_count

    for g_idx in range(generations):
        gen_minute = t0 + g_idx * period_minutes
        gen_events = [e for e in sim_swaps if e.minute == gen_minute]
        assert len(gen_events) == mules_count
        assert {e.account_id for e in gen_events} == set(acquired_mules)


def test_planning_rotation_gang_in_validation_split_refused() -> None:
    """Verifies that attempting to plan a rotation gang under validation split is refused fail-closed."""
    world = build_world(TEST_SETTINGS.seed, TEST_SETTINGS.n_people, TEST_SETTINGS.n_shops)
    rng = np.random.default_rng(42)

    gang = _make_rotation_gang()
    with pytest.raises(ValueError) as excinfo:
        plan_gang(gang, world, rng, world_split="validation")

    message = str(excinfo.value)
    assert "rotate_every_days" in message
    assert "device_rotation" in message
    assert "validation" in message
