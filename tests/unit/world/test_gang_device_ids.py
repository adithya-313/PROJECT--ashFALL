"""
WHAT: Regression tests verifying unique external hardware device numbering across multiple gangs in one world.

WHY: Previously, every syndicate numbered its external devices starting at dev_ext_000001, causing artificial
hardware identifier collisions between independent gangs in multi-gang worlds. These tests verify that device
identifiers increment continuously across sequential gangs, preserving deterministic generation while preventing
false shared-device signals in ground truth and client telemetry.

ISSUE IDs: 12-8
"""

from world.fast_mode import FastRunSettings
from world.gang_config import BlockStep, FixtureSource, GangBudget, GangConfig
from world.interactive_records import RunnerSettings
from world.interactive_runner import run_interactive_world

# Copied from tests/unit/world/test_gang_runs.py (TINY, BUDGET, SOURCE_1, SOURCE_2) with take_control added.
TINY = FastRunSettings(101, 60, 12, 61, "training")
BUDGET = GangBudget(money_paise=10_000_000, time_minutes=10_000, max_risk=0.5)
SOURCE_1 = FixtureSource("acc_gang_src_01", 0, 10_000_000, 1440)
SOURCE_2 = FixtureSource("acc_gang_src_02", 1, 10_000_000, 2880)

# Expected device identifiers for gang 1 computed on base commit 90aee69 via git worktree execution.
EXPECTED_GANG_1_DEVICES: list[str] = ["dev_ext_000001", "dev_ext_000002"]


def _make_two_gangs_with_devices() -> tuple[GangConfig, GangConfig]:
    """
    WHAT: Constructs two simple gangs each possessing take_control blocks with 2 external hardware devices.
    WHY: Sets up multi-gang device allocation based on the configuration pattern in test_gang_runs.py.
    """
    steps_1 = (
        BlockStep("acquire", (("count", 3),)),
        BlockStep("take_control", (("devices", 2),)),
        BlockStep("route", (("layers", 2), ("accounts_per_layer", 1))),
    )
    steps_2 = (
        BlockStep("acquire", (("count", 4),)),
        BlockStep("take_control", (("devices", 2),)),
        BlockStep("route", (("layers", 2), ("accounts_per_layer", 2))),
    )
    g1 = GangConfig("gang_bravo", 1, BUDGET, SOURCE_1, steps_1)
    g2 = GangConfig("gang_charlie", 2, BUDGET, SOURCE_2, steps_2)
    return g1, g2


def test_unique_device_ids_across_gangs_in_one_world() -> None:
    """
    WHAT: Asserts that gang device identifiers do not collide across gangs and increment consecutively.
    WHY: Verifies that gang 1 retains its baseline device identifiers while gang 2 starts immediately after gang 1.
    """
    g1, g2 = _make_two_gangs_with_devices()
    output = run_interactive_world(
        TINY,
        RunnerSettings(fraud_enabled=True),
        gang_configs=(g1, g2),
    )
    device_spans = output.result.truth.device_control

    # 1. Assert no device_id is controlled by two different gang_ids in the truth.
    device_to_gangs: dict[str, set[str]] = {}
    for span in device_spans:
        device_to_gangs.setdefault(span.device_id, set()).add(span.gang_id)

    for device_id, gangs in device_to_gangs.items():
        assert len(gangs) == 1, f"Device '{device_id}' is controlled by multiple gangs: {gangs}"

    # 2. Assert gang 1's device ids are exactly what they were before this commit (computed on 90aee69).
    g1_devices = sorted(list({s.device_id for s in device_spans if s.gang_id == "gang_bravo"}))
    assert g1_devices == EXPECTED_GANG_1_DEVICES

    # 3. Assert gang 2's first device id is the one right after gang 1's last.
    g2_devices = sorted(list({s.device_id for s in device_spans if s.gang_id == "gang_charlie"}))
    assert len(g2_devices) == 2
    assert g2_devices[0] == "dev_ext_000003"
    assert g2_devices[1] == "dev_ext_000004"
