"""
WHAT: Integration and regression tests for device event execution in calendar simulations.
WHY: Proves device lifecycle events mutate personal devices in time order and stamp payment logs (issue R5).
TERM: device event: a discrete occurrence altering handset state such as phone replacement or OS update,
      e.g. upgrading an operating system generation while preserving hardware serial numbers.
TERM: device stamp: telemetry metadata recording sender hardware fingerprint and novelty per transaction,
      e.g. tagging a payment with the sender phone fingerprint and marking new devices.
TERM: golden hash: a cryptographic SHA-256 digest ensuring simulation outputs reproduce identically,
      e.g. verifying that a repeated simulation run produces the exact same payment ledger.
"""

from pathlib import Path

import numpy as np

from world.world_plan import spawn_life_streams, spawn_world_streams

REPO_ROOT = Path(__file__).resolve().parents[3]
DEVICE_CONFIG_PATH = REPO_ROOT / "config" / "device_realism.yaml"


def test_spawn_life_streams() -> None:
    """
    WHAT: Tests reproducibility and independence of auxiliary device and episode pseudo-random streams.
    WHY: Proves child streams 2 and 3 do not collide with each other or perturb the primary calendar streams.
    Example:
        dev_rng, ep_rng = spawn_life_streams(101)
        assert dev_rng.uniform() != ep_rng.uniform()
    """
    # 1. Test reproducibility across repeated calls with identical seed.
    dev1, ep1 = spawn_life_streams(101)
    dev2, ep2 = spawn_life_streams(101)
    d1 = [float(dev1.uniform()) for _ in range(5)]
    d2 = [float(dev2.uniform()) for _ in range(5)]
    e1 = [float(ep1.uniform()) for _ in range(5)]
    e2 = [float(ep2.uniform()) for _ in range(5)]
    assert d1 == d2
    assert e1 == e2
    # 2. Test independence between the two life streams.
    assert d1 != e1
    # 3. Test independence from primary plan and run simulation streams.
    plan_rng, run_rng = spawn_world_streams(101)
    p_draws = [float(plan_rng.uniform()) for _ in range(5)]
    r_draws = [float(run_rng.uniform()) for _ in range(5)]
    assert d1 != p_draws and d1 != r_draws
    assert e1 != p_draws and e1 != r_draws
    # 4. Prove spawn_world_streams matches children 0 and 1 of SeedSequence.spawn(4).
    s4 = np.random.SeedSequence(101).spawn(4)
    rng0 = np.random.default_rng(s4[0])
    rng1 = np.random.default_rng(s4[1])
    assert p_draws == [float(rng0.uniform()) for _ in range(5)]
    assert r_draws == [float(rng1.uniform()) for _ in range(5)]
