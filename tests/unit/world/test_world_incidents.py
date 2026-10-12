"""
WHAT: Unit tests for simulated incident scheduling, timing bounds, and partial degradation.
WHY: Proves that NPCI incidents and bank downtimes occur at random intervals within the world,
     with partial success rates rather than total blackouts (issue R7).
ISSUE IDs: R7 (outage schedule and partial degradation).
TERM: incident schedule: A sequence of time-windowed service degradation events across banks and central switch.
"""

import hashlib
import subprocess
from pathlib import Path

import numpy as np
import pytest

from world.background_config import read_background_realism
from world.incidents import draw_incident_schedule

CONFIG_PATH = Path("config/failure_realism.yaml")


def test_i1_incident_bounds_and_structure() -> None:
    """
    WHAT: Tests incident time windows, kinds, bank indices, and ordering over 100 seeds.
    WHY: Verifies incidents sit inside the world and adhere to structural invariants.
    """
    data = read_background_realism(CONFIG_PATH)
    npci_dur = data["parameters"]["npci_incident_duration_minutes"]["training"]
    bank_dur = data["parameters"]["bank_downtime_duration_minutes"]["training"]
    npci_rate = data["parameters"]["npci_incident_success_rate"]["training"]
    bank_rate = data["parameters"]["bank_downtime_success_rate"]["training"]

    for seed in range(100):
        rng = np.random.default_rng(seed)
        sched = draw_incident_schedule(data, rng, n_days=60, n_banks=3, range_name="training")
        for inc in sched:
            assert 0 <= inc.start_minute
            assert inc.start_minute + inc.duration_minutes <= 60 * 1440
            if inc.kind == "npci_incident":
                assert inc.bank_index is None
                assert npci_dur[0] <= inc.duration_minutes <= npci_dur[1]
                assert npci_rate[0] <= inc.success_rate <= npci_rate[1]
            else:
                assert inc.bank_index in (0, 1, 2)
                assert bank_dur[0] <= inc.duration_minutes <= bank_dur[1]
                assert bank_rate[0] <= inc.success_rate <= bank_rate[1]
        keys = [(it.start_minute, it.kind, -1 if it.bank_index is None else it.bank_index) for it in sched]
        assert keys == sorted(keys)


def test_i2_incident_counts_over_seeds() -> None:
    """
    WHAT: Tests that NPCI and bank downtime counts match scaled config expectations.
    WHY: Guarantees Poisson/uniform incident generation adheres to configured rates.
    """
    data = read_background_realism(CONFIG_PATH)
    npci_cfg = data["parameters"]["npci_incident_count_per_30_days"]["training"]
    bank_cfg = data["parameters"]["bank_downtime_count_per_bank_per_30_days"]["training"]
    max_npci = round(npci_cfg[1] * 60 / 30)
    min_bank = round(bank_cfg[0] * 60 / 30)
    max_bank = round(bank_cfg[1] * 60 / 30)

    for seed in range(100):
        rng = np.random.default_rng(seed)
        sched = draw_incident_schedule(data, rng, n_days=60, n_banks=3, range_name="training")
        npci_count = sum(1 for it in sched if it.kind == "npci_incident")
        assert 0 <= npci_count <= max_npci
        for b_idx in range(3):
            b_count = sum(1 for it in sched if it.bank_index == b_idx)
            assert min_bank <= b_count <= max_bank


def test_i3_no_hard_coded_time_diversity() -> None:
    """
    WHAT: Checks diversity of incident start minutes across 200 random seeds.
    WHY: Proves incident starts are drawn uniformly without fixed days or schedules.
    """
    data = read_background_realism(CONFIG_PATH)
    starts: set[int] = set()
    for seed in range(200):
        rng = np.random.default_rng(seed)
        sched = draw_incident_schedule(data, rng, n_days=60, n_banks=3, range_name="training")
        for it in sched:
            starts.add(it.start_minute)
    assert len(starts) >= 50


def test_i4_partial_degradation_success_rates() -> None:
    """
    WHAT: Asserts that all incident success rates are strictly between 0 and 1.
    WHY: Enforces partial degradation semantics so outages never drop to total blackouts.
    """
    data = read_background_realism(CONFIG_PATH)
    for seed in range(100):
        rng = np.random.default_rng(seed)
        sched = draw_incident_schedule(data, rng, n_days=60, n_banks=3, range_name="training")
        for it in sched:
            assert 0.0 < it.success_rate < 1.0


def test_i5_refusals_and_determinism() -> None:
    """
    WHAT: Tests input refusals and cross-process SHA-256 determinism of schedules.
    WHY: Enforces rigorous parameter validation and reproducible simulation replay.
    """
    d = read_background_realism(CONFIG_PATH)
    rng = np.random.default_rng(42)
    for bad_d in (1, 0, -5, 2.5, "60"):
        with pytest.raises((ValueError, TypeError)):
            draw_incident_schedule(d, rng, n_days=bad_d, n_banks=3)  # type: ignore[arg-type]
    for bad_b in (0, 2.5):
        with pytest.raises((ValueError, TypeError)):
            draw_incident_schedule(d, rng, n_days=60, n_banks=bad_b)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        draw_incident_schedule(d, rng, n_days=60, n_banks=3, range_name="unknown")

    # In-process determinism and seed divergence.
    s0 = draw_incident_schedule(d, np.random.default_rng(0), n_days=60, n_banks=3)
    assert s0 == draw_incident_schedule(d, np.random.default_rng(0), n_days=60, n_banks=3)
    assert any(draw_incident_schedule(d, np.random.default_rng(s), n_days=60, n_banks=3) != s0 for s in range(1, 21))

    # Cross-process SHA-256 determinism check via subprocess.
    script = (
        "import hashlib, numpy as np; from world.background_config import read_background_realism; "
        "from world.incidents import draw_incident_schedule; d=read_background_realism('config/failure_realism.yaml'); "
        "sch=draw_incident_schedule(d, np.random.default_rng(0), 60, 3); "
        "print(hashlib.sha256(repr(sch).encode()).hexdigest())"
    )
    res = subprocess.run(["python", "-B", "-c", script], capture_output=True, text=True, check=True)
    assert res.stdout.strip() == hashlib.sha256(repr(s0).encode()).hexdigest()

