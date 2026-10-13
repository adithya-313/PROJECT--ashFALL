"""
WHAT: Unit tests E1-E7 for ordinary citizen episode schedule generation and temporal states.
WHY: Verifies time-limited deceptive spikes for non-mule personas (issue A1).
ISSUE IDs: A1 (episode and background realism).
TERM: episode: A temporary period where an ordinary person receives high-velocity incoming transfers.
TERM: ordinary person: An individual whose persona does not mimic mule patterns in daily life.
"""

import hashlib
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from world.build_world import build_world
from world.config_reader import read_world_ranges
from world.episodes import Episode, draw_episode_schedule, episode_state

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
RANGES_PATH = REPO_ROOT / "config" / "world_ranges.yaml"


@dataclass
class StandinPerson:
    """
    WHAT: Lightweight stand-in entity carrying person identifier and persona name.
    WHY: Isolates episode unit testing from full world construction overhead.
    """

    person_id: str
    persona_name: str


def _make_param_entry(val: float | int) -> dict[str, list[Any]]:
    """
    WHAT: Constructs a 3-tier dictionary with identical bounds across splits.
    WHY: Supplies mock configuration parameter blocks for tests.
    """
    return {"training": [val, val], "test": [val, val], "stress": [val, val]}


def _make_test_episode_ranges(
    share: float = 0.05,
    length: int = 14,
    payers: int = 50,
) -> dict[str, Any]:
    """
    WHAT: Constructs customized ranges dictionary for episode parameters.
    WHY: Enables precise unit tests with controlled fixed parameter values.
    """
    return {
        "parameters": {
            "episode_share_of_ordinary_people_per_year": _make_param_entry(share),
            "episode_length_days": _make_param_entry(length),
            "episode_payer_count": _make_param_entry(payers),
        }
    }


def _check_episode_bounds(ep: Episode, p_len: list[int], p_pay: list[int], n_days: int) -> None:
    """
    WHAT: Asserts temporal boundary and payer count constraints on a single episode.
    WHY: Verifies episode schedule invariants against configuration limits.
    """
    expected_kinds = [
        "wedding_or_medical_collection",
        "chit_fund_or_kitty",
        "college_fest_treasurer",
        "sell_then_repay_loan",
        "pay_bills_for_relatives",
    ]
    assert 0 <= ep.start_minute < ep.end_minute <= n_days * 1440
    duration_days = (ep.end_minute - ep.start_minute) // 1440
    assert p_len[0] <= duration_days <= p_len[1]
    assert p_pay[0] <= ep.payer_count <= p_pay[1]
    assert ep.kind in expected_kinds


def test_e1_shape_and_invariants() -> None:
    """
    WHAT: Tests episode bounds, uniqueness, ordering, and kind set across 50 seeds.
    WHY: Verifies structural invariants using real world ranges and population.
    """
    cfg = read_world_ranges(RANGES_PATH)
    p_len = cfg["parameters"]["episode_length_days"]["training"]
    p_pay = cfg["parameters"]["episode_payer_count"]["training"]
    world = build_world(seed=6, n_people=600, n_shops=40)
    for s in range(50):
        rng = np.random.default_rng(s)
        episodes = draw_episode_schedule(cfg, rng, world.people, n_days=60)
        seen_pids: set[str] = set()
        for ep in episodes:
            _check_episode_bounds(ep, p_len, p_pay, 60)
            assert ep.person_id not in seen_pids
            seen_pids.add(ep.person_id)
        for i in range(len(episodes) - 1):
            assert (episodes[i].start_minute, episodes[i].person_id) <= (
                episodes[i + 1].start_minute,
                episodes[i + 1].person_id,
            )


def test_e2_only_ordinary_people() -> None:
    """
    WHAT: Asserts only ordinary personas receive episodes, excluding mule look-alikes.
    WHY: Protects small shopkeepers and gig workers who already appear suspicious in routine life.
    """
    ranges = _make_test_episode_ranges(share=1.0, length=5, payers=20)
    personas = ["salaried_employee", "student", "retiree", "small_shopkeeper", "gig_worker"]
    people: list[StandinPerson] = []
    for p_name in personas:
        for i in range(1500):
            people.append(StandinPerson(person_id=f"{p_name}_{i}", persona_name=p_name))
    rng = np.random.default_rng(21)
    episodes = draw_episode_schedule(ranges, rng, people, n_days=60)
    pids = {ep.person_id for ep in episodes}
    for i in range(1500):
        assert f"small_shopkeeper_{i}" not in pids
        assert f"gig_worker_{i}" not in pids
    for p_name in ["salaried_employee", "student", "retiree"]:
        assert any(ep.person_id.startswith(p_name) for ep in episodes)


def test_e3_episode_rate() -> None:
    """
    WHAT: Tests aggregate episode count matches annualized rate within 15 percent margin.
    WHY: Confirms ordinary person episode frequencies conform to configured expectations.
    """
    ranges = _make_test_episode_ranges(share=1.0, length=5, payers=20)
    people = [StandinPerson(person_id=f"p_{i}", persona_name="salaried_employee") for i in range(3000)]
    rng = np.random.default_rng(22)
    episodes = draw_episode_schedule(ranges, rng, people, n_days=60)
    expected = 3000 * 1.0 * 60 / 365
    measured = len(episodes)
    assert abs(measured - expected) / expected <= 0.15


def test_e4_start_and_end_states() -> None:
    """
    WHAT: Tests episode_state transition precision at start, during, and end boundaries.
    WHY: Verifies ordinary individuals resume normal state after episode completion.
    """
    ranges = _make_test_episode_ranges(share=1.0, length=5, payers=20)
    people = [StandinPerson(person_id=f"p_{i}", persona_name="salaried_employee") for i in range(100)]
    rng = np.random.default_rng(23)
    episodes = draw_episode_schedule(ranges, rng, people, n_days=60)
    ep = episodes[0]
    assert episode_state(ep.person_id, ep.start_minute - 1, episodes) == "normal"
    assert episode_state(ep.person_id, ep.start_minute, episodes) == ep.kind
    assert episode_state(ep.person_id, ep.end_minute - 1, episodes) == ep.kind
    assert episode_state(ep.person_id, ep.end_minute, episodes) == "normal"
    assert episode_state(ep.person_id, 60 * 1440 - 1, episodes) == "normal"
    assert episode_state("unassigned_person", ep.start_minute, episodes) == "normal"


def test_e5_no_hard_coded_time() -> None:
    """
    WHAT: Tests diversity of episode start minutes across 100 seeds.
    WHY: Proves episodes do not anchor to hard-coded calendar schedules.
    """
    ranges = _make_test_episode_ranges(share=1.0, length=5, payers=20)
    people = [StandinPerson(person_id=f"p_{i}", persona_name="salaried_employee") for i in range(200)]
    start_minutes: set[int] = set()
    for s in range(100):
        rng = np.random.default_rng(s)
        episodes = draw_episode_schedule(ranges, rng, people, n_days=60)
        for ep in episodes:
            start_minutes.add(ep.start_minute)
    assert len(start_minutes) >= 50


def _check_e6_refusals(ranges: dict[str, Any], people: list[Any]) -> None:
    """
    WHAT: Asserts input validation errors for invalid duration, empty entities, and splits.
    WHY: Verifies parameter boundaries defensively reject nonsensical simulation inputs.
    """
    rng = np.random.default_rng(0)
    for bad_days in [1, 0, -3, 2.5, "60"]:
        with pytest.raises(ValueError):
            draw_episode_schedule(ranges, rng, people, n_days=bad_days)  # type: ignore
    with pytest.raises(ValueError, match="people"):
        draw_episode_schedule(ranges, rng, [], n_days=60)
    with pytest.raises(ValueError, match="range name"):
        draw_episode_schedule(ranges, rng, people, n_days=60, range_name="unknown")
    cfg = read_world_ranges(RANGES_PATH)
    with pytest.raises(ValueError, match="smaller than"):
        draw_episode_schedule(cfg, rng, people, n_days=20, range_name="training")
    with pytest.raises(ValueError, match="smaller than"):
        draw_episode_schedule(cfg, rng, people, n_days=60, range_name="stress")


def test_e6_refusals() -> None:
    """
    WHAT: Tests argument refusals for draw_episode_schedule and episode_state.
    WHY: Verifies defensive programming on schedule generators and state query routines.
    """
    ranges = _make_test_episode_ranges(share=1.0, length=5, payers=20)
    people = [StandinPerson(person_id="p1", persona_name="salaried_employee")]
    _check_e6_refusals(ranges, people)
    for bad_min in [-1, -50, 1.5, "100", True]:
        with pytest.raises(ValueError, match="minute"):
            episode_state("p1", bad_min, [])  # type: ignore


def _run_subprocess_episode_hash(seed: int) -> str:
    """
    WHAT: Executes fresh Python process to compute episode schedule hash.
    WHY: Proves cross-process determinism without in-memory state caching.
    """
    py_code = (
        "import hashlib, numpy as np; "
        "from world.episodes import draw_episode_schedule; "
        "from tests.unit.world.test_world_episodes import StandinPerson, _make_test_episode_ranges; "
        f"rng = np.random.default_rng({seed}); "
        "p = [StandinPerson(f'p_{i}', 'salaried_employee') for i in range(100)]; "
        "r = _make_test_episode_ranges(share=0.5, length=5); "
        "ev = draw_episode_schedule(r, rng, p, n_days=60); "
        "print(hashlib.sha256(repr(ev).encode('utf-8')).hexdigest())"
    )
    cmd = [sys.executable, "-B", "-c", py_code]
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()


def test_e7_determinism() -> None:
    """
    WHAT: Tests cryptographic determinism across seeds and distinct processes.
    WHY: Enforces reproducible audit trails for all episode schedules.
    """
    ranges = _make_test_episode_ranges(share=0.5, length=5)
    people = [StandinPerson(person_id=f"p_{i}", persona_name="salaried_employee") for i in range(100)]
    rng1 = np.random.default_rng(7)
    ev1 = draw_episode_schedule(ranges, rng1, people, n_days=60)
    rng2 = np.random.default_rng(7)
    ev2 = draw_episode_schedule(ranges, rng2, people, n_days=60)
    assert ev1 == ev2

    h1 = hashlib.sha256(repr(ev1).encode("utf-8")).hexdigest()
    assert _run_subprocess_episode_hash(7) == h1

    differs = False
    for s in range(1, 21):
        rng_diff = np.random.default_rng(s)
        if draw_episode_schedule(ranges, rng_diff, people, n_days=60) != ev1:
            differs = True
            break
    assert differs
