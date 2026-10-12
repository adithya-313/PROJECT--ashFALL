"""
WHAT: Unit tests for diurnal, weekly, monthly, and festival rhythm multipliers.
WHY: Proves rhythm multipliers are normalized, deterministic, persona-differentiated,
     and respect configured parameter bounds.
ISSUE IDs: R2 (always-on rhythms).
TERM: rhythm multiplier: A time-varying factor adjusting payment activity rates.
"""

import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from world.background_config import read_background_realism
from world.clock import WORLD_START_DATE, SimClock
from world.festivals import draw_festival_calendar
from world.personas import PERSONAS
from world.rhythms import (
    RhythmSettings,
    activity_multiplier,
    daily_curve,
    daily_multiplier,
    draw_rhythm_settings,
    hour_weight,
    month_start_multiplier,
    weekend_multiplier,
)

CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "background_realism.yaml"


def _build_dummy_settings(
    uplift: float = 0.10,
    easing_days: int = 10,
    weekend_diff: float = -0.05,
    night_floor: float = 0.10,
    peak_width: float = 3.0,
) -> RhythmSettings:
    """
    WHAT: Creates a deterministic RhythmSettings fixture for unit testing.
    WHY: Isolates specific multiplier mathematical formulas from parameter drawing.
    """
    peaks = {
        "salaried_employee": 19.0,
        "student": 21.0,
        "retiree": 11.0,
        "small_shopkeeper": 15.0,
        "gig_worker": 18.0,
    }
    return RhythmSettings(
        month_start_uplift=uplift,
        month_start_easing_days=easing_days,
        weekend_difference=weekend_diff,
        night_floor=night_floor,
        peak_width_hours=peak_width,
        peak_hours=peaks,
    )


def test_r1_month_start_multiplier() -> None:
    """
    WHAT: Tests R1 month-start multiplier curve over days of month.
    WHY: Verifies day 1 peak, non-increasing fade, and exact 1.0 beyond easing window.
    """
    settings = _build_dummy_settings(uplift=0.10, easing_days=10)
    assert month_start_multiplier(1, settings) == pytest.approx(1.10)
    prev = month_start_multiplier(1, settings)
    for day in range(2, 11):
        curr = month_start_multiplier(day, settings)
        assert curr <= prev
        prev = curr
    assert month_start_multiplier(11, settings) == pytest.approx(1.0)
    assert month_start_multiplier(15, settings) == pytest.approx(1.0)


def test_r2_weekend_multiplier() -> None:
    """
    WHAT: Tests R2 weekend multiplier for Saturday/Sunday vs weekdays.
    WHY: Verifies weekend adjustment factor and identity when difference is zero.
    """
    settings = _build_dummy_settings(weekend_diff=-0.05)
    assert weekend_multiplier(True, settings) == pytest.approx(0.95)
    assert weekend_multiplier(False, settings) == pytest.approx(1.0)

    neutral = _build_dummy_settings(weekend_diff=0.0)
    assert weekend_multiplier(True, neutral) == pytest.approx(1.0)
    assert weekend_multiplier(False, neutral) == pytest.approx(1.0)


def test_r3_hour_weight_properties() -> None:
    """
    WHAT: Tests R3 diurnal Gaussian curve normalization and night-floor bound.
    WHY: Verifies daily average weight is 1.0 and distance decay matches spec formula.
    """
    settings = _build_dummy_settings(night_floor=0.10, peak_width=3.0)
    for persona in PERSONAS:
        weights = [hour_weight(h, persona, settings) for h in range(24)]
        assert sum(weights) / 24.0 == pytest.approx(1.0, abs=1e-9)
        peak_h = settings.peak_hours[persona]
        peak_weight = hour_weight(peak_h, persona, settings)
        assert all(w <= peak_weight + 1e-9 for w in weights)
        h_12 = (peak_h + 12.0) % 24.0
        ratio = hour_weight(h_12, persona, settings) / peak_weight
        expected_raw = 0.10 + 0.90 * np.exp(-0.5 * (12.0 / 3.0) ** 2)
        assert ratio == pytest.approx(expected_raw, rel=1e-6)


def test_r4_personas_differ() -> None:
    """
    WHAT: Tests R4 diurnal weight variation across distinct personas.
    WHY: Verifies retiree activity peaks before student activity in daytime vs evening.
    """
    settings = _build_dummy_settings()
    retiree_11 = hour_weight(11, "retiree", settings)
    student_11 = hour_weight(11, "student", settings)
    assert retiree_11 > student_11
    student_21 = hour_weight(21, "student", settings)
    retiree_21 = hour_weight(21, "retiree", settings)
    assert student_21 > retiree_21


def test_r5_daily_and_activity_multipliers() -> None:
    """
    WHAT: Tests R5 composite daily and persona activity multipliers at minute counter.
    WHY: Confirms product composition of calendar, monthly, weekend, and hourly terms.
    """
    clock = SimClock(WORLD_START_DATE)
    settings = _build_dummy_settings()
    festivals = draw_festival_calendar(
        read_background_realism(CONFIG_PATH), np.random.default_rng(10), n_days=61
    )
    minute = 180  # Day 0, 03:00, 2026-09-01
    dm = daily_multiplier(minute, clock, settings, festivals)
    hw = hour_weight(3, "salaried_employee", settings)
    act = activity_multiplier(minute, "salaried_employee", clock, settings, festivals)
    assert act == pytest.approx(dm * hw)


def test_r6_daily_curve_length_and_positivity() -> None:
    """
    WHAT: Tests R6 daily multiplier curve across an entire world timeline.
    WHY: Verifies generated curve matches total day count and values are all positive.
    """
    clock = SimClock(WORLD_START_DATE)
    ranges = read_background_realism(CONFIG_PATH)
    rng = np.random.default_rng(42)
    settings = draw_rhythm_settings(ranges, rng)
    festivals = draw_festival_calendar(ranges, rng, n_days=61)
    curve = daily_curve(clock, 61, settings, festivals)
    assert len(curve) == 61
    assert all(val > 0.0 for val in curve)


def _compute_simulation_hash(seed: int) -> str:
    """
    WHAT: Evaluates rhythm simulation for a seed and returns a SHA-256 digest.
    WHY: Verifies in-process and cross-process reproducibility.
    """
    clock = SimClock(WORLD_START_DATE)
    ranges = read_background_realism(CONFIG_PATH)
    rng = np.random.default_rng(seed)
    settings = draw_rhythm_settings(ranges, rng)
    festivals = draw_festival_calendar(ranges, rng, n_days=61)
    curve = daily_curve(clock, 61, settings, festivals)
    payload = f"{settings}|{festivals}|{curve}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _subprocess_simulation_hash(seed: int) -> str:
    """
    WHAT: Runs simulation in a fresh Python subprocess and returns SHA-256 digest.
    WHY: Verifies out-of-process determinism without shared interpreter memory.
    """
    script = (
        "from world.clock import WORLD_START_DATE, SimClock;"
        "from world.background_config import read_background_realism;"
        "from world.festivals import draw_festival_calendar;"
        "from world.rhythms import draw_rhythm_settings, daily_curve;"
        "import numpy as np, hashlib;"
        "clock = SimClock(WORLD_START_DATE);"
        "ranges = read_background_realism();"
        "rng = np.random.default_rng(" + str(seed) + ");"
        "settings = draw_rhythm_settings(ranges, rng);"
        "festivals = draw_festival_calendar(ranges, rng, n_days=61);"
        "curve = daily_curve(clock, 61, settings, festivals);"
        "payload = f'{settings}|{festivals}|{curve}';"
        "print(hashlib.sha256(payload.encode('utf-8')).hexdigest())"
    )
    proc = subprocess.run([sys.executable, "-B", "-c", script], capture_output=True, text=True, check=True)
    return proc.stdout.strip()


def test_r7_determinism_same_seed_and_subprocess() -> None:
    """
    WHAT: Tests R7 determinism across repeated runs and an isolated subprocess.
    WHY: Guarantees reproducible simulations without hidden state or random drift.
    """
    hash1 = _compute_simulation_hash(12345)
    hash2 = _compute_simulation_hash(12345)
    assert hash1 == hash2
    ranges = read_background_realism(CONFIG_PATH)
    fest0 = draw_festival_calendar(ranges, np.random.default_rng(0), n_days=61)[0].peak_day
    different = any(
        draw_festival_calendar(ranges, np.random.default_rng(s), n_days=61)[0].peak_day != fest0
        for s in range(1, 21)
    )
    assert different
    assert _subprocess_simulation_hash(12345) == hash1


def test_r8_range_names_and_refusal() -> None:
    """
    WHAT: Tests R8 rhythm parameter sampling across test and stress bounds.
    WHY: Verifies wider bounds sampling and plain refusal on invalid range name.
    """
    ranges = read_background_realism(CONFIG_PATH)
    rng = np.random.default_rng(99)
    test_settings = draw_rhythm_settings(ranges, rng, range_name="test")
    stress_settings = draw_rhythm_settings(ranges, rng, range_name="stress")
    assert test_settings.month_start_easing_days >= 4
    assert stress_settings.month_start_easing_days >= 3
    with pytest.raises(ValueError, match="Unknown range name 'invalid'"):
        draw_rhythm_settings(ranges, rng, range_name="invalid")
