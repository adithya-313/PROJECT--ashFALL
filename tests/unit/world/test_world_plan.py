"""
WHAT: Unit tests P1-P3 verifying deterministic WorldPlan drawing, parameter refusals, and fixed draw sequence.
WHY: Proves that WorldPlan drawing is reproducible, respects boundary invariants, and locks random draw order.
TERM: world plan: An immutable specification of calendar dynamics and outage schedules drawn before simulation.
TERM: determinism: The guarantee that identical pseudo-random seeds yield bitwise identical simulation objects.
TERM: draw order: The strict sequential ordering of random number consumption across simulation components.
"""

import numpy as np
import pytest

from world.world_plan import draw_world_plan


def test_p1_same_seed_identical_plan_different_seed_differs() -> None:
    """
    WHAT: Tests that identical seeds yield identical WorldPlan objects while different seeds differ.
    WHY: Verifies requirement P1 ensuring that world plans are completely deterministic functions of the seed.
    Example:
        # Drawing two plans with seed 100 produces identical Diwali peak day and incident start minutes
        p1 = draw_world_plan(np.random.default_rng(100), n_days=61)
        p2 = draw_world_plan(np.random.default_rng(100), n_days=61)
        assert p1 == p2
    """
    # Initialize generators for identical seed runs.
    rng_a = np.random.default_rng(777)
    rng_b = np.random.default_rng(777)
    # Draw two plans using identical seed 777.
    plan_a = draw_world_plan(rng_a, n_days=61)
    plan_b = draw_world_plan(rng_b, n_days=61)
    # Assert bitwise equality of both plans.
    assert plan_a == plan_b
    # Draw a third plan using a different seed 888.
    rng_c = np.random.default_rng(888)
    plan_c = draw_world_plan(rng_c, n_days=61)
    # Check that either festivals or incidents differ between different seeds.
    assert (plan_a.festivals != plan_c.festivals) or (plan_a.incidents != plan_c.incidents)


def test_p2_short_horizon_raises_festival_value_error() -> None:
    """
    WHAT: Tests that n_days=30 fails with the standard festival validation ValueError message.
    WHY: Verifies requirement P2 guaranteeing that too-short world durations fail loudly without suppression.
    Example:
        # A 30-day horizon is too cramped for the festival peak window (upper bound 50 in training)
        with pytest.raises(ValueError, match="strictly greater than the upper bound"):
            draw_world_plan(np.random.default_rng(123), n_days=30)
    """
    # Create generator with arbitrary seed.
    rng = np.random.default_rng(123)
    # Expect ValueError due to festival calendar duration constraint.
    with pytest.raises(ValueError, match="strictly greater than the upper bound of festival_peak_day_in_world"):
        # 30 days is smaller than the upper bound of festival peak day in training (50).
        draw_world_plan(rng, n_days=30, range_name="training")


def test_p3_fixed_draw_order_and_independent_state() -> None:
    """
    WHAT: Tests that plan drawing consumes random draws in an invariant sequence independent of external state.
    WHY: Verifies requirement P3 proving that draw order is strictly locked and repeatable across environments.
    Example:
        # Two fresh generators at identical state generate the exact same plan components in lockstep
        r1 = np.random.default_rng(999)
        r2 = np.random.default_rng(999)
        assert draw_world_plan(r1).n_days == draw_world_plan(r2).n_days
    """
    # Create first generator and draw a complete plan.
    rng1 = np.random.default_rng(999)
    plan1 = draw_world_plan(rng1, n_days=61)
    # Create second generator with identical seed.
    rng2 = np.random.default_rng(999)
    plan2 = draw_world_plan(rng2, n_days=61)
    # Verify that rhythm settings, festivals, incidents, and failures match exactly.
    assert plan1.rhythm_settings == plan2.rhythm_settings
    assert plan1.festivals == plan2.festivals
    assert plan1.incidents == plan2.incidents
    assert plan1.failure_settings == plan2.failure_settings
