"""
WHAT: World plan orchestration module bundling pre-drawn temporal rhythms, festival calendars, and outages.
WHY: Draws all world-level schedule variations once upfront to ensure consistent and deterministic playback.
TERM: world plan: An immutable specification of calendar dynamics and outage schedules drawn before simulation.
TERM: simulated clock: A chronological mapper translating continuous elapsed minutes into calendar dates.
TERM: temporal rhythms: Periodic diurnal, weekly, and monthly payment activity scaling curves.
TERM: festival calendar: Stochastically generated holiday peaks with piecewise linear rise and fade multipliers.
TERM: incident schedule: Chronologically ordered downtime intervals for central switches and banks.
TERM: failure settings: Background transaction decline rates covering PIN errors, network drops, and technical declines.
TERM: seed sequence: An algorithm expanding a root integer into high-entropy, independent seed states.
TERM: spawn: The process of creating isolated, non-overlapping pseudo-random streams from one parent seed.
"""

from dataclasses import dataclass

import numpy as np

from world.background_config import read_background_realism
from world.clock import WORLD_START_DATE, SimClock
from world.failures import FailureSettings, draw_failure_settings
from world.festivals import Festival, draw_festival_calendar
from world.incidents import Incident, draw_incident_schedule
from world.rhythms import RhythmSettings, draw_rhythm_settings


@dataclass(frozen=True)
class WorldPlan:
    """
    WHAT: Container holding drawn temporal rhythms, festival calendars, and system failure schedules.
    WHY: Encapsulates all stochastic world dynamics in a single immutable structure for reproducible runs.
    """

    n_days: int
    clock: SimClock
    festivals: list[Festival]
    rhythm_settings: RhythmSettings
    incidents: list[Incident]
    failure_settings: FailureSettings
    range_name: str


def draw_world_plan(
    rng: np.random.Generator, n_days: int = 61, range_name: str = "training", n_banks: int = 3
) -> WorldPlan:
    """
    WHAT: Pre-draws all calendar and failure schedule dynamics for a simulated world run.
    WHY: Establishes an immutable, seed-driven temporal timeline before payments execute.
    Args: rng: Generator for deterministic draws; n_days: horizon in days; range_name: split; n_banks: count.
    Returns: WorldPlan holding frozen clock, festivals, rhythms, incidents, and failure settings.
    Example: plan = draw_world_plan(np.random.default_rng(42), n_days=61, range_name="training", n_banks=3)
    """
    # 1. Initialize simulation clock with world start date (2026-09-01).
    clock = SimClock(WORLD_START_DATE)
    # 2. Read background realism and failure realism configurations.
    bg_ranges = read_background_realism()
    fail_ranges = read_background_realism("config/failure_realism.yaml")
    # FIXED DRAW ORDER: rhythm settings -> festival calendar -> incident schedule -> failure settings.
    # Step 1: Draw rhythm settings for diurnal, weekly, and monthly multipliers.
    rhythm_settings = draw_rhythm_settings(bg_ranges, rng, range_name=range_name)
    # Step 2: Draw festival calendar; do not catch ValueError if n_days is too short.
    festivals = draw_festival_calendar(bg_ranges, rng, n_days=n_days, range_name=range_name)
    # Step 3: Draw incident schedule for central switch and bank downtime outages.
    incidents = draw_incident_schedule(fail_ranges, rng, n_days=n_days, n_banks=n_banks, range_name=range_name)
    # Step 4: Draw background failure probability settings (PIN, network, technical).
    failure_settings = draw_failure_settings(fail_ranges, rng, range_name=range_name)
    # Return the assembled, frozen world plan container.
    return WorldPlan(n_days, clock, festivals, rhythm_settings, incidents, failure_settings, range_name)


def spawn_world_streams(seed: int) -> tuple[np.random.Generator, np.random.Generator]:
    """
    WHAT: Derives two independent NumPy pseudo-random number generators from a single root seed integer.
    WHY: Prevents hidden correlation between static world plan calendar draws and dynamic simulation execution.
         A seed sequence expands an integer into independent states, and spawn splits it without sequence overlap.
    Args: seed: Root integer seed initializing the high-entropy SeedSequence algorithm.
    Returns: Tuple of (plan_rng, run_rng) providing independent, reproducible random streams.
    Example:
        # plan_rng, run_rng = spawn_world_streams(42)
        # assert plan_rng.uniform() != run_rng.uniform()
    """
    # Initialize a master seed sequence to generate statistically independent sub-seeds.
    seed_seq = np.random.SeedSequence(seed)
    # Spawn two child seed sequences for world plan generation and transaction execution streams.
    children = seed_seq.spawn(2)
    # Return independent random number generators for the static plan and the live runner.
    return np.random.default_rng(children[0]), np.random.default_rng(children[1])
