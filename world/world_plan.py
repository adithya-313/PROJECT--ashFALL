"""
WHAT: World plan orchestration module bundling pre-drawn temporal rhythms, festival calendars, and outages.
WHY: Draws all world-level schedule variations once upfront to ensure consistent and deterministic playback.
TERM: world plan: a complete calendar schedule made before the run starts,
      e.g. picking which days have holidays or bank slowdowns.
TERM: simulated clock: a counter turning minutes into calendar dates,
      e.g. minute 1440 means day 2 at midnight.
TERM: temporal rhythms: daily and weekly patterns of human activity,
      e.g. people buy more groceries at noon than at 3 AM.
TERM: festival calendar: a list of holiday dates that boost shopping,
      e.g. Diwali increasing store sales for five days.
TERM: incident schedule: a planned timetable of outages,
      e.g. a bank server going down from 10 AM to 2 PM.
TERM: failure settings: everyday reasons payments fail,
      e.g. typing the wrong PIN or losing phone signal.
TERM: seed sequence: a method turning one starting number into several separate random number streams,
      e.g. splitting seed 42 so planning and running never share numbers.
TERM: spawn: creating new independent random number streams from one parent,
      e.g. giving one stream to the planner and another to the payment runner.
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
    # Child streams 0 and 1 are the plan and run streams. Never reorder them: every saved world would change.
    return np.random.default_rng(children[0]), np.random.default_rng(children[1])


def spawn_life_streams(seed: int) -> tuple[np.random.Generator, np.random.Generator]:
    """
    WHAT: Derives two auxiliary independent NumPy random number generators from a root seed integer.
    WHY: Decouples device and episode lifecycle generation from the primary plan and run simulation streams.
         Spawning four children guarantees that children 2 and 3 never perturb the random draws of children 0 and 1.
    Args:
        seed: Root integer seed initializing the high-entropy SeedSequence algorithm.
    Returns:
        Tuple of (device_rng, episode_rng) providing independent random streams.
    Example:
        dev_rng, ep_rng = spawn_life_streams(42)
        assert dev_rng.uniform() != ep_rng.uniform()
    """
    # Initialize a master seed sequence to generate statistically independent sub-seeds.
    seed_seq = np.random.SeedSequence(seed)
    # Spawn four child seed sequences, reserving slots 2 for devices and 3 for future episodes.
    children = seed_seq.spawn(4)
    # Child streams 2 and 3 are the device and episode streams, added later so streams 0 and 1 stay unchanged.
    return np.random.default_rng(children[2]), np.random.default_rng(children[3])
