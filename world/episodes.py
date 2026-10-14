"""
WHAT: Episode schedule generator and person episode state evaluator.
WHY: Simulates temporary burst periods of incoming payments for ordinary citizens (issue A1).
ISSUE IDs: A1 (episode and background realism).
TERM: episode: A bounded temporal window during which an ordinary citizen's payments mimic mule activity.
TERM: episode state: The current behavioral mode of an individual (either 'normal' or the active episode kind).

NOTE: This phase only SCHEDULES episodes. Generating the episode's payments and checking the
ledger-level return to normal belongs to the wiring phase (11B-3).
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from world.background_config import draw_whole_number
from world.config_reader import draw_in_range, get_parameter_entry
from world.personas import PERSONAS

EPISODE_KINDS: list[str] = [
    "wedding_or_medical_collection",
    "chit_fund_or_kitty",
    "college_fest_treasurer",
    "sell_then_repay_loan",
    "pay_bills_for_relatives",
]

MINUTES_PER_DAY: int = 1440
DAYS_PER_YEAR: int = 365
VALID_RANGES: tuple[str, ...] = ("training", "test", "stress")


@dataclass(frozen=True)
class Episode:
    """
    WHAT: Immutable specification of a time-limited high-inflow episode for one individual.
    WHY: Encapsulates episode category, timing bounds, and expected counterparty count.
    """

    person_id: str
    kind: str
    start_minute: int
    end_minute: int
    payer_count: int


def _get_entry(ranges: dict[str, Any], key: str) -> dict[str, Any]:
    """
    WHAT: Retrieves parameter dictionary block from raw or nested config dictionary.
    WHY: Supports both master world ranges file and lightweight hand-built test dictionaries.

    Args:
        ranges: Configuration dictionary containing parameters under "parameters" or direct keys.
        key: Name of the configuration parameter to retrieve.

    Returns:
        Dictionary mapping split names to parameter bound tuples or lists.

    Raises:
        ValueError: If key is not found in either nested or top-level dictionary.

    Example:
        >>> from world.episodes import _get_entry
        >>> d = {"episode_length_days": {"training": [3, 14]}}
        >>> entry = _get_entry(d, "episode_length_days")
        >>> entry["training"]
        [3, 14]
    """
    # Check for master world ranges hierarchy with top-level parameters dictionary.
    if "parameters" in ranges:
        return get_parameter_entry(ranges, key)
    # Check for direct key mapping used in test configuration dictionaries.
    if key in ranges:
        return ranges[key]
    raise ValueError(f"Unknown config parameter key: '{key}'")


def _validate_episode_args(ranges: dict[str, Any], people: list[Any], n_days: int, range_name: str) -> None:
    """
    WHAT: Validates duration, population, range split, and upper bound constraints.
    WHY: Rejects invalid horizons, empty populations, unknown splits, and cramped windows.

    Args:
        ranges: Configuration dictionary containing episode parameter blocks.
        people: List of simulated Person instances in the world.
        n_days: Integer duration of the simulated world in full days.
        range_name: Target dataset split name (one of "training", "test", "stress").

    Returns:
        None.

    Raises:
        ValueError: If n_days is not an integer >= 2, people is empty, range_name is
            unknown, or n_days is shorter than the upper bound of episode_length_days.

    Example:
        >>> from world.episodes import _validate_episode_args
        >>> ranges = {"episode_length_days": {"training": [3, 14]}}
        >>> _validate_episode_args(ranges, ["dummy_person"], 30, "training")
    """
    # Verify that the simulation horizon is an integer of at least two days.
    if not isinstance(n_days, int) or isinstance(n_days, bool) or n_days < 2:
        raise ValueError(f"n_days must be an integer >= 2, got {n_days!r}")
    # Ensure that the population is non-empty.
    if len(people) == 0:
        raise ValueError("people list cannot be empty")
    # Validate the dataset split against recognized simulation ranges.
    if range_name not in VALID_RANGES:
        raise ValueError(f"Unknown range name '{range_name}'; expected one of {VALID_RANGES}")

    # Ensure simulation duration is long enough to fit the longest possible episode.
    len_entry = _get_entry(ranges, "episode_length_days")
    upper_bound = len_entry[range_name][1]
    if n_days < upper_bound:
        raise ValueError(
            f"n_days ({n_days}) is smaller than upper bound of episode_length_days ({upper_bound}) "
            f"for range '{range_name}'"
        )


def _draw_single_person_episode(
    person_id: str,
    n_days: int,
    len_entry: dict[str, Any],
    payer_entry: dict[str, Any],
    range_name: str,
    rng: np.random.Generator,
) -> Episode:
    """
    WHAT: Draws episode attributes (kind, length, payers, timing) for an ordinary person.
    WHY: Samples discrete episode parameters uniformly from configured bands.

    Args:
        person_id: Unique string identifier of the individual undergoing the episode.
        n_days: Total duration of the simulation in full days.
        len_entry: Parameter dictionary bounding episode length in days.
        payer_entry: Parameter dictionary bounding number of incoming counterparties.
        range_name: Target dataset split name (e.g. "training").
        rng: NumPy pseudo-random generator instance for deterministic draws.

    Returns:
        Episode dataclass instance specifying kind, timing bounds, and payer count.

    Example:
        >>> import numpy as np
        >>> from world.episodes import _draw_single_person_episode
        >>> rng = np.random.default_rng(42)
        >>> len_entry = {"training": [3, 14]}
        >>> payer_entry = {"training": [10, 30]}
        >>> ep = _draw_single_person_episode("p1", 30, len_entry, payer_entry, "training", rng)
        >>> (ep.kind, ep.start_minute, ep.end_minute, ep.payer_count)
        ('wedding_or_medical_collection', 11376, 28656, 23)
    """
    # Uniform selection across EPISODE_KINDS is a structural choice; no source provides kind shares.
    kind = EPISODE_KINDS[int(rng.integers(0, 5))]
    # Draw duration and counterparty count from configured uniform parameter ranges.
    length_days = draw_whole_number(len_entry, rng, range_name=range_name, param_name="episode_length_days")
    payers = draw_whole_number(payer_entry, rng, range_name=range_name, param_name="episode_payer_count")
    # Uniformly position the episode within the simulation horizon without overflowing.
    max_start_min = n_days * MINUTES_PER_DAY - length_days * MINUTES_PER_DAY
    start_minute = int(rng.integers(0, max_start_min, endpoint=True))
    end_minute = start_minute + length_days * MINUTES_PER_DAY
    return Episode(
        person_id=person_id,
        kind=kind,
        start_minute=start_minute,
        end_minute=end_minute,
        payer_count=payers,
    )


def _episode_sort_key(ep: Episode) -> tuple[int, str]:
    """
    WHAT: Sort key extracting start_minute and person_id.
    WHY: Orders episodes chronologically and stably without using lambdas.

    Args:
        ep: Episode instance to extract ordering tuple from.

    Returns:
        Tuple of (start_minute, person_id) for chronological ordering.

    Example:
        >>> from world.episodes import Episode, _episode_sort_key
        >>> ep = Episode("p1", "chit_fund_or_kitty", 1440, 4320, 15)
        >>> _episode_sort_key(ep)
        (1440, 'p1')
    """
    return (ep.start_minute, ep.person_id)


def _collect_ordinary_episodes(
    people: list[Any],
    chance: float,
    n_days: int,
    len_entry: dict[str, Any],
    payer_entry: dict[str, Any],
    range_name: str,
    rng: np.random.Generator,
) -> list[Episode]:
    """
    WHAT: Iterates over population and samples episodes for non-mule personas.
    WHY: Isolates random person loop to keep function lengths strictly under 30 lines.

    Args:
        people: List of citizen Person instances eligible for episode evaluation.
        chance: Float probability that an ordinary citizen experiences an episode.
        n_days: Total duration of the simulation in full days.
        len_entry: Parameter dictionary bounding episode length in days.
        payer_entry: Parameter dictionary bounding number of incoming counterparties.
        range_name: Target dataset split name (e.g. "training").
        rng: NumPy pseudo-random generator instance for deterministic draws.

    Returns:
        List of Episode instances scheduled for qualifying ordinary citizens.

    Example:
        >>> import numpy as np
        >>> from world.people import Person
        >>> from world.episodes import _collect_ordinary_episodes
        >>> p1 = Person("p1", "salaried_employee", 1, 0, "a1", False, None, None, None, 0)
        >>> p2 = Person("p2", "gig_worker", 1, 0, "a2", False, None, None, None, 0)
        >>> bounds = {"training": [3, 14]}
        >>> eps = _collect_ordinary_episodes([p1, p2], 1.0, 30, bounds, bounds, "training", np.random.default_rng(42))
        >>> [e.person_id for e in eps]
        ['p1']
    """
    episodes: list[Episode] = []
    # Filter out mule look-alikes to prevent redundant deceptive overlays.
    for p in people:
        persona = PERSONAS.get(p.persona_name)
        if persona is not None and persona.looks_like_mule_in_normal_life:
            continue
        # Evaluate independent Bernoulli trial against calibrated annual episode frequency.
        if float(rng.random()) < chance:
            episodes.append(
                _draw_single_person_episode(p.person_id, n_days, len_entry, payer_entry, range_name, rng)
            )
    return episodes


def draw_episode_schedule(
    ranges: dict[str, Any],
    rng: np.random.Generator,
    people: list[Any],
    n_days: int,
    range_name: str = "training",
) -> list[Episode]:
    """
    WHAT: Generates full schedule of deceptive episodes for ordinary individuals.
    WHY: Provides background adversarial distractors without changing people or ledgers.

    Args:
        ranges: Configuration dictionary containing episode parameter ranges.
        rng: NumPy pseudo-random generator instance for deterministic draws.
        people: List of simulated Person instances in the world.
        n_days: Integer duration of the simulated world in full days.
        range_name: Target dataset split name (defaults to "training").

    Returns:
        Chronologically sorted list of Episode instances across the simulation horizon.

    Raises:
        ValueError: If validation of arguments or horizon bounds fails.

    Example:
        >>> import numpy as np
        >>> from world.config_reader import read_world_ranges
        >>> from world.people import Person
        >>> from world.episodes import draw_episode_schedule
        >>> ranges = read_world_ranges()
        >>> people = [Person(f"p{i}", "student", 1, 0, f"a{i}", False, None, None, None, 0) for i in range(100)]
        >>> eps = draw_episode_schedule(ranges, np.random.default_rng(42), people, 60)
        >>> (len(eps), eps[0].person_id, eps[0].kind, eps[0].start_minute)
        (1, 'p50', 'sell_then_repay_loan', 33509)
    """
    # Enforce simulation horizon, non-empty population, and range constraints.
    _validate_episode_args(ranges, people, n_days, range_name)
    share_entry = _get_entry(ranges, "episode_share_of_ordinary_people_per_year")
    len_entry = _get_entry(ranges, "episode_length_days")
    payer_entry = _get_entry(ranges, "episode_payer_count")

    # Sample annualized episode participation rate and scale proportionally to days simulated.
    share = draw_in_range(
        share_entry, rng, range_name=range_name, param_name="episode_share_of_ordinary_people_per_year"
    )
    # Annual share scaled to world duration: 365 days per year is a structural choice.
    chance = share * n_days / DAYS_PER_YEAR
    # Generate individual burst episodes for eligible non-mule citizens.
    episodes = _collect_ordinary_episodes(people, chance, n_days, len_entry, payer_entry, range_name, rng)
    return sorted(episodes, key=_episode_sort_key)


def episode_state(person_id: str, minute: int, episodes: list[Episode]) -> str:
    """
    WHAT: Returns active episode kind if person is currently in an episode, else 'normal'.
    WHY: Allows simulators to determine whether an account is undergoing a high-velocity spike.

    Args:
        person_id: Unique string identifier of the person to inspect.
        minute: Non-negative integer minute offset within the simulation.
        episodes: List of scheduled Episode instances for the world.

    Returns:
        Active episode kind string if the minute falls in an episode, otherwise "normal".

    Raises:
        ValueError: If minute is not a non-negative integer.

    Example:
        >>> from world.episodes import Episode, episode_state
        >>> ep = Episode("p1", "chit_fund_or_kitty", 1440, 2880, 15)
        >>> episode_state("p1", 2000, [ep])
        'chit_fund_or_kitty'
        >>> episode_state("p1", 3000, [ep])
        'normal'
    """
    # Verify that the minute offset is a non-negative integer.
    if not isinstance(minute, int) or isinstance(minute, bool) or minute < 0:
        raise ValueError(f"Event minute must be a non-negative integer, got {minute!r}")

    # Inspect all scheduled episodes to find an active time window for this citizen.
    for ep in episodes:
        if ep.person_id == person_id and ep.start_minute <= minute < ep.end_minute:
            return ep.kind
    return "normal"
