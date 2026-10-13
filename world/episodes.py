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
    """
    if "parameters" in ranges:
        return get_parameter_entry(ranges, key)
    if key in ranges:
        return ranges[key]
    raise ValueError(f"Unknown config parameter key: '{key}'")


def _validate_episode_args(ranges: dict[str, Any], people: list[Any], n_days: int, range_name: str) -> None:
    """
    WHAT: Validates duration, population, range split, and upper bound constraints.
    WHY: Rejects invalid horizons, empty populations, unknown splits, and cramped windows.
    """
    if not isinstance(n_days, int) or isinstance(n_days, bool) or n_days < 2:
        raise ValueError(f"n_days must be an integer >= 2, got {n_days!r}")
    if len(people) == 0:
        raise ValueError("people list cannot be empty")
    if range_name not in VALID_RANGES:
        raise ValueError(f"Unknown range name '{range_name}'; expected one of {VALID_RANGES}")

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
    """
    # Uniform selection across EPISODE_KINDS is a structural choice; no source provides kind shares.
    kind = EPISODE_KINDS[int(rng.integers(0, 5))]
    length_days = draw_whole_number(len_entry, rng, range_name=range_name, param_name="episode_length_days")
    payers = draw_whole_number(payer_entry, rng, range_name=range_name, param_name="episode_payer_count")
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
    """
    episodes: list[Episode] = []
    for p in people:
        persona = PERSONAS.get(p.persona_name)
        if persona is not None and persona.looks_like_mule_in_normal_life:
            continue
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
    """
    _validate_episode_args(ranges, people, n_days, range_name)
    share_entry = _get_entry(ranges, "episode_share_of_ordinary_people_per_year")
    len_entry = _get_entry(ranges, "episode_length_days")
    payer_entry = _get_entry(ranges, "episode_payer_count")

    share = draw_in_range(
        share_entry, rng, range_name=range_name, param_name="episode_share_of_ordinary_people_per_year"
    )
    # Annual share scaled to world duration: 365 days per year is a structural choice.
    chance = share * n_days / DAYS_PER_YEAR
    episodes = _collect_ordinary_episodes(people, chance, n_days, len_entry, payer_entry, range_name, rng)
    return sorted(episodes, key=_episode_sort_key)


def episode_state(person_id: str, minute: int, episodes: list[Episode]) -> str:
    """
    WHAT: Returns active episode kind if person is currently in an episode, else 'normal'.
    WHY: Allows simulators to determine whether an account is undergoing a high-velocity spike.
    """
    if not isinstance(minute, int) or isinstance(minute, bool) or minute < 0:
        raise ValueError(f"Event minute must be a non-negative integer, got {minute!r}")

    for ep in episodes:
        if ep.person_id == person_id and ep.start_minute <= minute < ep.end_minute:
            return ep.kind
    return "normal"
