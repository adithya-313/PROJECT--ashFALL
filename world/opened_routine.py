"""
WHAT: Gives a genuine person who opens an account mid-run a normal routine life (Level 12c C-3a): decides whether the
person is idle or has a routine, and plans the routine's payments from the opening minute to the end of the run with
Level 11's own payment generators.

WHY: A new account with no payments at all would be recognisable as new and idle, which would single out gang-opened
accounts. Using Level 11's generators, called and not changed, gives new people the same kinds of payments, amounts and
times of day as everyone else. New shops get no customers: that would change world.shops and shift every existing
shopper's draws.

TERM: idle person: a new genuine person with no routine payments at all, drawn with the no-routine share.
TERM: routine: salary, rent and bills for a salaried person, everyday shop payments, and inbound peer transfers for
      a persona that looks like a mule in normal life.
TERM: stand-in world: a small object with only the people and shops a Level 11 generator reads.
TERM: no-routine share: the share of new genuine people who stay idle, drawn once per run from config.
"""

import calendar
from types import SimpleNamespace
from typing import Any

import numpy as np

from world.calendar_runner import (
    PlannedPayment,
    _generate_day_everyday_payments,
    _generate_fixed_items,
    _sample_lookalike_inbounds,
    persona_hour_probabilities,
)
from world.personas import get_persona
from world.placeholders import LOOKALIKE_SMALL_INCOMING_PER_MONTH
from world.rhythms import daily_multiplier
from world.world_plan import WorldPlan

MINUTES_PER_DAY: int = 1440
NO_ROUTINE_SETTING: str = "share_of_new_genuine_people_with_no_routine"


def resolve_no_routine_share_range(split: str, config: dict[str, Any]) -> tuple[float, float]:
    """
    WHAT: Returns the range of the share of new genuine people who stay idle, for one split, from an account openings
    configuration mapping.

    WHY: The share is a config value with a label and a reason; the planner must not type it.

    Args:
        split: Split name of the world, e.g. "training".
        config: Parsed config/account_openings.yaml, or a narrowed copy.

    Returns:
        (lower, upper) share bounds.

    Raises:
        ValueError: If the setting or the split is missing, or the range is not two ordered numbers in [0, 1].

    Example:
        resolve_no_routine_share_range("training", load_account_openings_config()) == (0.2, 0.6)
    """
    parameters = config["parameters"]
    if NO_ROUTINE_SETTING not in parameters:
        raise ValueError(f"Account openings config has no setting '{NO_ROUTINE_SETTING}'.")
    entry = parameters[NO_ROUTINE_SETTING]
    if split not in entry:
        raise ValueError(f"Account openings setting '{NO_ROUTINE_SETTING}' has no range for split '{split}'.")
    bounds = entry[split]
    if not isinstance(bounds, list) or len(bounds) != 2 or not 0.0 <= bounds[0] <= bounds[1] <= 1.0:
        raise ValueError(
            f"Account openings setting '{NO_ROUTINE_SETTING}' split '{split}' must be two ordered shares in [0, 1]."
        )
    return float(bounds[0]), float(bounds[1])


def draw_no_routine_share(split: str, config: dict[str, Any], rng: np.random.Generator) -> float:
    """
    WHAT: Draws this run's share of new genuine people who stay idle, uniformly from the configured range.

    WHY: One share per run keeps the idle people a stable fraction of the run's openings.

    Args:
        split: Split name of the world.
        config: Parsed account openings configuration.
        rng: The opened account routine stream (child 10); one number is read.

    Returns:
        The share in [0, 1].

    Example:
        With the range narrowed to [1, 1] the share is 1.0 and every new person is idle.
    """
    low_share, high_share = resolve_no_routine_share_range(split, config)
    return float(rng.uniform(low_share, high_share))


def is_idle(no_routine_share: float, rng: np.random.Generator) -> bool:
    """
    WHAT: Decides whether one new person is idle: a fresh number below the no-routine share means idle.

    WHY: One draw per new person, taken the moment the account opens, keeps the decision independent of the others.

    Args:
        no_routine_share: This run's share of idle new people.
        rng: The opened account routine stream (child 10); one number is read.

    Returns:
        True when the person gets no routine.

    Example:
        is_idle(1.0, rng) is True; is_idle(0.0, rng) is False.
    """
    return float(rng.uniform(0.0, 1.0)) < no_routine_share


def _lookalike_payments_for_day(
    person: Any, people: list[Any], plan: WorldPlan, day: int, probs: dict[str, np.ndarray],
    rng: np.random.Generator, start_index: int,
) -> tuple[list[PlannedPayment], int]:
    """
    WHAT: Plans one day of inbound peer transfers for a lookalike person: a Poisson count at the daily rate, then
    Level 11's _sample_lookalike_inbounds picks the senders and times from every person except the new one.

    WHY: Level 11 sizes this count inside a function that loops over all people, so it cannot be called for one new
    person. The rate is rebuilt from the same constant and rhythm function Level 11 uses; only the loop is not reused.

    Args:
        person: The new lookalike person, last in people.
        people: Every person that can send, with the new person last.
        plan: The WorldPlan (clock, rhythm and festivals).
        day: Day index.
        probs: Hourly probabilities per persona.
        rng: The opened account routine stream (child 10).
        start_index: First creation index to use.

    Returns:
        (payments, next creation index).

    Example:
        A lookalike person gets Poisson(rate) small transfers per day from other people.
    """
    date = plan.clock.to_datetime(day * MINUTES_PER_DAY)
    days_in_month = calendar.monthrange(date.year, date.month)[1]
    base_rate = LOOKALIKE_SMALL_INCOMING_PER_MONTH / float(days_in_month)
    boost = daily_multiplier(day * MINUTES_PER_DAY, plan.clock, plan.rhythm_settings, plan.festivals)
    count = int(rng.poisson(base_rate * boost))
    if count == 0:
        return [], start_index
    return _sample_lookalike_inbounds(person, len(people) - 1, people, day, probs, count, rng, start_index)


def plan_opened_routine(
    world: Any, plan: WorldPlan, person: Any, open_minute: int, rng: np.random.Generator, start_index: int
) -> tuple[list[PlannedPayment], int]:
    """
    WHAT: Plans a new person's routine payments from the opening minute to the end of the run. Salary, rent and bills
    come from Level 11's fixed item generator, everyday payments from its daily generator, both called with a stand-in
    world holding only this person; a lookalike persona also gets inbound peer transfers. Only payments at or after
    the opening minute are kept. Days before the opening day are not generated.

    WHY: The person did not exist before the opening minute, and Level 11's generators already know how a person of
    each persona pays and is paid. All draws come from one stream so they never touch Level 11's.

    Args:
        world: The World; its shops are the stand-in world's shops and its people are the possible senders.
        plan: The WorldPlan.
        person: The new Person, already registered with the runner.
        open_minute: Minute at which the account was opened.
        rng: The opened account routine stream (child 10).
        start_index: First creation index to use, so ids stay unique across people.

    Returns:
        (payments in generation order, next creation index).

    Example:
        A new salaried person opened on day 20 gets salary and bills of later months and everyday payments from day 20.
    """
    stand_in = SimpleNamespace(people=[person], shops=world.shops)
    probs = persona_hour_probabilities(plan)
    payments: list[PlannedPayment] = []
    index = start_index

    fixed, index = _generate_fixed_items(stand_in, plan, rng, index)
    payments.extend(fixed)

    senders = list(world.people) + [person]
    is_lookalike = get_persona(person.persona_name).looks_like_mule_in_normal_life
    for day in range(open_minute // MINUTES_PER_DAY, plan.n_days):
        everyday, index = _generate_day_everyday_payments(stand_in, plan, day, probs, rng, index)
        payments.extend(everyday)
        if is_lookalike and len(senders) > 1:
            inbound, index = _lookalike_payments_for_day(person, senders, plan, day, probs, rng, index)
            payments.extend(inbound)

    return [payment for payment in payments if payment.minute >= open_minute], index
