"""
WHAT: Building block procedural execution routines for gang laundering tactics (Level 12b B-2a).
Provides helper functions for account acquisition, warm-up delay, routing payment generation, and timing disguise.

WHY: Procedural synthesis turns declarative gang configurations into executable tactical schedules.
Isolating block-level logic ensures individual building blocks remain modular, testable, and deterministic.

TERM: building block: atomic tactical unit (acquire, warm_up, route, disguise).
TERM: mule account: ordinary citizen bank account leased or coerced into moving illicit funds.
TERM: routing chain: sequence of multi-hop transfers propagating stolen money across distinct layers.
TERM: timing disguise: intentional perturbation of transfer dispatch times within an account's dwell window.
"""

from typing import Sequence
import numpy as np

from world.build_world import World
from world.calendar_runner import PlannedPayment
from world.tactic_config import (
    get_disguise_jitter_range,
    get_dwell_range,
    get_route_amount_range,
)
from world.truth_records import PaymentStory, RoleSpan

MINUTES_PER_DAY: int = 1440


def acquire_mule_accounts(
    world: World, count: int, rng: np.random.Generator, excluded_accounts: set[str]
) -> list[str]:
    """
    WHAT: Selects a specified count of distinct citizen bank accounts from the world to serve as mules.
    It collects all person account identifiers, strictly filters out shop accounts and excluded accounts
    (such as fixture sources or previously leased mules), sorts candidates for determinism, and draws
    the requested count without replacement using the provided gang random number generator.

    WHY: Ground Rule 15 and Decision D7 prohibit syndicates from operating commercial shop accounts or stealing
    from accounts assigned to other concurrent campaigns; mules must be authentic citizen accounts.

    Args:
        world: The simulated World instance containing registered people and shops.
        count: Integer number of distinct mule accounts to recruit.
        rng: Gang-specific random number generator stream.
        excluded_accounts: Set of account identifier strings that must not be selected.

    Returns:
        List of selected unique account identifier strings.

    Example:
        # mules = acquire_mule_accounts(world, 3, gang_rng, {"acc_collector_01"})
        # assert len(mules) == 3 and not any(m in excluded_accounts for m in mules)
    """
    # 1. Identify all shop accounts to ensure commercial entities are never recruited as mules.
    shop_accounts = {shop.account_id for shop in world.shops}
    # 2. Filter eligible citizen accounts that are not shops and not in the exclusion set.
    eligible = [
        person.account_id
        for person in world.people
        if person.account_id not in shop_accounts and person.account_id not in excluded_accounts
    ]
    if len(eligible) < count:
        raise ValueError(
            f"Cannot acquire {count} mule accounts; only {len(eligible)} eligible citizen accounts available."
        )
    # 3. Sort eligible accounts to ensure permutation determinism across platforms.
    eligible.sort()
    chosen_indices = rng.choice(len(eligible), size=count, replace=False)
    return [eligible[idx] for idx in chosen_indices]


def advance_warmup_clock(current_minute: int, days: int) -> int:
    """
    WHAT: Advances the tactical simulation clock forward by a fixed number of days for account seasoning.
    It multiplies the requested seasoning duration in calendar days by 1,440 minutes per day and adds
    the resulting elapsed duration to the current simulation minute without scheduling any payment events.

    WHY: Decision D7 models account seasoning (warm_up) as an elapsed time barrier where newly leased
    accounts remain dormant to evade heuristic new-account detection before active routing begins.

    Args:
        current_minute: Current simulation minute timestamp of the gang campaign.
        days: Integer number of calendar days allocated to account warming.

    Returns:
        New integer simulation minute timestamp after warming completes.

    Example:
        # new_minute = advance_warmup_clock(current_minute=100, days=2)
        # assert new_minute == 100 + 2 * 1440
    """
    # 1. Advance simulation clock by 1,440 minutes for each seasoning day.
    minutes_elapsed = days * MINUTES_PER_DAY
    return current_minute + minutes_elapsed


def draw_route_stream_amount(
    strength: int, amount_param_inr: int | None, rng: np.random.Generator
) -> int:
    """
    WHAT: Determines the transfer amount in integer paise for a laundering stream based on gang strength.
    Under strength 1, amounts are fixed multiples of Rs 1,000 (either from configuration or drawn from the
    training range). Under strength 2, amounts vary down to the single rupee to camouflage against round-number filters.

    WHY: Decision D7 specifies that naive syndicates (strength 1) move obvious round sums, whereas
    camouflaged syndicates (strength 2) inject fine-grained rupee variances across parallel chains.

    Args:
        strength: Integer strength level of the gang (1 for fixed round sums, 2 for camouflaged variance).
        amount_param_inr: Optional configured transfer amount in rupees.
        rng: Gang-specific random number generator stream.

    Returns:
        Integer transfer amount in paise.

    Example:
        # amt = draw_route_stream_amount(1, 5000, rng)
        # assert amt == 500_000 and amt % 100_000 == 0
    """
    # 1. Read calibrated amount bounds dynamically from world ranges configuration.
    min_amount, max_amount = get_route_amount_range()

    # 2. Evaluate strength 1 fixed round-thousand transfers.
    if strength == 1:
        if amount_param_inr is not None:
            inr = (amount_param_inr // 1000) * 1000
        else:
            inr = int(round(rng.uniform(min_amount, max_amount) / 1000.0) * 1000)
        return inr * 100

    # 3. Evaluate strength 2 variable single-rupee transfers.
    if amount_param_inr is not None:
        inr = int(amount_param_inr)
    else:
        inr = int(round(rng.uniform(min_amount, max_amount)))
    return inr * 100


def draw_hop_dwell_minutes(
    dwell_param_minutes: int | None, rng: np.random.Generator
) -> int:
    """
    WHAT: Determines the dwell holding duration in simulated minutes for a payment hop in the laundering chain.
    If a dwell parameter is explicitly configured, it validates and returns that value; otherwise, it samples
    an integer dwell duration within the calibrated training range [5, 60] minutes using the gang RNG stream.

    WHY: Decision D7 requires funds to reside in intermediate mule accounts for realistic holding windows
    prior to forwarding, reflecting operational delays and human communication latencies in laundering syndicates.

    Args:
        dwell_param_minutes: Optional explicit dwell window duration in simulated minutes.
        rng: Gang-specific random number generator stream.

    Returns:
        Integer dwell duration in simulated minutes.

    Example:
        # dwell = draw_hop_dwell_minutes(None, rng)
        # assert 5 <= dwell <= 60
    """
    # 1. Use explicitly configured dwell or sample from calibrated training bounds.
    if dwell_param_minutes is not None:
        return dwell_param_minutes
    min_dwell, max_dwell = get_dwell_range()
    return int(rng.integers(min_dwell, max_dwell + 1))


def calculate_send_minute(
    receive_minute: int, hop_dwell: int, has_disguise: bool, jitter_minutes: int | None,
    rng: np.random.Generator
) -> int:
    """
    WHAT: Computes the dispatch minute timestamp for a forwarding payment relative to funds receipt.
    Without disguise, forwarding occurs exactly at the expiration of the dwell window. With disguise,
    the dispatch minute is perturbed with gang RNG to fall strictly within the dwell window [receive, receive + dwell].

    WHY: Decision D7 specifies that disguise timing jitter scatters payments across time, preventing
    deterministic inter-arrival pattern detection while ensuring money is never forwarded before receipt.

    Args:
        receive_minute: Simulation minute timestamp when money arrived in the account.
        hop_dwell: Planned dwell holding duration in simulated minutes.
        has_disguise: Boolean indicating whether disguise timing camouflage is active.
        jitter_minutes: Optional maximum minute jitter window configured for disguise.
        rng: Gang-specific random number generator stream.

    Returns:
        Integer simulation minute timestamp when the outbound transfer should be dispatched.

    Example:
        # send_min = calculate_send_minute(100, 30, True, 10, rng)
        # assert 100 <= send_min <= 130
    """
    # 1. Dispatch at end of dwell window when disguise is inactive.
    if not has_disguise:
        return receive_minute + hop_dwell

    # 2. Perturb dispatch minute inside the dwell interval when disguise is active.
    min_jitter, max_jitter = get_disguise_jitter_range()
    jitter_bound = jitter_minutes if jitter_minutes is not None else max_jitter
    limit = min(jitter_bound, hop_dwell)
    offset = int(rng.integers(min_jitter, limit + 1)) if limit >= min_jitter else 0
    return receive_minute + (hop_dwell - offset)
