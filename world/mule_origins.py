"""
WHAT: Mule origins for the acquire step (Level 12c C-3a): draws how each acquired mule account came to be controlled
(rented from an ordinary owner, opened on borrowed identity documents, or opened as a business account), names the
accounts that do not exist yet with handles, and builds the open_account requests that create them.

WHY: Only rented mules exist in the world when a gang is planned. Borrowed-KYC and business mule accounts are new
accounts, opened mid-run in the standard id formats, so a plan can only refer to them by handle until the runner
assigns their real ids.

TERM: mule origin: rented, borrowed_kyc or business; stored in truth on the mule's role span.
TERM: handle: a placeholder account name "new:<gang_id>:<n>" for an account the gang will open; the runner replaces
      every handle by the real account id before the run starts.
TERM: origin mix: the acquire parameter that decides where origins come from: all_rented or world_ranges.
TERM: share: the chance that a mule has a given origin, drawn from a range and normalised so the shares sum to 1.
"""

import numpy as np

from world.outside_events import OutsideEvent
from world.tactic_config import get_mule_origin_share_ranges, get_new_account_owner_kind

HANDLE_PREFIX: str = "new:"
ORIGIN_RENTED: str = "rented"
OPENING_STREAM: str = "gang_account_openings"


def make_opening_handle(gang_id: str, opening_number: int) -> str:
    """
    WHAT: Builds the handle for a gang's opening_number-th new account, counting from 0, e.g. "new:g1:0".

    WHY: A handle is unique per gang and carries no account-id shape, so it can never be mistaken for a real id.

    Args:
        gang_id: The gang's id.
        opening_number: Position of the new account among this gang's openings, from 0.

    Returns:
        The handle string.

    Example:
        make_opening_handle("g1", 2) == "new:g1:2"
    """
    return f"{HANDLE_PREFIX}{gang_id}:{opening_number}"


def is_opening_handle(account_id: str) -> bool:
    """
    WHAT: Tells whether an account name is a handle rather than a real account id.

    WHY: Replacement and checking code must recognise a handle that was never resolved.

    Args:
        account_id: Any account name.

    Returns:
        True when the name starts with the handle prefix.

    Example:
        is_opening_handle("new:g1:0") is True; is_opening_handle("acc_person_000001") is False.
    """
    return account_id.startswith(HANDLE_PREFIX)


def draw_origin_shares(split: str, rng: np.random.Generator) -> dict[str, float]:
    """
    WHAT: Draws one share for each origin uniformly from its configured range and divides each by their sum.
    Origins are drawn in the order the config lists them (rented, borrowed_kyc, business), one number each.

    WHY: The three ranges are independent and do not sum to 1, so the draws must be normalised before they can be
    used as chances.

    Args:
        split: Split name of the world, e.g. "training".
        rng: The gang planning stream (child 7).

    Returns:
        Origin -> normalised share; the shares sum to 1.

    Raises:
        ValueError: If the drawn shares sum to 0, so nothing can be normalised.

    Example:
        With ranges [1, 1], [0, 0] and [0, 0] the result is {"rented": 1.0, "borrowed_kyc": 0.0, "business": 0.0}.
    """
    ranges = get_mule_origin_share_ranges(split)
    raw_shares = {origin: float(rng.uniform(low, high)) for origin, (low, high) in ranges.items()}
    total = sum(raw_shares.values())
    if total <= 0.0:
        raise ValueError("The drawn mule origin shares sum to 0, so they cannot be normalised.")
    return {origin: share / total for origin, share in raw_shares.items()}


def draw_mule_origins(count: int, split: str, rng: np.random.Generator) -> list[str]:
    """
    WHAT: Gives each of count mules an origin: first the shares are drawn, then one uniform number per mule picks
    the origin whose cumulative share first exceeds it.

    WHY: One draw per mule keeps the origin sequence reproducible and independent of how many mules share an origin.

    Args:
        count: Number of mules.
        split: Split name of the world.
        rng: The gang planning stream (child 7).

    Returns:
        One origin per mule, in mule order.

    Example:
        draw_mule_origins(3, "training", rng) might return ["rented", "business", "rented"].
    """
    shares = draw_origin_shares(split, rng)
    origins: list[str] = []
    for _ in range(count):
        draw = float(rng.uniform(0.0, 1.0))
        cumulative = 0.0
        chosen = list(shares)[-1]
        for origin, share in shares.items():
            cumulative += share
            if draw < cumulative:
                chosen = origin
                break
        origins.append(chosen)
    return origins


def build_opening_event(gang_id: str, opening_number: int, origin: str, minute: int) -> OutsideEvent:
    """
    WHAT: Builds the open_account request for a new mule account: a person account for borrowed_kyc, a shop account
    for business, on the gang account opening stream, tagged with the account's handle.

    WHY: The owner kind of each origin comes from config, so a new origin needs no code change here.

    Args:
        gang_id: The gang's id.
        opening_number: Position of the new account among this gang's openings, from 0.
        origin: The mule's origin.
        minute: Minute at which the account is opened.

    Returns:
        The OutsideEvent.

    Raises:
        ValueError: If the origin needs no new account (rented).

    Example:
        build_opening_event("g1", 0, "business", 1440) opens a shop account at minute 1440 under handle "new:g1:0".
    """
    owner_kind = get_new_account_owner_kind(origin)
    if owner_kind is None:
        raise ValueError(f"Mule origin '{origin}' uses an existing account and needs no opening.")
    return OutsideEvent(
        minute=minute, kind="open_account", owner_kind=owner_kind, opening_stream=OPENING_STREAM,
        opening_handle=make_opening_handle(gang_id, opening_number),
    )
