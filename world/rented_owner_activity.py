"""
WHAT: Decides how much of its owner's normal life a rented mule account keeps (Level 12c C-3a). For each mule role
with origin "rented" it draws a kept share once, and later answers, for one genuine payment, whether the owner keeps
making it or it is paused.

WHY: A rented account keeps part of its owner's normal payments next to the illicit flows. Level 11 plans the owner's
whole routine life for every person, so the runner pauses the share that is not kept instead of planning anything new,
and a paused payment reads exactly the failure draws a held payment reads, which keeps every other genuine payment's
draws where they were.

TERM: rented mule: a mule account whose ordinary owner still exists and still has a planned routine life.
TERM: kept share: the chance, drawn per rented mule from config, that one outgoing owner payment is still made.
TERM: paused payment: a genuine payment from a rented mule's account, inside its mule span, that is not made.
TERM: role span: the half-open interval [valid_from, valid_to) in which the account is a mule.
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from world.tactic_config import get_rented_owner_kept_share_range
from world.truth_records import RoleSpan

RENTED_ORIGIN: str = "rented"


@dataclass(frozen=True)
class OwnerActivity:
    """
    WHAT: What one rented mule keeps of its owner's life: the kept share and the mule span it applies in.

    WHY: Pausing needs both the chance and the interval in one lookup keyed by account.

    Args:
        kept_share: Chance that an outgoing owner payment inside the span is made, in [0, 1].
        valid_from: First minute of the mule span.
        valid_to: First minute after the span, or None when it never ends.

    Returns:
        An initialized immutable OwnerActivity.

    Example:
        OwnerActivity(0.5, 1440, None) keeps half of the owner's outgoing payments from minute 1440 on.
    """

    kept_share: float
    valid_from: int
    valid_to: int | None

    def covers(self, minute: int) -> bool:
        """
        WHAT: Tells whether a minute lies inside the mule span.

        WHY: Payments before the account was rented, or after, are the owner's own and are never paused.

        Args:
            minute: A payment's minute.

        Returns:
            True when valid_from <= minute and the span has not ended.

        Example:
            OwnerActivity(0.5, 1440, 2880).covers(2880) is False, because the span is half-open.
        """
        return self.valid_from <= minute and (self.valid_to is None or minute < self.valid_to)


@dataclass(frozen=True)
class PausedOwnerPayment:
    """
    WHAT: A genuine payment of a rented mule's owner that was paused: when it was planned, who would pay whom, and
    how much.

    WHY: Pausing records no event and moves no money, so this list is the only trace of what did not happen.

    Args:
        minute: The planned minute of the payment.
        from_account: The rented mule account that would have paid.
        to_account: The planned receiver.
        amount_paise: The planned amount in paise.

    Returns:
        An initialized immutable PausedOwnerPayment.

    Example:
        PausedOwnerPayment(2000, "acc_person_000006", "acc_shop_0003", 45000)
    """

    minute: int
    from_account: str
    to_account: str
    amount_paise: int


def plan_owner_activity(
    roles: Sequence[RoleSpan], split: str, rng: np.random.Generator
) -> dict[str, OwnerActivity]:
    """
    WHAT: Draws a kept share for every rented mule role, in sorted account-id order, uniformly from the configured
    range for the split, and records the role's span with it.

    WHY: Sorting makes the draws independent of the order roles were listed in. Each rented mule gets one draw, so
    adding a gang elsewhere never changes another mule's share.

    Args:
        roles: The truth role spans of the run.
        split: Split name of the world, e.g. "training".
        rng: The rented owner activity stream (child 12).

    Returns:
        Account id -> OwnerActivity for rented mules only; empty when there are none, and then nothing is read.

    Raises:
        ValueError: If a rented account has two mule spans, which the pausing rule cannot tell apart.

    Example:
        With one rented mule and a range [0.3, 0.8] the result has one entry with kept_share in that range.
    """
    rented_roles = sorted(
        (role for role in roles if role.role == "mule" and role.origin == RENTED_ORIGIN),
        key=lambda role: role.account_id,
    )
    if not rented_roles:
        return {}
    low_share, high_share = get_rented_owner_kept_share_range(split)
    activity: dict[str, OwnerActivity] = {}
    for role in rented_roles:
        if role.account_id in activity:
            raise ValueError(f"Rented mule '{role.account_id}' has more than one mule span.")
        kept_share = float(rng.uniform(low_share, high_share))
        activity[role.account_id] = OwnerActivity(kept_share, role.valid_from, role.valid_to)
    return activity
