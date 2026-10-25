"""
WHAT: The things that can happen to an account from outside the payment flow while a world runs: a bank places a
lien (keeps part of the money from being spent), places a debit hold (blocks all outgoing payments), releases one
of those, or lets a held payment go ahead. Each is an OutsideEvent with a minute, and validate_outside_event
refuses a malformed one with one plain sentence.

WHY: In real life banks and police act on accounts at moments of their own choosing, not as part of a payment.
Keeping these as their own small records means the runner can put them on the same timeline as payments and
apply them in a fixed order. Checking each one when it arrives turns a typo into a clear error instead of a
silently wrong world.

TERM: outside event: something done to an account by a bank or authority, not by the account holder,
      e.g. a lien of Rs 500 placed at minute 1,440.
TERM: lien: a block on part of an account's money; the money stays in the account but cannot be spent,
      e.g. a balance of Rs 2,000 with a Rs 500 lien leaves Rs 1,500 spendable.
TERM: debit hold: a block on every outgoing payment from an account, whatever its balance; incoming money still
      arrives.
TERM: release: the end of one lien or debit hold, named by its encumbrance id.
TERM: encumbrance id: the name of one lien or debit hold, e.g. "enc-000001", used later to release it.
TERM: resume payment: an instruction to let a payment that the bank held go ahead now.
TERM: planned payment id: the name of one payment the world planned, shared by all of its pieces when it is
      split, e.g. "66db9aa439b092ee-p-000017".
"""

from dataclasses import dataclass

# Every kind the runner understands today. Level 21 adds the banks' own decisions here as new kinds.
OUTSIDE_EVENT_KINDS: tuple[str, ...] = ("lien", "debit_hold", "release", "resume_payment", "open_account")
VALID_OPENING_STREAMS: tuple[str, ...] = ("genuine_account_openings", "gang_account_openings")


@dataclass(frozen=True)
class OutsideEvent:
    """
    WHAT: One outside event: when it happens (minute), what it is (kind), and the details that kind needs.
    A lien uses account_id and amount_paise; a debit hold uses account_id; a release uses encumbrance_id; a
    resume uses planned_payment_id; an open_account uses owner_kind and opening_stream. Unused fields keep
    their defaults.

    WHY: One record shape for every kind lets all of them share one queue and one validation function. Freezing
    it means nobody can change an event after it was checked.

    Args:
        minute: When the event happens, in minutes since the world started, e.g. 1440 for the start of day 2.
        kind: One of OUTSIDE_EVENT_KINDS.
        account_id: The account a lien or debit hold applies to, e.g. "acc_person_000003".
        amount_paise: The amount a lien keeps back, in paise (100 paise = Rs 1), e.g. 50000 for Rs 500.
        encumbrance_id: The lien or hold to name (when placing) or to end (when releasing).
        planned_payment_id: The held payment a resume_payment lets go ahead.
        owner_kind: For open_account, whether the owner is a "person" or a "shop".
        opening_stream: For open_account, the random stream name from random_stream_children.
        opening_handle: For an open_account planned by a gang, the placeholder name ("new:<gang_id>:<n>") of the
            account to open; the runner replaces it by the real id before the run. None for genuine openings.

    Returns:
        A frozen OutsideEvent.

    Example:
        OutsideEvent(1440, "lien", account_id="acc_person_000003", amount_paise=50000) is a Rs 500 lien placed at
        the start of day 2.
    """

    minute: int
    kind: str
    account_id: str | None = None
    amount_paise: int = 0
    encumbrance_id: str | None = None
    planned_payment_id: str | None = None
    owner_kind: str | None = None
    opening_stream: str | None = None
    opening_handle: str | None = None


def validate_outside_event(event: OutsideEvent) -> None:
    """
    WHAT: Checks one outside event and raises ValueError with one plain sentence if it cannot be applied: an unknown
    kind, a negative minute, a lien without an account or with an amount of 0 or less, a debit hold without an
    account, a release without an encumbrance id, a resume without a planned payment id, or an open_account with
    an account_id, missing or invalid owner_kind, or unknown opening_stream.

    WHY: The runner applies outside events in the middle of a world. A bad one found there would stop the run far
    from its cause, or worse, be applied wrongly. Checking each event as it is queued names the problem at once.

    Args:
        event: The OutsideEvent to check.

    Returns:
        None. It only raises when something is wrong.

    Example:
        validate_outside_event(OutsideEvent(1440, "lien", "acc_person_000003", 50000)) passes quietly.
        validate_outside_event(OutsideEvent(1440, "lien", "acc_person_000003", 0)) raises
        "a lien needs an account_id and an amount_paise above 0".
    """
    # The kind must be one the runner knows how to apply.
    if event.kind not in OUTSIDE_EVENT_KINDS:
        raise ValueError(f"outside event kind '{event.kind}' is not one of {', '.join(OUTSIDE_EVENT_KINDS)}")
    # Time starts at minute 0, so a negative minute is a mistake.
    if event.minute < 0:
        raise ValueError(f"outside event minute must be 0 or more, got {event.minute}")
    # A lien must say which account and how much money it keeps back.
    if event.kind == "lien" and (not event.account_id or event.amount_paise <= 0):
        raise ValueError("a lien needs an account_id and an amount_paise above 0")
    # A debit hold blocks a whole account, so it needs only the account.
    if event.kind == "debit_hold" and not event.account_id:
        raise ValueError("a debit_hold needs an account_id")
    # A release must name the lien or hold it ends.
    if event.kind == "release" and not event.encumbrance_id:
        raise ValueError("a release needs the encumbrance_id it releases")
    # A resume must name the held payment it lets go ahead.
    if event.kind == "resume_payment" and not event.planned_payment_id:
        raise ValueError("a resume_payment needs the planned_payment_id of the held payment")
    # An open_account creates a new person or shop account mid-run using a designated stream.
    if event.kind == "open_account":
        if event.account_id is not None:
            raise ValueError("an open_account event must not specify an account_id")
        if event.owner_kind not in ("person", "shop"):
            raise ValueError("an open_account event needs owner_kind to be 'person' or 'shop'")
        if not event.opening_stream or event.opening_stream not in VALID_OPENING_STREAMS:
            raise ValueError(
                f"an open_account event needs opening_stream to be one of {', '.join(VALID_OPENING_STREAMS)}"
            )
        # A gang opening may carry the handle its plan uses; a genuine opening never does.
        if event.opening_stream == "genuine_account_openings" and event.opening_handle is not None:
            raise ValueError("a genuine open_account event must not carry an opening_handle")
    elif event.opening_handle is not None:
        raise ValueError("only an open_account event may carry an opening_handle")
