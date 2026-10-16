"""
WHAT: Unit tests for world/outside_events.py and world/encumbrance.py, without running a world. They check that each
malformed outside event is refused with its own plain message, and that the encumbrance book places, releases and
answers "is this debit blocked?" exactly as described.

WHY: The interactive runner trusts these two small modules on every outgoing payment piece. Testing them alone,
with small hand-made numbers, makes a failure point straight at the rule that broke instead of at a whole world.

TERM: outside event: something a bank or authority does to an account during a run, e.g. a Rs 500 lien.
TERM: lien: keeps part of an account's money from being spent, e.g. Rs 500 of a Rs 2,000 balance.
TERM: debit hold: blocks every outgoing payment from an account; incoming money still arrives.
TERM: encumbrance book: the record of liens and holds, which are still active, and what each account may spend.
TERM: available balance: balance minus active liens, e.g. 200,000 - 50,000 = 150,000 paise.
TERM: paise: Rs 1 = 100 paise.
"""

import pytest

from world.encumbrance import EncumbranceBook
from world.outside_events import OutsideEvent, validate_outside_event


@pytest.mark.parametrize(
    ("event", "message"),
    [
        (OutsideEvent(10, "freeze", "acc_a"), "is not one of lien, debit_hold, release, resume_payment"),
        (OutsideEvent(-1, "lien", "acc_a", 500), "outside event minute must be 0 or more, got -1"),
        (OutsideEvent(10, "lien", None, 500), "a lien needs an account_id and an amount_paise above 0"),
        (OutsideEvent(10, "lien", "acc_a", 0), "a lien needs an account_id and an amount_paise above 0"),
        (OutsideEvent(10, "debit_hold", None), "a debit_hold needs an account_id"),
        (OutsideEvent(10, "release"), "a release needs the encumbrance_id it releases"),
        (OutsideEvent(10, "resume_payment"), "a resume_payment needs the planned_payment_id of the held payment"),
    ],
)
def test_each_bad_outside_event_is_refused(event: OutsideEvent, message: str) -> None:
    """
    WHAT: Feeds validate_outside_event one malformed event per rule (unknown kind, negative minute, lien without
    account, lien with 0 amount, hold without account, release without id, resume without payment id) and checks
    that each raises ValueError containing its own sentence.

    WHY: Each rule protects the runner from a different mistake. One case per rule shows every rule is live and that
    its message tells the reader exactly what to fix.

    Args:
        event: The malformed OutsideEvent for this case.
        message: Text the error must contain.

    Returns:
        None. The test fails if no error is raised or the message differs.

    Example:
        OutsideEvent(10, "lien", "acc_a", 0) raises "a lien needs an account_id and an amount_paise above 0".
    """
    # Each bad event must be stopped with its own explanation.
    with pytest.raises(ValueError, match=message):
        validate_outside_event(event)


def test_good_outside_events_pass() -> None:
    """
    WHAT: Checks that one well-formed event of every kind passes validation without an error.

    WHY: Validation that rejected good input would be as harmful as validation that accepted bad input; this pins
    the positive side so the rules cannot drift into being too strict.

    Args:
        None.

    Returns:
        None. The test fails if any good event raises.

    Example:
        OutsideEvent(1440, "lien", "acc_a", 50000) passes, as do a hold, a release and a resume.
    """
    # One correct example of each of the four kinds.
    good_events = [
        OutsideEvent(1440, "lien", "acc_a", 50000),
        OutsideEvent(1440, "debit_hold", "acc_a"),
        OutsideEvent(2880, "release", encumbrance_id="enc-000001"),
        OutsideEvent(2880, "resume_payment", planned_payment_id="run-p-000003"),
    ]
    for event in good_events:
        validate_outside_event(event)


def test_lien_reduces_what_can_be_spent() -> None:
    """
    WHAT: Places a 50,000-paise (Rs 500) lien on an account with a 200,000-paise balance, then checks the available
    balance is 150,000, a 150,000 debit is allowed, a 150,001 debit is blocked, and after release 200,000 is
    available again.

    WHY: This is the arithmetic the runner relies on for every lien. The edge at exactly the available amount
    matters: a debit equal to what is available must still go through.

    Args:
        None.

    Returns:
        None. The test fails if any number differs.

    Example:
        200,000 - 50,000 = 150,000, so blocks_debit(..., 150000) is False and blocks_debit(..., 150001) is True.
    """
    # Place a Rs 500 lien on account acc_a at minute 1440.
    book = EncumbranceBook()
    book.place(OutsideEvent(1440, "lien", "acc_a", 50000), "enc-000001")
    assert book.available_balance("acc_a", 200000) == 200000 - 50000
    # A debit equal to the available money passes; one paise more is blocked.
    assert book.blocks_debit("acc_a", 200000, 150000) is False
    assert book.blocks_debit("acc_a", 200000, 150001) is True
    # Releasing the lien makes the full balance available again.
    book.release("enc-000001", 2880)
    assert book.available_balance("acc_a", 200000) == 200000
    assert book.log == [(1440, "place_lien", "enc-000001"), (2880, "release", "enc-000001")]


def test_debit_hold_blocks_until_released_and_only_on_its_account() -> None:
    """
    WHAT: Places a debit hold on acc_a and checks that even a 1-paise debit from acc_a is blocked, that acc_b is not
    affected, and that after the release acc_a's debits are allowed again.

    WHY: A debit hold is all or nothing for one account. It must not leak onto other accounts, and it must end
    exactly when released, otherwise money stays frozen longer than the outside event says.

    Args:
        None.

    Returns:
        None. The test fails if the hold blocks too little, too much, or too long.

    Example:
        With a hold on acc_a: blocks_debit("acc_a", 200000, 1) is True and blocks_debit("acc_b", 200000, 1) is False.
    """
    # A hold on acc_a blocks the smallest possible debit there.
    book = EncumbranceBook()
    book.place(OutsideEvent(1440, "debit_hold", "acc_a"), "enc-000002")
    assert book.blocks_debit("acc_a", 200000, 1) is True
    # Another account is untouched.
    assert book.blocks_debit("acc_b", 200000, 1) is False
    # After release, acc_a can pay again.
    book.release("enc-000002", 2880)
    assert book.blocks_debit("acc_a", 200000, 1) is False


def test_release_of_unknown_or_released_encumbrance_is_refused() -> None:
    """
    WHAT: Checks that releasing an id that was never placed raises "was never placed", and that releasing the same
    id twice raises "has already been released". Placing the same id twice is also refused.

    WHY: These are signs that the outside events contradict each other. Failing loudly prevents a block from being
    silently left on, or silently assumed off.

    Args:
        None.

    Returns:
        None. The test fails if any of the three mistakes is accepted.

    Example:
        EncumbranceBook().release("enc-999999", 10) raises "encumbrance enc-999999 was never placed, ...".
    """
    # Releasing an id nobody placed.
    book = EncumbranceBook()
    with pytest.raises(ValueError, match="was never placed"):
        book.release("enc-999999", 10)
    # Placing one id twice.
    book.place(OutsideEvent(10, "lien", "acc_a", 100), "enc-000001")
    with pytest.raises(ValueError, match="has already been placed"):
        book.place(OutsideEvent(20, "lien", "acc_a", 100), "enc-000001")
    # Releasing the same id twice.
    book.release("enc-000001", 30)
    with pytest.raises(ValueError, match="has already been released"):
        book.release("enc-000001", 40)
