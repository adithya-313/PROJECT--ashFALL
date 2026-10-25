"""
WHAT: Assigns real account ids to the handles a gang plan uses for accounts it will open mid-run, and rewrites every
place a plan stores an account id (payments, truth roles, device control spans, gang device events, illicit seeds,
outside events, gang holdings, cash withdrawals and account resales) with the real ids.

WHY: A gang is planned before the run, but the id of an account opened mid-run depends on how many accounts were
opened before it, genuine ones included. Replaying the openings in the exact order the queue will apply them gives
every handle its final id before the first payment runs, so no plan ever has to guess and no handle reaches output.

TERM: handle: a placeholder account name "new:<gang_id>:<n>" for an account a gang will open.
TERM: opening replay: walking the open_account events in queue order and counting persons and shops the way
      the runner state will, to find the id each opening will receive.
TERM: queue order: outside events are applied by minute, then by their position in the outside event list.
"""

import dataclasses

from world.calendar_runner import PlannedPayment
from world.cash_withdrawals import CashWithdrawal
from world.interactive_records import FraudInputs
from world.interactive_state import person_account_id, shop_account_id
from world.mule_origins import is_opening_handle
from world.outside_events import OutsideEvent
from world.truth_records import TruthFacts


def replay_openings(
    outside_events: tuple[OutsideEvent, ...], next_person_index: int, next_shop_index: int
) -> tuple[dict[str, str], dict[str, int]]:
    """
    WHAT: Walks the open_account events in the order the queue applies them (minute, then list position) and counts
    persons and shops from the given starting indexes. Every event with a handle gets the id its opening will receive.

    WHY: Outside events share rank 0 in the queue, so position in the list decides ties at the same minute. Counting
    with the same starting indexes the runner state uses makes the ids match the ones open_account will produce.

    Args:
        outside_events: All outside events of the run, genuine openings included.
        next_person_index: The runner state's next person index at run start.
        next_shop_index: The runner state's next shop index at run start.

    Returns:
        (handle -> account id, handle -> opening minute).

    Raises:
        ValueError: If two events carry the same handle.

    Example:
        Two openings at minute 100, a genuine person first and a gang person second, with next_person_index 60, give
        the gang handle "acc_person_000061".
    """
    openings = [(index, event) for index, event in enumerate(outside_events) if event.kind == "open_account"]
    openings.sort(key=lambda item: (item[1].minute, item[0]))
    person_index = next_person_index
    shop_index = next_shop_index
    assigned_ids: dict[str, str] = {}
    opening_minutes: dict[str, int] = {}
    for _position, event in openings:
        if event.owner_kind == "person":
            account_id = person_account_id(person_index)
            person_index += 1
        else:
            account_id = shop_account_id(shop_index)
            shop_index += 1
        if event.opening_handle is None:
            continue
        if event.opening_handle in assigned_ids:
            raise ValueError(f"Two open_account events carry the handle '{event.opening_handle}'.")
        assigned_ids[event.opening_handle] = account_id
        opening_minutes[event.opening_handle] = event.minute
    return assigned_ids, opening_minutes


def _real_id(account_id: str | None, assigned_ids: dict[str, str]) -> str | None:
    """
    WHAT: Returns the real id for a handle, or the name unchanged when it is a real account id or None.

    WHY: Every field that may hold an account id goes through this one rule.

    Args:
        account_id: A real account id, a handle, or None.
        assigned_ids: Handle -> real id.

    Returns:
        The real id, the unchanged name, or None.

    Raises:
        ValueError: If the name is a handle that no open_account event defines.

    Example:
        _real_id("new:g1:0", {"new:g1:0": "acc_person_000060"}) == "acc_person_000060"
    """
    if account_id is None:
        return None
    if account_id in assigned_ids:
        return assigned_ids[account_id]
    if is_opening_handle(account_id):
        raise ValueError(f"Account '{account_id}' is a handle but no open_account event opens it.")
    return account_id


def _check_payment_after_opening(
    payment: PlannedPayment, opening_minutes: dict[str, int]
) -> None:
    """
    WHAT: Asserts that a payment touching a handle is not dated before the opening of that handle's account.

    WHY: A payment to or from an account that does not exist yet could only fail with an unclear lookup error.

    Args:
        payment: The planned payment, with handles still in place.
        opening_minutes: Handle -> opening minute.

    Returns:
        None. Raises ValueError naming the handle and both minutes when the payment is too early.

    Example:
        A payment to "new:g1:0" at minute 100 with that account opened at minute 200 raises ValueError.
    """
    for account_id in (payment.from_account, payment.to_account):
        if account_id in opening_minutes and payment.minute < opening_minutes[account_id]:
            raise ValueError(
                f"A payment at minute {payment.minute} uses account '{account_id}' "
                f"before it is opened at minute {opening_minutes[account_id]}."
            )


def _check_withdrawal_after_opening(withdrawal: CashWithdrawal, opening_minutes: dict[str, int]) -> None:
    """
    WHAT: Asserts that a cash withdrawal from a handle's account is not dated before that account opens.

    WHY: Money cannot leave an account that does not exist yet, and the lookup error would not say why.

    Args:
        withdrawal: The planned withdrawal, with its handle still in place.
        opening_minutes: Handle -> opening minute.

    Returns:
        None. Raises ValueError naming the handle and both minutes when the withdrawal is too early.

    Example:
        A withdrawal from "new:g1:0" at minute 100 with that account opened at minute 200 raises ValueError.
    """
    account_id = withdrawal.account_id
    if account_id in opening_minutes and withdrawal.minute < opening_minutes[account_id]:
        raise ValueError(
            f"A withdrawal at minute {withdrawal.minute} uses account '{account_id}' "
            f"before it is opened at minute {opening_minutes[account_id]}."
        )


def resolve_opening_handles(
    fraud_inputs: FraudInputs, next_person_index: int, next_shop_index: int
) -> tuple[FraudInputs, dict[str, str]]:
    """
    WHAT: Gives every handle its real id and returns the inputs with all handles replaced. First checks that no
    payment or cash withdrawal uses a handle before its opening minute. Inputs without handles are returned unchanged.

    WHY: Everything downstream, from the queue to the truth files, works with real ids only.

    Args:
        fraud_inputs: The run's fraud inputs, genuine openings included in outside_events.
        next_person_index: The runner state's next person index at run start.
        next_shop_index: The runner state's next shop index at run start.

    Returns:
        (inputs with real ids, handle -> real id). The outside events keep their handles so that each opening can be
        checked against its assigned id when it is applied.

    Raises:
        ValueError: If a payment uses a handle early, a handle is repeated, or a handle has no opening.

    Example:
        resolve_opening_handles(FraudInputs(), 60, 12) returns (FraudInputs(), {}).
    """
    has_handles = any(event.opening_handle is not None for event in fraud_inputs.outside_events)
    if not has_handles:
        return fraud_inputs, {}

    assigned_ids, opening_minutes = replay_openings(fraud_inputs.outside_events, next_person_index, next_shop_index)
    for payment in fraud_inputs.payments:
        _check_payment_after_opening(payment, opening_minutes)
    for withdrawal in fraud_inputs.withdrawals:
        _check_withdrawal_after_opening(withdrawal, opening_minutes)

    payments = tuple(
        dataclasses.replace(
            payment,
            from_account=str(_real_id(payment.from_account, assigned_ids)),
            to_account=str(_real_id(payment.to_account, assigned_ids)),
        )
        for payment in fraud_inputs.payments
    )
    roles = tuple(
        dataclasses.replace(role, account_id=str(_real_id(role.account_id, assigned_ids)))
        for role in fraud_inputs.truth.roles
    )
    device_control = tuple(
        dataclasses.replace(span, account_id=str(_real_id(span.account_id, assigned_ids)))
        for span in fraud_inputs.truth.device_control
    )
    # gangs=fraud_inputs.truth.gangs passes gang truth records through so tags do not vanish
    # for gangs that open accounts
    truth = TruthFacts(
        roles=roles,
        device_control=device_control,
        payment_stories=fraud_inputs.truth.payment_stories,
        gangs=fraud_inputs.truth.gangs,
    )
    device_events = tuple(
        dataclasses.replace(event, account_id=_real_id(event.account_id, assigned_ids))
        for event in fraud_inputs.gang_device_events
    )
    seeds = tuple(
        dataclasses.replace(seed, account_id=str(_real_id(seed.account_id, assigned_ids)))
        for seed in fraud_inputs.illicit_seeds
    )
    outside_events = tuple(
        dataclasses.replace(event, account_id=_real_id(event.account_id, assigned_ids))
        for event in fraud_inputs.outside_events
    )
    holdings = tuple(
        dataclasses.replace(h, account_id=str(_real_id(h.account_id, assigned_ids)))
        for h in fraud_inputs.gang_holdings
    )
    # Withdrawals and resales were missed until Level 12f; fixtures used all_rented, so no withdrawal ever came from
    # a gang-opened account and the unresolved handle was never reached.
    withdrawals = tuple(
        dataclasses.replace(w, account_id=str(_real_id(w.account_id, assigned_ids)))
        for w in fraud_inputs.withdrawals
    )
    resales = tuple(
        dataclasses.replace(
            r, new_account=str(_real_id(r.new_account, assigned_ids)),
            sold_account=str(_real_id(r.sold_account, assigned_ids)),
        )
        for r in fraud_inputs.account_resales
    )
    resolved = dataclasses.replace(
        fraud_inputs, payments=payments, truth=truth, gang_device_events=device_events,
        illicit_seeds=seeds, outside_events=outside_events, gang_holdings=holdings,
        withdrawals=withdrawals, account_resales=resales,
    )
    return resolved, assigned_ids

