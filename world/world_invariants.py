"""
WHAT: Checks a finished world against facts that must hold in every run: the ledger checks of Level 11 with the right
expected total for a fraud-on world, no account moves money before it is opened, every account named in the truth
exists, ATM withdrawals stay inside their daily card limit, payment ids are unique, and complaints arrive after the
payments they name. Each check returns plain problem sentences; an empty list means every check passed.

WHY: A sampled-world sweep runs many worlds, so a broken rule must show up as a sentence naming the account, minute and
payment id, not as a wrong number later. The checks read only the run's own records (ledger entries, opened accounts,
withdrawal records, payment ids) and never repeat the runner's logic, so they can disagree with it.

TERM: invariant: a statement that must be true of every finished world, whatever its seed or gang mix.
TERM: expected total: the money that must be in the ledgers at the end: every opening balance minus cash withdrawn.
TERM: opening entry: the ledger entry of kind "opening" that gives an account its starting balance.
"""

from collections import Counter
from typing import Sequence

from world.checks import check_balances_equal_entry_sums, check_money_conserved, check_no_negative_balances
from world.interactive_records import InteractiveWorldOutput
from world.label_records import LabelEvent

MINUTES_PER_DAY = 1440


def expected_total_paise(output: InteractiveWorldOutput) -> int:
    """
    WHAT: Returns the money that must be in the ledgers at the end of a fraud-on run: the sum of every opening entry
    (the starting balances of the world and of accounts opened during the run) minus the cash withdrawn.

    WHY: Level 11 expected the opening total alone. A fraud-on run also registers accounts mid-run with balances and
    lets cash leave through withdrawals, and both are in the run's own records, so nothing from outside is needed.

    Args:
        output: A finished run.

    Returns:
        The expected total in paise.
    """
    opening = 0
    for ledger in output.world.ledgers:
        for entry in ledger.entries:
            if entry.kind == "opening":
                opening += entry.amount_paise
    return opening - output.result.money_withdrawn_paise


def check_ledgers(output: InteractiveWorldOutput) -> list[str]:
    """
    WHAT: Runs the three Level 11 ledger checks (no negative balance, balance equals entry sum, money conserved) with
    the fraud-on expected total, and tags each problem with the check's name.

    WHY: Level 11 proved the ledger sound for ordinary worlds; fraud, openings and withdrawals must not weaken it.

    Args:
        output: A finished run.

    Returns:
        Problem sentences, each starting with [no_negative_balances], [balances_equal_entry_sums] or [money_conserved].
    """
    ledgers = output.world.ledgers
    problems = [f"[no_negative_balances] {text}" for text in check_no_negative_balances(ledgers)]
    problems += [f"[balances_equal_entry_sums] {text}" for text in check_balances_equal_entry_sums(ledgers)]
    problems += [f"[money_conserved] {text}" for text in check_money_conserved(ledgers, expected_total_paise(output))]
    return problems


def check_opening_times(output: InteractiveWorldOutput) -> list[str]:
    """
    WHAT: Finds ledger entries that move money on an account opened mid-run before its opening minute.

    WHY: Level 12c lets gangs and genuine people open accounts during the run; a payment or withdrawal before the
    opening would mean the plan and the queue disagree about when the account exists.

    Args:
        output: A finished run.

    Returns:
        Problem sentences starting with [opening_time], naming the account, entry minute and opening minute.
    """
    opening_minute: dict[str, int] = {}
    for minute, account_id, _owner, _stream, _bank, _balance in output.result.opened_accounts:
        opening_minute[account_id] = minute
    problems: list[str] = []
    for ledger in output.world.ledgers:
        for entry in ledger.entries:
            opened_at = opening_minute.get(entry.account_id)
            if entry.kind != "opening" and opened_at is not None and entry.minute < opened_at:
                problems.append(
                    f"[opening_time] account {entry.account_id} has a {entry.kind} entry "
                    f"(transfer {entry.transfer_id}) at minute {entry.minute}, "
                    f"before its opening at minute {opened_at}"
                )
    return problems


def check_truth_accounts_exist(output: InteractiveWorldOutput) -> list[str]:
    """
    WHAT: Finds accounts named in the truth roles or device control spans that are not in any ledger at the end.

    WHY: The truth is what a later evaluation scores against; an account it names that never existed would make a label
    point at nothing (rule 15 keeps the truth world-side, so only the world can notice this).

    Args:
        output: A finished run.

    Returns:
        Problem sentences starting with [truth_accounts], naming the account and where the truth names it.
    """
    known: set[str] = set()
    for ledger in output.world.ledgers:
        known.update(ledger.accounts)
    problems: list[str] = []
    for role in output.result.truth.roles:
        if role.account_id not in known:
            problems.append(
                f"[truth_accounts] role '{role.role}' of gang {role.gang_id} names missing account {role.account_id}"
            )
    for span in output.result.truth.device_control:
        if span.account_id not in known:
            problems.append(
                f"[truth_accounts] device {span.device_id} of gang {span.gang_id} "
                f"names missing account {span.account_id}"
            )
    return problems


def check_atm_daily_limits(output: InteractiveWorldOutput, atm_limits: dict[str, int] | None) -> list[str]:
    """
    WHAT: Adds up each account's successful ATM withdrawals per day and flags any day above that account's daily card
    limit. The limits come from the planned withdrawals, which the run's records do not carry; with no limits given the
    check has nothing to compare and reports nothing.

    WHY: The runner enforces a daily ATM ceiling (Level 12c); reading only the success records means a bug in that
    enforcement shows up here.

    Args:
        output: A finished run.
        atm_limits: Account id -> card_daily_limit_paise from the gang plans, or None.

    Returns:
        Problem sentences starting with [atm_daily_limit], naming the account, day, total and limit.
    """
    if not atm_limits:
        return []
    totals: dict[tuple[str, int], int] = {}
    for record in output.result.cash_withdrawals:
        if record.status == "success" and record.channel == "atm":
            key = (record.account, record.minute // MINUTES_PER_DAY)
            totals[key] = totals.get(key, 0) + record.withdrawn
    problems: list[str] = []
    for (account_id, day), total in sorted(totals.items()):
        limit = atm_limits.get(account_id)
        if limit is not None and total > limit:
            problems.append(
                f"[atm_daily_limit] account {account_id} withdrew {total} paise by ATM on day {day}, "
                f"above its daily limit of {limit}"
            )
    return problems


def check_payment_ids_unique(output: InteractiveWorldOutput) -> list[str]:
    """
    WHAT: Finds payment ids that appear more than once in the run.

    WHY: Labels, truth and the event output all refer to payments by id; a repeated id would make two payments
    indistinguishable.

    Args:
        output: A finished run.

    Returns:
        Problem sentences starting with [payment_ids], naming each repeated id and how often it appears.
    """
    counts = Counter(output.result.payment_ids)
    return [
        f"[payment_ids] payment id {payment_id} appears {count} times"
        for payment_id, count in sorted(counts.items()) if count > 1
    ]


def check_labels_after_payments(output: InteractiveWorldOutput, labels: Sequence[LabelEvent]) -> list[str]:
    """
    WHAT: Checks that every payment a complaint names exists in the run and happened at or before the complaint arrives.

    WHY: A complaint cannot arrive before the payment it is about (Level 12e); one that does would give a detector the
    future.

    Args:
        output: A finished run.
        labels: The run's label events.

    Returns:
        Problem sentences starting with [labels], naming the label, payment id and both minutes.
    """
    minute_of: dict[str, int] = {}
    for payment_id, event in zip(output.result.payment_ids, output.result.run_result.events):
        minute_of[payment_id] = event.minute
    problems: list[str] = []
    for label in labels:
        for payment_id in label.payment_ids:
            if payment_id not in minute_of:
                problems.append(f"[labels] label {label.label_id} names payment {payment_id}, which is not in the run")
            elif label.arrival_minute < minute_of[payment_id]:
                problems.append(
                    f"[labels] label {label.label_id} arrives at minute {label.arrival_minute}, before payment "
                    f"{payment_id} at minute {minute_of[payment_id]}"
                )
    return problems


def check_world(
    output: InteractiveWorldOutput,
    labels: Sequence[LabelEvent] | None = None,
    atm_limits: dict[str, int] | None = None,
) -> list[str]:
    """
    WHAT: Runs every world invariant on a finished run and returns all problems found (an empty list means all passed).
    The label check runs only when labels are given, and the ATM check only when the planned limits are given.

    WHY: One call per world makes a sweep simple: a world is sound when the list is empty.

    Args:
        output: A finished run.
        labels: The run's label events, or None to skip the label check.
        atm_limits: Account id -> daily ATM card limit from the gang plans, or None to skip the ATM check.

    Returns:
        Problem sentences, each tagged with the name of the check that found it.
    """
    problems: list[str] = []
    problems += check_ledgers(output)
    problems += check_opening_times(output)
    problems += check_truth_accounts_exist(output)
    problems += check_atm_daily_limits(output, atm_limits)
    problems += check_payment_ids_unique(output)
    if labels is not None:
        problems += check_labels_after_payments(output, labels)
    return problems

