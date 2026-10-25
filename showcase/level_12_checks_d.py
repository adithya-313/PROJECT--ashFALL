"""
WHAT: Implementation of Level 12 Showcase Part D checks (Checks 16 through 20).
Shows two gangs sharing accounts bought on the account market, a training world with no held-out tactic family,
fraud activity counted by tactic family, a lien that a gang learns of only after the feed delay, and the refusal of a
held-out family in a training world.

WHY: Verifies Level 12d: gangs that trade accounts, react to what they can observe, and are held to the tactic
families that were pre-registered before any gang world existed (amendments A-014 and A-017).

TERM: account market: resale of a burned rented mule account from one gang to a later gang.
TERM: feed delay: minutes between a lien being placed and the gang learning of it.
TERM: tactic family: one named way a mule gang moves stolen money, defined in config/tactic_families.yaml.
TERM: held-out family: a family allowed only in test and stress worlds, so it cannot shape training.
"""

from collections import Counter
from typing import Any

from showcase.framework import CheckOutcome
from showcase.level_12_support_d import (
    get_families_run,
    get_lien_run,
    get_market_run,
    run_held_out_in_training,
)
from world.interactive_records import InteractiveWorldOutput
from world.tactic_families import load_tactic_families


def _fraud_events_with_gang(output: InteractiveWorldOutput) -> list[tuple[Any, str]]:
    """
    WHAT: Lists every gang (fraud) event together with the id of the gang whose payment it belongs to.
    WHY: The gang of an event is world side knowledge, found through the fraud input index and the payment stories.
    Args:
        output: The InteractiveWorldOutput of a run with gangs.
    Returns:
        Pairs of (PaymentEvent, gang id) in execution order.
    Example:
        pairs = _fraud_events_with_gang(output); assert all(isinstance(gang, str) for _, gang in pairs)
    """
    result = output.result
    pairs: list[tuple[Any, str]] = []
    for position, event in enumerate(result.run_result.events):
        fraud_index = result.event_fraud_input_index[position]
        if result.event_sources[position] == "fraud" and fraud_index is not None:
            pairs.append((event, result.truth.payment_stories[fraud_index].gang_id))
    return pairs


async def run_account_market_check(workspace: Any) -> CheckOutcome:
    """
    WHAT: Evaluates the account market: a burned rented mule of the first gang is bought by the second gang.
    It reports the sold account, the seller, the buyer and the resale minute, and asserts that the truth owner of the
    sold account is the seller one minute before the resale and the buyer from the resale minute on.
    WHY: Proves Check 16: gangs can share an account without any overlap of ownership in the truth.
    Args:
        workspace: Optional execution workspace supplied by the showcase framework.
    Returns:
        CheckOutcome reporting the number of resold accounts and the first resale in detail.
    Example:
        outcome = await run_account_market_check(None)
        assert "resold" in outcome.key_number
    """
    output, store = get_market_run()
    resales = output.result.account_resales
    assert resales, "Expected at least one account resold between the two gangs"
    resale = resales[0]

    # The owner must change exactly at the resale minute: seller before it, buyer from it on.
    before = store.owner_at(resale.sold_account, resale.resale_minute - 1)
    at_minute = store.owner_at(resale.sold_account, resale.resale_minute)
    assert before == resale.previous_gang, f"Owner before resale is {before!r}, expected {resale.previous_gang!r}"
    assert at_minute == resale.new_gang, f"Owner at resale is {at_minute!r}, expected {resale.new_gang!r}"

    key_number = (
        f"{len(resales)} accounts resold; {resale.sold_account}: {resale.previous_gang} -> {resale.new_gang} "
        f"at minute {resale.resale_minute}, owner changes exactly then"
    )
    evidence = {
        "resold_accounts": len(resales), "sold_account": resale.sold_account, "seller": resale.previous_gang,
        "buyer": resale.new_gang, "resale_minute": resale.resale_minute,
        "owner_before": before, "owner_at_resale": at_minute,
    }
    return CheckOutcome(key_number=key_number, evidence=evidence)


async def run_training_world_families_check(workspace: Any) -> CheckOutcome:
    """
    WHAT: Evaluates a training world built from gangs with different training families. It reads the held-out list
    from config and asserts that no gang's family is held out, and that every gang family is a training family.
    WHY: Proves Check 17: a training world contains no held-out family, so a model trained on it cannot see one.
    Args:
        workspace: Optional execution workspace supplied by the showcase framework.
    Returns:
        CheckOutcome listing the gang families and the held-out list that none of them is on.
    Example:
        outcome = await run_training_world_families_check(None)
        assert "no held-out" in outcome.key_number
    """
    gangs, output = get_families_run()
    families = load_tactic_families()
    held_out = sorted(families["held_out"])
    used = sorted(str(gang.family) for gang in gangs)

    assert len(set(used)) >= 2, f"Expected gangs with different families, got {used}"
    assert output.result.run_result.events, "Expected the training world to produce events"
    for family in used:
        assert family not in held_out, f"Held-out family {family!r} found in a training world"
        assert family in families["training"], f"Family {family!r} is not a training family"

    key_number = f"{len(gangs)} gangs with families {', '.join(used)}; none is held out ({', '.join(held_out)})"
    evidence = {"gang_families": used, "held_out_families": held_out, "range_name": "training"}
    return CheckOutcome(key_number=key_number, evidence=evidence)


async def run_counts_by_family_check(workspace: Any) -> CheckOutcome:
    """
    WHAT: Counts, for each tactic family in the training world, the fraud payments and the gang accounts of the
    gangs that use it. The family of a gang comes from the gang configs (world side), not from the truth files.
    WHY: Proves Check 18: a gang world can be summarised by family, and every family really produced activity.
    Args:
        workspace: Optional execution workspace supplied by the showcase framework.
    Returns:
        CheckOutcome reporting payments and accounts per family.
    Example:
        outcome = await run_counts_by_family_check(None)
        assert "payments" in outcome.key_number
    """
    gangs, output = get_families_run()
    family_of_gang = {gang.gang_id: str(gang.family) for gang in gangs}
    payments: Counter[str] = Counter()
    for _event, gang_id in _fraud_events_with_gang(output):
        payments[family_of_gang[gang_id]] += 1
    accounts: dict[str, set[str]] = {family: set() for family in family_of_gang.values()}
    for role in output.result.truth.roles:
        accounts[family_of_gang[role.gang_id]].add(role.account_id)

    for family in sorted(accounts):
        assert payments[family] > 0, f"Family {family!r} has no fraud payments"
        assert accounts[family], f"Family {family!r} has no gang accounts"

    parts = [f"{family}: {payments[family]} payments, {len(accounts[family])} accounts" for family in sorted(accounts)]
    evidence = {family: {"payments": payments[family], "accounts": len(accounts[family])} for family in accounts}
    return CheckOutcome(key_number="; ".join(parts), evidence=evidence)


async def run_late_lien_observation_check(workspace: Any) -> CheckOutcome:
    """
    WHAT: Evaluates a lien the gang learns of late. A lien is placed at minute L on a mule; the gang sees it at L+D
    where D is the feed delay. It counts fraud payments through the mule in [L, L+D) and at or after L+D.
    WHY: Proves Check 19: a gang keeps using an account until it observes the lien, then stops.
    Args:
        workspace: Optional execution workspace supplied by the showcase framework.
    Returns:
        CheckOutcome reporting L, L+D and the two payment counts.
    Example:
        outcome = await run_late_lien_observation_check(None)
        assert "money flows" in outcome.key_number
    """
    output, lien, delay = get_lien_run()
    mule = str(lien.account_id)
    seen_minute = lien.minute + delay
    mules = {role.account_id for role in output.result.truth.roles if role.role == "mule"}
    assert mule in mules, f"Expected {mule} to be a mule of the gang"

    gang_events = [event for event, _gang in _fraud_events_with_gang(output)]
    events = [event for event in gang_events if mule in (event.from_account, event.to_account)]
    before_seen = [event for event in events if lien.minute <= event.minute < seen_minute]
    after_seen = [event for event in events if event.minute >= seen_minute]
    assert before_seen, f"Expected payments through {mule} between {lien.minute} and {seen_minute}"
    assert not after_seen, f"Found {len(after_seen)} fraud payments through {mule} at or after {seen_minute}"

    key_number = (
        f"lien at minute {lien.minute}, seen at {seen_minute} (delay {delay}); {mule}: "
        f"{len(before_seen)} fraud payments in [{lien.minute}, {seen_minute}), {len(after_seen)} at or after"
    )
    evidence = {
        "lien_minute": lien.minute, "seen_minute": seen_minute, "delay": delay, "mule": mule,
        "payments_before_seen": len(before_seen), "payments_after_seen": len(after_seen),
        "cancelled_by_gang": len(output.result.cancelled_by_gang),
    }
    return CheckOutcome(key_number=key_number, evidence=evidence)


async def run_held_out_in_training_refusal_check(workspace: Any) -> CheckOutcome:
    """
    WHAT: Runs a gang that names a held-out family in a training world and shows the plain refusal. A counting
    bank hook and a counting runner state factory prove that nothing was processed or built before the refusal.
    WHY: Proves Check 20: a world file that names a held-out family cannot be used for training.
    Args:
        workspace: Optional execution workspace supplied by the showcase framework.
    Returns:
        CheckOutcome carrying the refusal sentence.
    Example:
        outcome = await run_held_out_in_training_refusal_check(None)
        assert outcome.key_number.startswith("refused")
    """
    gang, message, calls = run_held_out_in_training()
    held_out = load_tactic_families()["held_out"]
    assert gang.family in held_out, f"Fixture family {gang.family!r} must be a held-out family"
    assert "held-out family" in message and str(gang.family) in message, f"Unexpected refusal text: {message}"
    assert calls == 0, f"Expected no payments or state before the refusal, saw {calls} calls"

    key_number = f"refused before planning, 0 events: {message}"
    return CheckOutcome(key_number=key_number, evidence={"family": gang.family, "refusal": message, "calls": calls})
