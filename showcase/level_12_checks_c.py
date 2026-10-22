"""
WHAT: Implementation of Level 12 Showcase Part C checks (Checks 11 through 15).
Evaluates victim payment scam generation, multi-hop route forwarding, multi-channel cash-out,
rented mule routine preservation, temporal lifecycle phase transitions, downstream genuine merchant
interactions, and outage-driven forward suppression.

WHY: Verifies Level 12c reactive world engine capabilities: dynamic victim fraud injection,
temporal lifecycle phases from warm-up to burned, non-illicit activity co-existence for rented owners,
and double-entry money preservation across crypto, ATM, branch, and retail merchant exits.

TERM: Level 12: World engine II supporting reactive outside events and interactive fraud simulation.
TERM: cash-out: Channel through which laundered money exits mule accounts into physical cash or retail.
TERM: rented mule: An existing account whose owner rented credentials while maintaining normal routine.
TERM: temporal phase: Account status resolving to warm_up, active, or burned depending on simulation minute.
"""

from typing import Any

from showcase.framework import CheckOutcome
from showcase.level_12_support_c import get_showcase_c_outage_run, get_showcase_c_run


async def run_victim_to_cashout_check(workspace: Any) -> CheckOutcome:
    """
    WHAT: Evaluates end-to-end victim-to-cash-out flow: victim scam payments into layer 1 mules,
    layered forward routing across intermediate mules, and terminal cash-out extractions.
    WHY: Proves Check 11: a single scam gang successfully conducts victim fraud, forwards funds,
    and disburses laundered money across cash-out channels.
    Args:
        workspace: Optional execution workspace supplied by the showcase framework.
    Returns:
        CheckOutcome reporting counts of victim payments, forwards, cash-out transactions, and total Rs.
    Example:
        outcome = await run_victim_to_cashout_check(None)
        assert "victim payments ->" in outcome.key_number
    """
    output, _store = get_showcase_c_run()
    events = output.result.run_result.events
    sources = output.result.event_sources
    stories = output.result.truth.payment_stories

    # 1. Filter fraud transactions by story block categorization.
    fraud_pairs = [
        (e, stories[output.result.event_fraud_input_index[i]])
        for i, (e, s) in enumerate(zip(events, sources))
        if s == "fraud" and output.result.event_fraud_input_index[i] is not None
    ]
    victim_pmts = [p for p in fraud_pairs if p[1].block == "scam" and "payout back" not in p[1].description]
    forwards = [p for p in fraud_pairs if p[1].block == "route"]
    shop_pmts = [p for p in fraud_pairs if p[1].block == "cash_out" and "genuine retail merchant" in p[1].description]
    crypto_pmts = [p for p in fraud_pairs if p[1].block == "cash_out" and "crypto collector" in p[1].description]
    withdrawals = [w for w in output.result.cash_withdrawals if w.status == "success"]

    # 2. Identify active cash-out channels and compute total extracted currency.
    channels: set[str] = set()
    if crypto_pmts:
        channels.add("crypto")
    if any(w.channel == "atm" for w in withdrawals):
        channels.add("atm")
    if any(w.channel == "branch" for w in withdrawals):
        channels.add("branch")
    if shop_pmts:
        channels.add("genuine_shop")

    total_withdrawn = output.result.money_withdrawn_paise
    total_shop = sum(p[0].amount_paise for p in shop_pmts)
    total_crypto = sum(p[0].amount_paise for p in crypto_pmts)
    total_cashout_rs = (total_withdrawn + total_shop + total_crypto) // 100
    cashout_count = len(withdrawals) + len(shop_pmts) + len(crypto_pmts)

    assert len(victim_pmts) > 0, "Expected at least one victim fraud payment"
    assert len(forwards) > 0, "Expected at least one route forward payment"
    assert cashout_count > 0, "Expected at least one cash-out transaction"

    channels_str = ", ".join(sorted(channels))
    flow_desc = f"{len(victim_pmts)} victim payments -> {len(forwards)} forwards"
    key_number = f"{flow_desc} -> {cashout_count} cash-outs ({channels_str}), Rs {total_cashout_rs:,}"
    evidence = {
        "victim_payments": len(victim_pmts),
        "forwards": len(forwards),
        "cashouts": cashout_count,
        "channels": sorted(channels),
        "total_cashout_rs": total_cashout_rs,
    }
    return CheckOutcome(key_number=key_number, evidence=evidence)


async def run_rented_mule_normal_life_check(workspace: Any) -> CheckOutcome:
    """
    WHAT: Asserts that a rented mule continues making genuine payments during its rental window
    while selective outgoing owner payments are paused per Decision D2.
    WHY: Proves Check 12: renting an account compromises financial integrity without halting
    all genuine owner routines, maintaining realistic mixed-traffic behavior.
    Args:
        workspace: Optional execution workspace supplied by the showcase framework.
    Returns:
        CheckOutcome reporting genuine payments made during rental and the number paused.
    Example:
        outcome = await run_rented_mule_normal_life_check(None)
        assert "genuine payments made" in outcome.key_number
    """
    output, store = get_showcase_c_run()
    events = output.result.run_result.events
    sources = output.result.event_sources

    # 1. Locate rented mule role from truth store.
    rented_roles = [r for r in store._roles if r["role"] == "mule" and r.get("origin") == "rented"]
    assert rented_roles, "Expected at least one rented mule role in truth"
    mule_role = rented_roles[0]
    acct = mule_role["account_id"]
    valid_from = mule_role["valid_from"]
    burned_from = mule_role["burned_from"] or float("inf")

    # 2. Count genuine payments executed by this owner inside rental span and paused items.
    made = [
        e for e, s in zip(events, sources)
        if s == "planned" and e.from_account == acct and valid_from <= e.minute < burned_from
    ]
    paused = [p for p in output.result.paused_owner_payments if p.from_account == acct]

    assert len(made) > 0, f"Expected rented mule {acct} to execute genuine payments during rental"
    assert len(paused) > 0, f"Expected rented mule {acct} to have paused owner payments recorded"

    key_number = f"rented mule {acct}: {len(made)} genuine payments made, {len(paused)} paused"
    evidence = {
        "account_id": acct,
        "payments_made_during_rental": len(made),
        "payments_paused_during_rental": len(paused),
        "valid_from": valid_from,
        "burned_from": mule_role["burned_from"],
    }
    return CheckOutcome(key_number=key_number, evidence=evidence)


async def run_mule_lifecycle_check(workspace: Any) -> CheckOutcome:
    """
    WHAT: Evaluates temporal phase transitions of a mule account across warm_up, active, and burned stages.
    WHY: Proves Check 13: mule lifecycle progression strictly enforces temporal validity intervals.
    Args:
        workspace: Optional execution workspace supplied by the showcase framework.
    Returns:
        CheckOutcome reporting phase_at evaluations at the three transition boundary minutes.
    Example:
        outcome = await run_mule_lifecycle_check(None)
        assert "warm_up" in outcome.key_number and "burned" in outcome.key_number
    """
    _output, store = get_showcase_c_run()

    # 1. Select mule account possessing complete lifecycle metadata.
    mule_roles = [
        r for r in store._roles
        if r["role"] == "mule" and r.get("active_from") is not None and r.get("burned_from") is not None
    ]
    assert mule_roles, "Expected at least one mule with complete lifecycle fields in truth"
    selected = mule_roles[0]
    acct = selected["account_id"]
    min_warmup = selected["valid_from"]
    min_active = selected["active_from"]
    min_burned = selected["burned_from"]

    # 2. Query truth store phase_at at transition minutes.
    phase_warmup = store.phase_at(acct, min_warmup)
    phase_active = store.phase_at(acct, min_active)
    phase_burned = store.phase_at(acct, min_burned)

    assert phase_warmup == "warm_up", f"Expected warm_up at minute {min_warmup}, got {phase_warmup}"
    assert phase_active == "active", f"Expected active at minute {min_active}, got {phase_active}"
    assert phase_burned == "burned", f"Expected burned at minute {min_burned}, got {phase_burned}"

    phases_desc = (
        f"minute {min_warmup} {phase_warmup}, "
        f"minute {min_active} {phase_active}, "
        f"minute {min_burned} {phase_burned}"
    )
    key_number = f"mule {acct}: {phases_desc}"
    evidence = {
        "account_id": acct,
        "valid_from": min_warmup,
        "active_from": min_active,
        "burned_from": min_burned,
        "phases": [phase_warmup, phase_active, phase_burned],
    }
    return CheckOutcome(key_number=key_number, evidence=evidence)


async def run_genuine_shop_downstream_check(workspace: Any) -> CheckOutcome:
    """
    WHAT: Verifies that a genuine retail merchant receives illicit mule money through a cash-out
    payment while maintaining ordinary genuine legal status (role_at = genuine) in truth.
    WHY: Proves Check 14: retail cash-out transfers stolen funds into legitimate economic circulation
    without erroneously branding innocent commercial merchants as criminal conspirators.
    Args:
        workspace: Optional execution workspace supplied by the showcase framework.
    Returns:
        CheckOutcome reporting the shop identifier, received paise amount, and genuine role status.
    Example:
        outcome = await run_genuine_shop_downstream_check(None)
        assert "role_at = genuine" in outcome.key_number
    """
    output, store = get_showcase_c_run()
    events = output.result.run_result.events
    sources = output.result.event_sources
    stories = output.result.truth.payment_stories

    # 1. Identify retail merchant payments emitted by cash_out tactic block.
    shop_pmts = [
        (e, stories[output.result.event_fraud_input_index[i]])
        for i, (e, s) in enumerate(zip(events, sources))
        if s == "fraud" and output.result.event_fraud_input_index[i] is not None
        and stories[output.result.event_fraud_input_index[i]].block == "cash_out"
        and "genuine retail merchant" in stories[output.result.event_fraud_input_index[i]].description
    ]
    assert shop_pmts, "Expected at least one payment to a genuine merchant shop under cash_out"
    sp = shop_pmts[0]
    shop_id = sp[0].to_account
    amt_paise = sp[0].amount_paise
    minute = sp[0].minute

    # 2. Confirm the shop retains role_at = genuine in truth at transaction minute.
    role = store.role_at(shop_id, minute)
    assert role == "genuine", f"Shop '{shop_id}' unexpectedly holds role '{role}' in truth"

    key_number = f"shop {shop_id} received Rs {amt_paise // 100:,} ({amt_paise} paise), role_at = {role}"
    evidence = {
        "shop_account": shop_id,
        "amount_paise": amt_paise,
        "amount_rs": amt_paise // 100,
        "role_at": role,
        "minute": minute,
    }
    return CheckOutcome(key_number=key_number, evidence=evidence)


async def run_victim_payment_outage_check(workspace: Any) -> CheckOutcome:
    """
    WHAT: Reuses the pinned bank downtime outage scenario from test_victim_money_route.py to assert
    that a victim payment failing in an outage causes dependent forwarding transfers to be skipped.
    WHY: Proves Check 15: external bank infrastructure failures prevent stolen money generation
    and safely suppress subsequent gang routing without ledger side effects.
    Args:
        workspace: Optional execution workspace supplied by the showcase framework.
    Returns:
        CheckOutcome reporting the failure reason, failure domain, and skipped forward details.
    Example:
        outcome = await run_victim_payment_outage_check(None)
        assert "bank_downtime" in outcome.key_number
    """
    output = get_showcase_c_outage_run()
    events = output.result.run_result.events
    sources = output.result.event_sources

    # 1. Locate failed victim payment transaction.
    failed_pmts = [
        e for e, s in zip(events, sources)
        if s == "fraud" and e.status == "failed"
    ]
    assert failed_pmts, "Expected at least one failed fraud payment in outage run"
    failed = failed_pmts[0]

    # 2. Confirm forward was skipped due to unreceived illicit money.
    assert output.result.skipped_forwards, "Expected skipped forward records when victim payment failed"
    skipped = output.result.skipped_forwards[0]

    key_number = (
        f"victim payment failed with {failed.reason} ({failed.failure_domain}); "
        f"forward from {skipped.from_account} of Rs {skipped.planned_paise // 100:,} skipped ({skipped.reason})"
    )
    evidence = {
        "failed_payment_minute": failed.minute,
        "failed_reason": failed.reason,
        "failure_domain": failed.failure_domain,
        "skipped_from_account": skipped.from_account,
        "skipped_planned_paise": skipped.planned_paise,
        "skipped_reason": skipped.reason,
    }
    return CheckOutcome(key_number=key_number, evidence=evidence)
