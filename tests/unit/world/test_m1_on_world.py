# File: tests/unit/world/test_m1_on_world.py
# Purpose: Real-world simulation integration test proving routine life features on genuine and rented mule accounts.
# Issue: Level 15b Round 2b-2 real-world demonstration (G1, Task 1).
"""
WHAT: Integration test executing a >= 70 day fraud-enabled simulation run:
  1. Spawns a syndicate recruiting rented mule accounts, including a salaried employee.
  2. Projects simulation telemetry to BankFeed external API via project_run().
  3. Converts all bank-visible payments, lifecycle events, and withdrawals to feature records
     with arrival_time = event_time + 60s.
  4. Ingests all records into FakeRedis account state.
  5. Computes M1 routine_life features at Day 65:
     - Asserts rented mule (salaried employee) has routine_life_present == True and salary_like_payer_count >= 1.
     - Asserts genuine salaried person also has routine_life_present == True and salary_like_payer_count >= 1.
     - Asserts early decision (10 days history) has routine_life_present is None (not enough history).
  6. Prints the required MEASURED line documenting seed, duration, and feature values.
"""

from datetime import timedelta

import fakeredis

from core.event_clock import world_minute_to_ist
from features.account_keys import load_account_features_settings
from features.account_records import (
    AccountEventRecord,
    PaymentSideRecord,
    WithdrawalRecord,
)
from features.account_updater import (
    apply_account_event,
    apply_payment_side,
    apply_withdrawal,
)
from features.m1_features import compute_account_features
from world.bank_feed import project_run
from world.fast_mode import FastRunSettings
from world.gang_config import BlockStep, GangBudget, GangConfig
from world.interactive_records import RunnerSettings
from world.interactive_runner import run_interactive_world


def test_m1_routine_life_on_real_world() -> None:
    """Verifies routine life feature calculation on a real 70-day world with rented mules and genuine employees."""
    seed = 1
    days = 70
    budget = GangBudget(money_paise=50_000_000, time_minutes=500_000, max_risk=0.99)
    gang = GangConfig(
        "g1",
        1,
        budget,
        None,
        (
            BlockStep("acquire", (("count", 6), ("origin_mix", "all_rented"))),
            BlockStep("scam", (("receiving_mules", 2), ("victims_per_receiving_mule", 2))),
            BlockStep("route", (("layers", 2), ("accounts_per_layer", 2))),
        ),
        1440,
    )

    out = run_interactive_world(
        FastRunSettings(seed, 60, 12, days, "training"),
        runner_settings=RunnerSettings(fraud_enabled=True),
        gang_configs=(gang,),
    )

    feed = project_run(out)
    client = fakeredis.FakeRedis()
    settings = load_account_features_settings()

    # Ingest payments into Redis state with arrival_time = event_time + 60s
    for sides in feed.payments_by_bank.values():
        for s in sides:
            rec_side = PaymentSideRecord(
                run_id=out.result.run_id,
                bank_id=s.bank_id,
                account_id=s.account_id,
                payment_id=s.payment_id,
                side=s.side,
                counterparty_bank_id=s.counterparty_bank_id,
                counterparty_vpa=s.counterparty_vpa,
                amount_paise=s.amount_paise,
                status=s.status,
                failure_domain=s.failure_domain,
                payee_type=s.payee_type,
                channel=s.channel,
                sender_device_fingerprint=s.sender_device_fingerprint,
                sender_os_generation=s.sender_os_generation,
                balance_after_paise=s.balance_after_paise,
                event_time=s.event_time,
                arrival_time=s.event_time + timedelta(seconds=60),
            )
            apply_payment_side(client, rec_side, settings)

    # Ingest account events
    for events in feed.account_events_by_bank.values():
        for ev in events:
            rec_ev = AccountEventRecord(
                run_id=out.result.run_id,
                bank_id=ev.bank_id,
                account_id=ev.account_id,
                event_id=ev.event_id,
                kind=ev.kind,
                event_time=ev.event_time,
                arrival_time=ev.event_time + timedelta(seconds=60),
                details=ev.details,
            )
            apply_account_event(client, rec_ev, settings)

    # Ingest withdrawals
    for withdrawals in feed.withdrawals_by_bank.values():
        for w in withdrawals:
            rec_w = WithdrawalRecord(
                run_id=out.result.run_id,
                bank_id=w.bank_id,
                account_id=w.account_id,
                withdrawal_id=w.withdrawal_id,
                channel=w.channel,
                amount_paise=w.amount_paise,
                status=w.status,
                failure_domain=w.failure_domain,
                balance_after_paise=w.balance_after_paise,
                event_time=w.event_time,
                arrival_time=w.event_time + timedelta(seconds=60),
            )
            apply_withdrawal(client, rec_w, settings)

    start_time = world_minute_to_ist(0, out.plan.clock.start_date)
    dec_65 = start_time + timedelta(days=65)

    # 1. Identify rented mule who is a salaried employee
    rented_roles = [r for r in out.result.truth.roles if r.role == "mule" and r.origin == "rented"]
    mule_acc = next(
        r.account_id
        for r in rented_roles
        if any(p.account_id == r.account_id and p.persona_name == "salaried_employee" for p in out.world.people)
    )
    mule_bank = next(
        s.bank_id for sides in feed.payments_by_bank.values() for s in sides if s.account_id == mule_acc
    )

    f_mule = compute_account_features(
        client, out.result.run_id, mule_bank, mule_acc, dec_65, dec_65, settings, groups=["routine_life"]
    )
    assert f_mule["routine_life_present"] is True
    assert isinstance(f_mule["salary_like_payer_count"], int)
    assert f_mule["salary_like_payer_count"] >= 1

    # 2. Identify genuine salaried employee (never rented)
    rented_accounts = {r.account_id for r in rented_roles}
    gen_person = next(
        p for p in out.world.people
        if p.persona_name == "salaried_employee" and p.account_id not in rented_accounts
    )
    gen_acc = gen_person.account_id
    gen_bank = next(
        s.bank_id for sides in feed.payments_by_bank.values() for s in sides if s.account_id == gen_acc
    )

    f_gen = compute_account_features(
        client, out.result.run_id, gen_bank, gen_acc, dec_65, dec_65, settings, groups=["routine_life"]
    )
    assert f_gen["routine_life_present"] is True
    assert isinstance(f_gen["salary_like_payer_count"], int)
    assert f_gen["salary_like_payer_count"] >= 1

    # 3. 10-day history evaluation -> routine_life_present is None (not enough history)
    dec_10 = start_time + timedelta(days=10)
    f_10d = compute_account_features(
        client, out.result.run_id, gen_bank, gen_acc, dec_10, dec_10, settings, groups=["routine_life"]
    )
    assert f_10d["history_days_available"] == 10
    assert f_10d["routine_life_present"] is None
    assert f_10d["salary_like_payer_count"] is None

    # 4. MEASURED statement line
    print(
        f"\nMEASURED: seed={seed}, days={days}, "
        f"mule ({mule_acc}) salary_count={f_mule['salary_like_payer_count']} "
        f"history_days={f_mule['history_days_available']}, "
        f"genuine ({gen_acc}) salary_count={f_gen['salary_like_payer_count']} "
        f"history_days={f_gen['history_days_available']}"
    )
