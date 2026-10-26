"""
tests/unit/nationalsim/test_hold_flow.py

Unit test suite for pure 1930 administrative hold flow logic (nationalsim/hold_flow.py).
Verifies exactly-once complaint hold issuance, composite latency formula across time-of-day
and weekend boundaries, unreachable delay conditional addition, four-way outcome classification,
and strictly single-hop forwarding constraints under Amendment A-029.
"""

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from nationalsim.hold_flow import (
    HoldFlowTracker,
    HoldOutcome,
    HoldRequest,
    OutgoingTransfer,
    classify_hold_outcome,
    compute_hold_latency,
    decide_one_hop_forwarding,
    is_night_hour,
)

IST = ZoneInfo("Asia/Kolkata")


def test_one_complaint_gives_exactly_one_hold_and_duplicate_gives_none() -> None:
    """Proves that a complaint creates exactly one hold directive and duplicate complaint_ids return None.

    What it is for:
        Guarantees that 1930 hold generation is idempotent. Re-submitting the same complaint
        must never spawn multiple liens or duplicate hold requests against a receiving bank.

    Inputs:
        None. Submits a complaint_id once, then attempts to submit the identical complaint_id again.

    What it returns:
        None. Asserts the first call returns a HoldRequest and the second call returns None.

    How it works:
        1. Initialize a clean HoldFlowTracker instance.
        2. Call create_hold_request for 'complaint-001'; verify valid HoldRequest is returned.
        3. Call create_hold_request again with 'complaint-001'; verify it returns None.
        4. Call create_hold_request with a distinct 'complaint-002'; verify it returns a second request.

    Errors and side effects:
        Raises AssertionError if first call fails or duplicate call does not return None.

    Limits:
        Evaluates in-memory idempotency tracker.
    """
    tracker = HoldFlowTracker()

    # 1. First submission succeeds and delivers a valid hold request
    hold1 = tracker.create_hold_request(
        complaint_id="complaint-001",
        payment_id="pay-101",
        receiving_bank_index=1,
        account_id="acc_mule_01",
        amount_paise=500000,
        landing_minute=120,
    )
    assert hold1 is not None
    assert hold1.complaint_id == "complaint-001"
    assert hold1.account_id == "acc_mule_01"
    assert hold1.amount_paise == 500000
    assert hold1.landing_minute == 120
    assert hold1.is_forwarded is False

    # 2. Duplicate submission with identical complaint_id returns None
    hold_dup = tracker.create_hold_request(
        complaint_id="complaint-001",
        payment_id="pay-101",
        receiving_bank_index=1,
        account_id="acc_mule_01",
        amount_paise=500000,
        landing_minute=125,
    )
    assert hold_dup is None, "Duplicate complaint_id must return None rather than duplicating hold"

    # 3. Fresh complaint_id produces a new hold request
    hold2 = tracker.create_hold_request(
        complaint_id="complaint-002",
        payment_id="pay-102",
        receiving_bank_index=2,
        account_id="acc_mule_02",
        amount_paise=750000,
        landing_minute=140,
    )
    assert hold2 is not None
    assert hold2.complaint_id == "complaint-002"


def test_latency_formula_exact_minutes_day_night_weekday_weekend() -> None:
    """Proves that compute_hold_latency applies exact multipliers for day vs night and weekday vs weekend.

    What it is for:
        Verifies that operational hold latency accurately scales according to institutional
        delay factors and staffing variations across diurnal and weekly cycles.

    Inputs:
        None. Evaluates four deterministic timestamps covering all permutations:
        (day weekday, night weekday, day weekend, night weekend).

    What it returns:
        None. Asserts returned integer minutes match exact theoretical calculations.

    How it works:
        1. Set fixed base latency = 20.0 minutes, bank factor = 1.5, night hours = (22, 6).
        2. Set night_multiplier = 2.0, weekend_multiplier = 1.5, unreachable extra = 0.
        3. Day Weekday (Wednesday 14:00): 20 * 1.5 * 1.0 * 1.0 = 30 minutes.
        4. Night Weekday (Wednesday 23:00): 20 * 1.5 * 2.0 * 1.0 = 60 minutes.
        5. Day Weekend (Saturday 14:00): 20 * 1.5 * 1.0 * 1.5 = 45 minutes.
        6. Night Weekend (Saturday 23:00): 20 * 1.5 * 2.0 * 1.5 = 90 minutes.

    Errors and side effects:
        Raises AssertionError if computed minutes deviate from formula bounds.

    Limits:
        Tests clean integer product combinations.
    """
    base_latency = 20.0
    bank_factor = 1.5
    night_window = (22, 6)
    night_mult = 2.0
    wknd_mult = 1.5

    # 2026-09-02 is Wednesday (weekday=2)
    dt_day_weekday = datetime(2026, 9, 2, 14, 0, tzinfo=IST)
    dt_night_weekday = datetime(2026, 9, 2, 23, 0, tzinfo=IST)

    # 2026-09-05 is Saturday (weekday=5)
    dt_day_weekend = datetime(2026, 9, 5, 14, 0, tzinfo=IST)
    dt_night_weekend = datetime(2026, 9, 5, 23, 0, tzinfo=IST)

    # 1. Day Weekday: 20 * 1.5 * 1.0 * 1.0 = 30
    lat_dw = compute_hold_latency(
        base_latency_minutes=base_latency,
        bank_hold_delay_factor=bank_factor,
        arrival_dt=dt_day_weekday,
        night_hours=night_window,
        night_multiplier=night_mult,
        weekend_multiplier=wknd_mult,
        u_unreachable=0.5,
        unreachable_share=0.2,  # 0.5 >= 0.2: reachable, extra delay 0
        unreachable_extra_delay_minutes=50.0,
    )
    assert lat_dw == 30

    # 2. Night Weekday: 20 * 1.5 * 2.0 * 1.0 = 60
    lat_nw = compute_hold_latency(
        base_latency_minutes=base_latency,
        bank_hold_delay_factor=bank_factor,
        arrival_dt=dt_night_weekday,
        night_hours=night_window,
        night_multiplier=night_mult,
        weekend_multiplier=wknd_mult,
        u_unreachable=0.5,
        unreachable_share=0.2,
        unreachable_extra_delay_minutes=50.0,
    )
    assert lat_nw == 60

    # 3. Day Weekend: 20 * 1.5 * 1.0 * 1.5 = 45
    lat_dwe = compute_hold_latency(
        base_latency_minutes=base_latency,
        bank_hold_delay_factor=bank_factor,
        arrival_dt=dt_day_weekend,
        night_hours=night_window,
        night_multiplier=night_mult,
        weekend_multiplier=wknd_mult,
        u_unreachable=0.5,
        unreachable_share=0.2,
        unreachable_extra_delay_minutes=50.0,
    )
    assert lat_dwe == 45

    # 4. Night Weekend: 20 * 1.5 * 2.0 * 1.5 = 90
    lat_nwe = compute_hold_latency(
        base_latency_minutes=base_latency,
        bank_hold_delay_factor=bank_factor,
        arrival_dt=dt_night_weekend,
        night_hours=night_window,
        night_multiplier=night_mult,
        weekend_multiplier=wknd_mult,
        u_unreachable=0.5,
        unreachable_share=0.2,
        unreachable_extra_delay_minutes=50.0,
    )
    assert lat_nwe == 90


def test_unreachable_extra_delay_added_only_when_draw_says_so() -> None:
    """Proves that unreachable_1930_extra_delay is added when u < share and omitted when u >= share.

    What it is for:
        Verifies that congestion latency on the 1930 reporting hotline affects only the designated
        fraction of callers whose random uniform draw indicates initial connection failure.

    Inputs:
        None. Compares identical baseline configurations with u_unreachable below and above threshold.

    What it returns:
        None. Asserts extra delay is applied conditionally and exactly.

    How it works:
        1. Set base = 10, bank factor = 1.0, day weekday (multiplier = 1.0).
        2. Set unreachable_share = 0.25, extra delay = 45.0.
        3. Test u = 0.10 (< 0.25): expected latency = 10 + 45 = 55 minutes.
        4. Test u = 0.25 (== 0.25): expected latency = 10 + 0 = 10 minutes.
        5. Test u = 0.80 (> 0.25): expected latency = 10 + 0 = 10 minutes.

    Errors and side effects:
        Raises AssertionError if extra delay is added unconditionally or omitted when drawn.

    Limits:
        Evaluates boundary inequality comparison u < share.
    """
    dt_day = datetime(2026, 9, 2, 12, 0, tzinfo=IST)
    night_window = (22, 6)

    # 1. Congested case: u=0.10 < share=0.25 -> extra delay added
    latency_congested = compute_hold_latency(
        base_latency_minutes=10.0,
        bank_hold_delay_factor=1.0,
        arrival_dt=dt_day,
        night_hours=night_window,
        night_multiplier=2.0,
        weekend_multiplier=1.5,
        u_unreachable=0.10,
        unreachable_share=0.25,
        unreachable_extra_delay_minutes=45.0,
    )
    assert latency_congested == 55

    # 2. Boundary case: u=0.25 is not strictly less than share=0.25 -> extra delay omitted
    latency_boundary = compute_hold_latency(
        base_latency_minutes=10.0,
        bank_hold_delay_factor=1.0,
        arrival_dt=dt_day,
        night_hours=night_window,
        night_multiplier=2.0,
        weekend_multiplier=1.5,
        u_unreachable=0.25,
        unreachable_share=0.25,
        unreachable_extra_delay_minutes=45.0,
    )
    assert latency_boundary == 10

    # 3. Successful first-try case: u=0.80 > share=0.25 -> extra delay omitted
    latency_reachable = compute_hold_latency(
        base_latency_minutes=10.0,
        bank_hold_delay_factor=1.0,
        arrival_dt=dt_day,
        night_hours=night_window,
        night_multiplier=2.0,
        weekend_multiplier=1.5,
        u_unreachable=0.80,
        unreachable_share=0.25,
        unreachable_extra_delay_minutes=45.0,
    )
    assert latency_reachable == 10


def test_outcome_classification_covers_applied_too_late_partial_missed() -> None:
    """Proves that classify_hold_outcome exhaustively covers applied, partial, too_late, and missed.

    What it is for:
        Guarantees that every administrative hold maps to a mathematically consistent outcome
        based on available account balance at landing minute and bank action status.

    Inputs:
        None. Tests all four outcome states with explicit monetary balances.

    What it returns:
        None. Asserts status and held_amount_paise for each case.

    How it works:
        1. Missed: missed=True -> status='missed', held=0.
        2. Applied: balance >= requested (10,000 >= 10,000) -> status='applied', held=10,000.
        3. Too Late: balance <= 0 (0 <= 10,000) -> status='too_late', held=0.
        4. Partial: 0 < balance < requested (4,000 < 10,000) -> status='partial', held=4,000.

    Errors and side effects:
        Raises AssertionError if any outcome status or held balance is misclassified.

    Limits:
        Evaluates point-in-time financial classification.
    """
    req_paise = 1000000  # Rs 10,000
    acc = "acc_target_01"
    minute = 300

    # 1. Missed: Bank never acted on the directive
    res_missed = classify_hold_outcome(
        available_balance_paise=1500000,
        requested_amount_paise=req_paise,
        account_id=acc,
        landing_minute=minute,
        missed=True,
    )
    assert res_missed.status == "missed"
    assert res_missed.held_amount_paise == 0
    assert res_missed.requested_amount_paise == req_paise

    # 2. Applied: Entire requested balance is available
    res_applied = classify_hold_outcome(
        available_balance_paise=1200000,
        requested_amount_paise=req_paise,
        account_id=acc,
        landing_minute=minute,
        missed=False,
    )
    assert res_applied.status == "applied"
    assert res_applied.held_amount_paise == req_paise

    # 3. Too late: Account was depleted prior to hold landing
    res_too_late = classify_hold_outcome(
        available_balance_paise=0,
        requested_amount_paise=req_paise,
        account_id=acc,
        landing_minute=minute,
        missed=False,
    )
    assert res_too_late.status == "too_late"
    assert res_too_late.held_amount_paise == 0

    # 4. Partial: Some funds remain, but less than full amount
    res_partial = classify_hold_outcome(
        available_balance_paise=400000,
        requested_amount_paise=req_paise,
        account_id=acc,
        landing_minute=minute,
        missed=False,
    )
    assert res_partial.status == "partial"
    assert res_partial.held_amount_paise == 400000


def test_forwarding_is_exactly_one_hop() -> None:
    """Proves that naive forwarding issues one hop for departed funds and never forwards a second time.

    What it is for:
        Enforces Amendment A-029 boundary constraint: receiving banks trace money only one step
        further. Transitive multi-hop chaining is explicitly prevented.

    Inputs:
        None. Evaluates primary hold forwarding followed by attempting to forward the forwarded hold.

    What it returns:
        None. Asserts primary hold forwards to the next bank and forwarded hold returns None.

    How it works:
        1. Construct original primary HoldRequest (is_forwarded=False).
        2. Set outcome='too_late' with outgoing transfer to bank 2.
        3. Call decide_one_hop_forwarding; verify forwarded request targets bank 2 with is_forwarded=True.
        4. Attempt to pass the forwarded request back into decide_one_hop_forwarding; verify it returns None.
        5. Verify that an 'applied' outcome does not forward.

    Errors and side effects:
        Raises AssertionError if forwarding chains beyond one hop or forwards an applied hold.

    Limits:
        Evaluates single-institution ledger traceback rules.
    """
    primary_hold = HoldRequest(
        complaint_id="c-100",
        payment_id="pay-origin",
        receiving_bank_index=0,
        account_id="acc_mule_1",
        amount_paise=1000000,
        landing_minute=180,
        is_forwarded=False,
    )

    outcome_too_late = HoldOutcome(
        status="too_late",
        held_amount_paise=0,
        requested_amount_paise=1000000,
        account_id="acc_mule_1",
        landing_minute=180,
    )

    departed_transfers = [
        OutgoingTransfer(
            transfer_id="tx_depart_1",
            from_account="acc_mule_1",
            to_account="acc_mule_2",
            to_bank_index=2,
            amount_paise=800000,
            minute=150,  # Posted before hold landed at minute 180
        )
    ]

    # 1. Primary hold with too_late outcome creates a single forwarded hold
    forwarded_hold = decide_one_hop_forwarding(primary_hold, outcome_too_late, departed_transfers)
    assert forwarded_hold is not None
    assert forwarded_hold.is_forwarded is True
    assert forwarded_hold.receiving_bank_index == 2
    assert forwarded_hold.account_id == "acc_mule_2"
    assert forwarded_hold.amount_paise == 800000
    assert forwarded_hold.originating_hold_id == "c-100"

    # 2. Attempting to forward the already-forwarded hold MUST return None (strictly one hop)
    forwarded_outcome = HoldOutcome(
        status="too_late",
        held_amount_paise=0,
        requested_amount_paise=800000,
        account_id="acc_mule_2",
        landing_minute=220,
    )
    second_hop_transfers = [
        OutgoingTransfer(
            transfer_id="tx_depart_2",
            from_account="acc_mule_2",
            to_account="acc_mule_3",
            to_bank_index=1,
            amount_paise=800000,
            minute=200,
        )
    ]
    second_forward = decide_one_hop_forwarding(forwarded_hold, forwarded_outcome, second_hop_transfers)
    assert second_forward is None, "Forwarding must terminate after exactly one hop"

    # 3. An applied outcome does not forward because funds were successfully held
    outcome_applied = HoldOutcome(
        status="applied",
        held_amount_paise=1000000,
        requested_amount_paise=1000000,
        account_id="acc_mule_1",
        landing_minute=180,
    )
    forward_for_applied = decide_one_hop_forwarding(primary_hold, outcome_applied, departed_transfers)
    assert forward_for_applied is None, "Applied hold must not emit forwarded hold"
