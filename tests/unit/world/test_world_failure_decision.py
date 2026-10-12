"""
WHAT: Unit tests D1-D9 for payment failure decision evaluation, degradation, and determinism.
WHY: Proves that payments fail only within active incident windows or per-configured background rates,
     attributing each failure to its correct operational failure domain (issue R7).
ISSUE IDs: R7 (failure decision and domain tagging), E4 (bank differences), R2 (background realism).
TERM: failure decision: The deterministic function evaluating whether an attempted payment fails and why.
"""

import hashlib
import subprocess

import numpy as np
import pytest

from world.failure_decision import decide_failure
from world.failures import FAILURE_DOMAINS, FailureSettings, make_event
from world.incidents import Incident


def test_d1_zero_rates_no_incidents_zero_failures() -> None:
    """
    WHAT: Asserts that with no incidents and zero rates, all 5000 payments succeed.
    WHY: Verifies that the failure engine introduces zero spurious payment declines.
    """
    settings = FailureSettings(0.0, 0.0, 0.0)
    rng = np.random.default_rng(42)
    for _ in range(5000):
        res = decide_failure(100, 0, 1, [], settings, rng)
        assert res is None


def test_d2_npci_incident_bounds_and_failure_rate() -> None:
    """
    WHAT: Tests NPCI incident window bounds (minutes 999, 1000, 1099, 1100) and failure rate.
    WHY: Confirms exact [start, start+duration) window edges and 0.5 degradation probability.
    """
    inc = Incident("npci_incident", 1000, 100, None, 0.5)
    settings = FailureSettings(0.0, 0.0, 0.0)
    rng = np.random.default_rng(101)
    # Edge checks: outside window must never fail.
    assert all(decide_failure(999, 0, 1, [inc], settings, rng) is None for _ in range(1000))
    assert all(decide_failure(1100, 0, 1, [inc], settings, rng) is None for _ in range(1000))
    # Active window checks: failure fraction within 0.02 of 0.5.
    for m in (1000, 1099):
        fails = [decide_failure(m, 0, 1, [inc], settings, rng) for _ in range(20000)]
        non_none = [f for f in fails if f is not None]
        frac = len(non_none) / 20000.0
        assert abs(frac - 0.5) <= 0.02
        assert all(f == ("npci_central", "npci_incident") for f in non_none)


def test_d3_bank_downtime_routes_and_rates() -> None:
    """
    WHAT: Tests bank downtime routing: sender vs receiver vs unaffected institutions.
    WHY: Enforces that bank downtime only affects payment legs attached to that bank.
    """
    inc = Incident("bank_downtime", 500, 60, 1, 0.3)
    settings = FailureSettings(0.0, 0.0, 0.0)
    rng = np.random.default_rng(202)
    # Bank 1 -> Bank 0 (sender down, failure rate 0.7)
    res_10 = [decide_failure(510, 1, 0, [inc], settings, rng) for _ in range(20000)]
    f_10 = [r for r in res_10 if r is not None]
    assert abs(len(f_10) / 20000.0 - 0.7) <= 0.02
    assert all(r == ("sender_bank", "bank_downtime") for r in f_10)
    # Bank 0 -> Bank 1 (receiver down, failure rate 0.7)
    res_01 = [decide_failure(510, 0, 1, [inc], settings, rng) for _ in range(20000)]
    f_01 = [r for r in res_01 if r is not None]
    assert abs(len(f_01) / 20000.0 - 0.7) <= 0.02
    assert all(r == ("receiver_bank", "bank_downtime") for r in f_01)
    # Bank 0 -> Bank 2 (unaffected)
    assert all(decide_failure(510, 0, 2, [inc], settings, rng) is None for _ in range(20000))
    # Bank 1 -> Bank 1 (internal transfer: counted once as sender)
    res_11 = [decide_failure(510, 1, 1, [inc], settings, rng) for _ in range(20000)]
    f_11 = [r for r in res_11 if r is not None]
    assert abs(len(f_11) / 20000.0 - 0.7) <= 0.02
    assert all(r == ("sender_bank", "bank_downtime") for r in f_11)


def test_d4_partial_degradation_mixed_outcomes() -> None:
    """
    WHAT: Tests that 2000 payments during success=0.5 incident produce both outcomes.
    WHY: Guarantees partial degradation functions as a degraded channel, not binary blackout.
    """
    inc = Incident("npci_incident", 100, 50, None, 0.5)
    settings = FailureSettings(0.0, 0.0, 0.0)
    rng = np.random.default_rng(303)
    outcomes = [decide_failure(120, 0, 1, [inc], settings, rng) for _ in range(2000)]
    successes = sum(1 for o in outcomes if o is None)
    failures = sum(1 for o in outcomes if o is not None)
    assert successes > 0
    assert failures > 0


def test_d5_background_failure_proportions() -> None:
    """
    WHAT: Tests background failure distribution (customer, app/network, technical decline).
    WHY: Verifies multinomial partition across PIN errors, drops, and 3-way technical split.
    """
    settings = FailureSettings(0.01, 0.005, 0.0075)
    rng = np.random.default_rng(404)
    n_total = 100000
    res = [decide_failure(50, 0, 1, [], settings, rng) for _ in range(n_total)]
    c_count = sum(1 for r in res if r is not None and r[0] == "customer")
    a_count = sum(1 for r in res if r is not None and r[0] == "app_or_network")
    tech = [r for r in res if r is not None and r[1] == "technical_decline"]
    t_npci = sum(1 for r in tech if r[0] == "npci_central")
    t_send = sum(1 for r in tech if r[0] == "sender_bank")
    t_recv = sum(1 for r in tech if r[0] == "receiver_bank")

    assert abs(c_count / n_total - 0.01) <= 0.002
    assert abs(a_count / n_total - 0.005) <= 0.002
    assert abs(len(tech) / n_total - 0.0075) <= 0.002
    assert abs(t_npci / n_total - 0.0025) <= 0.002
    assert abs(t_send / n_total - 0.0025) <= 0.002
    assert abs(t_recv / n_total - 0.0025) <= 0.002


def test_d6_incident_overlap_precedence() -> None:
    """
    WHAT: Verifies that active NPCI incident takes precedence over simultaneous bank downtime.
    WHY: Establishes clear hierarchical precedence: central switch disruption preempts banks.
    """
    inc_npci = Incident("npci_incident", 200, 60, None, 0.0)
    inc_bank = Incident("bank_downtime", 200, 60, 0, 0.5)
    settings = FailureSettings(0.0, 0.0, 0.0)
    rng = np.random.default_rng(505)
    for _ in range(1000):
        res = decide_failure(220, 0, 1, [inc_npci, inc_bank], settings, rng)
        assert res == ("npci_central", "npci_incident")


def test_d7_make_event_status_and_validation() -> None:
    """
    WHAT: Tests make_event creation and rejection of illegal argument combinations.
    WHY: Protects PaymentEvent invariant: None domain = success, non-None = failed.
    """
    ev_ok = make_event(10, "acc1", "acc2", 1000, 0, 1, None, "ok")
    assert ev_ok.status == "success" and ev_ok.reason == "ok"
    for d in FAILURE_DOMAINS:
        ev_fail = make_event(10, "acc1", "acc2", 1000, 0, 1, d, "declined")
        assert ev_fail.status == "failed" and ev_fail.failure_domain == d

    # Rejection of illegal inputs.
    with pytest.raises(ValueError, match="Unknown failure_domain"):
        make_event(10, "acc1", "acc2", 1000, 0, 1, "invalid_domain", "declined")
    with pytest.raises(ValueError, match="reason 'ok'"):
        make_event(10, "acc1", "acc2", 1000, 0, 1, None, "not_ok")
    with pytest.raises(ValueError, match="cannot have reason 'ok'"):
        make_event(10, "acc1", "acc2", 1000, 0, 1, "customer", "ok")
    with pytest.raises(ValueError, match="positive integer"):
        make_event(10, "acc1", "acc2", 0, 0, 1, None, "ok")
    with pytest.raises(ValueError, match="non-negative integer"):
        make_event(-5, "acc1", "acc2", 1000, 0, 1, None, "ok")
    with pytest.raises(ValueError, match="non-negative integer"):
        make_event("10", "acc1", "acc2", 1000, 0, 1, None, "ok")  # type: ignore[arg-type]


def test_d8_failure_domains_exact_list() -> None:
    """
    WHAT: Validates that FAILURE_DOMAINS matches the exact specification list.
    WHY: Prevents domain additions or spelling drift across the failure subsystem.
    """
    expected = ["npci_central", "sender_bank", "receiver_bank", "app_or_network", "customer"]
    assert FAILURE_DOMAINS == expected
    settings = FailureSettings(0.1, 0.1, 0.1)
    rng = np.random.default_rng(606)
    for _ in range(500):
        res = decide_failure(1, 0, 1, [], settings, rng)
        if res is not None:
            assert res[0] in expected


def test_d9_determinism_and_input_refusals() -> None:
    """
    WHAT: Tests decision reproducibility across seeds and processes, plus input refusals.
    WHY: Proves strict seed determinism and parameter boundary defense for failure decisions.
    """
    inc = Incident("npci_incident", 100, 50, None, 0.7)
    settings = FailureSettings(0.01, 0.01, 0.01)
    d0 = [decide_failure(120, 0, 1, [inc], settings, np.random.default_rng(0)) for _ in range(2000)]
    d0_dup = [decide_failure(120, 0, 1, [inc], settings, np.random.default_rng(0)) for _ in range(2000)]
    assert d0 == d0_dup

    # Refusals for bad minute and bad bank index.
    rng = np.random.default_rng(42)
    for bad_m in (-1, "10", 2.5):
        with pytest.raises((ValueError, TypeError)):
            decide_failure(bad_m, 0, 1, [], settings, rng)  # type: ignore[arg-type]
    for bad_b in (-1, "0", 1.5):
        with pytest.raises((ValueError, TypeError)):
            decide_failure(10, bad_b, 1, [], settings, rng)  # type: ignore[arg-type]

    # Cross-process determinism.
    script = (
        "import hashlib, numpy as np; from world.failures import FailureSettings; "
        "from world.incidents import Incident; from world.failure_decision import decide_failure; "
        "inc=Incident('npci_incident', 100, 50, None, 0.7); s=FailureSettings(0.01, 0.01, 0.01); "
        "res=[decide_failure(120, 0, 1, [inc], s, np.random.default_rng(0)) for _ in range(2000)]; "
        "print(hashlib.sha256(repr(res).encode()).hexdigest())"
    )
    res = subprocess.run(["python", "-B", "-c", script], capture_output=True, text=True, check=True)
    assert res.stdout.strip() == hashlib.sha256(repr(d0).encode()).hexdigest()
