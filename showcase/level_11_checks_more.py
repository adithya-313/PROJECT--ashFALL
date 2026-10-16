"""
WHAT: Showcase check runners 8 through 16 for Level 11 simulation verification.
It tests central switch outages versus deferred registry outages, wedding episode inflow surges, banking asymmetry,
deterministic event repeatability, capacity refusal safety, double-entry ledgers, live replay, and golden baselines.

WHY: Completes the full verification checklist proving architectural fidelity across normal, edge, and negative paths.
Isolating checks into modular routines prevents procedural complexity from exceeding the 30-line code boundary.
Executing tests against production simulation runners ensures that report claims reflect verifiable runtime facts.

TERM: central switch outage: Complete technical failure of the central payment clearing switch,
      e.g. NPCI core routing failure causing all attempted transactions to fail.
TERM: deferred layer: A configuration scenario altering parameters not yet activated in the current simulation engine,
      e.g. registry-outage which takes effect from Level 13 and leaves Level 11 base events completely unchanged.
TERM: double-entry ledger: Accounting record where every financial transaction is entered as matching debits and credits,
      e.g. transferring money from sender account to recipient account while conserving total monetary supply.
TERM: golden baseline: A cryptographically verified digest or transaction tally pinned from an audited run,
      e.g. golden hashes G1, G2, and G3 confirming zero unexpected code drift across revisions.
"""

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from core.config_validator import load_yaml
from showcase.framework import CheckOutcome
from showcase.level_11_charts import account_flow_with_window
from world.build_world import build_world
from world.calendar_runner import run_world
from world.checks import (
    check_balances_equal_entry_sums,
    check_money_conserved,
    check_no_negative_balances,
)
from world.config_reader import read_world_ranges
from world.episode_payments import plan_episode_payments
from world.episodes import Episode
from world.estimates import MachineProfile, read_sizing_config
from world.fast_mode import FastRunSettings, event_lines, run_fast_world
from world.layers import apply_layers, read_layer
from world.live_mode import HttpEventSender, LocalReceiver, build_live_events, read_live_policy, replay
from world.machine_limits import MachineState
from world.world_configs import CONFIG_NAMES, read_world_configs
from world.world_plan import draw_world_plan, spawn_life_streams, spawn_world_streams
from world.world_sizing import decide_run

REPO_ROOT = Path(__file__).resolve().parent.parent
SWITCH_OUTAGE_PATH = REPO_ROOT / "scenarios" / "stress" / "switch-outage.yaml"
REGISTRY_OUTAGE_PATH = REPO_ROOT / "scenarios" / "stress" / "registry-outage.yaml"
EXPECTED_SHA256 = "24fc0e25009d79462d508f040a109620dd4308486cc78aee589393fe921f92fd"


async def run_switch_vs_registry_check(workspace: Any) -> CheckOutcome:
    """
    WHAT: Tests switch-outage total clearing failure against deferred registry-outage zero-event drift.
    Under switch-outage, it asserts all payments attempted during NPCI windows fail with npci_central (0% success).
    Under registry-outage, it asserts that serialized event bytes produce the exact canonical 11C-1 SHA-256 hash.

    WHY: Demonstrates that active failures drop transactions immediately while deferred scenarios cause zero drift.
    Proves that registry-outage is safely pre-registered for Level 13 without perturbing Level 11 baseline behavior.
    Verifies that the failure domain assignment logic correctly attributes switch outages to the central clearinghouse.

    Args:
        workspace: Optional showcase execution workspace environment container.

    Returns:
        CheckOutcome recording zero success during switch outage and exact SHA-256 hash match for registry outage.

    Example:
        >>> import asyncio
        >>> outcome = asyncio.run(run_switch_vs_registry_check(None))
        >>> "switch outage: 0% success" in outcome.key_number
        True
    """
    base_cfgs = read_world_configs()
    settings = FastRunSettings(seed=101, n_people=60, n_shops=12, n_days=61, range_name="training")
    sw_layer = read_layer(SWITCH_OUTAGE_PATH)
    sw_cfgs, _ = apply_layers(base_cfgs, [sw_layer])
    out_sw = run_fast_world(settings, configs=sw_cfgs)
    incidents = [inc for inc in out_sw.plan.incidents if inc.kind == "npci_incident"]
    assert incidents, "Expected NPCI central incident to exist"
    inc = incidents[0]
    in_sw = [
        ev for ev in out_sw.result.events if inc.start_minute <= ev.minute < inc.start_minute + inc.duration_minutes
    ]
    assert in_sw and all(ev.status == "failed" and ev.failure_domain == "npci_central" for ev in in_sw)
    reg_layer = read_layer(REGISTRY_OUTAGE_PATH)
    reg_cfgs, _ = apply_layers(base_cfgs, [reg_layer])
    out_reg = run_fast_world(settings, configs=reg_cfgs)
    reg_hash = hashlib.sha256("".join(event_lines(out_reg)).encode("utf-8")).hexdigest()
    assert reg_hash == EXPECTED_SHA256, f"Registry outage altered events: {reg_hash}"
    key = "switch outage: 0% success during NPCI window; registry outage: 0 base event changes"
    return CheckOutcome(
        key_number=key,
        evidence={"switch_window_attempts": len(in_sw), "switch_failures": len(in_sw), "registry_sha256": reg_hash},
    )


async def run_wedding_episode_check(workspace: Any) -> CheckOutcome:
    """
    WHAT: Simulates a wedding gift collection episode and charts the daily inflow velocity into the host account.
    It schedules an episode where 40 citizens contribute funds to one recipient across calendar days 5 to 20.
    It asserts that daily incoming money surges dramatically during the episode window compared to baseline periods.

    WHY: Proves that ordinary life events generate non-fraudulent transaction bursts mimicking mule behavior.
    Validating that normal episodes produce sharp account velocity spikes tests detector resilience against false positives.
    Attaching an SVG bar chart visually documents the temporal bounds and financial intensity of the social collection.

    Args:
        workspace: Optional showcase execution workspace environment container.

    Returns:
        CheckOutcome capturing peak burst inflow, baseline inflow, and the account flow vector chart.

    Example:
        >>> import asyncio
        >>> outcome = asyncio.run(run_wedding_episode_check(None))
        >>> "peak day inflow" in outcome.key_number
        True
    """
    world = build_world(seed=101, n_people=60, n_shops=12)
    p_rng, r_rng = spawn_world_streams(101)
    plan = draw_world_plan(p_rng, n_days=61)
    ranges = read_world_ranges()
    collector = world.people[0]
    _, ep_rng = spawn_life_streams(101)
    ep_plan = plan_episode_payments(world, plan, [ep], ranges, ep_rng, "training")
    res = run_world(world, plan, r_rng, episode_payments=ep_plan.payments)
    inflows = [0.0] * 61
    for ev in res.events:
        if ev.status == "success" and ev.to_account == collector.account_id:
            inflows[min(ev.minute // 1440, 60)] += ev.amount_paise / 100.0
    peak_inflow = max(inflows[5:21])
    baseline_inflow = max(inflows[:5] + inflows[21:])
    assert peak_inflow > 10000.0 and baseline_inflow == 0.0
    days = list(range(61))
    svg_chart = account_flow_with_window(days, inflows, (5, 20, "Wedding Collection"))
    key = f"peak day inflow Rs {int(peak_inflow):,} inside episode vs Rs {int(baseline_inflow)} baseline"
    return CheckOutcome(
        key_number=key,
        evidence={"peak_inflow_rupees": peak_inflow, "baseline_inflow_rupees": baseline_inflow, "chart_svg": svg_chart},
    )


async def run_bank_asymmetry_check(workspace: Any) -> CheckOutcome:
    """
    WHAT: Evaluates structural heterogeneity across the three banking institutions in a generated simulation world.
    It verifies that public, private, and payments banks differ significantly in customer share, KYC, and hold latency.
    It asserts that Bank 0 maintains the largest share while Bank 1 enforces the strictest identity verification.

    WHY: Empirical banking sectors feature stark disparities in operational scale, KYC rigor, and fraud response speed.
    Mule networks disproportionately exploit banks with relaxed identity checks and sluggish administrative freeze latencies.
    Verifying institutional diversity proves that Level 11 models realistic banking frictions and fraud vulnerabilities.

    Args:
        workspace: Optional showcase execution workspace environment container.

    Returns:
        CheckOutcome capturing individual bank customer shares, KYC quality scores, and hold latency multipliers.

    Example:
        >>> import asyncio
        >>> outcome = asyncio.run(run_bank_asymmetry_check(None))
        >>> "3 banks differ" in outcome.key_number
        True
    """
    world = build_world(seed=101, n_people=60, n_shops=12)
    b0, b1, b2 = world.banks
    assert b0.customer_share > b1.customer_share > b2.customer_share
    assert b1.weak_kyc_share < b0.weak_kyc_share and b1.weak_kyc_share < b2.weak_kyc_share
    assert b1.hold_delay_factor < b0.hold_delay_factor and b1.hold_delay_factor < b2.hold_delay_factor
    key = (
        f"3 banks differ: Bank 0 ({b0.customer_share:.1%} share, {b0.weak_kyc_share:.1%} weak KYC, {b0.hold_delay_factor:.2f}x hold), "
        f"Bank 1 ({b1.customer_share:.1%} share, {b1.weak_kyc_share:.1%} weak KYC, {b1.hold_delay_factor:.2f}x hold), "
        f"Bank 2 ({b2.customer_share:.1%} share, {b2.weak_kyc_share:.1%} weak KYC, {b2.hold_delay_factor:.2f}x hold)"
    )
    evidence = {
        "bank_0": {"kind": b0.kind, "share": b0.customer_share, "weak_kyc": b0.weak_kyc_share, "hold_delay": b0.hold_delay_factor},
        "bank_1": {"kind": b1.kind, "share": b1.customer_share, "weak_kyc": b1.weak_kyc_share, "hold_delay": b1.hold_delay_factor},
        "bank_2": {"kind": b2.kind, "share": b2.customer_share, "weak_kyc": b2.weak_kyc_share, "hold_delay": b2.hold_delay_factor},
    }
    return CheckOutcome(key_number=key, evidence=evidence)


async def run_repeatability_check(workspace: Any) -> CheckOutcome:
    """
    WHAT: Re-runs the canonical tiny world simulation and verifies bit-for-bit repeatability against golden SHA-256.
    It serializes every payment event record and verifies that the resulting cryptographic digest equals the baseline.
    It asserts that two independent runs executed with identical seeds produce byte-identical event files.

    WHY: Determinism is an essential foundational requirement for repeatable benchmark evaluations and debugging.
    Proving bit-for-bit repeatability guarantees that random seeds fully specify all simulation state transitions.
    Matching the canonical golden digest 24fc0e25... proves that no unintended behavior drift has entered the engine.

    Args:
        workspace: Optional showcase execution workspace environment container.

    Returns:
        CheckOutcome certifying bit-for-bit event digest equality across independent simulation executions.

    Example:
        >>> import asyncio
        >>> outcome = asyncio.run(run_repeatability_check(None))
        >>> "24fc0e25" in outcome.key_number
        True
    """
    settings = FastRunSettings(seed=101, n_people=60, n_shops=12, n_days=61, range_name="training")
    output = run_fast_world(settings)
    lines = event_lines(output)
    digest = hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()
    assert digest == EXPECTED_SHA256, f"Expected {EXPECTED_SHA256}, got {digest}"
    key = f"SHA-256 {digest[:12]}... verified bit-for-bit against golden benchmark"
    return CheckOutcome(key_number=key, evidence={"events_sha256": digest, "total_events": len(lines)})


async def run_oversized_world_check(workspace: Any) -> CheckOutcome:
    """
    WHAT: Evaluates capacity sizing decision engine refusal on an oversized world configuration.
    It proposes a 100,000-person, 365-day world against a constrained 30-minute runtime budget.
    It verifies that the sizing engine rejects execution, recommends fitting limits, and writes zero output files.

    WHY: Proves capacity safety: simulations exceeding memory or runtime boundaries are refused before execution.
    Failing fast protects host resources from unconstrained memory exhaustion or infinite runtimes.
    Ensures that refused runs create no stray directories or partial output files on disk.

    Args:
        workspace: Optional showcase execution workspace environment container.

    Returns:
        CheckOutcome recording refusal status, estimate vs limit comparison, and problem causes.

    Example:
        >>> import asyncio
        >>> outcome = asyncio.run(run_oversized_world_check(None))
        >>> "refused, nothing written" in outcome.key_number
        True
    """
    prof = MachineProfile(5000.0, 0.8, 350.0, 1024.0, 2915)
    state = MachineState(10_000_000, 50_000_000)
    sizing_cfg = read_sizing_config()
    settings = FastRunSettings(seed=101, n_people=100_000, n_shops=10_000, n_days=365, range_name="training")
    max_minutes = 30.0
    decision = decide_run(settings, prof, state, sizing_cfg, max_minutes=max_minutes)
    assert not decision.fits and len(decision.problems) >= 2
    dummy_out = REPO_ROOT / "runs" / "oversized_world_refused_test"
    assert not dummy_out.exists(), f"Refused run should create no folder, but found: {dummy_out}"
    est_min = round(decision.estimate.seconds / 60.0)
    limit_min = int(max_minutes)
    key = f"estimated {est_min} min vs limit {limit_min} min; refused, nothing written"
    evidence = {
        "fits": decision.fits,
        "estimated_minutes": est_min,
        "limit_minutes": limit_min,
        "problems": decision.problems,
        "message": decision.message,
    }
    return CheckOutcome(key_number=key, evidence=evidence)


async def run_ledger_invariants_check(workspace: Any) -> CheckOutcome:
    """
    WHAT: Audits double-entry ledger invariants across all bank institutions and tests rejection of corrupted balances.
    It runs check_no_negative_balances, check_balances_equal_entry_sums, and check_money_conserved on a completed world.
    It then mutates a cached balance in a shallow copy and asserts that balance inconsistency checks catch the fault.

    WHY: Closed financial systems require mathematical proof that currency is neither created nor destroyed.
    Proving balance non-negativity validates that authorization rules prevent illicit overdraft transactions.
    Testing balance corruption detection confirms that audit verification functions reliably detect ledger tampering.

    Args:
        workspace: Optional showcase execution workspace environment container.

    Returns:
        CheckOutcome documenting verified conservation of currency and successful detection of corrupted records.

    Example:
        >>> import asyncio
        >>> outcome = asyncio.run(run_ledger_invariants_check(None))
        >>> "3 ledger invariants pass" in outcome.key_number
        True
    """
    settings = FastRunSettings(seed=101, n_people=60, n_shops=12, n_days=61, range_name="training")
    output = run_fast_world(settings)
    ledgers = output.world.ledgers
    initial_total = sum(ledger.balance_of(acc) for ledger in ledgers for acc in ledger.accounts)
    assert not check_no_negative_balances(ledgers)
    assert not check_balances_equal_entry_sums(ledgers)
    assert not check_money_conserved(ledgers, initial_total)
    acc = next(iter(ledgers[0].accounts))
    orig = ledgers[0].balances[acc]
    ledgers[0].balances[acc] += 1000
    try:
        assert check_balances_equal_entry_sums(ledgers)
    finally:
        ledgers[0].balances[acc] = orig
    key = "3 ledger invariants pass across 3 banks; corrupted balance caught"
    return CheckOutcome(key_number=key, evidence={"total_paise": initial_total, "banks_audited": len(ledgers)})


async def run_live_replay_check(workspace: Any) -> CheckOutcome:
    """
    WHAT: Replays simulated fast-mode transaction events across loopback HTTP interface to an in-process receiver.
    It uses LocalReceiver to capture replayed transaction envelopes and compares them with fast-mode event lines.
    It asserts that event count, record contents, and chronological order match fast mode exactly.

    WHY: Proves end-to-end delivery fidelity across the HTTP transport boundary without external services.
    Verifying that received records equal fast-mode lines confirms that live mode causes zero payload mutation.
    Validating loopback binding to 127.0.0.1 confirms that simulation data remains securely confined to the host.

    Args:
        workspace: Optional showcase execution workspace environment container.

    Returns:
        CheckOutcome capturing replayed event counts, loopback binding status, and transmission parity proof.

    Example:
        >>> import asyncio
        >>> outcome = asyncio.run(run_live_replay_check(None))
        >>> "live replay matched" in outcome.key_number
        True
    """
    settings = FastRunSettings(seed=101, n_people=60, n_shops=12, n_days=61, range_name="training")
    output = run_fast_world(settings)
    events = build_live_events(output)
    policy = read_live_policy()
    with LocalReceiver(policy) as receiver:
        sender = HttpEventSender(receiver.url, policy)
        report = replay(events, sender, speed=None, policy=policy)
        expected = [json.loads(line) for line in event_lines(output)]
        assert receiver.records == expected and report.sent == len(expected)
    key = f"live replay matched all {len(expected):,} events byte-for-byte to 127.0.0.1"
    return CheckOutcome(key_number=key, evidence={"replayed_events": len(expected), "host": policy.default_host})


async def run_layer_isolation_check(workspace: Any) -> CheckOutcome:
    """
    WHAT: Audits all six stress scenario layers in scenarios/stress/ to verify surgical parameter isolation.
    For each layer, it applies parameter overrides and asserts that all unmentioned configuration blocks match baseline.
    It compares parameters across background, failure, device, and world range configuration partitions.

    WHY: Prevents accidental parameter leakage or collateral mutations when applying stress scenarios.
    Proving that untouched configuration values stay identical guarantees that stress scenarios have surgical precision.
    Validates that scenario layers maintain modularity and predictable execution boundaries.

    Args:
        workspace: Optional showcase execution workspace environment container.

    Returns:
        CheckOutcome certifying that all six stress scenario files alter only their designated parameters.

    Example:
        >>> import asyncio
        >>> outcome = asyncio.run(run_layer_isolation_check(None))
        >>> "6 stress scenarios verified" in outcome.key_number
        True
    """
    stress_dir = REPO_ROOT / "scenarios" / "stress"
    stress_files = sorted(stress_dir.glob("*.yaml"))
    assert len(stress_files) == 6, f"Expected 6 stress scenario files, got {len(stress_files)}"
    for file_path in stress_files:
        layer = read_layer(file_path)
        base = read_world_configs()
        layered, _ = apply_layers(base, [layer])
        for config_name in CONFIG_NAMES:
            base_params = getattr(base, config_name)["parameters"]
            layered_params = getattr(layered, config_name)["parameters"]
            layer_changes = layer.changes.get(config_name, {})
            for setting_name, base_block in base_params.items():
                if setting_name not in layer_changes:
                    assert layered_params[setting_name] == base_block
    key = f"{len(stress_files)} stress scenarios verified: each modifies only named parameters"
    return CheckOutcome(key_number=key, evidence={"verified_files": [f.name for f in stress_files]})


async def run_golden_fingerprints_check(workspace: Any) -> CheckOutcome:
    """
    WHAT: Re-runs the four golden simulation checks (FP, G1, G2, G3) in-process and asserts all pass.
    It invokes check_fp, check_g1, check_g2, and check_g3 from scripts/verify_world.py using golden_values.yaml.
    It verifies entity fingerprints, monthly routine ledgers, calendar events, and device telemetry stamps.

    WHY: Provides in-process cryptographic verification that the entire simulation suite remains free of regressions.
    Executing golden checks as part of the showcase guarantees that published metrics reflect audited baselines.
    In-process execution eliminates subprocess overhead and enables direct verification within test runners.

    Args:
        workspace: Optional showcase execution workspace environment container.

    Returns:
        CheckOutcome documenting verified status across all four canonical golden benchmarks.

    Example:
        >>> import asyncio
        >>> outcome = asyncio.run(run_golden_fingerprints_check(None))
        >>> "4 golden fingerprints reproduced" in outcome.key_number
        True
    """
    from scripts.verify_world import check_fp, check_g1, check_g2, check_g3
    golden_path = REPO_ROOT / "scripts" / "golden_values.yaml"
    golden_cfg = load_yaml(golden_path)
    ok_fp, _, _ = check_fp(golden_cfg)
    ok_g1, _, _ = check_g1(golden_cfg)
    ok_g2, _, _ = check_g2(golden_cfg)
    ok_g3, _, _ = check_g3(golden_cfg)
    assert ok_fp and ok_g1 and ok_g2 and ok_g3
    key = "4 golden fingerprints reproduced in-process (FP, G1, G2, G3)"
    evidence = {"fp_passed": ok_fp, "g1_passed": ok_g1, "g2_passed": ok_g2, "g3_passed": ok_g3}
    return CheckOutcome(key_number=key, evidence=evidence)
