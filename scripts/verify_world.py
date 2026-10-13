"""
File: scripts/verify_world.py
Purpose: Run every golden simulation check (FP, G1, G2, G3) and issue register in one command.
"""

import argparse
import io
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import numpy as np

# Make repo imports work the same way scripts/check.py does:
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.config_validator import load_yaml  # noqa: E402
from core.issue_register import main as issue_register_main  # noqa: E402
from world.background_config import read_background_realism  # noqa: E402
from world.build_world import build_world, world_fingerprint  # noqa: E402
from world.calendar_runner import run_world  # noqa: E402
from world.device_schedule import draw_device_event_schedule  # noqa: E402
from world.routine import run_one_month  # noqa: E402
from world.run_hashes import (  # noqa: E402
    hash_device_stamps,
    hash_ledger_entries,
    hash_payment_events,
    hash_people_devices,
)
from world.world_plan import draw_world_plan, spawn_life_streams, spawn_world_streams  # noqa: E402


def check_fp(golden_cfg: dict[str, Any]) -> tuple[bool, str, str]:
    """
    WHAT: Re-computes world fingerprint for seed 7 and compares against golden baseline.
    WHY: Verifies Level 11A bit-for-bit entity initialization determinism.
    Args: golden_cfg: Dictionary loaded from golden_values.yaml.
    Returns: Tuple of (ok_flag, expected_fingerprint, actual_fingerprint).
    """
    # 1. Build world with standard Level 11A seed 7 configuration.
    world = build_world(seed=7, n_people=300, n_shops=40)
    actual = world_fingerprint(world)
    expected = str(golden_cfg["fp"]["expected"]["fingerprint"])
    # 2. Check equality of actual fingerprint against expected baseline.
    return (actual == expected, expected, actual)


def check_g1(golden_cfg: dict[str, Any]) -> tuple[bool, str, str]:
    """
    WHAT: Runs Level 11A one-month routine and verifies ledger entries hash and counts.
    WHY: Verifies monthly routine execution and double-entry bookkeeping journals.
    Args: golden_cfg: Dictionary loaded from golden_values.yaml.
    Returns: Tuple of (ok_flag, expected_summary, actual_summary).
    """
    # 1. Build world and run monthly routine cycle with fixed generator seed 11.
    world = build_world(seed=7, n_people=300, n_shops=40)
    counts = run_one_month(world, np.random.default_rng(11))
    actual_hash = hash_ledger_entries(world.ledgers)
    # 2. Compare calculated ledger entries hash and transaction counts to golden baseline.
    exp_h = golden_cfg["g1"]["expected"]["ledger_hash"]
    exp_c = golden_cfg["g1"]["expected"]["counts"]
    ok = (actual_hash == exp_h) and (dict(sorted(counts.items())) == exp_c)
    exp_repr = f"hash={exp_h[:12]}..., counts={exp_c}"
    act_repr = f"hash={actual_hash[:12]}..., counts={dict(sorted(counts.items()))}"
    return (ok, exp_repr, act_repr)


def check_g2(golden_cfg: dict[str, Any]) -> tuple[bool, str, str]:
    """
    WHAT: Runs 61-day calendar simulation without device events and checks G2 hash and counts.
    WHY: Verifies calendar runner transaction dispatch and failure breakdown against PR #47 baseline.
    Args: golden_cfg: Dictionary loaded from golden_values.yaml.
    Returns: Tuple of (ok_flag, expected_summary, actual_summary).
    """
    # 1. Execute calendar simulation without device schedule.
    world = build_world(seed=101, n_people=60, n_shops=12)
    p_rng, r_rng = spawn_world_streams(101)
    res = run_world(world, draw_world_plan(p_rng, n_days=61), r_rng)
    # 2. Verify payment event hash and transaction domain failure breakdown counts.
    act_h, act_c = hash_payment_events(res.events), res.counts
    exp_h = golden_cfg["g2"]["expected"]["event_hash"]
    exp_c = golden_cfg["g2"]["expected"]["counts"]
    ok = (act_h == exp_h) and (act_c == exp_c)
    return (ok, f"hash={exp_h[:12]}..., counts={exp_c}", f"hash={act_h[:12]}..., counts={act_c}")


def check_g3(golden_cfg: dict[str, Any]) -> tuple[bool, str, str]:
    """
    WHAT: Runs 61-day calendar simulation with device schedule and validates G3 baselines.
    WHY: Verifies device lifecycle events, hardware stamps, and people device state against PR #49.
    Args: golden_cfg: Dictionary loaded from golden_values.yaml.
    Returns: Tuple of (ok_flag, expected_summary, actual_summary).
    """
    # 1. Build world, plan, and device schedule from device realism configuration.
    world = build_world(seed=101, n_people=60, n_shops=12)
    p_rng, r_rng = spawn_world_streams(101)
    plan = draw_world_plan(p_rng, n_days=61)
    dev_rng, _ = spawn_life_streams(101)
    bg = read_background_realism(REPO_ROOT / "config" / "device_realism.yaml")
    sched = draw_device_event_schedule(bg, dev_rng, world.people, plan.festivals, plan.n_days)
    res = run_world(world, plan, r_rng, device_events=sched)
    # 2. Compute telemetry stamps hash and people devices hash.
    act_s, act_d, act_p = (
        hash_device_stamps(res.event_sender_devices),
        res.device_counts,
        hash_people_devices(world.people),
    )
    exp = golden_cfg["g3"]["expected"]
    ok = (act_s == exp["stamp_hash"]) and (act_d == exp["device_counts"]) and (act_p == exp["people_hash"])
    return (
        ok,
        f"stamps={exp['stamp_hash'][:12]}..., dev_counts={exp['device_counts']}, people={exp['people_hash'][:12]}...",
        f"stamps={act_s[:12]}..., dev_counts={act_d}, people={act_p[:12]}...",
    )


def check_issues(golden_cfg: dict[str, Any]) -> tuple[bool, str, str]:
    """
    WHAT: Invokes core.issue_register.main() in-process to verify all 31 research issues.
    WHY: Validates research issue registration and status invariants without running subprocess.
    Args: golden_cfg: Dictionary loaded from golden_values.yaml.
    Returns: Tuple of (ok_flag, expected_summary, actual_summary).
    """
    # 1. Capture stdout from core.issue_register.main execution.
    buf = io.StringIO()
    with redirect_stdout(buf):
        exit_code = issue_register_main([])
    # 2. Assert exit code is 0 and output contains expected summary text.
    output = buf.getvalue().strip()
    expected = golden_cfg["issues"]["expected"]["summary"]
    ok = (exit_code == 0) and (expected in output)
    return (ok, expected, output)


def run_fast_suite() -> tuple[bool, str]:
    """
    WHAT: Runs pytest fast test suite via subprocess and captures summary line.
    WHY: Verifies the entire fast test suite passes without errors or failures when --suites is set.
    Returns: Tuple of (passed_flag, last_summary_line).
    """
    # 1. Construct command invoking fast pytest test suite matching CI and check.py.
    cmd = [
        sys.executable,
        "-B",
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "-m",
        "not needs_stack and not gpu and not ml and not needs_dvc",
        "tests/unit",
        "tests/contract",
        "tests/property",
        "tests/schema",
        "tests/baselines",
    ]
    # 2. Execute pytest command, capture output, and extract last summary line.
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    summary = lines[-1] if lines else "no summary output"
    # 3. Check for failed or error indicators in exit code and summary text.
    ok = (proc.returncode == 0) and ("failed" not in summary) and ("error" not in summary.lower())
    return ok, summary


def run_all_golden_checks(golden_path: Path) -> list[tuple[str, bool, str, str]]:
    """
    WHAT: Loads golden configuration and executes FP, G1, G2, G3, and issues checks.
    WHY: Aggregates verification results across all five golden specification areas.
    Args: golden_path: Filesystem path to golden_values.yaml.
    Returns: List of (check_name, passed_boolean, expected_string, actual_string).
    """
    # 1. Load configuration catalog using core YAML loader.
    cfg = load_yaml(golden_path)
    # 2. Execute each golden verification check sequentially.
    r_fp = check_fp(cfg)
    r_g1 = check_g1(cfg)
    r_g2 = check_g2(cfg)
    r_g3 = check_g3(cfg)
    r_iss = check_issues(cfg)
    return [
        ("FP", r_fp[0], r_fp[1], r_fp[2]),
        ("G1", r_g1[0], r_g1[1], r_g1[2]),
        ("G2", r_g2[0], r_g2[1], r_g2[2]),
        ("G3", r_g3[0], r_g3[1], r_g3[2]),
        ("ISSUES", r_iss[0], r_iss[1], r_iss[2]),
    ]


def main(argv: list[str] | None = None) -> int:
    """
    WHAT: CLI entry point orchestrating golden checks and optional test suite execution.
    WHY: Provides single command exiting 0 on success and 1 on any failure.
    Args: argv: Optional CLI argument strings list.
    Returns: Integer exit code 0 on success or 1 on failure.
    """
    parser = argparse.ArgumentParser(description="Verify world golden values.")
    parser.add_argument("--golden-file", type=Path, default=REPO_ROOT / "scripts" / "golden_values.yaml")
    parser.add_argument("--suites", action="store_true", help="Run fast test suite.")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    # 1. Run golden checks and print PASS / FAIL per item.
    results = run_all_golden_checks(args.golden_file)
    fails = sum(not ok for _, ok, _, _ in results)
    for name, ok, exp, act in results:
        print(f"PASS {name}" if ok else f"FAIL {name}: expected {exp}, got {act}")
    # 2. If --suites requested, execute fast test suite and report summary.
    if args.suites:
        suites_ok, summary = run_fast_suite()
        print(f"SUITES: {summary}")
        fails += int(not suites_ok)
    # 3. Print final status line and return exit code.
    print("ALL GOLDEN CHECKS PASSED" if fails == 0 else f"{fails} CHECK(S) FAILED")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
