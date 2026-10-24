"""
WHAT: Measures sampled worlds and checks them. For each world it writes one CSV row: how long the world took, how much
of that was gang planning (per gang and in all), how many events, labels, mules and victims it yielded, how much of
each gang's budget was used, how many invariant problems world/world_invariants.py found, and a hash of its event
lines. It can run one block, run a fixed sweep across splits and gang mixes, use worker processes, and check that
parallel worlds are identical to serial ones. A world that crashes is recorded and the run continues.

WHY: The number of worlds a later amendment commits to should come from measured facts, and a sweep is the cheapest
way to find a world that breaks a rule. This script measures and decides nothing. It reads the world's in-memory truth
only to tell true complaints from false ones, as the Level 12e showcase does, and it never loads the stored truth files.

TERM: block: the block_size worlds that sample_block plans for one split and block number.
TERM: true complaint: a complaint that names at least one victim payment of a gang; false complaints name none.
TERM: sweep: the fixed list of block runs in SWEEP_PLAN, each reported as worlds run, crashed and with problems.
TERM: budget used share: a gang's planned total divided by its budget limit, for money and for time.
"""

import argparse
import concurrent.futures
import csv
import hashlib
import os
import platform
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from world.fast_mode import event_lines  # noqa: E402
from world.gang_mix import generate_gang_mix, load_gang_mix_config  # noqa: E402
from world.gang_planner import plan_gang  # noqa: E402
from world.interactive_records import FraudInputs, RunnerSettings  # noqa: E402
from world.interactive_runner import interactive_fast_output, prepare_world, run_interactive_world  # noqa: E402
from world.label_builder import build_label_events, build_registry_reports  # noqa: E402
from world.opening_handles import replay_openings  # noqa: E402
from world.sampled_world import fast_settings_for, pinned_configs, run_sampled_world  # noqa: E402
from world.truth_records import SCAM_STEP_VICTIM_PAYMENT  # noqa: E402
from world.world_configs import read_world_configs  # noqa: E402
from world.world_invariants import check_atm_daily_limits, check_world  # noqa: E402
from world.world_sampler import WorldSpec, load_sampler_config, sample_block  # noqa: E402

CODE_VERSION: str = "measure-world-block"
COLUMNS: tuple[str, ...] = (
    "world_id", "seed", "gang_count", "seconds_total", "seconds_planning_all_gangs", "seconds_planning_per_gang",
    "events", "payments_success", "true_complaints", "false_complaints", "registry_reports", "mule_accounts",
    "victims", "budget_money_used_share_max", "budget_time_used_share_max", "invariant_problems",
    "atm_checked", "atm_skipped", "liens_injected", "liens_on_gang_opened", "gang_observed_lien_count",
    "payments_blocked_funds_on_hold", "lien_note", "event_lines_sha256", "run_label",
)
NUMERIC_COLUMNS: tuple[str, ...] = (
    "gang_count", "seconds_total", "seconds_planning_all_gangs", "events", "payments_success", "true_complaints",
    "false_complaints", "registry_reports", "mule_accounts", "victims", "budget_money_used_share_max",
    "budget_time_used_share_max", "invariant_problems", "atm_checked", "atm_skipped", "liens_injected",
    "liens_on_gang_opened", "gang_observed_lien_count", "payments_blocked_funds_on_hold",
)
# The fixed sweep: (row name, label, split, block, limit or None for the whole block, mode). Modes: "plain" is
# run_sampled_world; "market" adds account market and gang reactions; "lien" adds both plus two injected liens.
SWEEP_PLAN: tuple[tuple[str, str, str, int, int | None, str], ...] = (
    ("training", "training block 0", "training", 0, None, "plain"),
    ("validation", "validation block 0 (10)", "validation", 0, 10, "plain"),
    ("test", "test block 0 (10)", "test", 0, 10, "plain"),
    ("stress", "stress block 0 (10)", "stress", 0, 10, "plain"),
    ("market", "training market+reactions (5)", "training", 0, 5, "market"),
    ("lien", "training lien+reactions (5)", "training", 0, 5, "lien"),
)
TRACEBACK_LINES = 15
# A test injection, like the fixture liens of Level 12d: large enough to block any debit of the account it sits on.
INJECTED_LIEN_PAISE = 500_000_000
Task = tuple[WorldSpec, str, str, str]


def read_arguments(argv: list[str]) -> argparse.Namespace:
    """Parses the command line and refuses an output path inside the repository or a compare without workers."""
    parser = argparse.ArgumentParser(description="Measure and check sampled worlds (ESTIMATE only).")
    parser.add_argument("--split", default="training", help="training, validation, test or stress")
    parser.add_argument("--block", type=int, default=0, help="block number inside the split")
    parser.add_argument("--sampler-seed", type=int, default=1, help="master seed of the sampler")
    parser.add_argument("--preset", default="small", help="size preset of config/world_sizing.yaml")
    parser.add_argument("--limit", type=int, default=None, help="measure only the first N worlds of the block")
    parser.add_argument("--workers", type=int, default=1, help="worker processes; 1 runs everything serially")
    parser.add_argument("--compare", type=int, default=0, help="also run the first K worlds serially and compare")
    parser.add_argument("--sweep", action="store_true", help="run the fixed sweep and ignore --split, --block, --limit")
    parser.add_argument(
        "--sweep-row", default=None, choices=[row[0] for row in SWEEP_PLAN],
        help="run only this named row of the sweep and ignore --split, --block, --limit",
    )
    parser.add_argument("--out", required=True, help="CSV file to write; must be outside the repository")
    args = parser.parse_args(argv)
    out_path = Path(args.out).resolve()
    if REPO_ROOT == out_path or REPO_ROOT in out_path.parents:
        parser.error(f"--out {out_path} is inside the repository; results must never be committed by accident")
    if args.compare > 0 and args.workers < 2:
        parser.error("--compare needs --workers of at least 2")
    return args


def _time_planning(
    spec: WorldSpec, preset: str, cfg: dict[str, Any]
) -> tuple[list[float], tuple[Any, ...], list[Any]]:
    """
    Times plan_gang for each generated gang, calling it as the runner does (children 6 and 7 of the seed, one shared
    planning stream, reserved accounts growing). This is a second planning pass done only to measure; the world itself
    is planned again inside the run.
    """
    settings = fast_settings_for(spec, preset)
    configs = pinned_configs(spec, cfg)
    world = prepare_world(settings, configs).world
    shop_accounts = {shop.account_id for shop in world.shops}
    eligible = len([person for person in world.people if person.account_id not in shop_accounts])
    gangs = generate_gang_mix(spec, cfg, load_gang_mix_config(), eligible, configs.world_ranges)
    children = np.random.SeedSequence(spec.seed).spawn(10)
    dev_rng = np.random.default_rng(children[6])
    plan_rng = np.random.default_rng(children[7])
    reserved = {gang.source.account_id for gang in gangs if gang.source is not None}
    seconds: list[float] = []
    plans: list[Any] = []
    next_device_index = 1
    for gang in gangs:
        started = time.perf_counter()
        plan = plan_gang(
            gang, world, plan_rng, reserved_accounts=reserved, gang_devices_rng=dev_rng,
            device_start_index=next_device_index, world_split=spec.split,
        )
        seconds.append(time.perf_counter() - started)
        dev_acquired_count = sum(1 for e in plan.gang_device_events if e.kind == "device_acquired")
        next_device_index += dev_acquired_count
        reserved.update(role.account_id for role in plan.truth.roles)
        plans.append(plan)
    return seconds, gangs, plans


def _budget_shares(plans: list[Any], gangs: tuple[Any, ...]) -> tuple[float, float]:
    """Returns the largest money share and the largest time share of any gang's planned totals over its budget."""
    money_share = 0.0
    time_share = 0.0
    for plan, gang in zip(plans, gangs):
        money_share = max(money_share, plan.totals.money_paise / gang.budget.money_paise)
        time_share = max(time_share, plan.totals.time_minutes / gang.budget.time_minutes)
    return money_share, time_share


def _handle_ids(plans: list[Any], n_people: int, n_shops: int) -> dict[str, str]:
    """
    Gives every account a gang will open its real id before the run, by replaying the planned opening events in the
    order the runner queues them (the gangs' events in gang order, then the run's own state counts). This is the same
    replay the runner does in resolve_opening_handles; no genuine openings exist because account openings are off.
    """
    events = tuple(event for plan in plans for event in plan.outside_events)
    assigned, _minutes = replay_openings(events, n_people, n_shops)
    return assigned


def _atm_limits(plans: list[Any], handle_ids: dict[str, str]) -> dict[str, int]:
    """
    Collects each planned ATM withdrawal's daily card limit by real account id (the last plan wins if two share a
    mule). Plans name gang-opened accounts by handle, so handles are resolved first; otherwise the limit would never
    match the run's withdrawal records and the check would silently skip them.
    """
    limits: dict[str, int] = {}
    for plan in plans:
        for withdrawal in plan.withdrawals:
            if withdrawal.channel == "atm" and withdrawal.card_daily_limit_paise is not None:
                limits[handle_ids.get(withdrawal.account_id, withdrawal.account_id)] = withdrawal.card_daily_limit_paise
    return limits


def _plan_liens(plans: list[Any], handle_ids: dict[str, str]) -> tuple[list[Any], str]:
    """
    Builds the two test liens (like the fixture liens of Level 12d, a test injection and not part of any plan).
    Lien 1 sits on the first route mule of gang_mix_0 at the minute halfway between its first and last planned payment.
    Lien 2 sits on a gang-opened account (other than lien 1's) at the halfway minute of its planned payments. If no gang
    opened a second account, only lien 1 is placed and the note says so; no account is ever faked.
    """
    from world.outside_events import OutsideEvent  # imported here: only the lien row builds events

    def span_of(account: str) -> tuple[int, int] | None:
        """Returns the first and last planned payment minute that touches the account (handles resolved), or None."""
        minutes = [
            payment.minute for plan in plans for payment in plan.payments
            if account in (handle_ids.get(payment.from_account, payment.from_account),
                           handle_ids.get(payment.to_account, payment.to_account))
        ]
        return (min(minutes), max(minutes)) if minutes else None

    liens: list[Any] = []
    notes: list[str] = []
    first_plan = plans[0]
    route_mule = ""
    for payment, story in zip(first_plan.payments, first_plan.truth.payment_stories):
        if story.block == "route":
            route_mule = handle_ids.get(payment.from_account, payment.from_account)
            break
    route_span = span_of(route_mule) if route_mule else None
    if route_span is None:
        notes.append("no route mule in gang_mix_0")
    else:
        liens.append(OutsideEvent(
            (route_span[0] + route_span[1]) // 2, "lien", account_id=route_mule,
            amount_paise=INJECTED_LIEN_PAISE, encumbrance_id="enc-test-1",
        ))
    opened = [real for real in handle_ids.values() if real != route_mule and span_of(real) is not None]
    if not opened:
        notes.append("no gang-opened account")
    else:
        opened_span = span_of(opened[0])
        assert opened_span is not None
        liens.append(OutsideEvent(
            (opened_span[0] + opened_span[1]) // 2, "lien", account_id=opened[0],
            amount_paise=INJECTED_LIEN_PAISE, encumbrance_id="enc-test-2",
        ))
    return liens, "; ".join(notes)


def _count_labels(output: Any, labels: list[Any]) -> tuple[int, int, int]:
    """
    Returns (true complaints, false complaints, registry reports) of a finished run. A complaint is true when it names
    a payment that is a victim payment of a gang, found by comparing payment ids with the run's in-memory truth, the
    same idea as the Level 12e showcase uses.
    """
    result = output.result
    victim_payment_ids: set[str] = set()
    for position, source in enumerate(result.event_sources):
        input_index = result.event_fraud_input_index[position]
        if source != "fraud" or input_index is None:
            continue
        if result.truth.payment_stories[input_index].scam_step == SCAM_STEP_VICTIM_PAYMENT:
            victim_payment_ids.add(result.payment_ids[position])
    true_count = 0
    for label in labels:
        if any(payment_id in victim_payment_ids for payment_id in label.payment_ids):
            true_count += 1
    reports = build_registry_reports(output, labels)
    return true_count, len(labels) - true_count, len(reports)


def _run_world(
    spec: WorldSpec, preset: str, cfg: dict[str, Any], gangs: tuple[Any, ...], mode: str, liens: list[Any]
) -> Any:
    """
    Runs the world. The "plain" mode is run_sampled_world itself. The "market" and "lien" modes switch on account market
    and gang reactions with their default configuration files and pass the same pinned configs, generated gangs and
    world split to the runner directly; "lien" also places the test liens through FraudInputs.outside_events, the path
    Level 12d uses for its fixture lien.
    """
    if mode == "plain":
        return run_sampled_world(spec, preset, None, cfg, CODE_VERSION)
    return run_interactive_world(
        fast_settings_for(spec, preset, cfg),
        RunnerSettings(fraud_enabled=True, account_market_enabled=True, gang_reactions_enabled=True),
        fraud_inputs=FraudInputs(outside_events=tuple(liens)),
        configs=pinned_configs(spec, cfg),
        gang_configs=gangs,
        code_version=CODE_VERSION,
        world_split=spec.split,
    )


def _lien_columns(output: Any, liens: list[Any], handle_ids: dict[str, str], note: str) -> dict[str, Any]:
    """
    Counts what the injected liens did. The runner keeps no record of what a gang observed, so a lien counts as observed
    when the liened account shows up in a cancelled_by_gang record (the gang stopped using it after learning of it).
    Payments blocked are the events that failed with reason funds_on_hold. A lien not found in the encumbrance log is
    noted, so a lien that never took effect cannot pass as a placed one.
    """
    ids = {lien.encumbrance_id for lien in liens}
    placed = {entry[2] for entry in output.result.encumbrance_log if entry[2] in ids}
    cancelled_accounts = {account for record in output.result.cancelled_by_gang for account in record.accounts}
    opened_by_gangs = set(handle_ids.values())
    notes = [note] if note else []
    if len(placed) != len(liens):
        notes.append(f"{len(placed)} of {len(liens)} liens found in the encumbrance log")
    return {
        "liens_injected": len(liens),
        "liens_on_gang_opened": len([lien for lien in liens if lien.account_id in opened_by_gangs]),
        "gang_observed_lien_count": len([lien for lien in liens if lien.account_id in cancelled_accounts]),
        "payments_blocked_funds_on_hold": sum(
            1 for event in output.result.run_result.events if event.reason == "funds_on_hold"
        ),
        "lien_note": "; ".join(notes),
    }


def measure_world(spec: WorldSpec, preset: str, mode: str, label: str) -> dict[str, Any]:
    """
    WHAT: Plans the spec's gangs once to time them, runs the world, counts its events, labels, mules and victims, checks
    the world invariants, and returns one CSV row as a dictionary (with the problem sentences under "_problems").

    WHY: Each world must be measured on its own with no shared state, so serial and parallel runs can be compared.

    Args:
        spec: A planned world from sample_block.
        preset: A size preset of config/world_sizing.yaml.
        mode: "plain", "market" (account market and gang reactions on) or "lien" (those plus two test liens).
        label: The sweep row or block name the world belongs to.

    Returns:
        A dictionary with one value per column of COLUMNS, plus "_problems".
    """
    cfg = load_sampler_config()
    planning_seconds, gangs, plans = _time_planning(spec, preset, cfg)
    money_share, time_share = _budget_shares(plans, gangs)
    settings = fast_settings_for(spec, preset, cfg)
    handle_ids = _handle_ids(plans, settings.n_people, settings.n_shops)
    liens, lien_note = _plan_liens(plans, handle_ids) if mode == "lien" else ([], "")
    started = time.perf_counter()
    output = _run_world(spec, preset, cfg, gangs, mode, liens)
    total_seconds = time.perf_counter() - started
    events = output.result.run_result.events
    labels = build_label_events(output)
    true_count, false_count, report_count = _count_labels(output, labels)
    atm_limits = _atm_limits(plans, handle_ids)
    problems = check_world(output, labels, atm_limits)
    atm = check_atm_daily_limits(output, atm_limits)
    lien_columns = _lien_columns(output, liens, handle_ids, lien_note)
    mules = {role.account_id for role in output.result.truth.roles if role.role == "mule"}
    victims: set[str] = set()
    for position, source in enumerate(output.result.event_sources):
        input_index = output.result.event_fraud_input_index[position]
        if source == "fraud" and input_index is not None:
            if output.result.truth.payment_stories[input_index].scam_step == SCAM_STEP_VICTIM_PAYMENT:
                victims.add(events[position].from_account)
    lines_text = "\n".join(event_lines(interactive_fast_output(output)))
    return {
        "world_id": spec.world_id, "seed": spec.seed, "gang_count": len(gangs),
        "seconds_total": round(total_seconds, 3),
        "seconds_planning_all_gangs": round(sum(planning_seconds), 3),
        "seconds_planning_per_gang": ";".join(f"{value:.3f}" for value in planning_seconds),
        "events": len(events), "payments_success": sum(1 for event in events if event.status == "success"),
        "true_complaints": true_count, "false_complaints": false_count, "registry_reports": report_count,
        "mule_accounts": len(mules), "victims": len(victims),
        "budget_money_used_share_max": round(money_share, 6), "budget_time_used_share_max": round(time_share, 6),
        "invariant_problems": len(problems), "atm_checked": atm.checked, "atm_skipped": atm.skipped,
        **lien_columns,
        "event_lines_sha256": hashlib.sha256(lines_text.encode("utf-8")).hexdigest(),
        "run_label": label, "_problems": problems,
    }


def measure_task(task: Task) -> dict[str, Any]:
    """
    Measures one task and never raises: a crash comes back as a record with "_crash" set, holding the world id, split,
    seed, exception type and message, and the last lines of the traceback, so a sweep goes on to the next world.
    """
    spec, preset, mode, label = task
    try:
        return measure_world(spec, preset, mode, label)
    except Exception as error:  # noqa: BLE001 - a sweep must record every kind of crash and continue
        trace = traceback.format_exc().splitlines()[-TRACEBACK_LINES:]
        return {
            "_crash": True, "world_id": spec.world_id, "split": spec.split, "seed": spec.seed, "run_label": label,
            "error_type": type(error).__name__, "error_message": str(error), "traceback": "\n".join(trace),
        }


def run_tasks(tasks: list[Task], workers: int) -> tuple[list[dict[str, Any]], float]:
    """Measures every task, serially or with a process pool; returns the results in task order and the wall time."""
    started = time.perf_counter()
    if workers <= 1:
        results = [measure_task(task) for task in tasks]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(measure_task, tasks))
    return results, time.perf_counter() - started


def build_tasks(args: argparse.Namespace) -> list[Task]:
    """Plans the worlds to measure: the fixed sweep, or one block (optionally limited), from the command line."""
    cfg = load_sampler_config()
    ranges = read_world_configs().world_ranges
    tasks: list[Task] = []
    if args.sweep or args.sweep_row:
        for name, label, split, block, limit, mode in SWEEP_PLAN:
            if args.sweep_row and name != args.sweep_row:
                continue
            specs = sample_block(split, block, args.sampler_seed, cfg, ranges)
            for spec in specs[:limit]:
                tasks.append((spec, args.preset, mode, label))
        return tasks
    specs = sample_block(args.split, args.block, args.sampler_seed, cfg, ranges)
    for spec in specs[: args.limit]:
        tasks.append((spec, args.preset, "plain", f"{args.split} block {args.block}"))
    return tasks


def write_outputs(results: list[dict[str, Any]], out_path: Path) -> None:
    """Writes the CSV of measured rows, the problems file and the crashes file next to it, with LF line endings."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [result for result in results if not result.get("_crash")]
    with open(out_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS), lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    problem_lines = [f"{row['world_id']}: {text}" for row in rows for text in row["_problems"]]
    # A skipped ATM withdrawal (no planned limit to compare with) is stated, never silent.
    atm_skipped = sum(int(row["atm_skipped"]) for row in rows)
    atm_checked = sum(int(row["atm_checked"]) for row in rows)
    problem_lines.append(
        f"ATM daily-limit check: {atm_checked} withdrawals checked, {atm_skipped} skipped (no limit known)"
    )
    with open(f"{out_path}.problems.txt", "w", encoding="utf-8", newline="") as handle:
        handle.write("\n".join(problem_lines) + "\n")
    crash_blocks = [format_crash(result) for result in results if result.get("_crash")]
    with open(f"{out_path}.crashes.txt", "w", encoding="utf-8", newline="") as handle:
        handle.write("\n\n".join(crash_blocks) + ("\n" if crash_blocks else ""))


def format_crash(crash: dict[str, Any]) -> str:
    """Formats one crash record: world id, split, seed, exception type and message, then the traceback tail."""
    return (
        f"CRASH {crash['world_id']} split={crash['split']} seed={crash['seed']} ({crash['run_label']})\n"
        f"{crash['error_type']}: {crash['error_message']}\n{crash['traceback']}"
    )


def print_summary(rows: list[dict[str, Any]], wall_seconds: float, workers: int) -> None:
    """Prints mean, median, min and max of each numeric column, the worlds per gang count and the machine facts."""
    print(f"worlds measured: {len(rows)}  workers: {workers}  wall-clock seconds for the run: {wall_seconds:.2f}")
    if rows:
        for name in NUMERIC_COLUMNS:
            values = [float(row[name]) for row in rows]
            print(
                f"{name}: mean={statistics.mean(values):.3f} median={statistics.median(values):.3f} "
                f"min={min(values):.3f} max={max(values):.3f}"
            )
        counts: dict[int, int] = {}
        for row in rows:
            counts[row["gang_count"]] = counts.get(row["gang_count"], 0) + 1
        print("worlds per gang_count: " + ", ".join(f"{key}: {counts[key]}" for key in sorted(counts)))
    print(f"python={platform.python_version()} cpu_count={os.cpu_count()} platform={platform.platform()}")
    print("ESTIMATE: one machine; measures, decides nothing")


def print_table(results: list[dict[str, Any]]) -> None:
    """Prints one line per run label: worlds run, crashed, and worlds with invariant problems."""
    labels: list[str] = []
    for result in results:
        if result["run_label"] not in labels:
            labels.append(result["run_label"])
    print(f"{'run':<34}{'worlds run':>11}{'crashed':>9}{'with problems':>15}")
    total = [0, 0, 0]
    for label in labels:
        group = [result for result in results if result["run_label"] == label]
        crashed = len([result for result in group if result.get("_crash")])
        with_problems = len([result for result in group if not result.get("_crash") and result["_problems"]])
        print(f"{label:<34}{len(group):>11}{crashed:>9}{with_problems:>15}")
        total[0] += len(group)
        total[1] += crashed
        total[2] += with_problems
    print(f"{'total':<34}{total[0]:>11}{total[1]:>9}{total[2]:>15}")


def compare_rows(serial: list[dict[str, Any]], parallel: list[dict[str, Any]]) -> list[str]:
    """Returns one message per world whose event-line hash (or crash state) differs between the two runs."""
    mismatches: list[str] = []
    for serial_row, parallel_row in zip(serial, parallel):
        if serial_row.get("_crash") or parallel_row.get("_crash"):
            if bool(serial_row.get("_crash")) != bool(parallel_row.get("_crash")):
                mismatches.append(f"{serial_row['world_id']}: crashed in only one of the two runs")
        elif serial_row["event_lines_sha256"] != parallel_row["event_lines_sha256"]:
            mismatches.append(
                f"{serial_row['world_id']}: serial {serial_row['event_lines_sha256'][:16]} "
                f"!= parallel {parallel_row['event_lines_sha256'][:16]}"
            )
    return mismatches


def main(argv: list[str] | None = None) -> int:
    """
    Plans the worlds, measures them, writes the outputs, prints the summary and table, runs the optional parallel
    comparison, and returns non-zero if any world crashed, had invariant problems, or differed between runs.
    """
    args = read_arguments(sys.argv[1:] if argv is None else argv)
    tasks = build_tasks(args)
    results, wall_seconds = run_tasks(tasks, args.workers)
    write_outputs(results, Path(args.out))
    rows = [result for result in results if not result.get("_crash")]
    crashes = [result for result in results if result.get("_crash")]
    print(",".join(COLUMNS))
    for row in rows:
        print(",".join(str(row[name]) for name in COLUMNS))
    print_summary(rows, wall_seconds, args.workers)
    print_table(results)
    for crash in crashes:
        print(format_crash(crash))
    problem_count = sum(len(row["_problems"]) for row in rows)
    exit_code = 1 if crashes or problem_count else 0
    if args.compare > 0:
        first = tasks[: args.compare]
        serial_results, serial_wall = run_tasks(first, 1)
        mismatches = compare_rows(serial_results, results[: len(first)])
        print(f"compare: {len(first)} worlds serial {serial_wall:.2f}s, parallel run {wall_seconds:.2f}s")
        if mismatches:
            print("MISMATCH: " + "; ".join(mismatches))
            exit_code = 1
        else:
            print("IDENTICAL")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
