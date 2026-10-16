"""
WHAT: Showcase check runners 1 through 7 for Level 11 simulation verification.
It validates pre-registration amendments, issue register integrity, tiny world execution, empirical parameter grounding,
independent seed divergence, bank outage partial degradation, and mobile OS rollout adoption curves.

WHY: Provides concrete programmatic evidence validating simulation behavior for the Level 11 showcase report.
Breaking checks into modular files guarantees that all routines remain well under the 30-line code ceiling.
Executing against canonical seeds verifies that simulation mechanics behave deterministically and match empirical data.

TERM: pre-registration: Formally committing model structures, evaluation rules, and ranges prior to running experiments,
      e.g. recording metrics and bounds in config/evaluation.yaml before generating test data.
TERM: issue register: A structured inventory of architectural questions and tasks with evidence and status tracking,
      e.g. issues.yaml tracking 31 numbered foundational topics.
TERM: fast mode: A self-contained simulation runtime producing entire multi-day world histories directly in memory,
      e.g. generating 61 days of transactions for 60 citizens in under a second.
TERM: empirical grounding: Connecting simulation parameters to real-world reports, studies, or explicit assumption
      records,
      e.g. linking technical decline rates to published NPCI monthly system metrics.
TERM: partial degradation: An incident state where system performance drops significantly without collapsing to zero,
      e.g. bank outage reducing overall transaction success rate while unaffected institutions process payments
      normally.
"""

import copy
import dataclasses
import hashlib
import math
import time
from pathlib import Path
from typing import Any

from core import issue_register, level10_check
from core.config_validator import load_yaml
from showcase.framework import CheckOutcome
from showcase.level_11_charts import adoption_curve, daily_bars_with_average, rate_line_with_window
from showcase.level_11_festival import (
    calendar_parts,
    choose_festival_people,
    measure_festival_uplift,
    split_daily_counts,
)
from world.background_config import read_background_realism
from world.build_world import build_world
from world.device_schedule import draw_device_event_schedule
from world.fast_mode import FastRunSettings, event_lines, run_fast_world
from world.grounding import count_by_label, grounding_rows
from world.layers import apply_layers, read_layer
from world.placeholders import OS_GENERATIONS
from world.world_configs import read_world_configs
from world.world_plan import draw_world_plan, spawn_life_streams, spawn_world_streams

REPO_ROOT = Path(__file__).resolve().parent.parent
AMENDMENTS_PATH = REPO_ROOT / "docs" / "research" / "amendments.md"
BANK_OUTAGE_LAYER_PATH = REPO_ROOT / "scenarios" / "stress" / "bank-outage-long.yaml"
OS_WAVE_LAYER_PATH = REPO_ROOT / "scenarios" / "stress" / "os-update-wave.yaml"
SHOWCASE_CONFIG_PATH = REPO_ROOT / "config" / "showcase_level_11.yaml"


async def run_amendments_check(workspace: Any) -> CheckOutcome:
    """
    WHAT: Audits pre-registration amendments recorded in amendments.md and verifies git commit ordering proof.
    It counts documented amendment headings and invokes level10_check.preregistration_proof in-process.
    It asserts that all amendments are accounted for and that no new-engine file predates pre-registered rules.

    WHY: Level 11 builds directly upon Level 10 pre-registration commitments and empirical research foundations.
    Proving that configuration adjustments were formally recorded ensures complete research provenance.
    Verifying commit dates confirms that the engine honors the pre-registered scientific methodology.

    Args:
        workspace: Optional showcase execution workspace environment container.

    Returns:
        CheckOutcome documenting verified amendment counts, git commit dates, and pre-registration status.

    Example:
        >>> import asyncio
        >>> outcome = asyncio.run(run_amendments_check(None))
        >>> "amendments recorded" in outcome.key_number
        True
    """
    content = AMENDMENTS_PATH.read_text(encoding="utf-8")
    amendments = [line for line in content.splitlines() if line.startswith("## A-")]
    count = len(amendments)
    assert count >= 11, f"Expected at least 11 amendments, found {count}"
    proof = level10_check.preregistration_proof()
    assert proof["result"] == "OK", f"Preregistration check failed: {proof['sentence']}"
    key = f"{count} amendments recorded; new-engine files all postdate evaluation.yaml"
    evidence = {
        "amendments_count": count,
        "amendment_ids": [line.split()[1] for line in amendments],
        "preregistration": proof,
        "proof_sentence": proof["sentence"],
    }
    return CheckOutcome(key_number=key, evidence=evidence)


async def run_issue_register_check(workspace: Any) -> CheckOutcome:
    """
    WHAT: Audits the central research issue register in issues.yaml and verifies rejection of corrupted copies.
    It counts issues categorized as built, planned, and deferred, matching expected status invariants.
    It then verifies that removing mandatory IDs and inserting unlisted IDs are both rejected by check_register.

    WHY: Verifies that the research foundation remains intact and that issues.yaml conforms to schema standards.
    Confirming that R2 has advanced to built status reflects the completion of the Level 11 stress library.
    Testing mutation rejection proves that the register validation routines actively defend against data loss.

    Args:
        workspace: Optional showcase execution workspace environment container.

    Returns:
        CheckOutcome recording verified issue status tallies and mutation rejection proof.

    Example:
        >>> import asyncio
        >>> outcome = asyncio.run(run_issue_register_check(None))
        >>> "28 built" in outcome.key_number
        True
    """
    reg_path = REPO_ROOT / "config" / "issues.yaml"
    data = issue_register.load_register(reg_path)
    errors = issue_register.check_register(data, REPO_ROOT)
    assert not errors, f"Issue register validation errors: {errors}"
    built = sum(1 for v in data.values() if v.get("status") == "built")
    planned = sum(1 for v in data.values() if v.get("status") == "planned")
    deferred = sum(1 for v in data.values() if v.get("status") == "deferred")
    assert built == 28 and planned == 2 and deferred == 1
    bad_copy = copy.deepcopy(data)
    del bad_copy["G1"]
    bad_errors = issue_register.check_register(bad_copy, REPO_ROOT)
    assert any("Missing expected IDs" in err and "G1" in err for err in bad_errors)
    unlisted_copy = copy.deepcopy(data)
    unlisted_copy["R999"] = {"status": "planned", "title": "Unlisted test issue", "later_work": ["Level 99"]}
    unlisted_errors = issue_register.check_register(unlisted_copy, REPO_ROOT)
    assert any("Unknown IDs found in register" in err and "R999" in err for err in unlisted_errors)
    key = f"{built} built, {planned} planned, {deferred} deferred; broken copy rejected"
    return CheckOutcome(
        key_number=key,
        evidence={
            "built": built, "planned": planned, "deferred": deferred,
            "corrupted_missing_rejected": True, "unlisted_id_rejected": True,
        },
    )


def _calc_festival_uplift(
    daily_counts: list[int], festivals: list[Any], n_days: int
) -> tuple[float, float]:
    """
    WHAT: Calculates average daily payment volumes inside and outside festival calendar windows.
    It identifies all calendar days falling within any active festival duration span.
    It returns the average transaction counts for festival days versus non-festival days.

    WHY: Helper ensuring run_tiny_world_check satisfies the 30-line code ceiling constraint.
    Provides verified numerical comparison of holiday payment uplift for showcase Check 3.

    Args:
        daily_counts: List of total transaction attempts per calendar day index.
        festivals: List of Festival objects with peak_day, rise_days, and fade_days properties.
        n_days: Total observation days in the simulation horizon.

    Returns:
        Tuple of (average payments per day in festivals, average payments per day outside festivals).

    Example:
        >>> avg_in, avg_out = _calc_festival_uplift([10, 20], [], 2)
        >>> avg_in == 0.0 and avg_out == 15.0
        True
    """
    fest_days = set()
    for f in festivals:
        s_d = max(f.peak_day - f.rise_days, 0)
        e_d = min(f.peak_day + f.fade_days, n_days - 1)
        fest_days.update(range(s_d, e_d + 1))
    in_fest = [daily_counts[d] for d in fest_days]
    out_fest = [daily_counts[d] for d in range(n_days) if d not in fest_days]
    avg_in = sum(in_fest) / len(in_fest) if in_fest else 0.0
    avg_out = sum(out_fest) / len(out_fest) if out_fest else 0.0
    return avg_in, avg_out


def _run_festival_world(tiny_scaled: list[int]) -> dict[str, Any]:
    """
    WHAT: Sizes and runs the festival world, then measures festival uplift on its calendar-scaled payments.
    The size comes from the config rule (smallest candidate whose predicted signal ratio reaches the minimum),
    evaluated with the plan's multipliers and the tiny world's base rate before the larger world is run.

    WHY: The tiny world is too small for a k = 4 festival test; Poisson noise there is larger than the rise.
    Fixing the size by a written rule keeps the choice independent of the festival-day results.

    Args:
        tiny_scaled: Calendar-scaled payment counts per day from the 60-person tiny world.

    Returns:
        Dictionary of the chosen size, predicted ratios, run seconds, per-day lists and the FestivalUplift.

    Example:
        >>> from showcase.level_11_festival import split_daily_counts
        >>> tiny = run_fast_world(FastRunSettings(seed=101, n_people=60, n_shops=12, n_days=61, range_name="training"))
        >>> split = split_daily_counts(tiny.result.events, tiny.result.event_purposes, 61)
        >>> _run_festival_world(split.calendar_scaled)["n_people"]
        4000
    """
    cfg = load_yaml(SHOWCASE_CONFIG_PATH)
    rule, k_sigma = cfg["festival_world"], float(cfg["expected_invariants"]["k_sigma"])
    plan = draw_world_plan(spawn_world_streams(rule["seed"])[0], rule["n_days"], rule["range_name"])
    other, fest = calendar_parts(plan, rule["n_days"])
    n_people, ratios = choose_festival_people(
        tiny_scaled, cfg["tiny_world"]["n_people"], rule["candidate_people"], other, fest, rule["min_planned_ratio"]
    )
    settings = FastRunSettings(
        seed=rule["seed"], n_people=n_people, n_shops=n_people // rule["people_per_shop"],
        n_days=rule["n_days"], range_name=rule["range_name"],
    )
    started = time.perf_counter()
    output = run_fast_world(settings)
    seconds = time.perf_counter() - started
    # The plan depends only on the seed, so the sized world must have the same calendar used for sizing.
    assert output.plan.festivals == plan.festivals, "Festival world plan differs from the sizing plan"
    split = split_daily_counts(output.result.events, output.result.event_purposes, settings.n_days)
    uplift = measure_festival_uplift(split.calendar_scaled, other, fest, k_sigma)
    return {
        "n_people": n_people, "n_shops": settings.n_shops, "run_seconds": seconds,
        "predicted_ratios": {str(n): r for n, r in ratios.items()},
        "k_sigma": k_sigma, "c_d": split.total, "u_d": split.fixed_date, "s_d": split.calendar_scaled,
        "other_d": other, "fest_d": fest, "u_total": sum(split.fixed_date), "uplift": uplift,
    }


async def run_tiny_world_check(workspace: Any) -> CheckOutcome:
    """
    WHAT: Runs the tiny world end to end, draws its daily view, and tests the festival uplift on a sized world.
    The tiny world (seed 101, 60 people, 12 shops, 61 days) supplies the hero chart and the plain daily averages.
    The festival test removes weekend, month-start and fixed-date payments and compares the extra festival-day
    payments with the plan at k = 4 standard errors.

    WHY: A plain inside-versus-outside average hides the festival effect behind weekends and month starts.
    Testing the isolated festival factor is the honest way to show that the calendar plan is applied.

    Args:
        workspace: Optional showcase execution workspace environment container.

    Returns:
        CheckOutcome with the festival excess, planned excess, standard errors, every per-day value and the chart.

    Example:
        >>> import asyncio
        >>> outcome = asyncio.run(run_tiny_world_check(None))
        >>> "standard errors" in outcome.key_number and outcome.evidence["chart_svg"].startswith("<svg")
        True
    """
    settings = FastRunSettings(seed=101, n_people=60, n_shops=12, n_days=61, range_name="training")
    output = run_fast_world(settings)
    events = output.result.events
    tiny = split_daily_counts(events, output.result.event_purposes, settings.n_days)
    dates = [output.plan.clock.to_datetime(d * 1440).date().isoformat() for d in range(settings.n_days)]
    festivals = [
        (max(f.peak_day - f.rise_days, 0), min(f.peak_day + f.fade_days, settings.n_days - 1), f.name)
        for f in output.plan.festivals
    ]
    svg_chart = daily_bars_with_average(tiny.total, dates, festivals)
    fails = sum(1 for e in events if e.status == "failed")
    avg_in, avg_out = _calc_festival_uplift(tiny.total, output.plan.festivals, settings.n_days)
    big = _run_festival_world(tiny.calendar_scaled)
    up = big["uplift"]
    assert up.rise_is_real, f"Festival excess {up.excess:.1f} is not above {big['k_sigma']} x se {up.se_excess:.1f}"
    assert up.size_matches_plan, f"Excess {up.excess:.1f} differs from planned {up.planned_excess:.1f} by > k se"
    key = (
        f"festival days ({big['n_people']:,}-person world): {up.excess:,.0f} more payments than expected "
        f"without festivals (planned {up.planned_excess:,.0f}; {up.excess / up.se_excess:.1f} standard errors)"
    )
    evidence = {
        "attempts": len(events), "successes": len(events) - fails, "failures": fails,
        "days": settings.n_days, "people": settings.n_people, "shops": settings.n_shops,
        "plain_daily_view": {"avg_in_festival": avg_in, "avg_outside_festival": avg_out},
        "festival_world": {**{k: v for k, v in big.items() if k != "uplift"}, **dataclasses.asdict(up)},
        "chart_svg": svg_chart,
    }
    return CheckOutcome(key_number=key, evidence=evidence)


async def run_grounding_check(workspace: Any) -> CheckOutcome:
    """
    WHAT: Compiles the full grounding catalog across background, failure, and device realism configurations.
    It parses parameter metadata using world.grounding.grounding_rows and tallies epistemic labels.
    It asserts that all 33 parameters are documented, with exactly 30 assumptions and 3 reported values.

    WHY: Enforces rigorous empirical traceability across all domain configuration inputs.
    Verifying that every parameter declares its epistemic origin defends against unvetted assumptions.
    Providing exact label counts ensures that the Level 11 grounding inventory satisfies audited standards.

    Args:
        workspace: Optional showcase execution workspace environment container.

    Returns:
        CheckOutcome containing grounding parameter tallies, epistemic label distributions, and row data.

    Example:
        >>> import asyncio
        >>> outcome = asyncio.run(run_grounding_check(None))
        >>> "33 parameters" in outcome.key_number
        True
    """
    cfgs = read_world_configs()
    rows = grounding_rows(cfgs)
    counts = count_by_label(rows)
    assert len(rows) == 33, f"Expected 33 grounded parameters, got {len(rows)}"
    assert counts.get("assumption", 0) == 30, f"Expected 30 assumptions, got {counts.get('assumption')}"
    assert counts.get("reported", 0) == 3, f"Expected 3 reported parameters, got {counts.get('reported')}"
    key = f"{len(rows)} parameters grounded: {counts['assumption']} assumption, {counts['reported']} reported"
    evidence = {
        "total_parameters": len(rows),
        "label_counts": counts,
        "parameters": [{"config": r.config_name, "setting": r.setting, "label": r.label} for r in rows],
    }
    return CheckOutcome(key_number=key, evidence=evidence)


def _collect_seed_timings(seed: int, n_days: int) -> tuple[dict[str, int], list[int], list[int]]:
    """
    WHAT: Extracts festival peak days, bank outage start days, and OS update wave days for a simulation seed.
    It builds a small world plan and device schedule under the given master seed across the observation horizon.
    The resulting timings allow direct side-by-side comparison between divergent pseudo-random seeds.

    WHY: Helper ensuring run_seed_divergence_check remains well under the 30-line code limit.
    Isolates calendar plan timing extraction and device schedule generation from table formatting logic.

    Args:
        seed: Master integer random seed initializing pseudo-random number generator streams.
        n_days: Total simulation horizon duration in full calendar days.

    Returns:
        Tuple of (festival peaks dictionary, bank outage start days list, OS update days list).

    Example:
        >>> fests, outages, os_days = _collect_seed_timings(101, 61)
        >>> 'diwali' in fests and len(outages) > 0 and len(os_days) > 0
        True
    """
    w = build_world(seed, 60, 12)
    p_rng, _ = spawn_world_streams(seed)
    plan = draw_world_plan(p_rng, n_days)
    d_rng, _ = spawn_life_streams(seed)
    bg = read_background_realism(REPO_ROOT / "config" / "device_realism.yaml")
    dev_ev = draw_device_event_schedule(bg, d_rng, w.people, plan.festivals, plan.n_days)
    fests = {f.name: f.peak_day for f in plan.festivals}
    outages = [inc.start_minute // 1440 for inc in plan.incidents if inc.kind == "bank_downtime"]
    os_days = sorted(set(e.minute // 1440 for e in dev_ev if e.kind == "os_update"))
    return fests, outages, os_days


def _nth_or_none(values: list[int], index: int) -> int | None:
    """
    WHAT: Returns values[index] when it exists, otherwise None.
    WHY: A seed may schedule fewer outages or OS update days than another; a missing timing must show as missing,
    not as day 0, so the side-by-side table does not invent a value.

    Args:
        values: Sorted day numbers.
        index: Zero-based position wanted.

    Returns:
        The day number, or None when the list is too short.

    Example:
        >>> _nth_or_none([5, 9], 1), _nth_or_none([5, 9], 2)
        (9, None)
    """
    return values[index] if index < len(values) else None


def _seed_timing_rows(first_seed: int, second_seed: int, n_days: int) -> list[dict[str, Any]]:
    """
    WHAT: Builds the side-by-side timing rows for two seeds: festival peaks, outage starts and OS update days.
    Each row holds the timing name, the day for each seed (None when absent) and whether the two differ.

    WHY: The page draws check 5's table from these rows and the key number counts them, so both always agree.

    Args:
        first_seed: The canonical seed (101).
        second_seed: The comparison seed from config (102).
        n_days: Simulation horizon in days.

    Returns:
        List of row dictionaries with keys timing, first_seed_day, second_seed_day and differs.

    Example:
        >>> rows = _seed_timing_rows(101, 102, 61)
        >>> rows[0]
        {'timing': 'Diwali peak', 'first_seed_day': 30, 'second_seed_day': 40, 'differs': True}
    """
    f1, o1, os1 = _collect_seed_timings(first_seed, n_days)
    f2, o2, os2 = _collect_seed_timings(second_seed, n_days)
    pairs: list[tuple[str, int | None, int | None]] = [
        ("Diwali peak", f1.get("diwali"), f2.get("diwali")),
        ("Festival 1 peak", f1.get("other_festival_1"), f2.get("other_festival_1")),
        ("Festival 2 peak", f1.get("other_festival_2"), f2.get("other_festival_2")),
    ]
    pairs += [(f"Outage {i + 1} start", _nth_or_none(o1, i), _nth_or_none(o2, i)) for i in range(5)]
    pairs += [("OS update first day", _nth_or_none(os1, 0), _nth_or_none(os2, 0))]
    pairs += [("OS update second day", _nth_or_none(os1, 1), _nth_or_none(os2, 1))]
    return [
        {"timing": name, "first_seed_day": d1, "second_seed_day": d2, "differs": d1 != d2}
        for name, d1, d2 in pairs
    ]


async def run_seed_divergence_check(workspace: Any) -> CheckOutcome:
    """
    WHAT: Compares seed 101 with the config's comparison seed on festival peaks, outage starts and OS update days.
    It stores the timing rows in evidence; the page builds the side-by-side table from those same rows.
    It also runs both worlds in fast mode and records each event file's SHA-256.

    WHY: Different seeds must give different calendars; identical timings would mean the random streams collapsed.

    Args:
        workspace: Optional showcase execution workspace environment container.

    Returns:
        CheckOutcome with "N of M timings differ", the timing rows and both event hashes.

    Example:
        >>> import asyncio
        >>> outcome = asyncio.run(run_seed_divergence_check(None))
        >>> outcome.key_number
        '10 of 10 timings differ'
    """
    second_seed = int(load_yaml(SHOWCASE_CONFIG_PATH)["comparison_world"]["seed"])
    rows = _seed_timing_rows(101, second_seed, 61)
    diffs = sum(1 for row in rows if row["differs"])
    assert diffs >= 8, f"Expected at least 8 differing timings, found {diffs}"
    s1 = FastRunSettings(seed=101, n_people=60, n_shops=12, n_days=61, range_name="training")
    s2 = FastRunSettings(seed=second_seed, n_people=60, n_shops=12, n_days=61, range_name="training")
    out1, out2 = run_fast_world(s1), run_fast_world(s2)
    h1 = hashlib.sha256("".join(event_lines(out1)).encode("utf-8")).hexdigest()
    h2 = hashlib.sha256("".join(event_lines(out2)).encode("utf-8")).hexdigest()
    assert h1 != h2, "Two seeds produced identical event files"
    evidence = {
        "first_seed": 101, "second_seed": second_seed, "timings": rows,
        "sha256_first_seed": h1, "sha256_second_seed": h2,
        "events_first_seed": len(out1.result.events), "events_second_seed": len(out2.result.events),
    }
    return CheckOutcome(key_number=f"{diffs} of {len(rows)} timings differ", evidence=evidence)


def _calc_hourly_rates(events: list[Any], max_minute: int) -> tuple[list[float], list[float]]:
    """
    WHAT: Computes hourly success rate percentages across the complete simulation observation period.
    It buckets payment attempts into 60-minute intervals and calculates the fraction of successful transactions.
    Empty hours without transactions inherit the baseline success rate of 100% to preserve curve continuity.

    WHY: Helper routine keeping run_bank_outage_check within the 30-line function length limit.
    Binning by hour exposes the sharp temporal dip during bank downtime while filtering minute noise.
    Produces clean coordinate sequences for rendering the rate_line_with_window vector chart.

    Args:
        events: List of PaymentEvent instances recorded during the simulation run.
        max_minute: Highest minute timestamp observed in the simulation timeline.

    Returns:
        Tuple of (hourly timestamp list, corresponding success percentage list).

    Example:
        >>> from world.failures import PaymentEvent
        >>> ev = PaymentEvent(10, "a1", "a2", 500, 0, 1, "success", "ok", "none")
        >>> hrs, rts = _calc_hourly_rates([ev], 59)
        >>> len(hrs) == 1 and rts[0] == 100.0
        True
    """
    total_hours = max(max_minute // 60 + 1, 1)
    attempts = [0] * total_hours
    successes = [0] * total_hours
    for ev in events:
        h = ev.minute // 60
        if h < total_hours:
            attempts[h] += 1
            if ev.status == "success":
                successes[h] += 1
    hours = [float(h) for h in range(total_hours)]
    rates = [(successes[h] / attempts[h]) * 100.0 if attempts[h] > 0 else 100.0 for h in range(total_hours)]
    return hours, rates


def _count_window_outcomes(events: list[Any], inc: Any) -> dict[str, int]:
    """
    WHAT: Counts attempts and successes inside one bank outage window, split by whether the outage bank is involved.
    A payment involves the outage bank when its sending or receiving bank index equals the incident's bank_index.
    It also counts failures with reason bank_downtime on each side.

    WHY: Partial degradation means the outage bank's payments partly fail while other banks keep working.
    Raw counts are reported because a few hours of a tiny world hold only a handful of payments.

    Args:
        events: PaymentEvent records from the run.
        inc: Incident with start_minute, duration_minutes and bank_index.

    Returns:
        Dictionary with outage_attempts, outage_successes, outage_downtime_failures, other_attempts,
        other_successes and other_downtime_failures.

    Example:
        >>> from types import SimpleNamespace as E
        >>> inc = E(start_minute=0, duration_minutes=60, bank_index=2)
        >>> evs = [E(minute=5, from_bank_index=2, to_bank_index=0, status="failed", reason="bank_downtime"),
        ...        E(minute=9, from_bank_index=0, to_bank_index=1, status="success", reason="ok"),
        ...        E(minute=99, from_bank_index=2, to_bank_index=2, status="failed", reason="bank_downtime")]
        >>> counts = _count_window_outcomes(evs, inc)
        >>> counts["outage_attempts"], counts["outage_downtime_failures"], counts["other_successes"]
        (1, 1, 1)
    """
    end = inc.start_minute + inc.duration_minutes
    names = ("attempts", "successes", "downtime_failures")
    counts = {f"{side}_{name}": 0 for side in ("outage", "other") for name in names}
    for ev in events:
        if not inc.start_minute <= ev.minute < end:
            continue
        # Payments touching the outage bank on either side are exposed to the outage.
        side = "outage" if inc.bank_index in (ev.from_bank_index, ev.to_bank_index) else "other"
        counts[f"{side}_attempts"] += 1
        counts[f"{side}_successes"] += int(ev.status == "success")
        counts[f"{side}_downtime_failures"] += int(ev.reason == "bank_downtime")
    return counts


def _percent(part: int, whole: int) -> float:
    """
    WHAT: Returns part / whole as a percentage, or 0.0 when whole is zero.
    WHY: Keeps the key number readable without dividing by zero when a window holds no payments.

    Args:
        part: Number of successes.
        whole: Number of attempts.

    Returns:
        Percentage between 0 and 100.

    Example:
        >>> round(_percent(2, 3), 1)
        66.7
        >>> _percent(0, 0)
        0.0
    """
    return 100.0 * part / whole if whole else 0.0


async def run_bank_outage_check(workspace: Any) -> CheckOutcome:
    """
    WHAT: Runs the tiny world with the bank-outage-long layer and counts outcomes inside its first bank outage.
    Payments touching the outage bank must include both successes and bank_downtime failures (partial, not zero),
    while payments between the other banks must show no bank_downtime failures.

    WHY: One bank's downtime should degrade only that bank's payments. The counts are small, so the check
    claims only what they show: some outage-bank payments still succeed and other banks are not hit by downtime.

    Args:
        workspace: Optional showcase execution workspace environment container.

    Returns:
        CheckOutcome with raw counts for the outage bank and the other banks, the baseline rate and the chart.

    Example:
        >>> import asyncio
        >>> outcome = asyncio.run(run_bank_outage_check(None))
        >>> outcome.key_number.startswith("outage bank ")
        True
    """
    layered_cfgs, _ = apply_layers(read_world_configs(), [read_layer(BANK_OUTAGE_LAYER_PATH)])
    settings = FastRunSettings(seed=101, n_people=60, n_shops=12, n_days=61, range_name="training")
    output = run_fast_world(settings, configs=layered_cfgs)
    incidents = [inc for inc in output.plan.incidents if inc.kind == "bank_downtime"]
    assert incidents, "Expected at least one bank downtime incident"
    inc = incidents[0]
    c = _count_window_outcomes(output.result.events, inc)
    assert 0 < c["outage_successes"] < c["outage_attempts"], f"Outage bank not partly degraded: {c}"
    assert c["outage_downtime_failures"] > 0, f"No bank_downtime failure at the outage bank: {c}"
    assert c["other_attempts"] > 0 and c["other_downtime_failures"] == 0, f"Other banks hit by downtime: {c}"
    # Baseline: every payment outside all bank downtime windows of this run.
    outside = [ev for ev in output.result.events if not any(
        i.start_minute <= ev.minute < i.start_minute + i.duration_minutes for i in incidents)]
    base_rate = _percent(sum(1 for ev in outside if ev.status == "success"), len(outside))
    hours, rates = _calc_hourly_rates(output.result.events, settings.n_days * 1440)
    win = (inc.start_minute / 60.0, (inc.start_minute + inc.duration_minutes) / 60.0, "Bank Outage Window")
    key = (
        f"outage bank {c['outage_successes']} of {c['outage_attempts']} succeeded "
        f"({_percent(c['outage_successes'], c['outage_attempts']):.1f}%); other banks {c['other_successes']} of "
        f"{c['other_attempts']} ({_percent(c['other_successes'], c['other_attempts']):.1f}%); "
        f"baseline {base_rate:.1f}%"
    )
    evidence = {**c, "outage_bank_index": inc.bank_index, "window_start_minute": inc.start_minute,
                "window_minutes": inc.duration_minutes, "baseline_rate_outside_all_outages": base_rate,
                "incidents": len(incidents), "chart_svg": rate_line_with_window(hours, rates, win)}
    return CheckOutcome(key_number=key, evidence=evidence)


def _newest_os_share_by_day(people: list[Any], device_events: list[Any], n_days: int) -> list[float]:
    """
    WHAT: Returns, for each day, the share of phones running the newest OS generation at the end of that day.
    It starts from each person's starting OS generation and applies device events in time order.

    WHY: Adoption over time is how the base and stressed OS update waves are compared on the chart and key number.

    Args:
        people: People from a freshly built world (their devices hold the starting OS generation).
        device_events: DeviceEvent records sorted by minute.
        n_days: Number of days.

    Returns:
        List of shares between 0 and 1, one per day.

    Example:
        >>> from types import SimpleNamespace as N
        >>> ppl = [N(person_id="a", device=N(os_generation=3)), N(person_id="b", device=N(os_generation=4))]
        >>> evs = [N(person_id="a", minute=1500, os_generation_after=4)]
        >>> _newest_os_share_by_day(ppl, evs, 2)
        [0.5, 1.0]
    """
    generation = {p.person_id: p.device.os_generation for p in people}
    shares: list[float] = []
    position = 0
    for day in range(n_days):
        # Apply every event that happened on or before this day.
        while position < len(device_events) and device_events[position].minute // 1440 <= day:
            generation[device_events[position].person_id] = device_events[position].os_generation_after
            position += 1
        shares.append(sum(1 for g in generation.values() if g == OS_GENERATIONS) / len(generation))
    return shares


def _os_wave_run(cfgs: Any, settings: FastRunSettings) -> dict[str, Any]:
    """
    WHAT: Runs one world and redraws its device schedule to measure OS updates and new-device flags.
    The schedule is redrawn with the same seed stream and inputs that fast mode uses, and its length is checked
    against the run's own count of applied device events so both describe the same run.

    WHY: Fast mode returns device stamps but not the device schedule; the schedule is needed for adoption over time.

    Args:
        cfgs: WorldConfigs (base or layered).
        settings: FastRunSettings for the tiny world.

    Returns:
        Dictionary with new_device_flags, os_update_events, os_update_days and newest_os_share per day.

    Example:
        >>> s = FastRunSettings(seed=101, n_people=60, n_shops=12, n_days=61, range_name="training")
        >>> _os_wave_run(read_world_configs(), s)["os_update_events"]
        25
    """
    output = run_fast_world(settings, configs=cfgs)
    # A fresh world is needed: the run changed people's devices in place.
    world = build_world(settings.seed, settings.n_people, settings.n_shops, ranges=cfgs.world_ranges)
    device_rng, _ = spawn_life_streams(settings.seed)
    schedule = draw_device_event_schedule(
        cfgs.device_realism, device_rng, world.people, output.plan.festivals, output.plan.n_days
    )
    assert len(schedule) == output.result.device_counts["events_applied"], "Redrawn device schedule differs"
    os_updates = [ev for ev in schedule if ev.kind == "os_update"]
    return {
        "new_device_flags": sum(1 for st in output.result.event_sender_devices if st.is_new_device),
        "os_update_events": len(os_updates),
        "os_update_days": len({ev.minute // 1440 for ev in os_updates}),
        "newest_os_share": _newest_os_share_by_day(world.people, schedule, settings.n_days),
    }


def _largest_daily_rise(shares: list[float]) -> float:
    """
    WHAT: Returns the largest one-day increase in a daily share series.
    WHY: A concentrated wave shows up as a large single-day jump; a gradual wave does not.

    Args:
        shares: Daily shares between 0 and 1.

    Returns:
        Largest day-over-day increase (0.0 for fewer than two days).

    Example:
        >>> round(_largest_daily_rise([0.2, 0.25, 0.5, 0.5]), 2)
        0.25
    """
    return max((shares[d] - shares[d - 1] for d in range(1, len(shares))), default=0.0)


async def run_os_wave_check(workspace: Any) -> CheckOutcome:
    """
    WHAT: Compares the base run with the os-update-wave layer: OS adoption over time and new-device flags (R5).
    The layer must move more phones to the newest OS and in sharper jumps than the base run, and new-device
    flags must stay within a four-sigma Poisson bound of the base: |layer - base| <= 4 x sqrt(max(base, 1)).

    WHY: The layer exaggerates OS update waves (short waves reaching 90-100% of eligible phones); issue R5 says
    an OS update is not a new device, so faster updates must not make new-device flags jump.

    Args:
        workspace: Optional showcase execution workspace environment container.

    Returns:
        CheckOutcome with both new-device counts and the bound, both adoption series, and the adoption chart.

    Example:
        >>> import asyncio
        >>> outcome = asyncio.run(run_os_wave_check(None))
        >>> outcome.key_number.startswith("new-device flags ")
        True
    """
    settings = FastRunSettings(seed=101, n_people=60, n_shops=12, n_days=61, range_name="training")
    base_cfgs = read_world_configs()
    layered_cfgs, _ = apply_layers(base_cfgs, [read_layer(OS_WAVE_LAYER_PATH)])
    base, stress = _os_wave_run(base_cfgs, settings), _os_wave_run(layered_cfgs, settings)
    bound = 4.0 * math.sqrt(max(base["new_device_flags"], 1))
    diff = abs(stress["new_device_flags"] - base["new_device_flags"])
    assert diff <= bound, f"R5: new-device flags {stress['new_device_flags']} vs {base['new_device_flags']}"
    base_rise = _largest_daily_rise(base["newest_os_share"])
    stress_rise = _largest_daily_rise(stress["newest_os_share"])
    assert stress["newest_os_share"][-1] > base["newest_os_share"][-1], "Layer did not raise final OS adoption"
    assert stress_rise > base_rise, "Layer waves are not more concentrated than base waves"
    key = (
        f"new-device flags {stress['new_device_flags']} vs {base['new_device_flags']} (bound +/-{bound:.1f}); "
        f"newest OS {base['newest_os_share'][0]:.1%} to {base['newest_os_share'][-1]:.1%} base vs "
        f"{stress['newest_os_share'][-1]:.1%} with layer; "
        f"largest one-day rise {base_rise * 100:.1f} vs {stress_rise * 100:.1f} points"
    )
    chart = adoption_curve(list(range(settings.n_days)), stress["newest_os_share"], base["newest_os_share"])
    evidence = {"base": base, "layer": stress, "r5_bound": bound, "r5_difference": diff,
                "largest_daily_rise_base": base_rise, "largest_daily_rise_layer": stress_rise, "chart_svg": chart}
    return CheckOutcome(key_number=key, evidence=evidence)
