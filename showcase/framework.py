# File: showcase/framework.py
# Purpose: The shared machinery that runs a level's showcase: the checks, the PASS / FAIL / NOT RUN rules, the
#          exit code and the terminal checklist. It knows nothing about any one level.
#
# A level file (showcase/level_NN.py) must define:
#   LEVEL_NUMBER (int), LEVEL_TITLE (str), WHAT_YOU_SAW (str), NEEDS_DATABASE (bool),
#   find_missing_preconditions() -> list[str]   (empty list = everything needed is there),
#   get_checks(quick: bool) -> list[CheckDefinition].
#
# A level that is built in parts (for example Level 12a, 12b, ...) tags each check with part="a", "b", ...
# Running one part (--part a) keeps only that part's checks, AFTER the coverage rules were checked on the whole
# level, because one part on its own may not hold a normal, two edge and a negative check.
# TERM: part = one letter naming a slice of a level that is built and reviewed on its own, e.g. "a" for Level 12a.
import inspect
import subprocess
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

# Every showcase uses this seed for anything random, so two runs behave the same way.
FIXED_SEED = 7

PASS = "PASS"
FAIL = "FAIL"
NOT_RUN = "NOT RUN"

VALID_KINDS = ("normal", "edge", "negative", "consistency")

# The coverage rules every level must meet.
MIN_NORMAL_CHECKS = 1
MIN_EDGE_CHECKS = 2
MIN_NEGATIVE_CHECKS = 1


class NotRun(Exception):
    """Raised by a check (or precondition) to say: this could not run, and here is why."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class CheckCoverageError(Exception):
    """Raised when a level's list of checks breaks the coverage rules. The framework never weakens the rules."""


class NoChecksForPart(Exception):
    """Raised when --part names a part that no check of the level belongs to. A run never passes with zero checks."""


@dataclass
class CheckOutcome:
    """What a check hands back when it passes."""

    key_number: str
    evidence: dict[str, Any] = field(default_factory=dict)
    chart: dict[str, Any] | None = None


@dataclass
class CheckDefinition:
    """One check: what it is called, what it proves, and the async function that does it."""

    name: str
    kind: str
    proves: str
    run: Callable[[Any], Awaitable[CheckOutcome]]
    likely_cause: str
    where_to_look: str
    # Which part of the level this check belongs to ("a", "b", ...); None for levels that are not split.
    part: str | None = None


@dataclass
class CheckResult:
    """What happened when one check ran."""

    name: str
    kind: str
    proves: str
    status: str
    key_number: str
    explanation: str
    evidence: dict[str, Any]
    chart: dict[str, Any] | None
    seconds: float


@dataclass
class LevelResult:
    """What happened when a whole level ran."""

    level_number: int
    title: str
    mode: str
    what_you_saw: str
    check_results: list[CheckResult]
    total_seconds: float
    code_version: str
    finished_at: str
    overall_status: str
    # The part that was run, or None when the whole level was run.
    part: str | None = None


def validate_checks(checks: list[CheckDefinition]) -> None:
    """Raise CheckCoverageError unless the checks meet the coverage rules (see the constants above)."""
    # Every check needs a known kind and a real "proves" sentence.
    for check in checks:
        if check.kind not in VALID_KINDS:
            raise CheckCoverageError(f"check '{check.name}' has unknown kind '{check.kind}'")
        if not check.proves.strip():
            raise CheckCoverageError(f"check '{check.name}' has an empty 'proves' sentence")

    # Names must be unique, because reports and evidence files are named after them.
    names = [check.name for check in checks]
    for name in names:
        if names.count(name) > 1:
            raise CheckCoverageError(f"duplicate check name '{name}'")

    # Count each kind and compare with the minimums.
    kinds = [check.kind for check in checks]
    if kinds.count("normal") < MIN_NORMAL_CHECKS:
        raise CheckCoverageError(f"need at least {MIN_NORMAL_CHECKS} normal check")
    if kinds.count("edge") < MIN_EDGE_CHECKS:
        raise CheckCoverageError(f"need at least {MIN_EDGE_CHECKS} edge checks, found {kinds.count('edge')}")
    if kinds.count("negative") < MIN_NEGATIVE_CHECKS:
        raise CheckCoverageError(f"need at least {MIN_NEGATIVE_CHECKS} negative check")


def select_part_checks(checks: list[CheckDefinition], part: str | None) -> list[CheckDefinition]:
    """
    WHAT: Keeps only the checks whose part tag equals the requested part, in their original order.
    When no part is requested (None), every check is kept unchanged, so whole-level runs behave as before.
    If a part is requested but no check carries that tag, it raises NoChecksForPart instead of returning nothing.

    WHY: A level built in pieces (12a, 12b, ...) needs to show one piece at a time while it is being reviewed.
    The filter lives here, in the shared framework, so level files only tag their checks and never filter.
    Refusing an empty selection matters because a run with zero checks would otherwise report PASSED.

    Args:
        checks: The level's full, already validated list of checks, in run order.
        part: One lowercase letter such as "a", or None to keep every check.

    Returns:
        The checks to run: all of them when part is None, otherwise only those tagged with that part.

    Example:
        Four checks tagged a, b, a, b with part="a" keep checks 1 and 3 (2 of 4); with part=None all 4 stay.
        With part="z" and no check tagged z, NoChecksForPart("z") is raised.
    """
    # No part requested: the whole level runs, exactly as it did before parts existed.
    if part is None:
        return checks
    # Keep the order of the level file, so check numbers in the report still read 1, 3, ... in sequence.
    selected = [check for check in checks if check.part == part]
    # Never hand back an empty list: the caller would run nothing and call that a pass.
    if not selected:
        raise NoChecksForPart(part)
    return selected


def get_code_version() -> str:
    """Short git commit hash, with '-dirty' if there are uncommitted changes, or 'unknown' without git."""
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True)
        status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    version = commit.stdout.strip() or "unknown"
    if status.stdout.strip():
        version += "-dirty"
    return version


async def find_preconditions(module: Any) -> list[str]:
    """Ask the level what is missing. The level's function may be plain or async."""
    answer = module.find_missing_preconditions()
    if inspect.isawaitable(answer):
        answer = await answer
    return list(answer)


def make_not_run_result(check: CheckDefinition, reason: str) -> CheckResult:
    """Build the result for a check that could not run."""
    return CheckResult(check.name, check.kind, check.proves, NOT_RUN, "-", reason, {}, None, 0.0)


def make_fail_result(check: CheckDefinition, error: Exception, seconds: float) -> CheckResult:
    """Build the result for a check that raised an unexpected error: a short line, the full traceback as evidence."""
    # One short line for people. The long traceback goes into the evidence file only.
    explanation = (f"{check.likely_cause} Look at: {check.where_to_look}. "
                   f"Error: {type(error).__name__}: {error}").replace("\n", " ")
    evidence = {"traceback": traceback.format_exception(type(error), error, error.__traceback__)}
    return CheckResult(check.name, check.kind, check.proves, FAIL, "-", explanation, evidence, None, seconds)


async def run_one_check(check: CheckDefinition, workspace: Any) -> CheckResult:
    """Run a single check and turn whatever happens into a CheckResult. This function never raises."""
    started = time.perf_counter()
    try:
        outcome = await check.run(workspace)
    except NotRun as skipped:
        return make_not_run_result(check, skipped.reason)
    except Exception as error:  # noqa: BLE001 - a broken check must be reported, not crash the runner
        return make_fail_result(check, error, time.perf_counter() - started)
    seconds = time.perf_counter() - started
    return CheckResult(check.name, check.kind, check.proves, PASS, outcome.key_number, "", outcome.evidence,
                       outcome.chart, seconds)


def decide_overall_status(results: list[CheckResult], allow_not_run: bool) -> str:
    """PASSED only if every check passed (NOT RUN is accepted only when allow_not_run is True)."""
    for result in results:
        if result.status == FAIL:
            return "FAILED"
        if result.status == NOT_RUN and not allow_not_run:
            return "FAILED"
    return "PASSED"


def decide_exit_code(level_result: LevelResult, allow_not_run: bool) -> int:
    """0 only when the level passed under the rules above, otherwise 1."""
    if decide_overall_status(level_result.check_results, allow_not_run) == "PASSED":
        return 0
    return 1


async def run_checks_in_workspace(
    module: Any,
    checks: list[CheckDefinition],
    workspace_factory: Callable[[], Awaitable[Any]],
    workspace_cleanup: Callable[[Any], Awaitable[None]] | None,
) -> list[CheckResult]:
    """Create the workspace (if the level needs one), run the checks one at a time, and always clean up."""
    workspace = None
    try:
        if module.NEEDS_DATABASE:
            workspace = await workspace_factory()
        results = []
        for check in checks:
            results.append(await run_one_check(check, workspace))
        return results
    finally:
        # Runs even if something blew up, so no throwaway schema or temp folder is left behind.
        if workspace is not None and workspace_cleanup is not None:
            await workspace_cleanup(workspace)


async def run_level_module(
    module: Any,
    quick: bool,
    allow_not_run: bool,
    workspace_factory: Callable[[], Awaitable[Any]],
    out_dir: Path,
    workspace_cleanup: Callable[[Any], Awaitable[None]] | None = None,
    part: str | None = None,
) -> LevelResult:
    """
    Run every check of one level, in order, and return the result.

    workspace_factory and workspace_cleanup are parameters so unit tests can pass fakes. out_dir is where this
    level's reports go; the run itself writes nothing there (the report writer does).
    part, when given, keeps only that part's checks. The coverage rules are checked on the WHOLE level first,
    because a single part may legitimately lack an edge or negative check of its own.
    """
    started = time.perf_counter()
    checks = module.get_checks(quick)
    validate_checks(checks)
    # Filter only after validation, so the coverage minimum always applies to the level as a whole.
    checks = select_part_checks(checks, part)

    missing = await find_preconditions(module)
    if missing:
        # Something the level needs is not there. Say so for EVERY check, never silently.
        reason = "; ".join(missing)
        results = [make_not_run_result(check, reason) for check in checks]
    else:
        results = await run_checks_in_workspace(module, checks, workspace_factory, workspace_cleanup)

    return LevelResult(
        level_number=module.LEVEL_NUMBER,
        title=module.LEVEL_TITLE,
        mode="quick" if quick else "full",
        what_you_saw=module.WHAT_YOU_SAW,
        check_results=results,
        total_seconds=time.perf_counter() - started,
        code_version=get_code_version(),
        finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        overall_status=decide_overall_status(results, allow_not_run),
        part=part,
    )


def plain_ascii(text: str) -> str:
    """Replace any non-ASCII character with '?', because some Windows consoles crash on fancy symbols."""
    return text.encode("ascii", errors="replace").decode("ascii")


def print_checklist(level_result: LevelResult) -> None:
    """Print the level's result to the terminal: header, one block per check, the story, then totals."""
    lines = [f"Level {level_result.level_number}: {level_result.title}  (mode: {level_result.mode})", ""]
    for result in level_result.check_results:
        lines.append(f"{result.name}  [{result.kind}]  {result.status}  {result.key_number}")
        lines.append(f"    proves: {result.proves}")
        if result.status != PASS and result.explanation:
            lines.append(f"    why: {result.explanation}")
    lines.append("")
    lines.append("What you just saw:")
    lines.append(level_result.what_you_saw)
    lines.append("")
    lines.append(f"Total time: {level_result.total_seconds:.1f}s   Overall: {level_result.overall_status}")
    print(plain_ascii("\n".join(lines)))
