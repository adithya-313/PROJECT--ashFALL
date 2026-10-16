# File: showcase/__main__.py
# Purpose: The command line for showcases.
#   python -m showcase --level 7 [--quick] [--allow-not-run]
#   python -m showcase --level 12 --part a [--quick] [--allow-not-run]
#   python -m showcase --all [--quick] [--allow-not-run]
# Exit codes: 0 = everything passed, 1 = a check failed (or was NOT RUN without --allow-not-run),
#             2 = no such level, a bad --part value, or a part with no checks.
# A part run writes its report to reports/showcase/level_NN/part_X/, so the --all summary (which reads only
# reports/showcase/level_NN/result.json) never mixes a part result with a whole-level result.
# TERM: part = one letter naming a slice of a level that is built and reviewed on its own, e.g. "a" for Level 12a.
import argparse
import asyncio
import re
import sys
from pathlib import Path
from types import ModuleType

from showcase.framework import (
    CheckCoverageError,
    NoChecksForPart,
    decide_exit_code,
    plain_ascii,
    print_checklist,
    run_level_module,
)
from showcase.reporting import write_level_report
from showcase.summary import (
    REPORTS_DIR,
    build_summary_rows,
    discover_level_modules,
    level_folder_name,
    write_summary_page,
)


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Read the command line. Either --level N or --all is required."""
    parser = argparse.ArgumentParser(prog="python -m showcase", description="Run level showcases.")
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--level", type=int, help="run the showcase of one level, for example 7")
    which.add_argument("--all", action="store_true", help="run every level that has a showcase")
    parser.add_argument("--quick", action="store_true", help="use the faster, smaller version of each check")
    parser.add_argument("--allow-not-run", action="store_true",
                        help="accept NOT RUN checks (for example when a service is missing) as success")
    parser.add_argument("--part", help="with --level only: run just one part of a split level, for example a")
    return parser.parse_args(argv)


# A part is exactly one lowercase letter, such as "a" for Level 12a.
PART_PATTERN = re.compile(r"^[a-z]$")


def check_part_argument(level: int | None, part: str | None) -> str | None:
    """
    WHAT: Checks the --part value before any level runs and returns a plain error sentence when it is not usable.
    It returns None when --part was not given, or when it was given together with --level and is one lowercase
    letter. Anything else (missing --level, an empty value, "ab", "A", "1") produces an error sentence.

    WHY: A bad --part value must stop the command with exit code 2 and a clear message, not silently run the whole
    level or nothing. Checking the value up front keeps that rule in one small place that is easy to test.

    Args:
        level: The --level number, or None when --all was used instead.
        part: The --part value exactly as typed, or None when --part was not given.

    Returns:
        None when the arguments are fine, otherwise one sentence explaining what is wrong.

    Example:
        check_part_argument(12, "a") gives None; check_part_argument(12, None) gives None.
        check_part_argument(None, "a") gives "--part can only be used together with --level".
        check_part_argument(12, "ab") gives "--part must be one lowercase letter such as a, got 'ab'".
    """
    # No --part at all: nothing to check, the run behaves exactly as before.
    if part is None:
        return None
    # A part only makes sense inside one level; --all has no single level to split.
    if level is None:
        return "--part can only be used together with --level"
    # Only a single lowercase letter names a part.
    if not PART_PATTERN.match(part):
        return f"--part must be one lowercase letter such as a, got '{part}'"
    return None


def report_folder(level_number: int, part: str | None) -> Path:
    """
    WHAT: Works out the folder a level's report is written to.
    A whole-level run uses reports/showcase/level_NN/ as always; a part run uses a sub-folder part_X/ inside it.

    WHY: Part results must never overwrite or be mistaken for the whole-level result. The --all summary only reads
    reports/showcase/level_NN/result.json, so a sub-folder keeps part results out of it.

    Args:
        level_number: The level number, for example 12.
        part: One lowercase letter such as "a", or None for a whole-level run.

    Returns:
        The folder path for this run's index.html, result.json and evidence files.

    Example:
        report_folder(7, None) ends in reports/showcase/level_07.
        report_folder(12, "a") ends in reports/showcase/level_12/part_a.
    """
    # Every level keeps its usual two-digit folder.
    level_dir = REPORTS_DIR / level_folder_name(level_number)
    # A part run gets its own sub-folder inside it.
    return level_dir if part is None else level_dir / f"part_{part}"


def run_one_level(module: ModuleType, quick: bool, allow_not_run: bool, part: str | None = None) -> int:
    """Run one level (or one part of it), print its checklist, write its report, and return its exit code."""
    # Imported here so that commands which never run a level (such as an unknown level number) do not need a
    # database setting at all.
    from showcase.isolation import cleanup_workspace, create_workspace

    out_dir: Path = report_folder(module.LEVEL_NUMBER, part)
    try:
        result = asyncio.run(run_level_module(module, quick, allow_not_run, create_workspace, out_dir,
                                              cleanup_workspace, part=part))
    except CheckCoverageError as error:
        print(plain_ascii(f"FAIL: level {module.LEVEL_NUMBER} breaks the coverage rules: {error}"))
        return 1
    except NoChecksForPart:
        print(f"Level {module.LEVEL_NUMBER} has no checks for part {part}")
        return 2
    print_checklist(result)
    write_level_report(result, out_dir)
    if hasattr(module, "write_custom_page"):
        module.write_custom_page(result, out_dir)
    print(plain_ascii(f"Report written to {out_dir / 'index.html'}"))
    return decide_exit_code(result, allow_not_run)


def refresh_summary() -> None:
    """Rebuild reports/showcase/index.html from whatever results exist on disk."""
    rows = build_summary_rows(discover_level_modules())
    write_summary_page(rows, REPORTS_DIR / "index.html")


def run_one_by_number(level_number: int, quick: bool, allow_not_run: bool, part: str | None = None) -> int:
    """Handle --level N (and --part X when given)."""
    modules = discover_level_modules()
    if level_number not in modules:
        print(f"No showcase exists for level {level_number}")
        return 2
    exit_code = run_one_level(modules[level_number], quick, allow_not_run, part)
    refresh_summary()
    return exit_code


def run_all(quick: bool, allow_not_run: bool) -> int:
    """Handle --all: run every discovered level in numeric order; fail if any level fails."""
    modules = discover_level_modules()
    if not modules:
        print("No level showcases found")
        return 0
    worst_exit_code = 0
    for level_number in sorted(modules):
        exit_code = run_one_level(modules[level_number], quick, allow_not_run)
        worst_exit_code = max(worst_exit_code, exit_code)
    refresh_summary()
    return worst_exit_code


def main(argv: list[str] | None = None) -> int:
    """Entry point. argv is a parameter so tests can call this directly."""
    args = parse_args(sys.argv[1:] if argv is None else argv)
    # Refuse a bad --part before anything runs or any report is written.
    part_error = check_part_argument(args.level, args.part)
    if part_error is not None:
        print(part_error)
        return 2
    if args.all:
        return run_all(args.quick, args.allow_not_run)
    return run_one_by_number(args.level, args.quick, args.allow_not_run, args.part)


if __name__ == "__main__":
    sys.exit(main())
