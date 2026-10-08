# File: showcase/__main__.py
# Purpose: The command line for showcases.
#   python -m showcase --level 7 [--quick] [--allow-not-run]
#   python -m showcase --all [--quick] [--allow-not-run]
# Exit codes: 0 = everything passed, 1 = a check failed (or was NOT RUN without --allow-not-run), 2 = no such level.
import argparse
import asyncio
import sys
from pathlib import Path
from types import ModuleType

from showcase.framework import CheckCoverageError, decide_exit_code, plain_ascii, print_checklist, run_level_module
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
    return parser.parse_args(argv)


def run_one_level(module: ModuleType, quick: bool, allow_not_run: bool) -> int:
    """Run one level, print its checklist, write its report, and return its exit code."""
    # Imported here so that commands which never run a level (such as an unknown level number) do not need a
    # database setting at all.
    from showcase.isolation import cleanup_workspace, create_workspace

    out_dir: Path = REPORTS_DIR / level_folder_name(module.LEVEL_NUMBER)
    try:
        result = asyncio.run(run_level_module(module, quick, allow_not_run, create_workspace, out_dir,
                                              cleanup_workspace))
    except CheckCoverageError as error:
        print(plain_ascii(f"FAIL: level {module.LEVEL_NUMBER} breaks the coverage rules: {error}"))
        return 1
    print_checklist(result)
    write_level_report(result, out_dir)
    print(plain_ascii(f"Report written to {out_dir / 'index.html'}"))
    return decide_exit_code(result, allow_not_run)


def refresh_summary() -> None:
    """Rebuild reports/showcase/index.html from whatever results exist on disk."""
    rows = build_summary_rows(discover_level_modules())
    write_summary_page(rows, REPORTS_DIR / "index.html")


def run_one_by_number(level_number: int, quick: bool, allow_not_run: bool) -> int:
    """Handle --level N."""
    modules = discover_level_modules()
    if level_number not in modules:
        print(f"No showcase exists for level {level_number}")
        return 2
    exit_code = run_one_level(modules[level_number], quick, allow_not_run)
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
    if args.all:
        return run_all(args.quick, args.allow_not_run)
    return run_one_by_number(args.level, args.quick, args.allow_not_run)


if __name__ == "__main__":
    sys.exit(main())
