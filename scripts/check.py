# File: scripts/check.py
# Purpose: Run on your own machine the same checks the CI jobs in .github/workflows/ci.yml run, so a pull request
#          does not fail on something you could have found locally. This is what `make check` calls.
#
#   python scripts/check.py                 all steps
#   python scripts/check.py --no-docker     only the steps that need no Docker
#   python scripts/check.py --only "ruff (api, core)"    one step by its exact name
#
# DECISION: locally this script NEVER starts, stops or restarts containers. Where CI boots the stack itself, this
# script checks the stack that is ALREADY running. If it is not running, the Docker steps report NOT RUN with the
# command to start it, and that counts as a failure. (The Level 7 showcase steps do stop and start the Redpanda broker
# on purpose, as part of what they prove; that is the showcase's own behaviour, not this script's.)
import argparse
import json
import re
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable
START_STACK = "docker compose up -d --wait"
SERVICES_WITH_HEALTHCHECK = ["postgres", "redis", "redpanda"]


@dataclass
class Step:
    """One check: its name, the CI job it mirrors, what to run, and what counts as success."""

    name: str
    ci_job: str
    command: list[str] = field(default_factory=list)
    check: Callable[[], tuple[bool, str]] | None = None  # used instead of a command for the stack checks
    needs_docker: bool = False
    expect_exit: int = 0
    must_contain: str = ""
    must_not_contain: str = ""
    is_showcase: bool = False  # True for the level showcase steps, which --skip-showcases leaves out


def ruff_step(name: str, job: str, folders: list[str]) -> Step:
    """A ruff step for the given folders."""
    return Step(name, job, [PYTHON, "-m", "ruff", "check", *folders])


def mypy_step(name: str, job: str, folders: list[str]) -> Step:
    """A mypy step for the given folders (same flags as CI)."""
    return Step(name, job, [PYTHON, "-m", "mypy", *folders, "--ignore-missing-imports", "--no-strict-optional"])


def check_stack_is_running() -> tuple[bool, str]:
    """The services with a healthcheck must be healthy, and http://localhost:8000/health must answer 200."""
    command = ["docker", "compose", "ps", "--format", "json"]
    result = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True)
    health = {}
    for line in result.stdout.splitlines():
        row = json.loads(line)
        health[row["Service"]] = row.get("Health", "")
    problems = [f"{name} is not healthy" for name in SERVICES_WITH_HEALTHCHECK if health.get(name) != "healthy"]
    try:
        status = urllib.request.urlopen("http://localhost:8000/health", timeout=5).status
    except OSError as error:
        status = 0
        problems.append(f"/health did not answer: {error}")
    if status not in (0, 200):
        problems.append(f"/health returned {status}")
    if problems:
        return False, "NOT RUN: " + "; ".join(problems) + f". Start the stack with: {START_STACK}"
    return True, "postgres, redis, redpanda healthy; /health returned 200"


def check_health_contract() -> tuple[bool, str]:
    """The /health answer must match tests/contract/health_response.schema.json (the same check as the CI boot job)."""
    try:
        import jsonschema
    except ImportError:
        return False, "NOT RUN: install jsonschema (pip install jsonschema)"
    try:
        body = json.loads(urllib.request.urlopen("http://localhost:8000/health", timeout=5).read())
    except OSError as error:
        return False, f"NOT RUN: /health did not answer ({error}). Start the stack with: {START_STACK}"
    schema = json.loads((REPO_ROOT / "tests" / "contract" / "health_response.schema.json").read_text(encoding="utf-8"))
    try:
        jsonschema.validate(instance=body, schema=schema)
    except jsonschema.ValidationError as error:
        return False, f"Contract test FAILED: {error.message}"
    return True, "response matches schema"


def level_numbers() -> list[int]:
    """The level showcases that exist (showcase/level_NN.py), in numeric order."""
    names = [path.name for path in (REPO_ROOT / "showcase").glob("level_*.py")]
    return sorted(int(m.group(1)) for n in names if (m := re.match(r"^level_(\d\d)\.py$", n)))


def build_steps() -> list[Step]:
    """Every step, in order. The commands mirror ci.yml; tests/unit/test_check_matches_ci.py guards the job names."""
    steps = [
        ruff_step("ruff (api, core)", "compose-boot", ["api/", "core/"]),
        mypy_step("mypy (api, core)", "compose-boot", ["api/", "core/"]),
        Step("pandera schema gate", "schema-validation", [PYTHON, "tests/schema/test_enriched_schema.py"]),
        Step("model quality gate", "model-quality-gate", [PYTHON, "scripts/ci_quality_gate.py"]),
        Step("degraded model is rejected", "model-quality-gate",
             [PYTHON, "scripts/ci_quality_gate.py", "--metrics", "tests/fixtures/metrics_degraded.json"],
             expect_exit=1, must_contain="FAIL PR-AUC"),
        ruff_step("ruff (showcase, api, core)", "level-showcase", ["showcase", "api", "core"]),
        mypy_step("mypy (showcase, api, core)", "level-showcase", ["showcase", "api", "core"]),
        Step("fast tests", "pytest-fast",
             [PYTHON, "-m", "pytest", "-q", "-rs", "-p", "no:cacheprovider", "-m", "not needs_stack", "tests/unit",
              "tests/contract", "tests/property", "tests/schema", "tests/baselines"]),
        Step("stack is running and healthy", "compose-boot", check=check_stack_is_running, needs_docker=True),
        Step("/health contract", "compose-boot", check=check_health_contract, needs_docker=True),
        Step("integration tests", "level-showcase",
             [PYTHON, "-m", "pytest", "-q", "-rs", "-p", "no:cacheprovider",
              "tests/integration/test_atomic_writer.py", "tests/integration/test_migration_runner.py",
              "tests/integration/test_showcase_isolation.py", "tests/integration/test_showcase_kafka_tools.py",
              "tests/contract/test_orjson_in_response.py", "tests/integration/test_pool_initialization.py"],
             needs_docker=True, must_not_contain="SKIPPED"),
    ]
    for number in level_numbers():
        steps.append(Step(f"showcase level {number} (quick)", "level-showcase",
                          [PYTHON, "-m", "showcase", "--level", str(number), "--quick"], needs_docker=True,
                          is_showcase=True))
    return steps


def parse_pinned_version(requirements_text: str, tool: str) -> str | None:
    """The version after 'tool==' at the start of a line in requirements.txt text, or None."""
    match = re.search(rf"^{re.escape(tool)}==([0-9][0-9A-Za-z.]*)", requirements_text, re.MULTILINE)
    return match.group(1) if match else None


def parse_installed_version(version_output: str) -> str | None:
    """The first version-looking number (such as 0.15.12) in the output of `<tool> --version`, or None."""
    match = re.search(r"[0-9]+(?:\.[0-9]+)+", version_output)
    return match.group(0) if match else None


def version_warning(tool: str, pinned: str | None, installed: str | None) -> str | None:
    """A warning text when the installed version differs from the CI pin (or cannot be read), otherwise None."""
    if pinned is None:
        return f"WARNING: no pinned {tool} version found in requirements.txt"
    if installed is None:
        return f"WARNING: could not read the installed {tool} version (CI uses {pinned})"
    if installed != pinned:
        return f"WARNING: {tool} {installed} is installed but CI uses {pinned}; results may differ"
    return None


def print_tool_versions() -> None:
    """Print installed versus CI-pinned versions of ruff and mypy. A difference is only a warning, never a failure."""
    requirements = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")
    for tool in ("ruff", "mypy"):
        try:
            result = subprocess.run([PYTHON, "-m", tool, "--version"], capture_output=True, text=True, timeout=60)
            installed = parse_installed_version(result.stdout + result.stderr)
        except (OSError, subprocess.TimeoutExpired):
            installed = None
        pinned = parse_pinned_version(requirements, tool)
        print(f"{tool}: installed {installed or 'unknown'}, CI pins {pinned or 'unknown'}")
        warning = version_warning(tool, pinned, installed)
        if warning:
            print(warning)
    print()


def run_step(step: Step) -> tuple[bool, str]:
    """Run one step and return (passed, output)."""
    if step.check is not None:
        return step.check()
    result = subprocess.run(step.command, cwd=REPO_ROOT, capture_output=True, text=True, timeout=1800)
    output = result.stdout + result.stderr
    ok = result.returncode == step.expect_exit
    ok = ok and (step.must_contain in output) and not (step.must_not_contain and step.must_not_contain in output)
    return ok, output


def main(argv: list[str] | None = None) -> int:
    """Run the chosen steps, print one line each, show the tail of any failure, and return 0 only if all passed."""
    parser = argparse.ArgumentParser(description="Run the CI checks locally.")
    parser.add_argument("--no-docker", action="store_true", help="skip the steps that need Docker")
    parser.add_argument("--only", help="run only the step with this exact name")
    parser.add_argument("--skip-showcases", action="store_true", help="skip only the level showcase steps")
    args = parser.parse_args(argv)
    steps = build_steps()
    if args.only:
        steps = [step for step in steps if step.name == args.only]
        if not steps:
            print("No step with that name. Names: " + ", ".join(step.name for step in build_steps()))
            return 2
    print_tool_versions()
    failures = 0
    for step in steps:
        if args.skip_showcases and step.is_showcase:
            print(f"SKIPPED  {'-':>6}  {step.name}  (skipped by --skip-showcases)")
            continue
        if args.no_docker and step.needs_docker:
            print(f"SKIPPED  {'-':>6}  {step.name}  (needs Docker; skipped by --no-docker)")
            continue
        started = time.perf_counter()
        passed, output = run_step(step)
        print(f"{'PASS' if passed else 'FAIL':8} {time.perf_counter() - started:5.1f}s  {step.name}  [{step.ci_job}]")
        if not passed:
            failures += 1
            print("    " + "\n    ".join(output.strip().splitlines()[-15:]))
    print(f"\n{failures} step(s) failed." if failures else "\nAll steps passed.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
