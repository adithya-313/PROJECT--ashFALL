# File: showcase/level_08.py
# Purpose: The Level 8 showcase ("Safety net"). Five checks prove that the protections added in Level 8 really work:
#          the checks all pass, a changed API field is caught, files can be restored from the DVC remote, the stack
#          works with the outgoing network blocked, and Level 7 still passes.
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

import orjson

from showcase.framework import CheckDefinition, CheckOutcome, NotRun
from showcase.level08_docker import (
    apply_migrations,
    build_sniffer_image,
    cleanup_copy,
    container_state,
    list_services,
    observe_copy,
    project_containers,
    run,
    run_compose,
    run_control_a,
    run_control_b,
    start_copy,
    verify_network_is_closed,
)
from showcase.level08_helpers import services_for_mode
from showcase.level08_support import (
    PYTHON,
    REPO_ROOT,
    RESPONSE_FILE,
    copy_tracked_files,
    count_differences,
    dvc_round_trip,
    hide,
    last_lines,
    parse_check_summary,
    remove_tree,
    rename_field_in_text,
    run_command,
)

LEVEL_NUMBER = 8
LEVEL_TITLE = "Safety net"
NEEDS_DATABASE = False  # the checks make their own temporary folders and containers; no database schema is needed
WHAT_YOU_SAW = (
    "The safety net held. Every check the project runs in CI also passes on this machine, and the saved baselines "
    "match. When a field of the API response was renamed in a temporary copy, the baseline test failed with a clear "
    "message, and passed again once the change was undone. Files stored with DVC were deleted, together with their "
    "cache, and restored from the remote with identical SHA-256 hashes. A second copy of the stack ran on a network "
    "with no way out and made no outgoing attempts, while two control experiments proved the counter can see such "
    "attempts. Last, the Level 7 showcase still passed. Quick mode (used in CI) restores generated files instead of "
    "the real artifacts, watches only Postgres, Redis and Redpanda for 20 seconds, and runs a smaller set of tests "
    "for the first check; full mode uses the real artifacts and the whole stack."
)

BASELINE_TESTS = ["tests/baselines", "tests/unit/test_feature_switches.py", "tests/unit/test_check_matches_ci.py"]


def docker_is_usable() -> bool:
    """True when the docker command works."""
    return run(["docker", "info", "--format", "{{.ServerVersion}}"]).returncode == 0


async def find_missing_preconditions() -> list[str]:
    """What is missing, in plain English (empty means we can run)."""
    problems = []
    if not docker_is_usable():
        problems.append("Docker is not usable - start Docker Desktop and check `docker info`")
    if shutil.which("dvc") is None:
        problems.append("dvc is not installed - run: pip install -r requirements-showcase.txt")
    return problems


def running_services() -> set[str]:
    """Compose service names that have a running container (any project)."""
    fmt = "{{.Label \"com.docker.compose.service\"}}"
    result = run(["docker", "ps", "--filter", "status=running", "--format", fmt])
    return set(result.stdout.split())


def local_remote_folder() -> str | None:
    """The folder of the DVC remote from `dvc remote list --local`, or None when there is none."""
    code, output = run_command(["dvc", "remote", "list", "--local"], REPO_ROOT, 60)
    if code != 0 or not output.strip():
        return None
    match = re.match(r"^\S+\s+(.*?)(?:\s+\(default\))?\s*$", output.strip().splitlines()[0])
    return match.group(1) if match else None


def require_full_mode_stack() -> None:
    """Full mode needs the real stack running; otherwise the check is NOT RUN, with the command to fix it."""
    missing = {"postgres", "redis", "redpanda", "api"} - running_services()
    if missing:
        names = ", ".join(sorted(missing))
        raise NotRun(f"the stack is not running (missing: {names}) - run: docker compose up -d --wait")


# ---------------------------------------------------------------------------
# Check 1: make check passes and the baselines match
# ---------------------------------------------------------------------------

def make_check_one(quick: bool) -> Any:
    """Build check 1 for quick or full mode."""
    async def run_check(workspace: Any) -> CheckOutcome:
        """Run the project's own checks and require success."""
        if quick:
            command = [PYTHON, "-m", "pytest", "-q", "-p", "no:cacheprovider", *BASELINE_TESTS]
            timeout = 300
        else:
            require_full_mode_stack()
            command, timeout = [PYTHON, "scripts/check.py", "--skip-showcases"], 600
        code, output = run_command(command, REPO_ROOT, timeout)
        assert code == 0, f"exit code {code}: {last_lines(output, 8)}"
        if quick:
            match = re.search(r"(\d+) passed", output)
            key = f"{match.group(1) if match else '?'} baseline tests passed"
        else:
            passed, total = parse_check_summary(output)
            key = f"{passed} of {total} steps passed"
        evidence = {"command": " ".join(command[1:]), "output_tail": last_lines(output)}
        return CheckOutcome(key_number=key, evidence=evidence)
    return run_check


# ---------------------------------------------------------------------------
# Check 2: a changed response field is caught (red, then green)
# ---------------------------------------------------------------------------

def baseline_run(folder: Path) -> tuple[int, str]:
    """Run the baseline tests inside the temporary copy, importing the copy's code and not the real one."""
    command = [PYTHON, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/baselines"]
    return run_command(command, folder, 300, {"PYTHONPATH": str(folder), "PYTHONDONTWRITEBYTECODE": "1"})


def imported_api_path(folder: Path) -> str:
    """Where `import api` comes from when run inside the copy."""
    command = [PYTHON, "-c", "import api; print(api.__file__)"]
    return run_command(command, folder, 60, {"PYTHONPATH": str(folder)})[1].strip().splitlines()[-1]


async def run_check_two(workspace: Any) -> CheckOutcome:
    """Rename a response field in a temporary copy: the baseline test must fail, and pass again after the undo."""
    folder = Path(tempfile.mkdtemp(prefix="ashfall-level8-"))
    try:
        copy_tracked_files(folder)
        target = folder / RESPONSE_FILE
        original = target.read_bytes().decode("utf-8")  # bytes in, bytes out: line endings stay exactly as they were
        changed = rename_field_in_text(original)
        assert changed is not None, "could not apply the change: the field line was not found exactly once"
        target.write_bytes(changed.encode("utf-8"))
        api_path = imported_api_path(folder)
        assert Path(api_path).resolve().is_relative_to(folder.resolve()), f"the test imported {api_path}, not the copy"
        red_code, red_output = baseline_run(folder)
        assert red_code != 0, "the baseline tests still passed after the field was renamed"
        assert "public shape changed" in red_output, "failed for the wrong reason: " + last_lines(red_output, 6)
        target.write_bytes(original.encode("utf-8"))
        green_code, green_output = baseline_run(folder)
        assert green_code == 0, "the baseline tests still fail after undoing the change: " + last_lines(green_output, 6)
        differences = count_differences(red_output)
        return CheckOutcome(
            key_number=f"{differences} differences reported, then 0 after the undo",
            evidence={"api_imported_from_copy": True, "red_exit_code": red_code, "green_exit_code": green_code,
                      "red_output_tail": last_lines(red_output, 12)},
        )
    finally:
        remove_tree(folder)


# ---------------------------------------------------------------------------
# Check 3: restore from the DVC remote with matching hashes
# ---------------------------------------------------------------------------

def make_check_three(quick: bool) -> Any:
    """Build check 3 for quick (generated files) or full (also the real 7 artifacts) mode."""
    async def run_check(workspace: Any) -> CheckOutcome:
        """Do the generated-files round trip, and in full mode also the real restore check."""
        folder = Path(tempfile.mkdtemp(prefix="ashfall-level8-dvc-"))
        try:
            matching, problem = dvc_round_trip(folder)
            assert problem is None, str(problem)
            evidence: dict[str, Any] = {"generated_files_restored": matching}
            key = f"{matching} of {matching} generated files restored"
            if not quick:
                remote = local_remote_folder()
                if remote is None:
                    raise NotRun("no local DVC remote is configured - run: dvc remote add --local -d backup <folder>")
                command = [PYTHON, "scripts/verify_dvc_restore.py", "--dvc-remote-dir", remote]
                code, output = run_command(command, REPO_ROOT, 1500)
                output = hide(output, remote)  # the remote folder is private: never save it in the report
                assert code == 0, "the real restore check failed (no network?): " + last_lines(output, 8)
                verified = len(re.findall(r"\sOK\s*$", output, re.MULTILINE))
                evidence.update({"real_artifacts_verified": verified, "output_tail": last_lines(output, 14)})
                key = f"{verified} real files verified, plus {matching} generated"
            return CheckOutcome(key_number=key, evidence=evidence)
        finally:
            remove_tree(folder)
    return run_check


# ---------------------------------------------------------------------------
# Check 4: the stack runs with the outgoing network blocked and makes zero outbound attempts
# ---------------------------------------------------------------------------

def stack_problems(states: list[dict[str, str]]) -> list[str]:
    """Plain-English problems with the copy's containers (not running, restarted, or a health check not healthy)."""
    problems = []
    for state in states:
        name = state["container"]
        if state["status"] != "running" or state["restarts"] != "0":
            problems.append(f"{name} is {state['status']} with {state['restarts']} restarts")
        if state["health"] not in ("healthy", "none"):
            problems.append(f"{name} health is {state['health']}")
    return problems


def api_health_status() -> str:
    """Ask /health from INSIDE the copy's api container (nothing is published on the host)."""
    code = "import urllib.request;print(urllib.request.urlopen('http://localhost:8000/health', timeout=5).status)"
    return run_compose(["exec", "-T", "api", "python", "-c", code]).stdout.strip()


def start_and_prepare_copy(quick: bool) -> list[str]:
    """Clean up leftovers, build the sniffer, start the copy and (full mode) apply migrations. Returns the services."""
    cleanup_copy()
    problem = build_sniffer_image()
    if problem:
        raise NotRun(f"the sniffer image could not be built ({problem[:120]}) - check Docker and network access")
    services = services_for_mode(quick, list_services())
    problem = start_copy(services)
    assert problem is None, f"the copy did not start: {problem}"
    if not quick:
        problem = apply_migrations()
        assert problem is None, str(problem)
    return services


def make_check_four(quick: bool) -> Any:
    """Build check 4 for quick (3 services, 20 s) or full (whole stack, 60 s) mode."""
    seconds = 20 if quick else 60

    async def run_check(workspace: Any) -> CheckOutcome:
        """Start the copy on an internal network, watch every container, run the controls, and demand zero attempts."""
        if not quick:
            require_full_mode_stack()
        try:
            services = start_and_prepare_copy(quick)
            closed_problem = verify_network_is_closed()
            assert closed_problem is None, str(closed_problem)
            if not quick:
                assert api_health_status() == "200", "the api did not answer /health with 200 inside the copy"
            results = observe_copy(seconds)
            incomplete = [r.container for r in results if not r.complete]
            assert not incomplete, f"capture incomplete for {incomplete}"
            states = [container_state(name) for name in project_containers()]
            problems = stack_problems(states)
            assert not problems, "; ".join(problems)
            control_a, control_b = run_control_a(), run_control_b()
            trusted = control_a.outside_packets >= 1 and control_b.dns_failures >= 1
            total = sum(r.outside_packets + r.dns_failures for r in results)
            if control_b.out_no_routes_delta >= 1:
                total += sum(r.out_no_routes_delta for r in results)
            offenders = [r.container for r in results if r.outside_packets or r.dns_failures or r.out_no_routes_delta]
            assert trusted, (f"the counter cannot be trusted: control A saw {control_a.outside_packets} packets, "
                             f"control B saw {control_b.dns_failures} DNS failures (real count: {total})")
            assert total == 0, f"{total} outbound attempts from: {offenders}"
            return CheckOutcome(
                key_number=f"{total} outbound attempts across {len(results)} services",
                evidence={
                    "services": services, "observed_seconds": seconds,
                    "per_container": [vars(r) for r in results], "states": states,
                    "control_a_open_network": vars(control_a), "control_b_internal_network": vars(control_b),
                    "blind_spots": [
                        "a literal-IP attempt that the kernel refuses leaves no packet; it is only seen as OutNoRoutes",
                        "traffic before the sniffer attached is not seen",
                        f"only {seconds} seconds are watched, so slow periodic phone-home is not seen",
                    ],
                },
            )
        finally:
            cleanup_copy()
    return run_check


# ---------------------------------------------------------------------------
# Check 5: Level 7 still passes in quick mode
# ---------------------------------------------------------------------------

async def run_check_five(workspace: Any) -> CheckOutcome:
    """Run the Level 7 showcase in quick mode and read its result file."""
    code, output = run_command([PYTHON, "-m", "showcase", "--level", "7", "--quick"], REPO_ROOT, 300)
    result_file = REPO_ROOT / "reports" / "showcase" / "level_07" / "result.json"
    assert result_file.is_file(), "Level 7 wrote no result.json: " + last_lines(output, 6)
    stable = orjson.loads(result_file.read_bytes())["stable"]
    checks = stable["checks"]
    passed = sum(1 for check in checks if check["status"] == "PASS")
    summary = f"{passed} of {len(checks)} checks"
    assert code == 0 and stable["overall_status"] == "PASSED", f"Level 7 did not pass ({summary})"
    return CheckOutcome(key_number=f"{passed} of {len(checks)} Level 7 checks passed",
                        evidence={"exit_code": code, "level_7_checks": checks})


def get_checks(quick: bool) -> list[CheckDefinition]:
    """The Level 8 checks, in order."""
    return [
        CheckDefinition(
            "1. make check passes and the baselines match", "normal",
            "The project's own checks pass and every saved baseline matches the current code"
            + (" (quick mode runs the baseline subset)." if quick else "."),
            make_check_one(quick), "A check or a baseline comparison fails.", "python scripts/check.py"),
        CheckDefinition(
            "2. a changed response field is caught", "edge",
            "Renaming an API response field in a temporary copy makes the baseline test fail, and undoing it fixes it.",
            run_check_two, "The baseline snapshot no longer covers the response models.",
            "tests/baselines/test_regression_baselines.py"),
        CheckDefinition(
            "3. all artifacts restore from the remote into an empty folder with matching hashes", "edge",
            "Files stored with DVC come back from the remote with identical SHA-256 hashes"
            + (" (quick mode restores generated files)." if quick else " (generated files and the real 7 artifacts)."),
            make_check_three(quick), "DVC, the remote folder or (full mode) the network is not available.",
            "scripts/verify_dvc_restore.py, docs/backup-and-restore.md"),
        CheckDefinition(
            "4. the stack runs with outgoing network blocked and makes zero outbound attempts", "negative",
            "A copy of the stack on an internal network works and makes no outside connection attempts, "
            "and two control experiments show the counter can see such attempts.",
            make_check_four(quick), "A service tries to reach the internet, or the network is not closed.",
            "showcase/fixtures/level_08/compose.noegress.yml, docker-compose.yml (telemetry settings)"),
        CheckDefinition(
            "5. Level 7's showcase still passes in quick mode", "consistency",
            "The Level 7 showcase still passes after the Level 8 changes.",
            run_check_five, "A Level 8 change broke something Level 7 proves.", "python -m showcase --level 7 --quick"),
    ]
