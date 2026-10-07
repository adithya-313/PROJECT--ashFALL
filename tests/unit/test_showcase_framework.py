# File: tests/unit/test_showcase_framework.py
# Purpose: Proves the showcase framework's rules (PASS / FAIL / NOT RUN, exit codes, coverage, cleanup) using FAKE
#          levels. Faking is fine here: the code under test is the framework itself, not a level.
import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from showcase.framework import (
    CheckCoverageError,
    CheckDefinition,
    CheckOutcome,
    NotRun,
    decide_exit_code,
    print_checklist,
    run_level_module,
    validate_checks,
)


def passing_run(label: str, order: list[str]) -> Any:
    """Build a check function that records its label and passes."""
    async def run(workspace: Any) -> CheckOutcome:
        order.append(label)
        return CheckOutcome(key_number="1 of 1")
    return run


def failing_run(error: Exception) -> Any:
    """Build a check function that raises the given error."""
    async def run(workspace: Any) -> CheckOutcome:
        raise error
    return run


def make_check(name: str, kind: str, run: Any) -> CheckDefinition:
    """Build a check with simple fixed text."""
    return CheckDefinition(name, kind, f"{name} works.", run, "Likely cause: demo.", "demo.py")


def good_checks(order: list[str]) -> list[CheckDefinition]:
    """A list of checks that meets the coverage rules and all pass."""
    return [
        make_check("normal-one", "normal", passing_run("normal-one", order)),
        make_check("edge-one", "edge", passing_run("edge-one", order)),
        make_check("edge-two", "edge", passing_run("edge-two", order)),
        make_check("negative-one", "negative", passing_run("negative-one", order)),
    ]


def make_module(checks: list[CheckDefinition], missing: list[str] | None = None, needs_db: bool = True) -> Any:
    """Build a fake level module with the contract attributes."""
    return SimpleNamespace(
        LEVEL_NUMBER=99, LEVEL_TITLE="Fake level", WHAT_YOU_SAW="You saw a fake level.", NEEDS_DATABASE=needs_db,
        find_missing_preconditions=lambda: missing or [],
        get_checks=lambda quick: checks,
    )


class FakeWorkspaceKeeper:
    """Records whether the workspace was created and cleaned up."""

    def __init__(self) -> None:
        self.created = 0
        self.cleaned = 0

    async def create(self) -> str:
        """Hand out a fake workspace."""
        self.created += 1
        return "fake-workspace"

    async def cleanup(self, workspace: Any) -> None:
        """Record the clean-up call."""
        self.cleaned += 1


def run_fake_level(module: Any, keeper: FakeWorkspaceKeeper, allow_not_run: bool = False) -> Any:
    """Run a fake level through the real framework."""
    return asyncio.run(run_level_module(module, False, allow_not_run, keeper.create, Path("unused"), keeper.cleanup))


def test_all_pass_gives_exit_code_zero():
    """When every check passes, the level is PASSED and the exit code is 0."""
    result = run_fake_level(make_module(good_checks([])), FakeWorkspaceKeeper())
    assert result.overall_status == "PASSED"
    assert decide_exit_code(result, False) == 0


def test_one_fail_gives_exit_code_one_and_shows_likely_cause():
    """A failing check makes the level FAILED, exit code 1, with the likely cause in the explanation."""
    checks = good_checks([])
    checks[1] = make_check("edge-one", "edge", failing_run(AssertionError("count was 2")))
    result = run_fake_level(make_module(checks), FakeWorkspaceKeeper())
    assert result.check_results[1].status == "FAIL"
    assert "Likely cause: demo." in result.check_results[1].explanation
    assert decide_exit_code(result, False) == 1


def test_not_run_check_fails_by_default_but_is_accepted_with_allow_not_run():
    """NOT RUN gives exit code 1 normally, and 0 only when allow_not_run is set."""
    checks = good_checks([])
    checks[0] = make_check("normal-one", "normal", failing_run(NotRun("broker is down")))
    result = run_fake_level(make_module(checks), FakeWorkspaceKeeper())
    assert result.check_results[0].status == "NOT RUN"
    assert "broker is down" in result.check_results[0].explanation
    assert decide_exit_code(result, False) == 1
    assert decide_exit_code(run_fake_level(make_module(checks), FakeWorkspaceKeeper(), True), True) == 0


def test_unexpected_exception_is_a_fail_and_later_checks_still_run():
    """A check that crashes becomes FAIL, the framework survives, and the next checks still run."""
    order: list[str] = []
    checks = good_checks(order)
    checks[0] = make_check("normal-one", "normal", failing_run(RuntimeError("boom")))
    result = run_fake_level(make_module(checks), FakeWorkspaceKeeper())
    assert result.check_results[0].status == "FAIL"
    assert "RuntimeError: boom" in result.check_results[0].explanation
    assert order == ["edge-one", "edge-two", "negative-one"]


def test_missing_preconditions_make_every_check_not_run():
    """If a precondition is missing, every check is NOT RUN with the reason, and no workspace is created."""
    keeper = FakeWorkspaceKeeper()
    module = make_module(good_checks([]), missing=["Kafka not reachable - run: docker compose up -d redpanda"])
    result = run_fake_level(module, keeper)
    assert [item.status for item in result.check_results] == ["NOT RUN"] * 4
    assert "Kafka not reachable" in result.check_results[0].explanation
    assert decide_exit_code(result, False) == 1
    assert keeper.created == 0


def test_validate_checks_rejects_missing_negative_case():
    """A level with no negative check is refused."""
    checks = [c for c in good_checks([]) if c.kind != "negative"]
    with pytest.raises(CheckCoverageError, match="negative"):
        validate_checks(checks)


def test_validate_checks_rejects_only_one_edge_case():
    """A level with only one edge check is refused."""
    checks = [c for c in good_checks([]) if c.name != "edge-two"]
    with pytest.raises(CheckCoverageError, match="edge"):
        validate_checks(checks)


def test_validate_checks_rejects_duplicate_names():
    """Two checks with the same name are refused."""
    checks = good_checks([])
    checks[1].name = "normal-one"
    with pytest.raises(CheckCoverageError, match="duplicate"):
        validate_checks(checks)


def test_checks_run_in_declared_order():
    """Checks run one at a time in the order the level lists them."""
    order: list[str] = []
    run_fake_level(make_module(good_checks(order)), FakeWorkspaceKeeper())
    assert order == ["normal-one", "edge-one", "edge-two", "negative-one"]


def test_workspace_is_cleaned_up_even_when_a_check_fails():
    """Clean-up is called exactly once even if a check fails."""
    keeper = FakeWorkspaceKeeper()
    checks = good_checks([])
    checks[0] = make_check("normal-one", "normal", failing_run(RuntimeError("boom")))
    run_fake_level(make_module(checks), keeper)
    assert keeper.created == 1
    assert keeper.cleaned == 1


def test_print_checklist_shows_names_and_status_words(capsys):
    """The terminal checklist contains every check name and a status word."""
    checks = good_checks([])
    checks[0] = make_check("normal-one", "normal", failing_run(NotRun("broker is down")))
    result = run_fake_level(make_module(checks), FakeWorkspaceKeeper())
    print_checklist(result)
    output = capsys.readouterr().out
    for name in ("normal-one", "edge-one", "edge-two", "negative-one"):
        assert name in output
    assert "NOT RUN" in output and "PASS" in output
    assert "What you just saw" in output
