"""
WHAT: Tests for the showcase --part option, which runs only one part of a level that is built in pieces.
Every test drives the real command-line entry point (showcase.__main__.main) against a small fake level defined
in this file, so no real level, database or network is involved.

WHY: Level 12 is reviewed part by part (12a, 12b, ...). The option must run exactly the requested part, leave
whole-level runs unchanged, and refuse anything it cannot honour with exit code 2 instead of passing with zero
checks. These tests pin each of those promises.

TERM: part: one letter naming a slice of a level that is built and reviewed on its own,
      e.g. "a" for Level 12a.
TERM: fake level: a small module object built inside the test that looks like showcase/level_NN.py,
      e.g. level 42 with four checks that only record that they ran.
TERM: exit code: the number a command hands back when it finishes,
      e.g. 0 when every check passed and 2 when the command line itself was wrong.
TERM: monkeypatch: a pytest helper that swaps a value for the length of one test and puts it back afterwards,
      e.g. pointing the report folder at a temporary folder so no real report is overwritten.
"""

from functools import partial
from pathlib import Path
from types import ModuleType
from typing import Any

import orjson
import pytest

import showcase.__main__ as showcase_main
from showcase.framework import CheckDefinition, CheckOutcome

# The level number of the fake level; no real showcase uses it.
FAKE_LEVEL_NUMBER: int = 42


async def record_and_pass(name: str, ran: list[str], workspace: Any) -> CheckOutcome:
    """
    WHAT: The body of every fake check. It writes the check's own name at the end of ran and then reports a pass
    with the key number "ok". The workspace argument is accepted only because the framework always passes one.

    WHY: Recording names is how the tests see exactly which checks ran and in what order. One shared function,
    with the name and list filled in beforehand, keeps the four fake checks identical apart from their names.

    Args:
        name: The check name to record, for example "1. normal".
        ran: The list that collects the names of the checks that ran.
        workspace: The framework's workspace; unused here and None for the fake level.

    Returns:
        A passing CheckOutcome with key_number "ok".

    Example:
        With ran = [], await record_and_pass("1. normal", ran, None) leaves ran == ["1. normal"].
    """
    # Note that this check ran, then pass.
    ran.append(name)
    return CheckOutcome(key_number="ok")


def make_fake_level(ran: list[str], tagged: bool) -> ModuleType:
    """
    WHAT: Builds a fake level module with four checks that meet the coverage rules as a whole level:
    one normal, two edge and one negative. When tagged is True, checks 1 and 3 belong to part "a" and checks
    2 and 4 to part "b"; when False, no check carries a part. Each check appends its own name to ran.

    WHY: The command line must be tested without running a real level, which would need minutes of simulation.
    Recording names shows exactly which checks ran and in what order, which is the behaviour under test.

    Args:
        ran: An empty list that each check appends its name to when it runs.
        tagged: True to tag the checks with parts "a" and "b", False to leave them untagged.

    Returns:
        A module object with LEVEL_NUMBER, LEVEL_TITLE, WHAT_YOU_SAW, NEEDS_DATABASE,
        find_missing_preconditions and get_checks, like a real showcase/level_NN.py.

    Example:
        make_fake_level([], tagged=True).get_checks(False) gives 4 checks tagged a, b, a, b.
        make_fake_level([], tagged=False).get_checks(False) gives the same 4 checks with part None.
    """
    # The four checks: (name, kind, part). Part "a" alone has no negative check, like a real Level 12 part.
    plan = [("1. normal", "normal", "a"), ("2. edge one", "edge", "b"),
            ("3. edge two", "edge", "a"), ("4. negative", "negative", "b")]
    # Build the CheckDefinitions. Each run function is record_and_pass with this check's name and list filled in.
    checks = [CheckDefinition(name, kind, f"{name} works.", partial(record_and_pass, name, ran), "demo cause.",
                              "demo.py", part=(part if tagged else None)) for name, kind, part in plan]
    # Assemble a module that offers everything the framework asks a level for.
    module = ModuleType("fake_level")
    module.LEVEL_NUMBER = FAKE_LEVEL_NUMBER  # type: ignore[attr-defined]
    module.LEVEL_TITLE = "Fake split level"  # type: ignore[attr-defined]
    module.WHAT_YOU_SAW = "A fake level."  # type: ignore[attr-defined]
    module.NEEDS_DATABASE = False  # type: ignore[attr-defined]
    module.find_missing_preconditions = lambda: []  # type: ignore[attr-defined]
    module.get_checks = lambda quick: checks  # type: ignore[attr-defined]
    return module


def use_fake_level(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, tagged: bool) -> list[str]:
    """
    WHAT: Points the command line at the fake level and at a temporary report folder for one test.
    It replaces level discovery with a function returning only the fake level, sends reports to tmp_path,
    and turns off the summary page refresh. It returns the list the fake checks write their names into.

    WHY: Without this, the test would find the real levels and overwrite the real reports/showcase folder.
    Monkeypatch puts every swapped value back after the test, so no other test is affected.

    Args:
        monkeypatch: pytest's monkeypatch helper for this test.
        tmp_path: pytest's temporary folder for this test.
        tagged: Passed on to make_fake_level: True for a split level, False for an untagged one.

    Returns:
        The list of check names that ran, empty until main() is called.

    Example:
        ran = use_fake_level(monkeypatch, tmp_path, tagged=True); main(["--level", "42"]) leaves ran with 4 names.
    """
    # The list the fake checks append to.
    ran: list[str] = []
    fake_level = make_fake_level(ran, tagged)
    # Only the fake level exists while this test runs.
    monkeypatch.setattr(showcase_main, "discover_level_modules", lambda: {FAKE_LEVEL_NUMBER: fake_level})
    # Reports go to the temporary folder, and the real summary page is left alone.
    monkeypatch.setattr(showcase_main, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(showcase_main, "refresh_summary", lambda: None)
    return ran


def test_part_a_runs_only_part_a_checks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """
    WHAT: Runs the fake split level with --part a and checks that only checks 1 and 3 ran, in that order.
    It also checks the report went to level_42/part_a/ and that its result.json records "part": "a".

    WHY: This is the main promise of --part. Part "a" alone has no negative check, so this run also proves the
    coverage rules were applied to the whole level before filtering, not to the part.

    Args:
        monkeypatch: pytest's monkeypatch helper.
        tmp_path: pytest's temporary folder.

    Returns:
        None. The test fails if other checks ran or the report is in the wrong place.

    Example:
        main(["--level", "42", "--part", "a"]) returns 0 and runs ["1. normal", "3. edge two"] (2 of 4 checks).
    """
    # Run only part a of the fake level.
    ran = use_fake_level(monkeypatch, tmp_path, tagged=True)
    exit_code = showcase_main.main(["--level", str(FAKE_LEVEL_NUMBER), "--part", "a"])
    assert exit_code == 0
    assert ran == ["1. normal", "3. edge two"]
    # The part report lives in its own sub-folder and says which part it is.
    result = orjson.loads((tmp_path / "level_42" / "part_a" / "result.json").read_bytes())
    assert result["stable"]["part"] == "a"
    assert [check["name"] for check in result["stable"]["checks"]] == ["1. normal", "3. edge two"]
    # Nothing was written where the whole-level result belongs.
    assert not (tmp_path / "level_42" / "result.json").exists()


def test_without_part_every_check_runs_in_order(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """
    WHAT: Runs the fake split level without --part and checks that all four checks ran in their file order.
    It also checks the report went to the usual level_42/ folder and that result.json has no "part" key.

    WHY: Adding parts must not change whole-level runs at all. Levels 7 to 11 rely on their report location and
    on the exact keys of result.json staying as they were.

    Args:
        monkeypatch: pytest's monkeypatch helper.
        tmp_path: pytest's temporary folder.

    Returns:
        None. The test fails if a check was skipped, the order changed, or the report changed shape.

    Example:
        main(["--level", "42"]) returns 0 and runs all 4 checks, 1 to 4.
    """
    # Run the whole fake level.
    ran = use_fake_level(monkeypatch, tmp_path, tagged=True)
    exit_code = showcase_main.main(["--level", str(FAKE_LEVEL_NUMBER)])
    assert exit_code == 0
    assert ran == ["1. normal", "2. edge one", "3. edge two", "4. negative"]
    # The whole-level report keeps its usual place and its usual keys.
    result = orjson.loads((tmp_path / "level_42" / "result.json").read_bytes())
    assert "part" not in result["stable"]
    assert set(result["stable"]) == {"level_number", "title", "mode", "overall_status", "checks"}


def test_unknown_part_exits_two(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                                capsys: pytest.CaptureFixture[str]) -> None:
    """
    WHAT: Asks the fake split level for part "z", which none of its checks carry, and expects exit code 2,
    the message "Level 42 has no checks for part z", and no check run.

    WHY: Running zero checks would otherwise count as a pass. A typo in --part must stop loudly, not succeed.

    Args:
        monkeypatch: pytest's monkeypatch helper.
        tmp_path: pytest's temporary folder.
        capsys: pytest's helper that captures what the command printed.

    Returns:
        None. The test fails if the command passed or printed something else.

    Example:
        main(["--level", "42", "--part", "z"]) returns 2 and prints "Level 42 has no checks for part z".
    """
    # Ask for a part that does not exist in the level.
    ran = use_fake_level(monkeypatch, tmp_path, tagged=True)
    exit_code = showcase_main.main(["--level", str(FAKE_LEVEL_NUMBER), "--part", "z"])
    assert exit_code == 2
    assert "Level 42 has no checks for part z" in capsys.readouterr().out
    assert ran == []


def test_part_on_untagged_level_exits_two(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                                          capsys: pytest.CaptureFixture[str]) -> None:
    """
    WHAT: Runs --part a on a fake level whose checks carry no part tag at all, the way Levels 7 to 11 are written,
    and expects exit code 2 with the "no checks for part" message.

    WHY: --part must never quietly fall back to running the whole level; that would hide a wrong command.

    Args:
        monkeypatch: pytest's monkeypatch helper.
        tmp_path: pytest's temporary folder.
        capsys: pytest's helper that captures what the command printed.

    Returns:
        None. The test fails if any check ran or the exit code is not 2.

    Example:
        main(["--level", "42", "--part", "a"]) on the untagged level returns 2 and runs 0 checks.
    """
    # The same four checks, but with no part tags.
    ran = use_fake_level(monkeypatch, tmp_path, tagged=False)
    exit_code = showcase_main.main(["--level", str(FAKE_LEVEL_NUMBER), "--part", "a"])
    assert exit_code == 2
    assert "Level 42 has no checks for part a" in capsys.readouterr().out
    assert ran == []


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--all", "--part", "a"], "--part can only be used together with --level"),
        (["--level", "42", "--part", "ab"], "--part must be one lowercase letter such as a, got 'ab'"),
        (["--level", "42", "--part", "A"], "--part must be one lowercase letter such as a, got 'A'"),
    ],
)
def test_bad_part_arguments_exit_two(argv: list[str], message: str, monkeypatch: pytest.MonkeyPatch,
                                     tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """
    WHAT: Gives --part without --level, and --part values that are not a single lowercase letter, and expects
    exit code 2 with a plain sentence saying what is wrong, before any check runs.

    WHY: Input is checked at the door. A malformed option should explain itself instead of starting a run that
    can only end in a confusing failure.

    Args:
        argv: The command-line words to run, for example ["--all", "--part", "a"].
        message: The sentence the command must print for those words.
        monkeypatch: pytest's monkeypatch helper.
        tmp_path: pytest's temporary folder.
        capsys: pytest's helper that captures what the command printed.

    Returns:
        None. The test fails if the command does not return 2 or prints a different sentence.

    Example:
        main(["--all", "--part", "a"]) returns 2 and prints "--part can only be used together with --level".
    """
    # Run the command with the bad arguments.
    ran = use_fake_level(monkeypatch, tmp_path, tagged=True)
    exit_code = showcase_main.main(argv)
    assert exit_code == 2
    assert message in capsys.readouterr().out
    assert ran == []
