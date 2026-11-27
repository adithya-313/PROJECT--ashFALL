"""
File: tests/unit/test_fail_on_skip.py
Purpose: Tests the FAIL_ON_SKIP guard on tiny test files that pytester writes and runs in a temporary folder.
"""
from types import SimpleNamespace

import pytest

from tests.fail_on_skip import EXEMPT_NODE_ID, EXEMPT_REASON, _is_exempt

pytest_plugins = ["pytester"]

HOOK = "from tests.fail_on_skip import pytest_sessionfinish  # noqa: F401\n"

SKIPPING_TEST = '''
import pytest

def test_a_skipped_test():
    pytest.skip("not here")

def test_a_normal_test():
    assert True
'''

MARKED_TEST = '''
import pytest

@pytest.mark.slowish
def test_marked_test():
    pytest.skip("not here")

def test_a_normal_test():
    assert True
'''


def run_inner(pytester, source, *arguments):
    pytester.makeconftest(HOOK)
    pytester.makepyfile(test_inner=source)
    return pytester.runpytest("-p", "no:cacheprovider", "-W", "ignore::pytest.PytestUnknownMarkWarning", *arguments)


def test_skip_is_allowed_when_the_guard_is_off(pytester, monkeypatch):
    monkeypatch.delenv("FAIL_ON_SKIP", raising=False)
    result = run_inner(pytester, SKIPPING_TEST)
    assert result.ret == 0


def test_skip_fails_when_the_guard_is_on(pytester, monkeypatch):
    monkeypatch.setenv("FAIL_ON_SKIP", "1")
    result = run_inner(pytester, SKIPPING_TEST)
    assert result.ret == 1
    result.stdout.fnmatch_lines(["*FAIL_ON_SKIP: 1 test(s) were skipped*", "*test_a_skipped_test*"])


def test_a_deselected_test_does_not_count_as_a_skip(pytester, monkeypatch):
    monkeypatch.setenv("FAIL_ON_SKIP", "1")
    result = run_inner(pytester, MARKED_TEST, "-m", "not slowish")
    assert result.ret == 0


def test_all_passing_is_fine_with_the_guard_on(pytester, monkeypatch):
    monkeypatch.setenv("FAIL_ON_SKIP", "1")
    result = run_inner(pytester, "def test_ok():\n    assert True\n")
    assert result.ret == 0


@pytest.mark.parametrize("value", ["0", ""])
def test_other_values_leave_the_guard_off(pytester, monkeypatch, value):
    monkeypatch.setenv("FAIL_ON_SKIP", value)
    assert run_inner(pytester, SKIPPING_TEST).ret == 0


# No fake test_world_estimates.py is written here: the real module already has that name, so writing
# a same-named fake file made pytest report "import file mismatch" between the two.


def fake_skip_report(node_id: str, message: str) -> SimpleNamespace:
    """Build a fake pytest skip report with the shape the matcher reads: (path, line, message)."""
    return SimpleNamespace(nodeid=node_id, longrepr=("some/path.py", 1, message))


@pytest.mark.parametrize(
    ("node_id", "message", "expected"),
    [
        (EXEMPT_NODE_ID, f"Skipped: {EXEMPT_REASON}", True),
        (
            "tests/unit/world/test_world_estimates.py::test_speed_stays_steady_as_worlds_grow",
            f"Skipped: {EXEMPT_REASON}",
            False,
        ),
        (EXEMPT_NODE_ID + "_again", f"Skipped: {EXEMPT_REASON}", False),
        (EXEMPT_NODE_ID, "Skipped: not here", False),
        (EXEMPT_NODE_ID, f"Skipped: {EXEMPT_REASON} today", False),
    ],
)
def test_only_the_exact_test_and_reason_are_exempt(node_id: str, message: str, expected: bool) -> None:
    """Check the exemption rule on its own: only the exact node id with the exact reason matches."""
    report = fake_skip_report(node_id, message)
    assert _is_exempt(report) is expected


def test_real_skip_reports_have_the_shape_the_matcher_reads(pytester) -> None:
    """Prove a real skipped report has the same shape the matcher reads, so the checks above are meaningful."""
    source = (
        "import pytest\n\n"
        f"def test_probe():\n    pytest.skip({EXEMPT_REASON!r})\n"
    )
    pytester.makepyfile(test_shape_probe=source)
    recorder = pytester.inline_run("-p", "no:cacheprovider")
    reports = recorder.getreports("pytest_runtest_logreport")
    skipped = [report for report in reports if report.outcome == "skipped"]
    assert len(skipped) == 1
    report = skipped[0]
    assert isinstance(report.longrepr, tuple)
    assert report.longrepr[2] == f"Skipped: {EXEMPT_REASON}"
    # Wrong node id: the probe is not the exempt test_world_estimates test.
    assert _is_exempt(report) is False
