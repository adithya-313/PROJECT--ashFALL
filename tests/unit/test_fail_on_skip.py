"""
File: tests/unit/test_fail_on_skip.py
Purpose: Tests the FAIL_ON_SKIP guard on tiny test files that pytester writes and runs in a temporary folder.
"""
import pytest

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
