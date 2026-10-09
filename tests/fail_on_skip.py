"""
File: tests/fail_on_skip.py
Purpose: A pytest hook that makes the test run FAIL if any test was skipped, when the environment variable
         FAIL_ON_SKIP is set to 1 (CI sets it).

Why it exists: A skipped test looks like a quiet pass. Tests that cannot run in CI must be left out on purpose with a
marker (for example -m "not ml"), which deselects them. This hook turns every other skip into a failure.
"""
import os


def pytest_sessionfinish(session, exitstatus):
    """After the run: if FAIL_ON_SKIP=1 and something was skipped, print what and make the exit code 1."""
    if os.environ.get("FAIL_ON_SKIP") != "1":
        return
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    skipped = reporter.stats.get("skipped", []) if reporter else []
    if not skipped:
        return
    reporter.write_line(f"FAIL_ON_SKIP: {len(skipped)} test(s) were skipped. Deselect them with a marker instead:",
                        red=True)
    for report in skipped:
        reporter.write_line(f"  {report.nodeid}", red=True)
    session.exitstatus = 1
