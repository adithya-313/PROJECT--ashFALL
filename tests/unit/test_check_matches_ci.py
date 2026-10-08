"""
File: tests/unit/test_check_matches_ci.py
Purpose: Drift guard. Proves scripts/check.py (what `make check` runs) covers every job in .github/workflows/ci.yml,
         so adding a CI job without adding a matching step to check.py makes this test fail.
"""

from pathlib import Path

import yaml

from scripts.check import build_steps

CI_FILE = Path(__file__).resolve().parent.parent.parent / ".github" / "workflows" / "ci.yml"


def job_ids_in_ci(ci_text: str) -> set[str]:
    """The ids of all jobs in a ci.yml text."""
    return set(yaml.safe_load(ci_text)["jobs"])


def job_ids_in_check() -> set[str]:
    """The CI job ids that check.py says its steps mirror."""
    return {step.ci_job for step in build_steps()}


def missing_from_check(ci_text: str) -> set[str]:
    """CI jobs that no step in check.py mirrors."""
    return job_ids_in_ci(ci_text) - job_ids_in_check()


def test_every_ci_job_has_a_step_in_check_py() -> None:
    """No CI job is missing from scripts/check.py."""
    missing = missing_from_check(CI_FILE.read_text(encoding="utf-8"))
    assert not missing, (
        f"These CI jobs have no step in scripts/check.py: {sorted(missing)}. "
        "Add a step that runs the same commands, with ci_job set to the job id."
    )


def test_every_step_names_a_real_ci_job() -> None:
    """Every step's ci_job is a job that exists in ci.yml (catches typos and removed jobs)."""
    unknown = job_ids_in_check() - job_ids_in_ci(CI_FILE.read_text(encoding="utf-8"))
    assert not unknown, f"scripts/check.py mentions CI jobs that do not exist in ci.yml: {sorted(unknown)}"


def test_the_guard_notices_a_new_ci_job() -> None:
    """If a job is added to the CI text, the guard reports it as missing (shown on an in-memory copy)."""
    changed = CI_FILE.read_text(encoding="utf-8") + "\n  brand-new-job:\n    runs-on: ubuntu-latest\n    steps: []\n"
    assert missing_from_check(changed) == {"brand-new-job"}
