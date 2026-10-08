"""
File: tests/unit/test_check_version_warning.py
Purpose: Proves the small version-reading helpers in scripts/check.py: they read the CI pin from requirements text, read
         an installed version from `--version` output, and warn (never fail) when the two differ. Standard library only.
"""

from scripts.check import parse_installed_version, parse_pinned_version, version_warning

REQUIREMENTS = "# a comment about ruff==9.9.9\nruff==0.4.4\n\nmypy==1.10.0\n"


def test_pinned_version_is_read_from_the_start_of_a_line() -> None:
    """Only a line that starts with 'tool==' counts; the same text inside a comment is ignored."""
    assert parse_pinned_version(REQUIREMENTS, "ruff") == "0.4.4"
    assert parse_pinned_version(REQUIREMENTS, "mypy") == "1.10.0"
    assert parse_pinned_version(REQUIREMENTS, "pytest") is None
    assert parse_pinned_version("", "ruff") is None


def test_installed_version_is_read_from_version_output() -> None:
    """The first dotted number in the output is taken, whatever words surround it."""
    assert parse_installed_version("ruff 0.15.12") == "0.15.12"
    assert parse_installed_version("mypy 1.20.2 (compiled: yes)") == "1.20.2"
    assert parse_installed_version("no number here") is None
    assert parse_installed_version("") is None


def test_a_difference_gives_a_warning_and_a_match_gives_none() -> None:
    """Different versions warn, equal versions do not, and unreadable versions also warn."""
    assert version_warning("ruff", "0.4.4", "0.4.4") is None
    assert "0.15.12" in str(version_warning("ruff", "0.4.4", "0.15.12"))
    assert "could not read" in str(version_warning("ruff", "0.4.4", None))
    assert "no pinned" in str(version_warning("ruff", None, "0.4.4"))
