"""Hand examples for the two readers the Level 18 handoff generator relies on."""

import pytest

from scripts.level18_handoff import amendment_change, spread

TEXT = (
    "## A-070\n\n| Field | Value |\n| What changed | first change |\n\n"
    "## A-070a\n\n| Field | Value |\n| What changed | second change |\n"
)


def test_amendment_change_reads_only_the_named_amendment() -> None:
    assert amendment_change(TEXT, "A-070") == "first change"
    assert amendment_change(TEXT.replace("\n", "\r\n"), "A-070a") == "second change"
    with pytest.raises(ValueError, match="no section for A-071"):
        amendment_change(TEXT, "A-071")


def test_spread_shows_median_p10_and_p90() -> None:
    assert spread({"median": 0.5, "p10": 0.25, "p90": 1.0}) == "0.5000 (0.2500 to 1.0000)"
