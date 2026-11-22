"""
WHAT: Unit tests for Section 23 leakage checks 5 and 6.

WHY: The static scans must find a planted simulator-only field and a planted identifier parse, and
must leave a clean lookup alone; the measurement's pass rule must turn hand numbers into the right
verdict.
"""

from pathlib import Path

import pytest

from evaluation.leakage_checks_5_6 import (
    FORBIDDEN_IDENTIFIER_NAMES,
    id_leak_passes,
    scan_identifier_derivations,
    scan_simulator_field_usage,
)

SIMULATOR_ONLY = frozenset({"purpose", "sender_is_new_device"})


def test_check_5a_flags_a_planted_simulator_field(tmp_path: Path) -> None:
    """A feature that reads a simulator-only field is reported, and a clean feature is not."""
    planted = tmp_path / "planted_feature.py"
    planted.write_text('def feature(row):\n    return row["purpose"]\n', encoding="utf-8")
    hits, excluded = scan_simulator_field_usage([planted], SIMULATOR_ONLY)
    assert [(hit.name, hit.line, hit.kind) for hit in hits] == [("purpose", 2, "string literal")]
    assert excluded == []

    clean = tmp_path / "clean_feature.py"
    clean.write_text("def feature(row):\n    return row.amount_paise\n", encoding="utf-8")
    clean_hits, clean_excluded = scan_simulator_field_usage([clean], SIMULATOR_ONLY)
    assert clean_hits == []
    assert clean_excluded == []


def test_check_6a_flags_a_planted_identifier_parse(tmp_path: Path) -> None:
    """A feature that parses the account id is reported, and using it as a lookup key is not."""
    assert "account_id" in FORBIDDEN_IDENTIFIER_NAMES
    planted = tmp_path / "planted_feature.py"
    planted.write_text(
        'def feature(account_id):\n    return int(account_id.rsplit("_", 1)[-1])\n',
        encoding="utf-8",
    )
    hits, excluded = scan_identifier_derivations([planted], FORBIDDEN_IDENTIFIER_NAMES)
    assert [(hit.name, hit.line, hit.kind) for hit in hits] == [("account_id", 2, "string method rsplit()")]
    assert excluded == []

    clean = tmp_path / "clean_feature.py"
    clean.write_text("def feature(account_id, ledger):\n    return ledger[account_id]\n", encoding="utf-8")
    clean_hits, clean_excluded = scan_identifier_derivations([clean], FORBIDDEN_IDENTIFIER_NAMES)
    assert clean_hits == []
    assert clean_excluded == []


def test_check_6b_pass_rule_on_hand_numbers() -> None:
    """Average precision within the allowed multiple of the base rate passes; above it fails."""
    assert id_leak_passes(0.10, 0.06, 2.0) is True
    assert id_leak_passes(0.13, 0.06, 2.0) is False
    assert id_leak_passes(0.05, 0.25, 2.0) is True
    with pytest.raises(ValueError):
        id_leak_passes(0.10, 0.0, 2.0)
