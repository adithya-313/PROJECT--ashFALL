"""Checks that config/research_status.yaml keeps its agreed shape and counts."""
import copy
from pathlib import Path

import pytest
import yaml

STATUS_PATH = Path(__file__).resolve().parents[2] / "config" / "research_status.yaml"

ALLOWED_STATUSES = ["confirmed", "changed", "reported", "unverified"]


def load_findings():
    """Read the YAML file and return the list of findings."""
    with open(STATUS_PATH, encoding="utf-8") as status_file:
        return yaml.safe_load(status_file)["findings"]


def check_findings(findings):
    """Raise ValueError with a clear message if the findings are not well formed."""
    seen_ids = set()
    for finding in findings:
        finding_id = finding.get("id")

        # Each id must be used only once.
        if finding_id in seen_ids:
            raise ValueError(f"duplicate id {finding_id!r}")
        seen_ids.add(finding_id)

        # Only the four agreed status words are allowed.
        if finding.get("status") not in ALLOWED_STATUSES:
            raise ValueError(f"{finding_id}: status {finding.get('status')!r} is not allowed")

        # hand_checked must be a real true/false value.
        if not isinstance(finding.get("hand_checked"), bool):
            raise ValueError(f"{finding_id}: hand_checked must be true or false")


def count_by_status(findings):
    counts = {status: 0 for status in ALLOWED_STATUSES}
    for finding in findings:
        counts[finding["status"]] += 1
    return counts


def test_file_is_well_formed():
    check_findings(load_findings())


def test_there_are_27_findings():
    assert len(load_findings()) == 27


def test_ids_are_unique():
    ids = [finding["id"] for finding in load_findings()]
    assert len(ids) == len(set(ids))


def test_status_counts():
    assert count_by_status(load_findings()) == {"confirmed": 11, "changed": 4, "reported": 6, "unverified": 6}


def test_exactly_three_are_hand_checked():
    hand_checked = [finding for finding in load_findings() if finding["hand_checked"]]
    assert len(hand_checked) == 3


def test_bad_status_word_is_rejected():
    findings = copy.deepcopy(load_findings())
    findings[0]["status"] = "maybe"
    with pytest.raises(ValueError, match="is not allowed"):
        check_findings(findings)


def test_duplicate_id_is_rejected():
    findings = copy.deepcopy(load_findings())
    findings[1]["id"] = findings[0]["id"]
    with pytest.raises(ValueError, match="duplicate id"):
        check_findings(findings)
