"""
File: tests/unit/test_issue_register.py
Purpose: Tests the issue register checker to ensure it catches missing or malformed issues without breaking on valid ones.
"""

import copy
from pathlib import Path

from core import issue_register

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
REAL_REGISTER_PATH = REPO_ROOT / "config" / "issues.yaml"

def load_real():
    return issue_register.load_register(REAL_REGISTER_PATH)

def test_real_register_is_valid():
    """check_register on the real file returns []."""
    data = load_real()
    errors = issue_register.check_register(data, REPO_ROOT)
    assert not errors

def test_register_has_all_31_ids():
    """The real register has exactly the 31 EXPECTED_IDS."""
    data = load_real()
    assert set(data.keys()) == issue_register.EXPECTED_IDS

def test_status_counts():
    """exactly {"built": 28, "planned": 2, "deferred": 1}."""
    data = load_real()
    built = sum(1 for v in data.values() if v.get("status") == "built")
    planned = sum(1 for v in data.values() if v.get("status") == "planned")
    deferred = sum(1 for v in data.values() if v.get("status") == "deferred")
    assert built == 28
    assert planned == 2
    assert deferred == 1

def test_r9_is_deferred_with_a_reason():
    """R9 must be deferred and have a reason."""
    data = load_real()
    assert data["R9"]["status"] == "deferred"
    assert "reason" in data["R9"]
    assert data["R9"]["reason"]

def test_missing_id_is_rejected():
    """delete 'G1' from the dict."""
    data = load_real()
    del data["G1"]
    errors = issue_register.check_register(data, REPO_ROOT)
    assert any("Missing expected IDs" in err and "G1" in err for err in errors)

def test_unknown_id_is_rejected():
    """add 'ZZ9'."""
    data = load_real()
    data["ZZ9"] = {"title": "Test", "status": "deferred", "reason": "Test"}
    errors = issue_register.check_register(data, REPO_ROOT)
    assert any("Unknown IDs found" in err and "ZZ9" in err for err in errors)

def test_unknown_status_is_rejected():
    """change a status to unknown."""
    data = load_real()
    data["G1"]["status"] = "magical"
    errors = issue_register.check_register(data, REPO_ROOT)
    assert any("Unknown status" in err for err in errors)

def test_deferred_without_reason_is_rejected():
    """deferred must have a reason."""
    data = load_real()
    data["R9"].pop("reason")
    errors = issue_register.check_register(data, REPO_ROOT)
    assert any("missing a reason" in err for err in errors)

def test_built_without_evidence_is_rejected():
    """built must have evidence."""
    data = load_real()
    data["G1"]["evidence"] = []
    errors = issue_register.check_register(data, REPO_ROOT)
    assert any("must have at least one evidence item" in err for err in errors)

def test_planned_without_later_work_is_rejected():
    """planned must have later_work."""
    data = load_real()
    data["R7"]["later_work"] = []
    errors = issue_register.check_register(data, REPO_ROOT)
    assert any("must have non-empty later_work list" in err for err in errors)

def test_bad_later_work_note_is_rejected():
    """note not starting with 'Level '."""
    data = load_real()
    data["R7"]["later_work"] = ["Soon"]
    errors = issue_register.check_register(data, REPO_ROOT)
    assert any("Bad later_work note" in err for err in errors)

def test_config_evidence_with_missing_parameter_is_rejected():
    """change the parameter name in config evidence."""
    data = load_real()
    data["G1"]["evidence"][0]["name"] = "does_not_exist"
    errors = issue_register.check_register(data, REPO_ROOT)
    assert any("Block does_not_exist not found" in err for err in errors)

def test_config_evidence_without_this_issue_id_is_rejected():
    """keep parameter but change issue ID."""
    data = load_real()
    data["G1"]["evidence"][0]["name"] = "mule_origin_share_rented"
    # test it under R2
    data["R2"]["evidence"] = [data["G1"]["evidence"][0]]
    data["R2"]["status"] = "built"
    errors = issue_register.check_register(data, REPO_ROOT)
    assert any("Issue ID R2 not in issue_ids" in err for err in errors)

def test_missing_amendment_is_rejected():
    """change amendment ID to something not there."""
    data = load_real()
    data["G1"]["evidence"][2]["id"] = "A-999"
    errors = issue_register.check_register(data, REPO_ROOT)
    assert any("Amendment A-999 not found" in err for err in errors)

def test_amendment_without_this_issue_id_is_rejected():
    """amendment that exists but doesn't mention G1."""
    data = load_real()
    # A-001 doesn't have G1
    data["G1"]["evidence"][2]["id"] = "A-001"
    errors = issue_register.check_register(data, REPO_ROOT)
    assert any("Issue ID G1 not in amendment" in err for err in errors)

def test_doc_text_missing_is_rejected():
    """doc evidence with missing text."""
    data = load_real()
    data["N1"]["evidence"][1]["text"] = "magical string not there"
    errors = issue_register.check_register(data, REPO_ROOT)
    assert any("Text 'magical string not there' not found" in err for err in errors)

def test_test_function_missing_is_rejected():
    """test evidence with missing function."""
    data = load_real()
    data["R6"]["evidence"][0]["name"] = "test_does_not_exist"
    errors = issue_register.check_register(data, REPO_ROOT)
    assert any("Test function test_does_not_exist not found" in err for err in errors)

def test_unknown_evidence_kind_is_rejected():
    """unknown kind."""
    data = load_real()
    data["G1"]["evidence"][0]["kind"] = "magic"
    errors = issue_register.check_register(data, REPO_ROOT)
    assert any("Unknown evidence kind 'magic'" in err for err in errors)

def test_unknown_config_issue_id_is_rejected():
    """check_config_ids_known({"ZZ9"})."""
    errors = issue_register.check_config_ids_known({"ZZ9"})
    assert any("Unknown ID in config files: ZZ9" in err for err in errors)

def test_main_returns_0_for_real_file_and_1_for_broken_copy(tmp_path):
    """write a broken copy into tmp_path and call main([path])."""
    # real
    assert issue_register.main([str(REAL_REGISTER_PATH)]) == 0
    
    # broken
    broken = tmp_path / "broken.yaml"
    broken.write_text("ZZ9: {title: 'Test', status: 'deferred', reason: 'Test'}", encoding="utf-8")
    assert issue_register.main([str(broken)]) == 1
