"""
File: core/level10_check.py
Purpose: Small read-only checks used by the Level 10 showcase (research catalogue, committed rules, related work).
         They validate the config files, count the research status table, show that broken configs are rejected,
         show from git history that the evaluation rules were committed before any new-engine code, and read the
         AMLGentex verdict.

Why it exists: Level 10 is about rules and research written down BEFORE the new engine exists. These functions let
anyone re-check that claim in one command. Nothing here writes a file.
"""

import copy
import subprocess
from datetime import datetime
from pathlib import Path

from core.config_validator import check_block, count_labels, load_yaml, validate_file

REPO_ROOT = Path(__file__).resolve().parent.parent

EVALUATION_PATH = REPO_ROOT / "config" / "evaluation.yaml"
WORLD_RANGES_PATH = REPO_ROOT / "config" / "world_ranges.yaml"
LEGAL_ACTIONS_PATH = REPO_ROOT / "config" / "legal_actions.yaml"
STATUS_PATH = REPO_ROOT / "config" / "research_status.yaml"
DECISIONS_PATH = REPO_ROOT / "docs" / "decisions" / "level-10-evaluation-decisions.md"
AMLGENTEX_DECISION_PATH = REPO_ROOT / "docs" / "decisions" / "level-10-amlgentex-decision.md"
VERDICT_PATH = REPO_ROOT / "docs" / "research" / "amlgentex_spike_verdict.md"

CONFIG_PATHS = [EVALUATION_PATH, WORLD_RANGES_PATH, LEGAL_ACTIONS_PATH]
STATUS_WORDS = ["confirmed", "changed", "reported", "unverified"]
TOOL_RECHECK_WORDS = ["none", "single", "both", "disagree"]

# Paths the new engine will use. They do not exist yet, and nothing under them may be older than the rules.
NEW_ENGINE_PATHS = ["sim/", "world/", "worlds/", "engine/", "decisions/"]

# One file from the v1 PaySim era. It is OLDER than evaluation.yaml, on purpose (see its earlier_work_note).
V1_MODEL_FILE = "scripts/data_split.py"


def check_configs():
    """Run the shared validator on the three rule files.

    Returns (errors, counts): errors is a list of 'file: error' strings (empty = all valid),
    counts is {file name: {label: number of blocks}}.
    """
    errors = []
    counts = {}
    for path in CONFIG_PATHS:
        for error in validate_file(path):
            errors.append(f"{path.name}: {error}")
        counts[path.name] = count_labels(path)
    return errors, counts


def status_counts():
    """Read the research status table.

    Returns (counts per status word, number hand checked, counts per tool_rechecked word).
    """
    findings = load_yaml(STATUS_PATH)["findings"]
    status = {word: 0 for word in STATUS_WORDS}
    tool = {word: 0 for word in TOOL_RECHECK_WORDS}
    hand_checked = 0
    for finding in findings:
        status[finding["status"]] += 1
        tool[finding["tool_rechecked"]] += 1
        if finding["hand_checked"]:
            hand_checked += 1
    return status, hand_checked, tool


def broken_copy_is_rejected():
    """Break three blocks IN MEMORY (no file is written) and return the validator's errors for each.

    Returns {name of the broken copy: list of error messages}. Every list must be non-empty.
    """
    parameters = load_yaml(WORLD_RANGES_PATH)["parameters"]

    # A block that loses its reason.
    no_reason = copy.deepcopy(parameters["mule_transfer_amount_inr"])
    del no_reason["reason"]

    # A sourced block that loses its source.
    no_source = copy.deepcopy(parameters["hops_to_cash_out"])
    del no_source["source"]

    # A range whose training part sticks out of the test part.
    bad_range = copy.deepcopy(parameters["hops_to_cash_out"])
    bad_range["training"] = [0, 5]

    return {
        "reason removed": check_block("copy_without_reason", no_reason),
        "source removed": check_block("copy_without_source", no_source),
        "training outside test": check_block("copy_with_bad_range", bad_range),
    }


def run_git(arguments):
    """Run a read-only git command in the repo and return its text output ('' if git fails)."""
    result = subprocess.run(["git", "-C", str(REPO_ROOT)] + arguments, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else ""


def first_add_date(paths):
    """Date of the earliest commit that ADDED a file under these paths, or None if there is none."""
    output = run_git(["log", "--diff-filter=A", "--format=%aI", "--"] + paths)
    dates = [datetime.fromisoformat(line) for line in output.splitlines() if line.strip()]
    return min(dates) if dates else None


def preregistration_proof():
    """Show from git history that the rules came before any new-engine code.

    Returns a dict with the dates, a result ("OK" or "NOT OK") and one plain sentence.
    The v1 PaySim code is older than the rules by design, so the proof is about the NEW engine only.
    """
    evaluation_date = first_add_date(["config/evaluation.yaml"])
    decisions_date = first_add_date(["docs/decisions/level-10-evaluation-decisions.md"])
    v1_date = first_add_date([V1_MODEL_FILE])

    # Collect every new-engine path that already has a commit.
    new_engine_dates = {}
    for path in NEW_ENGINE_PATHS:
        date = first_add_date([path])
        if date is not None:
            new_engine_dates[path] = date

    if evaluation_date is None:
        result = "NOT OK"
        sentence = "config/evaluation.yaml has no commit in this history, so the order cannot be shown."
    else:
        too_early = [path for path, date in new_engine_dates.items() if date < evaluation_date]
        result = "NOT OK" if too_early else "OK"
        if too_early:
            sentence = f"New-engine files are older than evaluation.yaml: {', '.join(too_early)}."
        elif new_engine_dates:
            sentence = "New-engine files exist, and all of them were added after evaluation.yaml."
        else:
            sentence = "No new-engine code exists yet (none yet), so nothing predates evaluation.yaml."
        sentence += (" The v1 code is older than evaluation.yaml by design (earlier_work_note);"
                     " the pre-registration claim covers the new engine only.")

    return {"evaluation_date": evaluation_date, "decisions_date": decisions_date, "v1_date": v1_date,
            "new_engine_dates": new_engine_dates, "result": result, "sentence": sentence}


def spike_verdict_line():
    """Return the one line that starts with 'Verdict:' in the AMLGentex spike file."""
    lines = VERDICT_PATH.read_text(encoding="utf-8").splitlines()
    verdict_lines = [line for line in lines if line.startswith("Verdict:")]
    if len(verdict_lines) != 1:
        raise ValueError(f"expected exactly one 'Verdict:' line, found {len(verdict_lines)}")
    return verdict_lines[0]
