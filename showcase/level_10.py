# File: showcase/level_10.py
# Purpose: The Level 10 showcase ("Research catalogue, committed rules, related-work study"). It READS the saved rule
#          files and research documents (it never trains or simulates anything) and shows: every config validates;
#          the research status table and how little of it was hand checked; broken configs are rejected; the rules
#          were committed before any new-engine code; and the AMLGentex verdict.
from typing import Any

from core import level10_check
from showcase.framework import CheckDefinition, CheckOutcome, NotRun

LEVEL_NUMBER = 10
LEVEL_TITLE = "Research catalogue, committed rules, related-work study"
NEEDS_DATABASE = False  # the checks read files and git history; no database is needed
WHAT_YOU_SAW = (
    "The evaluation rules, the world ranges and the legal actions are written down as config files, and each "
    "number says whether it is sourced, reported, an assumption or a policy. A validator accepts all of them and "
    "rejects copies that lose a reason or a source, or whose ranges do not nest. The research status table shows "
    "how many claims are confirmed, how few the author checked by hand, and how many a search tool re-read. Git "
    "history shows the rules were committed before any new-engine code, and the older v1 PaySim code is stated "
    "openly. The AMLGentex spike gave its verdict: borrow ideas and use as an outside check."
)

REQUIRED_FILES = [level10_check.EVALUATION_PATH, level10_check.WORLD_RANGES_PATH, level10_check.LEGAL_ACTIONS_PATH,
                  level10_check.STATUS_PATH, level10_check.DECISIONS_PATH, level10_check.AMLGENTEX_DECISION_PATH,
                  level10_check.VERDICT_PATH]


async def find_missing_preconditions() -> list[str]:
    """What is missing, in plain English (empty means we can run)."""
    return [f"{path.name} is missing - it is part of the Level 10 files" for path in REQUIRED_FILES if not path.exists()]


def need_git_history() -> None:
    """The proof needs git history. Say plainly if it is not there."""
    if not level10_check.run_git(["rev-parse", "--git-dir"]):
        raise NotRun("this folder is not a git repository, so the commit order cannot be checked")


def date_text(date: Any) -> str:
    """A date as text, or 'none yet' when there is no commit."""
    return date.isoformat() if date else "none yet"


# ---------------------------------------------------------------------------
# Check 1 (normal): every config validates
# ---------------------------------------------------------------------------

async def run_configs_check(workspace: Any) -> CheckOutcome:
    """The three rule files pass the shared validator, and each shows its label counts."""
    errors, counts = level10_check.check_configs()
    assert not errors, "; ".join(errors)
    summary = {name: ", ".join(f"{label} {number}" for label, number in file_counts.items())
               for name, file_counts in counts.items()}
    total = sum(sum(file_counts.values()) for file_counts in counts.values())
    return CheckOutcome(key_number=f"3 files valid, {total} labelled blocks", evidence={"labels_per_file": summary})


# ---------------------------------------------------------------------------
# Check 2 (edge): the research status table, and how little was hand checked
# ---------------------------------------------------------------------------

async def run_status_check(workspace: Any) -> CheckOutcome:
    """Show the status counts, the hand-checked count and the tool re-check counts."""
    status, hand_checked, tool = level10_check.status_counts()
    total = sum(status.values())
    assert total > 0, "the research status table is empty"
    status_text = ", ".join(f"{word} {number}" for word, number in status.items())
    tool_text = ", ".join(f"{word} {number}" for word, number in tool.items())
    return CheckOutcome(
        key_number=f"{total} findings: {status_text}; {hand_checked} hand checked; tool re-check: {tool_text}",
        evidence={"status": status, "hand_checked": hand_checked, "tool_rechecked": tool, "total": total},
    )


# ---------------------------------------------------------------------------
# Check 3 (negative): broken copies are rejected
# ---------------------------------------------------------------------------

async def run_broken_copy_check(workspace: Any) -> CheckOutcome:
    """Each broken copy (no reason, no source, bad range) must be rejected. Nothing is written to disk."""
    rejected = level10_check.broken_copy_is_rejected()
    for name, errors in rejected.items():
        assert errors, f"the copy with '{name}' was NOT rejected"
    return CheckOutcome(key_number=f"{len(rejected)} of {len(rejected)} broken copies rejected", evidence=rejected)


# ---------------------------------------------------------------------------
# Check 4 (edge): the rules were committed before any new-engine code
# ---------------------------------------------------------------------------

async def run_preregistration_check(workspace: Any) -> CheckOutcome:
    """Compare commit dates: no new-engine file may be older than config/evaluation.yaml."""
    need_git_history()
    proof = level10_check.preregistration_proof()
    assert proof["result"] == "OK", proof["sentence"]
    new_engine = {path: date_text(date) for path, date in proof["new_engine_dates"].items()}
    return CheckOutcome(
        key_number=f"{proof['result']}: {proof['sentence']}",
        evidence={"evaluation_yaml_added": date_text(proof["evaluation_date"]),
                  "decisions_doc_added": date_text(proof["decisions_date"]),
                  "v1_scripts_data_split_added": date_text(proof["v1_date"]),
                  "new_engine_files": new_engine or "none yet"},
    )


# ---------------------------------------------------------------------------
# Check 5 (normal): the AMLGentex verdict
# ---------------------------------------------------------------------------

async def run_spike_check(workspace: Any) -> CheckOutcome:
    """Show the verdict line and where the evidence and the decision are."""
    verdict = level10_check.spike_verdict_line()
    return CheckOutcome(
        key_number=verdict,
        evidence={"evidence_file": "docs/research/amlgentex_spike_verdict.md",
                  "decision_file": "docs/decisions/level-10-amlgentex-decision.md"},
    )


def get_checks(quick: bool) -> list[CheckDefinition]:
    """The Level 10 checks, in order (quick and full mode are the same: everything is a small read-only check)."""
    return [
        CheckDefinition(
            "1. every config file validates", "normal",
            "The evaluation rules, world ranges and legal actions all pass the shared validator.",
            run_configs_check, "A label, reason or source was removed or edited in a config file.",
            "python -m core.config_validator config/evaluation.yaml config/world_ranges.yaml config/legal_actions.yaml"),
        CheckDefinition(
            "2. research status table counts", "edge",
            "The table of research claims shows its status counts, how many were hand checked and tool re-checked.",
            run_status_check, "config/research_status.yaml is missing or empty.", "config/research_status.yaml"),
        CheckDefinition(
            "3. broken config copies are rejected", "negative",
            "A block without its reason, a sourced block without its source and a range outside its test range are refused.",
            run_broken_copy_check, "The validator no longer enforces reasons, sources or nested ranges.",
            "core/config_validator.py"),
        CheckDefinition(
            "4. rules were committed before any new-engine code", "edge",
            "No new-engine file is older than config/evaluation.yaml; the older v1 code is stated openly.",
            run_preregistration_check, "A file was added under a new-engine path before the rules.",
            "git log --diff-filter=A -- config/evaluation.yaml"),
        CheckDefinition(
            "5. AMLGentex spike verdict", "normal",
            "The spike decided between building on AMLGentex and only borrowing ideas from it.",
            run_spike_check, "The verdict file lost its single 'Verdict:' line.",
            "docs/research/amlgentex_spike_verdict.md"),
    ]
