"""
WHAT: Unit tests for pattern files: the basic example validates and previews, each kind of bad file is refused with
a plain reason, the preview is repeatable, and text in a pattern is kept as data.
Expected values come from the example YAML files or from construction, never from the code under test.

WHY: A pattern file is written by hand, so the first mistake in it should be named at once, and a pattern's text must
never be run as code.

TERM: preview: a plan of a pattern's gang in one small training world; it is an estimate, not a run.
"""

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

from world.gang_planner import GangPlan
from world.pattern_files import (
    PatternFileError,
    PatternPreview,
    load_pattern_file,
    plan_pattern,
    validate_pattern_file,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
BASIC_PATH = REPO_ROOT / "patterns" / "examples" / "scam_ring_basic.yaml"
OVER_BUDGET_PATH = REPO_ROOT / "patterns" / "examples" / "scam_ring_over_budget.yaml"


def _basic_document() -> dict[str, Any]:
    """Reads the basic example straight with yaml, as a fresh copy a test may change."""
    with open(BASIC_PATH, encoding="utf-8") as handle:
        document: dict[str, Any] = yaml.safe_load(handle)
    return copy.deepcopy(document)


def _write(tmp_path: Path, name: str, document: dict[str, Any]) -> Path:
    """Writes a document as YAML into the temporary folder and returns its path."""
    path = tmp_path / name
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def basic_preview() -> PatternPreview:
    """Validates the basic example once, so the preview world is planned only once for the module."""
    return validate_pattern_file(BASIC_PATH)


@pytest.fixture(scope="module")
def basic_plan() -> GangPlan:
    """Plans the basic example once more, to give the tests the plan itself to check the preview against."""
    return plan_pattern(load_pattern_file(BASIC_PATH))


def test_basic_example_validates_and_previews(basic_preview: PatternPreview) -> None:
    """The summary text names a mule count and a weekly figure and marks itself as an estimate."""
    text = basic_preview.summary_text()
    assert "about" in text
    assert "per week" in text
    assert "ESTIMATE" in text


def test_mule_count_is_bounded_and_every_mule_receives_a_payment(
    basic_preview: PatternPreview, basic_plan: GangPlan
) -> None:
    """Mules are at most the accounts acquired, at least one, and each counted mule receives a planned payment."""
    steps = _basic_document()["pattern"]["gang"]["steps"]
    acquired = sum(step["params"]["count"] for step in steps if step["block"] == "acquire")
    assert 1 <= basic_preview.mule_accounts <= acquired
    assert basic_preview.mule_accounts == len(basic_preview.mule_account_ids)
    receivers = {payment.to_account for payment in basic_plan.payments}
    for account_id in basic_preview.mule_account_ids:
        assert account_id in receivers


def test_money_taken_in_equals_the_victim_payments(basic_preview: PatternPreview, basic_plan: GangPlan) -> None:
    """The basic example is scam-funded, so money entering the mules equals the sum of its victim payments."""
    victim_total = 0
    for payment, story in zip(basic_plan.payments, basic_plan.truth.payment_stories):
        if story.scam_step == "victim_payment":
            victim_total += payment.amount_paise
    assert victim_total > 0
    assert basic_preview.money_taken_in_paise == victim_total
    assert basic_preview.money_per_week_paise == victim_total * 7 // basic_preview.span_days


def test_invalid_yaml_is_refused(tmp_path: Path) -> None:
    """Text that is not YAML must be refused, with the file name first."""
    path = tmp_path / "broken.yaml"
    path.write_text("pattern: [unclosed\n  - : :", encoding="utf-8")
    with pytest.raises(PatternFileError, match="broken.yaml: malformed: not valid YAML"):
        load_pattern_file(path)


def test_missing_gang_is_refused(tmp_path: Path) -> None:
    """A pattern block without a gang must be refused by naming the missing key."""
    document = _basic_document()
    del document["pattern"]["gang"]
    with pytest.raises(PatternFileError, match="malformed: missing key 'gang'"):
        load_pattern_file(_write(tmp_path, "no_gang.yaml", document))


def test_unknown_top_level_key_is_refused(tmp_path: Path) -> None:
    """An extra top-level key must be refused so a typo cannot be silently ignored."""
    document = _basic_document()
    document["patern"] = {}
    with pytest.raises(PatternFileError, match="malformed: unknown key 'patern'"):
        load_pattern_file(_write(tmp_path, "typo.yaml", document))


def test_wrong_version_is_refused(tmp_path: Path) -> None:
    """A version other than the supported one must be refused, naming the version."""
    document = _basic_document()
    document["pattern"]["pattern_version"] = 2
    with pytest.raises(PatternFileError, match="unsupported version 2"):
        load_pattern_file(_write(tmp_path, "version.yaml", document))


def test_unknown_block_is_refused_with_the_gang_wording(tmp_path: Path) -> None:
    """An unknown tactic block keeps the wording of the gang validator, prefixed with 'invalid gang:'."""
    document = _basic_document()
    document["pattern"]["gang"]["steps"][0]["block"] = "no_such_block"
    with pytest.raises(PatternFileError, match="invalid gang: .*unknown block"):
        load_pattern_file(_write(tmp_path, "block.yaml", document))


def test_missing_budget_and_bad_steps_give_clear_reasons(tmp_path: Path) -> None:
    """A missing budget key and a steps value that is not a list are named, not left as KeyError or TypeError."""
    no_budget = _basic_document()
    del no_budget["pattern"]["gang"]["budget"]
    with pytest.raises(PatternFileError, match="malformed: missing key 'budget'"):
        load_pattern_file(_write(tmp_path, "no_budget.yaml", no_budget))
    bad_steps = _basic_document()
    bad_steps["pattern"]["gang"]["steps"] = 5
    with pytest.raises(PatternFileError, match="malformed: 'steps' must be a list"):
        load_pattern_file(_write(tmp_path, "bad_steps.yaml", bad_steps))


def test_over_budget_example_is_refused() -> None:
    """The over-budget example loads but planning it must be refused with 'over budget:'."""
    with pytest.raises(PatternFileError, match="scam_ring_over_budget.yaml: over budget:"):
        validate_pattern_file(OVER_BUDGET_PATH)


def test_preview_is_deterministic(basic_preview: PatternPreview) -> None:
    """Validating the same example again gives an equal PatternPreview record."""
    assert validate_pattern_file(BASIC_PATH) == basic_preview


def test_pattern_text_is_kept_as_data(tmp_path: Path) -> None:
    """A description that looks like code loads and stays the same string; loading never runs it."""
    document = _basic_document()
    hostile = "{__import__('os')}"
    document["pattern"]["description"] = hostile
    loaded = load_pattern_file(_write(tmp_path, "text.yaml", document))
    assert loaded.description == hostile
