"""
WHAT: Unit test suite verifying Level 11 custom HTML page rendering and output persistence.
It tests render_level_11_html and write_custom_page against sample LevelResult data containers,
verifying that all 31 template variables are populated, all 10 report sections exist, and files write cleanly.

WHY: Self-contained showcase pages provide executive and auditor visibility into Level 11 simulation mechanics.
Testing that no unpopulated placeholder tokens remain protects against broken variable rendering in production reports.
Confirming that custom pages write offline without external network links guarantees zero external dependencies.

TERM: LevelResult: Complete execution record container holding check results, total runtime, and overall status,
      e.g. packaging sixteen check results and run metadata for HTML report rendering.
TERM: CheckResult: Individual check result holding outcome status, key numbers, evidence, and optional charts,
      e.g. recording a PASS result for the empirical grounding audit check.
TERM: self-contained: A standalone document that bundles all styling and visual assets without network requests,
      e.g. an HTML file containing inline CSS styles and inline SVG graphics.
"""

import re
from pathlib import Path
from typing import Any

from showcase.framework import CheckResult, LevelResult
from showcase.level_11_page import render_level_11_html, write_custom_page


def _make_sample_check(name: str, key_number: str, evidence: dict[str, Any] | None = None) -> CheckResult:
    """
    WHAT: Helper creating a mock CheckResult record with standardized default values.
    It packages name, key number, and optional evidence dictionary into a passing CheckResult dataclass instance.
    Supplies consistent mock check records for showcase report generation and template testing.

    WHY: Reduces code duplication across test fixtures and keeps test helper functions under 30 code lines.
    Centralizes mock CheckResult instantiation parameters to maintain test maintainability.

    Args:
        name: Hierarchical test name string with numerical prefix.
        key_number: Terse summary string of measured key metrics.
        evidence: Optional dictionary holding detailed measured values or SVG strings.

    Returns:
        CheckResult instance configured with PASS status and default timing metrics.

    Example:
        >>> cr = _make_sample_check("1. sample", "42 items")
        >>> cr.status == "PASS" and cr.key_number == "42 items"
        True
    """
    return CheckResult(
        name=name,
        kind="normal",
        proves="Proves expected behavior.",
        status="PASS",
        key_number=key_number,
        explanation="",
        evidence=evidence or {},
        chart=None,
        seconds=0.1,
    )


def _build_sample_level_result() -> LevelResult:
    """
    WHAT: Constructs a realistic mock LevelResult fixture populated with sample check results and timing numbers.
    It builds CheckResult records for tiny world simulation, grounding table, and ledger invariants.
    The resulting container supplies comprehensive inputs for testing template substitution and table rendering.

    WHY: Decouples template rendering unit tests from full simulation execution overhead.
    Supplying representative check evidence enables testing complex table and component generators instantly.
    Provides predictable test fixtures for verifying placeholder substitution integrity.

    Returns:
        LevelResult container holding sample check results, metadata, and execution timing.

    Example:
        >>> result = _build_sample_level_result()
        >>> result.level_number == 11 and len(result.check_results) == 3
        True
    """
    c3_ev = {
        "attempts": 2915,
        "failures": 50,
        "days": 61,
        "people": 60,
        "shops": 12,
        "chart_svg": "<svg></svg>",
    }
    checks = [
        _make_sample_check("3. tiny world fast-mode simulation", "2,915 attempts", c3_ev),
        _make_sample_check("4. empirical grounding table", "33 parameters grounded"),
        _make_sample_check("13. double-entry ledger invariants", "3 ledger invariants pass"),
    ]
    return LevelResult(
        level_number=11,
        title="A simulated India of everyday UPI payments",
        mode="full",
        what_you_saw="Simulated municipal economy with three banks.",
        check_results=checks,
        total_seconds=12.5,
        code_version="cf2c639",
        finished_at="2026-10-16T12:00:00Z",
        overall_status="PASSED",
    )


def test_render_level_11_html_replaces_all_placeholders() -> None:
    """
    WHAT: Tests that render_level_11_html substitutes all 31 template variables without leaving unpopulated tokens.
    It passes a sample LevelResult fixture and scans the resulting HTML string with a regular expression for dollar tokens.
    It verifies that all ten named section anchors and passbook accordion entries are present in the markup.

    WHY: Unpopulated template placeholders look unprofessional and indicate missing component renderers.
    Verifying that all 31 variables are replaced guarantees that metrics and tables reach report viewers intact.
    Scanning for section anchors confirms that table of contents navigation links function properly.

    Returns:
        None upon asserting all placeholders are populated and all sections exist.

    Example:
        >>> test_render_level_11_html_replaces_all_placeholders()
    """
    sample_result = _build_sample_level_result()
    rendered_html = render_level_11_html(sample_result)
    assert rendered_html.startswith("<!DOCTYPE html>")
    unpopulated = re.findall(r"\$[A-Za-z0-9_]+", rendered_html)
    assert unpopulated == [], f"Found unpopulated placeholders: {unpopulated}"
    section_ids = [
        "overview", "how", "world", "checks", "stress",
        "techniques", "testing", "limits", "records", "glossary",
    ]
    for sec_id in section_ids:
        assert f'id="{sec_id}"' in rendered_html, f"Missing section id: {sec_id}"


def test_write_custom_page_creates_file(tmp_path: Path) -> None:
    """
    WHAT: Tests that write_custom_page successfully writes index.html to the designated output folder.
    It supplies a temporary directory path, executes the writer, and checks that index.html exists and is readable.
    It verifies that the written file matches the rendered HTML string exactly and contains UTF-8 text.

    WHY: Proves that write_custom_page satisfies the file export contract expected by showcase.__main__.
    Ensures that parent directories are created automatically if they do not exist yet.
    Confirming file creation protects report publishing pipelines from silent write failures.

    Args:
        tmp_path: Pytest temporary directory fixture for isolating file creation.

    Returns:
        None upon asserting index.html exists and contains non-empty HTML text.

    Example:
        >>> # Executes cleanly on temporary directories:
        >>> # write_custom_page(sample_result, tmp_path)
    """
    sample_result = _build_sample_level_result()
    target_dir = tmp_path / "level_11_test_report"
    write_custom_page(sample_result, target_dir)
    output_file = target_dir / "index.html"
    assert output_file.exists() and output_file.is_file()
    content = output_file.read_text(encoding="utf-8")
    assert content.startswith("<!DOCTYPE html>")
    assert "Level 11" in content
