# File: tests/unit/test_showcase_reporting.py
# Purpose: Proves the report writers create the expected files, escape dangerous text, load nothing from the
#          network, keep the 'stable' part of result.json identical between runs, and draw the chart safely.
from dataclasses import replace
from pathlib import Path

import orjson

from showcase.framework import CheckResult, LevelResult
from showcase.reporting import make_timeline_chart_svg, write_level_report

EVIL_NAME = "<script>alert(1)</script>"


def make_result(name: str, status: str, seconds: float = 0.5, chart: dict | None = None) -> CheckResult:
    """Build one check result by hand."""
    explanation = "something went wrong" if status != "PASS" else ""
    return CheckResult(name, "normal", f"{name} works.", status, "3 of 3", explanation, {"count": 3}, chart, seconds)


def make_level(check_results: list[CheckResult], seconds: float = 1.0,
               finished_at: str = "2026-01-01T00:00:00") -> LevelResult:
    """Build a level result by hand."""
    return LevelResult(7, "Demo level", "quick", "You saw a demo.", check_results, seconds, "abc1234", finished_at,
                       "PASSED")


def test_report_files_are_created(tmp_path: Path):
    """Writing a report creates index.html, result.json and one evidence file per check."""
    level = make_level([make_result("first check", "PASS"), make_result("second check", "FAIL")])
    write_level_report(level, tmp_path)
    assert (tmp_path / "index.html").exists()
    assert (tmp_path / "result.json").exists()
    evidence_names = sorted(path.name for path in (tmp_path / "evidence").iterdir())
    assert evidence_names == ["01_first-check.json", "02_second-check.json"]


def test_html_has_names_and_status_words_and_no_external_links(tmp_path: Path):
    """The page shows each check name and status as words and never refers to an outside web address."""
    level = make_level([make_result("first check", "PASS"), make_result("second check", "NOT RUN")])
    write_level_report(level, tmp_path)
    html = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert "first check" in html and "second check" in html
    assert "PASS" in html and "NOT RUN" in html
    assert "http://" not in html and "https://" not in html
    assert 'href="evidence/01_first-check.json"' in html


def test_dangerous_check_name_is_escaped(tmp_path: Path):
    """A check named with a script tag appears as harmless text, never as a real tag."""
    chart = {"type": "timeline", "points": [[0, 1], [1, 2]], "shaded": [[0, 1, EVIL_NAME]], "title": EVIL_NAME,
             "x_label": "s", "y_label": "n"}
    level = make_level([make_result(EVIL_NAME, "PASS", chart=chart)])
    write_level_report(level, tmp_path)
    html = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_stable_part_of_result_json_ignores_timings_and_dates(tmp_path: Path):
    """Two runs that differ only in seconds and finished_at have an identical 'stable' section."""
    first = make_level([make_result("a", "PASS", seconds=0.1)], seconds=1.0, finished_at="2026-01-01T00:00:00")
    second = make_level([make_result("a", "PASS", seconds=9.9)], seconds=7.0, finished_at="2026-02-02T12:00:00")
    write_level_report(first, tmp_path / "one")
    write_level_report(replace(second), tmp_path / "two")
    data_one = orjson.loads((tmp_path / "one" / "result.json").read_bytes())
    data_two = orjson.loads((tmp_path / "two" / "result.json").read_bytes())
    assert data_one["stable"] == data_two["stable"]
    assert data_one["run_info"] != data_two["run_info"]


def test_chart_contains_shaded_label_and_handles_empty_points():
    """The chart draws the shaded band's label, and shows 'no data' instead of crashing when empty."""
    svg = make_timeline_chart_svg([(0, 0), (5, 10)], [(2, 4, "broker down")], "Events", "seconds", "count")
    assert "broker down" in svg
    assert "<polyline" in svg
    empty = make_timeline_chart_svg([], [], "Events", "seconds", "count")
    assert "no data" in empty
