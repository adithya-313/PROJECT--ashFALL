# File: tests/unit/test_showcase_cli.py
# Purpose: Proves the summary page rows, the page's escaping, the command-line exit code for an unknown level, and
#          the argument parsing.
from pathlib import Path
from types import SimpleNamespace

import orjson

from showcase.__main__ import main, parse_args
from showcase.summary import build_summary_rows, write_summary_page


def write_result_json(reports_dir: Path, level_number: int, overall_status: str) -> None:
    """Create a small result.json like the report writer would."""
    folder = reports_dir / f"level_{level_number:02d}"
    folder.mkdir(parents=True)
    data = {"stable": {"overall_status": overall_status},
            "run_info": {"finished_at": "2026-01-01T00:00:00", "code_version": "abc1234"}}
    (folder / "result.json").write_bytes(orjson.dumps(data))


def test_row_for_level_with_module_and_result_shows_real_status(tmp_path: Path):
    """A level with a module and a result.json shows its real status and a link to its report."""
    write_result_json(tmp_path, 7, "PASSED")
    modules = {7: SimpleNamespace(LEVEL_TITLE="Seven")}
    rows = build_summary_rows(modules, tmp_path)  # type: ignore[arg-type]
    seven = [row for row in rows if row.level == 7][0]
    assert seven.status == "passed"
    assert seven.link == "level_07/index.html"
    assert seven.code_version == "abc1234"


def test_failed_result_and_missing_result_are_reported_correctly(tmp_path: Path):
    """A failed result shows 'failed'; a module with no result file yet shows 'not run yet'."""
    write_result_json(tmp_path, 7, "FAILED")
    modules = {7: SimpleNamespace(LEVEL_TITLE="Seven"), 8: SimpleNamespace(LEVEL_TITLE="Eight")}
    rows = {row.level: row for row in build_summary_rows(modules, tmp_path)}  # type: ignore[arg-type]
    assert rows[7].status == "failed"
    assert rows[8].status == "not run yet"


def test_planned_level_without_module_is_not_built_yet(tmp_path: Path):
    """A planned level that has no showcase module shows 'not built yet'."""
    rows = build_summary_rows({}, tmp_path)
    assert [row.status for row in rows] == ["not built yet"]


def test_summary_page_escapes_dangerous_titles(tmp_path: Path):
    """A level title with a script tag is shown as text and never as a real tag."""
    write_result_json(tmp_path, 9, "PASSED")
    modules = {9: SimpleNamespace(LEVEL_TITLE="<script>alert(1)</script>")}
    rows = build_summary_rows(modules, tmp_path)  # type: ignore[arg-type]
    page_path = tmp_path / "index.html"
    write_summary_page(rows, page_path)
    html = page_path.read_text(encoding="utf-8")
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert "http://" not in html and "https://" not in html


def test_unknown_level_returns_exit_code_two_and_says_so(capsys):
    """Asking for a level with no showcase prints a clear message and returns exit code 2."""
    exit_code = main(["--level", "99"])
    assert exit_code == 2
    assert "No showcase exists for level 99" in capsys.readouterr().out


def test_flags_are_parsed():
    """--quick and --allow-not-run are read when given, and are off when not given."""
    with_flags = parse_args(["--level", "7", "--quick", "--allow-not-run"])
    assert with_flags.level == 7 and with_flags.quick is True and with_flags.allow_not_run is True
    without_flags = parse_args(["--all"])
    assert without_flags.all is True and without_flags.quick is False and without_flags.allow_not_run is False
