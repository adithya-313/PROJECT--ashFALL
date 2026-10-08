# File: showcase/summary.py
# Purpose: Finds the level showcases that exist and builds the one-page summary (reports/showcase/index.html) that
#          lists every planned level with its latest result, or "not built yet".
import importlib
import re
from dataclasses import dataclass
from html import escape
from pathlib import Path
from types import ModuleType

import orjson

SHOWCASE_DIR = Path(__file__).resolve().parent
REPORTS_DIR = SHOWCASE_DIR.parent / "reports" / "showcase"

# Add the other levels from the master plan here so the summary can show "not built yet" rows.
PLANNED_LEVELS: dict[int, str] = {
    7: "Transactional Integrity (Outbox)",
}

LEVEL_FILE_PATTERN = re.compile(r"^level_(\d\d)\.py$")

SUMMARY_STYLE = """
body { font-family: Arial, sans-serif; margin: 24px auto; max-width: 900px; padding: 0 16px; color: #1a1a1a; }
table { border-collapse: collapse; width: 100%; }
th, td { text-align: left; padding: 8px; border-bottom: 1px solid #ccc; font-size: 14px; }
.status { font-weight: bold; }
"""


@dataclass
class SummaryRow:
    """One line of the summary page."""

    level: int
    title: str
    status: str  # "passed", "failed", "not run yet" or "not built yet"
    date: str
    code_version: str
    link: str  # relative link to that level's index.html, or "" when there is none


def level_folder_name(level_number: int) -> str:
    """Folder name for a level's report, always two digits: level_07."""
    return f"level_{level_number:02d}"


def discover_level_modules() -> dict[int, ModuleType]:
    """Find showcase/level_NN.py files (NN = two digits), import each, and return them by level number."""
    modules: dict[int, ModuleType] = {}
    for path in sorted(SHOWCASE_DIR.glob("level_*.py")):
        match = LEVEL_FILE_PATTERN.match(path.name)
        if match is None:
            continue
        modules[int(match.group(1))] = importlib.import_module(f"showcase.{path.stem}")
    return modules


def read_result_json(level_number: int, reports_dir: Path) -> dict | None:
    """Read that level's result.json, or return None if there is no (readable) file yet."""
    result_path = reports_dir / level_folder_name(level_number) / "result.json"
    if not result_path.exists():
        return None
    try:
        return orjson.loads(result_path.read_bytes())
    except orjson.JSONDecodeError:
        return None


def make_row_for_level(level_number: int, title: str, has_module: bool, reports_dir: Path) -> SummaryRow:
    """Work out the summary row for one level."""
    if not has_module:
        return SummaryRow(level_number, title, "not built yet", "", "", "")
    result = read_result_json(level_number, reports_dir)
    if result is None:
        return SummaryRow(level_number, title, "not run yet", "", "", "")
    overall = result["stable"]["overall_status"]
    status = "passed" if overall == "PASSED" else "failed"
    link = f"{level_folder_name(level_number)}/index.html"
    return SummaryRow(level_number, title, status, result["run_info"]["finished_at"],
                      result["run_info"]["code_version"], link)


def build_summary_rows(modules: dict[int, ModuleType], reports_dir: Path = REPORTS_DIR) -> list[SummaryRow]:
    """One row for every planned level and every level that has a module, in numeric order."""
    titles = dict(PLANNED_LEVELS)
    for level_number, module in modules.items():
        titles[level_number] = module.LEVEL_TITLE
    rows = []
    for level_number in sorted(titles):
        rows.append(make_row_for_level(level_number, titles[level_number], level_number in modules, reports_dir))
    return rows


def make_row_html(row: SummaryRow) -> str:
    """One table row, every dynamic string escaped."""
    title = escape(row.title)
    if row.link:
        title = f'<a href="{escape(row.link)}">{title}</a>'
    return (f"<tr><td>{row.level}</td><td>{title}</td><td class=\"status\">{escape(row.status)}</td>"
            f"<td>{escape(row.date)}</td><td>{escape(row.code_version)}</td></tr>")


def write_summary_page(rows: list[SummaryRow], path: Path) -> None:
    """Write the self-contained summary page (no network, no external files)."""
    body = "".join(make_row_html(row) for row in rows)
    page = (
        '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>Showcase summary</title>'
        f"<style>{SUMMARY_STYLE}</style></head><body><h1>Showcase summary</h1>"
        "<table><tr><th>Level</th><th>Title</th><th>Status</th><th>Date</th><th>Code version</th></tr>"
        f"{body}</table></body></html>"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf-8")
