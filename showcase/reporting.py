# File: showcase/reporting.py
# Purpose: Writes a level's results to disk: one self-contained index.html (opens offline, no network), a
#          result.json, and one evidence JSON file per check. Also draws the simple timeline chart as inline SVG.
#
# Everything dynamic that goes into HTML passes through html.escape, so a check named "<script>..." is shown as
# text and never runs.
import re
from html import escape
from pathlib import Path
from typing import Any

import orjson

from showcase.framework import CheckResult, LevelResult

# Chart size and margins, in SVG pixels.
CHART_WIDTH = 640
CHART_HEIGHT = 280
MARGIN_LEFT = 56
MARGIN_RIGHT = 16
MARGIN_TOP = 32
MARGIN_BOTTOM = 44

PAGE_STYLE = """
body { font-family: Arial, sans-serif; margin: 24px auto; max-width: 900px; padding: 0 16px; color: #1a1a1a; }
h1 { font-size: 22px; margin-bottom: 4px; }
.meta { color: #555; font-size: 14px; margin-bottom: 16px; }
.banner { padding: 10px 14px; font-weight: bold; border: 2px solid #333; margin-bottom: 20px; }
.banner.PASSED { background: #e3f4e3; border-color: #2d7a2d; }
.banner.FAILED { background: #fbe3e3; border-color: #a02020; }
table { border-collapse: collapse; width: 100%; }
th, td { text-align: left; padding: 8px; border-bottom: 1px solid #ccc; vertical-align: top; font-size: 14px; }
.status { font-weight: bold; white-space: nowrap; }
.status.PASS { color: #1d6b1d; }
.status.FAIL { color: #a02020; }
.status.NOTRUN { color: #8a5a00; }
.proves { color: #444; }
.why { color: #a02020; font-size: 13px; }
.story { background: #f4f4f4; padding: 12px; margin: 20px 0; }
svg { max-width: 100%; height: auto; border: 1px solid #ccc; background: #fff; }
"""


def make_slug(text: str) -> str:
    """Turn a check name into a safe file-name piece: lowercase letters, digits and dashes only."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "check"


def evidence_file_name(index: int, check_name: str) -> str:
    """File name for one check's evidence, such as 01_events-reach-kafka.json."""
    return f"{index:02d}_{make_slug(check_name)}.json"


def to_json_bytes(data: Any) -> bytes:
    """Encode to pretty JSON. default=str means UUIDs, Decimals and datetimes are written as text, never a crash."""
    return orjson.dumps(data, default=str, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS)


def status_css_class(status: str) -> str:
    """CSS class name for a status word ('NOT RUN' has a space, class names cannot)."""
    return status.replace(" ", "")


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------

def scale_value(value: float, low: float, high: float, out_low: float, out_high: float) -> float:
    """Map a value from the range low..high onto the range out_low..out_high (no divide-by-zero)."""
    if high == low:
        return out_low
    return out_low + (value - low) / (high - low) * (out_high - out_low)


def make_empty_chart_svg(title: str) -> str:
    """The chart shown when there are no points: a box that says 'no data'."""
    return (f'<svg viewBox="0 0 {CHART_WIDTH} {CHART_HEIGHT}" role="img" aria-label="{escape(title)}">'
            f'<text x="{CHART_WIDTH // 2}" y="{CHART_HEIGHT // 2}" text-anchor="middle">no data</text></svg>')


def make_shaded_band_svg(start: float, end: float, label: str, x_low: float, x_high: float) -> str:
    """One light band (for example 'broker down') with its label at the top."""
    left = scale_value(start, x_low, x_high, MARGIN_LEFT, CHART_WIDTH - MARGIN_RIGHT)
    right = scale_value(end, x_low, x_high, MARGIN_LEFT, CHART_WIDTH - MARGIN_RIGHT)
    height = CHART_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM
    return (f'<rect x="{left:.1f}" y="{MARGIN_TOP}" width="{max(right - left, 1):.1f}" height="{height}" '
            f'fill="#f0c9c9" opacity="0.6"/>'
            f'<text x="{left + 4:.1f}" y="{MARGIN_TOP + 14}" font-size="12">{escape(label)}</text>')


def make_timeline_chart_svg(
    points: list[tuple[float, float]],
    shaded_regions: list[tuple[float, float, str]],
    title: str,
    x_label: str,
    y_label: str,
) -> str:
    """Build a simple line chart (x = seconds, y = a count) as an inline SVG string. Standard library only."""
    if not points:
        return make_empty_chart_svg(title)

    # Work out the ranges so everything (line and bands) fits inside the plot area.
    x_values = [x for x, _ in points]
    for start, end, _ in shaded_regions:
        x_values.extend([start, end])
    x_low, x_high = min(x_values), max(x_values)
    y_high = max(max(y for _, y in points), 1)

    parts = [f'<svg viewBox="0 0 {CHART_WIDTH} {CHART_HEIGHT}" role="img" aria-label="{escape(title)}">']
    parts.append(f'<text x="{MARGIN_LEFT}" y="18" font-size="14" font-weight="bold">{escape(title)}</text>')

    # Light bands first, so the line is drawn on top of them.
    for start, end, label in shaded_regions:
        parts.append(make_shaded_band_svg(start, end, label, x_low, x_high))

    # The line itself.
    coordinates = []
    for x, y in points:
        pixel_x = scale_value(x, x_low, x_high, MARGIN_LEFT, CHART_WIDTH - MARGIN_RIGHT)
        pixel_y = scale_value(y, 0, y_high, CHART_HEIGHT - MARGIN_BOTTOM, MARGIN_TOP)
        coordinates.append(f"{pixel_x:.1f},{pixel_y:.1f}")
    parts.append(f'<polyline points="{" ".join(coordinates)}" fill="none" stroke="#1f5fa8" stroke-width="2"/>')

    # Axes and their labels, with the largest value written on the y axis.
    bottom = CHART_HEIGHT - MARGIN_BOTTOM
    right_edge = CHART_WIDTH - MARGIN_RIGHT
    parts.append(f'<line x1="{MARGIN_LEFT}" y1="{bottom}" x2="{right_edge}" y2="{bottom}" stroke="#333"/>')
    parts.append(f'<line x1="{MARGIN_LEFT}" y1="{MARGIN_TOP}" x2="{MARGIN_LEFT}" y2="{bottom}" stroke="#333"/>')
    parts.append(f'<text x="{MARGIN_LEFT - 6}" y="{MARGIN_TOP + 4}" font-size="11" text-anchor="end">{y_high:g}</text>')
    parts.append(f'<text x="{MARGIN_LEFT - 6}" y="{bottom}" font-size="11" text-anchor="end">0</text>')
    parts.append(f'<text x="{CHART_WIDTH // 2}" y="{CHART_HEIGHT - 8}" font-size="12" text-anchor="middle">'
                 f'{escape(x_label)} ({x_low:g} to {x_high:g})</text>')
    parts.append(f'<text x="14" y="{CHART_HEIGHT // 2}" font-size="12" text-anchor="middle" '
                 f'transform="rotate(-90 14 {CHART_HEIGHT // 2})">{escape(y_label)}</text>')
    parts.append("</svg>")
    return "".join(parts)


SCATTER_COLORS = ("#1f5fa8", "#c8501e", "#2a8a4a", "#8a3fa0")
SCATTER_MARKERS = ("circle", "square", "diamond", "triangle")
SCATTER_GLYPHS = ("&#9679;", "&#9632;", "&#9670;", "&#9650;")


def make_scatter_chart_svg(
    series: list[tuple[str, list[tuple[float, float]]]], title: str, x_label: str, y_label: str
) -> str:
    """
    Build a scatter plot as an inline SVG string, standard library only: one marker shape and colour per series
    (circle, square, diamond, triangle, then repeating), axes with their lowest and highest values, and a legend.
    """
    all_points = [point for _, points in series for point in points]
    if not all_points:
        return make_empty_chart_svg(title)
    x_low, x_high = min(x for x, _ in all_points), max(x for x, _ in all_points)
    y_low, y_high = min(y for _, y in all_points), max(y for _, y in all_points)
    left, right = MARGIN_LEFT, CHART_WIDTH - MARGIN_RIGHT
    top, bottom = MARGIN_TOP, CHART_HEIGHT - MARGIN_BOTTOM
    parts = [f'<svg viewBox="0 0 {CHART_WIDTH} {CHART_HEIGHT}" role="img" aria-label="{escape(title)}">']
    parts.append(f'<text x="{MARGIN_LEFT}" y="18" font-size="14" font-weight="bold">{escape(title)}</text>')
    for number, (name, points) in enumerate(series):
        color = SCATTER_COLORS[number % len(SCATTER_COLORS)]
        shape = SCATTER_MARKERS[number % len(SCATTER_MARKERS)]
        for x, y in points:
            pixel_x = scale_value(x, x_low, x_high, left, right)
            pixel_y = scale_value(y, y_low, y_high, bottom, top)
            parts.append(_scatter_marker(shape, pixel_x, pixel_y, color))
        legend_x = left + 8 + 150 * number
        # The legend uses a text glyph of the same shape, so a series' circles in the plot are only its data points.
        glyph = SCATTER_GLYPHS[number % len(SCATTER_GLYPHS)]
        parts.append(f'<text x="{legend_x}" y="{top + 2}" font-size="12" fill="{color}">{glyph}</text>')
        parts.append(f'<text x="{legend_x + 14}" y="{top + 2}" font-size="11">{escape(name)}</text>')
    parts.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#333"/>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#333"/>')
    parts.append(f'<text x="{left - 6}" y="{top + 4}" font-size="11" text-anchor="end">{y_high:g}</text>')
    parts.append(f'<text x="{left - 6}" y="{bottom}" font-size="11" text-anchor="end">{y_low:g}</text>')
    parts.append(f'<text x="{CHART_WIDTH // 2}" y="{CHART_HEIGHT - 8}" font-size="12" text-anchor="middle">'
                 f'{escape(x_label)} ({x_low:g} to {x_high:g})</text>')
    parts.append(f'<text x="14" y="{CHART_HEIGHT // 2}" font-size="12" text-anchor="middle" '
                 f'transform="rotate(-90 14 {CHART_HEIGHT // 2})">{escape(y_label)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def _scatter_marker(shape: str, x: float, y: float, color: str) -> str:
    """One marker of a scatter plot: a circle, a square, a diamond or a triangle, in the series' colour."""
    if shape == "circle":
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{color}" opacity="0.8"/>'
    if shape == "square":
        return f'<rect x="{x - 3.5:.1f}" y="{y - 3.5:.1f}" width="7" height="7" fill="{color}" opacity="0.8"/>'
    if shape == "diamond":
        points = f"{x:.1f},{y - 4.5:.1f} {x + 4.5:.1f},{y:.1f} {x:.1f},{y + 4.5:.1f} {x - 4.5:.1f},{y:.1f}"
    else:
        points = f"{x:.1f},{y - 4.5:.1f} {x + 4.5:.1f},{y + 3.5:.1f} {x - 4.5:.1f},{y + 3.5:.1f}"
    return f'<polygon points="{points}" fill="{color}" opacity="0.8"/>'


def render_chart(chart: dict[str, Any]) -> str:
    """
    Turn the chart dict a check attached into SVG text: {"type": "timeline", ...} as before, or
    {"type": "scatter", "title", "x_label", "y_label", "series": [{"name", "points": [[x, y], ...]}, ...]}.
    """
    if chart.get("type") == "scatter":
        series = [
            (str(item["name"]), [(float(x), float(y)) for x, y in item["points"]]) for item in chart.get("series", [])
        ]
        return make_scatter_chart_svg(series, str(chart.get("title", "")), str(chart.get("x_label", "")),
                                      str(chart.get("y_label", "")))
    if chart.get("type") != "timeline":
        return f"<p>Unknown chart type: {escape(str(chart.get('type')))}</p>"
    points = [(float(x), float(y)) for x, y in chart.get("points", [])]
    shaded = [(float(a), float(b), str(label)) for a, b, label in chart.get("shaded", [])]
    return make_timeline_chart_svg(points, shaded, str(chart.get("title", "")), str(chart.get("x_label", "")),
                                   str(chart.get("y_label", "")))


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

def make_check_row_html(index: int, result: CheckResult) -> str:
    """One table row for one check, with every dynamic string escaped."""
    css_class = status_css_class(result.status)
    why = f'<div class="why">{escape(result.explanation)}</div>' if result.explanation else ""
    evidence_link = escape(f"evidence/{evidence_file_name(index, result.name)}")
    return (
        "<tr>"
        f"<td>{escape(result.name)}</td>"
        f"<td>{escape(result.kind)}</td>"
        f'<td class="status {css_class}">{escape(result.status)}</td>'
        f"<td>{escape(result.key_number)}</td>"
        f'<td><span class="proves">{escape(result.proves)}</span>{why}'
        f'<div><a href="{evidence_link}">evidence</a></div></td>'
        "</tr>"
    )


def make_charts_html(level_result: LevelResult) -> str:
    """A section for every check that attached a chart (empty text if none did)."""
    sections = []
    for result in level_result.check_results:
        if result.chart:
            sections.append(f"<h2>Chart: {escape(result.name)}</h2>{render_chart(result.chart)}")
    return "".join(sections)


def make_level_page_html(level_result: LevelResult) -> str:
    """The whole index.html for one level, as one string."""
    rows = "".join(make_check_row_html(index, result)
                   for index, result in enumerate(level_result.check_results, start=1))
    title = f"Level {level_result.level_number}: {level_result.title}"
    status_word = escape(level_result.overall_status)
    return (
        '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">'
        f"<title>{escape(title)}</title><style>{PAGE_STYLE}</style></head><body>"
        f"<h1>{escape(title)}</h1>"
        f'<div class="meta">mode: {escape(level_result.mode)} | code version: {escape(level_result.code_version)} | '
        f"finished: {escape(level_result.finished_at)} | total time: {level_result.total_seconds:.1f}s</div>"
        f'<div class="banner {status_word}">Overall: {status_word}</div>'
        "<table><tr><th>Check</th><th>Kind</th><th>Status</th><th>Key number</th><th>What it proves</th></tr>"
        f"{rows}</table>"
        f'<div class="story"><strong>What you just saw</strong><p>{escape(level_result.what_you_saw)}</p></div>'
        f"{make_charts_html(level_result)}"
        "</body></html>"
    )


# ---------------------------------------------------------------------------
# JSON and files
# ---------------------------------------------------------------------------

def make_result_json_data(level_result: LevelResult) -> dict[str, Any]:
    """
    Two clearly separated parts. 'stable' is identical on every run of the same code; 'run_info' holds the only
    things allowed to differ (dates, timings, code version).
    """
    stable = {
        "level_number": level_result.level_number,
        "title": level_result.title,
        "mode": level_result.mode,
        "overall_status": level_result.overall_status,
        "checks": [
            {"name": r.name, "kind": r.kind, "status": r.status, "key_number": r.key_number}
            for r in level_result.check_results
        ],
    }
    # A part run (--part a) says which part it was; whole-level runs keep exactly the keys they always had.
    if level_result.part is not None:
        stable["part"] = level_result.part
    run_info = {
        "finished_at": level_result.finished_at,
        "total_seconds": level_result.total_seconds,
        "code_version": level_result.code_version,
        "check_seconds": {r.name: r.seconds for r in level_result.check_results},
    }
    return {"stable": stable, "run_info": run_info}


def write_evidence_files(level_result: LevelResult, evidence_dir: Path) -> None:
    """Write one JSON file per check into the evidence folder."""
    evidence_dir.mkdir(parents=True, exist_ok=True)
    for index, result in enumerate(level_result.check_results, start=1):
        content = {"check": result.name, "status": result.status, "explanation": result.explanation,
                   "evidence": result.evidence}
        (evidence_dir / evidence_file_name(index, result.name)).write_bytes(to_json_bytes(content))


def write_level_report(level_result: LevelResult, out_dir: Path) -> None:
    """Write index.html, result.json and evidence/ for one level into out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "index.html").write_text(make_level_page_html(level_result), encoding="utf-8")
    (out_dir / "result.json").write_bytes(to_json_bytes(make_result_json_data(level_result)))
    write_evidence_files(level_result, out_dir / "evidence")

