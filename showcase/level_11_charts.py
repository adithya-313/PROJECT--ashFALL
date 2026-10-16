"""
WHAT: Standalone SVG chart builders rendering clean visual telemetry for Level 11 showcase reports.
It generates inline vector graphics illustrating daily transactions, bank outage windows, OS adoption curves,
and money burst flows without requiring any third-party rendering libraries or external browser dependencies.

WHY: Self-contained HTML reports must visualize simulation dynamics completely offline without external network calls.
Emitting pure SVG strings guarantees fast, deterministic rendering across modern browsers and automated headless tests.
Isolating graphic builders ensures styling changes remain decoupled from underlying simulation calculation routines.

TERM: SVG: Scalable Vector Graphics, an XML-based format for describing two-dimensional vector graphics,
      e.g. rendering interactive charts with rect, polyline, and text elements directly in web pages.
TERM: moving average: A calculation that averages data points within a sliding window to smooth out short-term fluctuations,
      e.g. calculating a 7-day rolling mean to highlight multi-week transaction volume trends.
TERM: adoption curve: A visual curve tracing the percentage of entities adopting an innovation or software version over time,
      e.g. charting the cumulative fraction of handsets updating to a new operating system release.
TERM: hero chart: The primary high-level summary visual prominently displayed at the top of an analytical report,
      e.g. the full 61-day daily transaction volume chart showing festival peaks in the Level 11 overview.
"""

from html import escape
from typing import Any


def _scale_coord(val: float, in_min: float, in_max: float, out_min: float, out_max: float) -> float:
    """
    WHAT: Linearly interpolates a real numeric value from an input range into an output screen pixel coordinate.
    It clamps identical boundary limits to the lower output bound to prevent division-by-zero runtime exceptions.
    The resulting floating-point value maps domain measurements directly onto SVG rendering canvases.

    WHY: Visual chart rendering requires mapping abstract metrics such as timestamps and transaction counts
    onto absolute pixel coordinates within SVG viewport bounding boxes.
    Centralizing coordinate transformation prevents duplicated scaling arithmetic across different chart generators.

    Args:
        val: The input real numeric measurement value to be linearly mapped onto canvas coordinates.
        in_min: Lower numeric bound of the input domain scale representing the baseline value.
        in_max: Upper numeric bound of the input domain scale representing the peak value.
        out_min: Lower pixel coordinate boundary on the target SVG canvas representing the origin.
        out_max: Upper pixel coordinate boundary on the target SVG canvas representing the extent.

    Returns:
        Mapped floating-point coordinate value positioned precisely within the target pixel range.

    Example:
        >>> y_px = _scale_coord(50.0, 0.0, 100.0, 300.0, 0.0)
        >>> round(y_px, 1)
        150.0
    """
    if in_max == in_min:
        return out_min
    ratio = (val - in_min) / (in_max - in_min)
    return out_min + ratio * (out_max - out_min)


def _compute_moving_average(values: list[int], window_size: int = 7) -> list[float]:
    """
    WHAT: Computes a backward-looking sliding window moving average across an ordered sequence of integer counts.
    For each position, it averages preceding values up to the designated window size to produce smooth curves.
    Early positions prior to filling a complete window average all available historical elements from index zero.

    WHY: Daily transaction counts exhibit sharp high-frequency noise from weekend cycles and spontaneous purchases.
    A 7-day smoothing curve exposes structural underlying trends such as multi-day festival spending surges.
    Providing trailing averages avoids forward-looking bias and accurately reflects observed momentum.

    Args:
        values: Ordered sequence of daily non-negative integer transaction attempt counts.
        window_size: Number of consecutive daily history elements included in the rolling mean.

    Returns:
        List of smoothed floating-point average counts corresponding to each input day index.

    Example:
        >>> raw = [10, 20, 30, 40, 50, 60, 70, 80]
        >>> smoothed = _compute_moving_average(raw, window_size=3)
        >>> [round(x, 1) for x in smoothed[:4]]
        [10.0, 15.0, 20.0, 30.0]
    """
    smoothed: list[float] = []
    for idx in range(len(values)):
        start = max(0, idx - window_size + 1)
        window = values[start : idx + 1]
        smoothed.append(sum(window) / len(window) if window else 0.0)
    return smoothed


def _render_festival_bands(
    windows: list[tuple[int, int, str]], n: int, plot_w: float, plot_h: float, m_left: float, m_top: float
) -> list[str]:
    """
    WHAT: Generates SVG rect and text elements representing highlighted festival periods on the timeline.
    It converts day indices into canvas pixel bounds and draws warm saffron background bands with labels.
    The resulting elements highlight periods of expected transaction volume spikes for visual inspection.

    WHY: Decomposing chart rendering into focused helper routines preserves strict 30-line code bounds.
    Separating festival background annotations ensures visual styling remains independent of bar geometry.
    Rendering background bands first ensures transaction bars and trend lines remain visible on top.

    Args:
        windows: List of tuples specifying festival start day, end day, and text banner label.
        n: Total number of days in the simulation horizon used to scale horizontal bounds.
        plot_w: Available horizontal width of the active chart plotting area in pixels.
        plot_h: Available vertical height of the active chart plotting area in pixels.
        m_left: Left margin pixel offset from the SVG canvas boundary.
        m_top: Top margin pixel offset from the SVG canvas boundary.

    Returns:
        List of SVG XML string snippets rendering shaded festival bands and descriptive text labels.

    Example:
        >>> win = [(3, 5, "Diwali")]
        >>> snippets = _render_festival_bands(win, 10, 800.0, 200.0, 50.0, 30.0)
        >>> len(snippets) == 2 and "Diwali" in snippets[1]
        True
    """
    elements: list[str] = []
    for start_idx, end_idx, label in windows:
        x1 = m_left + (start_idx / n) * plot_w
        x2 = m_left + (min(end_idx + 1, n) / n) * plot_w
        elements.append(
            f'<rect x="{x1:.1f}" y="{m_top}" width="{max(x2 - x1, 2):.1f}" height="{plot_h}" '
            f'fill="#FBEBD6" stroke="#D9822B" stroke-width="0.8" opacity="0.85"/>'
        )
        elements.append(
            f'<text x="{x1 + 4:.1f}" y="{m_top + 16}" fill="#9A5A17" font-size="13" '
            f'font-family="sans-serif" font-weight="600">{escape(label)}</text>'
        )
    return elements


def _render_daily_bars(
    counts: list[int],
    dates: list[str],
    n: int,
    max_c: int,
    plot_w: float,
    plot_h: float,
    m_left: float,
    m_top: float,
) -> list[str]:
    """
    WHAT: Generates SVG rect elements for daily transaction count bars with accessible hover titles.
    It calculates proportional heights and positions for each day, applying semi-transparent dark ink styling.
    Hover title tooltips display the exact date and volume count for interactive desktop inspection.

    WHY: Decomposes hero chart generation into modular helpers meeting the 30-line function standard.
    Isolates discrete bar geometry calculations from continuous moving average polyline generation.
    Semi-transparent opacity ensures background festival bands and foreground polylines remain visible.

    Args:
        counts: List of daily payment attempt counts across the simulation horizon.
        dates: List of formatted calendar date strings corresponding to each simulated day.
        n: Total number of days in the simulation horizon determining bar spacing.
        max_c: Peak transaction count across all days used to scale vertical bar heights.
        plot_w: Usable plotting area width in pixels excluding canvas margins.
        plot_h: Usable plotting area height in pixels excluding canvas margins.
        m_left: Left margin pixel offset from canvas origin.
        m_top: Top margin pixel offset from canvas origin.

    Returns:
        List of SVG XML string snippets rendering individual daily transaction bars.

    Example:
        >>> c = [50, 100]
        >>> d = ["2026-01-01", "2026-01-02"]
        >>> bars = _render_daily_bars(c, d, 2, 100, 800.0, 200.0, 50.0, 30.0)
        >>> len(bars) == 2 and "title" in bars[0]
        True
    """
    elements: list[str] = []
    bar_w = max((plot_w / n) * 0.75, 1.0)
    for idx, count in enumerate(counts):
        bx = m_left + (idx / n) * plot_w + ((plot_w / n) - bar_w) / 2
        by = m_top + plot_h - (count / max_c) * plot_h
        bh = (count / max_c) * plot_h
        date_lbl = dates[idx] if idx < len(dates) else str(idx)
        elements.append(
            f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w:.1f}" height="{bh:.1f}" '
            f'fill="#1A2B48" opacity="0.22"><title>{escape(date_lbl)}: {count}</title></rect>'
        )
    return elements


def _render_moving_avg_polyline(
    avg_counts: list[float], n: int, max_c: int, plot_w: float, plot_h: float, m_left: float, m_top: float
) -> str:
    """
    WHAT: Generates an SVG polyline element tracing smoothed 7-day rolling average transaction counts.
    It maps moving average counts across horizontal day positions and vertical volume levels.
    The resulting continuous dark curve overlays discrete transaction bars to expose macro volume momentum.

    WHY: Modular helper ensuring daily_bars_with_average adheres to the 30-line code length ceiling.
    Isolates trendline coordinate calculations from background grid lines and discrete bar geometry.
    Smooth polyline formatting highlights long-term trends while filtering out daily transactional noise.

    Args:
        avg_counts: Sequence of smoothed floating-point transaction count averages.
        n: Total number of days in the simulation horizon determining horizontal spacing.
        max_c: Peak daily transaction count used to scale vertical coordinates.
        plot_w: Available plot width in pixels excluding horizontal margins.
        plot_h: Available plot height in pixels excluding vertical margins.
        m_left: Left margin pixel offset from canvas origin.
        m_top: Top margin pixel offset from canvas origin.

    Returns:
        Complete SVG polyline element string configured with mapped coordinates and styling.

    Example:
        >>> avgs = [10.0, 20.0, 30.0]
        >>> p = _render_moving_avg_polyline(avgs, 3, 50, 600.0, 200.0, 50.0, 30.0)
        >>> 'points="' in p and '<polyline' in p
        True
    """
    pts = [
        f"{m_left + ((i + 0.5) / n) * plot_w:.1f},{m_top + plot_h - (val / max_c) * plot_h:.1f}"
        for i, val in enumerate(avg_counts)
    ]
    return f'<polyline points="{" ".join(pts)}" fill="none" stroke="#1A2B48" stroke-width="2.4" stroke-linecap="round"/>'


def daily_bars_with_average(
    daily_counts: list[int], dates: list[str], festival_windows: list[tuple[int, int, str]]
) -> str:
    """
    WHAT: Renders an SVG hero chart combining daily transaction volume bars, a 7-day average, and festival bands.
    Each daily count is drawn as a semi-transparent vertical bar, overlaid with a continuous smoothed average line.
    Shaded background rectangles annotate designated festival periods with distinct styling and text banners.

    WHY: Provides the principal overview graphic for the Level 11 showcase report, visualizing full-horizon activity.
    Combining daily discrete volumes with continuous trend lines demonstrates both baseline stability and festival spikes.
    Inline SVG allows immediate rendering in standalone offline HTML documents without external visual dependencies.

    Args:
        daily_counts: List of daily payment attempt counts spanning the entire simulated calendar period.
        dates: List of human-readable calendar date strings corresponding to each simulated day.
        festival_windows: Tuples defining start day index, end day index, and banner label for festivals.

    Returns:
        Complete standalone SVG XML string ready for embedding within report document bodies.

    Example:
        >>> counts = [100, 120, 110, 180, 220, 130, 105]
        >>> dts = [f"2026-01-0{i+1}" for i in range(7)]
        >>> win = [(3, 5, "Diwali")]
        >>> svg = daily_bars_with_average(counts, dts, win)
        >>> '<svg class="chart"' in svg and 'Diwali' in svg
        True
    """
    width, height = 1000, 300
    m_left, m_right, m_top, m_bottom = 50, 20, 30, 40
    plot_w, plot_h = width - m_left - m_right, height - m_top - m_bottom
    n, max_c = max(len(daily_counts), 1), max(max(daily_counts, default=1), 1)
    parts = [
        f'<svg class="chart" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="100%" height="auto" role="img" aria-label="Daily payment attempts over time">'
    ]
    parts.extend(_render_festival_bands(festival_windows, n, plot_w, plot_h, m_left, m_top))
    parts.extend(_render_daily_bars(daily_counts, dates, n, max_c, plot_w, plot_h, m_left, m_top))
    avg_counts = _compute_moving_average(daily_counts, window_size=7)
    parts.append(_render_moving_avg_polyline(avg_counts, n, max_c, plot_w, plot_h, m_left, m_top))
    parts.append(
        f'<line x1="{m_left}" y1="{m_top + plot_h}" x2="{width - m_right}" y2="{m_top + plot_h}" stroke="#D3DCD7"/>'
    )
    parts.append(
        f'<text x="{m_left - 8}" y="{m_top + 10}" fill="#6B778A" font-size="11" text-anchor="end">{max_c}</text>'
    )
    parts.append(
        f'<text x="{m_left - 8}" y="{m_top + plot_h}" fill="#6B778A" font-size="11" text-anchor="end">0</text></svg>'
    )
    return "".join(parts)


def _render_incident_band(
    window: tuple[float, float, str], min_h: float, max_h: float, m_left: float, plot_w: float, m_top: float, plot_h: float
) -> list[str]:
    """
    WHAT: Generates SVG rect and text elements highlighting an active downtime incident window.
    It maps floating-point starting and ending hours to horizontal pixel boundaries across the timeline.
    The resulting shaded rect and text banner clearly distinguish normal operations from incident intervals.

    WHY: Isolates incident window markup generation to maintain the 30-line function length constraint.
    Ensures that shaded incident bands are drawn behind transaction success polylines for visual clarity.
    Reusable coordinate interpolation guarantees precision when mapping arbitrary fractional hour stamps.

    Args:
        window: Tuple specifying start hour, end hour, and descriptive incident label.
        min_h: Earliest chronological hour stamp in the observation sequence.
        max_h: Latest chronological hour stamp in the observation sequence.
        m_left: Left margin pixel offset from canvas origin.
        plot_w: Available plot width in pixels.
        m_top: Top margin pixel offset from canvas origin.
        plot_h: Available plot height in pixels.

    Returns:
        List of SVG XML string snippets rendering the incident band and text label.

    Example:
        >>> win = (2.0, 4.0, "Outage")
        >>> elms = _render_incident_band(win, 0.0, 10.0, 40.0, 500.0, 20.0, 150.0)
        >>> len(elms) == 2 and "Outage" in elms[1]
        True
    """
    w_start, w_end, w_label = window
    x1 = _scale_coord(w_start, min_h, max_h, m_left, m_left + plot_w)
    x2 = _scale_coord(w_end, min_h, max_h, m_left, m_left + plot_w)
    rect = (
        f'<rect x="{x1:.1f}" y="{m_top}" width="{max(x2 - x1, 2):.1f}" height="{plot_h}" '
        f'fill="#FBEBD6" stroke="#D9822B" stroke-width="0.8" opacity="0.8"/>'
    )
    label = (
        f'<text x="{x1 + 4:.1f}" y="{m_top + 14}" fill="#9A5A17" font-size="11" '
        f'font-family="sans-serif" font-weight="600">{escape(w_label)}</text>'
    )
    return [rect, label]


def rate_line_with_window(
    hours: list[float], rates: list[float], window: tuple[float, float, str]
) -> str:
    """
    WHAT: Generates an SVG line graph displaying hourly payment success rates with a highlighted incident window.
    The continuous line charts percentage values over time, while a shaded vertical band highlights active downtime.
    The chart visualizes system degradation, demonstrating that service drops during bank outages without reaching zero.

    WHY: Visual proof for bank downtime check (Check 6) demonstrating partial degradation across the multi-bank system.
    Displaying the exact downtime interval establishes empirical alignment between planned incidents and observed drops.
    Standalone vector graphics provide clear visual evidence within the expandable check details accordion.

    Args:
        hours: Sequence of chronological simulation elapsed hour timestamps.
        rates: Corresponding transaction success rate percentages between 0.0 and 100.0.
        window: Tuple specifying the incident start hour, end hour, and explanatory label.

    Returns:
        Complete standalone SVG XML string illustrating success rate fluctuations and incident bounds.

    Example:
        >>> hrs = [10.0, 11.0, 12.0, 13.0, 14.0]
        >>> rts = [98.5, 99.0, 65.2, 64.8, 98.2]
        >>> win = (11.5, 13.5, "Bank-0 Outage")
        >>> chart = rate_line_with_window(hrs, rts, win)
        >>> 'Bank-0 Outage' in chart and '<polyline' in chart
        True
    """
    width, height = 640, 200
    m_left, m_right, m_top, m_bottom = 45, 15, 20, 30
    plot_w, plot_h = width - m_left - m_right, height - m_top - m_bottom
    min_h, max_h = min(hours, default=0.0), max(hours, default=1.0)
    parts = [
        f'<svg class="chart" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="100%" height="auto" role="img" aria-label="Success rate over time">'
    ]
    parts.extend(_render_incident_band(window, min_h, max_h, m_left, plot_w, m_top, plot_h))
    pts = [
        f"{_scale_coord(h, min_h, max_h, m_left, m_left + plot_w):.1f},"
        f"{_scale_coord(r, 0.0, 100.0, m_top + plot_h, m_top):.1f}"
        for h, r in zip(hours, rates)
    ]
    parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="#1A2B48" stroke-width="2"/>')
    parts.append(
        f'<line x1="{m_left}" y1="{m_top + plot_h}" x2="{width - m_right}" y2="{m_top + plot_h}" stroke="#D3DCD7"/>'
    )
    parts.append(
        f'<text x="{m_left - 6}" y="{m_top + 10}" fill="#6B778A" font-size="10" text-anchor="end">100%</text>'
    )
    parts.append(
        f'<text x="{m_left - 6}" y="{m_top + plot_h}" fill="#6B778A" font-size="10" text-anchor="end">0%</text></svg>'
    )
    return "".join(parts)


def _render_curve_polyline(
    days: list[int],
    shares: list[float],
    stroke: str,
    stroke_w: float,
    dash: str,
    min_d: int,
    max_d: int,
    m_left: float,
    plot_w: float,
    m_top: float,
    plot_h: float,
) -> str:
    """
    WHAT: Generates an SVG polyline element tracing adoption percentage coordinates across calendar days.
    It maps each integer day and floating-point adoption fraction to exact canvas pixel locations.
    Configurable stroke color, width, and dasharray enable rendering primary or comparative baseline curves.

    WHY: Modular helper ensuring the adoption_curve function stays within the 30-line code ceiling.
    Encapsulates coordinate transformation and attribute assembly for reusable polyline rendering.
    Enables rendering both solid observed adoption curves and dashed baseline comparisons.

    Args:
        days: Chronological list of integer day numbers.
        shares: Corresponding adoption shares bounded between 0.0 and 1.0.
        stroke: CSS color string specifying the line stroke color.
        stroke_w: Floating-point pixel width of the polyline stroke.
        dash: SVG stroke-dasharray attribute string or empty string for solid lines.
        min_d: Earliest day index on the horizontal timeline.
        max_d: Latest day index on the horizontal timeline.
        m_left: Left margin pixel offset from canvas origin.
        plot_w: Total horizontal plot width in pixels.
        m_top: Top margin pixel offset from canvas origin.
        plot_h: Total vertical plot height in pixels.

    Returns:
        Complete SVG polyline element string configured with mapped coordinates and styling.

    Example:
        >>> ds = [0, 1]
        >>> shs = [0.1, 0.9]
        >>> p = _render_curve_polyline(ds, shs, "#000", 2.0, "", 0, 1, 40.0, 500.0, 20.0, 150.0)
        >>> 'points="' in p and 'stroke="#000"' in p
        True
    """
    pts = [
        f"{_scale_coord(float(d), float(min_d), float(max_d), m_left, m_left + plot_w):.1f},"
        f"{_scale_coord(s, 0.0, 1.0, m_top + plot_h, m_top):.1f}"
        for d, s in zip(days, shares)
    ]
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<polyline points="{" ".join(pts)}" fill="none" stroke="{stroke}" stroke-width="{stroke_w}"{dash_attr}/>'


def _render_adoption_legend(right_x: float, top_y: float) -> str:
    """
    WHAT: Draws a two-line legend naming the solid curve (with stress layer) and the dashed curve (base run).
    It anchors both labels to the right edge of the plot area, just below its top margin.

    WHY: Check 7 overlays the base and layered adoption curves; without a legend the reader cannot tell them apart.

    Args:
        right_x: X coordinate of the plot's right edge.
        top_y: Y coordinate of the plot's top edge.

    Returns:
        SVG text fragment with two text elements.

    Example:
        >>> legend = _render_adoption_legend(625, 20)
        >>> "with stress layer" in legend and "base run" in legend
        True
    """
    return (
        f'<text x="{right_x}" y="{top_y + 12}" fill="#1A2B48" font-size="11" text-anchor="end">'
        "solid: with stress layer</text>"
        f'<text x="{right_x}" y="{top_y + 26}" fill="#6B778A" font-size="11" text-anchor="end">'
        "dashed: base run</text>"
    )


def adoption_curve(
    days: list[int], adoption_shares: list[float], baseline_shares: list[float] | None = None
) -> str:
    """
    WHAT: Draws an SVG adoption curve charting the cumulative market share percentage of updated operating systems.
    It plots the daily share of phones on the newest OS generation over simulated calendar days.
    When baseline shares are provided, it overlays a secondary dashed reference curve for visual comparison.

    WHY: Visual proof for OS update wave stress check (Check 7) illustrating logistic adoption kinetics.
    Validates that mobile devices upgrade smoothly across citizen cohorts rather than jumping instantaneously.
    Standalone vector graphics document hardware environment realism within analytical audit reports.

    Args:
        days: Chronological list of integer day numbers spanning the observation window.
        adoption_shares: Fraction (0.0 to 1.0) of active devices operating the upgraded operating system.
        baseline_shares: Optional baseline adoption shares for un-stressed comparison benchmarking.

    Returns:
        Complete standalone SVG XML string visualizing the empirical software rollout trajectory.

    Example:
        >>> ds = list(range(10))
        >>> shs = [0.01 * (i ** 2) for i in ds]
        >>> chart = adoption_curve(ds, shs)
        >>> '<polyline' in chart and 'viewBox' in chart
        True
    """
    width, height = 640, 200
    m_left, m_right, m_top, m_bottom = 45, 15, 20, 30
    plot_w, plot_h = width - m_left - m_right, height - m_top - m_bottom
    min_d, max_d = min(days, default=0), max(days, default=1)
    parts = [
        f'<svg class="chart" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="100%" height="auto" role="img" aria-label="OS update adoption curve">'
    ]
    if baseline_shares and len(baseline_shares) == len(days):
        base_line = _render_curve_polyline(
            days, baseline_shares, "#6B778A", 1.6, "4,4", min_d, max_d, m_left, plot_w, m_top, plot_h
        )
        parts.extend([base_line, _render_adoption_legend(width - m_right, m_top)])
    main_line = _render_curve_polyline(
        days, adoption_shares, "#1A2B48", 2.2, "", min_d, max_d, m_left, plot_w, m_top, plot_h
    )
    parts.append(main_line)
    parts.append(
        f'<line x1="{m_left}" y1="{m_top + plot_h}" x2="{width - m_right}" y2="{m_top + plot_h}" stroke="#D3DCD7"/>'
    )
    parts.append(
        f'<text x="{m_left - 6}" y="{m_top + 10}" fill="#6B778A" font-size="10" text-anchor="end">1.0</text>'
    )
    parts.append(
        f'<text x="{m_left - 6}" y="{m_top + plot_h}" fill="#6B778A" font-size="10" text-anchor="end">0.0</text></svg>'
    )
    return "".join(parts)


def _render_inflow_window(
    window: tuple[int, int, str], n: int, plot_w: float, plot_h: float, m_left: float, m_top: float
) -> list[str]:
    """
    WHAT: Generates SVG rect and text elements marking the active episode window for account inflows.
    It translates integer day limits into screen coordinates and displays the episode name banner.
    The resulting background rectangle visually identifies the specific duration of the burst event.

    WHY: Factoring out episode band generation ensures account_flow_with_window remains under 30 code lines.
    Separates background event delineation from discrete financial transaction bar calculations.
    Standardized styling maintains visual harmony with festival and downtime incident markers.

    Args:
        window: Tuple specifying start day index, end day index, and banner label for the burst episode.
        n: Total number of days in the observation window used to scale horizontal bounds.
        plot_w: Available horizontal plotting width in pixels.
        plot_h: Available vertical plotting height in pixels.
        m_left: Left margin pixel offset from canvas origin.
        m_top: Top margin pixel offset from canvas origin.

    Returns:
        List of SVG XML string snippets rendering the shaded burst window and banner text.

    Example:
        >>> win = (2, 4, "Wedding")
        >>> snippets = _render_inflow_window(win, 7, 500.0, 150.0, 50.0, 20.0)
        >>> len(snippets) == 2 and "Wedding" in snippets[1]
        True
    """
    w_start, w_end, w_label = window
    x1 = m_left + (w_start / n) * plot_w
    x2 = m_left + (min(w_end + 1, n) / n) * plot_w
    rect = (
        f'<rect x="{x1:.1f}" y="{m_top}" width="{max(x2 - x1, 2):.1f}" height="{plot_h}" '
        f'fill="#FBEBD6" stroke="#D9822B" stroke-width="0.8" opacity="0.8"/>'
    )
    label = (
        f'<text x="{x1 + 4:.1f}" y="{m_top + 14}" fill="#9A5A17" font-size="11" '
        f'font-family="sans-serif" font-weight="600">{escape(w_label)}</text>'
    )
    return [rect, label]


def _render_inflow_bars(
    inflows: list[float], n: int, max_flow: float, plot_w: float, plot_h: float, m_left: float, m_top: float
) -> list[str]:
    """
    WHAT: Generates SVG rect elements for daily account inflow bars with proportional heights.
    It calculates horizontal bar positions and vertical elevations relative to peak observed volume.
    The resulting semi-transparent bars visualize aggregate incoming funds per simulated calendar day.

    WHY: Modular helper ensuring account_flow_with_window adheres to the 30-line code length limit.
    Isolates discrete bar geometry calculations from background episode band and axis line generation.
    Semi-transparent opacity ensures underlying shaded burst windows remain clearly visible.

    Args:
        inflows: Sequence of daily incoming currency amounts in rupees.
        n: Total number of days in the simulation calendar determining horizontal bar spacing.
        max_flow: Peak daily incoming currency amount used to scale vertical bar elevations.
        plot_w: Usable plotting area width in pixels excluding canvas margins.
        plot_h: Usable plotting area height in pixels excluding canvas margins.
        m_left: Left margin pixel offset from canvas origin.
        m_top: Top margin pixel offset from canvas origin.

    Returns:
        List of SVG XML string snippets rendering individual daily account inflow bars.

    Example:
        >>> flows = [100.0, 5000.0]
        >>> bars = _render_inflow_bars(flows, 2, 5000.0, 500.0, 150.0, 50.0, 20.0)
        >>> len(bars) == 2 and '<rect' in bars[0]
        True
    """
    elements: list[str] = []
    bar_w = max((plot_w / n) * 0.7, 1.0)
    for idx, flow in enumerate(inflows):
        bx = m_left + (idx / n) * plot_w + ((plot_w / n) - bar_w) / 2
        bh = (flow / max_flow) * plot_h
        by = m_top + plot_h - bh
        elements.append(
            f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w:.1f}" height="{bh:.1f}" '
            f'fill="#1A2B48" opacity="0.45"/>'
        )
    return elements


def account_flow_with_window(
    days: list[int], inflows: list[float], window: tuple[int, int, str]
) -> str:
    """
    WHAT: Renders an SVG bar chart illustrating daily incoming funds into an account with an episode window highlight.
    Daily inflow amounts are drawn as vertical bars, overlaid with a shaded background band marking the active burst event.
    The visualization highlights sudden spikes in account velocity characteristic of weddings or chit fund collections.

    WHY: Visual proof for wedding collection check (Check 9) demonstrating distinct non-fraud transaction bursts.
    Proves that legitimate social payment episodes create dramatic inflow surges without triggering system failures.
    Standalone vector formatting allows seamless integration into audit reports and compliance documentation.

    Args:
        days: Sequence of simulated calendar day numbers.
        inflows: Aggregate daily incoming transaction amount in rupees received by the host account.
        window: Tuple specifying episode start day, end day, and descriptive name label.

    Returns:
        Complete standalone SVG XML string representing account inflows across the simulation calendar.

    Example:
        >>> ds = [1, 2, 3, 4, 5, 6, 7]
        >>> flows = [500.0, 600.0, 450.0, 25000.0, 32000.0, 700.0, 550.0]
        >>> win = (4, 5, "Wedding Gift Collection")
        >>> chart = account_flow_with_window(ds, flows, win)
        >>> 'Wedding Gift Collection' in chart and '<rect' in chart
        True
    """
    width, height = 640, 200
    m_left, m_right, m_top, m_bottom = 55, 15, 20, 30
    plot_w, plot_h = width - m_left - m_right, height - m_top - m_bottom
    n, max_flow = max(len(days), 1), max(max(inflows, default=1.0), 1.0)
    parts = [
        f'<svg class="chart" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="100%" height="auto" role="img" aria-label="Daily account incoming funds">'
    ]
    parts.extend(_render_inflow_window(window, n, plot_w, plot_h, m_left, m_top))
    parts.extend(_render_inflow_bars(inflows, n, max_flow, plot_w, plot_h, m_left, m_top))
    parts.append(
        f'<line x1="{m_left}" y1="{m_top + plot_h}" x2="{width - m_right}" y2="{m_top + plot_h}" stroke="#D3DCD7"/>'
    )
    parts.append(
        f'<text x="{m_left - 6}" y="{m_top + 10}" fill="#6B778A" font-size="10" text-anchor="end">₹{int(max_flow):,}</text>'
    )
    parts.append(
        f'<text x="{m_left - 6}" y="{m_top + plot_h}" fill="#6B778A" font-size="10" text-anchor="end">₹0</text></svg>'
    )
    return "".join(parts)


def _render_pipeline_node(
    title: str, subtitle: str, bx: float, y_pos: float, box_w: float, box_h: float
) -> list[str]:
    """
    WHAT: Renders a rounded rectangular stage node with header and multi-line subtitle descriptions.
    It builds SVG rect and text elements formatted according to design system styling tokens.
    The resulting component represents a single discrete functional stage in the simulation architecture.

    WHY: Decomposing pipeline diagram construction ensures pipeline_diagram satisfies the 30-line rule.
    Standardizes node boundary geometry, corner radiuses, and typographic styling across all diagram stages.
    Enables clean multi-line text wrapping within each architectural process node.

    Args:
        title: Primary bold header label identifying the architectural stage.
        subtitle: Multi-line description text detailing component responsibilities.
        bx: Left horizontal pixel position for placing the stage node.
        y_pos: Top vertical pixel position for placing the stage node.
        box_w: Fixed pixel width of the stage node rectangle.
        box_h: Fixed pixel height of the stage node rectangle.

    Returns:
        List of SVG XML string snippets rendering the node container and formatted text labels.

    Example:
        >>> node = _render_pipeline_node("Stage", "Sub1\\nSub2", 10.0, 20.0, 100.0, 60.0)
        >>> len(node) >= 3 and "Stage" in node[1]
        True
    """
    elements = [
        f'<rect x="{bx:.1f}" y="{y_pos}" width="{box_w}" height="{box_h}" rx="6" '
        f'fill="#FBFCFA" stroke="#AFBEB6" stroke-width="1.4"/>',
        f'<text x="{bx + box_w / 2:.1f}" y="{y_pos + 28}" fill="#1A2B48" font-size="13" '
        f'font-family="sans-serif" font-weight="600" text-anchor="middle">{escape(title)}</text>',
    ]
    for s_idx, line in enumerate(subtitle.split("\n")):
        elements.append(
            f'<text x="{bx + box_w / 2:.1f}" y="{y_pos + 48 + s_idx * 16}" fill="#6B778A" '
            f'font-size="11" font-family="sans-serif" text-anchor="middle">{escape(line)}</text>'
        )
    return elements


def pipeline_diagram() -> str:
    """
    WHAT: Constructs an SVG block diagram displaying the simulation pipeline stages and data flow arrows.
    It renders labeled rectangular process nodes representing configurations, entity builder, calendar planning,
    transaction dispatch runner, and output artifact serializations connected by sequential arrows.

    WHY: Visual representation of Section 2 ('How a run works') in the Level 11 showcase report.
    Clarifies architectural boundaries between static configuration, entity graphs, and dynamic execution runners.
    Standalone vector formatting guarantees pristine rendering at any display resolution without external images.

    Returns:
        Complete standalone SVG XML string depicting the architectural flow of simulation components.

    Example:
        >>> diag = pipeline_diagram()
        >>> '<svg class="chart"' in diag and 'Calendar Runner' in diag
        True
    """
    width, height = 960, 160
    stages = [
        ("Config & Layers", "world_ranges.yaml\nstress layers"),
        ("Build World", "300 people, 40 shops\n3 bank ledgers"),
        ("Plan & Schedule", "festivals, outages\ndevice updates"),
        ("Calendar Runner", "minute-by-minute\npayment dispatch"),
        ("Run Artifacts", "events.jsonl\nmanifest & card"),
    ]
    parts = [
        f'<svg class="chart" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="100%" height="auto" role="img" aria-label="Simulation pipeline architecture">'
    ]
    box_w, box_h, y_pos, gap = 160, 80, 40, 35
    total_w = len(stages) * box_w + (len(stages) - 1) * gap
    start_x = (width - total_w) / 2
    for idx, (title, subtitle) in enumerate(stages):
        bx = start_x + idx * (box_w + gap)
        parts.extend(_render_pipeline_node(title, subtitle, bx, y_pos, box_w, box_h))
        if idx < len(stages) - 1:
            ax = bx + box_w + 6
            parts.append(
                f'<path d="M {ax:.1f} {y_pos + box_h / 2} L {ax + gap - 12:.1f} {y_pos + box_h / 2} '
                f'M {ax + gap - 18:.1f} {y_pos + box_h / 2 - 4} L {ax + gap - 12:.1f} {y_pos + box_h / 2} '
                f'L {ax + gap - 18:.1f} {y_pos + box_h / 2 + 4}" fill="none" stroke="#D9822B" stroke-width="2"/>'
            )
    parts.append("</svg>")
    return "".join(parts)
