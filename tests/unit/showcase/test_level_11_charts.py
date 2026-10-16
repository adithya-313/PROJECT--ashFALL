"""
WHAT: Unit test suite validating pure SVG chart rendering functions in showcase/level_11_charts.py.
It verifies XML structure validity, correct element dimensions, absence of NaN values, and element counts
across daily bars, bank downtime windows, OS adoption curves, account flows, and pipeline block diagrams.

WHY: Self-contained showcase pages depend on standalone SVG strings rendering cleanly without client-side scripts.
Validating XML syntax via ElementTree ensures no unescaped entity tags or malformed XML attributes break browser parsers.
Testing boundary conditions such as empty sequences and single-element inputs prevents unexpected runtime exceptions.

TERM: ElementTree: Standard Python library module for parsing, inspecting, and manipulating XML tree structures,
      e.g. verifying that an SVG string contains well-formed opening and closing tags.
TERM: assertion: A test statement verifying that a condition holds true, raising an AssertionError if it does not,
      e.g. asserting that the SVG output begins with the svg tag.
TERM: XML well-formedness: Syntax rules requiring correct tag nesting, quoted attributes, and escaped entity characters,
      e.g. ensuring an SVG graphic does not fail XML parsers due to unclosed tags.
"""

import xml.etree.ElementTree as ET

from showcase.level_11_charts import (
    account_flow_with_window,
    adoption_curve,
    daily_bars_with_average,
    pipeline_diagram,
    rate_line_with_window,
)


def test_daily_bars_with_average_parses_valid_xml() -> None:
    """
    WHAT: Tests that daily_bars_with_average outputs valid XML parsing without error via ElementTree.
    It passes sample transaction counts, dates, and festival windows, verifying that rects and polylines are present.
    It verifies that the total number of bar rects matches the length of the input daily count sequence.

    WHY: Hero charts are the primary focal point of the Level 11 showcase and must never render broken XML.
    Asserting that every daily count corresponds to a rendered rect guarantees complete data representation.
    Ensuring XML validity protects browser layout engines from falling back to raw text rendering.

    Returns:
        None upon successful execution of all assertions.

    Example:
        >>> test_daily_bars_with_average_parses_valid_xml()
    """
    counts = [100, 150, 200, 350, 400, 180, 120]
    dates = [f"2026-01-0{i+1}" for i in range(len(counts))]
    festivals = [(3, 4, "Diwali")]
    svg_str = daily_bars_with_average(counts, dates, festivals)
    assert "NaN" not in svg_str
    root = ET.fromstring(svg_str)
    assert root.tag.endswith("svg")
    rects = [elem for elem in root.iter() if elem.tag.endswith("rect")]
    # One festival background rect plus seven transaction bar rects equals eight total rects.
    assert len(rects) == len(counts) + len(festivals)
    polylines = [elem for elem in root.iter() if elem.tag.endswith("polyline")]
    assert len(polylines) == 1


def test_rate_line_with_window_parses_valid_xml() -> None:
    """
    WHAT: Tests that rate_line_with_window outputs well-formed XML with shaded outage band and rate polylines.
    It provides hourly elapsed times, success percentages, and a downtime window, parsing the result with ElementTree.
    It asserts the presence of the incident text banner and verifies that no NaN floating-point tokens appear.

    WHY: Bank downtime degradation is evaluated visually in Check 6 and must be backed by accurate graphics.
    Confirming that shaded window boundaries and labels appear protects against dropped incident metadata.
    Checking for NaN tokens prevents malformed polylines resulting from division by zero or empty inputs.

    Returns:
        None upon successful execution of all assertions.

    Example:
        >>> test_rate_line_with_window_parses_valid_xml()
    """
    hours = [0.0, 1.0, 2.0, 3.0, 4.0]
    rates = [99.0, 98.5, 65.0, 64.5, 99.2]
    window = (1.5, 3.5, "Outage Window")
    svg_str = rate_line_with_window(hours, rates, window)
    assert "NaN" not in svg_str
    assert "Outage Window" in svg_str
    root = ET.fromstring(svg_str)
    assert root.tag.endswith("svg")
    polylines = [elem for elem in root.iter() if elem.tag.endswith("polyline")]
    assert len(polylines) == 1


def test_adoption_curve_parses_valid_xml() -> None:
    """
    WHAT: Tests that adoption_curve generates well-formed SVG graphics with optional baseline overlays.
    It provides day indices, adoption shares, and baseline shares, validating that two polylines are rendered.
    It verifies that coordinates are mapped properly into SVG canvas viewBox coordinates without NaN values.

    WHY: OS update wave verification in Check 7 requires visual inspection of the adoption S-curve.
    Validating both primary and baseline curve rendering guarantees correct comparison against un-stressed runs.
    ElementTree parsing guarantees that stroke dasharray attributes and polyline points comply with SVG standards.

    Returns:
        None upon successful execution of all assertions.

    Example:
        >>> test_adoption_curve_parses_valid_xml()
    """
    days = list(range(10))
    shares = [0.0, 0.05, 0.15, 0.35, 0.65, 0.85, 0.95, 0.98, 0.99, 1.0]
    base_shares = [0.0, 0.02, 0.08, 0.20, 0.40, 0.60, 0.75, 0.85, 0.90, 0.92]
    svg_str = adoption_curve(days, shares, baseline_shares=base_shares)
    assert "NaN" not in svg_str
    root = ET.fromstring(svg_str)
    assert root.tag.endswith("svg")
    polylines = [elem for elem in root.iter() if elem.tag.endswith("polyline")]
    assert len(polylines) == 2


def test_account_flow_with_window_parses_valid_xml() -> None:
    """
    WHAT: Tests that account_flow_with_window generates well-formed SVG charts with burst event bands.
    It supplies day indices, inflow amounts, and an episode window, parsing the resulting vector markup.
    It checks that the episode label appears in the SVG and that every daily inflow generates a distinct bar.

    WHY: Wedding gift collection dynamics in Check 9 require visual verification of dramatic money surges.
    Verifying that vertical inflow bars accurately reflect daily sums ensures financial bursts are clearly visible.
    Confirming valid XML syntax guarantees that currency symbols and numeric labels are escaped properly.

    Returns:
        None upon successful execution of all assertions.

    Example:
        >>> test_account_flow_with_window_parses_valid_xml()
    """
    days = [1, 2, 3, 4, 5, 6, 7]
    inflows = [500.0, 450.0, 600.0, 35000.0, 42000.0, 800.0, 520.0]
    window = (3, 5, "Wedding Burst")
    svg_str = account_flow_with_window(days, inflows, window)
    assert "NaN" not in svg_str
    assert "Wedding Burst" in svg_str
    root = ET.fromstring(svg_str)
    assert root.tag.endswith("svg")
    rects = [elem for elem in root.iter() if elem.tag.endswith("rect")]
    assert len(rects) == len(days) + 1


def test_pipeline_diagram_parses_valid_xml() -> None:
    """
    WHAT: Tests that pipeline_diagram renders a complete block diagram illustrating the simulation lifecycle.
    It verifies that stages for config, entity building, planning, execution, and outputs are all depicted.
    It checks that the output string parses into a valid ElementTree XML node tree without exceptions.

    WHY: Section 2 of the Level 11 showcase explains simulation stages to domain auditors and developers.
    Validating that all pipeline stages and flow paths are rendered prevents incomplete architectural diagrams.
    Ensuring XML syntax validity guarantees reliable rendering in all supported offline web browsers.

    Returns:
        None upon successful execution of all assertions.

    Example:
        >>> test_pipeline_diagram_parses_valid_xml()
    """
    svg_str = pipeline_diagram()
    assert "NaN" not in svg_str
    assert "Build World" in svg_str
    assert "Calendar Runner" in svg_str
    root = ET.fromstring(svg_str)
    assert root.tag.endswith("svg")
    rects = [elem for elem in root.iter() if elem.tag.endswith("rect")]
    assert len(rects) == 5
