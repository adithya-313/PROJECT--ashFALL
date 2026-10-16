"""
WHAT: Compiles, inspects, and tabulates grounding provenance and empirical sources for simulation parameters.
It audits parameter settings across background realism, failure realism, and device dynamics configs,
extracting author labels, training ranges, and published citations or assumption rationales.

WHY: Enforces rigorous empirical traceability across all domain configuration inputs.
Providing a unified grounding table ensures developers and auditors can immediately identify which
parameters rely on measured historical benchmarks versus exploratory modeling assumptions.

TERM: grounding: Connecting simulation parameters to real-world measurements, studies, or assumption records,
      e.g. citing an NPCI press release for monthly technical decline percentages.
TERM: provenance: The documented origin, measurement date, and literature citation verifying a parameter value,
      e.g. recording the publication URL and release date for device replacement cycles.
TERM: assumption: An unverified parameter estimate chosen for simulation mechanics without published data,
      e.g. estimating daily active user payment hour distributions without public banking telemetry.
"""

from dataclasses import dataclass
from typing import Any

from world.world_configs import WorldConfigs, read_world_configs

GROUNDING_CONFIGS: list[str] = [
    "background_realism",
    "failure_realism",
    "device_realism",
]


@dataclass(frozen=True)
class GroundingRow:
    """
    WHAT: Immutable record holding empirical grounding details for a single simulation parameter.
    It encapsulates the domain config partition, parameter name, epistemic label, training range,
    and source citation or assumption rationale.

    WHY: Standardizes parameter metadata into a typed, structured container.
    Supporting both attribute access and sequence iteration allows flexible consumption by table renderers.

    Args:
        config_name: Name of the domain configuration container holding the parameter.
        setting: Parameter setting identifier name within the configuration.
        label: Epistemic status label ('sourced', 'reported', 'assumption', or 'policy').
        training_range: Two-element numeric list representing default training bounds.
        source: Citation string containing URL and date when present, or assumption reason text.

    Returns:
        GroundingRow dataclass instance holding parsed grounding properties.

    Example:
        >>> src = "https://example.com (2024)"
        >>> row = GroundingRow("failure_realism", "tech_decline", "reported", [0.007, 0.008], src)
        >>> (row.config_name, row.label, row.training_range)
        ('failure_realism', 'reported', [0.007, 0.008])
    """

    config_name: str
    setting: str
    label: str
    training_range: list[Any]
    source: str


def _format_source_entry(block: dict[str, Any]) -> str:
    """
    WHAT: Formats the source citation string from a parameter configuration block.
    If a source mapping with URL and date is present, it constructs a citation string;
    otherwise, it formats the fallback assumption rationale.

    WHY: Isolates string formatting rules for parameter provenance into a compact helper.
    Ensures consistent citation formatting across all tabulated parameters.

    Args:
        block: Parameter definition dictionary loaded from a configuration file.

    Returns:
        Formatted citation string with URL and date, or assumption reason description.

    Example:
        >>> src_block = {"source": {"url": "https://example.com", "date": "2024-01-01"}}
        >>> _format_source_entry(src_block)
        'https://example.com (2024-01-01)'
        >>> asm_block = {"reason": "No data found."}
        >>> _format_source_entry(asm_block)
        'assumption: No data found.'
    """
    # 1. Check for explicit source dictionary containing citation URL and date.
    source = block.get("source")
    if isinstance(source, dict) and source.get("url"):
        url = str(source["url"])
        date = str(source.get("date", "")).strip()
        return f"{url} ({date})" if date else url

    # 2. Hand back assumption rationale when external source citation is absent.
    reason = str(block.get("reason", "")).strip()
    return f"assumption: {reason}"


def grounding_rows(configs: WorldConfigs) -> list[GroundingRow]:
    """
    WHAT: Extracts grounding records for every parameter in background, failure, and device configs.
    It iterates sequentially across all three domain configs and compiles a GroundingRow for each setting,
    recording config name, parameter name, label, training range, and source citation.

    WHY: Assembles an exhaustive tabular dataset of simulation parameter provenance.
    Enables programmatic auditing of assumption counts and empirical coverage across the codebase.

    Args:
        configs: WorldConfigs container holding loaded domain configuration dictionaries.

    Returns:
        List of GroundingRow records representing all parameters in the three configs.

    Example:
        >>> from world.world_configs import read_world_configs
        >>> cfgs = read_world_configs()
        >>> rows = grounding_rows(cfgs)
        >>> len(rows)
        33
        >>> rows[0].config_name
        'background_realism'
    """
    # 1. Initialize record collection for grounded parameter entries.
    rows: list[GroundingRow] = []

    # 2. Iterate across background_realism, failure_realism, and device_realism configs.
    for config_name in GROUNDING_CONFIGS:
        cfg_dict = getattr(configs, config_name)
        params_map = cfg_dict.get("parameters", {})

        # 3. Extract parameter attributes and append typed GroundingRow instances.
        for setting, block in params_map.items():
            label = str(block.get("label", "assumption"))
            training = list(block.get("training", []))
            source_text = _format_source_entry(block)
            rows.append(
                GroundingRow(
                    config_name=config_name,
                    setting=setting,
                    label=label,
                    training_range=training,
                    source=source_text,
                )
            )

    return rows


def count_by_label(rows: list[GroundingRow]) -> dict[str, int]:
    """
    WHAT: Tallies the count of parameter grounding rows grouped by epistemic status label.
    It counts occurrences of 'sourced', 'reported', 'assumption', and any other assigned labels.

    WHY: Provides high-level metrics summarizing the proportion of empirically grounded versus assumed settings.
    Enables quick verification that assumption counts match golden repository expectations.

    Args:
        rows: List of GroundingRow instances to be categorized and counted.

    Returns:
        Dictionary mapping label string identifiers to their integer occurrence counts.

    Example:
        >>> r1 = GroundingRow("cfg", "s1", "assumption", [1, 2], "reason 1")
        >>> r2 = GroundingRow("cfg", "s2", "reported", [3, 4], "http://src")
        >>> count_by_label([r1, r2])
        {'assumption': 1, 'reported': 1}
    """
    # 1. Initialize label frequency tally dictionary.
    counts: dict[str, int] = {}

    # 2. Count occurrences of each epistemic label across all supplied rows.
    for row in rows:
        counts[row.label] = counts.get(row.label, 0) + 1

    # 3. Hand back computed label distribution dictionary.
    return counts


def describe_grounding(rows: list[GroundingRow]) -> str:
    """
    WHAT: Formats grounding rows into an aligned human-readable text table ending with label counts.
    It renders column headers for Config, Setting, Label, Training, and Source, followed by individual
    parameter rows and a trailing summary section displaying counts by label.

    WHY: Produces clear operational documentation for terminal inspection and audit reports.
    Aligning columns and concluding with summary counts enables rapid review of parameter origins.

    Args:
        rows: List of GroundingRow records to be formatted into tabular representation.

    Returns:
        Formatted multi-line plain text table string ending with summary counts.

    Example:
        >>> r1 = GroundingRow("cfg", "s1", "assumption", [1, 2], "reason text")
        >>> text = describe_grounding([r1])
        >>> "Config" in text and "assumption: 1" in text
        True
    """
    # 1. Render table header line and separation bar.
    lines = [
        f"{'Config':<20} | {'Setting':<36} | {'Label':<12} | {'Training':<14} | Source",
        f"{'-'*20}-+-{'-'*36}-+-{'-'*12}-+-{'-'*14}-+-{'-'*40}",
    ]

    # 2. Render each parameter row with consistent column formatting.
    for r in rows:
        range_str = str(r.training_range)
        lines.append(f"{r.config_name:<20} | {r.setting:<36} | {r.label:<12} | {range_str:<14} | {r.source}")

    # 3. Compute and append label breakdown counts and total parameter tally.
    counts = count_by_label(rows)
    total_settings = len(rows)
    lines.extend([
        "",
        "Grounding counts by label:",
    ])
    for label, count in sorted(counts.items()):
        lines.append(f"  {label}: {count}")
    lines.append(f"  Total settings: {total_settings}")

    return "\n".join(lines)


def main() -> int:
    """
    WHAT: Command-line entry point reading baseline configurations and printing the grounding table.
    It loads world configs, generates grounding rows, formats the audit table, and prints to standard output.

    WHY: Allows operators and CI scripts to inspect parameter grounding directly via 'python -m world.grounding'.

    Returns:
        Integer exit status code 0 upon successful execution.

    Example:
        >>> import contextlib, io
        >>> buf = io.StringIO()
        >>> with contextlib.redirect_stdout(buf):
        ...     code = main()
        >>> code
        0
        >>> "Total settings: 33" in buf.getvalue()
        True
    """
    # 1. Load domain configurations and compile grounding records.
    configs = read_world_configs()
    rows = grounding_rows(configs)

    # 2. Format and display complete plain text grounding summary.
    print(describe_grounding(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
