"""Evaluation of national registry coverage across hierarchical mule account depths.

What it is for:
    Measures how thoroughly the Suspect Registry covers illicit accounts across different
    routing depths (layer 1 direct receiver mules vs deeper intermediate hops).
    Compares observed coverage against reference intervals in config/world_ranges.yaml.

Inputs:
    - roles: Sequence of simulation RoleSpan objects or role dictionaries from truth.
    - registry_reports: Sequence of RegistryReport objects or report dictionaries.
    - world_ranges: Optional loaded world ranges dictionary; defaults to config/world_ranges.yaml.
    - split: Dataset partition name ('training', 'test', or 'stress').

What it returns:
    A RegistryCoverageResult containing detailed metrics for depth 1, depth 2, depth 3+,
    and aggregate deeper mules, along with configured bounds and a formatted ASCII table.

How it works:
    1. Collects unique mule accounts and their assigned depth levels from simulation truth.
    2. Collects unique account identifiers reported to the national registry.
    3. Groups mules into depth 1 (first layer), depth 2, and depth 3+ (layer 3 and beyond),
       as well as an aggregate 'deeper' category (all depths >= 2).
    4. Computes observed coverage fraction (reported mules / all mules) for each tier.
    5. Reads reference intervals for layer 1 and deeper tiers from world_ranges.
    6. Evaluates the pass condition: layer-1 coverage must be strictly greater than deeper coverage.
    7. Formats a clear comparison table showing counts, percentages, and reference bounds.

Errors and side effects:
    Raises ValueError if split is unrecognized or required configuration keys are missing.
    Does not modify simulation state or input collections.

Limits:
    Evaluates registry presence only; does not evaluate whether reports resulted in holds
    or successful asset recovery (handled in later evaluation modules).
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from world.config_reader import get_parameter_entry, read_world_ranges


@dataclass(frozen=True)
class MuleDepthMetrics:
    """Coverage statistics for a single mule depth tier.

    What it is for:
        Encapsulates population counts, reported counts, coverage percentage,
        and reference bounds for one depth category.

    Inputs:
        depth_label: Descriptive label for the depth (e.g. '1', '2', '3+', 'deeper').
        total_mules: Total distinct mule accounts at this depth in truth.
        reported_mules: Count of those mule accounts present in registry reports.
        coverage: Ratio of reported mules to total mules (0.0 if total is 0).
        configured_range: Expected reference interval [min, max] from world configuration.
    """

    depth_label: str
    total_mules: int
    reported_mules: int
    coverage: float
    configured_range: tuple[float, float]


@dataclass(frozen=True)
class RegistryCoverageResult:
    """Aggregate registry coverage evaluation report across all mule depths.

    What it is for:
        Summarizes hierarchical detection coverage, verifies the layer-1 dominance rule,
        and provides printable tabular output for reports and showcase checks.

    Inputs:
        by_depth: Mapping from depth label to individual MuleDepthMetrics.
        layer1_coverage: Observed coverage fraction for layer 1 mules.
        deeper_coverage: Observed coverage fraction for deeper mules (depth >= 2).
        layer1_range: Configured reference bounds for layer 1 mules.
        deeper_range: Configured reference bounds for deeper mules.
        passed: True if layer 1 coverage strictly exceeds deeper coverage.
        table_text: Pre-formatted ASCII table displaying the results.
    """

    by_depth: dict[str, MuleDepthMetrics]
    layer1_coverage: float
    deeper_coverage: float
    layer1_range: tuple[float, float]
    deeper_range: tuple[float, float]
    passed: bool
    table_text: str


def compute_registry_coverage(
    roles: Sequence[Any],
    registry_reports: Sequence[Any],
    world_ranges: dict[str, Any] | None = None,
    split: str = "training",
) -> RegistryCoverageResult:
    """Compute registry coverage fractions broken down by mule depth.

    What it is for:
        Evaluates how effectively victim complaints and bank confirmations detect
        first-hop mules versus downstream money-laundering mules.

    Inputs:
        roles: Sequence of truth RoleSpan instances or dictionaries with role, account_id, depth.
            Example: [RoleSpan(account_id='acc_1', role='mule', depth=1, ...)]
        registry_reports: Sequence of RegistryReport instances or dictionaries with account_id.
            Example: [RegistryReport(account_id='acc_1', ...)]
        world_ranges: Optional loaded configuration dictionary; defaults to reading disk.
        split: Range partition name ('training', 'test', 'stress').

    What it returns:
        A RegistryCoverageResult containing metrics by depth and pass/fail status.

    How it works:
        1. Identifies all accounts holding a 'mule' role in truth and records their integer depth.
        2. Collects all accounts that appear in at least one registry report.
        3. Partitions mules into depth 1, depth 2, depth 3+, and deeper (depth >= 2).
        4. Calculates coverage share (reported / total) for each group.
        5. Looks up configured reference ranges in world_ranges.
        6. Asserts pass rule: layer 1 coverage > deeper coverage.
        7. Builds an aligned table string for display.

    Errors and side effects:
        Raises ValueError if an unexpected role depth (< 1) is encountered or split is invalid.
    """
    # 1. Collect mule accounts and their depths from truth roles
    mules_by_depth: dict[str, int] = {}
    for r in roles:
        role_name = getattr(r, "role", None) if not isinstance(r, dict) else r.get("role")
        if role_name != "mule":
            continue

        account_id = getattr(r, "account_id", None) if not isinstance(r, dict) else r.get("account_id")
        depth = getattr(r, "depth", None) if not isinstance(r, dict) else r.get("depth")

        if account_id is None:
            raise ValueError(f"Truth mule role record is missing account_id: {r}")
        if depth is None or depth < 1:
            raise ValueError(f"Truth mule '{account_id}' must have depth >= 1, got {depth}")

        # Account occupies this depth in the gang's laundering chain
        mules_by_depth[str(account_id)] = int(depth)

    # 2. Collect unique accounts reported to the registry
    reported_accounts: set[str] = set()
    for rep in registry_reports:
        account_id = getattr(rep, "account_id", None) if not isinstance(rep, dict) else rep.get("account_id")
        if account_id is not None:
            reported_accounts.add(str(account_id))

    # 3. Categorize mules into depth subsets
    d1_accounts = {acc for acc, d in mules_by_depth.items() if d == 1}
    d2_accounts = {acc for acc, d in mules_by_depth.items() if d == 2}
    d3plus_accounts = {acc for acc, d in mules_by_depth.items() if d >= 3}
    deeper_accounts = {acc for acc, d in mules_by_depth.items() if d >= 2}

    # 4. Load configured reference ranges
    ranges_data = world_ranges if world_ranges is not None else read_world_ranges()
    layer1_entry = get_parameter_entry(ranges_data, "registry_coverage_layer1_mule_share")
    deeper_entry = get_parameter_entry(ranges_data, "registry_coverage_deeper_mule_share")

    if split not in layer1_entry:
        raise ValueError(f"Split '{split}' not found in registry_coverage_layer1_mule_share")
    if split not in deeper_entry:
        raise ValueError(f"Split '{split}' not found in registry_coverage_deeper_mule_share")

    layer1_range = (float(layer1_entry[split][0]), float(layer1_entry[split][1]))
    deeper_range = (float(deeper_entry[split][0]), float(deeper_entry[split][1]))

    # 5. Calculate metrics for each group
    def _make_metric(label: str, group: set[str], bounds: tuple[float, float]) -> MuleDepthMetrics:
        total = len(group)
        reported = len(group & reported_accounts)
        share = (reported / total) if total > 0 else 0.0
        return MuleDepthMetrics(
            depth_label=label,
            total_mules=total,
            reported_mules=reported,
            coverage=share,
            configured_range=bounds,
        )

    m_d1 = _make_metric("1", d1_accounts, layer1_range)
    m_d2 = _make_metric("2", d2_accounts, deeper_range)
    m_d3plus = _make_metric("3+", d3plus_accounts, deeper_range)
    m_deeper = _make_metric("deeper", deeper_accounts, deeper_range)

    by_depth = {
        "1": m_d1,
        "2": m_d2,
        "3+": m_d3plus,
        "deeper": m_deeper,
    }

    layer1_coverage = m_d1.coverage
    deeper_coverage = m_deeper.coverage
    # Pass rule: layer-1 coverage must be strictly greater than deeper coverage
    passed = layer1_coverage > deeper_coverage

    # 6. Build printable ASCII comparison table
    def _format_row(label: str, m: MuleDepthMetrics, bounds: tuple[float, float]) -> str:
        counts = f"{m.reported_mules:4d} / {m.total_mules:4d}"
        pct = f"{m.coverage * 100:6.2f}%"
        range_str = f"[{bounds[0]:.2f}, {bounds[1]:.2f}]"
        return f"{label:<10} | {counts:<16} | {pct:>9} | {range_str}"

    lines = [
        "Mule Depth | Reported / Total |  Coverage | Configured Range",
        "-----------+------------------+-----------+-----------------",
        _format_row("1", m_d1, layer1_range),
        _format_row("2", m_d2, deeper_range),
        _format_row("3+", m_d3plus, deeper_range),
        _format_row("Deeper(2+)", m_deeper, deeper_range),
    ]
    table_text = "\n".join(lines)

    return RegistryCoverageResult(
        by_depth=by_depth,
        layer1_coverage=layer1_coverage,
        deeper_coverage=deeper_coverage,
        layer1_range=layer1_range,
        deeper_range=deeper_range,
        passed=passed,
        table_text=table_text,
    )


def evaluate_world_registry_coverage(
    output: Any,
    registry_reports: Sequence[Any] | None = None,
    split: str | None = None,
) -> RegistryCoverageResult:
    """Evaluate registry coverage directly for a completed simulation run output.

    What it is for:
        Convenience wrapper extracting truth roles, registry reports, and world settings
        from an InteractiveWorldOutput object.

    Inputs:
        output: Completed InteractiveWorldOutput instance.
        registry_reports: Optional sequence of RegistryReport objects; built from labels if omitted.
        split: Optional range name override; defaults to world's effective range setting.

    What it returns:
        RegistryCoverageResult summarizing coverage across mule depths.
    """
    effective = output.effective_config
    world_ranges = effective["world_configs"]["world_ranges"]
    effective_split = split if split is not None else effective["settings"]["range_name"]

    reports = registry_reports
    if reports is None:
        # Build registry reports from complaints if not supplied directly
        from world.label_builder import build_label_events, build_registry_reports

        complaints = build_label_events(output)
        reports = build_registry_reports(output, complaints)

    return compute_registry_coverage(
        roles=output.result.truth.roles,
        registry_reports=reports,
        world_ranges=world_ranges,
        split=effective_split,
    )
