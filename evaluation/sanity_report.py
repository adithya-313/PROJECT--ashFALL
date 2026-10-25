"""
WHAT: Sanity report comparing synthetic world payment distributions against external baseline datasets (PaySim).
Computes amount quantile shapes divided by the median, payer repeat transaction share, payments per payer,
and payment type distribution over genuine transactions. Formats side-by-side comparison tables.

WHY: Verifies plausibility of simulated banking dynamics against established external benchmarks without using
external data to tune or set simulation parameters. Strictly adheres to Ground Rule 15 and issue 12-16.

TERM: genuine transaction: a successful payment where both payer and payee hold no gang role at that minute.
TERM: quantile shape: distribution quantiles normalized by dividing by the median amount (currency-independent).
TERM: payer repeat share: fraction of payments whose (payer, payee) directed pair appeared previously.
"""

import argparse
import importlib
import json
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from core.config_validator import load_yaml
from world.interactive_records import InteractiveWorldOutput
from world.run_identity import REPO_ROOT
from world.sampled_world import run_sampled_world
from world.world_configs import read_world_configs
from world.world_sampler import load_sampler_config, sample_block

DEFAULT_CONFIG_PATH: Path = REPO_ROOT / "config" / "sanity_report.yaml"
DEFAULT_PAYSIM_PATH: Path = REPO_ROOT / "data" / "enriched" / "paysim_enriched.parquet"
DEFAULT_SUMMARY_PATH: Path = REPO_ROOT / "reports" / "sanity" / "summary.md"
GANG_ROLES: frozenset[str] = frozenset({"mule", "collector", "cash_out"})
CODE_VERSION: str = "sanity-report"


def _compute_quantile(sorted_data: Sequence[float], q: float) -> float:
    """
    Computes empirical quantile with linear interpolation matching standard distribution conventions.
    Calculates fractional rank positions and interpolates between adjacent observations.

    Args:
        sorted_data: Sequence of real numbers already sorted in ascending order.
        q: Quantile specification value strictly in [0.0, 1.0].

    Returns:
        Interpolated real-valued quantile metric.

    Raises:
        ValueError: If the sequence is empty or q is outside [0.0, 1.0].
    """
    if not sorted_data:
        raise ValueError("Cannot compute quantile on empty sequence")
    if not (0.0 <= q <= 1.0):
        raise ValueError(f"Quantile parameter q must be in [0.0, 1.0], got {q}")
    idx = (len(sorted_data) - 1) * q
    low = int(idx)
    high = min(low + 1, len(sorted_data) - 1)
    weight = idx - float(low)
    return float(sorted_data[low] + weight * (sorted_data[high] - sorted_data[low]))


def load_sanity_report_config(path: Path | str | None = None) -> dict[str, Any]:
    """
    Parses configuration settings for outside data comparisons from sanity_report.yaml.
    Extracts policy rules, used/excluded column lists, world sample specifications, and quantile bounds.

    Args:
        path: Optional filesystem Path to configuration file; defaults to config/sanity_report.yaml.

    Returns:
        Dictionary containing the sanity_report configuration mapping.

    Raises:
        FileNotFoundError: If the specified configuration path does not exist on disk.
    """
    target = DEFAULT_CONFIG_PATH if path is None else Path(path)
    data = load_yaml(target)
    if "sanity_report" in data and isinstance(data["sanity_report"], dict):
        return data["sanity_report"]
    return data


def paysim_genuine_stats(path: Path | str, cfg: dict[str, Any]) -> dict[str, Any]:
    """
    Computes statistical distribution metrics over genuine rows from a PaySim parquet or csv dataset.
    Loads strictly the designated columns, validates that excluded balance columns are absent,
    filters out fraudulent transactions (isFraud == 0), and calculates quantiles, repeats, and payer frequencies.

    Args:
        path: Filesystem path to the PaySim dataset file (parquet or csv).
        cfg: Configuration dictionary defining column whitelists, blacklists, and quantiles.

    Returns:
        Dictionary containing genuine row count, amount quantile ratios, repeat share, and payer frequencies.

    Raises:
        ValueError: If excluded columns are detected in the loaded data or if genuine rows are empty.
    """
    resolved_path = Path(path)
    if not resolved_path.exists():
        raise ValueError(f"PaySim file not found at {resolved_path}")

    try:
        pq = importlib.import_module("pyarrow.parquet")
        pc = importlib.import_module("pyarrow.compute")
    except ModuleNotFoundError as err:
        raise ValueError(f"pyarrow is required to read PaySim dataset: {err}") from err

    columns_used = list(cfg["paysim_columns_used"])
    columns_excluded = list(cfg["paysim_columns_excluded"])
    quantiles = [float(q) for q in cfg["quantiles"]]

    if resolved_path.suffix in (".parquet", ".pq"):
        table = pq.read_table(resolved_path, columns=columns_used)
    elif resolved_path.suffix == ".csv":
        pcsv = importlib.import_module("pyarrow.csv")
        table = pcsv.read_csv(
            resolved_path,
            convert_options=pcsv.ConvertOptions(include_columns=columns_used),
        )
    else:
        raise ValueError(f"Unsupported file format for PaySim dataset: {resolved_path.suffix}")

    loaded_names = set(table.column_names)
    for excluded in columns_excluded:
        if excluded in loaded_names:
            raise ValueError(f"PaySim reader loaded forbidden excluded column: '{excluded}'")

    fraud_index = loaded_names.intersection({"isFraud", "isfraud"}).pop()
    mask = pc.equal(table[fraud_index], 0)
    genuine_table = table.filter(mask)
    total_genuine = genuine_table.num_rows
    if total_genuine == 0:
        raise ValueError("PaySim dataset contains zero genuine (isFraud == 0) records")

    sorted_amounts = sorted(float(a) for a in genuine_table["amount"].to_pylist())
    median_amount = _compute_quantile(sorted_amounts, 0.5)
    if median_amount <= 0.0:
        raise ValueError(f"PaySim genuine amount median must be positive, got {median_amount}")

    quantile_values = {q: _compute_quantile(sorted_amounts, q) for q in quantiles}
    quantile_ratios = {q: quantile_values[q] / median_amount for q in quantiles}

    orig_list = genuine_table["nameOrig"].to_pylist()
    dest_list = genuine_table["nameDest"].to_pylist()
    seen_pairs: set[tuple[str, str]] = set()
    repeat_count = 0
    payer_frequencies: dict[str, int] = {}

    for orig, dest in zip(orig_list, dest_list):
        payer_frequencies[orig] = payer_frequencies.get(orig, 0) + 1
        pair = (orig, dest)
        if pair in seen_pairs:
            repeat_count += 1
        else:
            seen_pairs.add(pair)

    repeat_share = float(repeat_count / total_genuine)
    sorted_payer_counts = sorted(float(c) for c in payer_frequencies.values())
    payer_median = _compute_quantile(sorted_payer_counts, 0.5)
    payer_p90 = _compute_quantile(sorted_payer_counts, 0.9)

    type_counts = genuine_table["type"].value_counts().to_pylist()
    type_shares = {
        str(item["values"]): float(item["counts"]) / float(total_genuine) for item in type_counts
    }

    return {
        "genuine_count": total_genuine,
        "amount_median": median_amount,
        "amount_quantiles": quantile_values,
        "amount_quantile_ratios": quantile_ratios,
        "payer_repeat_share": repeat_share,
        "payments_per_payer": {
            "median": payer_median,
            "p90": payer_p90,
        },
        "type_shares": type_shares,
    }


def _extract_gang_spans(output: Any) -> list[Any]:
    """
    Extracts active criminal role spans from world truth artifacts or test result fixtures.
    Inspects output result metadata to gather RoleSpan entities with roles matching mule, collector, or cash_out.

    Args:
        output: InteractiveWorldOutput instance or dictionary fixture from unit test scenarios.

    Returns:
        List of RoleSpan instances holding designated criminal syndicate functions.
    """
    if isinstance(output, dict):
        roles = output.get("roles", [])
    elif hasattr(output, "result") and hasattr(output.result, "truth"):
        roles = output.result.truth.roles
    else:
        roles = []
    return [r for r in roles if getattr(r, "role", "") in GANG_ROLES]


def _extract_events(output: Any) -> Sequence[Any]:
    """
    Retrieves chronological payment events from simulation results or test dictionaries.
    Provides uniform event sequence extraction across full InteractiveWorldOutput records and lightweight test fixtures.

    Args:
        output: Simulation output object or test dictionary holding payment event instances.

    Returns:
        Sequence of payment events emitted across the simulation timeframe.
    """
    if isinstance(output, dict):
        return output.get("events", [])
    if hasattr(output, "result") and hasattr(output.result, "run_result"):
        return output.result.run_result.events
    return []


def world_genuine_stats(outputs: Sequence[Any], cfg: dict[str, Any]) -> dict[str, Any]:
    """
    Calculates empirical distribution statistics across genuine successful payments pooled over the world sample.
    Filters transactions where both payer and payee hold no gang role at that minute, and calculates
    amount quantiles divided by median, payer repeat share, and payments per payer.

    Args:
        outputs: Sequence of completed InteractiveWorldOutput simulation results or test fixtures.
        cfg: Configuration dictionary defining quantile evaluation points.

    Returns:
        Dictionary containing pooled genuine payment statistics and not-comparable type indicator.

    Raises:
        ValueError: If zero genuine successful payments are found across the pooled sample.
    """
    quantiles = [float(q) for q in cfg["quantiles"]]
    all_amounts: list[float] = []
    seen_pairs: set[tuple[int, str, str]] = set()
    repeat_count = 0
    payer_frequencies: dict[tuple[int, str], int] = {}

    for world_idx, output in enumerate(outputs):
        gang_spans = _extract_gang_spans(output)
        events = _extract_events(output)

        for ev in events:
            if getattr(ev, "status", "") != "success":
                continue

            minute = getattr(ev, "minute", 0)
            from_acc = getattr(ev, "from_account", "")
            to_acc = getattr(ev, "to_account", "")

            payer_gang = any(
                r.account_id == from_acc and r.valid_from <= minute and (r.valid_to is None or minute < r.valid_to)
                for r in gang_spans
            )
            payee_gang = any(
                r.account_id == to_acc and r.valid_from <= minute and (r.valid_to is None or minute < r.valid_to)
                for r in gang_spans
            )
            if payer_gang or payee_gang:
                continue

            amount = float(getattr(ev, "amount_paise", 0))
            all_amounts.append(amount)

            pair = (world_idx, from_acc, to_acc)
            if pair in seen_pairs:
                repeat_count += 1
            else:
                seen_pairs.add(pair)

            payer_key = (world_idx, from_acc)
            payer_frequencies[payer_key] = payer_frequencies.get(payer_key, 0) + 1

    total_genuine = len(all_amounts)
    if total_genuine == 0:
        raise ValueError("Pooled world sample contains zero genuine successful payments")

    sorted_amounts = sorted(all_amounts)
    median_amount = _compute_quantile(sorted_amounts, 0.5)
    if median_amount <= 0.0:
        raise ValueError(f"World genuine amount median must be positive, got {median_amount}")

    quantile_values = {q: _compute_quantile(sorted_amounts, q) for q in quantiles}
    quantile_ratios = {q: quantile_values[q] / median_amount for q in quantiles}

    repeat_share = float(repeat_count / total_genuine)
    sorted_payer_counts = sorted(float(c) for c in payer_frequencies.values())
    payer_median = _compute_quantile(sorted_payer_counts, 0.5)
    payer_p90 = _compute_quantile(sorted_payer_counts, 0.9)

    return {
        "genuine_count": total_genuine,
        "amount_median": median_amount,
        "amount_quantiles": quantile_values,
        "amount_quantile_ratios": quantile_ratios,
        "payer_repeat_share": repeat_share,
        "payments_per_payer": {
            "median": payer_median,
            "p90": payer_p90,
        },
        "type_shares": "not comparable (the world has no PaySim types)",
    }


def _format_ratio(world_val: float, paysim_val: float) -> tuple[str, float]:
    """
    Calculates the numerical ratio between world and PaySim metrics and returns display string and float value.
    Handles zero-valued denominators gracefully by producing infinity indicators while tracking divergence.

    Args:
        world_val: Real-valued metric computed from synthetic world simulation.
        paysim_val: Real-valued benchmark metric computed from PaySim genuine data.

    Returns:
        Tuple of (formatted_ratio_string, numerical_ratio_float).
    """
    if paysim_val == 0.0:
        return ("inf", float("inf")) if world_val > 0.0 else ("1.0000", 1.0)
    ratio = world_val / paysim_val
    return f"{ratio:.4f}", ratio


def build_report(
    world_stats: dict[str, Any],
    paysim_stats: dict[str, Any],
    cfg: dict[str, Any],
) -> str:
    """
    Formats markdown sanity comparison report containing side-by-side tables and plain-text differences.
    Precedes all content with the required disclaimer sentence, produces structured markdown tables,
    and enumerates every comparison where the world-to-PaySim ratio falls outside the [0.5, 2.0] interval.

    Args:
        world_stats: Output dictionary from world_genuine_stats.
        paysim_stats: Output dictionary from paysim_genuine_stats.
        cfg: Configuration dictionary loaded from sanity_report.yaml.

    Returns:
        Structured markdown report string ready for publication in reports/sanity/summary.md.
    """
    lines: list[str] = [
        "Both datasets are synthetic, so agreement with them shows plausibility, not realism. "
        "This report is recorded and never used to set or tune any world setting.",
        "",
        "## Amount Quantiles Divided by Median",
        "",
        "| Quantile | World (shape) | PaySim (shape) | Ratio (World / PaySim) |",
        "| --- | --- | --- | --- |",
    ]

    quantiles = [float(q) for q in cfg["quantiles"]]
    differences: list[str] = []

    for q in quantiles:
        w_shape = world_stats["amount_quantile_ratios"][q]
        p_shape = paysim_stats["amount_quantile_ratios"][q]
        ratio_str, ratio_val = _format_ratio(w_shape, p_shape)
        lines.append(f"| {q} | {w_shape:.4f} | {p_shape:.4f} | {ratio_str} |")
        if ratio_val < 0.5 or ratio_val > 2.0:
            differences.append(
                f"- Amount quantile {q} / median: World is {w_shape:.4f}, PaySim is {p_shape:.4f} "
                f"(ratio {ratio_str}, outside [0.5, 2.0])."
            )

    lines.extend([
        "",
        "## Payer Repeat Share",
        "",
        "| Metric | World | PaySim | Ratio (World / PaySim) |",
        "| --- | --- | --- | --- |",
    ])
    w_repeat = world_stats["payer_repeat_share"]
    p_repeat = paysim_stats["payer_repeat_share"]
    repeat_ratio_str, repeat_ratio_val = _format_ratio(w_repeat, p_repeat)
    lines.append(f"| Payer repeat share | {w_repeat:.4f} | {p_repeat:.4f} | {repeat_ratio_str} |")
    if repeat_ratio_val < 0.5 or repeat_ratio_val > 2.0:
        differences.append(
            f"- Payer repeat share: World is {w_repeat:.4f}, PaySim is {p_repeat:.4f} "
            f"(ratio {repeat_ratio_str}, outside [0.5, 2.0])."
        )

    lines.extend([
        "",
        "## Payments Per Payer",
        "",
        "| Metric | World | PaySim | Ratio (World / PaySim) |",
        "| --- | --- | --- | --- |",
    ])
    w_med_payer = world_stats["payments_per_payer"]["median"]
    p_med_payer = paysim_stats["payments_per_payer"]["median"]
    med_ratio_str, med_ratio_val = _format_ratio(w_med_payer, p_med_payer)
    lines.append(f"| Median payments per payer | {w_med_payer:.1f} | {p_med_payer:.1f} | {med_ratio_str} |")
    if med_ratio_val < 0.5 or med_ratio_val > 2.0:
        differences.append(
            f"- Median payments per payer: World is {w_med_payer:.1f}, PaySim is {p_med_payer:.1f} "
            f"(ratio {med_ratio_str}, outside [0.5, 2.0])."
        )

    w_p90_payer = world_stats["payments_per_payer"]["p90"]
    p_p90_payer = paysim_stats["payments_per_payer"]["p90"]
    p90_ratio_str, p90_ratio_val = _format_ratio(w_p90_payer, p_p90_payer)
    lines.append(f"| P90 payments per payer | {w_p90_payer:.1f} | {p_p90_payer:.1f} | {p90_ratio_str} |")
    if p90_ratio_val < 0.5 or p90_ratio_val > 2.0:
        differences.append(
            f"- P90 payments per payer: World is {w_p90_payer:.1f}, PaySim is {p_p90_payer:.1f} "
            f"(ratio {p90_ratio_str}, outside [0.5, 2.0])."
        )

    lines.extend([
        "",
        "## Payment Types",
        "",
        "| Type | PaySim Share | World Share | Comparison Note |",
        "| --- | --- | --- | --- |",
    ])
    for p_type, share in sorted(paysim_stats["type_shares"].items()):
        lines.append(f"| {p_type} | {share:.4f} | not comparable | the world has no PaySim types |")

    lines.extend([
        "",
        "## Differences",
        "",
    ])
    if differences:
        lines.extend(differences)
    else:
        lines.append("- No metric ratio sits outside the [0.5, 2.0] plausibility corridor.")

    lines.extend([
        "",
        "## External Datasets Status",
        "",
        "IBM AML-Data: pending (not downloaded); register 12-16 records it.",
        "",
    ])

    return "\n".join(lines)


def run_world_sample(cfg: dict[str, Any], count_override: int | None = None) -> list[InteractiveWorldOutput]:
    """
    Executes the designated batch of simulated worlds using the many-worlds sampler and pinned configs.
    Loads training block 0 under the small preset, runs full interactive simulations with fraud on,
    and returns completed InteractiveWorldOutput records for distribution evaluation.

    Args:
        cfg: Configuration dictionary defining split, block index, preset, and world count.
        count_override: Optional integer to limit the executed world count for unit test execution.

    Returns:
        List of completed InteractiveWorldOutput instances.
    """
    sampler_cfg = load_sampler_config()
    world_configs = read_world_configs()
    sample_cfg = cfg["world_sample"]

    split = str(sample_cfg.get("split", "training"))
    block_num = int(sample_cfg.get("block", 0))
    preset_name = str(sample_cfg.get("preset", "small"))
    total_count = int(sample_cfg.get("count", 40)) if count_override is None else count_override

    specs = sample_block(split, block_num, 1, sampler_cfg, world_configs.world_ranges)
    selected_specs = specs[:total_count]

    outputs: list[InteractiveWorldOutput] = []
    for spec in selected_specs:
        out = run_sampled_world(spec, preset_name, None, sampler_cfg, CODE_VERSION)
        outputs.append(out)

    return outputs


def main(argv: list[str] | None = None) -> int:
    """
    Command line orchestrator reading PaySim, running sampled worlds, and generating the sanity report.
    Validates external output paths, records markdown summary tables to reports/sanity/summary.md,
    and dumps raw metrics to a JSON file outside the repository directory.

    Args:
        argv: Optional list of command-line argument strings to parse instead of sys.argv.

    Returns:
        Integer exit code (0 on success).
    """
    parser = argparse.ArgumentParser(description="World sanity report against PaySim genuine rows.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to config/sanity_report.yaml")
    parser.add_argument("--paysim-path", default=str(DEFAULT_PAYSIM_PATH), help="Path to PaySim parquet file")
    parser.add_argument("--summary-out", default=str(DEFAULT_SUMMARY_PATH), help="Output path for summary.md")
    parser.add_argument(
        "--raw-json",
        default=str(Path(tempfile.gettempdir()) / "ashfall_sanity_raw_numbers.json"),
        help="Output path for raw numbers JSON; must reside outside repository",
    )
    parser.add_argument("--world-count", type=int, default=None, help="Optional count of worlds to execute")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    raw_json_path = Path(args.raw_json).resolve()
    if REPO_ROOT == raw_json_path or REPO_ROOT in raw_json_path.parents:
        parser.error(f"--raw-json path {raw_json_path} must reside outside repository root")

    cfg = load_sanity_report_config(Path(args.config))
    paysim_stats = paysim_genuine_stats(Path(args.paysim_path), cfg)
    outputs = run_world_sample(cfg, count_override=args.world_count)
    world_stats = world_genuine_stats(outputs, cfg)

    report_md = build_report(world_stats, paysim_stats, cfg)

    summary_out_path = Path(args.summary_out)
    summary_out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_out_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    raw_data = {
        "paysim": paysim_stats,
        "world": world_stats,
    }
    raw_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(raw_json_path, "w", encoding="utf-8") as f:
        json.dump(raw_data, f, indent=2)

    print(report_md)
    print(f"\nSummary markdown written to {summary_out_path}")
    print(f"Raw numbers JSON written to {raw_json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
