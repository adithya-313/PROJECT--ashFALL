"""
WHAT: Builds the label trust curve for config/mule_model.yaml from the measured label-delay report.

WHY: An unlabelled training row with k days left in its world is trusted as a negative with the
share of bank labels that had arrived k days after the first illicit credit. The curve must come
from the measured delays by a script, never typed by hand, so its source stays reproducible.

TERM: label delay: days from an account's first illicit credit to the arrival of its bank label.
TERM: trust curve: share_k = (number of delays <= k days) / (number of delays), for k = 0..max_days.

Usage:
    py -3.12 scripts/build_label_trust_curve.py --report runs/full/label_delay_r7.json \
        --expected-sha256 <hex> --max-days 90

Prints only the YAML list block holding the curve values. Every guard failure raises and prints
nothing else; the report is never modified.
"""

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

import orjson

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from evaluation.measure_label_delay import _percentiles  # noqa: E402

EXPECTED_WORLD_COUNT = 160
PERCENTILE_TOLERANCE = 1e-9
SHARE_DECIMALS = 6


def verify_report_hash(report_path: Path, expected_sha256: str) -> None:
    """
    WHAT: Refuses to continue unless the report file's sha256 equals the expected value.

    Raises:
        ValueError: If the hashes differ, so a changed report can never feed the curve.
    """
    actual_sha256 = hashlib.sha256(report_path.read_bytes()).hexdigest()
    if actual_sha256.lower() != expected_sha256.lower():
        raise ValueError(f"Report sha256 {actual_sha256} does not equal the expected {expected_sha256}")


def pool_delays(report: dict[str, Any]) -> list[float]:
    """
    WHAT: Concatenates every by_world entry's delays_days in file order and checks the pool.

    WHY: The report's pooled section holds only summary statistics, so the pooled list is rebuilt
    from the per-world lists and proven equal to those statistics before it is used.

    Raises:
        ValueError: If the world count, the pooled length, the negative count or the p50/p90
            percentiles disagree with the report's own pooled section.
    """
    by_world = report["by_world"]
    pooled_section = report["pooled"]
    if len(by_world) != EXPECTED_WORLD_COUNT:
        raise ValueError(f"by_world has {len(by_world)} entries; expected {EXPECTED_WORLD_COUNT}")

    pooled_delays: list[float] = []
    for world_entry in by_world:
        pooled_delays.extend(world_entry["delays_days"])

    if len(pooled_delays) != pooled_section["labelled_accounts"]:
        raise ValueError(
            f"Pooled delay count {len(pooled_delays)} does not equal "
            f"pooled.labelled_accounts {pooled_section['labelled_accounts']}"
        )

    negative_count = sum(1 for delay in pooled_delays if delay < 0)
    if negative_count != pooled_section["negative_delay_count"]:
        raise ValueError(
            f"Pooled negative count {negative_count} does not equal "
            f"pooled.negative_delay_count {pooled_section['negative_delay_count']}"
        )

    recomputed = _percentiles(sorted(pooled_delays))
    reported = pooled_section["delay_percentiles_days"]
    for name in ("p50", "p90"):
        if abs(recomputed[name] - reported[name]) > PERCENTILE_TOLERANCE:
            raise ValueError(f"Recomputed {name} {recomputed[name]} differs from the report's {reported[name]}")
    # Goes to stderr so stdout stays exactly the YAML block.
    summary = f"recomputed p50={recomputed['p50']!r} p90={recomputed['p90']!r} delays={len(pooled_delays)}"
    print(summary, file=sys.stderr)
    return pooled_delays


def trust_curve(delays: list[float], max_days: int) -> list[float]:
    """
    WHAT: Returns share_k for k = 0..max_days, with negative delays counted as 0 days.

    WHY: A negative delay means the label arrived no later than the credit, so it is trusted from day 0.

    Raises:
        ValueError: If there are no delays or the curve is not non-decreasing.
    """
    if not delays:
        raise ValueError("No delays to build a curve from")
    clamped_delays = [max(delay, 0.0) for delay in delays]
    total = len(clamped_delays)
    curve: list[float] = []
    for day in range(max_days + 1):
        arrived = sum(1 for delay in clamped_delays if delay <= day)
        curve.append(round(arrived / total, SHARE_DECIMALS))
    for index in range(1, len(curve)):
        if curve[index] < curve[index - 1]:
            raise ValueError(f"Trust curve decreases at day {index}")
    return curve


def format_yaml_block(curve: list[float], expected_sha256: str) -> str:
    """
    WHAT: Returns the label_trust_curve YAML block, indented to sit under `parameters:`.

    WHY: The block text is generated here so the config's numbers and their provenance cannot drift.
    """
    values_text = ", ".join(str(share) for share in curve)
    short_hash = f"{expected_sha256[:8]}...{expected_sha256[-4:]}"
    derived_from = (
        "scripts/build_label_trust_curve.py over runs/full/label_delay_r7.json "
        f"sha256 {short_hash} (pooled from the 160 by_world delays_days lists, 160 training worlds)"
    )
    reason = (
        "Share of bank labels that had arrived k days after the first illicit credit; an unlabelled "
        "training row with k days left in its world is trusted as a negative with this weight. Labelled "
        "delays only (censored accounts are missing), so the curve is optimistic - recorded as a limitation."
    )
    block_lines = [
        "  label_trust_curve:",
        "    label: measured",
        f"    value: [{values_text}]",
        f'    derived_from: "{derived_from}"',
        f'    reason: "{reason}"',
    ]
    return "\n".join(block_lines)


def main(argv: list[str] | None = None) -> int:
    """WHAT: Parses arguments, runs the guards and prints the YAML value block."""
    parser = argparse.ArgumentParser(description="Build the label trust curve YAML block.")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--max-days", type=int, required=True)
    args = parser.parse_args(argv)

    verify_report_hash(args.report, args.expected_sha256)
    report = orjson.loads(args.report.read_bytes())
    delays = pool_delays(report)
    curve = trust_curve(delays, args.max_days)

    print(format_yaml_block(curve, args.expected_sha256))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
