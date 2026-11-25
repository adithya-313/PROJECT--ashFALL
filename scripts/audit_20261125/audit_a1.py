"""A1: recompute official Level 18 headline metrics from the frozen per-world records.

Imports nothing from evaluation/: the medians use a local sort-based median, and every value is
read from the committed or frozen JSON artifacts, never recomputed by the modules under audit.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs" / "full" / "audit_20261125"
RUN_A = ROOT / "runs" / "full" / "run_a320_stage5c" / "final" / "45c496799170" / "result.json"
EXTRA = ROOT / "runs" / "full" / "run_a320_stage5c" / "extra_reporting.json"
V2 = ROOT / "runs" / "full" / "payment_model_v2" / "test_score" / "result.json"


def median(values: list[float]) -> float:
    """Sort-based median (average of the two middle values for even counts)."""
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def load(path: Path) -> dict:
    """Read one JSON document."""
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    """Recompute the headline metrics and write a comparison table."""
    run_a = load(RUN_A)
    extra = load(EXTRA)
    v2 = load(V2)

    rows: list[dict] = []

    def record(name: str, official: float, independent: float, source: str) -> None:
        rows.append(
            {
                "metric": name,
                "official": official,
                "independent": independent,
                "abs_diff": abs(official - independent),
                "source": source,
            }
        )

    xgb_recall = [block["scorers"]["xgboost"]["recall_at_capacity"] for block in run_a["per_world"].values()]
    official_recall = run_a["across_worlds"]["test"]["xgboost"]["recall_at_capacity"]["median"]
    record("run_a_test_recall_at_capacity_median", official_recall, median(xgb_recall), "run_a result.json")

    ap_values = [block["pure"]["account_day_average_precision"] for block in extra["test"]["per_world"].values()]
    official_ap = extra["test"]["aggregate"]["pure"]["account_day_average_precision"]["median"]
    record("run_a_test_account_day_ap_median", official_ap, median(ap_values), "extra_reporting.json")

    test_worlds = [w["truth_ap"] for w in v2["splits"]["test"]["worlds"] if w.get("truth_ap") is not None]
    official_test = 0.6493820260824092
    record("payment_v2_test_truth_ap_median", official_test, median(test_worlds), "v2 result.json worlds")

    stress_worlds = [w["truth_ap"] for w in v2["splits"]["stress"]["worlds"] if w.get("truth_ap") is not None]
    official_stress = 0.12203008503375444
    record("payment_v2_stress_truth_ap_median", official_stress, median(stress_worlds), "v2 result.json worlds")

    payee = v2["splits"]["test"]["payee_precision"]
    record(
        "payment_v2_test_payee_precision",
        payee["payee_precision"],
        payee["positive_payees"] / payee["distinct_payees"],
        "v2 result.json payee counts",
    )

    for row in rows:
        row["flag"] = row["abs_diff"] > 1e-6
    document = {
        "simulated": True,
        "note": (
            "Medians recomputed from the frozen per-world records with a local median; per-world AP and "
            "recall values themselves are the frozen scorer's, so this checks aggregation only, not the "
            "AP implementation. Payee precision is positive_payees / distinct_payees from stored counts."
        ),
        "rows": rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "a1_recompute.json").write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8", newline="\n")
    for row in rows:
        print(
            f"{row['metric']:<40} official {row['official']:.12f} independent {row['independent']:.12f} "
            f"diff {row['abs_diff']:.3e} flag={row['flag']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
