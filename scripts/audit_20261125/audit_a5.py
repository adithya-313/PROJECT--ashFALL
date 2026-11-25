"""A5: world-level bootstrap 95% confidence intervals for three frozen headline metrics.

Resamples the 120 test worlds with replacement 2,000 times (seed 1717) and reports the 2.5th and
97.5th percentiles of the recomputed median. Reads frozen records only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs" / "full" / "audit_20261125"
RUN_A = ROOT / "runs" / "full" / "run_a320_stage5c" / "final" / "45c496799170" / "result.json"
V2 = ROOT / "runs" / "full" / "payment_model_v2" / "test_score" / "result.json"
RESAMPLES = 2000
SEED = 1717


def bootstrap_median(values: np.ndarray, rng: np.random.Generator) -> dict:
    """Returns the point median and the 2.5/97.5 percentile band of resampled medians."""
    n = values.size
    samples = np.empty(RESAMPLES, dtype=np.float64)
    for index in range(RESAMPLES):
        draw = values[rng.integers(0, n, size=n)]
        samples[index] = np.median(draw)
    return {
        "point_median": float(np.median(values)),
        "ci95_low": float(np.percentile(samples, 2.5)),
        "ci95_high": float(np.percentile(samples, 97.5)),
        "resamples": RESAMPLES,
        "worlds": int(n),
    }


def main() -> int:
    """Computes the three intervals and writes them to a5_bootstrap.json."""
    rng = np.random.default_rng(SEED)
    run_a = json.loads(RUN_A.read_text(encoding="utf-8"))
    v2 = json.loads(V2.read_text(encoding="utf-8"))

    worlds = list(run_a["per_world"].values())
    xgb = np.asarray([w["scorers"]["xgboost"]["recall_at_capacity"] for w in worlds])
    rules = np.asarray([w["scorers"]["rules"]["recall_at_capacity"] for w in worlds])
    gap = xgb - rules

    test_ap = np.asarray([w["truth_ap"] for w in v2["splits"]["test"]["worlds"] if w.get("truth_ap") is not None])

    document = {
        "simulated": True,
        "seed": SEED,
        "run_a_test_recall_at_capacity_median": bootstrap_median(xgb, rng),
        "run_a_test_xgboost_minus_rules_gap_median": bootstrap_median(gap, rng),
        "payment_v2_test_truth_ap_median": bootstrap_median(test_ap, rng),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "a5_bootstrap.json").write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8", newline="\n")
    for name, block in document.items():
        if isinstance(block, dict):
            print(
                f"{name:<48} median {block['point_median']:.6f} "
                f"95% CI [{block['ci95_low']:.6f}, {block['ci95_high']:.6f}] over {block['worlds']} worlds"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
