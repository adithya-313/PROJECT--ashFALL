# File: scripts/audit_chart.py
# Purpose: Draw the Level 9 bar chart from reports/model/v1_audit_results.json: one horizontal bar per variant (original
#          score, control, ablations, out-of-time runs, every single feature), a vertical line at the base rate (what a
#          model with no skill scores) and the value written at the end of each bar.
#
# Run from the repo root:   python scripts/audit_chart.py
import io
import sys
from pathlib import Path
from typing import Any

import orjson

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from core.frozen_artifacts import guarded_write_bytes  # noqa: E402

RESULTS_PATH = REPO_ROOT / "reports" / "model" / "v1_audit_results.json"
CHART_PATH = REPO_ROOT / "reports" / "model" / "v1_audit_chart.png"

# Bar colours by group, so the groups are easy to tell apart.
GROUP_COLOURS = {"original": "#1b4f72", "control": "#2e86c1", "ablation": "#c0392b", "out_of_time": "#d68910",
                 "single_feature": "#7f8c8d"}


def ordered_rows(results: dict[str, Any]) -> list[dict[str, Any]]:
    """The rows in the order they are drawn: as saved (original, control, ablations, out-of-time, single features)."""
    return list(results["rows"])


def make_chart_png(results: dict[str, Any]) -> bytes:
    """Draw the chart and return it as PNG bytes. matplotlib is imported here so other scripts stay light."""
    import matplotlib

    matplotlib.use("Agg")  # draw into memory, no window
    import matplotlib.pyplot as plt

    rows = ordered_rows(results)
    figure, axis = plt.subplots(figsize=(10, 0.38 * len(rows) + 1.5))
    positions = list(range(len(rows)))
    scores = [row["pr_auc"] for row in rows]
    axis.barh(positions, scores, color=[GROUP_COLOURS.get(row["group"], "#7f8c8d") for row in rows])
    for position, score in zip(positions, scores):
        axis.text(score + 0.01, position, f"{score:.3f}", va="center", fontsize=8)
    axis.axvline(results["base_rate"], color="black", linestyle="--", linewidth=1)
    axis.text(results["base_rate"] + 0.01, -0.9, f"base rate {results['base_rate']:.3f}", fontsize=8)
    axis.set_yticks(positions)
    axis.set_yticklabels([row["name"] for row in rows], fontsize=8)
    axis.invert_yaxis()  # first row at the top
    axis.set_xlim(0, 1.12)
    axis.set_xlabel("PR-AUC on the test set (higher is better)")
    axis.set_title("Model v1 audit: where does the score come from?")
    figure.tight_layout()
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=130)
    plt.close(figure)
    return buffer.getvalue()


def main() -> int:
    """Read the saved results and write the chart."""
    results = orjson.loads(RESULTS_PATH.read_bytes())
    guarded_write_bytes(CHART_PATH, make_chart_png(results))
    print(f"Wrote {CHART_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
