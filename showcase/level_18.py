# File: showcase/level_18.py
# Purpose: Level 18 showcase: the mule account and payment results exactly as the console files record them.
"""
WHAT: Level 18 freezes the mule account and payment models and publishes their results as nine console
files under reports/level_18/console/. This showcase reads those files and shows seven things: the
account comparison with its spreads, the two held-out families, recall by hidden depth, all seven
leakage checks including the recorded 6b failure, the calibration bins and ECE before and after
isotonic, ONNX parity against the native model, and the measured yield against the E1 re-weighting
with its inflation factor.

WHY: The numbers are frozen; the only honest way to show them is to read them and check that they are
complete and agree with what was pre-registered, not to recompute or restate them from elsewhere.
"""

from __future__ import annotations

from typing import Any

from showcase.framework import CheckDefinition

LEVEL_NUMBER = 18
LEVEL_TITLE = "Mule account and payment models on frozen results (SIMULATED)"
NEEDS_DATABASE = False

WHAT_YOU_SAW = (
    "Level 18's results are SIMULATED: they come from the committed console files, which copy the frozen "
    "Run A and Level 18b results. The account model's xgboost recall at capacity has a median of 0.642857 "
    "across 120 test worlds, above both the rules (0.368112) and rules-and-anomaly (0.375000) baselines, "
    "with p10 0.477045 and p90 0.833333. The held-out device-rotation and slow-drip families keep their "
    "world counts and both show better recall at the cut than at capacity. Recall by hidden depth 1 to 5 "
    "keeps its spreads and world counts. The leakage page shows all seven published checks plus check 4's "
    "recorded trivial-world pass: seven pass and the payment check_6b is a FINDING — a documented "
    "limitation (A-051) — with an average precision over the base rate of 10.415126 against its 2.000000 "
    "limit; check 4 passes with its best single feature at 0.028270 against the 0.900000 threshold. "
    "Calibration shows the reliability "
    "bins before and after isotonic: the account ECE falls from 0.127137 to 0.001843, while the payment "
    "ECE stays close (0.017176 to 0.013593) and is shown as recorded. The ONNX files match the native "
    "models within the 1e-5 tolerance with zero rows over it, and the merge proof keeps zero decision "
    "flips. The E1 re-weighting shows the xgboost median yield falling from 0.923077 measured to 0.272955 "
    "re-weighted, with a median inflation factor of 39.316691 over 118 of the 120 worlds."
)


def find_missing_preconditions() -> list[str]:
    """Checks the committed Level 18 console files are present before any check runs."""
    from showcase import level_18_checks as c

    return c.find_missing_preconditions()


def get_checks(quick: bool = False) -> list[CheckDefinition]:
    """Returns the Level 18 checks in the A-065 order; quick mode runs the same seven reads."""
    from showcase import level_18_checks as c

    def d(name: str, kind: str, proves: str, run: Any, cause: str, where: str, part: str) -> CheckDefinition:
        """One check definition with its part tag."""
        return CheckDefinition(
            name=name, kind=kind, proves=proves, run=run, likely_cause=cause, where_to_look=where, part=part
        )

    return [
        d(
            "18-1 comparison: recall at capacity with p10 and p90 spreads",
            "normal",
            "The account comparison console reports ordered test spreads for all four scorers over 120 worlds, "
            "and the xgboost recall at capacity median sits above both rules baselines.",
            c.check_18_1_comparison,
            "A console value was changed or a spread lost its p10 or p90.",
            "reports/level_18/console/comparison.json, evaluation/rules_metrics.py",
            "a",
        ),
        d(
            "18-2 held-out families: device rotation and slow drip at capacity and at the cut",
            "normal",
            "Both held-out families account for every world as used or absent, and their recall at capacity "
            "and at the cut are ordered spreads.",
            c.check_18_2_held_out_families,
            "A family block lost its world counts or a recall spread.",
            "reports/level_18/console/held_out_families.json",
            "a",
        ),
        d(
            "18-3 hidden depth: recall by hidden mule depth 1 to 5",
            "edge",
            "Depths 1 to 5 are all present, every recall is an ordered spread, and the worlds used equal the "
            "worlds counted for that depth.",
            c.check_18_3_hidden_depth,
            "A depth was dropped or a world count no longer matches.",
            "reports/level_18/console/hidden_depth.json",
            "a",
        ),
        d(
            "18-4 leakage: all seven published checks, including the recorded 6b failure",
            "negative",
            "All seven published leakage outcomes and check 4 are present with a recorded pass or fail status; "
            "the only one that does not pass is check_6b, shown as the documented A-051 finding with its "
            "numbers, and a missing or not-run record fails the check.",
            c.check_18_4_leakage,
            "A leakage record was removed, reordered or left not run, or check 4's record was dropped.",
            "reports/level_18/console/leakage.json (data.checks, data.trivial_world), "
            "runs/full/run_a320_stage5c/final/45c496799170/leakage_checks.json:checks.trivial_world, "
            "docs/research/amendments.md (A-051, A-065a)",
            "b",
        ),
        d(
            "18-5 calibration: reliability bins before and after isotonic, with a chart",
            "normal",
            "Four reliability series hold ten ordered bins each whose counts sum to the published rows, and the "
            "account calibrated ECE is below its raw ECE; the scatter chart shows both curves.",
            c.check_18_5_calibration,
            "A bin was dropped, a count no longer sums to the rows, or the ECE moved above raw.",
            "reports/level_18/console/calibration.json, evaluation/calibration.py",
            "b",
        ),
        d(
            "18-6 ONNX vs native: parity on validation and test within the 1e-5 tolerance",
            "consistency",
            "Both ONNX models stay within the console's parity tolerance on validation and test with zero rows "
            "over it, and the account merge proof keeps zero decision flips, an identical top set and the "
            "Spearman floor.",
            c.check_18_6_onnx_parity,
            "A parity value exceeded the tolerance or the merge proof changed.",
            "reports/level_18/console/onnx_parity.json, evaluation/onnx_export.py",
            "b",
        ),
        d(
            "18-7 yield: measured against E1 re-weighting with the inflation factor",
            "edge",
            "Every one of the 120 worlds carries a re-weighted yield exactly when its inflation factor is "
            "positive, the published factor counts and median agree with the worlds, and the xgboost "
            "re-weighted median is below its measured median.",
            c.check_18_7_yield,
            "A factor and its re-weighted yield disagree, or the published median no longer matches.",
            "reports/level_18/console/yield_e1.json, config/evaluation.yaml",
            "a",
        ),
    ]
