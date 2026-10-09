# File: scripts/audit_report.py
# Purpose: Turn the saved audit results (reports/model/v1_audit_results.json) into a plain-language verdict and the
#          report reports/model/v1_audit.md. The verdict is built ONLY from the numbers with simple if/else rules, so
#          every claim in it can be traced back to a row of the results.
#
# Run from the repo root:   python scripts/audit_report.py
import sys
from pathlib import Path
from typing import Any

import orjson

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.frozen_artifacts import guarded_write_bytes  # noqa: E402

RESULTS_PATH = REPO_ROOT / "reports" / "model" / "v1_audit_results.json"
PAYSIM_SOURCE_PATH = REPO_ROOT / "reports" / "model" / "paysim_warning_source.json"
REPORT_PATH = REPO_ROOT / "reports" / "model" / "v1_audit.md"

# Rules for the verdict.
LARGE_DROP = 0.05  # an ablation that lowers PR-AUC by at least this much (absolute) counts as a large drop
SHORTCUT_SHARE = 0.8  # one feature alone reaching this share of the original score counts as a shortcut
WELL_ABOVE_BASE = 10  # "well above the base rate" means at least this many times the base rate

ABLATIONS = {"ablate_no_balance": "the balance columns", "ablate_no_type": "the type flags",
             "ablate_no_generated": "the five generated columns"}
NOT_VERIFIED_LINE = "NOT VERIFIED — source sentence still to be added"


def row_named(results: dict[str, Any], name: str) -> dict[str, Any]:
    """The result row with this name (a clear error if the audit did not produce it)."""
    for row in results["rows"]:
        if row["name"] == name:
            return row
    raise KeyError(f"the results have no row named '{name}'")


def strongest_single_feature(results: dict[str, Any]) -> dict[str, Any]:
    """The single-feature row with the highest PR-AUC."""
    singles = [row for row in results["rows"] if row["group"] == "single_feature"]
    return max(singles, key=lambda row: row["pr_auc"])


def most_separable_generated_column(results: dict[str, Any]) -> tuple[str, float, float]:
    """The generated column whose non-zero share differs most between fraud and other rows: (name, fraud, other)."""
    diagnostics = results["diagnostics"]

    def gap(column: str) -> float:
        return abs(diagnostics[column]["share_nonzero_fraud"] - diagnostics[column]["share_nonzero_nonfraud"])

    name = max(diagnostics, key=gap)
    return name, diagnostics[name]["share_nonzero_fraud"], diagnostics[name]["share_nonzero_nonfraud"]


def ablation_sentences(results: dict[str, Any], control: float) -> list[str]:
    """One sentence per ablation: the score without that group and the drop from the control."""
    sentences = []
    for name, label in ABLATIONS.items():
        score = row_named(results, name)["pr_auc"]
        drop = control - score
        size = "a large drop" if drop >= LARGE_DROP else "a small drop"
        sentences.append(f"Without {label}, PR-AUC fell from {control:.3f} to {score:.3f} ({size} of {drop:.3f}).")
    return sentences


def diagnostic_sentence(results: dict[str, Any]) -> str:
    """The pointer to the separability diagnostic, with its most telling column."""
    column, fraud_share, other_share = most_separable_generated_column(results)
    return (
        "The separability diagnostic (the table in section 3) shows how easy the injected patterns are to tell apart: "
        f"{column} is non-zero in {fraud_share:.0%} of fraud rows and {other_share:.0%} of other rows."
    )


def balance_sentences(results: dict[str, Any], control: float, base_rate: float) -> list[str]:
    """What the balance ablation means, plus the pointer to the separability diagnostic."""
    no_balance = row_named(results, "ablate_no_balance")["pr_auc"]
    if control - no_balance >= LARGE_DROP:
        if no_balance >= WELL_ABOVE_BASE * base_rate:
            above = f"it still stayed well above the base rate of {base_rate:.3f}"
        else:
            above = f"it also fell close to the base rate of {base_rate:.3f}"
        return [
            f"Removing the balance columns dropped PR-AUC from {control:.3f} to {no_balance:.3f}, and {above}. "
            "So the score DEPENDS on the balance group, which is recorded after the transaction (a leak). The "
            "injected patterns may also be easy to separate on their own. " + diagnostic_sentence(results)
        ]
    return [
        f"No ablation lowered PR-AUC by {LARGE_DROP:.2f} or more, so the score survived taking each group away. "
        "That does not by itself prove the score is sound. " + diagnostic_sentence(results)
    ]


def out_of_time_sentence(results: dict[str, Any], control: float) -> str:
    """The out-of-time scores and their drop from the control."""
    oot = row_named(results, "oot_all15")["pr_auc"]
    oot_no_balance = row_named(results, "oot_no_balance")["pr_auc"]
    return (
        f"On an out-of-time split (train on steps up to {results['oot_cutoff_step']}, test on later steps) all 15 "
        f"features scored {oot:.3f}, a drop of {control - oot:.3f} from the control; without the balance columns the "
        f"out-of-time score was {oot_no_balance:.3f}."
    )


def single_feature_sentence(results: dict[str, Any], original: float) -> str:
    """The strongest single feature, and whether it is a shortcut. It never concludes 'the score is fine'."""
    best = strongest_single_feature(results)
    share = best["pr_auc"] / original
    name = best["name"].removeprefix("single_")
    start = (f"The strongest single feature was {name} with PR-AUC {best['pr_auc']:.3f}, "
             f"{share:.0%} of the original score.")
    if share >= SHORTCUT_SHARE:
        return start + " That is a shortcut: one column alone nearly explains the score."
    return (start + " No single feature explains the score; its strength comes from combining columns, so this is not "
            "evidence that the score was fine.")


def payment_sentence(results: dict[str, Any], original: float, base_rate: float) -> str:
    """type_PAYMENT alone: score, share of the original score and lift over the base rate."""
    score = row_named(results, "single_type_PAYMENT")["pr_auc"]
    return (f"type_PAYMENT alone scored {score:.3f}, which is {score / original:.1%} of the original score and "
            f"{score / base_rate:.1f} times the base rate of {base_rate:.3f}.")


def build_verdict(results: dict[str, Any]) -> str:
    """The plain-language verdict paragraph, built only from the numbers in the results."""
    original = row_named(results, "original_v1")["pr_auc"]
    control = row_named(results, "control_all15")["pr_auc"]
    base_rate = results["base_rate"]
    sentences = [
        f"Model v1's original PR-AUC was {original:.3f}, while a model with no skill would score about the base "
        f"rate of {base_rate:.3f}.",
        f"A copy retrained on the CPU with all 15 features scored {control:.3f}; the gap of {original - control:.4f} "
        "to the original is an ESTIMATE of CPU versus GPU training (it was not tested).",
    ]
    sentences += ablation_sentences(results, control)
    sentences += balance_sentences(results, control, base_rate)
    sentences += [out_of_time_sentence(results, control), single_feature_sentence(results, original),
                  payment_sentence(results, original, base_rate)]
    return " ".join(sentences)


def results_table(results: dict[str, Any]) -> str:
    """A markdown table of every row with its MLflow run id."""
    lines = ["| Variant | Group | Split | PR-AUC | MLflow run id |", "|---|---|---|---|---|"]
    for row in results["rows"]:
        cells = [row["name"], row["group"], row["split"], f"{row['pr_auc']:.5f}", row["mlflow_run_id"]]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def diagnostics_table(results: dict[str, Any]) -> str:
    """A markdown table of the generated-column separability diagnostic."""
    lines = ["| Generated column | Non-zero in fraud rows | Non-zero in other rows |", "|---|---|---|"]
    for column, shares in results["diagnostics"].items():
        lines.append(f"| {column} | {shares['share_nonzero_fraud']:.1%} | {shares['share_nonzero_nonfraud']:.1%} |")
    return "\n".join(lines)


def paysim_section(paysim_source: dict[str, Any]) -> str:
    """Section 5: PaySim's own guidance, quoted only if it was really read; otherwise a visible NOT VERIFIED line."""
    if paysim_source.get("verified") and paysim_source.get("exact_sentence"):
        return (f"> {paysim_source['exact_sentence']}\n\nSource: {paysim_source['source_url']} "
                f"(retrieved on {paysim_source['retrieved_on']}).")
    return (f"**{NOT_VERIFIED_LINE}**\n\nSource to check: {paysim_source.get('source_url', 'unknown')}. The page text "
            "could not be read automatically, so no sentence is quoted here. Nothing in this report depends on it.")


def bullets(items: list[str]) -> str:
    """A markdown bullet list."""
    return "\n".join(f"- {item}" for item in items)


DESIGN_RULES = [
    "Never use balances recorded after the transaction (new balances) as model inputs.",
    "Evaluate with an out-of-time split, not only a random split.",
    "Ablate groups of features and test features alone before trusting a score.",
    "Keep frozen artifacts behind a hash check, and never write to them from analysis code.",
]
LIMITS = [
    "The audit models were trained on the CPU; model v1 was trained on the GPU, so scores differ slightly.",
    "One seed (42) and one out-of-time cutoff (step 372) were used.",
    "The data is synthetic (PaySim plus injected patterns): scores say how well a model learned this simulator.",
    "Only PR-AUC was compared; thresholds, calibration and fairness were not re-audited.",
]


def how_section(results: dict[str, Any]) -> str:
    """Section 1: what was audited and how."""
    return (
        "## 1. What was audited and how\n\n"
        "Model v1 (XGBoost, PR-AUC 0.99) was left untouched; its files are hashed before and after the audit and "
        f"are unchanged. Copies of its training recipe were retrained in the MLflow experiment `level9_v1_audit` "
        f"(seed {results['seed']}, CPU): with all 15 features (control), without the balance columns, without the "
        "type flags, without the five generated columns, on an out-of-time split, and with each feature alone."
    )


def learned_section(results: dict[str, Any]) -> str:
    """Section 3: the numbers behind what the model learned (top single features and the separability table)."""
    singles = sorted((row for row in results["rows"] if row["group"] == "single_feature"),
                     key=lambda row: row["pr_auc"], reverse=True)[:3]
    top = bullets([f"{row['name'].removeprefix('single_')} alone: PR-AUC {row['pr_auc']:.3f}" for row in singles])
    return ("## 3. What the model learned\n\nThe three strongest single features:\n\n" + top
            + "\n\nHow easy the generated columns are to separate:\n\n" + diagnostics_table(results))


def render_report(results: dict[str, Any], paysim_source: dict[str, Any]) -> str:
    """The full markdown report with its seven sections."""
    verdict = build_verdict(results)
    sections = [
        "# Audit of model v1",
        how_section(results),
        "## 2. Results\n\n" + results_table(results) + f"\n\nBase rate of the test split: {results['base_rate']:.4f}.",
        learned_section(results),
        "## 4. Why the score misled (or did not)\n\n" + verdict,
        "## 5. PaySim's own guidance\n\n" + paysim_section(paysim_source),
        "## 6. Design rules that follow\n\n" + bullets(DESIGN_RULES),
        "## 7. Limits of this audit\n\n" + bullets(LIMITS),
    ]
    return "\n\n".join(sections) + "\n"


def main() -> int:
    """Read the saved results and write reports/model/v1_audit.md."""
    results = orjson.loads(RESULTS_PATH.read_bytes())
    paysim_source = orjson.loads(PAYSIM_SOURCE_PATH.read_bytes())
    guarded_write_bytes(REPORT_PATH, render_report(results, paysim_source).encode("utf-8"))
    print(f"Wrote {REPORT_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
