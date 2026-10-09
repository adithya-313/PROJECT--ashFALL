"""
File: tests/unit/test_audit_report.py
Purpose: Proves the verdict generator: it takes the right branch for a big drop, for no drop and for a shortcut
         feature, it never says the score was fine just because no single feature explains it, the report lists every
         MLflow run id, and the NOT VERIFIED line appears when the PaySim sentence was not verified.
"""

from typing import Any

from scripts.audit_report import NOT_VERIFIED_LINE, build_verdict, render_report

ALL_SINGLES = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "type_CASH_IN",
               "type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER", "new_device_flag", "geo_velocity_spike",
               "account_avg_amount", "many_small_inbound_count", "large_outbound_amount"]


def make_results(no_balance: float = 0.80, no_type: float = 0.98, no_generated: float = 0.96,
                 best_single: tuple[str, float] = ("newbalanceOrig", 0.19), payment: float = 0.008) -> dict[str, Any]:
    """A small made-up results set shaped like reports/model/v1_audit_results.json."""
    def row(name: str, group: str, score: float) -> dict[str, Any]:
        return {"name": name, "group": group, "split": "level5_random", "pr_auc": score, "mlflow_run_id": f"run-{name}"}

    rows = [row("original_v1", "original", 0.995), row("control_all15", "control", 0.994),
            row("ablate_no_balance", "ablation", no_balance), row("ablate_no_type", "ablation", no_type),
            row("ablate_no_generated", "ablation", no_generated), row("oot_all15", "out_of_time", 0.99),
            row("oot_no_balance", "out_of_time", 0.66)]
    for feature in ALL_SINGLES:
        score = payment if feature == "type_PAYMENT" else 0.01
        if feature == best_single[0]:
            score = best_single[1]
        rows.append(row(f"single_{feature}", "single_feature", score))
    diagnostics = {"new_device_flag": {"share_nonzero_fraud": 0.46, "share_nonzero_nonfraud": 0.15}}
    return {"base_rate": 0.005, "rows": rows, "diagnostics": diagnostics, "oot_cutoff_step": 372, "seed": 42}


def test_big_drop_says_the_score_depends_on_the_balance_group() -> None:
    """A large drop without balance columns gives the DEPENDS sentence, the from/to numbers and the diagnostic."""
    verdict = build_verdict(make_results())
    assert "DEPENDS on the balance group" in verdict
    assert "from 0.994 to 0.800" in verdict
    assert "well above the base rate" in verdict
    assert "separability diagnostic" in verdict


def test_no_drop_says_the_score_survived_and_does_not_call_it_misleading() -> None:
    """When no ablation drops by LARGE_DROP, the verdict says it survived and still points to the diagnostic."""
    verdict = build_verdict(make_results(no_balance=0.99, no_type=0.99, no_generated=0.99))
    assert "the score survived" in verdict
    assert "DEPENDS" not in verdict and "misleading" not in verdict
    assert "separability diagnostic" in verdict


def test_a_shortcut_feature_is_named_as_a_shortcut() -> None:
    """One feature reaching SHORTCUT_SHARE of the original score is called a shortcut."""
    verdict = build_verdict(make_results(best_single=("type_PAYMENT", 0.95), payment=0.95))
    assert "That is a shortcut" in verdict


def test_no_single_feature_does_not_mean_the_score_is_fine() -> None:
    """When no single feature explains the score, the verdict says the strength comes from combining columns."""
    verdict = build_verdict(make_results())
    assert "combining columns" in verdict and "not evidence that the score was fine" in verdict


def test_payment_sentence_and_cpu_gap_wording() -> None:
    """type_PAYMENT alone has its own sentence with its lift, and the CPU-vs-GPU gap is called an ESTIMATE."""
    verdict = build_verdict(make_results(payment=0.0078))
    assert "type_PAYMENT alone scored 0.008" in verdict and "1.6 times the base rate" in verdict
    assert "ESTIMATE of CPU versus GPU" in verdict


def test_report_lists_every_run_id_and_shows_not_verified() -> None:
    """Every row's run id appears in the report, and an unverified PaySim source gives the visible NOT VERIFIED line."""
    results = make_results()
    source = {"source_url": "https://example.test/paysim", "retrieved_on": "2026-01-01", "exact_sentence": "",
              "verified": False}
    report = render_report(results, source)
    assert all(row["mlflow_run_id"] in report for row in results["rows"])
    assert NOT_VERIFIED_LINE in report
    assert all(f"## {number}." in report for number in range(1, 8))


def test_report_quotes_the_sentence_only_when_verified() -> None:
    """A verified source sentence is quoted, and the NOT VERIFIED line is then absent."""
    source = {"source_url": "https://example.test/paysim", "retrieved_on": "2026-01-01",
              "exact_sentence": "An exact test sentence.", "verified": True}
    report = render_report(make_results(), source)
    assert "> An exact test sentence." in report and NOT_VERIFIED_LINE not in report
