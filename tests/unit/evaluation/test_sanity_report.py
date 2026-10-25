"""
WHAT: Unit tests for external dataset sanity reporting against PaySim genuine rows (Level 12h, issue 12-16).
Verifies statistical computations against closed-form hand examples, proves that the PaySim reader never loads
excluded balance columns, enforces architectural isolation against tuning feedback loops, and checks real files.

WHY: Verifies that sanity comparison metrics are mathematically exact, ensures strict exclusion of forbidden balance
features from external datasets, and guarantees that synthetic world models are never tuned against external baselines.
"""

import ast
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from evaluation.sanity_report import (
    build_report,
    load_sanity_report_config,
    paysim_genuine_stats,
    world_genuine_stats,
)
from world.failures import PaymentEvent
from world.run_identity import REPO_ROOT
from world.truth_records import RoleSpan

DEFAULT_CONFIG: dict[str, Any] = {
    "paysim_columns_used": ["step", "type", "amount", "nameOrig", "nameDest", "isFraud"],
    "paysim_columns_excluded": [
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "isFlaggedFraud",
    ],
    "quantiles": [0.1, 0.25, 0.5, 0.75, 0.9, 0.99],
}


def test_hand_computed_paysim_and_world_stats(tmp_path: Path) -> None:
    """
    Validates statistical distribution calculations on hand-crafted analytical datasets.
    Proves mathematical exactness of quantile ratios, payer repeat transaction share, and payments per payer.
    """
    fixture_parquet = tmp_path / "mini_paysim.parquet"
    table_data = {
        "step": [1, 1, 2, 2, 3, 3],
        "type": ["PAYMENT", "PAYMENT", "TRANSFER", "PAYMENT", "PAYMENT", "TRANSFER"],
        "amount": [10.0, 20.0, 50.0, 70.0, 100.0, 999.0],
        "nameOrig": ["A", "A", "B", "A", "C", "F"],
        "nameDest": ["B", "C", "C", "B", "A", "G"],
        "isFraud": [0, 0, 0, 0, 0, 1],
        "oldbalanceOrg": [100.0, 90.0, 50.0, 70.0, 100.0, 999.0],
        "newbalanceOrig": [90.0, 70.0, 0.0, 0.0, 0.0, 0.0],
    }
    arrow_table = pa.Table.from_pydict(table_data)
    pq.write_table(arrow_table, fixture_parquet)

    stats = paysim_genuine_stats(fixture_parquet, DEFAULT_CONFIG)

    assert stats["genuine_count"] == 5
    assert stats["amount_median"] == 50.0
    assert abs(stats["amount_quantile_ratios"][0.5] - 1.0) < 1e-6
    assert abs(stats["amount_quantile_ratios"][0.1] - (14.0 / 50.0)) < 1e-6

    assert abs(stats["payer_repeat_share"] - 0.2) < 1e-6
    assert stats["payments_per_payer"]["median"] == 1.0
    assert abs(stats["payments_per_payer"]["p90"] - 2.6) < 1e-6
    assert abs(stats["type_shares"]["PAYMENT"] - 0.8) < 1e-6
    assert abs(stats["type_shares"]["TRANSFER"] - 0.2) < 1e-6

    mock_events = [
        PaymentEvent(
            minute=10, from_account="u1", to_account="u2", amount_paise=1000,
            from_bank_index=0, to_bank_index=0, status="success", reason="ok",
        ),
        PaymentEvent(
            minute=20, from_account="u1", to_account="u3", amount_paise=2000,
            from_bank_index=0, to_bank_index=0, status="success", reason="ok",
        ),
        PaymentEvent(
            minute=30, from_account="u2", to_account="u3", amount_paise=5000,
            from_bank_index=0, to_bank_index=0, status="success", reason="ok",
        ),
        PaymentEvent(
            minute=40, from_account="u1", to_account="u2", amount_paise=7000,
            from_bank_index=0, to_bank_index=0, status="success", reason="ok",
        ),
        PaymentEvent(
            minute=50, from_account="u3", to_account="u1", amount_paise=10000,
            from_bank_index=0, to_bank_index=0, status="success", reason="ok",
        ),
        PaymentEvent(
            minute=60, from_account="m1", to_account="m2", amount_paise=50000,
            from_bank_index=0, to_bank_index=0, status="success", reason="ok",
        ),
        PaymentEvent(
            minute=70, from_account="u1", to_account="u2", amount_paise=1000,
            from_bank_index=0, to_bank_index=0, status="insufficient_funds", reason="err",
        ),
    ]
    mock_roles = [
        RoleSpan(account_id="m1", role="mule", gang_id="gang1", depth=1, valid_from=0, valid_to=None),
        RoleSpan(account_id="m2", role="cash_out", gang_id="gang1", depth=2, valid_from=0, valid_to=None),
    ]
    mock_output = {"events": mock_events, "roles": mock_roles}

    w_stats = world_genuine_stats([mock_output], DEFAULT_CONFIG)
    assert w_stats["genuine_count"] == 5
    assert w_stats["amount_median"] == 5000.0
    assert abs(w_stats["amount_quantile_ratios"][0.5] - 1.0) < 1e-6
    assert abs(w_stats["payer_repeat_share"] - 0.2) < 1e-6
    assert w_stats["payments_per_payer"]["median"] == 1.0
    assert "not comparable" in str(w_stats["type_shares"])

    report_text = build_report(w_stats, stats, DEFAULT_CONFIG)
    expected_header = (
        "Both datasets are synthetic, so agreement with them shows plausibility, not realism. "
        "This report is recorded and never used to set or tune any world setting."
    )
    assert report_text.startswith(expected_header)
    assert "## Amount Quantiles Divided by Median" in report_text
    assert "## Differences" in report_text
    assert "IBM AML-Data: pending (not downloaded); register 12-16 records it." in report_text


def test_paysim_reader_never_reads_excluded_columns(tmp_path: Path) -> None:
    """
    Asserts that the PaySim reader explicitly excludes Kaggle balance columns from loaded dataframes.
    Passes a synthetic parquet dataset containing forbidden balance columns and confirms they are omitted.
    """
    fixture_parquet = tmp_path / "has_balances.parquet"
    data = {
        "step": [1, 2],
        "type": ["PAYMENT", "TRANSFER"],
        "amount": [100.0, 200.0],
        "nameOrig": ["orig_1", "orig_2"],
        "nameDest": ["dest_1", "dest_2"],
        "isFraud": [0, 0],
        "oldbalanceOrg": [500.0, 600.0],
        "newbalanceOrig": [400.0, 400.0],
        "oldbalanceDest": [10.0, 20.0],
        "newbalanceDest": [110.0, 220.0],
        "isFlaggedFraud": [0, 0],
    }
    table = pa.Table.from_pydict(data)
    pq.write_table(table, fixture_parquet)

    stats = paysim_genuine_stats(fixture_parquet, DEFAULT_CONFIG)
    assert stats["genuine_count"] == 2

    loaded_table = pq.read_table(fixture_parquet, columns=DEFAULT_CONFIG["paysim_columns_used"])
    loaded_columns = set(loaded_table.column_names)
    for excluded in DEFAULT_CONFIG["paysim_columns_excluded"]:
        assert excluded not in loaded_columns, f"Forbidden column '{excluded}' was loaded"


def test_never_used_to_tune_isolation() -> None:
    """
    Enforces architectural separation preventing simulation parameters from tuning against sanity reports.
    Verifies that no file under world/ imports sanity_report or mentions reports/sanity.
    """
    world_dir = REPO_ROOT / "world"
    forbidden_terms = ["sanity_report", "reports/sanity"]

    for file_path in world_dir.rglob("*"):
        if file_path.is_file() and file_path.suffix in (".py", ".yaml", ".json", ".md"):
            content = file_path.read_text(encoding="utf-8")
            for term in forbidden_terms:
                assert term not in content, f"{file_path} mentions forbidden external benchmark term '{term}'"

        if file_path.is_file() and file_path.suffix == ".py":
            source = file_path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(file_path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        assert "sanity_report" not in alias.name, f"{file_path} imports {alias.name}"
                elif isinstance(node, ast.ImportFrom):
                    mod = node.module or ""
                    assert "sanity_report" not in mod, f"{file_path} imports from {mod}"
                    assert not mod.startswith("evaluation"), f"{file_path} imports from evaluation: {mod}"


@pytest.mark.needs_dvc
def test_real_paysim_file_if_available() -> None:
    """
    Executes distribution checks on the real PaySim enriched parquet file when present locally.
    Guards with the needs_dvc pytest marker to avoid breaking environments without large ML test fixtures.
    """
    real_path = REPO_ROOT / "data" / "enriched" / "paysim_enriched.parquet"
    if not real_path.exists():
        pytest.skip("data/enriched/paysim_enriched.parquet not present on disk")

    cfg = load_sanity_report_config()
    stats = paysim_genuine_stats(real_path, cfg)

    assert stats["genuine_count"] > 1_000_000
    assert abs(stats["amount_quantile_ratios"][0.5] - 1.0) < 1e-6
    assert stats["payer_repeat_share"] >= 0.0
    assert stats["payments_per_payer"]["median"] >= 1.0
    assert len(stats["type_shares"]) > 0
