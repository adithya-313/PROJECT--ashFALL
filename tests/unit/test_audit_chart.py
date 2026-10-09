"""
File: tests/unit/test_audit_chart.py
Purpose: Proves the Level 9 chart function produces a real, non-empty PNG from a small made-up results set.
"""

import pytest

pytest.importorskip("matplotlib", reason="matplotlib is only installed with requirements-ml.txt")

from scripts.audit_chart import make_chart_png  # noqa: E402

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def fake_results() -> dict:
    """A tiny results set with one row per group."""
    def row(name: str, group: str, score: float) -> dict:
        return {"name": name, "group": group, "pr_auc": score}

    return {"base_rate": 0.005, "rows": [
        row("original_v1", "original", 0.99), row("control_all15", "control", 0.98),
        row("ablate_no_balance", "ablation", 0.80), row("oot_all15", "out_of_time", 0.97),
        row("single_amount", "single_feature", 0.01),
    ]}


def test_chart_is_a_non_empty_png() -> None:
    """The returned bytes start with the PNG signature and are not tiny."""
    png = make_chart_png(fake_results())
    assert png.startswith(PNG_SIGNATURE)
    assert len(png) > 5000
