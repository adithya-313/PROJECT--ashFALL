# File: tests/unit/test_level_07_module.py
# Purpose: Proves the Level 7 showcase module has the contract the framework needs, lists its checks in the right
#          order, and reports missing services. No database or Kafka is needed.
import asyncio

from showcase import level_07
from showcase.framework import validate_checks
from showcase.level07_helpers import build_outage_chart, choose_kill_points, compress_timeline

EXPECTED_NAMES = [
    "100 transactions produce 100 ordered events",
    "Broker stopped mid-run",
    "Duplicate delivery counted once",
    "Forced database error leaves nothing",
    "Poison message is parked and the rest still delivered",
    "Same input gives same event IDs",
    "Repeated broker kills lose nothing",
]
EXPECTED_KINDS = ["normal", "edge", "edge", "negative", "edge", "consistency", "edge"]


def test_module_has_the_contract_attributes():
    """The module defines every attribute the framework reads."""
    assert level_07.LEVEL_NUMBER == 7
    assert level_07.LEVEL_TITLE
    assert level_07.WHAT_YOU_SAW
    assert level_07.NEEDS_DATABASE is True
    assert callable(level_07.find_missing_preconditions)
    assert callable(level_07.get_checks)


def test_there_are_seven_checks_in_both_modes():
    """Quick and full mode both list seven checks."""
    assert len(level_07.get_checks(False)) == len(level_07.get_checks(True)) == 7


def test_checks_have_the_exact_names_in_order():
    """The checks appear in the documented order, in quick and full mode."""
    assert [check.name for check in level_07.get_checks(False)] == EXPECTED_NAMES
    assert [check.name for check in level_07.get_checks(True)] == EXPECTED_NAMES


def test_missing_services_are_reported_with_a_fix(monkeypatch):
    """When Kafka (or Postgres) is unreachable, the problems list is not empty and says how to fix it."""
    async def database_fine(url: str, fix_hint: str) -> None:
        return None

    def kafka_down(host: str, port: int, what: str, fix_hint: str) -> str:
        return f"{what} not reachable - run: {fix_hint}"

    monkeypatch.setattr(level_07, "check_database", database_fine)
    monkeypatch.setattr(level_07, "check_tcp_port", kafka_down)
    problems = asyncio.run(level_07.find_missing_preconditions())
    assert len(problems) == 1
    assert "docker compose up -d redpanda" in problems[0]


def test_checks_meet_the_framework_coverage_rules_and_kinds():
    """Both modes pass the framework's coverage rules and list the kinds in the documented order."""
    for quick in (False, True):
        checks = level_07.get_checks(quick)
        validate_checks(checks)
        assert [check.kind for check in checks] == EXPECTED_KINDS


def test_choose_kill_points_spreads_kills_across_the_writes():
    """One kill lands about a third in; five kills are spread evenly and strictly increasing."""
    assert choose_kill_points(100, 1) == [33]
    five = choose_kill_points(100, 5)
    assert five == [14, 29, 43, 57, 71]
    assert five == sorted(set(five))


def test_compress_timeline_keeps_changes_first_and_last():
    """Repeated delivered counts are dropped, while the first point, every change and the last point stay."""
    points = [(0.0, 0), (0.2, 0), (0.4, 5), (0.6, 5), (0.8, 5), (1.0, 9), (1.2, 9)]
    assert compress_timeline(points) == [(0.0, 0), (0.4, 5), (1.0, 9), (1.2, 9)]


def test_outage_chart_has_the_timeline_format():
    """The chart dict has the keys the report writer expects, including the 'broker down' band."""
    chart = build_outage_chart([(0.0, 0), (1.0, 3)], [(0.2, 0.8, "broker down")])
    assert chart["type"] == "timeline"
    assert chart["shaded"] == [(0.2, 0.8, "broker down")]
    assert chart["points"] == [(0.0, 0), (1.0, 3)]
    assert chart["x_label"] and chart["y_label"] and chart["title"]
