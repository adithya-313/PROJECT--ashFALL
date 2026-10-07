# File: tests/unit/test_level_07_module.py
# Purpose: Proves the Level 7 showcase module has the contract the framework needs, lists its checks in the right
#          order, and reports missing services. No database or Kafka is needed.
import asyncio

from showcase import level_07

EXPECTED_NAMES = [
    "100 transactions produce 100 ordered events",
    "Duplicate delivery counted once",
    "Forced database error leaves nothing",
    "Same input gives same event IDs",
]


def test_module_has_the_contract_attributes():
    """The module defines every attribute the framework reads."""
    assert level_07.LEVEL_NUMBER == 7
    assert level_07.LEVEL_TITLE
    assert level_07.WHAT_YOU_SAW
    assert level_07.NEEDS_DATABASE is True
    assert callable(level_07.find_missing_preconditions)
    assert callable(level_07.get_checks)


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
