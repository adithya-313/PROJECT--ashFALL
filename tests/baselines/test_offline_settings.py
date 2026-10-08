# File: tests/baselines/test_offline_settings.py
# Purpose: Guards the settings that stop Grafana from calling the internet. Level 8 found (with the stack on a
#          network that has no outside access) that Grafana looked up grafana.com at start-up; these settings turned
#          that off. If one of them is removed from docker-compose.yml by accident, this test fails with an
#          explanation.
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml", reason="PyYAML is only installed in the pytest-fast job")

COMPOSE_FILE = Path(__file__).resolve().parent.parent.parent / "docker-compose.yml"

# Each setting, and what it switches off.
GRAFANA_OFFLINE_SETTINGS = {
    "GF_ANALYTICS_REPORTING_ENABLED": "false",  # usage counters sent to stats.grafana.org
    "GF_ANALYTICS_CHECK_FOR_UPDATES": "false",  # asks grafana.com for new Grafana versions
    "GF_ANALYTICS_CHECK_FOR_PLUGIN_UPDATES": "false",  # asks grafana.com for new plugin versions
    "GF_PLUGINS_PUBLIC_KEY_RETRIEVAL_DISABLED": "true",  # downloads the plugin signature key from grafana.com
}


def grafana_environment() -> dict[str, str]:
    """The grafana service's environment entries from docker-compose.yml, as a dictionary."""
    compose = yaml.safe_load(COMPOSE_FILE.read_text(encoding="utf-8"))
    entries = compose["services"]["grafana"]["environment"]
    pairs = [entry.split("=", 1) for entry in entries]
    return {name: value for name, value in pairs}


@pytest.mark.parametrize("name", sorted(GRAFANA_OFFLINE_SETTINGS))
def test_grafana_phone_home_setting_is_still_off(name: str) -> None:
    """Each setting that stops Grafana from calling the internet is still present with its safe value."""
    environment = grafana_environment()
    expected = GRAFANA_OFFLINE_SETTINGS[name]
    assert environment.get(name) == expected, (
        f"docker-compose.yml no longer sets {name}={expected} for grafana. That setting stops Grafana from calling the "
        "internet, and the Level 8 no-egress check depends on it. If you removed it on purpose, update this test "
        "and explain why in your pull request."
    )
