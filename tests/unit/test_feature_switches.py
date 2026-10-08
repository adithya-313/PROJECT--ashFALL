"""
File: tests/unit/test_feature_switches.py
Purpose: Proves the feature switches in core/config.py are all OFF by default, can each be turned on from the
         environment, reject values that are not true/false, and that a new "enable_" switch cannot be added without
         being listed here (and so without a default-is-off test).
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from core.config import AppConfig

# Every feature switch. Add a new switch here when you add it to AppConfig: the last test fails otherwise.
SWITCH_NAMES = [
    "enable_bank_integration_api",
    "enable_model_v2",
    "enable_gnn_ensemble",
    "enable_research_agent_internet",
]
REPO_ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove any switch from the environment, and set the two required settings, so each test starts the same."""
    for name in SWITCH_NAMES:
        monkeypatch.delenv(name.upper(), raising=False)
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@localhost:5432/testdb")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")


def make_config() -> AppConfig:
    """Build the config from the environment only (no .env file)."""
    return AppConfig(_env_file=None)  # type: ignore[call-arg]


def test_every_switch_is_off_by_default() -> None:
    """With nothing set, every feature switch is False."""
    config = make_config()
    for name in SWITCH_NAMES:
        assert getattr(config, name) is False, f"{name} must default to False"


@pytest.mark.parametrize("name", SWITCH_NAMES)
def test_each_switch_can_be_turned_on_from_the_environment(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Setting NAME=true turns that one switch on and leaves the others off."""
    monkeypatch.setenv(name.upper(), "true")
    config = make_config()
    assert getattr(config, name) is True
    others = [other for other in SWITCH_NAMES if other != name]
    assert all(getattr(config, other) is False for other in others)


def test_a_value_that_is_not_true_or_false_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """pydantic refuses a switch value like 'maybe' and names the field in its error."""
    monkeypatch.setenv("ENABLE_MODEL_V2", "maybe")
    with pytest.raises(ValidationError) as error_info:
        make_config()
    assert "enable_model_v2" in str(error_info.value)


def test_the_example_env_file_still_loads_with_all_switches_off() -> None:
    """The committed .env.example loads, and every switch in it is off."""
    config = AppConfig(_env_file=REPO_ROOT / ".env.example")  # type: ignore[call-arg]
    assert all(getattr(config, name) is False for name in SWITCH_NAMES)


def test_the_switch_list_matches_the_config_fields() -> None:
    """The names listed above are exactly the config fields that start with 'enable_'."""
    fields_in_config = {name for name in AppConfig.model_fields if name.startswith("enable_")}
    assert fields_in_config == set(SWITCH_NAMES)
