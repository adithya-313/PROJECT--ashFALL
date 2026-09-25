"""
File: tests/unit/test_config_loads.py
Purpose: Unit tests for application configuration loading and validation.
Connects: Validates core/config.py AppConfig behaviour against valid and invalid environments.
"""

import pytest
from pydantic import ValidationError

from core.config import AppConfig


def test_config_loads_all_required_keys(monkeypatch: pytest.MonkeyPatch, tmp_path: pytest.TempPathFactory) -> None:
    """
    Asserts that AppConfig correctly loads all four required fields from an environment file:
    environment, log_level, database_url, and redis_url.
    """
    env_content = (
        "ENVIRONMENT=testing\n"
        "LOG_LEVEL=DEBUG\n"
        "DATABASE_URL=postgresql://postgres:secret@db:5432/testdb\n"
        "REDIS_URL=redis://redis:6379/1\n"
    )
    test_env_file = tmp_path / ".test.env"
    test_env_file.write_text(env_content, encoding="utf-8")

    # Prevent docker-compose environment vars from overriding the test file
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("REDIS_URL", raising=False)

    config = AppConfig(_env_file=str(test_env_file))

    # Assert presence of all four required keys
    assert hasattr(config, "environment")
    assert hasattr(config, "log_level")
    assert hasattr(config, "database_url")
    assert hasattr(config, "redis_url")

    # Assert values match test environment
    assert config.environment == "testing"
    assert config.log_level == "DEBUG"
    assert config.database_url == "postgresql://postgres:secret@db:5432/testdb"
    assert config.redis_url == "redis://redis:6379/1"


def test_config_missing_database_url_raises_validation_error(
    tmp_path: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Asserts that AppConfig raises a Pydantic ValidationError when DATABASE_URL is missing.
    """
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("REDIS_URL", raising=False)

    env_content = (
        "ENVIRONMENT=testing\n"
        "LOG_LEVEL=INFO\n"
        "REDIS_URL=redis://redis:6379/0\n"
    )
    test_env_file = tmp_path / ".missing_db.env"
    test_env_file.write_text(env_content, encoding="utf-8")

    with pytest.raises(ValidationError) as exc_info:
        AppConfig(_env_file=str(test_env_file))

    assert "database_url" in str(exc_info.value)


def test_config_missing_redis_url_raises_validation_error(
    tmp_path: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Asserts that AppConfig raises a Pydantic ValidationError when REDIS_URL is missing.
    """
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("REDIS_URL", raising=False)

    env_content = (
        "ENVIRONMENT=testing\n"
        "LOG_LEVEL=INFO\n"
        "DATABASE_URL=postgresql://postgres:secret@db:5432/testdb\n"
    )
    test_env_file = tmp_path / ".missing_redis.env"
    test_env_file.write_text(env_content, encoding="utf-8")

    with pytest.raises(ValidationError) as exc_info:
        AppConfig(_env_file=str(test_env_file))

    assert "redis_url" in str(exc_info.value)
