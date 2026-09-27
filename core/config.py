"""
File: core/config.py
Purpose: Defines the application configuration schema and loads variables from the environment.
Connects: Used throughout the application (api/main.py, core/logging.py) whenever a
          configuration value is needed. Relies on pydantic-settings to validate types
          and require mandatory environment variables.

Required Fields:
  - database_url: Connection string for PostgreSQL (no default; fails loudly if missing).
  - redis_url: Connection string for Redis (no default; fails loudly if missing).

Note on WORKER_COUNT:
  The worker count is read directly in infra/docker/gunicorn_conf.py via os.environ.get()
  rather than through this class. This is because gunicorn reads its config file
  before the application is imported, so pydantic-settings has not run yet.
  WORKER_COUNT is documented in .env.example for visibility, but it does not
  appear here to avoid creating two sources of truth for the same variable.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseSettings):
    """
    Defines the typed configuration schema for the application.

    pydantic-settings reads each attribute name as an environment variable name
    (case-insensitive). If the variable is present in the environment or in the
    .env file, its value overrides the default. If a required field lacks a default
    and is not present in the environment, pydantic-settings raises ValidationError.

    Attributes:
        environment: The deployment context — affects log labelling and can be
                     used to gate behaviour (e.g., more verbose errors in development).
        log_level: Controls which log messages are emitted. Only messages at this
                   severity or higher will appear in the log output.
        database_url: Mandatory PostgreSQL connection URI (no default).
        redis_url: Mandatory Redis connection URI (no default).
    """
    environment: str = "development"
    log_level: str = "INFO"
    database_url: str
    redis_url: str
    db_pool_min_size: int = 2
    db_pool_max_size: int = 10
    # OTLP endpoint for sending traces. Points to Jaeger in Docker; override for local dev.
    otlp_endpoint: str = "http://jaeger:4318/v1/traces"

    # SettingsConfigDict tells pydantic-settings where to look for values.
    # env_file=".env" reads variables from a .env file if it exists.
    # extra="ignore" means unknown variables in .env do not cause validation errors.
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


def get_config() -> AppConfig:
    """
    Instantiates and validates the application configuration from the environment.

    Returns:
        AppConfig: A fully-validated configuration object. pydantic-settings will
                   raise a ValidationError at startup if a required variable is
                   missing or has the wrong type — failing fast is intentional.

    Why a factory function:
        This factory is the single place where AppConfig is constructed. If we
        later add caching (via functools.lru_cache) or dependency injection
        (via FastAPI's Depends), we change only this function — all consumers
        remain unchanged.
    """
    return AppConfig()  # type: ignore[call-arg]  # pydantic-settings reads fields from env


# Module-level singleton: imported by name in other modules.
# Constructed once at import time, which happens once per gunicorn worker process.
config = get_config()
