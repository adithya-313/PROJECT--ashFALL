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

from pydantic import Field, SecretStr
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
    # OTLP endpoint for sending traces to Tempo. Override in .env for local dev outside Docker.
    otlp_endpoint: str = "http://tempo:4318/v1/traces"
    # Address of the Kafka-compatible broker (Redpanda). The default works for scripts run on the host machine, where
    # docker-compose publishes the broker on localhost:19092. Containers on the compose network should set
    # KAFKA_BOOTSTRAP_SERVERS=redpanda:9092 instead.
    kafka_bootstrap_servers: str = "localhost:19092"
    # The Kafka topic the outbox relay publishes transaction events to.
    kafka_outbox_topic: str = "transactions.created"
    # The Kafka topic the outbox relay publishes schema-2 UPI payment events to.
    kafka_upi_topic: str = "upi.payments"
    # Outbox relay tuning: how long to wait between checks when there is nothing to send, how many events to send
    # per round, and the longest wait between retries after a Kafka failure (the wait doubles up to this cap).
    relay_poll_interval_seconds: float = 1.0
    relay_batch_size: int = 100
    relay_max_backoff_seconds: float = 30.0
    # How many times a message that Kafka rejects for its own sake (for example because it is too large) is tried
    # before the relay parks it in the outbox_dead_letter table. Connection problems never count. Minimum 1.
    relay_max_attempts: int = Field(default=5, ge=1)
    # The port on which the relay serves its metrics (for Prometheus) inside the compose network.
    relay_metrics_port: int = 9108

    # Feature switches for later levels. They are only DECLARED here: nothing reads them yet. False means today's
    # behaviour, so leaving them unset changes nothing. Each is set by the environment variable with the same name in
    # upper case (for example ENABLE_MODEL_V2=true).
    enable_bank_integration_api: bool = False
    enable_model_v2: bool = False
    enable_gnn_ensemble: bool = False
    enable_research_agent_internet: bool = False

    # Bank HMAC request signing secrets (Level 14b). Stored securely as SecretStr
    # so they are never printed, logged, or serialized to JSON in plain text.
    bank_signing_secret_bank_0: SecretStr | None = None
    bank_signing_secret_bank_1: SecretStr | None = None
    bank_signing_secret_bank_2: SecretStr | None = None

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
