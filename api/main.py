"""
File: api/main.py
Purpose: The application entrypoint. Defines the FastAPI application instance and its core routes.
Connects: Gunicorn (infra/docker/gunicorn_conf.py) boots workers pointing at this module's `app` object.
          The logging module (core/logging.py) is initialised here at startup.
          The config module (core/config.py) provides runtime configuration.

Preserved from Level 1:
  default_response_class=ORJSONResponse is set on the FastAPI application.
  Every route returning a dictionary (like /health) is serialised using orjson
  for high-throughput, low-latency JSON handling.

Current endpoints:
  GET /health — returns a minimal JSON payload confirming the service is alive.
"""

import fastapi
from fastapi.responses import ORJSONResponse

from core.config import config
from core.logging import configure_logging, get_logger

# Initialise logging before the application begins serving requests.
# Module-level execution ensures logging is ready before any worker startup handlers execute.
configure_logging(config.log_level)

logger = get_logger(__name__)

# The FastAPI application instance.
# default_response_class=ORJSONResponse uses orjson for JSON serialisation across all routes.
app = fastapi.FastAPI(
    title="PROJECT-ashFALL",
    default_response_class=ORJSONResponse,
)


@app.on_event("startup")
async def on_startup() -> None:
    """
    Runs once when each ASGI worker starts, before any requests are handled.

    Logs a startup line per worker process so that the active worker count
    and startup configuration are visible in the structured log stream.
    """
    logger.info(
        "Worker started",
        environment=config.environment,
        log_level=config.log_level,
    )


@app.get("/health")
async def health_check() -> dict:
    """
    Returns a minimal JSON payload confirming the service is alive.

    Exercises the full HTTP stack (HTTP parser -> router -> response serialiser).
    Logs one structured JSON line per request.
    """
    logger.info("health check requested", endpoint="/health")
    return {"status": "ok", "environment": config.environment}
