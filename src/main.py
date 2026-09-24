"""
File: src/main.py
Purpose: The application entrypoint. Defines the FastAPI application instance and its routes.
Connects: Gunicorn (src/gunicorn_conf.py) boots workers pointing at this module's `app` object.
          The logging module (src/core/logging.py) is initialised here at startup.
          The config module (src/core/config.py) provides LOG_LEVEL and ENVIRONMENT.

Phase 1b change from Phase 1a:
  default_response_class=ORJSONResponse is set on the FastAPI application.
  This means every route that returns a dict (like /health) will be serialised
  using orjson (a Rust-based JSON encoder) instead of Python's stdlib json.
  orjson is faster for small payloads and handles more types natively (e.g. UUIDs,
  datetime objects) without custom encoders.

Current endpoints:
  GET /health — returns a small JSON payload confirming the service is alive.
                Benchmark target for Levels 1a and 1b.
"""

import fastapi
from fastapi.responses import ORJSONResponse

from src.core.config import config
from src.core.logging import configure_logging, get_logger

# Initialise logging before the application begins serving requests.
# This must happen at module level so that logging is ready before any
# startup event handlers or route handlers execute.
configure_logging(config.log_level)

logger = get_logger(__name__)

# The FastAPI application instance.
# default_response_class=ORJSONResponse replaces the default JSONResponse
# (which uses stdlib json) with ORJSONResponse (which uses orjson).
# This applies to every route that returns a plain dict — routes can still
# override it per-endpoint by specifying response_class explicitly.
app = fastapi.FastAPI(
    title="PROJECT-ashFALL",
    default_response_class=ORJSONResponse,
)


@app.on_event("startup")
async def on_startup() -> None:
    """
    Runs once when each ASGI worker starts, before any requests are handled.

    Logs a startup line per worker process so that the number of active workers
    and their startup timestamps are visible in the log stream.

    Note: with gunicorn spawning multiple workers, this will fire once per worker,
    not once per gunicorn invocation. This is intentional and expected behaviour.
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

    This is the primary benchmark target: a realistic but minimal endpoint that
    exercises the full HTTP stack (HTTP parser → router → response serialiser)
    without any business logic that would mask the infrastructure overhead.

    Returns:
        A dictionary serialised via orjson (Phase 1b) or stdlib json (Phase 1a):
          - status: "ok" — confirms this handler was reached successfully.
          - environment: the deployment environment name from the config,
                         confirming that environment variable loading works.

    Logs one structured JSON line per request. In Level 2 this moves to middleware
    with a per-request request_id — for now it is explicit in the handler.
    """
    logger.info("health check requested", endpoint="/health")
    return {"status": "ok", "environment": config.environment}
