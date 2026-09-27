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

import asyncio
from contextlib import asynccontextmanager

import asyncpg

import fastapi
import uvloop
from fastapi.responses import ORJSONResponse

from api.middleware.request_id import RequestIDMiddleware
from api.middleware.logging_middleware import RequestLoggingMiddleware
from core.config import config
from core.logging import configure_logging, get_logger

# Initialise logging before the application begins serving requests.
# Module-level execution ensures logging is ready before any worker startup handlers execute.
configure_logging(config.log_level)

logger = get_logger(__name__)

# Explicitly set uvloop as the asyncio event loop policy.
# While UvicornWorker implicitly uses uvloop, explicitly setting it removes any
# ambiguity about which event loop is active.
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    """
    Lifespan context manager for the FastAPI application.
    Runs startup logic before the application begins accepting requests,
    and shutdown logic after it stops.
    """
    logger.info(
        "Worker started",
        environment=config.environment,
        log_level=config.log_level,
    )

    # Create the asyncpg connection pool
    pool = await asyncpg.create_pool(
        dsn=config.database_url,
        min_size=config.db_pool_min_size,
        max_size=config.db_pool_max_size,
    )
    # Store pool on app.state for dependency injection
    app.state.db_pool = pool
    logger.info("Database pool initialized", min_size=config.db_pool_min_size, max_size=config.db_pool_max_size)

    yield

    # Cleanly close the pool on shutdown
    await pool.close()
    logger.info("Worker stopped")


# The FastAPI application instance.
# default_response_class=ORJSONResponse uses orjson for JSON serialisation across all routes.
app = fastapi.FastAPI(
    title="PROJECT-ashFALL",
    default_response_class=ORJSONResponse,
    lifespan=lifespan,
)

# Register request_id propagation middleware.
# Must be added first so request_id is in the contextvar before logging middleware runs.
app.add_middleware(RequestIDMiddleware)

# Register per-request access logging (path + status_code on every request).
# Added after RequestIDMiddleware so the request_id contextvar is already set when
# RequestLoggingMiddleware calls the logger.
app.add_middleware(RequestLoggingMiddleware)


@app.get("/health")
async def health_check() -> dict:
    """
    Returns a minimal JSON payload confirming the service is alive.

    Exercises the full HTTP stack (HTTP parser -> router -> response serialiser).
    Logs one structured JSON line per request with request_id attached.
    """
    logger.info("health check requested", endpoint="/health")
    return {"status": "ok", "environment": config.environment}
