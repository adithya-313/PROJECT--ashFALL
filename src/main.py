"""
File: src/main.py
Purpose: The application entrypoint. Defines the FastAPI application instance and its routes.
Connects: The Dockerfile CMD boots uvicorn/gunicorn pointing at this module's `app` object.
          The logging module (src/core/logging.py) is initialised here at startup.
          The config module (src/core/config.py) provides the LOG_LEVEL used at startup.

Why this file exists at this level:
A FastAPI 'app' object is the object that an ASGI server (uvicorn, gunicorn) serves.
It must be importable as a module-level attribute — which is why `app` is defined at
module scope rather than inside a function.

Current endpoints:
  GET /health — returns a small JSON payload confirming the service is alive.
                Used as the benchmark target in Levels 1a and 1b.
"""

import time
import fastapi

from src.core.config import config
from src.core.logging import configure_logging, get_logger

# Initialise logging at module import time, before the application starts serving.
# We use the log level from the environment/config so that it is controllable
# without rebuilding the image.
configure_logging(config.log_level)

# Retrieve a logger bound to this module's name for per-module log attribution.
logger = get_logger(__name__)

# The FastAPI application instance. The ASGI server targets this object.
# 'title' appears in the auto-generated OpenAPI docs at /docs.
app = fastapi.FastAPI(title="PROJECT-ashFALL")


@app.on_event("startup")
async def on_startup() -> None:
    """
    Runs once when the ASGI server starts, before any requests are handled.

    Logs a structured startup line so that the timestamp of service readiness
    is recorded in the log stream — useful when diagnosing slow cold-starts later.
    """
    logger.info(
        "Application started",
        environment=config.environment,
        log_level=config.log_level,
    )


@app.get("/health")
async def health_check() -> dict:
    """
    Returns a minimal JSON payload confirming the service is alive.

    This endpoint exists for two reasons:
    1. It serves as the benchmark target for Level 1 — a realistic but minimal
       response that exercises the full HTTP stack without business logic noise.
    2. It acts as the container liveness probe endpoint in future orchestration.

    Returns:
        A dictionary with:
          - status: the string "ok" — indicates the service reached this handler.
          - environment: the deployment environment name from config, confirming
                         that environment variable loading is working end-to-end.

    Logs one structured JSON line per request, which is the pattern all future
    endpoints will follow.
    """
    # Record request arrival in the structured log stream.
    # In Level 2 this will be moved into middleware so every route gets it
    # automatically with a request_id — for now it is explicit here.
    logger.info("health check requested", endpoint="/health")

    return {"status": "ok", "environment": config.environment}
