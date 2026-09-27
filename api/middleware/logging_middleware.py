"""
File: api/middleware/logging_middleware.py
Purpose: Logs one structured JSON line for every HTTP request, after the response
         is returned so that status_code is known.
Connects: Registered in api/main.py after RequestIDMiddleware.
          Uses get_logger from core/logging.py — the same structlog setup used everywhere.
          The request_id is already in the contextvar by the time this middleware runs,
          so structlog's inject_request_id processor attaches it automatically.
"""

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from core.logging import get_logger

logger = get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware that writes one log line per request with path and status_code.

    It calls the next middleware/route handler first (await call_next),
    then logs once the response object is available. This is the only safe
    way to capture status_code — it isn't known until the handler finishes.

    timestamp and request_id are injected automatically by structlog processors
    defined in core/logging.py, so we don't need to add them here manually.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Run the actual route handler and wait for the response.
        response = await call_next(request)

        # Log after the response so we have the real status code.
        logger.info(
            "request handled",
            path=request.url.path,
            status_code=response.status_code,
        )

        return response
