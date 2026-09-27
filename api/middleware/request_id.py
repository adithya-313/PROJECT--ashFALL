"""
File: api/middleware/request_id.py
Purpose: ASGI middleware to capture or generate a request_id for every incoming HTTP request,
         store it in a contextvar for structlog propagation, and attach it to response headers.
Connects: Registered in api/main.py; sets core.logging.request_id_ctx_var.
"""

import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from core.logging import request_id_ctx_var


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Middleware that ensures every HTTP request has an associated request_id.
    
    If the caller provides an 'X-Request-ID' header, that value is preserved.
    Otherwise, a new UUID4 string is generated.
    The ID is set in the contextvar for the duration of the request call stack,
    and echoed back to the client in the 'X-Request-ID' response header.
    """
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        token = request_id_ctx_var.set(request_id)
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            request_id_ctx_var.reset(token)
