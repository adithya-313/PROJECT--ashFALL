# File: api/v1/errors.py
# Purpose: One error envelope for every /v1 bank endpoint, and the handlers that turn exceptions into it.
#
# Every error a bank can see has the same shape: {"error": {"code": "...", "message": "..."}}. The code is
# stable and meant for programs; the message is for people and never contains secrets, signatures or raw input.
from fastapi import Request
from fastapi.responses import ORJSONResponse

from api.v1.request_auth import BankAuthError


class ApiError(Exception):
    """
    An expected failure of a bank request that must become an HTTP error response.

    Attributes:
        status: HTTP status code to return (for example 404, 409 or 422).
        code: Short machine-readable code such as "invalid_body" or "account_not_found".
        message: Safe human-readable sentence; it must never echo secrets or raw request values.
    """

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


def error_response(status: int, code: str, message: str) -> ORJSONResponse:
    """
    Builds the standard error response.

    Args:
        status: HTTP status code.
        code: Machine-readable error code.
        message: Safe human-readable message.

    Returns:
        An ORJSONResponse with the body {"error": {"code": code, "message": message}}.
    """
    return ORJSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


async def handle_api_error(request: Request, error: ApiError) -> ORJSONResponse:
    """Turns an ApiError raised by a route into the standard error response."""
    return error_response(error.status, error.code, error.message)


async def handle_bank_auth_error(request: Request, error: BankAuthError) -> ORJSONResponse:
    """Turns a BankAuthError raised during request authentication into the standard error response."""
    return error_response(error.http_status, error.code, error.message)
