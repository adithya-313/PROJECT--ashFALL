# File: api/v1/bank_clients.py
# Purpose: Signed HTTP client implementation for commercial bank REST API integrations.
#
# Connectors and mock banks communicate over HTTP using this client. It computes canonical
# HMAC-SHA256 signatures, attaches authentication headers (API key, timestamp, nonce, signature),
# and formats responses as IdempotentOutcome objects, raising BankCallError on non-2xx responses.

import uuid
from typing import Any

import httpx

from api.v1.bank_service import IdempotentOutcome
from api.v1.signing import sign_outgoing
from core.event_clock import Clock


class BankCallError(Exception):
    """
    WHAT: Raised when a bank HTTP API request returns a non-2xx HTTP status.
    WHY: Exposes status code and machine-readable error code to calling clients,
    ensuring callers and orchestrators do not treat error responses as successful outcomes.
    """

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(f"Bank call failed with status {status} ({code}): {message}")
        self.status = status
        self.code = code
        self.message = message


class SignedHttpBankClient:
    """
    WHAT: Authenticated HTTP client that signs outgoing requests using HMAC-SHA256.
    WHY: Connectors interact with bank HTTP endpoints over wire transports (including ASGI in tests).
    This client computes transport signatures, supplies idempotency headers, and standardises outcomes.
    """

    def __init__(
        self,
        http_client: httpx.AsyncClient,
        bank_id: str,
        api_key: str,
        clock: Clock,
    ) -> None:
        """
        Initialises the signed HTTP bank client.

        Args:
            http_client: Configured httpx.AsyncClient instance (or ASGITransport client).
            bank_id: Bank identifier (e.g., 'bank_0').
            api_key: Plaintext API key corresponding to the bank's active credential.
            clock: Clock implementation providing current IST datetime for request signing.
        """
        self.http_client = http_client
        self.bank_id = bank_id
        self.api_key = api_key
        self.clock = clock

    async def post(
        self,
        path: str,
        body_bytes: bytes,
        idempotency_key: str,
    ) -> IdempotentOutcome:
        """
        Issues a signed POST request to the specified bank API endpoint.

        Args:
            path: Relative request path (e.g. '/v1/decisions', '/v1/transactions').
            body_bytes: UTF-8 encoded JSON payload bytes.
            idempotency_key: Client-provided idempotency key string.

        Returns:
            IdempotentOutcome containing status code, response body dict, and replayed flag.

        Raises:
            BankCallError: If the server returns a non-2xx response.
            ValueError: If the response cannot be parsed or lacks standard error fields.
        """
        # Generate a fresh random UUIDv4 nonce per request to guarantee freshness and prevent replay attacks.
        nonce = uuid.uuid4().hex

        # Construct canonical authentication headers using HMAC-SHA256.
        headers = sign_outgoing(
            bank_id=self.bank_id,
            method="POST",
            path=path,
            body_bytes=body_bytes,
            clock=self.clock,
            nonce=nonce,
        )

        # Attach standard identity, content-type, and idempotency headers.
        headers["X-Ashfall-Key"] = self.api_key
        headers["Content-Type"] = "application/json"
        headers["Idempotency-Key"] = idempotency_key

        response = await self.http_client.post(path, content=body_bytes, headers=headers)

        if response.status_code < 200 or response.status_code >= 300:
            try:
                data: Any = response.json()
            except Exception as exc:
                raise BankCallError(
                    status=response.status_code,
                    code="invalid_error_response",
                    message=f"Non-JSON error response from bank API: {response.text}",
                ) from exc

            if not isinstance(data, dict) or "error" not in data or not isinstance(data["error"], dict):
                raise ValueError(f"Malformed error response structure from bank API: {data!r}")

            err_dict = data["error"]
            if "code" not in err_dict or "message" not in err_dict:
                raise ValueError(f"Missing 'code' or 'message' in bank API error payload: {err_dict!r}")

            raise BankCallError(
                status=response.status_code,
                code=str(err_dict["code"]),
                message=str(err_dict["message"]),
            )

        # Parse success response.
        replayed = response.headers.get("Idempotent-Replayed", "").lower() == "true"
        body = response.json()
        return IdempotentOutcome(
            status=response.status_code,
            body=body,
            replayed=replayed,
        )

