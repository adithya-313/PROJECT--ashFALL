# File: api/v1/bank_service.py
# Purpose: Decoupled business logic and in-process bank client execution for Level 14c.
#
# Enables invoking bank API domain operations directly without HTTP transport or signing overhead,
# while preserving exact database transaction semantics, schema validation, and idempotency guarantees.

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, TypeVar

import asyncpg
import orjson
from pydantic import BaseModel, ValidationError

from api.v1.bank_scope import bank_scoped_transaction
from api.v1.errors import ApiError
from api.v1.idempotency import (
    IdempotencyStateError,
    Outcome,
    claim_idempotency_key,
    hash_request,
    store_idempotent_response,
)
from api.v1.models import (
    POLICY,
    AccountEventRequest,
    AccountEventResponse,
    BankMode,
    ComplaintRequest,
    ComplaintResponse,
    DecisionRequest,
    DecisionResponse,
    TransactionResponse,
    TransactionSideRequest,
    WithdrawalRequest,
    WithdrawalResponse,
)
from api.v1.upi_outbox import UpiPaymentSide, record_upi_side, validate_side
from core.config import config
from core.event_clock import Clock
from features.account_keys import AccountFeaturesSettings
from features.decision_features import compute_decision_row
from features.feature_log import FeatureLogRow, insert_feature_log_row
from features.registry_screening import RegistryScreener

logger = logging.getLogger(__name__)

ModelT = TypeVar("ModelT", bound=BaseModel)

DECISION_UNIQUE_CONSTRAINT: str = "uq_bank_decision_requests_bank_payment"
COMPLAINT_UNIQUE_CONSTRAINT: str = "uq_bank_complaints_bank_complaint"
DECISION_REASON: str = "scoring is not enabled until Level 19"


def get_bank_mode(bank_id: str) -> BankMode:
    """Returns the configured operating mode (shadow or live) for the specified bank identifier."""
    if bank_id not in POLICY.bank_modes:
        raise ValueError(f"No bank mode configured for bank {bank_id!r}")
    return POLICY.bank_modes[bank_id]


@dataclass(frozen=True)
class IdempotentOutcome:
    """
    WHAT: Represents the final result of an idempotently executed or replayed bank operation.
    It encapsulates the numeric HTTP-equivalent status code, the response body dictionary,
    and a boolean indicating whether the response was replayed from a previously cached execution.

    WHY: Decouples idempotency result representation from FastAPI ORJSONResponse objects,
    allowing service-layer callers and in-process clients to consume the response natively.
    """

    status: int
    body: dict[str, Any]
    replayed: bool


async def run_idempotent_operation(
    pool: asyncpg.Pool,
    bank_id: str,
    idempotency_key: str,
    request_hash: str,
    now: datetime,
    handler: Callable[[asyncpg.Connection], Awaitable[tuple[int, dict[str, Any]]]],
) -> IdempotentOutcome:
    """
    WHAT: Coordinates atomic execution of an idempotent write operation within a bank-scoped transaction.
    It acquires a database connection, begins a bank-scoped transaction, and attempts to claim the
    provided idempotency key. If a conflict is detected, it raises an ApiError(409); if the key was
    previously completed, it replays the stored status and body; otherwise, it executes the business handler,
    caches the returned status and body, and commits atomically.

    WHY: Establishes a shared, transport-independent idempotency boundary for both HTTP endpoint requests
    and InProcessBankClient calls, guaranteeing identical deduplication and conflict behavior across all callers.

    Args:
        pool: Connection pool for acquiring tenant database connections.
        bank_id: Commercial bank identifier scoping the transaction.
        idempotency_key: Client-supplied idempotency key string.
        request_hash: SHA-256 digest of the request method, path, and body bytes.
        now: Current timezone-aware transport timestamp.
        handler: Async callable executing the business write on the scoped connection.

    Returns:
        An IdempotentOutcome containing status code, response body dictionary, and replay boolean.

    Raises:
        ApiError: 409 'idempotency_key_reused' if the key was previously claimed with a different request hash.
        IdempotencyStateError: If a replayed record lacks stored response status or body.
    """
    async with pool.acquire() as connection:
        async with bank_scoped_transaction(connection, bank_id) as scoped:
            claim = await claim_idempotency_key(scoped, bank_id, idempotency_key, request_hash, now)
            if claim.outcome == Outcome.CONFLICT:
                raise ApiError(409, "idempotency_key_reused", "This Idempotency-Key was used with a different request")
            if claim.outcome == Outcome.REPLAY:
                if claim.stored_status is None or claim.stored_body is None:
                    raise IdempotencyStateError("A replayed idempotency key has no stored response")
                return IdempotentOutcome(
                    status=claim.stored_status,
                    body=claim.stored_body,
                    replayed=True,
                )
            status, body = await handler(scoped)
            await store_idempotent_response(scoped, bank_id, idempotency_key, status, body)
            return IdempotentOutcome(status=status, body=body, replayed=False)


async def record_decision(
    connection: asyncpg.Connection,
    bank_id: str,
    request: DecisionRequest,
    *,
    redis_client: Any = None,
    decision_clock: Any = None,
    settings: AccountFeaturesSettings | None = None,
    screener: RegistryScreener | None = None,
    feature_row: FeatureLogRow | None = None,
) -> tuple[int, dict[str, Any]]:
    """
    WHAT: Executes the business logic for recording a bank payment decision evaluation request.
    It inserts a record into bank_decision_requests with the bank's configured operating mode
    and returns an unscored evaluation decision response. When decision_feature_logging_enabled
    is configured, also records point-in-time canonical features in bank_feature_log atomically.

    WHY: Moves the decision write logic from the HTTP router into a reusable service function
    callable both by the HTTP route and in-process simulation connectors.

    Args:
        connection: Active bank-scoped database connection.
        bank_id: Commercial bank identifier.
        request: Validated DecisionRequest model.
        redis_client: Optional Redis client for watermark and account feature lookups.
        decision_clock: Optional clock instance for evaluation timestamp.
        settings: Optional AccountFeaturesSettings.
        screener: Optional RegistryScreener instance.
        feature_row: Pre-computed FeatureLogRow to insert (if computed prior to transaction).

    Returns:
        Tuple of status code (200) and response body dictionary.

    Raises:
        ApiError: 409 'payment_already_decided' if a decision was already requested for this payment_id.
    """
    request_json = orjson.dumps(request.model_dump(mode="json")).decode("utf-8")
    mode = get_bank_mode(bank_id)
    try:
        await connection.execute(
            """
            INSERT INTO bank_decision_requests (bank_id, payment_id, request, decision, mode)
            VALUES ($1, $2, $3::jsonb, 'unscored', $4)
            """,
            bank_id,
            request.payment_id,
            request_json,
            mode,
        )
    except asyncpg.UniqueViolationError as error:
        if error.constraint_name != DECISION_UNIQUE_CONSTRAINT:
            raise
        raise ApiError(409, "payment_already_decided", "A decision was already requested for this payment") from error

    if config.decision_feature_logging_enabled:
        row = feature_row
        if row is None:
            try:
                row = await asyncio.to_thread(
                    compute_decision_row,
                    redis_client=redis_client,
                    request=request,
                    clock=decision_clock,
                    settings=settings,
                    screener=screener,
                    bank_id=bank_id,
                )
            except Exception as exc:
                logger.error("Failed to compute decision features: %s", exc)
                row = None
        if row is not None:
            await insert_feature_log_row(connection, row)

    response = DecisionResponse(
        payment_id=request.payment_id,
        decision="unscored",
        mode=mode,
        reason=DECISION_REASON,
    )
    return 200, response.model_dump(mode="json")


async def record_transaction(
    connection: asyncpg.Connection,
    bank_id: str,
    request: TransactionSideRequest,
) -> tuple[int, dict[str, Any]]:
    """
    WHAT: Validates and persists one side (debit or credit) of a UPI payment into transactions and outbox.
    It constructs the internal UpiPaymentSide, validates side-level rules, and calls record_upi_side.

    WHY: Provides the core business transaction ingestion logic independently of HTTP request envelopes.

    Args:
        connection: Active bank-scoped database connection.
        bank_id: Commercial bank identifier.
        request: Validated TransactionSideRequest model.

    Returns:
        Tuple of status code (201) and response body dictionary.

    Raises:
        ApiError: 422 'invalid_body' if the transaction side violates consistency rules.
    """
    upi_side = UpiPaymentSide(
        payment_id=request.payment_id,
        bank_id=bank_id,
        side=request.side,
        account_id=request.account_id,
        counterparty_bank_id=request.counterparty_bank_id,
        sender_bank_id=request.sender_bank_id,
        receiver_bank_id=request.receiver_bank_id,
        amount_paise=request.amount_paise,
        event_time=request.event_time,
        status=request.status,
        counterparty_vpa=request.counterparty_vpa,
        failure_domain=request.failure_domain,
        payee_type=request.payee_type,
        channel=request.channel,
        sender_device_fingerprint=request.sender_device_fingerprint,
        sender_os_generation=request.sender_os_generation,
        balance_after_paise=request.balance_after_paise,
    )
    try:
        validate_side(upi_side)
    except ValueError as error:
        raise ApiError(
            422, "invalid_body", "Transaction side is inconsistent: check side, bank ids, status and failure_domain"
        ) from error

    stored = await record_upi_side(connection, upi_side)
    response = TransactionResponse(
        transaction_id=stored.transaction_id,
        event_id=stored.event_id,
        was_replay=stored.was_replay,
    )
    return 201, response.model_dump(mode="json")


async def record_account_event(
    connection: asyncpg.Connection,
    bank_id: str,
    request: AccountEventRequest,
) -> tuple[int, dict[str, Any]]:
    """
    WHAT: Persists an account life-cycle security or provisioning event into bank_account_events.
    If the event identifier is already recorded for this bank, it ignores the duplicate and answers 200;
    otherwise, it inserts the new record and answers 201.

    WHY: Encapsulates account event persistence logic for use across HTTP routers and internal connectors.

    Args:
        connection: Active bank-scoped database connection.
        bank_id: Commercial bank identifier.
        request: Validated AccountEventRequest model.

    Returns:
        Tuple of status code (200 on duplicate, 201 on new insert) and response body dictionary.
    """
    details_json = orjson.dumps(request.details).decode("utf-8")
    inserted_id = await connection.fetchval(
        """
        INSERT INTO bank_account_events (bank_id, event_id, account_id, kind, event_time, details)
        VALUES ($1, $2, $3, $4, $5, $6::jsonb)
        ON CONFLICT (bank_id, event_id) DO NOTHING
        RETURNING id
        """,
        bank_id,
        request.event_id,
        request.account_id,
        request.kind,
        request.event_time,
        details_json,
    )
    duplicate = inserted_id is None
    response = AccountEventResponse(event_id=request.event_id, duplicate=duplicate)
    return (200 if duplicate else 201), response.model_dump(mode="json")


async def record_complaint(
    connection: asyncpg.Connection,
    bank_id: str,
    request: ComplaintRequest,
) -> tuple[int, dict[str, Any]]:
    """
    WHAT: Stores a customer fraud complaint referencing one or more payment identifiers into bank_complaints.
    If the complaint identifier already exists for this bank, it raises an ApiError(409).

    WHY: Standardizes fraud complaint ingestion across external HTTP endpoints and in-process simulation connectors.

    Args:
        connection: Active bank-scoped database connection.
        bank_id: Commercial bank identifier.
        request: Validated ComplaintRequest model.

    Returns:
        Tuple of status code (201) and response body dictionary.

    Raises:
        ApiError: 409 'duplicate_record' if a complaint with this complaint_id already exists.
    """
    try:
        await connection.execute(
            """
            INSERT INTO bank_complaints (
                bank_id, complaint_id, complainant_account, reported_account, reported_bank_id,
                payment_ids, claimed_type, received_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """,
            bank_id,
            request.complaint_id,
            request.complainant_account,
            request.reported_account,
            request.reported_bank_id,
            request.payment_ids,
            request.claimed_type,
            request.received_at,
        )
    except asyncpg.UniqueViolationError as error:
        if error.constraint_name != COMPLAINT_UNIQUE_CONSTRAINT:
            raise
        raise ApiError(409, "duplicate_record", "A record with this id already exists") from error
    return 201, ComplaintResponse(complaint_id=request.complaint_id).model_dump(mode="json")


async def record_withdrawal(
    pool_or_conn: asyncpg.Pool | asyncpg.Connection,
    bank_id: str,
    request: WithdrawalRequest,
) -> tuple[bool, Any]:
    """
    WHAT: Persists an ATM or branch cash withdrawal into bank_withdrawals.
    Returns (True, row) if inserted, (False, row) if duplicate.
    Run under the same bank tenant role / app.bank_id setting as other bank writes.

    Args:
        pool_or_conn: Active connection pool or connection.
        bank_id: Authorized commercial bank identifier.
        request: Validated WithdrawalRequest model.

    Returns:
        Tuple of (inserted_boolean, row_record).
    """
    if request.bank_id != bank_id:
        raise ApiError(422, "invalid_body", "bank_id must match authenticated bank")

    async def _execute_insert(conn: asyncpg.Connection) -> tuple[bool, Any]:
        row = await conn.fetchrow(
            """
            INSERT INTO bank_withdrawals (
                bank_id, withdrawal_id, account_id, channel, amount_paise,
                status, failure_domain, event_time, balance_after_paise
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            ON CONFLICT (bank_id, withdrawal_id) DO NOTHING
            RETURNING *
            """,
            bank_id,
            request.withdrawal_id,
            request.account_id,
            request.channel,
            request.amount_paise,
            request.status,
            request.failure_domain,
            request.event_time,
            request.balance_after_paise,
        )
        if row is not None:
            return True, row
        existing = await conn.fetchrow(
            "SELECT * FROM bank_withdrawals WHERE bank_id = $1 AND withdrawal_id = $2",
            bank_id,
            request.withdrawal_id,
        )
        return False, existing

    if isinstance(pool_or_conn, asyncpg.Pool):
        async with pool_or_conn.acquire() as connection:
            async with bank_scoped_transaction(connection, bank_id) as scoped:
                return await _execute_insert(scoped)
    return await _execute_insert(pool_or_conn)


def parse_raw_body(body_bytes: bytes, model_class: type[ModelT]) -> ModelT:
    """
    WHAT: Parses raw JSON bytes into a validated Pydantic model instance.
    If validation fails, it extracts error locations and raises an ApiError(422, 'invalid_body').

    WHY: Ensures identical body validation logic and error structures between the HTTP parser and InProcessBankClient.

    Args:
        body_bytes: Raw JSON payload bytes to validate.
        model_class: Target Pydantic model class.

    Returns:
        Validated model instance.

    Raises:
        ApiError: 422 'invalid_body' naming the offending field locations without echoing raw input values.
    """
    try:
        return model_class.model_validate_json(body_bytes)
    except ValidationError as error:
        locations: list[str] = []
        for problem in error.errors():
            location = ".".join(str(part) for part in problem["loc"]) or "body"
            if location not in locations:
                locations.append(location)
        raise ApiError(422, "invalid_body", f"Validation failed at: {', '.join(locations)}") from error


class InProcessBankClient:
    """
    WHAT: In-process client for commercial banks calling bank API routes directly without network overhead.
    It validates request body bytes using official Pydantic models, computes matching SHA-256 request hashes,
    and executes service functions under bank-scoped idempotency control.

    WHY: Level 14c connector integration requires simulation components to feed bank telemetry into the API
    datastore with full schema validation and idempotency guarantees, but without the latency, network
    transports, cryptographic signing, and nonce management overhead of HTTP endpoints.

    Note: InProcessBankClient operates entirely in-memory within the trusted application process.
    It does not sign requests and records no nonces because there is no transport layer to protect against replay.
    """

    def __init__(
        self,
        pool: asyncpg.Pool,
        bank_id: str,
        clock: Clock,
        *,
        redis_client: Any = None,
        decision_clock: Any = None,
        settings: Any = None,
        screener: Any = None,
    ) -> None:
        """
        Initializes an in-process client instance.

        Args:
            pool: Connection pool for tenant-scoped database access.
            bank_id: Authorized commercial bank identifier.
            clock: Clock providing transport timestamps for idempotency expiration.
            redis_client: Optional Redis client for feature computation.
            decision_clock: Optional decision evaluation clock.
            settings: Optional AccountFeaturesSettings.
            screener: Optional RegistryScreener instance.
        """
        self._pool = pool
        self._bank_id = bank_id
        self._clock = clock
        self._redis_client = redis_client
        self._decision_clock = decision_clock
        self._settings = settings
        self._screener = screener

    async def post(self, path: str, body_bytes: bytes, idempotency_key: str) -> IdempotentOutcome:
        """
        WHAT: Dispatches an in-process POST operation to the corresponding bank service function.
        It parses body_bytes using the matching request model, derives the request_hash, and invokes
        run_idempotent_operation.

        Args:
            path: Relative API endpoint path (e.g. '/v1/decisions', '/v1/transactions',
                '/v1/account-events', '/v1/complaints').
            body_bytes: Raw UTF-8 JSON request body bytes.
            idempotency_key: Client-provided idempotency key string.

        Returns:
            An IdempotentOutcome record containing status, response body, and replayed flag.

        Raises:
            ValueError: If path is not recognized among supported in-process routes.
            ApiError: If body validation fails (422) or idempotency conflict occurs (409).
        """
        if path == "/v1/decisions":
            decision_model = parse_raw_body(body_bytes, DecisionRequest)

            feature_row = None
            if config.decision_feature_logging_enabled:
                try:
                    feature_row = await asyncio.to_thread(
                        compute_decision_row,
                        redis_client=self._redis_client,
                        request=decision_model,
                        clock=self._decision_clock,
                        settings=self._settings,
                        screener=self._screener,
                        bank_id=self._bank_id,
                    )
                except Exception as exc:
                    logger.error("Failed to compute decision features: %s", exc)
                    feature_row = None

            async def handler(connection: asyncpg.Connection) -> tuple[int, dict[str, Any]]:
                return await record_decision(
                    connection,
                    self._bank_id,
                    decision_model,
                    redis_client=self._redis_client,
                    decision_clock=self._decision_clock,
                    settings=self._settings,
                    screener=self._screener,
                    feature_row=feature_row,
                )

        elif path == "/v1/transactions":
            tx_model = parse_raw_body(body_bytes, TransactionSideRequest)

            async def handler(connection: asyncpg.Connection) -> tuple[int, dict[str, Any]]:
                return await record_transaction(connection, self._bank_id, tx_model)

        elif path == "/v1/account-events":
            acct_model = parse_raw_body(body_bytes, AccountEventRequest)

            async def handler(connection: asyncpg.Connection) -> tuple[int, dict[str, Any]]:
                return await record_account_event(connection, self._bank_id, acct_model)

        elif path == "/v1/complaints":
            complaint_model = parse_raw_body(body_bytes, ComplaintRequest)

            async def handler(connection: asyncpg.Connection) -> tuple[int, dict[str, Any]]:
                return await record_complaint(connection, self._bank_id, complaint_model)

        elif path == "/v1/withdrawals":
            withdrawal_model = parse_raw_body(body_bytes, WithdrawalRequest)

            async def handler(connection: asyncpg.Connection) -> tuple[int, dict[str, Any]]:
                inserted, _ = await record_withdrawal(connection, self._bank_id, withdrawal_model)
                status_code = 201 if inserted else 200
                resp_status: Literal["recorded", "duplicate"] = "recorded" if inserted else "duplicate"
                resp = WithdrawalResponse(
                    status=resp_status,
                    withdrawal_id=withdrawal_model.withdrawal_id,
                    bank_id=self._bank_id,
                )
                return status_code, resp.model_dump(mode="json")

        else:
            raise ValueError(f"Unknown in-process path: {path!r}")

        request_hash = hash_request("POST", path, body_bytes)
        now = self._clock.now_ist()

        return await run_idempotent_operation(
            pool=self._pool,
            bank_id=self._bank_id,
            idempotency_key=idempotency_key,
            request_hash=request_hash,
            now=now,
            handler=handler,
        )

    async def record_withdrawal(
        self,
        withdrawal: WithdrawalRequest | dict[str, Any] | bytes,
        idempotency_key: str,
    ) -> IdempotentOutcome:
        """Records an ATM or branch withdrawal idempotently."""
        if isinstance(withdrawal, bytes):
            body_bytes = withdrawal
        elif isinstance(withdrawal, dict):
            body_bytes = orjson.dumps(withdrawal)
        else:
            body_bytes = orjson.dumps(withdrawal.model_dump(mode="json"))
        return await self.post("/v1/withdrawals", body_bytes, idempotency_key)
