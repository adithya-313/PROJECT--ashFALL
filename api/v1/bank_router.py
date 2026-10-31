# File: api/v1/bank_router.py
# Purpose: The /v1 endpoints a commercial bank calls: decisions, transactions, account events, account risk,
#          complaints and feedback.
#
# Order of work in every route: (1) authenticate the request (key, signature, freshness, nonce) BEFORE the body
# is parsed, so an unsigned request with a broken body gets 401 and not 422; (2) parse and validate the body;
# (3) for POST routes, claim the idempotency key and do the business write in ONE bank-scoped transaction.
import logging
import re
from collections.abc import Awaitable, Callable
from typing import Any, Literal, TypeVar

import asyncpg
from fastapi import APIRouter, Request
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel, ValidationError

from api.v1.bank_scope import bank_scoped_transaction
from api.v1.bank_service import (
    get_bank_mode,
    record_account_event,
    record_complaint,
    record_decision,
    record_transaction,
    record_withdrawal,
    run_idempotent_operation,
)
from api.v1.errors import ApiError
from api.v1.idempotency import hash_request
from api.v1.models import (
    IDENTIFIER_PATTERN,
    POLICY,
    AccountEventRequest,
    AccountEventResponse,
    ComplaintRequest,
    ComplaintResponse,
    DecisionRequest,
    DecisionResponse,
    FeedbackRequest,
    FeedbackResponse,
    RiskResponse,
    TransactionResponse,
    TransactionSideRequest,
    WithdrawalRequest,
    WithdrawalResponse,
)
from api.v1.request_auth import BankRequestContext, authenticate_bank_request, raw_request_path
from core.config import config
from core.event_clock import Clock, WallClock

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1")

ModelT = TypeVar("ModelT", bound=BaseModel)
# A handler does the business write on a bank-scoped connection and returns (HTTP status, response body).
Handler = Callable[[asyncpg.Connection], Awaitable[tuple[int, dict[str, Any]]]]

IDENTIFIER_REGEX = re.compile(IDENTIFIER_PATTERN)
FEEDBACK_UNIQUE_CONSTRAINT = "uq_bank_feedback_bank_feedback"


def get_transport_clock(request: Request) -> Clock:
    """
    Returns the clock used for transport checks (signature freshness and idempotency lifetime).

    Tests put a ManualClock on app.state.transport_clock so time is under their control. Production sets nothing
    and gets the real wall clock, as ground rule 30 requires for transport time.
    """
    if hasattr(request.app.state, "transport_clock"):
        clock: Clock = request.app.state.transport_clock
        return clock
    return WallClock()


async def get_bank_context(request: Request) -> BankRequestContext:
    """
    Authenticates the request and returns who the bank is, the raw body and the idempotency key.

    Raises:
        BankAuthError: If authentication fails (the app turns it into a 4xx/5xx error response).
    """
    pool: asyncpg.Pool = request.app.state.db_pool
    return await authenticate_bank_request(request, pool, get_transport_clock(request))


def parse_body(ctx: BankRequestContext, model_class: type[ModelT]) -> ModelT:
    """
    Parses the already-authenticated raw body into a request model.

    Args:
        ctx: The authenticated request context holding the raw body bytes.
        model_class: The pydantic model the body must match.

    Returns:
        The validated model.

    Raises:
        ApiError: 422 "invalid_body" naming only the field locations that failed, never the values sent.
    """
    try:
        return model_class.model_validate_json(ctx.body_bytes)
    except ValidationError as error:
        locations: list[str] = []
        for problem in error.errors():
            location = ".".join(str(part) for part in problem["loc"]) or "body"
            if location not in locations:
                locations.append(location)
        raise ApiError(422, "invalid_body", "Request body is invalid; check: " + ", ".join(locations)) from error


def as_duplicate_record_error(error: asyncpg.UniqueViolationError, constraint_name: str) -> ApiError:
    """
    Turns a violation of one named business-id constraint into the 409 "duplicate_record" error.

    Args:
        error: The unique violation the database raised.
        constraint_name: The one constraint that means "this business id already exists".

    Returns:
        The ApiError to raise.

    Raises:
        asyncpg.UniqueViolationError: The same error again, if it came from any other constraint (that would be a
            real fault, not a duplicate id).
    """
    if error.constraint_name != constraint_name:
        raise error
    return ApiError(409, "duplicate_record", "A record with this id already exists")


def request_schema(model_class: type[BaseModel]) -> dict[str, Any]:
    """
    Builds the openapi_extra entry that documents a route's request body.

    Routes read the raw body themselves (so authentication can run first), so FastAPI cannot see the model; this
    keeps the model in the OpenAPI document for clients and for fuzzing.
    """
    return {
        "requestBody": {
            "required": True,
            "content": {"application/json": {"schema": model_class.model_json_schema()}},
        }
    }


async def run_idempotent_post(request: Request, ctx: BankRequestContext, handler: Handler) -> ORJSONResponse:
    """
    Runs one POST under the bank's idempotency key.

    Claiming the key, doing the business write and storing the response all happen in ONE bank-scoped
    transaction. If the write fails, the claim rolls back with it, so the bank can retry the same key; if the
    process dies after the commit, the stored response is already there to replay. Two requests with the same key
    at the same moment queue on the key's row lock, so the second sees the first one's committed response.

    Args:
        request: The incoming request (its method and path are part of the request hash).
        ctx: The authenticated context (bank id, raw body, idempotency key).
        handler: Does the business write and returns (status, body).

    Returns:
        The handler's response; a stored response with the header "Idempotent-Replayed: true" on a replay.

    Raises:
        ApiError: 409 "idempotency_key_reused" if the key was used with a different request.
        IdempotencyStateError: If a replayed key has no stored response (an impossible state).
    """
    if ctx.idempotency_key is None:
        raise ApiError(400, "bad_idempotency_key", "Missing Idempotency-Key header on POST")
    pool: asyncpg.Pool = request.app.state.db_pool
    path = raw_request_path(request)
    request_hash = hash_request(request.method, path, ctx.body_bytes)
    now = get_transport_clock(request).now_ist()

    outcome = await run_idempotent_operation(
        pool=pool,
        bank_id=ctx.bank_id,
        idempotency_key=ctx.idempotency_key,
        request_hash=request_hash,
        now=now,
        handler=handler,
    )
    headers = {"Idempotent-Replayed": "true"} if outcome.replayed else None
    return ORJSONResponse(status_code=outcome.status, content=outcome.body, headers=headers)


@router.post("/decisions", response_model=DecisionResponse, openapi_extra=request_schema(DecisionRequest))
async def post_decision(request: Request) -> ORJSONResponse:
    """
    Records a bank's request for a decision on a payment. Scoring does not exist yet, so the answer is always
    "unscored" in the bank's mode. The same payment id under a different idempotency key is a 409.
    """
    ctx = await get_bank_context(request)
    decision_request = parse_body(ctx, DecisionRequest)

    redis_client = getattr(request.app.state, "redis_client", None)
    decision_clock = getattr(request.app.state, "decision_clock", None)
    settings = getattr(request.app.state, "account_settings", None)
    screener = getattr(request.app.state, "registry_screener", None)
    registry = getattr(request.app.state, "registry_gateway", None)
    dpip_key = (
        config.dpip_participant_key.get_secret_value()
        if config.dpip_participant_key is not None
        else None
    )

    async def handler(connection: asyncpg.Connection) -> tuple[int, dict[str, Any]]:
        return await record_decision(
            connection,
            ctx.bank_id,
            decision_request,
            redis_client=redis_client,
            decision_clock=decision_clock,
            settings=settings,
            screener=screener,
            dpip_key=dpip_key,
            registry=registry,
        )

    return await run_idempotent_post(request, ctx, handler)


@router.post(
    "/transactions",
    status_code=201,
    response_model=TransactionResponse,
    openapi_extra=request_schema(TransactionSideRequest),
)
async def post_transaction(request: Request) -> ORJSONResponse:
    """
    Records one side (debit or credit) of a payment together with its outbox event. The bank on the record is the
    authenticated bank, never a value from the body.
    """
    ctx = await get_bank_context(request)
    side_request = parse_body(ctx, TransactionSideRequest)

    async def handler(connection: asyncpg.Connection) -> tuple[int, dict[str, Any]]:
        return await record_transaction(connection, ctx.bank_id, side_request)

    return await run_idempotent_post(request, ctx, handler)


@router.post(
    "/account-events",
    status_code=201,
    response_model=AccountEventResponse,
    openapi_extra=request_schema(AccountEventRequest),
)
async def post_account_event(request: Request) -> ORJSONResponse:
    """Records a life-cycle event of an account. A repeated event id is stored once and answered with 200."""
    ctx = await get_bank_context(request)
    event_request = parse_body(ctx, AccountEventRequest)

    async def handler(connection: asyncpg.Connection) -> tuple[int, dict[str, Any]]:
        return await record_account_event(connection, ctx.bank_id, event_request)

    return await run_idempotent_post(request, ctx, handler)


@router.post(
    "/complaints",
    status_code=201,
    response_model=ComplaintResponse,
    openapi_extra=request_schema(ComplaintRequest),
)
async def post_complaint(request: Request) -> ORJSONResponse:
    """
    Stores a fraud complaint the bank received from its customer. The same complaint id under a different
    idempotency key is a 409 "duplicate_record".
    """
    ctx = await get_bank_context(request)
    complaint = parse_body(ctx, ComplaintRequest)

    async def handler(connection: asyncpg.Connection) -> tuple[int, dict[str, Any]]:
        return await record_complaint(connection, ctx.bank_id, complaint)

    return await run_idempotent_post(request, ctx, handler)


@router.post(
    "/feedback",
    status_code=201,
    response_model=FeedbackResponse,
    openapi_extra=request_schema(FeedbackRequest),
)
async def post_feedback(request: Request) -> ORJSONResponse:
    """
    Stores the outcome a bank learned about a payment after the fact. The same feedback id under a different
    idempotency key is a 409 "duplicate_record".
    """
    ctx = await get_bank_context(request)
    feedback = parse_body(ctx, FeedbackRequest)

    async def handler(connection: asyncpg.Connection) -> tuple[int, dict[str, Any]]:
        try:
            await connection.execute(
                """
                INSERT INTO bank_feedback (bank_id, feedback_id, payment_id, outcome, note, event_time)
                VALUES ($1, $2, $3, $4, $5, $6)
                """,
                ctx.bank_id,
                feedback.feedback_id,
                feedback.payment_id,
                feedback.outcome,
                feedback.note,
                feedback.event_time,
            )
        except asyncpg.UniqueViolationError as error:
            raise as_duplicate_record_error(error, FEEDBACK_UNIQUE_CONSTRAINT) from error
        return 201, FeedbackResponse(feedback_id=feedback.feedback_id).model_dump(mode="json")

    return await run_idempotent_post(request, ctx, handler)


@router.post(
    "/withdrawals",
    status_code=201,
    response_model=WithdrawalResponse,
    openapi_extra=request_schema(WithdrawalRequest),
)
async def post_withdrawal(request: Request) -> ORJSONResponse:
    """Records an ATM or branch cash withdrawal. Repeated withdrawal id is stored once and answered with 200."""
    ctx = await get_bank_context(request)
    withdrawal = parse_body(ctx, WithdrawalRequest)
    if withdrawal.bank_id != ctx.bank_id:
        raise ApiError(422, "invalid_body", "bank_id must match authenticated bank")

    async def handler(connection: asyncpg.Connection) -> tuple[int, dict[str, Any]]:
        inserted, _ = await record_withdrawal(connection, ctx.bank_id, withdrawal)
        status_code = 201 if inserted else 200
        resp_status: Literal["recorded", "duplicate"] = "recorded" if inserted else "duplicate"
        resp = WithdrawalResponse(
            status=resp_status,
            withdrawal_id=withdrawal.withdrawal_id,
            bank_id=ctx.bank_id,
        )
        return status_code, resp.model_dump(mode="json")

    return await run_idempotent_post(request, ctx, handler)


@router.get("/accounts/{account_id}/risk", response_model=RiskResponse)
async def get_account_risk(account_id: str, request: Request) -> ORJSONResponse:
    """
    Returns the risk of one of the calling bank's accounts. Scoring does not exist yet, so it is "unscored".

    The bank-scoped query only sees the calling bank's rows, so "another bank's account" and "no such account"
    give the same 404 answer and nothing leaks about other banks.
    """
    ctx = await get_bank_context(request)
    if "\x00" in account_id:
        raise ApiError(422, "invalid_account_id", "Account id contains invalid characters")
    if len(account_id) > POLICY.max_identifier_length or not IDENTIFIER_REGEX.fullmatch(account_id):
        raise ApiError(422, "invalid_account_id", "Account id is not a valid identifier")

    pool: asyncpg.Pool = request.app.state.db_pool
    async with pool.acquire() as connection:
        async with bank_scoped_transaction(connection, ctx.bank_id) as scoped:
            account_known = await scoped.fetchval(
                """
                SELECT EXISTS (SELECT 1 FROM transactions WHERE account_id = $1)
                    OR EXISTS (SELECT 1 FROM bank_account_events WHERE account_id = $1)
                """,
                account_id,
            )
    if not account_known:
        raise ApiError(404, "account_not_found", "Account not found")
    mode = get_bank_mode(ctx.bank_id)
    response = RiskResponse(account_id=account_id, risk="unscored", mode=mode)
    return ORJSONResponse(status_code=200, content=response.model_dump(mode="json"))
