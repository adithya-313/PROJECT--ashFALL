# File: api/v1/bank_router.py
# Purpose: The /v1 endpoints a commercial bank calls: decisions, transactions, account events, account risk,
#          complaints and feedback.
#
# Order of work in every route: (1) authenticate the request (key, signature, freshness, nonce) BEFORE the body
# is parsed, so an unsigned request with a broken body gets 401 and not 422; (2) parse and validate the body;
# (3) for POST routes, claim the idempotency key and do the business write in ONE bank-scoped transaction.
import re
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

import asyncpg
import orjson
from fastapi import APIRouter, Request
from fastapi.responses import ORJSONResponse
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
)
from api.v1.request_auth import BankRequestContext, authenticate_bank_request, raw_request_path
from api.v1.upi_outbox import UpiPaymentSide, record_upi_side, validate_side
from core.event_clock import Clock, WallClock

router = APIRouter(prefix="/v1")

ModelT = TypeVar("ModelT", bound=BaseModel)
# A handler does the business write on a bank-scoped connection and returns (HTTP status, response body).
Handler = Callable[[asyncpg.Connection], Awaitable[tuple[int, dict[str, Any]]]]

IDENTIFIER_REGEX = re.compile(IDENTIFIER_PATTERN)
DECISION_UNIQUE_CONSTRAINT = "uq_bank_decision_requests_bank_payment"
COMPLAINT_UNIQUE_CONSTRAINT = "uq_bank_complaints_bank_complaint"
FEEDBACK_UNIQUE_CONSTRAINT = "uq_bank_feedback_bank_feedback"
DECISION_REASON = "scoring is not enabled until Level 19"


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

    async with pool.acquire() as connection:
        async with bank_scoped_transaction(connection, ctx.bank_id) as scoped:
            claim = await claim_idempotency_key(scoped, ctx.bank_id, ctx.idempotency_key, request_hash, now)
            if claim.outcome == Outcome.CONFLICT:
                raise ApiError(409, "idempotency_key_reused", "This Idempotency-Key was used with a different request")
            if claim.outcome == Outcome.REPLAY:
                if claim.stored_status is None or claim.stored_body is None:
                    raise IdempotencyStateError("A replayed idempotency key has no stored response")
                return ORJSONResponse(
                    status_code=claim.stored_status,
                    content=claim.stored_body,
                    headers={"Idempotent-Replayed": "true"},
                )
            status, body = await handler(scoped)
            await store_idempotent_response(scoped, ctx.bank_id, ctx.idempotency_key, status, body)
    return ORJSONResponse(status_code=status, content=body)


@router.post(
    "/decisions", response_model=DecisionResponse, openapi_extra=request_schema(DecisionRequest)
)
async def post_decision(request: Request) -> ORJSONResponse:
    """
    Records a bank's request for a decision on a payment. Scoring does not exist yet, so the answer is always
    "unscored" in the bank's mode. The same payment id under a different idempotency key is a 409.
    """
    ctx = await get_bank_context(request)
    decision_request = parse_body(ctx, DecisionRequest)
    request_json = orjson.dumps(decision_request.model_dump(mode="json")).decode("utf-8")

    async def handler(connection: asyncpg.Connection) -> tuple[int, dict[str, Any]]:
        try:
            await connection.execute(
                """
                INSERT INTO bank_decision_requests (bank_id, payment_id, request, decision, mode)
                VALUES ($1, $2, $3::jsonb, 'unscored', $4)
                """,
                ctx.bank_id,
                decision_request.payment_id,
                request_json,
                POLICY.default_bank_mode,
            )
        except asyncpg.UniqueViolationError as error:
            if error.constraint_name != DECISION_UNIQUE_CONSTRAINT:
                raise
            raise ApiError(
                409, "payment_already_decided", "A decision was already requested for this payment"
            ) from error
        response = DecisionResponse(
            payment_id=decision_request.payment_id,
            decision="unscored",
            mode=POLICY.default_bank_mode,
            reason=DECISION_REASON,
        )
        return 200, response.model_dump(mode="json")

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
    upi_side = UpiPaymentSide(
        payment_id=side_request.payment_id,
        bank_id=ctx.bank_id,
        side=side_request.side,
        account_id=side_request.account_id,
        counterparty_bank_id=side_request.counterparty_bank_id,
        sender_bank_id=side_request.sender_bank_id,
        receiver_bank_id=side_request.receiver_bank_id,
        amount_paise=side_request.amount_paise,
        event_time=side_request.event_time,
        status=side_request.status,
        counterparty_vpa=side_request.counterparty_vpa,
        failure_domain=side_request.failure_domain,
        payee_type=side_request.payee_type,
        channel=side_request.channel,
    )
    try:
        validate_side(upi_side)
    except ValueError as error:
        # The message names no values: it only says which kind of rule failed.
        raise ApiError(
            422, "invalid_body", "Transaction side is inconsistent: check side, bank ids, status and failure_domain"
        ) from error

    async def handler(connection: asyncpg.Connection) -> tuple[int, dict[str, Any]]:
        stored = await record_upi_side(connection, upi_side)
        response = TransactionResponse(
            transaction_id=stored.transaction_id, event_id=stored.event_id, was_replay=stored.was_replay
        )
        return 201, response.model_dump(mode="json")

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
    details_json = orjson.dumps(event_request.details).decode("utf-8")

    async def handler(connection: asyncpg.Connection) -> tuple[int, dict[str, Any]]:
        inserted_id = await connection.fetchval(
            """
            INSERT INTO bank_account_events (bank_id, event_id, account_id, kind, event_time, details)
            VALUES ($1, $2, $3, $4, $5, $6::jsonb)
            ON CONFLICT (bank_id, event_id) DO NOTHING
            RETURNING id
            """,
            ctx.bank_id,
            event_request.event_id,
            event_request.account_id,
            event_request.kind,
            event_request.event_time,
            details_json,
        )
        duplicate = inserted_id is None
        response = AccountEventResponse(event_id=event_request.event_id, duplicate=duplicate)
        return (200 if duplicate else 201), response.model_dump(mode="json")

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
        try:
            await connection.execute(
                """
                INSERT INTO bank_complaints (
                    bank_id, complaint_id, complainant_account, reported_account, reported_bank_id,
                    payment_ids, claimed_type, received_at
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """,
                ctx.bank_id,
                complaint.complaint_id,
                complaint.complainant_account,
                complaint.reported_account,
                complaint.reported_bank_id,
                complaint.payment_ids,
                complaint.claimed_type,
                complaint.received_at,
            )
        except asyncpg.UniqueViolationError as error:
            raise as_duplicate_record_error(error, COMPLAINT_UNIQUE_CONSTRAINT) from error
        return 201, ComplaintResponse(complaint_id=complaint.complaint_id).model_dump(mode="json")

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


@router.get("/accounts/{account_id}/risk", response_model=RiskResponse)
async def get_account_risk(account_id: str, request: Request) -> ORJSONResponse:
    """
    Returns the risk of one of the calling bank's accounts. Scoring does not exist yet, so it is "unscored".

    The bank-scoped query only sees the calling bank's rows, so "another bank's account" and "no such account"
    give the same 404 answer and nothing leaks about other banks.
    """
    ctx = await get_bank_context(request)
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
    response = RiskResponse(account_id=account_id, risk="unscored", mode=POLICY.default_bank_mode)
    return ORJSONResponse(status_code=200, content=response.model_dump(mode="json"))
