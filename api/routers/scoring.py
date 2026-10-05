"""
File: api/routers/scoring.py
Purpose: Provides the POST /score endpoint that accepts a transaction and returns a fraud score.

This is a STUB implementation. The fraud_score is a random number — there is no real model here.
Each scored transaction is also saved to Postgres together with an outbox event (api/services/outbox.py).
The purpose of this file is to establish the API contract (field names, types, response shape)
so that the rest of the system (clients, tests, CI) can be wired up before the model is ready.
The real model will be plugged in during a later level, replacing only the body of score_transaction.
"""

import random

import asyncpg
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.dependencies.db import get_db_pool
from api.services.outbox import record_transaction_with_outbox
from core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Request and response models
# These five request fields are frozen API-boundary fields. Their names and
# types must not be changed without updating all downstream consumers.
# ---------------------------------------------------------------------------


class TransactionRequest(BaseModel):
    """Incoming transaction to be scored for fraud risk."""

    amount: float
    account_id: str
    merchant_id: str
    device_fingerprint: str
    geo_coordinates: str


class ScoreResponse(BaseModel):
    """Fraud score result returned to the caller."""

    account_id: str
    fraud_score: float  # value between 0.0 (clean) and 1.0 (high fraud risk)
    model_version: str


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/score")


# STUB: This endpoint returns a random fraud_score. It will be replaced with a
# real model inference call in a later level. The request/response contract is
# intentionally locked in now so clients can integrate before the model exists.
@router.post("", response_model=ScoreResponse)
async def score_transaction(
    transaction: TransactionRequest, pool: asyncpg.Pool = Depends(get_db_pool)
) -> ScoreResponse:
    """
    Accept a transaction and return a fraud score.

    Currently returns a random score (stub-v0). Once the real model is wired
    up, only the body of this function changes — the route, request model, and
    response model stay the same.
    """
    # Log key fields so each scoring request is visible in the structured log.
    # Matches the logging style used in api/main.py (get_logger + keyword args).
    logger.info(
        "score request received",
        account_id=transaction.account_id,
        amount=transaction.amount,
    )

    # Generate a random score in [0.0, 1.0] as a stand-in for model inference.
    fraud_score = random.uniform(0.0, 1.0)
    model_version = "stub-v0"

    # Save the transaction and its outbox event together (one database transaction, see api/services/outbox.py).
    # If saving fails we do NOT return a score: a payment that was not durably stored must not look accepted, so the
    # caller gets a 500 and can retry. The failure is logged first so it is visible in the structured logs.
    try:
        async with pool.acquire() as connection:
            await record_transaction_with_outbox(connection, transaction, fraud_score, model_version)
    except Exception as error:
        logger.error(
            "failed to store transaction and outbox event",
            account_id=transaction.account_id,
            error_type=type(error).__name__,
            error=str(error),
        )
        raise HTTPException(status_code=500, detail="Transaction could not be stored") from error

    return ScoreResponse(
        account_id=transaction.account_id,
        fraud_score=fraud_score,
        model_version=model_version,
    )
