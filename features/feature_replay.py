# File: features/feature_replay.py
# Purpose: Level 15c historical feature replay and point-in-time recomputation verification.
# Issue: Level 15c Round 2a (recompute a logged feature row from history).
"""
WHAT: Provides historical event replay from PostgreSQL tables into an isolated scratch Redis
instance, and verifies point-in-time feature reproducibility against feature_log records.

WHY: Satisfies Section 22 validation: proves that feature values logged at decision time can be
bit-for-bit reproduced from relational transaction and event history up to the logged arrival watermark.
Scratch Redis ensures absolute isolation: live operational keys and watermarks are never mutated.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import orjson

from api.v1.models import DecisionRequest
from features.account_keys import (
    AccountFeaturesSettings,
    extract_run_id,
    get_default_settings,
)
from features.account_labels import apply_complaint_row, apply_feedback_row
from features.account_records import (
    account_event_from_row,
    payment_side_from_transaction_row,
    withdrawal_from_row,
)
from features.account_updater import (
    apply_account_event,
    apply_payment_side,
    apply_withdrawal,
)
from features.decision_features import compute_feature_values
from features.feature_log import (
    canonical_feature_names,
    read_feature_log_row,
    to_ordered_values,
)
from features.registry_screening import ScreenOutcome

__all__ = [
    "LoggedAnswerScreener",
    "RecomputeResult",
    "ReplayCounts",
    "recompute_logged_row",
    "replay_history_into_redis",
]


@dataclass(frozen=True)
class ReplayCounts:
    """
    Counts of historical domain rows replayed into scratch Redis.

    Attributes:
        transactions: Count of transaction payment sides applied.
        account_events: Count of bank account lifecycle events applied.
        withdrawals: Count of cash withdrawals applied.
        feedback: Count of bank feedback rows applied.
        complaints: Count of bank complaints applied.
    """

    transactions: int
    account_events: int
    withdrawals: int
    feedback: int
    complaints: int


@dataclass(frozen=True)
class RecomputeResult:
    """
    Comparison outcome between a logged feature row and its historical recomputation.

    Attributes:
        logged: List of logged feature values in canonical order.
        recomputed: List of recomputed feature values in canonical order.
        mismatches: List of (feature_name, logged_value, recomputed_value) tuples.
    """

    logged: list[Any]
    recomputed: list[Any]
    mismatches: list[tuple[str, Any, Any]]


class LoggedAnswerScreener:
    """
    WHAT: Registry screener adapter answering queries using recorded answers from feature_log.

    WHY: Proves historical determinism without calling external registries during replay.
    Enforces that recompute asks identical questions by raising LookupError if a query was not logged.
    """

    def __init__(self, registry_answers: list[dict[str, Any]] | None) -> None:
        self._answers: dict[tuple[str, str], ScreenOutcome] = {}
        if registry_answers:
            for item in registry_answers:
                ident_hash = str(item["identifier_hash"])
                ident_type = str(item["identifier_type"])
                outcome_data = item.get("outcome")
                if isinstance(outcome_data, dict):
                    status = str(outcome_data.get("status", "unknown"))
                    raw_vis = outcome_data.get("first_visible_at")
                    first_vis = datetime.fromisoformat(raw_vis) if raw_vis else None
                    outcome = ScreenOutcome(status=status, first_visible_at=first_vis)
                elif isinstance(outcome_data, ScreenOutcome):
                    outcome = outcome_data
                elif isinstance(outcome_data, str):
                    outcome = ScreenOutcome(status=outcome_data, first_visible_at=None)
                else:
                    outcome = ScreenOutcome(status="unknown", first_visible_at=None)
                self._answers[(ident_type, ident_hash)] = outcome

    def screen(
        self,
        identifier_hash: str,
        identifier_type: str,
        as_of: datetime | None = None,
    ) -> ScreenOutcome:
        """
        Returns the logged ScreenOutcome for (identifier_type, identifier_hash).

        Raises:
            LookupError: If the identifier was not queried in the logged decision.
        """
        key = (identifier_type, identifier_hash)
        if key not in self._answers:
            raise LookupError(
                f"Registry query for identifier_type={identifier_type!r}, "
                f"identifier_hash={identifier_hash!r} was not logged in feature_log"
            )
        return self._answers[key]


def _ensure_utc(dt: datetime) -> datetime:
    """Ensures datetime is timezone-aware and normalized to UTC for sorting."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


async def replay_history_into_redis(
    pg_connection: Any,
    scratch_redis: Any,
    run_id: str,
    bank_id: str,
    arrival_cutoff: datetime,
    settings: AccountFeaturesSettings | None = None,
) -> ReplayCounts:
    """
    WHAT: Replays historical database rows up to arrival_cutoff into scratch Redis.

    WHY: Reconstructs exact point-in-time account feature state without depending on Kafka
    or mutating live Redis data. Applies rows from all 5 feeds sorted in merged arrival order.
    Writes NO watermarks to Redis.

    NOTE: scratch_redis must be a separate Redis instance (or isolated Redis database)
    from the live operational store to avoid polluting real-time streaming state.

    Args:
        pg_connection: Active PostgreSQL database connection.
        scratch_redis: Isolated Redis or FakeRedis client instance.
        run_id: 16-hex simulation run identifier to isolate.
        bank_id: Tenant commercial bank identifier.
        arrival_cutoff: Ingestion watermark threshold (inclusive upper bound on arrival).
        settings: Optional AccountFeaturesSettings override.

    Returns:
        ReplayCounts tracking rows processed per domain table.
    """
    cfg = settings if settings is not None else get_default_settings()

    # 1. Fetch transactions (arrival = created_at)
    txn_rows = await pg_connection.fetch(
        """
        SELECT id, bank_id, payment_id, account_id, side, counterparty_bank_id,
               counterparty_vpa, amount_paise, status, failure_domain, payee_type,
               channel, sender_device_fingerprint, sender_os_generation,
               balance_after_paise, event_time, created_at
        FROM transactions
        WHERE bank_id = $1 AND created_at <= $2
        ORDER BY id
        """,
        bank_id,
        arrival_cutoff,
    )

    # 2. Fetch bank_account_events (arrival = created_at)
    event_rows = await pg_connection.fetch(
        """
        SELECT id, bank_id, event_id, account_id, kind, event_time, details, created_at
        FROM bank_account_events
        WHERE bank_id = $1 AND created_at <= $2
        ORDER BY id
        """,
        bank_id,
        arrival_cutoff,
    )

    # 3. Fetch bank_withdrawals (arrival = recorded_at)
    withdrawal_rows = await pg_connection.fetch(
        """
        SELECT id, bank_id, withdrawal_id, account_id, channel, amount_paise,
               status, failure_domain, event_time, balance_after_paise, recorded_at
        FROM bank_withdrawals
        WHERE bank_id = $1 AND recorded_at <= $2
        ORDER BY id
        """,
        bank_id,
        arrival_cutoff,
    )

    # 4. Fetch bank_feedback (arrival = created_at)
    feedback_rows = await pg_connection.fetch(
        """
        SELECT id, bank_id, feedback_id, payment_id, outcome, note, event_time, created_at
        FROM bank_feedback
        WHERE bank_id = $1 AND created_at <= $2
        ORDER BY id
        """,
        bank_id,
        arrival_cutoff,
    )

    # 5. Fetch bank_complaints (arrival = created_at)
    complaint_rows = await pg_connection.fetch(
        """
        SELECT id, bank_id, complaint_id, complainant_account, reported_account,
               reported_bank_id, payment_ids, claimed_type, received_at, created_at
        FROM bank_complaints
        WHERE bank_id = $1 AND created_at <= $2
        ORDER BY id
        """,
        bank_id,
        arrival_cutoff,
    )

    # Merge items into unified stream sorted by (arrival, table_order, id)
    # Table orders: 0=transactions, 1=account_events, 2=withdrawals, 3=feedback, 4=complaints
    items: list[tuple[datetime, int, str, str, Mapping[str, Any]]] = []

    count_txns = 0
    for r in txn_rows:
        pid = str(r["payment_id"])
        try:
            row_run = extract_run_id(pid)
        except ValueError as exc:
            raise ValueError(f"transactions row {pid!r} has unparseable id: {exc}") from exc
        if row_run != run_id:
            continue
        arr = _ensure_utc(r["created_at"])
        items.append((arr, 0, pid, "transaction", r))
        count_txns += 1

    count_events = 0
    for r in event_rows:
        eid = str(r["event_id"])
        try:
            row_run = extract_run_id(eid)
        except ValueError as exc:
            raise ValueError(f"bank_account_events row {eid!r} has unparseable id: {exc}") from exc
        if row_run != run_id:
            continue
        arr = _ensure_utc(r["created_at"])
        items.append((arr, 1, eid, "account_event", r))
        count_events += 1

    count_withdrawals = 0
    for r in withdrawal_rows:
        wid = str(r["withdrawal_id"])
        try:
            row_run = extract_run_id(wid)
        except ValueError as exc:
            raise ValueError(f"bank_withdrawals row {wid!r} has unparseable id: {exc}") from exc
        if row_run != run_id:
            continue
        arr = _ensure_utc(r["recorded_at"])
        items.append((arr, 2, wid, "withdrawal", r))
        count_withdrawals += 1

    count_feedback = 0
    for r in feedback_rows:
        pid = str(r["payment_id"])
        try:
            row_run = extract_run_id(pid)
        except ValueError as exc:
            raise ValueError(f"bank_feedback row {pid!r} has unparseable id: {exc}") from exc
        if row_run != run_id:
            continue
        arr = _ensure_utc(r["created_at"])
        items.append((arr, 3, str(r["feedback_id"]), "feedback", r))
        count_feedback += 1

    count_complaints = 0
    for r in complaint_rows:
        cid = str(r["complaint_id"])
        try:
            row_run = extract_run_id(cid)
        except ValueError as exc:
            raise ValueError(f"bank_complaints row {cid!r} has unparseable id: {exc}") from exc
        if row_run != run_id:
            continue
        arr = _ensure_utc(r["created_at"])
        items.append((arr, 4, cid, "complaint", r))
        count_complaints += 1

    # Sort merged order: (arrival, table_order, id)
    items.sort(key=lambda item: (item[0], item[1], item[2]))

    # Apply in merged chronological arrival sequence without writing watermarks
    for _arr, _order, _id_str, kind, row in items:
        if kind == "transaction":
            side_rec = payment_side_from_transaction_row(row)
            apply_payment_side(scratch_redis, side_rec, settings=cfg)
        elif kind == "account_event":
            event_rec = account_event_from_row(row)
            apply_account_event(scratch_redis, event_rec, settings=cfg)
        elif kind == "withdrawal":
            with_rec = withdrawal_from_row(row)
            apply_withdrawal(scratch_redis, with_rec, settings=cfg)
        elif kind == "feedback":
            await apply_feedback_row(pg_connection, scratch_redis, row, settings=cfg)
        elif kind == "complaint":
            apply_complaint_row(scratch_redis, row, settings=cfg)

    return ReplayCounts(
        transactions=count_txns,
        account_events=count_events,
        withdrawals=count_withdrawals,
        feedback=count_feedback,
        complaints=count_complaints,
    )


async def recompute_logged_row(
    pg_connection: Any,
    scratch_redis: Any,
    bank_id: str,
    payment_id: str,
    dpip_key: str | None = None,
    settings: AccountFeaturesSettings | None = None,
) -> RecomputeResult:
    """
    WHAT: Recomputes a decision feature vector from history and verifies it against feature_log.

    WHY: Proves reproducible Section 22 correctness. Reconstructs state in scratch Redis,
    re-evaluates features with LoggedAnswerScreener and logged timestamps, and compares
    canonical values with strict equality (==) and type match.

    Args:
        pg_connection: Active PostgreSQL connection.
        scratch_redis: Isolated Redis or FakeRedis client instance.
        bank_id: Commercial bank identifier.
        payment_id: Identifier of the evaluated payment.
        dpip_key: Optional explicit DPIP HMAC key.
        settings: Optional AccountFeaturesSettings.

    Returns:
        RecomputeResult containing logged values, recomputed values, and any mismatches.

    Raises:
        ValueError: If row not found, has no_watermark/error status, or request record missing.
    """
    log_row = await read_feature_log_row(pg_connection, bank_id, payment_id)
    if log_row is None:
        raise ValueError(f"Feature log row not found for bank {bank_id!r}, payment {payment_id!r}")

    if log_row.status in ("no_watermark", "error"):
        raise ValueError(f"Feature log row status is {log_row.status!r}; nothing to recompute")

    if log_row.arrival_cutoff is None:
        raise ValueError(f"Feature log row for payment {payment_id!r} has null arrival_cutoff")

    # Read original request from bank_decision_requests
    req_row = await pg_connection.fetchrow(
        """
        SELECT request
        FROM bank_decision_requests
        WHERE bank_id = $1 AND payment_id = $2
        """,
        bank_id,
        payment_id,
    )
    if req_row is None:
        raise ValueError(f"Decision request record not found for bank {bank_id!r}, payment {payment_id!r}")

    raw_req = req_row["request"]
    if isinstance(raw_req, (str, bytes)):
        req_dict = orjson.loads(raw_req)
    elif isinstance(raw_req, dict):
        req_dict = dict(raw_req)
    else:
        raise ValueError(f"Unexpected request format in bank_decision_requests: {type(raw_req)}")

    decision_req = DecisionRequest.model_validate(req_dict)

    # Replay historical state up to logged arrival cutoff into scratch Redis
    await replay_history_into_redis(
        pg_connection=pg_connection,
        scratch_redis=scratch_redis,
        run_id=log_row.run_id,
        bank_id=bank_id,
        arrival_cutoff=log_row.arrival_cutoff,
        settings=settings,
    )

    screener = LoggedAnswerScreener(log_row.registry_answers)

    comp = compute_feature_values(
        redis_client=scratch_redis,
        bank_id=bank_id,
        run_id=log_row.run_id,
        request=decision_req,
        decision_time=log_row.decision_time,
        arrival_cutoff=log_row.arrival_cutoff,
        screener=screener,
        dpip_key=dpip_key,
        settings=settings,
    )

    canonical_names = canonical_feature_names()
    recomputed_list = to_ordered_values(comp.feature_values)

    if isinstance(log_row.feature_values, list):
        logged_list = list(log_row.feature_values)
    elif isinstance(log_row.feature_values, dict):
        logged_list = to_ordered_values(log_row.feature_values)
    else:
        raise ValueError(f"Unexpected feature_values type in FeatureLogRow: {type(log_row.feature_values)}")

    mismatches: list[tuple[str, Any, Any]] = []
    for name, logged_val, recomp_val in zip(canonical_names, logged_list, recomputed_list):
        # Strict equality AND exact same type; no tolerance
        if not (logged_val == recomp_val and type(logged_val) is type(recomp_val)):
            mismatches.append((name, logged_val, recomp_val))

    return RecomputeResult(
        logged=logged_list,
        recomputed=recomputed_list,
        mismatches=mismatches,
    )
