# File: replay/dataset_builder.py
# Purpose: Offline training/validation/test dataset construction from replay workspace tables.
"""
WHAT: Extracts and builds multi-world datasets from replay workspaces into PyArrow tables and Parquet files.
WHY: Level 16c-2 / Architect decisions Q2-Q5:
  - Multi-world concatenation partitioned strictly by split (training, validation, test, stress).
  - Cutoff filtering (decision_time <= C, snapshot_time <= C, arrival_time <= C).
  - Derived convenience columns (label_receiver_reported_by_cutoff, label_payment_named_by_cutoff).
  - Preserves no_watermark rows, reports None share per feature, excludes labels from feature columns.
  - Refuses training datasets containing held-out fraud tactic families.
  - Generates canonical content hash across tables independent of server wall-clock timestamps.
"""

import hashlib
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import asyncpg
import orjson
import pyarrow as pa
import pyarrow.parquet as pq

from core.config_validator import load_yaml
from features.account_snapshots import account_feature_names
from features.feature_log import canonical_feature_names
from replay.runner import ReplayDropContext
from world.replay_recording import RecordedWorld
from world.run_identity import REPO_ROOT, find_code_version, load_runner_config

UTC = timezone.utc
IST = ZoneInfo("Asia/Kolkata")

__all__ = [
    "Dataset",
    "WorldExtract",
    "build_dataset",
    "content_hash",
    "extract_world",
    "write_dataset",
]


@dataclass(frozen=True)
class WorldExtract:
    """
    WHAT: In-memory dump of a single world's replay workspace tables and run configuration.
    WHY: Facilitates fast offline dataset composition and caching without persistent live DB connections.
    """

    world_id: str
    split: str
    seed: int
    run_id: str
    gang_families: tuple[str, ...]
    decisions_per_event_payment: bool
    registry_wired: bool
    registry_wired_reason: str
    payment_rows: list[dict[str, Any]]
    snapshot_rows: list[dict[str, Any]]
    label_rows: list[dict[str, Any]]
    clock_start: str = "2026-09-01"
    label_status_rows: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Dataset:
    """
    WHAT: Packaged PyArrow tables and descriptive manifest for offline ML pipelines.
    WHY: Represents a point-in-time training/validation dataset evaluated at a strict world-clock cutoff.
    """

    payments: pa.Table
    snapshots: pa.Table
    labels: pa.Table
    manifest: dict[str, Any]


def _ensure_utc(dt: Any) -> Any:
    """Converts timezone-aware or naive datetime to UTC."""
    if dt is None:
        return None
    if isinstance(dt, datetime):
        if dt.tzinfo is None:
            return dt.replace(tzinfo=IST).astimezone(UTC)
        return dt.astimezone(UTC)
    return dt


async def extract_world(
    ctx: ReplayDropContext,
    recorded_world: RecordedWorld,
    replay_flags: dict[str, Any],
) -> WorldExtract:
    """
    WHAT: Queries workspace feature_log, account_snapshots, and labels tables before workspace drop.
    WHY: Extracts simulation telemetry and operational decisions into a WorldExtract, reading workspace
    tables and recorded world events without accessing the hidden truth store, dropping real-clock columns.
    """
    # 1. Map payment identifiers to receiving accounts from simulation output and recorded calls
    payment_to_receiver: dict[str, str] = {}
    for pid, ev in zip(
        recorded_world.output.result.payment_ids,
        recorded_world.output.result.run_result.events,
    ):
        payment_to_receiver[pid] = ev.to_account
    for pview, _ in recorded_world.calls:
        payment_to_receiver[pview.planned_payment_id] = pview.to_account

    ws_conn = await asyncpg.connect(ctx.workspace_dsn)
    try:
        # 2. Extract feature_log rows (real-clock column created_at is omitted)
        p_records = await ws_conn.fetch(
            """
            SELECT bank_id, payment_id, decision_time, decided_at, watermark_lag_ms,
                   status, feature_values, missing_request_fields, registry_answers
            FROM feature_log
            ORDER BY decision_time, payment_id
            """
        )
        payment_rows: list[dict[str, Any]] = []
        for r in p_records:
            d = dict(r)
            d.pop("created_at", None)
            d["receiver_account_id"] = payment_to_receiver.get(d["payment_id"])
            payment_rows.append(d)

        # 3. Extract account_snapshots rows (real-clock column logged_at is omitted)
        s_records = await ws_conn.fetch(
            """
            SELECT bank_id, run_id, account_id, snapshot_time, snapshot_kind,
                   arrival_cutoff, feature_set_version, feature_names_sha256, feature_values
            FROM account_snapshots
            ORDER BY snapshot_time, bank_id, account_id, snapshot_kind
            """
        )
        snapshot_rows: list[dict[str, Any]] = []
        for r in s_records:
            d = dict(r)
            d.pop("logged_at", None)
            snapshot_rows.append(d)

        # 4. Extract labels rows (real-clock column recorded_at is omitted)
        l_records = await ws_conn.fetch(
            """
            SELECT run_id, bank_id, label_id, source, account_id, payment_ids,
                   verdict, arrival_time, region, batch_id, claimed_type, via
            FROM labels
            ORDER BY arrival_time, label_id
            """
        )
        label_rows: list[dict[str, Any]] = []
        for r in l_records:
            d = dict(r)
            d.pop("recorded_at", None)
            label_rows.append(d)

        # 4b. Label status changes (revoked / contested); the labels rows themselves are never edited
        s_records = await ws_conn.fetch(
            """
            SELECT bank_id, source, label_id, account_id, change, arrival_time
            FROM label_status_changes
            ORDER BY arrival_time, id
            """
        )
        label_status_rows = [dict(r) for r in s_records]
    finally:
        await ws_conn.close()

    # 5. Extract metadata from recorded_world
    run_id = recorded_world.output.result.run_id
    world_id = recorded_world.world_id or run_id
    split = recorded_world.split or "training"
    cs = recorded_world.output.plan.clock.start_date
    clock_start = cs.isoformat() if hasattr(cs, "isoformat") else str(cs)
    seed = (
        recorded_world.seed
        if recorded_world.seed is not None
        else (
            recorded_world.output.world.seed
            if recorded_world.output.world is not None
            else 0
        )
    )
    gang_families: tuple[str, ...] = (
        tuple(f for f in recorded_world.gang_families if f is not None)
        if recorded_world.gang_families is not None
        else ()
    )
    decisions_per_event = bool(replay_flags.get("decisions_per_event_payment", False))
    reg_wired = bool(replay_flags.get("registry_wired", False))
    reg_reason = str(
        replay_flags.get(
            "registry_wired_reason",
            "Registry wiring deferred until Level 17 (pre-decision registry answering)",
        )
    )

    return WorldExtract(
        world_id=world_id,
        split=split,
        seed=seed,
        run_id=run_id,
        gang_families=gang_families,
        decisions_per_event_payment=decisions_per_event,
        registry_wired=reg_wired,
        registry_wired_reason=reg_reason,
        payment_rows=payment_rows,
        snapshot_rows=snapshot_rows,
        label_rows=label_rows,
        label_status_rows=label_status_rows,
        clock_start=clock_start,
    )


def _canonical_table_bytes(table: pa.Table, sort_keys: tuple[str, ...]) -> bytes:
    """Serializes a PyArrow table to canonical JSON bytes sorted by sort_keys and column names."""
    rows = table.to_pylist()

    def sort_key_fn(r: dict[str, Any]) -> tuple[Any, ...]:
        return tuple(
            r[k].isoformat() if isinstance(r.get(k), (datetime, date)) else (r.get(k) or "")
            for k in sort_keys
        )

    rows.sort(key=sort_key_fn)
    canonical_rows: list[dict[str, Any]] = []
    for r in rows:
        c_row: dict[str, Any] = {}
        for k in sorted(r.keys()):
            val = r[k]
            if isinstance(val, (datetime, date)):
                c_row[k] = val.isoformat()
            else:
                c_row[k] = val
        canonical_rows.append(c_row)

    return orjson.dumps(canonical_rows, option=orjson.OPT_SORT_KEYS)


def content_hash(dataset: Dataset) -> str:
    """
    WHAT: Computes canonical SHA-256 digest across payments, snapshots, and labels tables.
    WHY: R6: Deterministic dataset fingerprint invariant to row order and devoid of wall-clock columns.
    """
    hasher = hashlib.sha256()
    hasher.update(b"payments:")
    hasher.update(
        _canonical_table_bytes(
            dataset.payments, ("world_id", "decision_time", "payment_id")
        )
    )
    hasher.update(b"snapshots:")
    hasher.update(
        _canonical_table_bytes(
            dataset.snapshots,
            ("world_id", "snapshot_time", "bank_id", "account_id", "snapshot_kind"),
        )
    )
    hasher.update(b"labels:")
    hasher.update(
        _canonical_table_bytes(
            dataset.labels, ("world_id", "arrival_time", "label_id", "account_id")
        )
    )
    return hasher.hexdigest()


def _labels_schema() -> pa.Schema:
    return pa.schema(
        [
            ("world_id", pa.string()),
            ("run_id", pa.string()),
            ("bank_id", pa.string()),
            ("label_id", pa.string()),
            ("source", pa.string()),
            ("account_id", pa.string()),
            ("payment_ids", pa.list_(pa.string())),
            ("verdict", pa.string()),
            ("arrival_time", pa.timestamp("us", tz="UTC")),
            ("region", pa.int32()),
            ("batch_id", pa.string()),
            ("claimed_type", pa.string()),
            ("via", pa.string()),
        ]
    )


def _snapshots_schema() -> pa.Schema:
    fields = [
        ("world_id", pa.string()),
        ("run_id", pa.string()),
        ("bank_id", pa.string()),
        ("account_id", pa.string()),
        ("snapshot_time", pa.timestamp("us", tz="UTC")),
        ("snapshot_kind", pa.string()),
        ("arrival_cutoff", pa.timestamp("us", tz="UTC")),
        ("feature_set_version", pa.string()),
        ("feature_names_sha256", pa.string()),
    ]
    for feat in account_feature_names():
        fields.append((feat, pa.float64()))
    return pa.schema(fields)


def _payments_schema() -> pa.Schema:
    fields = [
        ("world_id", pa.string()),
        ("bank_id", pa.string()),
        ("payment_id", pa.string()),
        ("receiver_account_id", pa.string()),
        ("decision_time", pa.timestamp("us", tz="UTC")),
        ("decided_at", pa.timestamp("us", tz="UTC")),
        ("watermark_lag_ms", pa.int64()),
        ("status", pa.string()),
    ]
    for feat in canonical_feature_names():
        fields.append((feat, pa.float64()))
    fields.append(("label_receiver_reported_by_cutoff", pa.bool_()))
    fields.append(("label_payment_named_by_cutoff", pa.bool_()))
    return pa.schema(fields)


def build_dataset(
    extracts: Sequence[WorldExtract],
    split: str,
    cutoff: datetime,
    code_version: str | None = None,
) -> Dataset:
    """
    WHAT: Builds a point-in-time dataset from single-world replay extracts filtered by a world-clock cutoff.
    WHY: Implements rules R1-R8 for reproducible training and validation datasets.

    Args:
        extracts: Sequence of WorldExtract instances.
        split: Target dataset split ('training', 'validation', 'test', 'stress').
        cutoff: Simulation time cutoff; events occurring strictly after this timestamp are excluded.
        code_version: Optional code version fingerprint recorded in manifest.

    Returns:
        A Dataset containing PyArrow tables for payments, snapshots, labels, and the metadata manifest.

    Raises:
        ValueError: On empty extracts, split mismatches, replay without event payment decisions,
                    or inclusion of held-out tactic families in training datasets.
    """
    if not extracts:
        raise ValueError("Cannot build dataset from empty extracts sequence")

    cutoff_utc = _ensure_utc(cutoff)

    # 1. Validation and refusals (R7)
    for e in extracts:
        if e.split != split:
            raise ValueError(
                f"Extract split {e.split!r} does not match requested dataset split {split!r}"
            )
        if not e.decisions_per_event_payment:
            raise ValueError(
                f"Extract for world {e.world_id!r} was not run with decisions_per_event_payment=True"
            )

    if split == "training":
        eval_cfg = load_yaml(Path("config/evaluation.yaml"))
        held_out_tactics = set(eval_cfg.get("held_out_tactics", {}).get("tactics", []))
        for e in extracts:
            for fam in e.gang_families:
                if fam in held_out_tactics:
                    raise ValueError(
                        f"Training dataset refused: extract for world {e.world_id!r} contains "
                        f"held-out tactic family {fam!r}"
                    )

    # 2. Extract and filter labels by cutoff (R2)
    labels_rows: list[dict[str, Any]] = []
    reported_payment_ids: set[str] = set()
    reported_accounts: set[str] = set()

    revoked_labels: set[tuple[str, str, str, str, str]] = set()
    contested_labels: set[tuple[str, str, str, str, str]] = set()
    for e in extracts:
        for s in e.label_status_rows:
            if _ensure_utc(s["arrival_time"]) <= cutoff_utc:  # a change after the cutoff does not exist yet
                status_key = (e.world_id, s["bank_id"], s["source"], s["label_id"], s["account_id"])
                (revoked_labels if s["change"] == "revoked" else contested_labels).add(status_key)
    revoked_excluded = 0

    for e in extracts:
        for r in e.label_rows:
            arr_time = _ensure_utc(r["arrival_time"])
            if arr_time <= cutoff_utc:
                if (e.world_id, r["bank_id"], r["source"], r["label_id"], r["account_id"]) in revoked_labels:
                    revoked_excluded += 1
                    continue
                p_ids = list(r.get("payment_ids") or [])
                l_dict = {
                    "world_id": e.world_id,
                    "run_id": r["run_id"],
                    "bank_id": r["bank_id"],
                    "label_id": r["label_id"],
                    "source": r["source"],
                    "account_id": r["account_id"],
                    "payment_ids": p_ids,
                    "verdict": r["verdict"],
                    "arrival_time": arr_time,
                    "region": r.get("region"),
                    "batch_id": r.get("batch_id"),
                    "claimed_type": r.get("claimed_type"),
                    "via": r.get("via"),
                }
                labels_rows.append(l_dict)
                reported_accounts.add(r["account_id"])
                for pid in p_ids:
                    reported_payment_ids.add(pid)

    # 3. Extract and filter payments by cutoff (R2, R3, R5)
    payments_rows: list[dict[str, Any]] = []
    canon_names = canonical_feature_names()
    no_watermark_count = 0

    for e in extracts:
        for r in e.payment_rows:
            dec_time = _ensure_utc(r["decision_time"])
            if dec_time <= cutoff_utc:
                if r.get("status") == "no_watermark":
                    no_watermark_count += 1

                if "receiver_account_id" not in r or r["receiver_account_id"] is None:
                    raise ValueError(
                        f"Payment row {r.get('payment_id')!r} is missing required 'receiver_account_id'"
                    )
                rec_acc = str(r["receiver_account_id"])
                p_dict: dict[str, Any] = {
                    "world_id": e.world_id,
                    "bank_id": r["bank_id"],
                    "payment_id": r["payment_id"],
                    "receiver_account_id": rec_acc,
                    "decision_time": dec_time,
                    "decided_at": _ensure_utc(r.get("decided_at")),
                    "watermark_lag_ms": r.get("watermark_lag_ms"),
                    "status": r.get("status"),
                }

                fv = r.get("feature_values")
                if isinstance(fv, str):
                    fv = orjson.loads(fv)
                if isinstance(fv, list):
                    for name, val in zip(canon_names, fv):
                        p_dict[name] = val
                elif isinstance(fv, dict):
                    for name in canon_names:
                        p_dict[name] = fv.get(name)
                else:
                    for name in canon_names:
                        p_dict[name] = None

                # Convenience columns (R3)
                p_dict["label_receiver_reported_by_cutoff"] = (
                    rec_acc in reported_accounts if rec_acc else False
                )
                p_dict["label_payment_named_by_cutoff"] = (
                    r["payment_id"] in reported_payment_ids
                )
                payments_rows.append(p_dict)

    # 4. Extract and filter snapshots by cutoff (R2)
    snapshots_rows: list[dict[str, Any]] = []
    acct_names = account_feature_names()

    for e in extracts:
        for r in e.snapshot_rows:
            snap_time = _ensure_utc(r["snapshot_time"])
            if snap_time <= cutoff_utc:
                s_dict: dict[str, Any] = {
                    "world_id": e.world_id,
                    "run_id": r["run_id"],
                    "bank_id": r["bank_id"],
                    "account_id": r["account_id"],
                    "snapshot_time": snap_time,
                    "snapshot_kind": r["snapshot_kind"],
                    "arrival_cutoff": _ensure_utc(r["arrival_cutoff"]),
                    "feature_set_version": r["feature_set_version"],
                    "feature_names_sha256": r["feature_names_sha256"],
                }
                sfv = r.get("feature_values")
                if isinstance(sfv, str):
                    sfv = orjson.loads(sfv)
                if isinstance(sfv, list):
                    for name, val in zip(acct_names, sfv):
                        s_dict[name] = val
                elif isinstance(sfv, dict):
                    for name in acct_names:
                        s_dict[name] = sfv.get(name)
                else:
                    for name in acct_names:
                        s_dict[name] = None
                snapshots_rows.append(s_dict)

    # 5. Feature columns list and None share computation (R4, R5)
    feature_cols = list(canon_names)
    for col in feature_cols:
        if col.startswith("label_"):
            raise ValueError(f"Feature column {col!r} begins with prohibited 'label_' prefix")

    none_shares: dict[str, float] = {}
    total_p = len(payments_rows)
    for col in feature_cols:
        count_none = sum(1 for row in payments_rows if row.get(col) is None)
        none_shares[col] = (count_none / total_p) if total_p > 0 else 0.0

    # 6. Build PyArrow tables
    if payments_rows:
        payments_table = pa.Table.from_pylist(payments_rows)
    else:
        payments_table = pa.Table.from_pylist([], schema=_payments_schema())

    if snapshots_rows:
        snapshots_table = pa.Table.from_pylist(snapshots_rows)
    else:
        snapshots_table = pa.Table.from_pylist([], schema=_snapshots_schema())

    if labels_rows:
        labels_table = pa.Table.from_pylist(labels_rows)
    else:
        labels_table = pa.Table.from_pylist([], schema=_labels_schema())

    # 7. Build Manifest (R8)
    if code_version is None:
        try:
            runner_cfg = load_runner_config()
            code_ver = find_code_version(
                os.environ,
                REPO_ROOT,
                runner_cfg["code_version_git_timeout_seconds"],
            )
        except Exception:
            code_ver = "unknown"
    else:
        code_ver = code_version

    manifest: dict[str, Any] = {
        "split": split,
        "cutoff": cutoff_utc.isoformat(),
        "registry_wired": all(e.registry_wired for e in extracts),
        "registry_wired_reason": (
            extracts[0].registry_wired_reason if extracts else ""
        ),
        "code_version": code_ver,
        "content_hash": "",
        "row_counts": {
            "payments": len(payments_table),
            "snapshots": len(snapshots_table),
            "labels": len(labels_table),
        },
        "no_watermark_count": no_watermark_count,
        "feature_columns": feature_cols,
        "feature_none_shares": none_shares,
        "labels_note": "False complaints count as reported by cutoff.",
        "labels_revoked_excluded": revoked_excluded,
        "labels_contested": sorted(":".join(k) for k in contested_labels - revoked_labels),
        "label_maturity_note": "Label maturity near cutoff contains unadjusted reporting lag.",
        "worlds": [
            {
                "world_id": e.world_id,
                "split": e.split,
                "seed": e.seed,
                "run_id": e.run_id,
                "clock_start": e.clock_start,
                "gang_families": list(e.gang_families),
            }
            for e in extracts
        ],
        "files": {},
    }

    dataset = Dataset(
        payments=payments_table,
        snapshots=snapshots_table,
        labels=labels_table,
        manifest=manifest,
    )
    chash = content_hash(dataset)
    dataset.manifest["content_hash"] = chash
    return dataset


def write_dataset(
    dataset: Dataset,
    root: Path = Path("data/datasets"),
) -> Path:
    """
    WHAT: Serializes a Dataset into Parquet tables and a manifest.json under a canonical directory name.
    WHY: Implements persistent storage for offline ML artifacts (<split>-<cutoff yyyymmddTHHMM>-<hash[:12]>).
    """
    split = dataset.manifest["split"]
    cutoff_dt = datetime.fromisoformat(dataset.manifest["cutoff"])
    cutoff_str = cutoff_dt.strftime("%Y%m%dT%H%M")
    chash = dataset.manifest["content_hash"]
    dir_name = f"{split}-{cutoff_str}-{chash[:12]}"
    dir_path = root / dir_name
    dir_path.mkdir(parents=True, exist_ok=True)

    # 1. Write Parquet tables
    pq.write_table(dataset.payments, dir_path / "payments.parquet")
    pq.write_table(dataset.snapshots, dir_path / "snapshots.parquet")
    pq.write_table(dataset.labels, dir_path / "labels.parquet")

    # 2. Compute file hashes and update manifest
    files_dict: dict[str, str] = {}
    for fname in ("payments.parquet", "snapshots.parquet", "labels.parquet"):
        data = (dir_path / fname).read_bytes()
        files_dict[fname] = hashlib.sha256(data).hexdigest()
    dataset.manifest["files"] = files_dict

    # 3. Write manifest.json
    manifest_bytes = orjson.dumps(
        dataset.manifest, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS
    )
    (dir_path / "manifest.json").write_bytes(manifest_bytes)

    return dir_path
