# File: tests/unit/test_dataset_builder.py
# Purpose: Unit tests for dataset builder (tables, cutoffs, convenience columns, manifest, content hash, refusals).
"""
WHAT: Unit tests covering rules R1-R8 for dataset_builder, plus D1b empty step check and recording defaults.
WHY: Level 16c-2 / Architect decisions Q2-Q5:
  - R1: Multi-world concatenation.
  - R2: Cutoff filtering (decision_time <= C, snapshot_time <= C, arrival_time <= C).
  - R3: Convenience label columns (reported receiver, named payment, false complaints).
  - R4: Feature columns exclusion of label prefixes.
  - R5: Preservation of no_watermark rows and feature None shares.
  - R6: Deterministic canonical content hash.
  - R7: Refusal on decisions_per_event_payment=False, held-out tactics in training, split mismatch.
  - R8: Manifest metadata fidelity and parquet file persistence.
  - D1b: run_replay raises ValueError on empty step list.
  - record_world_for_replay: default keyword arguments preserve gang_configs=().
"""

import hashlib
from datetime import datetime, timezone
from inspect import signature
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from features.account_snapshots import account_feature_names
from features.feature_log import canonical_feature_names
from replay.dataset_builder import (
    WorldExtract,
    build_dataset,
    write_dataset,
)
from world.replay_recording import record_world_for_replay

UTC = timezone.utc


def _make_dummy_extract(
    world_id: str = "w1",
    split: str = "training",
    seed: int = 101,
    run_id: str = "run_abc123",
    gang_families: tuple[str, ...] = ("fast_pass_through",),
    decisions_per_event_payment: bool = True,
    registry_wired: bool = False,
    registry_wired_reason: str = "Registry wiring deferred until Level 17",
    payment_rows: list[dict[str, Any]] | None = None,
    snapshot_rows: list[dict[str, Any]] | None = None,
    label_rows: list[dict[str, Any]] | None = None,
    clock_start: str = "2026-09-01",
) -> WorldExtract:
    canon = canonical_feature_names()
    acct = account_feature_names()

    if payment_rows is None:
        p1_feats: dict[str, Any] = {
            name: (1.0 if idx % 2 == 0 else 0.0) for idx, name in enumerate(canon)
        }
        p1_feats["payee_registry_match"] = None
        p1 = {
            "bank_id": "bank_0",
            "payment_id": "p1",
            "receiver_account_id": "acc_receiver_1",
            "decision_time": datetime(2026, 9, 10, 10, 0, 0, tzinfo=UTC),
            "decided_at": datetime(2026, 9, 10, 10, 0, 0, tzinfo=UTC),
            "watermark_lag_ms": 100,
            "status": "ok",
            "feature_values": p1_feats,
        }
        payment_rows = [p1]

    if snapshot_rows is None:
        s1 = {
            "bank_id": "bank_0",
            "run_id": run_id,
            "account_id": "acc_receiver_1",
            "snapshot_time": datetime(2026, 9, 10, 0, 0, 0, tzinfo=UTC),
            "snapshot_kind": "daily",
            "arrival_cutoff": datetime(2026, 9, 10, 0, 0, 0, tzinfo=UTC),
            "feature_set_version": "v1",
            "feature_names_sha256": "sha123",
            "feature_values": {name: 0.0 for name in acct},
        }
        snapshot_rows = [s1]

    if label_rows is None:
        l1 = {
            "run_id": run_id,
            "bank_id": "bank_0",
            "label_id": f"{run_id}-L001",
            "source": "complaint",
            "account_id": "acc_receiver_1",
            "payment_ids": ["p1"],
            "verdict": "fraud",
            "arrival_time": datetime(2026, 9, 10, 12, 0, 0, tzinfo=UTC),
            "region": 1,
            "batch_id": None,
            "claimed_type": None,
        }
        label_rows = [l1]

    return WorldExtract(
        world_id=world_id,
        split=split,
        seed=seed,
        run_id=run_id,
        gang_families=gang_families,
        decisions_per_event_payment=decisions_per_event_payment,
        registry_wired=registry_wired,
        registry_wired_reason=registry_wired_reason,
        payment_rows=payment_rows,
        snapshot_rows=snapshot_rows,
        label_rows=label_rows,
        clock_start=clock_start,
    )


def test_record_world_for_replay_default_parameters() -> None:
    """Verifies that record_world_for_replay retains keyword argument defaults gang_configs=()."""
    sig = signature(record_world_for_replay)
    assert "gang_configs" in sig.parameters
    assert sig.parameters["gang_configs"].default == ()
    assert "world_split" in sig.parameters
    assert sig.parameters["world_split"].default is None


@pytest.mark.anyio
async def test_run_replay_refuses_empty_step_list(monkeypatch: pytest.MonkeyPatch) -> None:
    """D1b: run_replay raises ValueError immediately when steps list is empty."""
    from replay.runner import run_replay
    from world.replay_workspace import ReplayWorkspace

    # Mock build_replay_steps to return empty list
    monkeypatch.setattr("replay.runner.build_replay_steps", lambda *args, **kwargs: [])

    # Mock workspace allocation and lifecycle to isolate unit test from external infra
    ws = ReplayWorkspace(
        name="test_ws",
        postgres_schema="test_schema",
        redis_db=0,
        kafka_topic="test_topic",
        kafka_group_id="test_group",
    )
    monkeypatch.setattr("replay.runner.allocate_workspace", AsyncMock(return_value=ws))
    monkeypatch.setattr("replay.runner.drop_workspace", AsyncMock(return_value=None))

    # Mock DB pool and setup
    mock_conn = AsyncMock()
    mock_pool = MagicMock()
    mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
    mock_pool.close = AsyncMock()
    monkeypatch.setattr("asyncpg.create_pool", AsyncMock(return_value=mock_pool))

    # Mock redis, producer, consumer, topic partition check
    monkeypatch.setattr("replay.runner.workspace_redis_client", MagicMock())
    monkeypatch.setattr("replay.runner.start_producer", AsyncMock(return_value=AsyncMock()))
    mock_consumer = AsyncMock()
    monkeypatch.setattr("replay.runner.AIOKafkaConsumer", MagicMock(return_value=mock_consumer))
    monkeypatch.setattr("replay.runner.check_workspace_topic_partitions", MagicMock())

    dummy_recorded = MagicMock()
    dummy_recorded.output.result.run_id = "test_run_12345"
    dummy_feed = MagicMock()

    with pytest.raises(
        ValueError, match="Cannot replay an empty step list: no replay steps were generated"
    ):
        await run_replay(
            dummy_recorded,
            dummy_feed,
            admin_pg="postgresql://dummy",
            base_redis_url="redis://dummy",
        )


def test_build_dataset_r1_multi_world_concatenation() -> None:
    """R1: Multi-world extracts with same split concatenate rows and preserve world_id."""
    e1 = _make_dummy_extract(world_id="world_1")
    e2 = _make_dummy_extract(world_id="world_2")

    cutoff = datetime(2026, 9, 11, 0, 0, 0, tzinfo=UTC)
    dataset = build_dataset([e1, e2], split="training", cutoff=cutoff)

    assert len(dataset.payments) == 2
    world_ids = dataset.payments["world_id"].to_pylist()
    assert world_ids == ["world_1", "world_2"]
    assert len(dataset.snapshots) == 2
    assert len(dataset.labels) == 2
    assert len(dataset.manifest["worlds"]) == 2


def test_build_dataset_r2_cutoff_filtering() -> None:
    """R2: Rows strictly after the cutoff are excluded; rows at or before cutoff are included."""
    cutoff = datetime(2026, 9, 10, 10, 0, 0, tzinfo=UTC)
    canon = canonical_feature_names()

    # Create 3 payment rows: before, exact, after
    p_before = {
        "bank_id": "b0",
        "payment_id": "p_before",
        "receiver_account_id": "acc_receiver_1",
        "decision_time": datetime(2026, 9, 10, 9, 59, 59, tzinfo=UTC),
        "status": "ok",
        "feature_values": {name: 1.0 for name in canon},
    }
    p_exact = {
        "bank_id": "b0",
        "payment_id": "p_exact",
        "receiver_account_id": "acc_1",
        "decision_time": cutoff,
        "status": "ok",
        "feature_values": {name: 1.0 for name in canon},
    }
    p_after = {
        "bank_id": "b0",
        "payment_id": "p_after",
        "receiver_account_id": "acc_2",
        "decision_time": datetime(2026, 9, 10, 10, 0, 1, tzinfo=UTC),
        "status": "ok",
        "feature_values": {name: 1.0 for name in canon},
    }

    # Create 2 label rows: exact cutoff, after cutoff
    l_exact = {
        "run_id": "run1",
        "bank_id": "b0",
        "label_id": "run1-L1",
        "source": "complaint",
        "account_id": "acc_1",
        "payment_ids": ["p_exact"],
        "verdict": "fraud",
        "arrival_time": cutoff,
    }
    l_after = {
        "run_id": "run1",
        "bank_id": "b0",
        "label_id": "run1-L2",
        "source": "complaint",
        "account_id": "acc_2",
        "payment_ids": ["p_after"],
        "verdict": "fraud",
        "arrival_time": datetime(2026, 9, 10, 10, 0, 1, tzinfo=UTC),
    }

    extract = _make_dummy_extract(
        payment_rows=[p_before, p_exact, p_after],
        label_rows=[l_exact, l_after],
        snapshot_rows=[],
    )

    dataset = build_dataset([extract], split="training", cutoff=cutoff)
    p_ids = set(dataset.payments["payment_id"].to_pylist())
    assert p_ids == {"p_before", "p_exact"}
    l_ids = set(dataset.labels["label_id"].to_pylist())
    assert l_ids == {"run1-L1"}


def test_build_dataset_r3_convenience_columns() -> None:
    """R3: Convenience columns derive properly, including for false complaints."""
    cutoff = datetime(2026, 9, 15, 0, 0, 0, tzinfo=UTC)
    canon = canonical_feature_names()

    # Labels arriving before cutoff:
    # L1: false complaint against acc_reported
    # L2: fraud complaint naming payment p_named
    labels = [
        {
            "run_id": "run1",
            "bank_id": "b0",
            "label_id": "run1-L1",
            "source": "complaint",
            "account_id": "acc_reported",
            "payment_ids": [],
            "verdict": "not_fraud",  # false complaint counts as reported
            "arrival_time": datetime(2026, 9, 10, 0, 0, 0, tzinfo=UTC),
        },
        {
            "run_id": "run1",
            "bank_id": "b0",
            "label_id": "run1-L2",
            "source": "complaint",
            "account_id": "acc_other",
            "payment_ids": ["p_named"],
            "verdict": "fraud",
            "arrival_time": datetime(2026, 9, 12, 0, 0, 0, tzinfo=UTC),
        },
    ]

    payments = [
        {
            "bank_id": "b0",
            "payment_id": "p_rec",
            "receiver_account_id": "acc_reported",
            "decision_time": datetime(2026, 9, 8, 0, 0, 0, tzinfo=UTC),
            "status": "ok",
            "feature_values": {name: 1.0 for name in canon},
        },
        {
            "bank_id": "b0",
            "payment_id": "p_named",
            "receiver_account_id": "acc_clean",
            "decision_time": datetime(2026, 9, 9, 0, 0, 0, tzinfo=UTC),
            "status": "ok",
            "feature_values": {name: 1.0 for name in canon},
        },
        {
            "bank_id": "b0",
            "payment_id": "p_clean",
            "receiver_account_id": "acc_clean",
            "decision_time": datetime(2026, 9, 9, 0, 0, 0, tzinfo=UTC),
            "status": "ok",
            "feature_values": {name: 1.0 for name in canon},
        },
    ]

    extract = _make_dummy_extract(payment_rows=payments, label_rows=labels, snapshot_rows=[])
    dataset = build_dataset([extract], split="training", cutoff=cutoff)

    df = dataset.payments.to_pydict()
    lookup = {df["payment_id"][i]: i for i in range(len(df["payment_id"]))}

    # p_rec receiver was reported (via false complaint)
    i1 = lookup["p_rec"]
    assert df["label_receiver_reported_by_cutoff"][i1] is True
    assert df["label_payment_named_by_cutoff"][i1] is False

    # p_named payment was named in L2
    i2 = lookup["p_named"]
    assert df["label_receiver_reported_by_cutoff"][i2] is False
    assert df["label_payment_named_by_cutoff"][i2] is True

    # p_clean neither
    i3 = lookup["p_clean"]
    assert df["label_receiver_reported_by_cutoff"][i3] is False
    assert df["label_payment_named_by_cutoff"][i3] is False


def test_build_dataset_r4_feature_columns_no_label_prefix() -> None:
    """R4: Manifest feature_columns lists explicit features without any label_ prefixes."""
    extract = _make_dummy_extract()
    dataset = build_dataset([extract], split="training", cutoff=datetime(2026, 9, 15, tzinfo=UTC))

    feature_cols = dataset.manifest["feature_columns"]
    assert len(feature_cols) == len(canonical_feature_names())
    for col in feature_cols:
        assert not col.startswith("label_")


def test_build_dataset_r5_no_watermark_rows_and_none_shares() -> None:
    """R5: Preserves no_watermark rows, counts them, and reports None shares accurately."""
    canon = canonical_feature_names()
    p1 = {
        "bank_id": "b0",
        "payment_id": "p1",
        "receiver_account_id": "acc_receiver_1",
        "decision_time": datetime(2026, 9, 10, tzinfo=UTC),
        "status": "no_watermark",
        "feature_values": {name: None for name in canon},
    }
    p2 = {
        "bank_id": "b0",
        "payment_id": "p2",
        "receiver_account_id": "acc_receiver_1",
        "decision_time": datetime(2026, 9, 10, tzinfo=UTC),
        "status": "ok",
        "feature_values": {name: (1.0 if name != "payee_registry_match" else None) for name in canon},
    }

    extract = _make_dummy_extract(payment_rows=[p1, p2], label_rows=[], snapshot_rows=[])
    dataset = build_dataset([extract], split="training", cutoff=datetime(2026, 9, 15, tzinfo=UTC))

    assert len(dataset.payments) == 2
    assert dataset.manifest["no_watermark_count"] == 1
    # payee_registry_match is None in both rows -> 1.0 share
    assert dataset.manifest["feature_none_shares"]["payee_registry_match"] == 1.0
    # other features are None in 1 of 2 rows -> 0.5 share
    assert dataset.manifest["feature_none_shares"]["distinct_payers_1h"] == 0.5


def test_build_dataset_r6_deterministic_content_hash() -> None:
    """R6: Content hash is deterministic and order-invariant across extracts."""
    e1 = _make_dummy_extract(world_id="w1")
    e2 = _make_dummy_extract(world_id="w2")

    cutoff = datetime(2026, 9, 15, tzinfo=UTC)
    dataset_a = build_dataset([e1, e2], split="training", cutoff=cutoff)
    dataset_b = build_dataset([e2, e1], split="training", cutoff=cutoff)

    assert dataset_a.manifest["content_hash"] == dataset_b.manifest["content_hash"]
    assert len(dataset_a.manifest["content_hash"]) == 64


def test_build_dataset_r7_refusals() -> None:
    """R7: Refuses decisions_per_event_payment=False, held-out tactics in training, split mismatch."""
    cutoff = datetime(2026, 9, 15, tzinfo=UTC)

    # 1. decisions_per_event_payment = False
    e_no_event = _make_dummy_extract(decisions_per_event_payment=False)
    with pytest.raises(ValueError, match="decisions_per_event_payment=True"):
        build_dataset([e_no_event], split="training", cutoff=cutoff)

    # 2. Held-out tactic slow_drip in training split
    e_slow_drip = _make_dummy_extract(gang_families=("slow_drip",))
    with pytest.raises(ValueError, match="held-out tactic family 'slow_drip'"):
        build_dataset([e_slow_drip], split="training", cutoff=cutoff)

    # 3. Held-out tactic device_rotation in training split
    e_dev_rot = _make_dummy_extract(gang_families=("device_rotation",))
    with pytest.raises(ValueError, match="held-out tactic family 'device_rotation'"):
        build_dataset([e_dev_rot], split="training", cutoff=cutoff)

    # 4. Split mismatch
    e_val = _make_dummy_extract(split="validation", gang_families=("fast_pass_through",))
    with pytest.raises(ValueError, match="does not match requested dataset split 'training'"):
        build_dataset([e_val], split="training", cutoff=cutoff)

    # 5. Empty extracts
    with pytest.raises(ValueError, match="empty extracts sequence"):
        build_dataset([], split="training", cutoff=cutoff)


def test_build_dataset_r8_manifest_and_write_dataset(tmp_path: Path) -> None:
    """R8: Manifest fidelity and write_dataset file persistence with verified SHA256 hashes."""
    extract = _make_dummy_extract()
    cutoff = datetime(2026, 9, 15, 14, 30, 0, tzinfo=UTC)
    dataset = build_dataset([extract], split="training", cutoff=cutoff, code_version="test-ver-001")

    assert dataset.manifest["split"] == "training"
    assert dataset.manifest["code_version"] == "test-ver-001"
    assert dataset.manifest["registry_wired"] is False
    assert dataset.manifest["registry_wired_reason"] == "Registry wiring deferred until Level 17"
    assert "False complaints count as reported" in dataset.manifest["labels_note"]
    assert "Label maturity near cutoff" in dataset.manifest["label_maturity_note"]

    # Write to disk
    out_dir = write_dataset(dataset, root=tmp_path)
    assert out_dir.exists()
    assert out_dir.name.startswith("training-20260915T1430-")

    # Verify files on disk
    for fname in ("payments.parquet", "snapshots.parquet", "labels.parquet", "manifest.json"):
        fpath = out_dir / fname
        assert fpath.exists()

    # Check file hashes in manifest match real disk content
    for fname in ("payments.parquet", "snapshots.parquet", "labels.parquet"):
        content = (out_dir / fname).read_bytes()
        expected_sha = hashlib.sha256(content).hexdigest()
        assert dataset.manifest["files"][fname] == expected_sha


def test_build_dataset_manifest_worlds_carry_clock_start() -> None:
    """Verifies that each world entry in manifest['worlds'] records clock_start."""
    e = _make_dummy_extract(world_id="w_custom_clock", clock_start="2026-10-01")
    cutoff = datetime(2026, 10, 15, tzinfo=UTC)
    dataset = build_dataset([e], split="training", cutoff=cutoff)

    assert len(dataset.manifest["worlds"]) == 1
    w_meta = dataset.manifest["worlds"][0]
    assert w_meta["world_id"] == "w_custom_clock"
    assert w_meta["clock_start"] == "2026-10-01"


def test_build_dataset_missing_receiver_account_raises_value_error() -> None:
    """Verifies that payment rows missing receiver_account_id fail fast with ValueError."""
    canon = canonical_feature_names()
    p_bad = {
        "bank_id": "b0",
        "payment_id": "p_missing_rec",
        "decision_time": datetime(2026, 9, 10, 10, 0, 0, tzinfo=UTC),
        "status": "ok",
        "feature_values": {name: 1.0 for name in canon},
        # receiver_account_id is omitted
    }
    extract = _make_dummy_extract(payment_rows=[p_bad])
    cutoff = datetime(2026, 9, 15, tzinfo=UTC)

    with pytest.raises(
        ValueError,
        match="Payment row 'p_missing_rec' is missing required 'receiver_account_id'",
    ):
        build_dataset([extract], split="training", cutoff=cutoff)

