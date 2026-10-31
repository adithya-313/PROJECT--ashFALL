# File: scripts/feature_log_size_estimate.py
# Purpose: Level 15c Round 2b measured feature log and snapshot size estimates per preset.
# Issue: Level 15c sizing estimate (D4).
"""
WHAT: Measures PostgreSQL storage footprints for feature_log and account_snapshots rows
and projects daily and weekly storage requirements across all four simulator presets
(tiny, small, medium, large).

WHY: Decision feature logging and account snapshots consume significant database capacity.
Empirically calibrating per-row column sizes against real database migrations guarantees
rigorous, reproducible storage forecasts for Level 16 training and Level 17 operations.
"""

import asyncio
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

__all__ = [
    "PresetEstimate",
    "compute_preset_estimates",
    "format_estimates_table",
    "measure_database_row_sizes",
    "measure_tiny_world_accounts",
]


@dataclass(frozen=True)
class PresetEstimate:
    """Projections for a single world scale preset."""

    name: str
    n_people: int
    n_shops: int
    n_days: int
    total_payments: int
    decisions_per_day: float
    estimated_accounts: int
    feature_log_mb_per_day: float
    daily_snapshot_mb_per_day: float
    full_snapshot_mb_per_week: float


async def measure_database_row_sizes() -> tuple[float, float]:
    """
    WHAT: Inserts 200 realistic feature_log rows and 200 account_snapshots rows into a
    temporary schema and measures average pg_column_size per row.

    WHY: Ground truth measurement from real PostgreSQL storage engine, avoiding theoretical estimates.
    """
    import asyncpg
    import orjson

    from features.account_snapshots import account_feature_names, account_feature_names_sha256
    from features.feature_log import (
        FeatureLogRow,
        canonical_feature_names,
        canonical_feature_names_sha256,
        insert_feature_log_row,
    )
    from scripts.run_migrations import apply_pending_migrations
    from tests.integration.conftest import (
        create_schema,
        drop_schema,
        get_test_database_url,
        make_schema_name,
    )

    schema_name = make_schema_name()
    database_url = get_test_database_url()
    await create_schema(database_url, schema_name)

    conn: asyncpg.Connection | None = None
    try:
        conn = await asyncpg.connect(
            database_url,
            server_settings={"search_path": schema_name},
        )
        await apply_pending_migrations(conn)

        c_names = canonical_feature_names()
        c_sha = canonical_feature_names_sha256()
        a_names = account_feature_names()
        a_sha = account_feature_names_sha256()

        now = datetime.now(timezone.utc)
        today = now.date()
        await conn.execute("SELECT create_feature_log_partition($1)", today)
        await conn.execute("SELECT create_account_snapshots_partition($1)", today)

        # Realistic feature vector values
        c_vals = [
            1000 if i % 4 == 0 else (0.75 if i % 4 == 1 else (True if i % 4 == 2 else None))
            for i in range(len(c_names))
        ]
        a_vals = [
            1000 if i % 4 == 0 else (0.75 if i % 4 == 1 else (True if i % 4 == 2 else None))
            for i in range(len(a_names))
        ]
        a_json = orjson.dumps(a_vals).decode("utf-8")

        snap_query = """
        INSERT INTO account_snapshots (
            bank_id, run_id, account_id, snapshot_time, snapshot_kind,
            arrival_cutoff, feature_set_version, feature_names_sha256,
            feature_values, logged_at
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb, $10
        )
        """

        for i in range(200):
            pid = f"66db9aa439b092ee-{i:06d}"
            log_row = FeatureLogRow(
                bank_id="bank_0",
                payment_id=pid,
                run_id="66db9aa439b092ee",
                account_id=f"acc_{i:04d}",
                decision_time=now,
                decided_at=now,
                arrival_cutoff=now,
                watermark_lag_ms=120,
                status="ok",
                feature_set_version="m1",
                feature_names_sha256=c_sha,
                feature_values=c_vals,
                missing_request_fields=[],
                registry_answers=[],
            )
            await insert_feature_log_row(conn, log_row)

            await conn.execute(
                snap_query,
                "bank_0",
                "66db9aa439b092ee",
                f"acc_{i:04d}",
                now,
                "daily",
                now,
                "m1",
                a_sha,
                a_json,
                now,
            )

        f_size = await conn.fetchval("SELECT avg(pg_column_size(t.*)) FROM feature_log t")
        s_size = await conn.fetchval("SELECT avg(pg_column_size(s.*)) FROM account_snapshots s")

        return float(f_size), float(s_size)

    finally:
        if conn is not None:
            await conn.close()
        await drop_schema(database_url, schema_name)


def measure_tiny_world_accounts() -> int:
    """
    WHAT: Measures real account count in the tiny world offline (seed 101, 60 people, 12 shops).

    WHY: Ground-truth reference entity count for scaling account numbers across presets.
    """
    from world.fast_mode import FastRunSettings, run_fast_world

    settings = FastRunSettings(seed=101, n_people=60, n_shops=12, n_days=61, range_name="training")
    out = run_fast_world(settings)
    return len(out.world.people) + len(out.world.shops)


def compute_preset_estimates(
    sizing_config: dict[str, Any],
    profile: Any,
    feature_log_row_bytes: float,
    snapshot_row_bytes: float,
    tiny_accounts: int,
) -> list[PresetEstimate]:
    """
    WHAT: Computes storage sizing forecasts across tiny, small, medium, and large presets.

    Decisions per day = payments / n_days (one decision per payment by sender's bank).
    Accounts = tiny_accounts scaled by (n_people + n_shops) / (60 + 12) [ESTIMATE].
    """
    from world.estimates import estimate_world

    presets = ["tiny", "small", "medium", "large"]
    results: list[PresetEstimate] = []

    for name in presets:
        p_cfg = sizing_config["presets"][name]
        n_people = int(p_cfg["n_people"])
        n_shops = int(p_cfg["n_shops"])
        n_days = int(p_cfg["n_days"])

        est = estimate_world(n_people, n_days, profile, sizing_config)
        payments = est.payments
        decisions_per_day = payments / n_days
        # Tagged ESTIMATE: scaled from measured tiny world accounts
        accounts = math.ceil(tiny_accounts * (n_people + n_shops) / (60 + 12))

        # Size calculations
        log_bytes_per_day = decisions_per_day * feature_log_row_bytes
        f_log_mb_day = log_bytes_per_day / (1024 * 1024)

        # Upper bound: assuming 100% of accounts are active on that day
        daily_snap_bytes_day = accounts * snapshot_row_bytes
        daily_snap_mb_day = daily_snap_bytes_day / (1024 * 1024)

        # Full snapshot once per week across all accounts
        full_snap_bytes_week = accounts * snapshot_row_bytes
        full_snap_mb_week = full_snap_bytes_week / (1024 * 1024)

        results.append(
            PresetEstimate(
                name=name,
                n_people=n_people,
                n_shops=n_shops,
                n_days=n_days,
                total_payments=payments,
                decisions_per_day=decisions_per_day,
                estimated_accounts=accounts,
                feature_log_mb_per_day=f_log_mb_day,
                daily_snapshot_mb_per_day=daily_snap_mb_day,
                full_snapshot_mb_per_week=full_snap_mb_week,
            )
        )

    return results


def format_estimates_table(
    estimates: list[PresetEstimate],
    feature_log_row_bytes: float,
    snapshot_row_bytes: float,
    tiny_accounts: int,
) -> str:
    """Formats markdown summary table with explicit derived_from citations for each number."""
    lines: list[str] = [
        "=" * 97,
        "LEVEL 15c FEATURE LOG & ACCOUNT SNAPSHOTS SIZE ESTIMATES",
        "=" * 97,
        f"Measured feature_log row size:        {feature_log_row_bytes:.2f} bytes (avg pg_column_size over 200 rows)",
        f"Measured account_snapshots row size:  {snapshot_row_bytes:.2f} bytes (avg pg_column_size over 200 rows)",
        f"Measured tiny world accounts offline: {tiny_accounts} accounts (60 people + 12 shops)",
        "",
        (
            "| Preset | People / Shops | Horizon | Payments | Decisions/Day | "
            "Accounts [ESTIMATE] | Feature Log (MB/day) | Daily Snapshot (MB/day) | Full Snapshot (MB/week) |"
        ),
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for e in estimates:
        lines.append(
            f"| {e.name:<6} | {e.n_people}/{e.n_shops:<5} | {e.n_days}d     | {e.total_payments:<8} | "
            f"{e.decisions_per_day:<13.1f} | {e.estimated_accounts:<19} | {e.feature_log_mb_per_day:<20.2f} | "
            f"{e.daily_snapshot_mb_per_day:<23.2f} | {e.full_snapshot_mb_per_week:<23.2f} |"
        )

    lines.extend(
        [
            "",
            "DERIVATIONS & RATIONALE:",
            "  * Payments: derived_from estimate_world(n_people, n_days, profile, sizing_config) with margin 1.15.",
            (
                "  * Decisions/Day: derived_from payments / n_days "
                "(exactly one fraud scoring decision per payment by sender's bank)."
            ),
            (
                "  * Accounts: derived_from 72 measured offline accounts in tiny world, "
                "scaled by (n_people + n_shops) / (60 + 12) [TAG: ESTIMATE]."
            ),
            "  * Feature Log (MB/day): derived_from (decisions_per_day * measured_feature_log_bytes) / 1048576.",
            (
                "  * Daily Snapshot (MB/day): derived_from (accounts * measured_snapshot_bytes) / 1048576 "
                "(upper bound: all accounts active)."
            ),
            (
                "  * Full Snapshot (MB/week): derived_from (accounts * measured_snapshot_bytes) / 1048576 "
                "(weekly cadence = full_interval_days 7)."
            ),
            "=" * 97,
        ]
    )
    return "\n".join(lines)


async def main_async() -> None:
    """Async main routine executing calibration, measurement, and formatting."""
    from world.estimates import calibrate_machine, read_sizing_config

    sizing_config = read_sizing_config()
    profile = calibrate_machine(sizing_config)

    f_bytes, s_bytes = await measure_database_row_sizes()
    tiny_accounts = measure_tiny_world_accounts()

    estimates = compute_preset_estimates(
        sizing_config=sizing_config,
        profile=profile,
        feature_log_row_bytes=f_bytes,
        snapshot_row_bytes=s_bytes,
        tiny_accounts=tiny_accounts,
    )

    table_text = format_estimates_table(
        estimates=estimates,
        feature_log_row_bytes=f_bytes,
        snapshot_row_bytes=s_bytes,
        tiny_accounts=tiny_accounts,
    )
    print(table_text)


def main() -> None:
    """Main CLI entrypoint."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
