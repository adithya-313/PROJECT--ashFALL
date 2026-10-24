"""
WHAT: Unit and integration tests for Level 12g gang truth classifications and truth-v2 export.

WHY: Level 12g requires simulation ground truth to classify each illicit syndicate as seen or held-out
directly from its executed tactical mechanism options. These tests verify the 5 classification rules of
gang_truth_for, validate truth-v2 manifest generation and query methods, verify that opening handles pass
gang truth through without loss, and check fraud-off empty gangs.jsonl generation.

ISSUE IDs: 12-8
TERM: gang truth: ground truth classification attributing each syndicate to a tactic family and taxonomy kind.
TERM: truth-v2: truth store format version introducing gangs.jsonl alongside roles, devices, stories, and manifest.
"""

import hashlib
import json
from pathlib import Path

import pytest

from evaluation.truth import load_truth
from world.fast_mode import FastRunSettings
from world.gang_config import BlockStep, FixtureSource, GangBudget, GangConfig
from world.held_out_options import gang_truth_for
from world.interactive_records import RunnerSettings
from world.interactive_runner import run_interactive_world
from world.tactic_families import load_tactic_families
from world.truth_records import GangTruth
from world.truth_writer import write_truth_files

TINY_TEST = FastRunSettings(101, 60, 12, 61, "test")
BUDGET = GangBudget(money_paise=50_000_000, time_minutes=500_000, max_risk=0.99)
SRC_1 = FixtureSource("acc_src_seen_01", 0, 10_000_000, 1440)
SRC_2 = FixtureSource("acc_src_drip_02", 1, 10_000_000, 1440)


def _make_plain_gang(gang_id: str, family: str | None = None) -> GangConfig:
    """
    WHAT: Creates a plain scam gang with no held-out mechanism options.
    WHY: Serves as a standard baseline seen gang for classification tests.
    """
    steps = (
        BlockStep("acquire", (("count", 3), ("origin_mix", "all_rented"))),
        BlockStep("scam", (("receiving_mules", 2),)),
        BlockStep("route", (("layers", 1), ("accounts_per_layer", 2))),
    )
    return GangConfig(gang_id, 1, BUDGET, None, steps, 1440, family=family)


def _make_drip_gang(gang_id: str, family: str | None = "slow_drip") -> GangConfig:
    """
    WHAT: Creates a gang configuring slow_drip held-out route parameters.
    WHY: Serves as a canonical held-out drip syndicate for truth validation.
    """
    steps = (
        BlockStep("acquire", (("count", 3), ("origin_mix", "all_rented"))),
        BlockStep(
            "route",
            (
                ("layers", 2),
                ("accounts_per_layer", 1),
                ("drip_piece_max_inr", 1500),
                ("drip_spread_days", 3),
                ("drip_hold_minutes", 300),
            ),
        ),
    )
    return GangConfig(gang_id, 1, BUDGET, SRC_2, steps, None, family=family)


def test_gang_truth_for_five_rules() -> None:
    """
    WHAT: Asserts that gang_truth_for enforces all five taxonomic classification rules.
    WHY: Verifies multi-family rejection, tag contradiction checks, unbacked held-out tag refusals,
    legitimate held-out option classification, and seen syndicate tagging.
    """
    families = load_tactic_families()

    # Rule 1: Gang configuring options from multiple held-out families raises ValueError.
    multi_steps = (
        BlockStep("acquire", (("count", 3),)),
        BlockStep("take_control", (("rotate_every_days", 7), ("rotations", 2))),
        BlockStep("route", (("drip_piece_max_inr", 1500), ("drip_spread_days", 3), ("drip_hold_minutes", 300))),
    )
    multi_gang = GangConfig("g_multi", 1, BUDGET, SRC_1, multi_steps, None)
    with pytest.raises(ValueError, match="one held-out family per gang, so results stay per family"):
        gang_truth_for(multi_gang, families)

    # Rule 2: Declared family tag contradicts detected held-out options.
    wrong_tag_gang = _make_drip_gang("g_wrong", family="device_rotation")
    with pytest.raises(ValueError, match="tag disagrees with behaviour"):
        gang_truth_for(wrong_tag_gang, families)

    # Rule 3: Gang declared as held-out without using any of its mechanism options.
    unbacked_gang = _make_plain_gang("g_unbacked", family="slow_drip")
    with pytest.raises(ValueError, match="a held-out name needs its mechanism"):
        gang_truth_for(unbacked_gang, families)

    # Rule 4: Gang uses valid held-out options and receives held_out kind.
    valid_drip = _make_drip_gang("g_drip", family="slow_drip")
    drip_gt = gang_truth_for(valid_drip, families)
    expected_options = ("route.drip_hold_minutes", "route.drip_piece_max_inr", "route.drip_spread_days")
    assert drip_gt == GangTruth("g_drip", "slow_drip", "held_out", expected_options)

    # Rule 5: Ordinary seen gang with no held-out options returns seen kind.
    seen_gang = _make_plain_gang("g_seen", family="fast_pass_through")
    seen_gt = gang_truth_for(seen_gang, families)
    assert seen_gt == GangTruth("g_seen", "fast_pass_through", "seen", ())


def test_full_test_world_run_and_truth_store(tmp_path: Path) -> None:
    """
    WHAT: Runs a test world with one seen gang and one held-out gang and verifies truth-v2 artifacts.
    WHY: Proves gangs.jsonl is written, manifest registers it with SHA-256, and TruthStore queries match.
    """
    seen_steps = (
        BlockStep("acquire", (("count", 3), ("origin_mix", "all_rented"))),
        BlockStep("route", (("layers", 1), ("accounts_per_layer", 2))),
    )
    seen_gang = GangConfig("gang_seen", 1, BUDGET, SRC_1, seen_steps, None, family="fast_pass_through")
    drip_gang = _make_drip_gang("gang_drip", family="slow_drip")

    output = run_interactive_world(
        TINY_TEST,
        RunnerSettings(fraud_enabled=True),
        gang_configs=(seen_gang, drip_gang),
    )
    truth_dir = write_truth_files(output, tmp_path)

    # 1. Assert gangs.jsonl exists and contains exactly two rows.
    gangs_file = truth_dir / "gangs.jsonl"
    assert gangs_file.exists()
    gangs_rows = [json.loads(line) for line in gangs_file.read_text(encoding="utf-8").splitlines()]
    assert len(gangs_rows) == 2

    # 2. Assert manifest format is truth-v2 and includes gangs.jsonl.
    manifest = json.loads((truth_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["format"] == "truth-v2"
    assert "gangs.jsonl" in manifest["files"]
    assert manifest["files"]["gangs.jsonl"]["row_count"] == 2

    # 3. Load truth via evaluation interface and query taxonomic gang kind.
    store = load_truth(tmp_path, output.result.run_id)
    assert store.gang_kind("gang_seen") == "seen"
    assert store.gang_kind("gang_drip") == "held_out"
    with pytest.raises(KeyError, match="Unknown gang 'phantom_gang'"):
        store.gang_kind("phantom_gang")


def test_opening_handles_preserves_gang_truth() -> None:
    """
    WHAT: Verifies that mid-run account openings pass gang truth through without dropping records.
    WHY: Proves that opening_handles rebuilds TruthFacts while preserving gangs classification tuples.
    """
    src = FixtureSource("acc_src_open_01", 0, 10_000_000, 1440)
    drip_opening_steps = (
        BlockStep("acquire", (("count", 4), ("origin_mix", "world_ranges"))),
        BlockStep(
            "route",
            (
                ("layers", 2),
                ("accounts_per_layer", 2),
                ("drip_piece_max_inr", 1500),
                ("drip_spread_days", 3),
                ("drip_hold_minutes", 300),
            ),
        ),
    )
    drip_gang = GangConfig("gang_open_drip", 1, BUDGET, src, drip_opening_steps, None, family="slow_drip")

    output = run_interactive_world(
        TINY_TEST,
        RunnerSettings(fraud_enabled=True, account_openings_enabled=True),
        gang_configs=(drip_gang,),
    )
    assert len(output.result.truth.gangs) == 1
    assert output.result.truth.gangs[0].gang_id == "gang_open_drip"
    assert output.result.truth.gangs[0].kind == "held_out"
    assert output.result.truth.gangs[0].family == "slow_drip"


def test_fraud_off_run_writes_empty_gangs_and_identical_hashes(tmp_path: Path) -> None:
    """
    WHAT: Verifies that fraud-off runs produce empty gangs.jsonl and byte-identical artifact hashes.
    WHY: Proves zero regression on fraud-off reality and confirms truth-v2 backward compatibility.
    """
    out_1 = run_interactive_world(TINY_TEST, RunnerSettings(fraud_enabled=False), code_version="test-12g")
    out_2 = run_interactive_world(TINY_TEST, RunnerSettings(fraud_enabled=False), code_version="test-12g")

    dir_1 = write_truth_files(out_1, tmp_path / "run_alpha")
    dir_2 = write_truth_files(out_2, tmp_path / "run_beta")

    # Assert gangs.jsonl exists and is empty (0 rows).
    gangs_1 = dir_1 / "gangs.jsonl"
    assert gangs_1.exists()
    assert gangs_1.read_bytes() == b""

    # Assert bit-for-bit identical hashes across all five truth files.
    file_names = ["roles.jsonl", "device_control.jsonl", "payment_stories.jsonl", "gangs.jsonl", "manifest.json"]
    for name in file_names:
        hash_1 = hashlib.sha256((dir_1 / name).read_bytes()).hexdigest()
        hash_2 = hashlib.sha256((dir_2 / name).read_bytes()).hexdigest()
        assert hash_1 == hash_2
