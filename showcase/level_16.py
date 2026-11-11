# File: showcase/level_16.py
# Purpose: Level 16 showcase: labels, replay datasets and truth kept apart (python -m showcase --level 16).
"""
WHAT: Level 16 turns a replayed world into labelled datasets. This showcase records one committed world
(config/showcase_level_16.yaml), replays it twice with the in-process registry wired, and runs 13 checks on the result.

WHY: To show with real counts that labels reach the right banks by all four paths, that datasets hold only what was
known at their cutoff, that truth stays outside the builder, and that two replays agree.
"""

from __future__ import annotations

from typing import Any

from core.config import config
from showcase.framework import CheckDefinition
from showcase.isolation import get_showcase_database_url
from showcase.level_16_support import host_side_redis_url
from showcase.preconditions import check_database, check_redis, check_tcp_port

LEVEL_NUMBER = 16
LEVEL_TITLE = "Labels, replay datasets and hidden truth kept apart"
NEEDS_DATABASE = True

WHAT_YOU_SAW = (
    "Level 16 replays a recorded world through the real bank pipeline and packages what banks knew into datasets. "
    "The showcase world (40 people, 90 days, one scam gang) holds a police crackdown, a complaint whose payee is at "
    "another bank, a recycled phone number listed before day 0, one account collecting complaints from several victims "
    "over weeks and a falsely reported account. Labels reach banks by four paths (complaint, registry, police_bulk, "
    "hold_request); datasets at three cutoffs contain only labels known by then, revokes respect the cutoff, truth is "
    "read only through the evaluation package, and two replays give identical hashes."
)


async def find_missing_preconditions() -> list[str]:
    """Checks that Postgres, Redpanda and Redis are reachable (the replay needs the whole stack)."""
    problems: list[str] = []
    database_problem = await check_database(get_showcase_database_url(), "docker compose up -d postgres redpanda")
    if database_problem is not None:
        problems.append(database_problem)
    host, _, port = config.kafka_bootstrap_servers.split(",")[0].partition(":")
    kafka_problem = check_tcp_port(
        host, int(port or 9092), "Redpanda (Kafka)", "docker compose up -d postgres redpanda"
    )
    if kafka_problem is not None:
        problems.append(kafka_problem)
    redis_problem = check_redis(host_side_redis_url(), "docker compose up -d redis")
    if redis_problem is not None:
        problems.append(redis_problem)
    return problems


def get_checks(quick: bool = False) -> list[CheckDefinition]:
    """Returns the 13 Level 16 checks in the architect's order."""
    from showcase import level_16_checks as c

    def d(name: str, kind: str, proves: str, run: Any, cause: str, where: str) -> CheckDefinition:
        """One check definition."""
        return CheckDefinition(name=name, kind=kind, proves=proves, run=run, likely_cause=cause, where_to_look=where)

    return [
        d(
            "16-1 a gang's true mule activity against label arrivals",
            "normal",
            "For one receiving mule, the first complaint label arrives after the mule's first payment, "
            "and truth is read only "
            "through the evaluation package.",
            c.check_16_1_gang_timeline,
            "Label arrival times or the truth join are off.",
            "replay/dataset_builder.py, evaluation/dataset_truth_join.py",
        ),
        d(
            "16-2 datasets at three cutoffs hold only labels known by then",
            "edge",
            "No dataset label arrives after its cutoff, earlier datasets are inside later ones, and the first cutoff "
            "withholds labels.",
            c.check_16_2_three_cutoffs,
            "The cutoff filter in build_dataset is missing or wrong.",
            "replay/dataset_builder.py",
        ),
        d(
            "16-3 a falsely reported account is a positive label",
            "edge",
            "A complaint naming a genuine account makes a fraud label, and the truth comparison shows the account was "
            "genuine.",
            c.check_16_3_false_report,
            "False complaints were not planned or the label verdict was changed.",
            "world/complaints.py",
        ),
        d(
            "16-4 two replays give identical hashes",
            "consistency",
            "The feature log hash and the dataset content hash at three cutoffs are identical across two replays.",
            c.check_16_4_two_replays,
            "Something in the replay or dataset is not deterministic.",
            "replay/runner.py, replay/dataset_builder.py",
        ),
        d(
            "16-5 the dataset builder is refused truth",
            "negative",
            "A caller in the replay package that asks to load truth gets PermissionError, and no replay module imports "
            "the evaluation package.",
            c.check_16_5_builder_refused_truth,
            "The caller check in load_truth was weakened.",
            "evaluation/truth.py",
        ),
        d(
            "16-6 no future leakage, including revokes",
            "edge",
            "A revoke after the cutoff leaves the dataset unchanged; a revoke before the cutoff removes the label.",
            c.check_16_6_no_future_leakage,
            "Label status changes are applied without comparing to the cutoff.",
            "replay/dataset_builder.py",
        ),
        d(
            "16-7 all four label paths reach the right banks",
            "normal",
            "Complaint, registry, police_bulk and hold_request labels exist at the banks the plan says, and registry "
            "labels only at DPIP participants.",
            c.check_16_7_four_label_paths,
            "A label consumer matched the wrong bank.",
            "features/registry_label_consumer.py, replay/hold_requests.py",
        ),
        d(
            "16-8 N1: the innocent holder of a recycled number is labelled",
            "edge",
            "The registry labels the innocent current holder of a recycled number listed before day 0, and truth says "
            "that holder is not a mule.",
            c.check_16_8_innocent_holder,
            "The pre-world seed or the label consumer changed.",
            "replay/registry_replay.py",
        ),
        d(
            "16-9 the registry is wired into decisions",
            "normal",
            "registry_wired is true, payee_registry_match has values at participants and none at the non-participant.",
            c.check_16_9_registry_wired,
            "The screener is not injected per bank.",
            "replay/runner.py",
        ),
        d(
            "16-10 a superseded dataset is refused unless asked",
            "edge",
            "superseded_by is set on the old manifest, old files are kept, and load and join refuse it.",
            c.check_16_10_superseded,
            "The superseded guard is missing.",
            "replay/dataset_builder.py",
        ),
        d(
            "16-11 G3: one account collects complaints from several victims over weeks",
            "normal",
            "One mule account has at least three complaint labels from different victims over at least 14 days.",
            c.check_16_11_g3,
            "The gang or complaint delay changed.",
            "showcase/fixtures/level_16/gang_stacking.yaml",
        ),
        d(
            "16-12 hygiene: no label column is a feature",
            "normal",
            "No feature column starts with label_, per-feature None shares and the no_watermark count are shown.",
            c.check_16_12_hygiene,
            "A label column was added to the feature list.",
            "replay/dataset_builder.py",
        ),
        d(
            "16-13 mid-run opened account replayed with its inherited region",
            "normal",
            "At least one mid-run opened account is replayed carrying its inherited home region, "
            "showing the account, minute, bank and region.",
            c.check_16_13_mid_run_account,
            "Account openings were not scheduled or mid-run accounts lost their region.",
            "world/interactive_state.py, world/bank_connectors.py, world/bank_feed.py",
        ),
    ]
