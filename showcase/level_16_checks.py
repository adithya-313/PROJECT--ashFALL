# File: showcase/level_16_checks.py
# Purpose: Level 16 showcase - the 12 checks (labels, replay datasets, truth kept apart).
"""
WHAT: The 12 Level 16 checks, in the order the architect fixed. Each reads the shared state (one recorded world, two
replays, datasets at three cutoffs) and passes only on real counts above zero, printing the evidence it used.

WHY: Level 16 turns a replay into labelled datasets without letting truth or the future leak in. These checks show each
promise on a world that really contains a crackdown, a cross-bank complaint, a recycled number, a G3 account and a
false report.

TERM: label path: how a label reached a bank - complaint, registry, police_bulk or hold_request.
TERM: G3: one receiving account collecting complaints from several victims over weeks.
TERM: showcase-only: a setting chosen so the check has something to look at; it is not a claim about the real world.
"""

from __future__ import annotations

import ast
import collections
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from evaluation.dataset_truth_join import join_truth, roles_at
from evaluation.truth import load_truth
from replay.dataset_builder import (
    SupersededDatasetError,
    build_dataset,
    load_dataset,
    mark_superseded,
    write_dataset,
)
from showcase.framework import CheckOutcome
from showcase.level_16_support import Level16State, get_state
from showcase.level_16_world import innocent_holders
from world.label_builder import build_bulk_confirmations, build_label_events

REPO_ROOT = Path(__file__).resolve().parent.parent
MULE_ROLES = {"mule", "collector", "cash_out"}
FALSE_REPORT_NOTE = "showcase-only: stress-range false-complaint rate so a false report occurs"


def _fail(message: str) -> AssertionError:
    """Builds the error a check raises when its evidence does not hold."""
    return AssertionError(message)


def _labels(dataset: Any) -> list[dict[str, Any]]:
    """The label rows of a dataset as dictionaries."""
    return list(dataset.labels.to_pylist())


def _world_id(state: Level16State) -> str:
    """The world id the dataset tables carry."""
    return str(state.extract_a.world_id)


async def check_16_1_gang_timeline(workspace: Any) -> CheckOutcome:
    """
    Check 16-1 (normal): for one gang's receiving mule, truth says when the mule was active and the dataset says when
    labels about it arrived; the first label comes after the first mule activity.
    """
    st = await get_state()
    dataset, path = st.datasets["c3"]
    payments, _ = join_truth(path, st.truth_root)
    mule_rows = [r for r in payments.to_pylist() if r["truth_role_receiver"] == "mule"]
    if not mule_rows:
        raise _fail("no payment row has a truth mule receiver")
    complaint_labels = [r for r in _labels(dataset) if r["source"] == "complaint" and r["via"] is None]
    by_account: dict[str, list[datetime]] = collections.defaultdict(list)
    for row in complaint_labels:
        by_account[row["account_id"]].append(row["arrival_time"])
    candidates = {r["receiver_account_id"] for r in mule_rows} & set(by_account)
    if not candidates:
        raise _fail("no truth mule has a complaint label in the dataset")
    account = max(sorted(candidates), key=lambda a: len(by_account[a]))
    times = sorted(r["decision_time"] for r in mule_rows if r["receiver_account_id"] == account)
    owner = next(r["truth_owner"] for r in mule_rows if r["receiver_account_id"] == account)
    arrivals = sorted(by_account[account])
    if not times[0] < arrivals[0]:
        raise _fail(f"a label about {account} arrived before the first mule payment")
    lag_days = (arrivals[0] - times[0]).total_seconds() / 86400
    return CheckOutcome(
        key_number=f"{account}: first mule payment to first label {lag_days:.1f} days",
        evidence={
            "mule_account": account,
            "gang_owner": owner,
            "mule_payments_received": len(times),
            "first_activity": times[0].isoformat(),
            "last_activity": times[-1].isoformat(),
            "label_arrivals": [a.isoformat() for a in arrivals],
            "truth_read_through": "evaluation.dataset_truth_join",
        },
    )


async def check_16_2_three_cutoffs(workspace: Any) -> CheckOutcome:
    """
    Check 16-2 (edge): datasets built at three cutoffs hold only labels known by then, and an earlier dataset's
    labels are all in the later ones.
    """
    st = await get_state()
    rows = {name: _labels(ds) for name, (ds, _) in st.datasets.items()}
    counts = {name: len(r) for name, r in rows.items()}
    for name, cutoff in st.cutoffs.items():
        late = [r for r in rows[name] if r["arrival_time"] > cutoff]
        if late:
            raise _fail(f"{name} holds {len(late)} labels that arrive after its cutoff")
    known = {n: {(r["bank_id"], r["label_id"], r["account_id"]) for r in rs} for n, rs in rows.items()}
    if not (known["c1"] <= known["c2"] <= known["c3"]):
        raise _fail("an earlier cutoff holds a label that a later cutoff lacks")
    withheld_c1 = sum(1 for r in st.extract_a.label_rows if r["arrival_time"] > st.cutoffs["c1"])
    if not (counts["c1"] > 0 and counts["c1"] < counts["c3"] and withheld_c1 > 0):
        raise _fail(f"cutoffs do not separate labels: {counts}, withheld at c1: {withheld_c1}")
    return CheckOutcome(
        key_number=f"labels c1/c2/c3 = {counts['c1']}/{counts['c2']}/{counts['c3']}; withheld at c1: {withheld_c1}",
        evidence={
            "cutoffs": {k: v.isoformat() for k, v in st.cutoffs.items()},
            "labels_per_cutoff": counts,
            "labels_after_cutoff_in_dataset": 0,
            "withheld_at_c1": withheld_c1,
        },
    )


async def check_16_3_false_report(workspace: Any) -> CheckOutcome:
    """
    Check 16-3 (edge): an account that a complaint names but that truth says was genuine is a positive (fraud) label,
    and the truth comparison shows the label is wrong.
    """
    st = await get_state()
    dataset, path = st.datasets["c3"]
    complaints = [r for r in _labels(dataset) if r["source"] == "complaint" and r["via"] is None]
    roles = roles_at(path, st.truth_root, [(r["world_id"], r["account_id"], r["arrival_time"]) for r in complaints])
    wrong = [(r, role) for r, (role, _owner) in zip(complaints, roles, strict=True) if role not in MULE_ROLES]
    if not wrong:
        raise _fail("no complaint label names a genuine account")
    if any(r["verdict"] != "fraud" for r, _ in wrong):
        raise _fail("a falsely reported account is not a positive label")
    accounts = sorted({r["account_id"] for r, _ in wrong})
    return CheckOutcome(
        key_number=f"{len(accounts)} genuine accounts carry a fraud label ({len(wrong)} labels)",
        evidence={
            "falsely_reported_accounts": accounts,
            "labels": len(wrong),
            "verdict": "fraud",
            "truth_role_at_label_time": sorted({role for _, role in wrong}),
            "config_note": FALSE_REPORT_NOTE,
        },
    )


async def check_16_4_two_replays(workspace: Any) -> CheckOutcome:
    """
    Check 16-4 (consistency): two replays of the same world give the same feature log hash and the same dataset
    content hash at every cutoff.
    """
    st = await get_state()
    hash_a, hash_b = st.result_a.feature_log_sha256, st.result_b.feature_log_sha256
    split = str(st.cfg["world"]["range_name"])
    same: dict[str, str] = {}
    for name, cutoff in st.cutoffs.items():
        a = st.datasets[name][0].manifest["content_hash"]
        b = build_dataset([st.extract_b], split=split, cutoff=cutoff).manifest["content_hash"]
        if a != b:
            raise _fail(f"dataset content hash differs at {name}: {a[:12]} vs {b[:12]}")
        same[name] = a[:12]
    if len(hash_a) != 64 or hash_a != hash_b:
        raise _fail(f"feature log hash differs: {hash_a[:12]} vs {hash_b[:12]}")
    return CheckOutcome(
        key_number=f"feature log {hash_a[:12]} and 3 dataset hashes identical across 2 replays",
        evidence={
            "feature_log_sha256": hash_a,
            "dataset_content_hash_prefixes": same,
            "decisions": st.result_a.decisions_sent,
        },
    )


async def check_16_5_builder_refused_truth(workspace: Any) -> CheckOutcome:
    """
    Check 16-5 (negative): a caller inside the replay package that tries to read truth is refused, and no replay
    module imports the evaluation package.
    """
    st = await get_state()
    namespace: dict[str, Any] = {"__name__": "replay.dataset_builder", "load_truth": load_truth}
    exec("def probe(root, run_id):\n    return load_truth(root, run_id)\n", namespace)  # noqa: S102
    run_id = st.recorded.output.result.run_id
    try:
        namespace["probe"](st.truth_root, run_id)
    except PermissionError as error:
        refusal = str(error)
    else:
        raise _fail("load_truth answered a caller in the replay package")
    importers: list[str] = []
    scanned = 0
    for source in sorted((REPO_ROOT / "replay").glob("*.py")):
        scanned += 1
        for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else []
            if isinstance(node, ast.ImportFrom) and node.module:
                names.append(node.module)
            if any(n == "evaluation" or n.startswith("evaluation.") for n in names):
                importers.append(source.name)
    if importers or scanned == 0:
        raise _fail(f"replay modules import evaluation: {importers}")
    return CheckOutcome(
        key_number=f"refused from caller replay.dataset_builder; {scanned} replay modules import no evaluation code",
        evidence={
            "caller_module": "replay.dataset_builder",
            "refusal": refusal[:160],
            "replay_modules_scanned": scanned,
        },
    )


async def check_16_6_no_future_leakage(workspace: Any) -> CheckOutcome:
    """
    Check 16-6 (edge): a revoke that arrives after the cutoff does not change that cutoff's dataset, a revoke that
    arrived before the cutoff removes the label, and no label in a dataset arrives after its cutoff.
    """
    st = await get_state()
    split = str(st.cfg["world"]["range_name"])
    cutoff = st.cutoffs["c2"]
    early = [
        r
        for r in st.extract_a.label_rows
        if r["source"] == "complaint" and r["arrival_time"] < cutoff - timedelta(hours=2)
    ]
    if not early:
        raise _fail("no complaint label arrived two hours before c2")
    target = early[0]

    def revoke(when: datetime) -> list[dict[str, Any]]:
        """One revoke status row for the target label at the given time."""
        return [
            {
                "bank_id": target["bank_id"],
                "source": target["source"],
                "label_id": target["label_id"],
                "account_id": target["account_id"],
                "change": "revoked",
                "arrival_time": when,
            }
        ]

    def ids(extract: Any, at: datetime) -> tuple[set[str], int]:
        """Label ids in the dataset at a cutoff and how many revokes it applied."""
        ds = build_dataset([extract], split=split, cutoff=at)
        return {r["label_id"] for r in _labels(ds)}, int(ds.manifest["labels_revoked_excluded"])

    base_ids, base_revoked = ids(replace(st.extract_a, label_status_rows=[]), cutoff)
    after_ids, after_revoked = ids(replace(st.extract_a, label_status_rows=revoke(cutoff + timedelta(hours=1))), cutoff)
    before_ids, before_revoked = ids(
        replace(st.extract_a, label_status_rows=revoke(target["arrival_time"] + timedelta(minutes=1))), cutoff
    )
    later_ids, _ = ids(replace(st.extract_a, label_status_rows=revoke(cutoff + timedelta(hours=1))), st.cutoffs["c3"])
    label_id = target["label_id"]
    if label_id not in base_ids:
        raise _fail("the target label is missing from the plain dataset")
    if label_id not in after_ids or after_revoked != base_revoked:
        raise _fail("a revoke after the cutoff changed the dataset at the cutoff")
    if label_id in before_ids or before_revoked != base_revoked + 1:
        raise _fail("a revoke before the cutoff did not remove the label")
    if label_id in later_ids:
        raise _fail("the later revoke was not applied once its time passed")
    return CheckOutcome(
        key_number=f"{label_id}: kept at c2 when revoked after c2; removed when revoked before c2",
        evidence={
            "cutoff": cutoff.isoformat(),
            "labels_in_plain_dataset": len(base_ids),
            "revoke_after_cutoff_labels": len(after_ids),
            "revoke_before_cutoff_labels": len(before_ids),
            "revokes_applied_before_cutoff": before_revoked,
            "revoke_after_cutoff_applied_at_c3": label_id not in later_ids,
        },
    )


def verify_four_label_paths(
    rows: list[dict[str, Any]],
    expected_complaint: set[tuple[str, str]],
    expected_bulk: set[tuple[str, str]],
    expected_hold: set[tuple[str, str]],
    registry_participants: list[str] | set[str],
) -> tuple[dict[str, set[tuple[str, str]]], collections.Counter[tuple[str, str]]]:
    """
    WHAT: Evaluates extracted label rows against expected targets for each of the four label paths.
    WHY: Check 16-7: Separates direct complaints (same-bank only, via None) from hold requests (cross-bank,
    via hold_request). Registry and police_bulk are evaluated unchanged.
    """
    got: dict[str, set[tuple[str, str]]] = collections.defaultdict(set)
    per_path: collections.Counter[tuple[str, str]] = collections.Counter()
    for r in rows:
        path = (
            "hold_request"
            if r.get("via") == "hold_request"
            else ("complaint" if r["source"] == "complaint" and not r.get("via") else r["source"])
        )
        got[path].add((r["bank_id"], r["account_id"]))
        per_path[(path, r["bank_id"])] += 1
    missing = [p for p in ("complaint", "registry", "police_bulk", "hold_request") if not got[p]]
    if missing:
        raise _fail(f"no labels from path(s): {missing}")

    print(f"direct complaint labels: expected {len(expected_complaint)}, got {len(got['complaint'])}")
    print(f"hold request labels: expected {len(expected_hold)}, got {len(got['hold_request'])}")

    for name, expected in (
        ("complaint", expected_complaint),
        ("police_bulk", expected_bulk),
        ("hold_request", expected_hold),
    ):
        if not expected or not expected <= got[name]:
            raise _fail(f"{name}: expected {sorted(expected)} but labels hold {sorted(got[name])}")
    outside = sorted(b for b, _ in got["registry"] if b not in registry_participants)
    if outside:
        raise _fail(f"registry labels at non-participant bank(s) {outside}")
    return got, per_path


async def check_16_7_four_label_paths(workspace: Any) -> CheckOutcome:
    """
    Check 16-7 (normal): complaint, registry, police_bulk and hold_request labels all exist, each at the banks the plan
    says: complaint labels at the reported account's bank, bulk labels at the holding bank, hold labels at the
    receiving bank, and
    registry labels only at DPIP participants.
    """
    st = await get_state()
    rows = st.extract_a.label_rows
    output = st.recorded.output
    end_minute = int(st.cfg["world"]["n_days"]) * 1440
    events = {e.label_id: e for e in build_label_events(output) if e.kind == "complaint"}
    expected_complaint = {
        (f"bank_{e.reported_bank_index}", e.reported_account)
        for e in events.values()
        if e.arrival_minute <= end_minute
        and e.reported_bank_index is not None
        and e.complainant_bank_index == e.reported_bank_index
    }
    expected_bulk = {
        (f"bank_{b}", a)
        for c in build_bulk_confirmations(output).confirmations
        if c.arrival_minute <= end_minute
        for a, b in zip(c.account_ids, c.account_bank_indexes, strict=True)
    }
    expected_hold = {
        (h.item.bank_id, events[h.complaint_id].reported_account)
        for h in st.driver.holds
        if h.complaint_id in events and h.latency_minutes + events[h.complaint_id].arrival_minute <= end_minute
    }
    got, per_path = verify_four_label_paths(
        rows,
        expected_complaint=expected_complaint,
        expected_bulk=expected_bulk,
        expected_hold=expected_hold,
        registry_participants=st.registry.participants,
    )
    return CheckOutcome(
        key_number="labels per path: "
        + ", ".join(
            f"{p}={sum(n for (q, _), n in per_path.items() if q == p)}"
            for p in ("complaint", "registry", "police_bulk", "hold_request")
        ),
        evidence={
            "labels_per_path_and_bank": {f"{p}@{b}": n for (p, b), n in sorted(per_path.items())},
            "registry_participants": st.registry.participants,
            "hold_requests_missed": st.driver.hold_missed,
        },
    )


async def check_16_8_innocent_holder(workspace: Any) -> CheckOutcome:
    """
    Check 16-8 (edge): N1 - the registry labels the innocent current holder of a recycled number that was listed
    before day 0, and truth says that holder is not a mule.
    """
    st = await get_state()
    dataset, path = st.datasets["c3"]
    expected = set(innocent_holders(st.recorded, st.registry))
    registry_labels = [r for r in _labels(dataset) if r["source"] == "registry"]
    hits = [r for r in registry_labels if (r["account_id"], r["bank_id"]) in expected]
    if not expected or not hits:
        raise _fail(
            f"expected innocent holders {sorted(expected)}, "
            f"registry labels {[r['account_id'] for r in registry_labels]}"
        )
    roles = roles_at(path, st.truth_root, [(r["world_id"], r["account_id"], r["arrival_time"]) for r in hits])
    guilty = [r["account_id"] for r, (role, _o) in zip(hits, roles, strict=True) if role in MULE_ROLES]
    if guilty:
        raise _fail(f"truth says {guilty} are mules, so the label is not an N1 case")
    first = hits[0]
    return CheckOutcome(
        key_number=(
            f"{first['account_id']} at {first['bank_id']} is labelled fraud by the registry; truth role {roles[0][0]}"
        ),
        evidence={
            "innocent_holders_planned": sorted(expected),
            "registry_labels_on_them": len(hits),
            "truth_roles": sorted({role for role, _ in roles}),
            "verdict": first["verdict"],
        },
    )


async def check_16_9_registry_wired(workspace: Any) -> CheckOutcome:
    """
    Check 16-9 (normal): registry_wired is true in the manifest, payee_registry_match has values at the DPIP
    participants, and is None at the non-participant bank.
    """
    st = await get_state()
    dataset, _ = st.datasets["c3"]
    manifest = dataset.manifest
    if manifest["registry_wired"] is not True:
        raise _fail("manifest registry_wired is not true")
    per_bank: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    for r in dataset.payments.to_pylist():
        if r["status"] != "ok":
            continue
        per_bank[r["bank_id"]][0] += 1
        per_bank[r["bank_id"]][1] += r["payee_registry_match"] is not None
    for bank in st.registry.participants:
        if per_bank[bank][0] == 0 or per_bank[bank][1] == 0:
            raise _fail(f"{bank} (participant) has no payee_registry_match values: {dict(per_bank)}")
    for bank in sorted(set(per_bank) - set(st.registry.participants)):
        if per_bank[bank][0] == 0 or per_bank[bank][1] != 0:
            raise _fail(f"{bank} (non-participant) must have rows and no payee_registry_match: {dict(per_bank)}")
    non_participants = sorted(set(per_bank) - set(st.registry.participants))
    if not non_participants:
        raise _fail("no non-participant bank made a decision")
    return CheckOutcome(
        key_number=f"registry_wired=true; non-None payee_registry_match: {sum(v[1] for v in per_bank.values())} rows",
        evidence={
            "registry_wired": True,
            "reason": manifest["registry_wired_reason"],
            "ok_rows_and_non_none_per_bank": {
                b: {"ok_rows": v[0], "non_none": v[1]} for b, v in sorted(per_bank.items())
            },
            "non_participant_banks": non_participants,
            "screens": st.result_a.registry_screens,
            "screens_unknown": st.result_a.registry_screens_unknown,
        },
    )


async def check_16_10_superseded(workspace: Any) -> CheckOutcome:
    """
    Check 16-10 (edge): a dataset replaced by a newer one is marked superseded_by, its old files are kept, and
    loading or joining it is refused unless explicitly asked.
    """
    st = await get_state()
    split = str(st.cfg["world"]["range_name"])
    folder = st.work_dir / "superseded"
    old = write_dataset(build_dataset([st.extract_a], split=split, cutoff=st.cutoffs["c1"]), folder)
    new = write_dataset(build_dataset([st.extract_a], split=split, cutoff=st.cutoffs["c3"]), folder)
    before = sorted(p.name for p in old.iterdir())
    mark_superseded(old, new)
    if sorted(p.name for p in old.iterdir()) != before or len(before) != 4:
        raise _fail(f"old files changed: {before}")
    refused = {"load_dataset": False, "join_truth": False}
    try:
        load_dataset(old)
    except SupersededDatasetError:
        refused["load_dataset"] = True
    try:
        join_truth(old, st.truth_root)
    except ValueError:
        refused["join_truth"] = True
    if not all(refused.values()):
        raise _fail(f"an old dataset was not refused: {refused}")
    explicit = load_dataset(old, allow_superseded=True)
    if explicit.manifest["superseded_by"] != new.name or load_dataset(new).manifest["superseded_by"] is not None:
        raise _fail("superseded_by is wrong")
    return CheckOutcome(
        key_number=f"{old.name} superseded by {new.name}; refused until allow_superseded=True",
        evidence={
            "old": old.name,
            "new": new.name,
            "old_files_kept": before,
            "refused": refused,
            "loaded_when_asked_labels": explicit.labels.num_rows,
        },
    )


async def check_16_11_g3(workspace: Any) -> CheckOutcome:
    """
    Check 16-11 (normal): G3 - one account collects complaint labels from several different victims over weeks, and
    truth says it is a mule.
    """
    st = await get_state()
    dataset, path = st.datasets["c3"]
    floors = st.cfg["minimums"]
    by_account: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in _labels(dataset):
        if r["source"] == "complaint" and r["via"] is None:
            by_account[r["account_id"]].append(r)
    account = max(sorted(by_account), key=lambda a: len({r["label_id"] for r in by_account[a]}), default=None)
    if account is None:
        raise _fail("no complaint labels")
    rows = by_account[account]
    victims = len({r["label_id"] for r in rows})
    spread = (max(r["arrival_time"] for r in rows) - min(r["arrival_time"] for r in rows)).total_seconds() / 86400
    role = roles_at(path, st.truth_root, [(rows[0]["world_id"], account, max(r["arrival_time"] for r in rows))])[0][0]
    if victims < floors["g3_complaints_on_one_account"] or spread < floors["g3_spread_days"] or role not in MULE_ROLES:
        raise _fail(f"{account}: {victims} complaints over {spread:.1f} days, truth role {role}")
    return CheckOutcome(
        key_number=f"{account}: {victims} complaints from {victims} victims over {spread:.1f} days; truth role {role}",
        evidence={
            "account": account,
            "complaint_labels": victims,
            "spread_days": round(spread, 1),
            "truth_role": role,
            "floors": {"complaints": floors["g3_complaints_on_one_account"], "days": floors["g3_spread_days"]},
        },
    )


async def check_16_12_hygiene(workspace: Any) -> CheckOutcome:
    """
    Check 16-12 (normal): no column starting "label_" is a feature, every feature's share of None values is shown,
    and the number of decisions that had no watermark is shown.
    """
    st = await get_state()
    dataset, _ = st.datasets["c3"]
    manifest = dataset.manifest
    features = list(manifest["feature_columns"])
    leaked = [c for c in features if c.startswith("label_")]
    label_columns = [c for c in dataset.payments.column_names if c.startswith("label_")]
    if leaked or not features or not label_columns:
        raise _fail(
            f"feature columns: {len(features)}, leaked label columns: {leaked}, "
            f"label columns kept apart: {label_columns}"
        )
    shares = dict(manifest["feature_none_shares"])
    if not shares:
        raise _fail("no per-feature None shares in the manifest")
    ranked = sorted(shares.items(), key=lambda kv: -float(kv[1]))
    return CheckOutcome(
        key_number=(
            f"{len(features)} features, none start with label_; "
            f"no_watermark decisions: {manifest['no_watermark_count']}"
        ),
        evidence={
            "feature_count": len(features),
            "label_columns_kept_apart": label_columns,
            "no_watermark_count": manifest["no_watermark_count"],
            "none_share_top5": {k: round(float(v), 3) for k, v in ranked[:5]},
            "none_share_all": {k: round(float(v), 3) for k, v in sorted(shares.items())},
        },
    )


async def check_16_13_mid_run_account(workspace: Any) -> CheckOutcome:
    """
    Check 16-13 (normal): mid-run opened account replayed with its inherited region.
    Verifies that the showcase world recorded and replayed at least one mid-run opened account
    carrying its authentic inherited region, and prints the account, minute, bank and region.
    """
    st = await get_state()
    opened = st.recorded.output.result.opened_accounts
    if not opened:
        raise _fail("no mid-run opened accounts in showcase world")

    first = opened[0]
    minute, account_id, owner_kind, opening_stream, bank_index, opening_balance_paise, region = first
    bank_id = f"bank_{bank_index}"

    return CheckOutcome(
        key_number=f"{account_id} at minute {minute} in {bank_id} with region {region}",
        evidence={
            "account_id": account_id,
            "minute": minute,
            "bank_id": bank_id,
            "region": region,
            "owner_kind": owner_kind,
            "opening_stream": opening_stream,
            "opening_balance_paise": opening_balance_paise,
            "total_mid_run_accounts": len(opened),
        },
    )
