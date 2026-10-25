"""
WHAT: Trivial-world check detecting synthetic worlds where fraud is trivially separable (Level 12h, A-026).
Computes six candidate account signals, evaluates area under the precision-recall curve (AUPRC) in both
directions, and gates training and validation worlds against a strict threshold of 0.90.

WHY: If a single heuristic signal separates fraud perfectly, benchmark models appear deceptively effective
without learning genuine behavioral patterns. Pre-registering six fixed signals and gating world generation
guarantees simulation complexity before model training begins.

TERM: trivial world: a synthetic simulation where a single feature separates illicit accounts at AUPRC >= 0.90.
TERM: average precision: discrete area under the precision-recall curve computed with tied group handling.
TERM: dwell time: elapsed minutes between receiving a credit and initiating the next outgoing payment.
TERM: pass-through: ratio of total outgoing funds to total incoming funds across simulation history.
"""

import bisect
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from core.config_validator import load_yaml
from world.interactive_records import InteractiveWorldOutput
from world.run_identity import REPO_ROOT

CONFIG_PATH: Path = REPO_ROOT / "config" / "evaluation.yaml"
MINUTES_PER_DAY: int = 1440


def load_evaluation_config(path: Path | None = None) -> dict[str, Any]:
    """
    WHAT: Reads the pre-registered evaluation configuration from evaluation.yaml.
    WHY: Loads pre-registered metric thresholds, gating rules, and policy bounds without hardcoding values.
    Args:
        path: Optional filesystem Path to evaluation configuration file; defaults to config/evaluation.yaml.
    Returns:
        Dictionary holding parsed YAML configuration mapping.
    Raises:
        FileNotFoundError: If the specified configuration path does not exist.
    """
    target = CONFIG_PATH if path is None else path
    return load_yaml(target)


def average_precision(scores: Sequence[float], labels: Sequence[int | bool]) -> float:
    """
    WHAT: Computes discrete Area Under the Precision-Recall Curve (AUPRC) using standard group evaluation.
    Sorts predictions descending, groups identical scores to calculate precision after each distinct threshold,
    and weights precision by the incremental positive recall fraction contributed by each group.

    WHY: Avoids interpolation artifacts and optimistic bias on tied score boundaries. Implemented in pure
    Python without numpy or scikit-learn dependencies to guarantee deterministic cross-platform behavior.

    Args:
        scores: Sequence of real-valued predictor scores for each candidate account.
        labels: Sequence of binary ground truth indicators (1/True for mules, 0/False for genuine).

    Returns:
        Float value in [0.0, 1.0] representing average precision.

    Raises:
        ValueError: If scores and labels differ in length, or if either positives or negatives are absent.
    """
    if len(scores) != len(labels):
        raise ValueError(f"scores length {len(scores)} does not match labels length {len(labels)}")
    n_pos = sum(1 for y in labels if y)
    n_neg = len(labels) - n_pos
    if n_pos == 0:
        raise ValueError("average_precision requires at least one positive label")
    if n_neg == 0:
        raise ValueError("average_precision requires at least one negative label")

    # Pair scores with binary integer labels and sort descending by predicted score.
    paired = sorted(zip(scores, labels, strict=True), key=lambda x: x[0], reverse=True)

    # Group adjacent observations having identical scores to evaluate precision collectively.
    groups: list[list[int]] = []
    current_score: float | None = None
    current_group: list[int] = []
    for score, lbl in paired:
        int_label = 1 if lbl else 0
        if current_score is None or score == current_score:
            current_score = score
            current_group.append(int_label)
        else:
            groups.append(current_group)
            current_score = score
            current_group = [int_label]
    if current_group:
        groups.append(current_group)

    # Accumulate true positives and evaluate precision after each distinct threshold group.
    cum_pos = 0
    cum_tot = 0
    ap = 0.0
    for group in groups:
        pos_in_group = sum(group)
        cum_pos += pos_in_group
        cum_tot += len(group)
        precision_after_group = cum_pos / cum_tot
        ap += (pos_in_group / n_pos) * precision_after_group

    return ap


def best_direction_ap(values: Sequence[float], labels: Sequence[int | bool]) -> float:
    """
    WHAT: Calculates two-tailed maximum average precision across positive and negated signal directions.
    WHY: Evaluates feature separability regardless of whether criminal behavior correlates positively or
         negatively with the signal (e.g. low dwell vs high device count).
    Args:
        values: Sequence of numeric feature values extracted per account.
        labels: Sequence of binary indicators designating mule vs genuine status.
    Returns:
        Maximum float AUPRC score achieved across direct or inverted signal orderings.
    """
    direct_ap = average_precision(values, labels)
    inverted_values = [-float(v) for v in values]
    inverted_ap = average_precision(inverted_values, labels)
    return max(direct_ap, inverted_ap)


def population(output: InteractiveWorldOutput) -> tuple[set[str], set[str]]:
    """
    WHAT: Extracts disjoint sets of mule accounts and genuine accounts from world truth and ledgers.
    Mules are accounts holding a 'mule' role in truth facts. Genuine accounts hold no gang role at any time
    (victims included). Fixture sources, collectors, cash-out and gang-opened collector accounts are excluded.

    WHY: Ground truth validation under A-026 requires comparing illicit money laundering nodes against
    innocent citizens, preventing intermediate operational infrastructure from distorting baseline distributions.

    Args:
        output: Completed InteractiveWorldOutput simulation result containing truth roles and bank ledgers.

    Returns:
        Tuple of (mule_accounts, genuine_accounts) account identifier string sets.
    """
    # 1. Gather all accounts registered across simulated bank ledgers.
    all_accounts: set[str] = set()
    for ledger in output.world.ledgers:
        all_accounts.update(ledger.accounts)

    # 2. Extract accounts assigned explicit fraud roles from simulation truth.
    truth_roles = output.result.truth.roles
    mules = {r.account_id for r in truth_roles if r.role == "mule"}

    # 3. Identify intermediate gang operational accounts excluded from both sets.
    neither: set[str] = {r.account_id for r in truth_roles if r.role in ("collector", "cash_out")}


    # Exclude non-mule accounts provisioned through gang account opening streams.
    for rec in getattr(output.result, "opened_accounts", []):
        if len(rec) >= 4 and rec[3] == "gang_account_openings" and rec[1] not in mules:
            neither.add(rec[1])

    # 4. Genuine accounts hold no gang role at any time; victims are explicitly included.
    genuine = (all_accounts - mules) - neither
    return mules, genuine


def account_signals(
    output: InteractiveWorldOutput, min_gap_days: int | None = None
) -> dict[str, dict[str, float]]:
    """
    WHAT: Computes six candidate behavioral signals for every account in the simulated world:
         - routine_life: number of distinct counterparties with >= 2 successful payments at least 20 days apart.
         - pass_through: ratio of total amount sent over total amount received (0.0 if nothing received).
         - dwell: median minutes from each received credit to the account's next outgoing payment (or world end).
         - devices: distinct device ids seen for the account (sender stamps plus gang device events naming it).
         - payer_count: distinct accounts that paid it.
         - payer_regions: distinct geographic regions among those payers (shop payers count as their shop's region;
           payers without a known region such as gang-opened accounts or fixture sources are skipped).

    WHY: Implements the pre-registered signal battery from amendment A-026 to measure trivial separability.

    Args:
        output: Interactive simulation run output container.
        min_gap_days: Optional integer day threshold for routine counterparties; defaults to evaluation.yaml policy.

    Returns:
        Mapping of account identifier to dictionary of signal name to float score.
    """
    # Resolve routine gap policy threshold from configuration if not explicitly provided.
    gap_days = min_gap_days
    if gap_days is None:
        cfg = load_evaluation_config()
        gap_days = int(cfg["trivial_world_check"]["routine_min_gap_days"])
    min_gap_minutes = gap_days * MINUTES_PER_DAY

    # Determine simulation boundary duration in minutes.
    world_end = float(output.plan.n_days * MINUTES_PER_DAY)

    # Index citizens and commercial shops by account identifier for geographic and device resolution.
    people_by_acc = {p.account_id: p for p in output.world.people}
    shops_by_acc = {s.account_id: s for s in output.world.shops}

    # Filter successful payment events and align corresponding sender telemetry stamps.
    all_events = output.result.run_result.events
    all_stamps = output.result.run_result.event_sender_devices
    success_events = [
        (ev, st) for ev, st in zip(all_events, all_stamps, strict=True) if ev.status == "success"
    ]

    # Pre-index transaction timestamps and transfer sums for fast signal calculation.
    cp_payments: dict[str, dict[str, list[int]]] = {}
    total_sent: dict[str, int] = {}
    total_recv: dict[str, int] = {}
    creds_by_acc: dict[str, list[int]] = {}
    debs_by_acc: dict[str, list[int]] = {}
    payers_by_acc: dict[str, set[str]] = {}

    for ev, _ in success_events:
        f, t, m, a = ev.from_account, ev.to_account, ev.minute, ev.amount_paise
        total_sent[f] = total_sent.get(f, 0) + a
        total_recv[t] = total_recv.get(t, 0) + a
        creds_by_acc.setdefault(t, []).append(m)
        debs_by_acc.setdefault(f, []).append(m)
        payers_by_acc.setdefault(t, set()).add(f)
        if f != t:
            cp_payments.setdefault(f, {}).setdefault(t, []).append(m)
            cp_payments.setdefault(t, {}).setdefault(f, []).append(m)

    # Collect distinct hardware fingerprints from sender stamps on payment attempts.
    devices_by_acc: dict[str, set[str]] = {}
    for ev, st in zip(all_events, all_stamps, strict=True):
        if st.fingerprint is not None:
            devices_by_acc.setdefault(ev.from_account, set()).add(st.fingerprint)

    # Ingest gang device events naming accounts into observed hardware profiles.
    for gde in getattr(output.result, "applied_gang_device_events", []):
        if gde.account_id is not None and gde.device_id:
            devices_by_acc.setdefault(gde.account_id, set()).add(gde.device_id)

    # Gather all registered accounts across banks to assemble the output feature table.
    all_accounts: set[str] = set()
    for ledger in output.world.ledgers:
        all_accounts.update(ledger.accounts)

    signals: dict[str, dict[str, float]] = {}
    for acc in sorted(all_accounts):
        # 1. Routine life: count distinct counterparties with payments separated by at least min_gap_minutes.
        rl_count = 0
        for _cp, minutes in cp_payments.get(acc, {}).items():
            if len(minutes) >= 2 and (max(minutes) - min(minutes)) >= min_gap_minutes:
                rl_count += 1

        # 2. Pass-through: ratio of total sent funds over total received funds.
        sent_paise = total_sent.get(acc, 0)
        recv_paise = total_recv.get(acc, 0)
        pass_through = (sent_paise / recv_paise) if recv_paise > 0 else 0.0

        # 3. Dwell: median duration between each received credit and the next outgoing debit.
        credits = creds_by_acc.get(acc, [])
        if not credits:
            dwell = world_end
        else:
            debits = debs_by_acc.get(acc, [])
            dwell_samples: list[float] = []
            for c_minute in credits:
                idx = bisect.bisect_left(debits, c_minute)
                if idx < len(debits):
                    dwell_samples.append(float(debits[idx] - c_minute))
                else:
                    dwell_samples.append(world_end - float(c_minute))
            dwell = float(statistics.median(dwell_samples))

        # 4. Devices: count of distinct hardware identifiers or stamps associated with this account.
        dev_count = float(len(devices_by_acc.get(acc, set())))

        # 5. Payer count: distinct accounts transferring funds into this account.
        payers = payers_by_acc.get(acc, set())
        p_count = float(len(payers))

        # 6. Payer regions: distinct municipal districts among incoming payers.
        regions: set[int] = set()
        for p_id in payers:
            if p_id in people_by_acc:
                regions.add(people_by_acc[p_id].region)
            elif p_id in shops_by_acc:
                regions.add(shops_by_acc[p_id].region)
            # Payers without a known region (e.g. gang-opened accounts or fixture sources) are skipped.
        p_regions = float(len(regions))

        signals[acc] = {
            "routine_life": float(rl_count),
            "pass_through": float(pass_through),
            "dwell": float(dwell),
            "devices": float(dev_count),
            "payer_count": float(p_count),
            "payer_regions": float(p_regions),
        }

    return signals


def check_world(output: InteractiveWorldOutput, config: dict[str, Any]) -> dict[str, Any]:
    """
    WHAT: Evaluates a completed world simulation against the trivial-world check policy.
    Computes AUPRC for all candidate signals across the mule vs genuine population, extracts the top 3
    features, determines whether the world passes the fail threshold, and scores held-out tactic gangs.

    WHY: Implements quality control gating under Level 12h, preventing trivially separable worlds from
    entering model training and auditing test/stress worlds for held-out tactic leakage.

    Args:
        output: Simulated InteractiveWorldOutput container.
        config: Evaluation configuration mapping containing the trivial_world_check policy section.

    Returns:
        Dictionary containing fraud_share, n_mules, n_genuine, per_signal_ap, top_3, best_signal,
        best_ap, passed, and held_out scores.
    """
    # Extract trivial world check parameters from configuration mapping.
    tw_cfg = config.get("trivial_world_check", config)
    fail_threshold = float(tw_cfg["fail_threshold"])
    routine_min_gap_days = int(tw_cfg["routine_min_gap_days"])
    top_features_count = int(tw_cfg.get("report_top_features", 3))

    # Calculate world fraud share across successful payment transactions.
    all_events = output.result.run_result.events
    event_sources = getattr(output.result, "event_sources", [])
    successful_pmts = sum(1 for ev in all_events if ev.status == "success")
    fraud_successful = sum(
        1 for ev, src in zip(all_events, event_sources, strict=True)
        if src == "fraud" and ev.status == "success"
    )
    fraud_share = (fraud_successful / successful_pmts) if successful_pmts > 0 else 0.0

    # Extract target evaluation populations and compute feature signals.
    mules, genuine = population(output)
    signals = account_signals(output, min_gap_days=routine_min_gap_days)

    # Evaluate feature separability across the combined population.
    pop_accounts = sorted(mules | genuine)
    labels = [1 if acc in mules else 0 for acc in pop_accounts]

    # Evaluate each signal present in the signal table.
    first_row = signals[pop_accounts[0]] if pop_accounts else {}
    signal_names = sorted(first_row.keys())

    per_signal_ap: dict[str, float] = {}
    for sig in signal_names:
        scores = [signals[acc][sig] for acc in pop_accounts]
        per_signal_ap[sig] = best_direction_ap(scores, labels)

    # Rank features descending by separability score.
    sorted_features = sorted(per_signal_ap.items(), key=lambda x: x[1], reverse=True)
    top_3 = sorted_features[:top_features_count]
    best_signal, best_ap = sorted_features[0]
    passed = best_ap < fail_threshold

    # Evaluate held-out tactical gangs on test and stress worlds as report-only metrics.
    held_out_tactics_list = config.get("held_out_tactics", {}).get(
        "tactics", ["slow_drip", "device_rotation"]
    )
    held_out_gang = None
    for g in getattr(output.result.truth, "gangs", []):
        if getattr(g, "family", None) in held_out_tactics_list:
            held_out_gang = g
            break

    held_out_result = None
    if held_out_gang is not None:
        ho_mules = {
            r.account_id
            for r in output.result.truth.roles
            if r.role == "mule" and r.gang_id == held_out_gang.gang_id
        }
        ho_pop = sorted(ho_mules | genuine)
        ho_labels = [1 if acc in ho_mules else 0 for acc in ho_pop]

        ho_per_signal: dict[str, float] = {}
        for sig in signal_names:
            scores = [signals[acc][sig] for acc in ho_pop]
            ho_per_signal[sig] = best_direction_ap(scores, ho_labels)

        ho_sorted = sorted(ho_per_signal.items(), key=lambda x: x[1], reverse=True)
        ho_top_3 = ho_sorted[:top_features_count]
        ho_best_signal, ho_best_ap = ho_sorted[0]

        held_out_result = {
            "family": held_out_gang.family,
            "gang_id": held_out_gang.gang_id,
            "n_mules": len(ho_mules),
            "per_signal_ap": ho_per_signal,
            "top_3": ho_top_3,
            "best_signal": ho_best_signal,
            "best_ap": ho_best_ap,
            "passed": ho_best_ap < fail_threshold,
        }

    return {
        "fraud_share": fraud_share,
        "n_mules": len(mules),
        "n_genuine": len(genuine),
        "per_signal_ap": per_signal_ap,
        "top_3": top_3,
        "best_signal": best_signal,
        "best_ap": best_ap,
        "passed": passed,
        "held_out": held_out_result,
    }
