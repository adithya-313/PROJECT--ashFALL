# File: features/m1_network.py
# Purpose: Level 15b M1 network feature calculation functions over commercial bank payment graphs.
# Issue: Level 15b M1 network features (plan Task 1, G1, R8).
"""
WHAT: Pure and state-backed functions computing M1 network behavioral features:
  - hops_to_confirmed_label: Shortest path distance (hops) to a confirmed fraud account at this bank.
  - hops_to_complained_account: Shortest path distance (hops) to an account with registered complaints at this bank.
  - shared_counterparties_with_flagged_7d: Count of raw-window counterparties shared with labelled accounts.
  - local_cluster_density: Clustering coefficient among this account's same-bank counterparties.

WHY: Fraud syndicates and money mules operate in dense, interconnected clusters. Graph distance
to known mules and shared counterparty fan-in signals provide strong topological indicators of collusion
while strictly maintaining commercial bank tenant isolation (edges never cross bank boundaries).
"""

import logging
from collections import deque
from datetime import datetime
from typing import Any

from features.account_keys import (
    AccountFeaturesSettings,
    get_default_settings,
)
from features.account_labels import (
    complained_accounts_visible,
    labelled_accounts_visible,
    same_bank_counterparties,
)
from features.account_reader import AccountSnapshot, read_account_snapshot

logger = logging.getLogger(__name__)


def _compute_hops_to_target(
    client: Any,
    run_id: str,
    bank_id: str,
    start_account: str,
    target_accounts: frozenset[str],
    max_hops: int,
    max_nodes: int,
    decision_time: datetime,
    arrival_cutoff: datetime,
    settings: AccountFeaturesSettings,
) -> int | None:
    """
    PURPOSE:
        Executes breadth-first search (BFS) over same_bank_counterparties starting from start_account.
        Returns smallest hop count <= max_hops reaching any account in target_accounts.

    CONSTRAINTS:
        - 0 if start_account is in target_accounts.
        - None if no target account is reachable within max_hops.
        - If max_nodes is reached, terminates traversal, logs a warning, and returns None.
    """
    if start_account in target_accounts:
        return 0

    if not target_accounts:
        return None

    visited: set[str] = {start_account}
    nodes_visited_count = 1
    queue: deque[tuple[str, int]] = deque([(start_account, 0)])

    while queue:
        curr_node, curr_hops = queue.popleft()
        if curr_hops >= max_hops:
            continue

        neighbors = same_bank_counterparties(
            client, run_id, bank_id, curr_node, decision_time, arrival_cutoff, settings
        )

        for nbr in sorted(neighbors):
            if nbr not in visited:
                visited.add(nbr)
                nodes_visited_count += 1
                if nodes_visited_count > max_nodes:
                    logger.warning(
                        "network_max_nodes_visited reached (%d > %d) during BFS on account %s",
                        nodes_visited_count,
                        max_nodes,
                        start_account,
                    )
                    return None
                if nbr in target_accounts:
                    return curr_hops + 1
                queue.append((nbr, curr_hops + 1))

    return None


def compute_shared_counterparties_with_flagged_7d(
    client: Any,
    run_id: str,
    bank_id: str,
    snapshot: AccountSnapshot,
    decision_time: datetime,
    arrival_cutoff: datetime,
    settings: AccountFeaturesSettings,
) -> int:
    """
    PURPOSE:
        Counts distinct counterparties in this account's 7-day raw window (any bank, as 'bank:vpa' strings)
        that ALSO appear as counterparties in the 7-day raw window of at least one labelled account at this bank.
    """
    this_cps: set[str] = set()
    for rec in snapshot.side_records:
        if rec.status == "success" and rec.counterparty_vpa:
            this_cps.add(f"{rec.counterparty_bank_id}:{rec.counterparty_vpa}")

    if not this_cps:
        return 0

    labelled = labelled_accounts_visible(client, run_id, bank_id, decision_time, arrival_cutoff)
    if not labelled:
        return 0

    flagged_cps: set[str] = set()
    for lbl_acc in sorted(labelled):
        lbl_snap = read_account_snapshot(
            client, run_id, bank_id, lbl_acc, decision_time, arrival_cutoff, settings
        )
        for rec in lbl_snap.side_records:
            if rec.status == "success" and rec.counterparty_vpa:
                flagged_cps.add(f"{rec.counterparty_bank_id}:{rec.counterparty_vpa}")

    return len(this_cps & flagged_cps)


def compute_local_cluster_density(
    client: Any,
    run_id: str,
    bank_id: str,
    account_id: str,
    decision_time: datetime,
    arrival_cutoff: datetime,
    settings: AccountFeaturesSettings,
) -> float | None:
    """
    PURPOSE:
        For this account's same-bank counterparties N (|N| >= 2), computes the share of pairs (a, b) in N
        where b is in a's same-bank counterparties.
        Returns None when |N| < 2. Each unordered pair {a, b} is counted exactly once.
    """
    N = same_bank_counterparties(
        client, run_id, bank_id, account_id, decision_time, arrival_cutoff, settings
    )
    if len(N) < 2:
        return None

    sorted_nodes = sorted(N)
    total_pairs = len(sorted_nodes) * (len(sorted_nodes) - 1) // 2
    connected_pairs = 0

    nbrs_by_node: dict[str, frozenset[str]] = {}
    for node in sorted_nodes:
        nbrs_by_node[node] = same_bank_counterparties(
            client, run_id, bank_id, node, decision_time, arrival_cutoff, settings
        )

    for i in range(len(sorted_nodes)):
        for j in range(i + 1, len(sorted_nodes)):
            a = sorted_nodes[i]
            b = sorted_nodes[j]
            if b in nbrs_by_node[a]:
                connected_pairs += 1

    return float(connected_pairs / total_pairs)


def compute_network_features(
    client: Any,
    run_id: str,
    bank_id: str,
    account_id: str,
    snapshot: AccountSnapshot,
    decision_time: datetime,
    arrival_cutoff: datetime,
    settings: AccountFeaturesSettings | None = None,
) -> dict[str, Any]:
    """
    PURPOSE:
        Computes all M1 network group features for an account at a decision moment:
          - hops_to_confirmed_label
          - shared_counterparties_with_flagged_7d
          - local_cluster_density
          - hops_to_complained_account
    """
    cfg = settings if settings is not None else get_default_settings()
    labelled = labelled_accounts_visible(client, run_id, bank_id, decision_time, arrival_cutoff)
    complained = complained_accounts_visible(client, run_id, bank_id, decision_time, arrival_cutoff)

    hops_confirmed = _compute_hops_to_target(
        client=client,
        run_id=run_id,
        bank_id=bank_id,
        start_account=account_id,
        target_accounts=labelled,
        max_hops=cfg.network_max_hops,
        max_nodes=cfg.network_max_nodes_visited,
        decision_time=decision_time,
        arrival_cutoff=arrival_cutoff,
        settings=cfg,
    )

    hops_complained = _compute_hops_to_target(
        client=client,
        run_id=run_id,
        bank_id=bank_id,
        start_account=account_id,
        target_accounts=complained,
        max_hops=cfg.network_max_hops,
        max_nodes=cfg.network_max_nodes_visited,
        decision_time=decision_time,
        arrival_cutoff=arrival_cutoff,
        settings=cfg,
    )

    shared_cps = compute_shared_counterparties_with_flagged_7d(
        client=client,
        run_id=run_id,
        bank_id=bank_id,
        snapshot=snapshot,
        decision_time=decision_time,
        arrival_cutoff=arrival_cutoff,
        settings=cfg,
    )

    cluster_density = compute_local_cluster_density(
        client=client,
        run_id=run_id,
        bank_id=bank_id,
        account_id=account_id,
        decision_time=decision_time,
        arrival_cutoff=arrival_cutoff,
        settings=cfg,
    )

    return {
        "hops_to_confirmed_label": hops_confirmed,
        "shared_counterparties_with_flagged_7d": shared_cps,
        "local_cluster_density": cluster_density,
        "hops_to_complained_account": hops_complained,
    }
