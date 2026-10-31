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
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from features.account_keys import (
    AccountFeaturesSettings,
    get_default_settings,
)
from features.account_labels import (
    complained_accounts_visible,
    labelled_accounts_visible,
    same_bank_counterparties_many,
)
from features.account_reader import AccountSnapshot, read_side_records_many

logger = logging.getLogger(__name__)


def _neighbours_of(
    client: Any,
    run_id: str,
    bank_id: str,
    account_ids: Sequence[str],
    decision_time: datetime,
    arrival_cutoff: datetime,
    settings: AccountFeaturesSettings,
    memo: dict[str, frozenset[str]],
) -> dict[str, frozenset[str]]:
    """
    WHAT: Same-bank neighbours of each account, fetching only the accounts not yet in `memo` (in one batch)
    and recording them there.

    WHY: Within one compute_network_features call the run, bank, times and settings never change, so an
    account's neighbours are the same every time they are asked for; the memo is that call's own dict.
    """
    missing = [account_id for account_id in dict.fromkeys(account_ids) if account_id not in memo]
    if missing:
        memo.update(
            same_bank_counterparties_many(client, run_id, bank_id, missing, decision_time, arrival_cutoff, settings)
        )
    return {account_id: memo[account_id] for account_id in account_ids}


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
    memo: dict[str, frozenset[str]] | None = None,
) -> int | None:
    """
    PURPOSE:
        Executes breadth-first search (BFS) over same_bank_counterparties starting from start_account.
        Returns smallest hop count <= max_hops reaching any account in target_accounts.
        Each hop level's accounts are looked up together (one batch per level); accounts are still
        expanded in first-seen order and their neighbours in sorted order, as a one-at-a-time queue would.

    CONSTRAINTS:
        - 0 if start_account is in target_accounts.
        - None if no target account is reachable within max_hops.
        - If max_nodes is reached, terminates traversal, logs a warning, and returns None.
    """
    if start_account in target_accounts:
        return 0

    if not target_accounts:
        return None

    neighbour_memo = memo if memo is not None else {}
    visited: set[str] = {start_account}
    nodes_visited_count = 1
    frontier: list[str] = [start_account]

    for curr_hops in range(max_hops):
        if not frontier:
            break
        neighbours_by_node = _neighbours_of(
            client, run_id, bank_id, frontier, decision_time, arrival_cutoff, settings, neighbour_memo
        )
        next_frontier: list[str] = []
        for curr_node in frontier:
            for nbr in sorted(neighbours_by_node[curr_node]):
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
                    next_frontier.append(nbr)
        frontier = next_frontier

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
    records_by_account = read_side_records_many(
        client, run_id, bank_id, sorted(labelled), decision_time, arrival_cutoff, settings
    )
    for lbl_acc in sorted(labelled):
        for rec in records_by_account[lbl_acc]:
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
    memo: dict[str, frozenset[str]] | None = None,
) -> float | None:
    """
    PURPOSE:
        For this account's same-bank counterparties N (|N| >= 2), computes the share of pairs (a, b) in N
        where b is in a's same-bank counterparties.
        Returns None when |N| < 2. Each unordered pair {a, b} is counted exactly once.
    """
    neighbour_memo = memo if memo is not None else {}
    N = _neighbours_of(
        client, run_id, bank_id, [account_id], decision_time, arrival_cutoff, settings, neighbour_memo
    )[account_id]
    if len(N) < 2:
        return None

    sorted_nodes = sorted(N)
    total_pairs = len(sorted_nodes) * (len(sorted_nodes) - 1) // 2
    connected_pairs = 0

    nbrs_by_node = _neighbours_of(
        client, run_id, bank_id, sorted_nodes, decision_time, arrival_cutoff, settings, neighbour_memo
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
    # One memo for this call only: both searches and the cluster density ask for many of the same accounts.
    neighbour_memo: dict[str, frozenset[str]] = {}

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
        memo=neighbour_memo,
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
        memo=neighbour_memo,
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
        memo=neighbour_memo,
    )

    return {
        "hops_to_confirmed_label": hops_confirmed,
        "shared_counterparties_with_flagged_7d": shared_cps,
        "local_cluster_density": cluster_density,
        "hops_to_complained_account": hops_complained,
    }
