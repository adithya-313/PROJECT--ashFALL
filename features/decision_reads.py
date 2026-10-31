# File: features/decision_reads.py
# Purpose: What one decision reads from Redis once and shares between its feature functions.
# Issue: Level 15 decision latency (duplicate reads within one decision).
"""
WHAT: DecisionReads holds the payer's AccountSnapshot and the visible labelled and complained account lists,
together with the run, bank, account and times they were read with. read_decision_reads() reads them once.

WHY: Within one decision the account features and the payment features read the same payer snapshot three
times and the same label lists twice. The inputs are identical (same run, bank, account, decision time and
arrival cutoff), so reading once changes no value; it only removes round trips.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from features.account_keys import AccountFeaturesSettings
from features.account_labels import complained_accounts_visible, labelled_accounts_visible
from features.account_reader import AccountSnapshot, read_account_snapshot


@dataclass(frozen=True)
class DecisionReads:
    """The reads shared by every feature function of one decision, and the arguments they were read with."""

    run_id: str
    bank_id: str
    account_id: str
    decision_time: datetime
    arrival_cutoff: datetime
    payer_snapshot: AccountSnapshot
    labelled_visible: frozenset[str]
    complained_visible: frozenset[str]

    def require_match(
        self,
        run_id: str,
        bank_id: str,
        account_id: str,
        decision_time: datetime,
        arrival_cutoff: datetime,
    ) -> None:
        """Raise ValueError unless these reads were made with exactly the given arguments."""
        mismatched = [
            name
            for name, held, given in (
                ("run_id", self.run_id, run_id),
                ("bank_id", self.bank_id, bank_id),
                ("account_id", self.account_id, account_id),
                ("decision_time", self.decision_time, decision_time),
                ("arrival_cutoff", self.arrival_cutoff, arrival_cutoff),
            )
            if held != given
        ]
        if mismatched:
            raise ValueError(
                f"DecisionReads were read with different {', '.join(mismatched)} than the feature call asks for"
            )


def read_decision_reads(
    client: Any,
    run_id: str,
    bank_id: str,
    account_id: str,
    decision_time: datetime,
    arrival_cutoff: datetime,
    settings: AccountFeaturesSettings | None = None,
) -> DecisionReads:
    """Read the payer's snapshot and the visible label lists once, using the same readers every feature uses."""
    return DecisionReads(
        run_id=run_id,
        bank_id=bank_id,
        account_id=account_id,
        decision_time=decision_time,
        arrival_cutoff=arrival_cutoff,
        payer_snapshot=read_account_snapshot(
            client, run_id, bank_id, account_id, decision_time, arrival_cutoff, settings
        ),
        labelled_visible=labelled_accounts_visible(client, run_id, bank_id, decision_time, arrival_cutoff),
        complained_visible=complained_accounts_visible(client, run_id, bank_id, decision_time, arrival_cutoff),
    )
