# File: replay/pipeline.py
# Purpose: Replay pipeline utilities.
"""
WHAT: Feed preparation pipeline for replay evaluation runs.
WHY: Encapsulates projecting simulation output into a BankFeed populated with complaint label events.
"""

from replay.runner import load_default_kyc_phone_on_account_opened
from world.bank_feed import BankFeed, project_run
from world.label_builder import build_label_events
from world.replay_recording import RecordedWorld

__all__ = ["feed_with_labels"]


def feed_with_labels(recorded_world: RecordedWorld) -> BankFeed:
    """Projects recorded world simulation output into a BankFeed including fraud complaint label events."""
    labels = build_label_events(recorded_world.output)
    return project_run(
        recorded_world.output,
        labels,
        phone_book=recorded_world.phone_book,
        kyc_phone=load_default_kyc_phone_on_account_opened(),
    )

