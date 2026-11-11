"""
Tests that the dataset builder applies label status changes at the cutoff (replay/dataset_builder.py): a label revoked
at or before the cutoff is not counted, one revoked after the cutoff still is (the "future trap"), and a contested
label is still counted and reported in the manifest.

WHY: A cutoff dataset may only use what was known at the cutoff; a revoke that arrives later does not exist yet.

TERM: revoked label: a label the bank later withdrew; it stops counting from the revoke's arrival time.
TERM: contested label: a disputed label; it still counts and is reported.
"""

from dataclasses import replace
from datetime import UTC, datetime

from replay.dataset_builder import build_dataset
from tests.unit.test_dataset_builder import _make_dummy_extract

CUTOFF = datetime(2026, 9, 20, tzinfo=UTC)
RUN = "0123456789abcdef"


def registry_label(label_id: str = f"{RUN}-S000000:phone") -> dict[str, object]:
    """A registry label arriving 12:00 on 10 Sept."""
    return {
        "run_id": RUN, "bank_id": "bank_0", "label_id": label_id, "source": "registry", "account_id": "acc_receiver_1",
        "payment_ids": [], "verdict": "fraud", "arrival_time": datetime(2026, 9, 10, 12, tzinfo=UTC),
        "region": None, "batch_id": None, "claimed_type": None,
    }


def status(change: str, when: datetime, label_id: str = f"{RUN}-S000000:phone") -> dict[str, object]:
    """A status change row for the registry label."""
    return {
        "bank_id": "bank_0", "source": "registry", "label_id": label_id, "account_id": "acc_receiver_1",
        "change": change, "arrival_time": when,
    }


def build(statuses: list[dict[str, object]]):  # type: ignore[no-untyped-def]
    """Builds a dataset at CUTOFF from one registry label and the given status changes."""
    extract = replace(_make_dummy_extract(label_rows=[registry_label()]), label_status_rows=statuses)
    return build_dataset([extract], split="training", cutoff=CUTOFF)


def test_no_status_change_keeps_the_label() -> None:
    """Checks the registry label is counted when nothing was revoked."""
    assert len(build([]).labels) == 1


def test_revoke_before_the_cutoff_drops_the_label_and_its_flag() -> None:
    """Checks a revoke at or before the cutoff removes the label and the receiver's reported flag."""
    ds = build([status("revoked", datetime(2026, 9, 15, tzinfo=UTC))])
    assert len(ds.labels) == 0
    assert ds.manifest["labels_revoked_excluded"] == 1
    assert ds.payments.column("label_receiver_reported_by_cutoff").to_pylist() == [False]


def test_revoke_exactly_at_the_cutoff_drops_the_label() -> None:
    """Checks the boundary: arrival_time equal to the cutoff counts as known."""
    assert len(build([status("revoked", CUTOFF)]).labels) == 0


def test_revoke_after_the_cutoff_keeps_the_label() -> None:
    """Checks the future trap: a revoke that arrives after the cutoff changes nothing at that cutoff."""
    ds = build([status("revoked", datetime(2026, 9, 25, tzinfo=UTC))])
    assert len(ds.labels) == 1
    assert ds.manifest["labels_revoked_excluded"] == 0


def test_dispute_keeps_the_label_and_reports_it_contested() -> None:
    """Checks a contested label is counted and named in the manifest."""
    ds = build([status("contested", datetime(2026, 9, 12, tzinfo=UTC))])
    assert len(ds.labels) == 1
    world_id = ds.manifest["worlds"][0]["world_id"]
    assert ds.manifest["labels_contested"] == [f"{world_id}:bank_0:registry:{RUN}-S000000:phone:acc_receiver_1"]
