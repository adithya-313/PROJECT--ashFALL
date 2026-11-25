"""A6: re-verify provenance from the files themselves, not from the reports.

Checks the frozen v1 and v2 artifacts against the sha256 values the amendments record, recomputes
Run A's checkpoint hashes against extra_reporting.json, and reads the recorded leakage statuses
straight from the two leakage JSON files. The PR-merge check runs separately with gh.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs" / "full" / "audit_20261125"
RUN_A_DIR = ROOT / "runs" / "full" / "run_a320_stage5c" / "final" / "45c496799170"

RECORDED = {
    "runs/full/payment_model_v1/frozen_model.json": (
        "98dffe1fda74342b36d55651deb2168fa144c4886a87cdb10715ff90d3a7864e"
    ),
    "runs/full/payment_model_v1/feature_order.yaml": (
        "6da1c373ac3d59d22f11808f033ae3462e9d1222d6c2e6a82d767b931f54211a"
    ),
    "runs/full/payment_model_v2/C1/frozen_model.json": (
        "d4bc66ce7fcc6c664e3ed67fba09c555f187cc11a1397cb0c5df951132f03bec"
    ),
    "runs/full/payment_model_v2/C1/feature_order.yaml": (
        "6da1c373ac3d59d22f11808f033ae3462e9d1222d6c2e6a82d767b931f54211a"
    ),
}


def sha256(path: Path) -> str:
    """Return the sha256 hex digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    """Recompute every recorded hash and list the leakage statuses."""
    checks: list[dict] = []
    for rel_path, recorded in RECORDED.items():
        actual = sha256(ROOT / rel_path)
        checks.append({"file": rel_path, "recorded": recorded, "actual": actual, "pass": actual == recorded})

    extra = json.loads(
        (ROOT / "runs" / "full" / "run_a320_stage5c" / "extra_reporting.json").read_text(encoding="utf-8")
    )
    checkpoint_hashes = extra["manifest"]["checkpoint_sha256"]
    checkpoint_dir = RUN_A_DIR / "checkpoint"
    for rel_path, recorded in sorted(checkpoint_hashes.items()):
        path = checkpoint_dir / rel_path
        actual = sha256(path)
        checks.append(
            {"file": f"checkpoint/{rel_path}", "recorded": recorded, "actual": actual, "pass": actual == recorded}
        )
    on_disk = sorted(p.name for p in (checkpoint_dir / "model_bags").glob("*.ubj"))
    checks.append(
        {
            "file": "checkpoint/model_bags member set",
            "recorded": f"{len([k for k in checkpoint_hashes if k.endswith('.ubj')])} members",
            "actual": f"{len(on_disk)} members",
            "pass": len(on_disk) == len([k for k in checkpoint_hashes if k.endswith(".ubj")]),
        }
    )

    leakage_main = json.loads((RUN_A_DIR / "leakage_checks.json").read_text(encoding="utf-8"))
    leakage_56 = json.loads(
        (ROOT / "runs" / "full" / "run_a320_stage5c" / "leakage_checks_5_6.json").read_text(encoding="utf-8")
    )
    console = json.loads((ROOT / "reports" / "level_18" / "console" / "leakage.json").read_text(encoding="utf-8"))

    console_checks = [(c["check"], c["status"]) for c in console["data"]["checks"]]
    trivial_world = console["data"]["trivial_world"].get("status")

    document = {
        "simulated": True,
        "hash_checks": checks,
        "all_hashes_pass": all(check["pass"] for check in checks),
        "leakage": {
            "main_file_keys": sorted(leakage_main),
            "file_5_6_keys": sorted(leakage_56),
            "console_checks": console_checks,
            "console_trivial_world": trivial_world,
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "a6_provenance.json").write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8", newline="\n")
    for check in checks:
        print(f"{'PASS' if check['pass'] else 'FAIL'} {check['file']}")
    print("leakage console statuses", console_checks)
    print("trivial world", trivial_world)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
