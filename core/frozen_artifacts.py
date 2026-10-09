"""
File: core/frozen_artifacts.py
Purpose: A guard for the FROZEN model v1 files (the model, its ONNX copy, the SHAP explainer and the calibrator).
         Each of them is tracked by DVC, and the .dvc file next to it records the md5 hash the file must have.
         This module can (1) check that the files on disk still match those hashes, and (2) refuse any attempt to
         write over them.

Why it exists: Level 9 audits model v1 by training COPIES of it. Nothing in that work may change the original files,
and this guard makes that impossible to do by accident.
"""

import hashlib
import os
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# The four frozen model v1 artifacts.
FROZEN_ARTIFACT_PATHS: list[Path] = [
    REPO_ROOT / "models" / "xgboost" / "model.ubj",
    REPO_ROOT / "models" / "xgboost" / "model.onnx",
    REPO_ROOT / "models" / "xgboost" / "shap_model.pkl",
    REPO_ROOT / "models" / "calibration" / "isotonic_calibrator.pkl",
]

CHUNK_SIZE = 1024 * 1024  # read files one megabyte at a time, so big files never fill the memory


class FrozenArtifactError(Exception):
    """Raised when something tries to write to a frozen model v1 file."""


def dvc_file_for(artifact_path: Path) -> Path:
    """The .dvc file that records the hash of an artifact (the artifact's path plus '.dvc')."""
    return artifact_path.with_name(artifact_path.name + ".dvc")


def md5_of_file(path: Path) -> str:
    """The md5 hash of a file, read in chunks."""
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def recorded_md5(dvc_file_path: Path) -> str:
    """
    Read the md5 that a .dvc file records.

    Plain text search on purpose (like scripts/ci_quality_gate.py): PyYAML is not installed in every environment.
    """
    text = dvc_file_path.read_text(encoding="utf-8")
    match = re.search(r"^\s*-?\s*md5:\s*([0-9a-f]{32})\s*$", text, re.MULTILINE)
    if match is None:
        raise ValueError(f"no md5 found in {dvc_file_path}")
    return match.group(1)


def verify_frozen_artifacts(allow_missing: bool = False, artifacts: list[Path] | None = None) -> list[str]:
    """
    Compare every frozen file on disk with the md5 recorded in its .dvc file.

    Returns a list of problems in plain English; an empty list means everything matches. A file that is not on disk is
    a problem unless allow_missing is True (a fresh clone has the .dvc files but not the model files until `dvc pull`).
    """
    problems: list[str] = []
    for path in artifacts if artifacts is not None else FROZEN_ARTIFACT_PATHS:
        expected = recorded_md5(dvc_file_for(path))
        if not path.exists():
            if not allow_missing:
                problems.append(f"{path.name} is missing from disk")
            continue
        actual = md5_of_file(path)
        if actual != expected:
            problems.append(f"{path.name} changed: md5 is {actual}, the .dvc file records {expected}")
    return problems


def protected_paths() -> set[str]:
    """Every protected path (the artifacts and their .dvc files) in a normal form for comparing."""
    paths = FROZEN_ARTIFACT_PATHS + [dvc_file_for(path) for path in FROZEN_ARTIFACT_PATHS]
    return {os.path.normcase(str(path.resolve())) for path in paths}


def refuse_if_frozen(path: Path) -> None:
    """Raise FrozenArtifactError if `path` is a frozen artifact or its .dvc file."""
    if os.path.normcase(str(Path(path).resolve())) in protected_paths():
        raise FrozenArtifactError(f"refusing to write to frozen model v1 file: {path}")


def guarded_write_bytes(path: Path, data: bytes) -> None:
    """Write bytes to a file, but only after checking that the file is not a frozen model v1 artifact."""
    refuse_if_frozen(path)
    Path(path).write_bytes(data)
