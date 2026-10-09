"""
File: tests/unit/test_frozen_artifacts.py
Purpose: Proves the frozen-artifact guard: writes to the four model v1 files (and their .dvc files) are refused,
         a changed file is noticed, and the pinned hashes in tests/baselines equal what the .dvc files record. The
         refusal tests never write anything, and the on-disk check only reads.
"""

from pathlib import Path

import orjson
import pytest

from core.frozen_artifacts import (
    FROZEN_ARTIFACT_PATHS,
    FrozenArtifactError,
    dvc_file_for,
    guarded_write_bytes,
    md5_of_file,
    recorded_md5,
    refuse_if_frozen,
    verify_frozen_artifacts,
)

PINNED_HASHES_FILE = Path(__file__).resolve().parent.parent / "baselines" / "model_v1_frozen_hashes.json"
REPO_ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.mark.parametrize("path", FROZEN_ARTIFACT_PATHS, ids=lambda path: path.name)
def test_every_frozen_file_and_its_dvc_file_is_refused(path: Path) -> None:
    """Writing to a frozen artifact, or to the .dvc file that records its hash, raises FrozenArtifactError."""
    with pytest.raises(FrozenArtifactError):
        refuse_if_frozen(path)
    with pytest.raises(FrozenArtifactError):
        refuse_if_frozen(dvc_file_for(path))


def test_guarded_write_refuses_without_writing(tmp_path: Path) -> None:
    """guarded_write_bytes stops before writing: a frozen path that does not exist is still absent afterwards."""
    frozen = FROZEN_ARTIFACT_PATHS[0]
    existed_before = frozen.exists()
    size_before = frozen.stat().st_size if existed_before else None
    with pytest.raises(FrozenArtifactError):
        guarded_write_bytes(frozen, b"dummy bytes")
    assert frozen.exists() == existed_before
    assert (frozen.stat().st_size if existed_before else None) == size_before


def test_a_normal_temp_file_can_be_written(tmp_path: Path) -> None:
    """A path that is not frozen is written normally."""
    target = tmp_path / "scratch.bin"
    guarded_write_bytes(target, b"ok")
    assert target.read_bytes() == b"ok"


def make_fake_artifact(folder: Path, content: bytes) -> Path:
    """A fake artifact file plus a .dvc file that records the md5 of the content as it is now."""
    artifact = folder / "fake_model.bin"
    artifact.write_bytes(content)
    record = f"outs:\n- md5: {md5_of_file(artifact)}\n  size: {len(content)}\n"
    dvc_file_for(artifact).write_text(record, encoding="utf-8")
    return artifact


def test_a_tampered_file_is_detected(tmp_path: Path) -> None:
    """An untouched file passes; after one byte changes, the problem is reported with both hashes."""
    artifact = make_fake_artifact(tmp_path, b"model bytes")
    assert verify_frozen_artifacts(artifacts=[artifact]) == []
    artifact.write_bytes(b"model byteX")
    problems = verify_frozen_artifacts(artifacts=[artifact])
    assert len(problems) == 1 and "changed" in problems[0]


def test_a_missing_file_is_a_problem_unless_allowed(tmp_path: Path) -> None:
    """A missing file is reported by default, and ignored when allow_missing is True."""
    artifact = make_fake_artifact(tmp_path, b"model bytes")
    artifact.unlink()
    assert "missing" in verify_frozen_artifacts(artifacts=[artifact])[0]
    assert verify_frozen_artifacts(allow_missing=True, artifacts=[artifact]) == []


def test_pinned_hashes_equal_the_hashes_in_the_dvc_files() -> None:
    """The hashes pinned in tests/baselines match the .dvc files (this reads only the .dvc files)."""
    pinned = orjson.loads(PINNED_HASHES_FILE.read_bytes())
    assert sorted(pinned) == sorted(path.relative_to(REPO_ROOT).as_posix() for path in FROZEN_ARTIFACT_PATHS)
    for path in FROZEN_ARTIFACT_PATHS:
        assert pinned[path.relative_to(REPO_ROOT).as_posix()] == recorded_md5(dvc_file_for(path)), (
            f"The pinned hash of {path.name} differs from its .dvc file. Frozen artifacts must never change."
        )


def test_frozen_files_on_disk_match_their_hashes() -> None:
    """If the model files are present, they match the recorded hashes. Skipped when they are not on this machine."""
    if not all(path.exists() for path in FROZEN_ARTIFACT_PATHS):
        pytest.skip("the model v1 files are not on disk here (they come from `dvc pull`)")
    assert verify_frozen_artifacts() == []
