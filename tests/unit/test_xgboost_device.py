"""
WHAT: Tests the xgboost_device config switch, the CUDA build check and that the chosen device
reaches every model the account trainers fit.

WHY: A run that records device "cuda" must really have trained on a GPU, and the default "cpu" must
leave every prediction bit-for-bit as before. CI has no GPU, so nothing here trains on one: the
tests check the device that is handed down and the refusal paths, never GPU numerics.
"""

from pathlib import Path
from typing import Any

import numpy
import pytest
import xgboost
import yaml

import decisions.pu_learning as pu_learning
from decisions.mule_account_model import (
    load_mule_model_config,
    require_cuda_build,
    train_account_model,
)
from decisions.pu_learning import train_elkan_noto_model, train_pu_bagged_model

REPO_ROOT = Path(__file__).resolve().parents[2]
COMMITTED_CONFIG = REPO_ROOT / "config" / "mule_model.yaml"
SEED = 11


def _config_with_device(tmp_path: Path, device: object) -> Path:
    """Returns a copy of the committed config whose xgboost_device block carries the given value."""
    raw = yaml.safe_load(COMMITTED_CONFIG.read_text(encoding="utf-8"))
    raw["parameters"]["xgboost_device"]["value"] = device
    path = tmp_path / "mule_model_device.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def _matrix_and_targets() -> tuple[numpy.ndarray, list[int]]:
    """Returns a seeded random float32 matrix and targets with every fifth row positive."""
    matrix = numpy.random.default_rng(2024).random((120, 4)).astype(numpy.float32)
    return matrix, [1 if index % 5 == 0 else 0 for index in range(120)]


def test_committed_config_defaults_to_cpu() -> None:
    assert load_mule_model_config(COMMITTED_CONFIG).xgboost_device == "cpu"


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_config_accepts_cpu_and_cuda(tmp_path: Path, device: str) -> None:
    assert load_mule_model_config(_config_with_device(tmp_path, device)).xgboost_device == device


@pytest.mark.parametrize("bad_device", ["gpu", "CUDA", "cuda:0", "", 1, None])
def test_config_refuses_any_other_device(tmp_path: Path, bad_device: object) -> None:
    with pytest.raises(ValueError, match="xgboost_device"):
        load_mule_model_config(_config_with_device(tmp_path, bad_device))


def test_trainer_refuses_an_unknown_device() -> None:
    matrix, targets = _matrix_and_targets()

    with pytest.raises(ValueError, match="device must be one of"):
        train_account_model(matrix, targets, SEED, 2, device="gpu")


def test_require_cuda_build_raises_when_the_build_has_no_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(xgboost, "build_info", lambda: {"USE_CUDA": False})

    with pytest.raises(ValueError, match="USE_CUDA"):
        require_cuda_build()


def test_cuda_training_on_a_build_without_cuda_never_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(xgboost, "build_info", lambda: {"USE_CUDA": False})
    matrix, targets = _matrix_and_targets()

    with pytest.raises(ValueError, match="USE_CUDA"):
        train_account_model(matrix, targets, SEED, 2, device="cuda")


def test_require_cuda_build_accepts_a_cuda_build(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(xgboost, "build_info", lambda: {"USE_CUDA": True})

    require_cuda_build()


def test_explicit_cpu_device_is_bit_identical_to_the_default_call() -> None:
    matrix, targets = _matrix_and_targets()
    probe = numpy.random.default_rng(99).random((40, 4)).astype(numpy.float32)

    explicit, _ = train_account_model(matrix, targets, SEED, 5, device="cpu")
    default, _ = train_account_model(matrix, targets, SEED, 5)

    numpy.testing.assert_array_equal(explicit.predict_proba(probe), default.predict_proba(probe))


def _record_devices(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replaces the trainer the PU module calls with a fake that records every device it receives."""
    devices: list[str] = []

    def fake_train(rows: Any, targets: Any, seed: int, n_estimators: int = 100, *, device: str = "cpu", **_: Any):  # type: ignore[no-untyped-def]
        devices.append(device)
        model = xgboost.XGBClassifier(n_estimators=1, random_state=seed)
        model.fit(numpy.asarray(rows, dtype=numpy.float32), targets)
        return model, 1.0

    monkeypatch.setattr(pu_learning, "train_account_model", fake_train)
    return devices


def test_every_bagging_member_receives_the_device(monkeypatch: pytest.MonkeyPatch) -> None:
    devices = _record_devices(monkeypatch)
    matrix, targets = _matrix_and_targets()

    train_pu_bagged_model(matrix, targets, SEED, 2, 3, 2, device="cuda")

    assert devices == ["cuda", "cuda", "cuda"]


def test_both_elkan_noto_models_receive_the_device(monkeypatch: pytest.MonkeyPatch) -> None:
    devices = _record_devices(monkeypatch)
    matrix, targets = _matrix_and_targets()

    train_elkan_noto_model(matrix, targets, matrix, targets, SEED, 2, 0.95, device="cuda")

    assert devices == ["cuda", "cuda"]
