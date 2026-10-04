"""
File: scripts/fraud_dvc_link.py
Purpose: 5.10 - record the link between the DVC-tracked model files and the MLflow runs that produced them.

What it does, in plain words:
  1. Reads the md5 hash that DVC stored in each model file's .dvc pointer file.
  2. Logs ONE new MLflow run holding those hashes plus the IDs of the runs that produced the files
     (finished runs are never edited).
  3. Writes the new run's ID back into each .dvc file as a top-level `meta:` entry, so the link works both ways.

Run from the repo root:  PYTHONPATH=. python scripts/fraud_dvc_link.py
"""

import sys
from pathlib import Path

import mlflow
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.mlflow_setup import setup_mlflow  # noqa: E402

EXPERIMENT_NAME = "level5_reproducibility"
RUN_NAME = "dvc_artifact_link"

# artifact name -> (path of its .dvc file, run that produced it, experiment of that run)
ARTIFACTS = {
    "model_ubj": ("models/xgboost/model.ubj.dvc", "cfd6628cc7fc4001a49f4009a3bd4d19", "level5_fraud_calibration"),
    "model_onnx": ("models/xgboost/model.onnx.dvc", "3ba14af13d31406ea8b613936d337fb5", "level5_fraud_onnx"),
    "isotonic_calibrator": ("models/calibration/isotonic_calibrator.pkl.dvc", "cfd6628cc7fc4001a49f4009a3bd4d19",
                            "level5_fraud_calibration"),
    "shap_model": ("models/xgboost/shap_model.pkl.dvc", "780831d8c7be4feabff20d894a1abd62",
                   "level5_fraud_evaluation"),
}


def read_md5(dvc_file: str) -> str:
    """Return the md5 DVC recorded for the single output in a .dvc file."""
    content = yaml.safe_load((REPO_ROOT / dvc_file).read_text(encoding="utf-8"))
    return content["outs"][0]["md5"]


def add_meta(dvc_file: str, run_id: str) -> None:
    """Append a top-level `meta:` block with the link run ID (skipped if the file already has one)."""
    path = REPO_ROOT / dvc_file
    text = path.read_text(encoding="utf-8")
    if "\nmeta:" in text or text.startswith("meta:"):
        return
    path.write_text(text.rstrip("\n") + f"\nmeta:\n  mlflow_link_run_id: {run_id}\n  mlflow_experiment: "
                    f"{EXPERIMENT_NAME}\n", encoding="utf-8")


def main() -> None:
    setup_mlflow(EXPERIMENT_NAME)
    hashes = {name: read_md5(info[0]) for name, info in ARTIFACTS.items()}
    with mlflow.start_run(run_name=RUN_NAME) as run:
        for name, (dvc_file, producer_run, producer_experiment) in ARTIFACTS.items():
            mlflow.log_param(f"dvc_md5_{name}", hashes[name])
            mlflow.log_param(f"dvc_file_{name}", dvc_file)
            mlflow.log_param(f"producer_run_{name}", producer_run)
            mlflow.log_param(f"producer_experiment_{name}", producer_experiment)
        mlflow.set_tag("dvc_remote", "none (local cache only)")
        run_id = run.info.run_id
    for dvc_file, _, _ in ARTIFACTS.values():
        add_meta(dvc_file, run_id)
    print("link run id:", run_id)
    for name, value in hashes.items():
        print(f"  dvc_md5_{name} = {value}")


if __name__ == "__main__":
    main()
