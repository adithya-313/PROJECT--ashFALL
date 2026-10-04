"""
File: core/mlflow_setup.py
Purpose: Small reusable helper that points MLflow at the local SQLite database (mlflow.db) and selects an experiment.

Why this exists: every ML sub-phase (baseline bake-off, tuning, evaluation) must log to the SAME place, so the
tracking-URI and experiment setup lives here once instead of being repeated in each script.

Why SQLite and not the ./mlruns folder: MLflow 3.15 treats the file-folder backend as maintenance-mode and refuses
it by default, so the supported local option is a single SQLite file. It still needs no separate server process.

How to see the runs: from the repo root run `mlflow ui --backend-store-uri sqlite:///mlflow.db` and open the printed
address (normally http://127.0.0.1:5000).
"""

import os
from pathlib import Path

import mlflow

# The repo root is one level above core/. Used so a relative database path like mlflow.db always means
# <repo>/mlflow.db, no matter which folder a script is launched from.
REPO_ROOT = Path(__file__).resolve().parent.parent

# Default location of the local database. Can be overridden with the MLFLOW_TRACKING_URI environment variable.
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"

SQLITE_URI_PREFIX = "sqlite:///"


def resolve_tracking_uri() -> str:
    """Return the tracking URI to use, turning a relative SQLite path into an absolute one under the repo root."""
    uri = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)

    # Only SQLite URIs need fixing. Anything else (http://..., a server URL) is passed through unchanged.
    if not uri.startswith(SQLITE_URI_PREFIX):
        return uri

    path_text = uri[len(SQLITE_URI_PREFIX):]

    # An absolute path (like C:/data/mlflow.db or /data/mlflow.db) is already unambiguous, so leave it alone.
    if Path(path_text).is_absolute() or path_text.startswith("/"):
        return uri

    # A relative path like mlflow.db is resolved against the repo root so every script uses the same database file,
    # no matter which folder it is launched from.
    return SQLITE_URI_PREFIX + (REPO_ROOT / path_text).resolve().as_posix()


def setup_mlflow(experiment_name: str) -> str:
    """Point MLflow at the local SQLite database, select (or create) the experiment, and return the tracking URI."""
    tracking_uri = resolve_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    return tracking_uri
