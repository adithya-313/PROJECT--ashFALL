# File: scripts/update_baselines.py
# Purpose: Build the "regression baselines" (saved snapshots of things that must not change by accident) and, when you
#          run this file, write them into tests/baselines/. The tests in tests/baselines/ build the same snapshots
#          again and compare them with the saved files.
#
# Run from the repo root, ONLY when a change is intentional:   python scripts/update_baselines.py
# Then look at the git diff of tests/baselines/ and explain it in your pull request.
import sys
from pathlib import Path
from typing import Any

import orjson
import yaml

# Make the repo root importable, so "from api.main import app" works when this file is run directly.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

BASELINE_DIR = REPO_ROOT / "tests" / "baselines"
METRICS_PATH = REPO_ROOT / "reports" / "model" / "metrics.json"
CALIBRATION_PATH = REPO_ROOT / "reports" / "calibration" / "calibration_bins.json"
COMPOSE_PATH = REPO_ROOT / "docker-compose.yml"

MODEL_BASELINE_FILE = "model_v1_metrics_baseline.json"
CALIBRATION_BASELINE_FILE = "calibration_bins_baseline.json"
OPENAPI_BASELINE_FILE = "api_openapi_snapshot.json"
COMPOSE_BASELINE_FILE = "compose_health_baseline.json"

# The keys of reports/model/metrics.json that make up the model baseline. "mlflow_evaluation_run_id" is left out on
# purpose: it changes every time the model is evaluated again, even when the numbers are the same.
MODEL_KEYS = [
    "model_name", "split", "test_rows", "test_fraud_rows", "pr_auc", "roc_auc",
    "dvc_data_md5", "dvc_model_ubj_md5", "dvc_model_onnx_md5",
]

# For these routes the snapshot keeps the full request and response description. Other routes only keep their methods.
DETAILED_PATHS = ["/health", "/score"]


def read_json(path: Path) -> Any:
    """Read a JSON file."""
    return orjson.loads(path.read_bytes())


def build_model_baseline() -> dict[str, Any]:
    """Pick the important values out of the current model metrics file."""
    metrics = read_json(METRICS_PATH)
    return {key: metrics[key] for key in MODEL_KEYS}


def build_calibration_baseline() -> Any:
    """A plain copy of the current calibration bins."""
    return read_json(CALIBRATION_PATH)


def describe_route(path: str, methods: dict[str, Any]) -> dict[str, Any]:
    """Describe one route: its methods, and for the detailed routes also the request and response shapes."""
    if path not in DETAILED_PATHS:
        return {"methods": sorted(methods)}
    details = {}
    for method, operation in methods.items():
        details[method] = {
            "request_body": operation.get("requestBody"),
            "responses": operation.get("responses"),
        }
    return {"methods": sorted(methods), "details": details}


def build_openapi_snapshot() -> dict[str, Any]:
    """
    Snapshot of the API's public shape.

    app.openapi() is called directly (not through a test client), so no database is needed: importing the app does
    not connect to anything, only starting it does.
    """
    from api.main import app  # imported here so that other functions in this file work without the app

    schema = app.openapi()
    routes = {path: describe_route(path, methods) for path, methods in schema["paths"].items()}
    return {
        "metrics_endpoint_present": "/metrics" in schema["paths"],
        "routes": routes,
        "component_schemas": schema.get("components", {}).get("schemas", {}),
    }


def build_compose_baseline() -> dict[str, bool]:
    """For every service in docker-compose.yml: does it define a healthcheck? (A config check; nothing is started.)"""
    compose = yaml.safe_load(COMPOSE_PATH.read_text(encoding="utf-8"))
    return {name: "healthcheck" in service for name, service in compose["services"].items()}


def all_baselines() -> dict[str, Any]:
    """Every baseline, keyed by its file name."""
    return {
        MODEL_BASELINE_FILE: build_model_baseline(),
        CALIBRATION_BASELINE_FILE: build_calibration_baseline(),
        OPENAPI_BASELINE_FILE: build_openapi_snapshot(),
        COMPOSE_BASELINE_FILE: build_compose_baseline(),
    }


def to_json_bytes(data: Any) -> bytes:
    """Stable JSON: sorted keys and 2-space indentation, so a re-run gives byte-identical files."""
    return orjson.dumps(data, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS) + b"\n"


def main() -> int:
    """Write every baseline file."""
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    for file_name, data in all_baselines().items():
        (BASELINE_DIR / file_name).write_bytes(to_json_bytes(data))
        print(f"wrote tests/baselines/{file_name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
