# File: tests/integration/test_triton_io_contract.py
# Purpose: Freezes the I/O schema of the Triton-served fraud model. Triton's own model metadata is the contract
#          surface: if the served input/output names, types or shapes ever drift, this test fails.
#
# The live tests need Triton running (docker compose --profile serving up -d --no-deps triton). When it is not
# reachable they are SKIPPED with a clear reason instead of failing, so CI without Triton does not false-fail.
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("tritonclient.grpc")
pytest.importorskip("joblib")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))  # lets fraud_baseline be imported by name, as the scripts do


def load_by_path(name: str, path: Path):
    """Load a module from a file path (serving/ and scripts/ are not packages)."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fraud_client = load_by_path("fraud_client", REPO_ROOT / "serving" / "fraud_client.py")

# The contract. Input and output names, types and shapes of the served model ([-1, ...] means "any batch size").
EXPECTED_INPUT = {"name": "input", "datatype": "FP32", "shape": [-1, 15]}
EXPECTED_OUTPUTS = {
    "label": {"datatype": "INT64", "shape": [-1, 1]},
    "probabilities": {"datatype": "FP32", "shape": [-1, 2]},
}


def triton_is_up() -> bool:
    """True when the Triton gRPC server answers on the usual address and the fraud model is ready."""
    try:
        client = fraud_client.make_client()
        return bool(client.is_server_live() and client.is_model_ready(fraud_client.MODEL_NAME))
    except Exception:  # noqa: BLE001 - any connection problem simply means "Triton is not available"
        return False


needs_triton = pytest.mark.skipif(not triton_is_up(), reason="Triton is not running on localhost:8001 (skipped)")


def training_feature_columns() -> list[str]:
    """The feature columns, in order, exactly as the training pipeline's prepare_features produces them."""
    fraud_baseline = load_by_path("fraud_baseline", REPO_ROOT / "scripts" / "fraud_baseline.py")
    one_row = {column: [1.0] for column in fraud_baseline.AMOUNT_AND_BALANCE_COLUMNS
               + fraud_baseline.ENGINEERED_NUMBER_COLUMNS}
    one_row.update({column: [True] for column in fraud_baseline.ENGINEERED_BOOL_COLUMNS})
    one_row.update({"type": ["TRANSFER"], fraud_baseline.TARGET_COLUMN: [0]})
    features, _ = fraud_baseline.prepare_features(pd.DataFrame(one_row))
    return list(features.columns)


def test_client_feature_order_is_the_training_contract():
    # Needs no server: the client's fixed 15-name list must equal what the training pipeline produces.
    assert fraud_client.FEATURE_ORDER == training_feature_columns()
    assert len(fraud_client.FEATURE_ORDER) == 15


@needs_triton
def test_served_input_schema_is_frozen():
    metadata = fraud_client.make_client().get_model_metadata(fraud_client.MODEL_NAME)
    assert len(metadata.inputs) == 1
    served = metadata.inputs[0]
    assert served.name == EXPECTED_INPUT["name"]
    assert served.datatype == EXPECTED_INPUT["datatype"]
    assert list(served.shape) == EXPECTED_INPUT["shape"]
    # Tie the 15 to the fixed feature list so the two cannot drift apart.
    assert served.shape[1] == len(fraud_client.FEATURE_ORDER)


@needs_triton
def test_served_output_schema_is_frozen():
    metadata = fraud_client.make_client().get_model_metadata(fraud_client.MODEL_NAME)
    served = {output.name: {"datatype": output.datatype, "shape": list(output.shape)} for output in metadata.outputs}
    assert served == EXPECTED_OUTPUTS


@needs_triton
def test_served_model_identity_and_batch_limit():
    client = fraud_client.make_client()
    metadata = client.get_model_metadata(fraud_client.MODEL_NAME)
    assert metadata.name == "fraud_xgboost"
    assert list(metadata.versions) == ["1"]
    assert client.get_model_config(fraud_client.MODEL_NAME).config.max_batch_size == fraud_client.MAX_BATCH
