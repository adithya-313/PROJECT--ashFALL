# File: benchmarks/profile_serialization.py
# Purpose: Level 6 - measure how much of a Triton gRPC request is spent on client-side serialization ("packing"), at
#          batch sizes 1, 8 and 32, and compare it with the whole round trip. Writes the "serialization_profiling"
#          section of reports/benchmarks/level6_batch.json (other sections are kept).
#
# What each number covers (read this before quoting any of them):
#   packing_ms        client only: build the InferInput and call set_data_from_numpy (checks dtype and shape, then
#                     copies the array into one block of raw bytes). This is the "tensor packing" step.
#   request_build_ms  client only: put the packed tensor into the gRPC request message (the call infer() makes next).
#                     Uses tritonclient's private helper _get_inference_request, so it can change between versions.
#   wire_encode_ms    client only: turn that message into the bytes that go on the network (protobuf SerializeToString).
#   roundtrip_p50_ms  the whole call as the application sees it: our input checks + packing + message build + the
#                     network hop + Triton queue and model compute + reading the answer back. One request in flight.
#
# Run from the repo root with Triton up:  PYTHONPATH=. python benchmarks/profile_serialization.py
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tritonclient.grpc as grpc_client
from tritonclient.grpc import _utils as grpc_utils
from tritonclient.grpc import service_pb2

from serving import fraud_client

REPORT_PATH = Path(__file__).resolve().parent.parent / "reports" / "benchmarks" / "level6_batch.json"
BATCH_SIZES = [1, 8, 32]
PACKING_ITERATIONS = 20_000  # how many times each client-side step is timed
ROUNDTRIP_WARMUP = 100
ROUNDTRIP_REQUESTS = 2000
SEED = 42
N_FEATURES = len(fraud_client.FEATURE_ORDER)


def make_table(batch_size: int) -> np.ndarray:
    """A deterministic float32 table of finite numbers, shaped (batch_size, 15)."""
    return np.random.default_rng(SEED + batch_size).random((batch_size, N_FEATURES), dtype=np.float32)


def median_ms(func, iterations: int) -> float:
    """Time func() `iterations` times (after a short warm-up) and return the median in milliseconds."""
    for _ in range(200):
        func()
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        times.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(times)


def pack(table: np.ndarray) -> grpc_client.InferInput:
    """The packing step exactly as serving/fraud_client.py does it."""
    request_input = grpc_client.InferInput(fraud_client.INPUT_NAME, list(table.shape), "FP32")
    request_input.set_data_from_numpy(table)
    return request_input


def build_request(request_input: grpc_client.InferInput):
    """Build the gRPC request message the way InferenceServerClient.infer does."""
    return grpc_utils._get_inference_request(
        model_name=fraud_client.MODEL_NAME, inputs=[request_input], model_version="", request_id="", outputs=None,
        sequence_id=0, sequence_start=False, sequence_end=False, priority=0, timeout=None, parameters=None)


def build_protobuf_float_request(table: np.ndarray):
    """The ALTERNATIVE encoding, for comparison only: floats inside the protobuf field instead of raw bytes."""
    request = service_pb2.ModelInferRequest()
    request.model_name = fraud_client.MODEL_NAME
    tensor = request.inputs.add()
    tensor.name, tensor.datatype = fraud_client.INPUT_NAME, "FP32"
    tensor.shape.extend(table.shape)
    tensor.contents.fp32_contents.extend(table.ravel().tolist())
    return request


def roundtrip_p50_ms(client, batch_size: int) -> float:
    """Median time of the full client call (the same code path as production), one request in flight."""
    batches = [make_table(batch_size) + i * 1e-6 for i in range(ROUNDTRIP_WARMUP + ROUNDTRIP_REQUESTS)]
    for batch in batches[:ROUNDTRIP_WARMUP]:
        fraud_client.raw_fraud_probability(client, batch.astype(np.float32))
    times = []
    for batch in batches[ROUNDTRIP_WARMUP:]:
        start = time.perf_counter()
        fraud_client.raw_fraud_probability(client, batch.astype(np.float32))
        times.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(times)


def profile_batch_size(client, batch_size: int) -> dict:
    """All measurements for one batch size."""
    table = make_table(batch_size)
    packed = pack(table)
    request = build_request(packed)
    float_request = build_protobuf_float_request(table)

    packing_ms = median_ms(lambda: pack(table), PACKING_ITERATIONS)
    build_ms = median_ms(lambda: build_request(packed), PACKING_ITERATIONS)
    encode_ms = median_ms(request.SerializeToString, PACKING_ITERATIONS)
    float_build_ms = median_ms(lambda: build_protobuf_float_request(table), PACKING_ITERATIONS)
    float_encode_ms = median_ms(float_request.SerializeToString, PACKING_ITERATIONS)
    roundtrip_ms = roundtrip_p50_ms(client, batch_size)

    return {
        "batch_size": batch_size,
        "packing_ms": round(packing_ms, 5),
        "request_build_ms": round(build_ms, 5),
        "wire_encode_ms": round(encode_ms, 5),
        "client_encode_total_ms": round(packing_ms + build_ms + encode_ms, 5),
        "roundtrip_p50_ms": round(roundtrip_ms, 3),
        "packing_fraction_of_roundtrip": round(packing_ms / roundtrip_ms, 5),
        "client_encode_total_fraction_of_roundtrip": round((packing_ms + build_ms + encode_ms) / roundtrip_ms, 5),
        "raw_tensor_bytes": batch_size * N_FEATURES * 4,
        "serialized_request_bytes": len(request.SerializeToString()),
        "raw_input_contents_bytes": len(request.raw_input_contents[0]),
        "alternative_protobuf_floats": {
            "build_plus_encode_ms": round(float_build_ms + float_encode_ms, 5),
            "serialized_request_bytes": len(float_request.SerializeToString()),
        },
    }


def confirm_wire_format() -> dict:
    """Look inside a real request: raw bytes (raw_input_contents) or floats inside the protobuf (contents)?"""
    request = build_request(pack(make_table(8)))
    return {
        "raw_input_contents_entries": len(request.raw_input_contents),
        "raw_input_contents_bytes": len(request.raw_input_contents[0]),
        "protobuf_fp32_contents_values": len(request.inputs[0].contents.fp32_contents),
        "verdict": "raw binary bytes" if request.raw_input_contents and not request.inputs[0].contents.fp32_contents
        else "protobuf float field",
    }


def main() -> int:
    client = grpc_client.InferenceServerClient(url=fraud_client.TRITON_URL)
    config = client.get_model_config(fraud_client.MODEL_NAME).config
    wire = confirm_wire_format()
    print("Wire format check:", wire)

    results = [profile_batch_size(client, n) for n in BATCH_SIZES]
    print(f"{'batch':>5} | {'packing_ms':>10} | {'roundtrip_p50_ms':>16} | {'packing_fraction':>16} | "
          f"{'payload_bytes':>13}")
    for row in results:
        print(f"{row['batch_size']:>5} | {row['packing_ms']:>10.5f} | {row['roundtrip_p50_ms']:>16.3f} | "
              f"{row['packing_fraction_of_roundtrip']:>16.4%} | {row['serialized_request_bytes']:>13}")

    section = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "what": "Client-side serialization cost of a Triton gRPC request versus the whole round trip",
        "wire_format": wire,
        "how_wire_format_was_confirmed": "Built a real request with the client's own helpers and inspected it: "
                                         "the tensor sits in raw_input_contents as bytes (set_data_from_numpy stores "
                                         "numpy tobytes()) and the protobuf fp32_contents field is empty.",
        "method": {"timed_iterations_per_step": PACKING_ITERATIONS, "roundtrip_requests": ROUNDTRIP_REQUESTS,
                   "roundtrip_warmup": ROUNDTRIP_WARMUP, "input_seed": SEED,
                   "roundtrip_measured_now": True,
                   "server_config_during_roundtrip": {
                       "dynamic_batching": config.HasField("dynamic_batching"),
                       "max_queue_delay_microseconds": config.dynamic_batching.max_queue_delay_microseconds},
                   "client": "host python via localhost:8001 (Docker Desktop port forward)"},
        "definitions": {
            "packing_ms": "client only: InferInput + set_data_from_numpy",
            "request_build_ms": "client only: put the packed tensor into the gRPC request message (private helper)",
            "wire_encode_ms": "client only: SerializeToString of the request",
            "roundtrip_p50_ms": "whole fraud_client.raw_fraud_probability call: checks, packing, message, network "
                                "hop, Triton queue + model compute, reading the answer",
            "packing_fraction_of_roundtrip": "packing_ms / roundtrip_p50_ms"},
        "results": results,
        "not_measured": ["shared memory (system or CUDA)", "server-side deserialization on its own",
                         "sending the protobuf-float alternative to Triton (only its client-side encode time and "
                         "size were measured)"],
    }
    report = json.loads(REPORT_PATH.read_text()) if REPORT_PATH.exists() else {}
    report["serialization_profiling"] = section
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote the 'serialization_profiling' section to {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
