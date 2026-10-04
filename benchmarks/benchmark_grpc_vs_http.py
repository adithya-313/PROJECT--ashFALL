# File: benchmarks/benchmark_grpc_vs_http.py
# Purpose: Micro-benchmark of Triton's gRPC versus HTTP transport for the SAME model and the SAME single-row inputs,
#          measured one request at a time (no concurrency). It isolates transport cost; it is not a load test.
import argparse
import json
import time
from pathlib import Path

import numpy as np

MODEL_NAME = "fraud_xgboost"
INPUT_NAME = "input"
OUTPUT_NAME = "probabilities"
N_FEATURES = 15


def make_grpc_caller(url: str):
    """Return a function that sends one row over gRPC and returns the fraud probability."""
    import tritonclient.grpc as grpc_client

    client = grpc_client.InferenceServerClient(url=url)
    output = grpc_client.InferRequestedOutput(OUTPUT_NAME)

    def call(row: np.ndarray) -> float:
        request = grpc_client.InferInput(INPUT_NAME, [1, N_FEATURES], "FP32")
        request.set_data_from_numpy(row.reshape(1, N_FEATURES))
        response = client.infer(MODEL_NAME, [request], outputs=[output])
        return float(response.as_numpy(OUTPUT_NAME)[0, 1])

    return call


def make_http_caller(url: str):
    """Return a function that sends one row over HTTP and returns the fraud probability."""
    import tritonclient.http as http_client

    client = http_client.InferenceServerClient(url=url)
    output = http_client.InferRequestedOutput(OUTPUT_NAME)

    def call(row: np.ndarray) -> float:
        request = http_client.InferInput(INPUT_NAME, [1, N_FEATURES], "FP32")
        request.set_data_from_numpy(row.reshape(1, N_FEATURES))
        response = client.infer(MODEL_NAME, [request], outputs=[output])
        return float(response.as_numpy(OUTPUT_NAME)[0, 1])

    return call


def time_one(call, row: np.ndarray) -> tuple[float, float]:
    """Time one full request (build, send, receive, read the answer) in milliseconds; also return the answer."""
    start = time.perf_counter()
    answer = call(row)
    return (time.perf_counter() - start) * 1000.0, answer


def run_benchmark(callers: dict, rows: np.ndarray, n_requests: int, warmup: int) -> dict:
    """Send the same rows through every transport, alternating which goes first, and collect the latencies."""
    names = list(callers)

    # Warm-up: the first requests are slower (connections opening, caches filling). They are not counted.
    for i in range(warmup):
        for name in names:
            callers[name](rows[i % len(rows)])

    latencies = {name: [] for name in names}
    answers = {name: [] for name in names}
    for i in range(n_requests):
        row = rows[i % len(rows)]
        # Alternate the order each time so any slow drift in the machine hits both transports equally.
        order = names if i % 2 == 0 else names[::-1]
        for name in order:
            milliseconds, answer = time_one(callers[name], row)
            latencies[name].append(milliseconds)
            answers[name].append(answer)
    return {"latencies": latencies, "answers": answers}


def summarize(latencies_ms: list[float]) -> dict:
    """p50, p95, p99, mean, min and max of a list of latencies (milliseconds)."""
    values = np.asarray(latencies_ms)
    return {
        "p50_ms": float(np.percentile(values, 50)), "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)), "mean_ms": float(values.mean()),
        "min_ms": float(values.min()), "max_ms": float(values.max()), "n": int(len(values)),
    }


def max_answer_gap(answers: dict) -> float:
    """Largest difference between the fraud probabilities two transports returned for the same rows."""
    names = list(answers)
    if len(names) < 2:
        return 0.0
    return float(np.max(np.abs(np.asarray(answers[names[0]]) - np.asarray(answers[names[1]]))))


def parse_arguments() -> argparse.Namespace:
    """Read the command-line options."""
    parser = argparse.ArgumentParser(description="Triton gRPC vs HTTP single-row latency micro-benchmark")
    parser.add_argument("--rows-npy", type=Path, required=True, help="float32 array of shape (rows, 15)")
    parser.add_argument("--grpc-url", default="triton:8001")
    parser.add_argument("--http-url", default="triton:8000")
    parser.add_argument("--transports", nargs="+", default=["grpc", "http"], choices=["grpc", "http"])
    parser.add_argument("--n", type=int, default=2000, help="timed requests per transport")
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    """Run the benchmark and print (and optionally save) the results."""
    args = parse_arguments()
    rows = np.load(args.rows_npy).astype(np.float32)
    assert rows.ndim == 2 and rows.shape[1] == N_FEATURES, f"Unexpected rows shape {rows.shape}"

    makers = {"grpc": lambda: make_grpc_caller(args.grpc_url), "http": lambda: make_http_caller(args.http_url)}
    callers = {name: makers[name]() for name in args.transports}
    outcome = run_benchmark(callers, rows, args.n, args.warmup)

    summary = {name: summarize(values) for name, values in outcome["latencies"].items()}
    gap = max_answer_gap(outcome["answers"])
    print(f"Requests per transport: {args.n} (after {args.warmup} warm-up), distinct input rows: {len(rows)}, "
          f"one row per request, one request at a time")
    print(f"{'transport':<10}{'p50 ms':>10}{'p95 ms':>10}{'p99 ms':>10}{'mean ms':>10}{'min ms':>10}{'max ms':>10}")
    for name, s in summary.items():
        print(f"{name:<10}{s['p50_ms']:>10.3f}{s['p95_ms']:>10.3f}{s['p99_ms']:>10.3f}{s['mean_ms']:>10.3f}"
              f"{s['min_ms']:>10.3f}{s['max_ms']:>10.3f}")
    print(f"Largest difference between the transports' fraud probabilities on the same rows: {gap:.3e}")

    if args.output_json is not None:
        report = {"n_requests": args.n, "warmup": args.warmup, "rows": int(len(rows)), "summary": summary,
                  "max_answer_gap": gap, "grpc_url": args.grpc_url, "http_url": args.http_url}
        args.output_json.write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
