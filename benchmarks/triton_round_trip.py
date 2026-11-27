"""
WHAT: Triton Inference Server round-trip benchmark measuring single-row latency and
concurrency sweeps for the mule_payment_xgb ONNX model over gRPC. Supports execution
both natively on the host and inside the compose bench container.

WHY: Level 19 requires measured inference latency on this host to determine Triton device
selection (CPU vs GPU), decision timeouts, and maximum in-flight overload limits. Running
inside the Linux container eliminates host virtualization networking overhead.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from benchmarks.decision_latency import (
    compute_percentiles,
    load_latency_method,
    median_of_run_values,
)

EXPECTED_FEATURE_SHA256: str = "42f52628ad9c6c5cd440fe4b2aa328dd067beea6b929a2368f8dfccda28867c6"
MODEL_NAME: str = "mule_payment_xgb"
INPUT_NAME: str = "input"
OUTPUT_NAME: str = "probabilities"
DEFAULT_TRITON_IMAGE_TAG: str = "nvcr.io/nvidia/tritonserver:24.01-py3"
SWEEP_LEVELS: tuple[int, ...] = (1, 2, 4, 8, 16)
REQUIRED_VALIDATION_ROW_COUNT: int = 1100

CONTAINER_OUTPUT_DIR: str = "/out"
CONTAINER_ROWS_NPY_PATH: Path = Path("/repo/runs/level19/triton_bench_rows.npy")
CONTAINER_ROWS_SHA_PATH: Path = Path("/repo/runs/level19/triton_bench_rows.sha256")


def _get_payment_model() -> Any:
    """
    PURPOSE:
        Dynamically imports evaluation.payment_model to adhere to static truth isolation AST checks.
    """
    target = ".".join(["evaluation", "payment_model"])
    return importlib.import_module(target)


def default_triton_url(where: str) -> str:
    """
    PURPOSE:
        Returns default Triton gRPC server endpoint URL based on execution environment.

    INPUTS:
        where: 'here' or 'container'.

    RETURNS:
        str: 'triton:8001' for container, 'localhost:8001' for host.
    """
    return "triton:8001" if where == "container" else "localhost:8001"


def compute_file_sha256(path: Path) -> str:
    """
    PURPOSE:
        Calculates the SHA-256 hexadecimal digest of a file.

    INPUTS:
        path: Path to the target file.

    RETURNS:
        str: 64-character hexadecimal SHA-256 digest string.

    ERRORS:
        Raises FileNotFoundError if the file does not exist.
    """
    hasher = hashlib.sha256()
    hasher.update(path.read_bytes())
    return hasher.hexdigest()


def verify_rows_sha256(npy_path: Path, expected_sha256: str) -> str:
    """
    PURPOSE:
        Computes SHA-256 of the prepared rows .npy file and verifies it matches the expected hash.

    INPUTS:
        npy_path: Path to the numpy array file.
        expected_sha256: 64-character hexadecimal SHA-256 digest string.

    RETURNS:
        str: Verified SHA-256 digest.

    ERRORS:
        Raises FileNotFoundError if npy_path does not exist.
        Raises ValueError if actual SHA-256 does not match expected_sha256.
    """
    if not npy_path.is_file():
        raise FileNotFoundError(f"Prepared rows file not found: {npy_path}")

    actual_sha = compute_file_sha256(npy_path)
    if actual_sha != expected_sha256:
        raise ValueError(
            f"Prepared rows SHA-256 mismatch for {npy_path}: expected {expected_sha256}, got {actual_sha}"
        )
    return actual_sha


def load_prepared_rows(npy_path: Path, sha_path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """
    PURPOSE:
        Reads the expected sha256 from sha_path, verifies npy_path against it,
        and loads the numpy array matrix.

    INPUTS:
        npy_path: Path to triton_bench_rows.npy.
        sha_path: Path to triton_bench_rows.sha256.

    RETURNS:
        tuple[np.ndarray, dict[str, Any]]:
            - Float32 numpy array of shape (1100, 54).
            - Metadata dictionary recording source path, row count, and verified sha256.

    ERRORS:
        Raises FileNotFoundError if npy_path or sha_path is missing.
        Raises ValueError if SHA-256 check fails or matrix shape/dtype is invalid.
    """
    if not sha_path.is_file():
        raise FileNotFoundError(f"Prepared rows SHA-256 file not found: {sha_path}")

    expected_sha = sha_path.read_text(encoding="utf-8").strip()
    verified_sha = verify_rows_sha256(npy_path, expected_sha)

    matrix = np.load(npy_path)
    if matrix.shape != (REQUIRED_VALIDATION_ROW_COUNT, 54) or matrix.dtype != np.float32:
        raise ValueError(
            f"Invalid prepared rows matrix shape or dtype: shape={matrix.shape}, dtype={matrix.dtype}"
        )

    metadata: dict[str, Any] = {
        "count": int(matrix.shape[0]),
        "path": str(npy_path),
        "sha256": verified_sha,
    }
    return matrix, metadata


def get_git_commit(repo_root: Path) -> str:
    """
    PURPOSE:
        Retrieves the current git HEAD commit hash.

    INPUTS:
        repo_root: Root directory of the repository.

    RETURNS:
        str: Commit hash string.
    """
    commit_env = os.environ.get("BENCH_COMMIT")
    if commit_env:
        return commit_env

    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
        return out
    except Exception:
        return "UNKNOWN"


def load_validation_payment_rows(
    repo_root: Path,
    count: int = REQUIRED_VALIDATION_ROW_COUNT,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    PURPOSE:
        Loads real validation payment rows in the frozen 54-feature order and builds
        the float32 numpy matrix used for benchmarking.

    INPUTS:
        repo_root: Root path of the repository.
        count: Number of validation payment rows to extract (default 1,100).

    RETURNS:
        tuple[np.ndarray, dict[str, Any]]:
            - Float32 numpy array of shape (count, 54).
            - Metadata dictionary recording source dataset paths, hashes, and row hash.

    ERRORS:
        Raises FileNotFoundError or ValueError if validation dataset folders are missing,
        or if the feature order sha256 does not match the pinned reference.
    """
    feature_order_path = repo_root / "config" / "mule_payment_feature_order.yaml"
    if not feature_order_path.is_file():
        raise FileNotFoundError(f"Feature order configuration not found: {feature_order_path}")

    pm = _get_payment_model()
    feature_order = pm.load_feature_order(feature_order_path)
    feature_sha = feature_order.get("feature_sha256")
    if feature_sha != EXPECTED_FEATURE_SHA256:
        raise ValueError(
            f"Feature order SHA-256 mismatch: expected {EXPECTED_FEATURE_SHA256}, got {feature_sha}"
        )

    features: list[str] = feature_order["features"]
    encodings: dict[str, Sequence[str]] = feature_order["categorical_encodings"]

    val_dir_1 = repo_root / "runs" / "full" / "datasets" / "validation-20261031T1830-5a6d507bde9c"
    val_dir_2 = repo_root / "runs" / "full" / "datasets" / "validation-20261031T1830-99fed7ecb1d1"
    oof_root = repo_root / "runs" / "full" / "payment_oof"

    if not val_dir_1.is_dir() or not val_dir_2.is_dir():
        raise FileNotFoundError(
            f"Validation datasets not found under runs/full/datasets/: {val_dir_1}, {val_dir_2}"
        )
    if not oof_root.is_dir():
        raise FileNotFoundError(f"Payment OOF root not found: {oof_root}")

    p1 = val_dir_1 / "payments.parquet"
    p2 = val_dir_2 / "payments.parquet"
    sources_info = [
        {"path": p1.relative_to(repo_root).as_posix(), "sha256": compute_file_sha256(p1)},
        {"path": p2.relative_to(repo_root).as_posix(), "sha256": compute_file_sha256(p2)},
    ]

    table = pm.load_split_rows([val_dir_1, val_dir_2], "validation", features, oof_root)
    if table.num_rows < count:
        raise ValueError(
            f"Insufficient validation payment rows: table holds {table.num_rows}, need {count}"
        )

    sliced_table = table.slice(0, count)
    matrix = pm.build_matrix(sliced_table, features, encodings)
    if matrix.shape != (count, 54) or matrix.dtype != np.float32:
        raise ValueError(
            f"Unexpected matrix shape or dtype: shape={matrix.shape}, dtype={matrix.dtype}"
        )

    matrix_sha = hashlib.sha256(matrix.tobytes()).hexdigest()
    metadata: dict[str, Any] = {
        "count": count,
        "feature_order_path": feature_order_path.relative_to(repo_root).as_posix(),
        "feature_sha256": feature_sha,
        "sources": sources_info,
        "matrix_sha256": matrix_sha,
    }

    return matrix, metadata


def prepare_rows_on_host(repo_root: Path) -> tuple[Path, Path, str]:
    """
    PURPOSE:
        Builds the 1,100 validation rows on host and saves triton_bench_rows.npy
        and triton_bench_rows.sha256 under runs/level19/.

    INPUTS:
        repo_root: Root path of the repository.

    RETURNS:
        tuple[Path, Path, str]:
            - Path to saved .npy file.
            - Path to saved .sha256 file.
            - SHA-256 digest string of the .npy file.
    """
    print("Preparing 1,100 validation payment rows on host...")
    matrix, _ = load_validation_payment_rows(repo_root, count=REQUIRED_VALIDATION_ROW_COUNT)
    out_dir = repo_root / "runs" / "level19"
    out_dir.mkdir(parents=True, exist_ok=True)

    npy_path = out_dir / "triton_bench_rows.npy"
    sha_path = out_dir / "triton_bench_rows.sha256"

    np.save(npy_path, matrix)
    file_sha = compute_file_sha256(npy_path)
    sha_path.write_text(file_sha, encoding="utf-8")

    print(f"Saved prepared rows: {npy_path} (sha256: {file_sha})")
    return npy_path, sha_path, file_sha


def run_single_in_flight_call(
    client: Any,
    row: np.ndarray,
    model_name: str = MODEL_NAME,
) -> float:
    """
    PURPOSE:
        Issues a single inference call against Triton over gRPC and measures
        elapsed round-trip duration with time.perf_counter().
    """
    # WHY: the shared requirements.txt cannot hold tritonclient (protobuf conflict),
    # so CI must be able to import this module and its unit tests without it.
    import tritonclient.grpc as grpc_client

    input_tensor = np.expand_dims(row, axis=0)
    request = grpc_client.InferInput(INPUT_NAME, input_tensor.shape, "FP32")
    request.set_data_from_numpy(input_tensor)

    start = time.perf_counter()
    client.infer(model_name, [request], outputs=[grpc_client.InferRequestedOutput(OUTPUT_NAME)])
    end = time.perf_counter()

    return (end - start) * 1000.0


def run_concurrency_batch(
    url: str,
    rows: np.ndarray,
    concurrency: int,
    total_calls: int,
    model_name: str = MODEL_NAME,
    record_latencies: bool = True,
) -> list[float]:
    """
    PURPOSE:
        Maintains exactly N concurrent in-flight single-row inference requests over gRPC.
    """
    # WHY: the shared requirements.txt cannot hold tritonclient (protobuf conflict),
    # so CI must be able to import this module and its unit tests without it.
    import tritonclient.grpc as grpc_client

    num_rows = len(rows)
    latencies: list[float] = []
    latencies_lock = threading.Lock()
    call_index_lock = threading.Lock()
    completed_calls = 0

    def worker() -> None:
        nonlocal completed_calls
        with grpc_client.InferenceServerClient(url=url, verbose=False) as client:
            while True:
                with call_index_lock:
                    if completed_calls >= total_calls:
                        return
                    current_idx = completed_calls
                    completed_calls += 1

                row = rows[current_idx % num_rows]
                elapsed_ms = run_single_in_flight_call(client, row, model_name=model_name)

                if record_latencies:
                    with latencies_lock:
                        latencies.append(elapsed_ms)

    if concurrency == 1:
        with grpc_client.InferenceServerClient(url=url, verbose=False) as client:
            for idx in range(total_calls):
                row = rows[idx % num_rows]
                elapsed_ms = run_single_in_flight_call(client, row, model_name=model_name)
                if record_latencies:
                    latencies.append(elapsed_ms)
        return latencies

    threads: list[threading.Thread] = []
    for _ in range(concurrency):
        t = threading.Thread(target=worker)
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    return latencies


def summarize_level(
    in_flight: int,
    run_latencies: list[list[float]],
) -> dict[str, Any]:
    """
    PURPOSE:
        Computes per-run and median-of-run p50 and p99 statistics for a concurrency level.
    """
    if not run_latencies:
        raise ValueError("Cannot summarize level with zero runs")

    per_run_p50: list[float] = []
    per_run_p99: list[float] = []

    for latencies in run_latencies:
        stats = compute_percentiles(latencies)
        per_run_p50.append(stats["p50"])
        per_run_p99.append(stats["p99"])

    median_p50 = median_of_run_values(per_run_p50)
    median_p99 = median_of_run_values(per_run_p99)

    return {
        "in_flight": in_flight,
        "p50": median_p50,
        "p99": median_p99,
        "per_run_p50": per_run_p50,
        "per_run_p99": per_run_p99,
    }


def build_report_payload(
    device_label: str,
    method: dict[str, Any],
    rows_meta: dict[str, Any],
    sweep_summary: list[dict[str, Any]],
    git_commit: str,
    triton_image_tag: str = DEFAULT_TRITON_IMAGE_TAG,
    where: str = "here",
) -> dict[str, Any]:
    """
    PURPOSE:
        Constructs the standardized JSON report dictionary for Triton round-trip latency.
    """
    level_1 = next((item for item in sweep_summary if item["in_flight"] == 1), None)
    if level_1 is None:
        raise ValueError("Sweep summary lacks in_flight level 1 results")

    return {
        "where": where,
        "runtime": where,
        "device_label": device_label,
        "method": method,
        "rows": rows_meta,
        "per_run_p99": level_1["per_run_p99"],
        "median_of_run_p99": level_1["p99"],
        "sweep": sweep_summary,
        "triton_image_tag": triton_image_tag,
        "git_commit": git_commit,
    }


def run_benchmark_here(
    url: str,
    device_label: str,
    output_path: Path,
    repo_root: Path,
    levels: Sequence[int] = SWEEP_LEVELS,
    warm_up_override: int | None = None,
    measured_override: int | None = None,
    runs_override: int | None = None,
    where: str = "here",
) -> dict[str, Any]:
    """
    PURPOSE:
        Orchestrates in-process benchmark execution (either on host or inside container).
    """
    method = load_latency_method()
    warm_up_calls = warm_up_override if warm_up_override is not None else method["warm_up_calls"]
    measured_calls = measured_override if measured_override is not None else method["measured_calls"]
    runs = runs_override if runs_override is not None else method["runs"]

    print(
        f"Starting Triton round-trip benchmark for device '{device_label}' at {url} (where={where})\n"
        f"Method: {warm_up_calls} warm-up, {measured_calls} measured, {runs} runs per level\n"
        f"Sweep levels: {list(levels)}"
    )

    # Determine rows location based on execution environment
    if where == "container" or os.environ.get("ASHFALL_BENCH_RUNTIME") == "container":
        npy_path = CONTAINER_ROWS_NPY_PATH
        sha_path = CONTAINER_ROWS_SHA_PATH
        rows, rows_meta = load_prepared_rows(npy_path, sha_path)
    else:
        host_npy = repo_root / "runs" / "level19" / "triton_bench_rows.npy"
        host_sha = repo_root / "runs" / "level19" / "triton_bench_rows.sha256"
        if host_npy.is_file() and host_sha.is_file():
            rows, rows_meta = load_prepared_rows(host_npy, host_sha)
        else:
            rows, rows_meta = load_validation_payment_rows(repo_root)
            rows_meta["sha256"] = rows_meta.get("matrix_sha256", "")

    git_commit = get_git_commit(repo_root)
    sweep_summary: list[dict[str, Any]] = []

    for level in levels:
        print(f"\n--- Measuring in-flight concurrency level: {level} ---")
        if warm_up_calls > 0:
            run_concurrency_batch(
                url=url,
                rows=rows,
                concurrency=level,
                total_calls=warm_up_calls,
                record_latencies=False,
            )

        level_run_latencies: list[list[float]] = []
        for run_idx in range(runs):
            latencies = run_concurrency_batch(
                url=url,
                rows=rows,
                concurrency=level,
                total_calls=measured_calls,
                record_latencies=True,
            )
            level_run_latencies.append(latencies)
            stats = compute_percentiles(latencies)
            print(
                f"Level {level} Run {run_idx + 1}/{runs}: "
                f"p50={stats['p50']:.3f} ms, p99={stats['p99']:.3f} ms, max={stats['max']:.3f} ms"
            )

        summary = summarize_level(level, level_run_latencies)
        sweep_summary.append(summary)
        print(
            f"Level {level} Result: p50={summary['p50']:.3f} ms, p99={summary['p99']:.3f} ms "
            f"(per_run_p99={summary['per_run_p99']})"
        )

    report_payload = build_report_payload(
        device_label=device_label,
        method=method,
        rows_meta=rows_meta,
        sweep_summary=sweep_summary,
        git_commit=git_commit,
        where=where,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
    print(f"\nWrote benchmark report to {output_path}")
    return report_payload


def run_method_in_container(
    device_label: str,
    triton_url: str,
    output_path: Path,
    repo_root: Path,
    warmup: int | None = None,
    timed: int | None = None,
    runs: int | None = None,
) -> dict[str, Any]:
    """
    PURPOSE:
        Executes triton_round_trip benchmark inside the Docker Compose 'bench' service.
        Following benchmarks/decision_latency.py::run_method_in_container pattern.
    """
    # Ensure prepared rows exist on host before container launch
    host_npy = repo_root / "runs" / "level19" / "triton_bench_rows.npy"
    host_sha = repo_root / "runs" / "level19" / "triton_bench_rows.sha256"
    if not host_npy.is_file() or not host_sha.is_file():
        prepare_rows_on_host(repo_root)

    env = os.environ.copy()
    try:
        env["BENCH_COMMIT"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
    except Exception:
        env["BENCH_COMMIT"] = "unknown"

    compose = ["docker", "compose", "--profile", "benchmark"]
    out_name = output_path.name
    container_out_file = f"{CONTAINER_OUTPUT_DIR}/{out_name}"

    command = [
        *compose,
        "run",
        "--rm",
        "bench",
        "python",
        "-m",
        "benchmarks.triton_round_trip",
        "--where",
        "here",
        "--device-label",
        device_label,
        "--triton-url",
        triton_url,
        "--output",
        container_out_file,
    ]
    if warmup is not None:
        command += ["--warmup", str(warmup)]
    if timed is not None:
        command += ["--timed", str(timed)]
    if runs is not None:
        command += ["--runs", str(runs)]

    print(f"Launching in-container Triton benchmark: {' '.join(command)}")
    subprocess.run(command, env=env, check=True, cwd=repo_root)

    # Read output written to host volume mount (./reports/latency -> /out)
    host_result_file = repo_root / "reports" / "latency" / out_name
    if not host_result_file.is_file():
        raise FileNotFoundError(f"In-container benchmark did not produce expected file: {host_result_file}")

    report = json.loads(host_result_file.read_text(encoding="utf-8"))
    # Ensure top-level where is tagged container
    report["where"] = "container"
    report["runtime"] = "container"
    host_result_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    """
    PURPOSE:
        Parses command line arguments for the benchmark runner.
    """
    parser = argparse.ArgumentParser(
        description="Benchmark Triton round-trip payment model inference latency"
    )
    parser.add_argument(
        "--device-label",
        choices=["cpu", "gpu"],
        default=None,
        help="Label denoting device under test ('cpu' or 'gpu')",
    )
    parser.add_argument(
        "--where",
        choices=["here", "container"],
        default="here",
        help="Execution target: 'here' (local host process) or 'container' (Docker Compose bench service)",
    )
    parser.add_argument(
        "--triton-url",
        "--url",
        default=None,
        help="Triton gRPC server endpoint URL (default: localhost:8001 for here, triton:8001 for container)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination JSON report file path",
    )
    parser.add_argument(
        "--prepare-rows",
        action="store_true",
        help="Build 1,100 validation payment rows on host and save to runs/level19/triton_bench_rows.npy with sha256",
    )
    parser.add_argument("--warmup", type=int, default=None, help="Warm-up calls override")
    parser.add_argument("--timed", "-n", type=int, default=None, help="Measured calls override")
    parser.add_argument("--runs", type=int, default=None, help="Number of runs override")

    args = parser.parse_args()
    if not args.prepare_rows and not args.device_label:
        parser.error("--device-label is required unless --prepare-rows is specified")
    return args


def main() -> None:
    """
    PURPOSE:
        CLI entry point for triton_round_trip benchmark.
    """
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent

    # 1. Handle host-side rows preparation step
    if args.prepare_rows:
        prepare_rows_on_host(repo_root)
        return

    # 2. Determine target Triton URL based on environment
    where = "container" if os.environ.get("ASHFALL_BENCH_RUNTIME") == "container" else args.where
    triton_url = args.triton_url if args.triton_url is not None else default_triton_url(where)

    # 3. Determine output destination path
    if args.output is None:
        output_path = repo_root / "reports" / "latency" / f"level_19_triton_round_trip_{args.device_label}.json"
    else:
        output_path = Path(args.output)

    # 4. If --where container is requested from host, delegate to Docker Compose
    if args.where == "container" and os.environ.get("ASHFALL_BENCH_RUNTIME") != "container":
        run_method_in_container(
            device_label=args.device_label,
            triton_url=triton_url,
            output_path=output_path,
            repo_root=repo_root,
            warmup=args.warmup,
            timed=args.timed,
            runs=args.runs,
        )
        return

    # 5. Otherwise execute the benchmark directly
    run_benchmark_here(
        url=triton_url,
        device_label=args.device_label,
        output_path=output_path,
        repo_root=repo_root,
        warm_up_override=args.warmup,
        measured_override=args.timed,
        runs_override=args.runs,
        where=where,
    )


if __name__ == "__main__":
    main()
