# File: benchmarks/benchmark_concurrency.py
# Purpose: Measures Triton's gRPC versus HTTP inference latency and throughput at several levels of concurrency
#          (requests in flight at once), using identical single-row inputs. A transport micro-benchmark, not the
#          production load test (that is a later level).
import argparse
import asyncio
import json
import multiprocessing
import os
import time
from pathlib import Path

import numpy as np

MODEL_NAME = "fraud_xgboost"
INPUT_NAME = "input"
OUTPUT_NAME = "probabilities"
N_FEATURES = 15

# How many separate client processes share each concurrency level. One Python process can only use one CPU core, so
# at high concurrency a single process could become the bottleneck instead of the transport. More processes
# spread the client work. Tasks per process = concurrency / processes.
PROCESSES_FOR_LEVEL = {1: 1, 10: 2, 50: 5, 100: 5}
REQUESTS_FOR_LEVEL = {1: 5000, 10: 20000, 50: 20000, 100: 20000}  # timed requests per transport per level
WARMUP_TOTAL = 500  # untimed requests per transport per level (connections opening, caches filling)


async def make_grpc_call(url: str, tasks: int):
    """Open an asyncio gRPC client; return (call, close). call(row) sends one row and returns its fraud probability."""
    import tritonclient.grpc.aio as grpc_aio

    client = grpc_aio.InferenceServerClient(url=url)
    output = grpc_aio.InferRequestedOutput(OUTPUT_NAME)

    async def call(row: np.ndarray) -> float:
        request = grpc_aio.InferInput(INPUT_NAME, [1, N_FEATURES], "FP32")
        request.set_data_from_numpy(row.reshape(1, N_FEATURES))
        response = await client.infer(MODEL_NAME, [request], outputs=[output])
        return float(response.as_numpy(OUTPUT_NAME)[0, 1])

    return call, client.close


async def make_http_call(url: str, tasks: int):
    """Open an asyncio HTTP client; return (call, close). call(row) sends one row and returns its fraud probability."""
    import tritonclient.http.aio as http_aio

    # conn_limit is the size of the connection pool; give it room for every task plus a little spare.
    client = http_aio.InferenceServerClient(url=url, conn_limit=tasks + 5)
    output = http_aio.InferRequestedOutput(OUTPUT_NAME)

    async def call(row: np.ndarray) -> float:
        request = http_aio.InferInput(INPUT_NAME, [1, N_FEATURES], "FP32")
        request.set_data_from_numpy(row.reshape(1, N_FEATURES))
        response = await client.infer(MODEL_NAME, [request], outputs=[output])
        return float(response.as_numpy(OUTPUT_NAME)[0, 1])

    return call, client.close


CALL_MAKERS = {"grpc": make_grpc_call, "http": make_http_call}


async def run_closed_loop(call, rows: np.ndarray, tasks: int, n_requests: int, record: bool) -> dict:
    """Keep `tasks` requests in flight until n_requests have finished; return the latencies, errors and timing."""
    latencies: list[float] = []
    errors = 0
    next_index = 0  # shared counter; safe because everything runs on one event loop

    async def worker():
        nonlocal next_index, errors
        while True:
            index = next_index
            if index >= n_requests:
                return
            next_index += 1
            start = time.perf_counter()
            try:
                await call(rows[index % len(rows)])
            except Exception:  # noqa: BLE001 - a failed request is counted, not hidden
                errors += 1
                continue
            if record:
                latencies.append((time.perf_counter() - start) * 1000.0)

    started = time.perf_counter()
    await asyncio.gather(*[worker() for _ in range(tasks)])
    return {"latencies": latencies, "errors": errors, "start": started, "end": time.perf_counter()}


def client_process(transport, url, rows, tasks, n_requests, warmup, barrier, results) -> None:
    """Body of one client process: warm up, wait for the others, then run the timed requests and report back."""

    async def main():
        call, close = await CALL_MAKERS[transport](url, tasks)
        try:
            await run_closed_loop(call, rows, tasks, warmup, record=False)
            # Wait until every client process has finished warming up, so they all start the timed part together.
            await asyncio.get_running_loop().run_in_executor(None, barrier.wait)
            return await run_closed_loop(call, rows, tasks, n_requests, record=True)
        finally:
            await close()

    results.put(asyncio.run(main()))


def run_level(transport: str, url: str, rows: np.ndarray, level: int) -> dict:
    """Run one (transport, concurrency level) measurement across several client processes."""
    processes = PROCESSES_FOR_LEVEL[level]
    tasks = level // processes
    per_process_requests = REQUESTS_FOR_LEVEL[level] // processes
    per_process_warmup = max(WARMUP_TOTAL // processes, tasks)

    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(processes)
    results = context.Queue()
    workers = [
        context.Process(target=client_process, args=(transport, url, rows, tasks, per_process_requests,
                                                     per_process_warmup, barrier, results))
        for _ in range(processes)
    ]
    for worker in workers:
        worker.start()
    outcomes = [results.get() for _ in workers]
    for worker in workers:
        worker.join()
    return summarize(outcomes, level, processes)


def summarize(outcomes: list[dict], level: int, processes: int) -> dict:
    """Combine the per-process results into latency percentiles and overall throughput."""
    latencies = np.concatenate([np.asarray(o["latencies"]) for o in outcomes])
    wall_seconds = max(o["end"] for o in outcomes) - min(o["start"] for o in outcomes)
    return {
        "concurrency": level, "processes": processes, "completed": int(len(latencies)),
        "errors": int(sum(o["errors"] for o in outcomes)),
        "p50_ms": float(np.percentile(latencies, 50)), "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)), "mean_ms": float(latencies.mean()),
        "max_ms": float(latencies.max()), "throughput_rps": float(len(latencies) / wall_seconds),
    }


async def check_same_answers(grpc_url: str, http_url: str, rows: np.ndarray, count: int = 20) -> float:
    """Send the first rows through both transports once and return the largest difference in the answers."""
    grpc_call, grpc_close = await make_grpc_call(grpc_url, 1)
    http_call, http_close = await make_http_call(http_url, 1)
    try:
        gaps = [abs(await grpc_call(rows[i]) - await http_call(rows[i])) for i in range(count)]
    finally:
        await grpc_close()
        await http_close()
    return max(gaps)


def parse_arguments() -> argparse.Namespace:
    """Read the command-line options."""
    parser = argparse.ArgumentParser(description="Triton gRPC vs HTTP concurrency benchmark")
    parser.add_argument("--rows-npy", type=Path, required=True)
    parser.add_argument("--grpc-url", default="triton:8001")
    parser.add_argument("--http-url", default="triton:8000")
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 10, 50, 100])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    """Run every level for both transports, several times, alternating which transport goes first."""
    args = parse_arguments()
    rows = np.load(args.rows_npy).astype(np.float32)
    urls = {"grpc": args.grpc_url, "http": args.http_url}
    print(f"client CPUs: {os.cpu_count()} | rows: {len(rows)} | levels: {args.levels} | repeats: {args.repeats}")

    gap = asyncio.run(check_same_answers(args.grpc_url, args.http_url, rows))
    print(f"Largest difference between the transports' answers on 20 rows: {gap:.2e}")

    runs = []
    for repeat in range(args.repeats):
        # Alternate which transport runs first from one repeat to the next, so slow drift favours neither.
        order = ["grpc", "http"] if repeat % 2 == 0 else ["http", "grpc"]
        run = {"repeat": repeat + 1, "order": order, "levels": {}}
        for level in args.levels:
            run["levels"][str(level)] = {}
            for transport in order:
                stats = run_level(transport, urls[transport], rows, level)
                run["levels"][str(level)][transport] = stats
                print(
                    f"run {repeat + 1} c={level:<3} {transport:<4} p50 {stats['p50_ms']:7.3f}  "
                    f"p95 {stats['p95_ms']:7.3f}  p99 {stats['p99_ms']:8.3f} ms  "
                    f"{stats['throughput_rps']:9.0f} req/s  errors {stats['errors']}",
                    flush=True,
                )
                time.sleep(1.0)
        runs.append(run)

    if args.output_json is not None:
        args.output_json.write_text(json.dumps({"answer_gap": gap, "levels": args.levels, "runs": runs,
                                                "grpc_url": args.grpc_url, "http_url": args.http_url,
                                                "requests_for_level": REQUESTS_FOR_LEVEL,
                                                "processes_for_level": PROCESSES_FOR_LEVEL,
                                                "warmup_total": WARMUP_TOTAL}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
