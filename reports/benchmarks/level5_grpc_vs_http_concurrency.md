# Level 5 - Triton gRPC vs HTTP under concurrency (micro-benchmark)

Companion to `level5_grpc_vs_http.md`, which measured one request at a time. This measures the same model, with the same single-row inputs, over Triton's gRPC and HTTP endpoints while several requests are in flight at once. It is a TRANSPORT micro-benchmark on one shared laptop. It is not the production load test (Level 6/10, around 500 users) and its numbers must not be read as production capacity.

## Method

- Client: asyncio clients (`tritonclient.grpc.aio` and `tritonclient.http.aio`), run from one throwaway container on the compose network and talking to `triton:8001` and `triton:8000` directly, so both transports share one network path.
- Concurrency = the number of requests in flight at once (a closed loop: each finished request is replaced by the next). To keep the Python client from becoming the bottleneck, the requests are spread over several client processes (client processes per concurrency level: 1: 1, 10: 2, 50: 5, 100: 5).
- Timed requests per transport per level: 1: 5,000, 10: 20,000, 50: 20,000, 100: 20,000; 500 untimed warm-up requests first. Inputs: the same 1,000 real test-split rows (50 fraud, 950 non-fraud), one row per request.
- Each level was run 3 times, alternating which transport ran first. Tables show the MEDIAN across the runs. A shared laptop is noisy, so look at the direction across runs, not at one figure.
- Both transports returned identical probabilities (largest difference 0.0e+00). Failed requests are counted in the Errors column.
- Latency is measured per request from sending to receiving the answer, so at high concurrency it includes time spent queueing. Throughput = completed requests / elapsed seconds.

## Results (median across runs)

| Concurrency | Transport | p50 ms | p95 ms | p99 ms | mean ms | Throughput req/s | Errors |
|---|---|---|---|---|---|---|---|
| 1 | grpc | 0.522 | 1.017 | 4.012 | 0.622 | 1,606 | 0 |
| 1 | http | 0.592 | 0.870 | 1.336 | 0.605 | 1,652 | 0 |
| 10 | grpc | 2.516 | 6.244 | 9.262 | 2.952 | 3,374 | 0 |
| 10 | http | 3.387 | 7.392 | 10.274 | 3.763 | 2,471 | 0 |
| 50 | grpc | 8.990 | 14.044 | 17.132 | 9.305 | 5,338 | 0 |
| 50 | http | 8.955 | 18.741 | 26.725 | 9.968 | 4,754 | 0 |
| 100 | grpc | 16.575 | 24.097 | 28.494 | 16.478 | 5,993 | 0 |
| 100 | http | 14.468 | 25.588 | 34.776 | 15.329 | 6,093 | 0 |

## Which transport was ahead, run by run

Cells show how many of the 3 runs gRPC was better (lower latency / higher throughput) than HTTP.

| Concurrency | p50 | p95 | p99 | mean | Throughput |
|---|---|---|---|---|---|
| 1 | 3 / 3 | 1 / 3 | 0 / 3 | 1 / 3 | 1 / 3 |
| 10 | 3 / 3 | 3 / 3 | 3 / 3 | 3 / 3 | 3 / 3 |
| 50 | 1 / 3 | 3 / 3 | 3 / 3 | 3 / 3 | 3 / 3 |
| 100 | 0 / 3 | 3 / 3 | 3 / 3 | 0 / 3 | 1 / 3 |

## Crossover finding

- gRPC had higher throughput than HTTP in every run at concurrency [10, 50].
- Throughput direction was mixed between runs (within noise) at concurrency [1, 100].
- gRPC had the lower p99 latency in every run at concurrency [10, 50, 100].
- HTTP had the lower median (p50) latency in every run at concurrency [100].

## Saturation: where the machine, not the transport, sets the limit

- grpc: from concurrency 50 to 100, median throughput changed 1.12x (5,338 to 5,993 req/s) while median p50 latency changed 1.84x (9.0 to 16.6 ms).
- http: from concurrency 50 to 100, median throughput changed 1.28x (4,754 to 6,093 req/s) while median p50 latency changed 1.62x (9.0 to 14.5 ms).
- Container CPU sampled during the whole run (100% = one core; Docker's VM has 12): Triton median 979%, peak 1200%; benchmark client container median 94%, peak 344%.

If throughput stops growing while latency keeps rising (Little's law: latency is roughly concurrency divided by throughput), the system has reached its service rate. Here Triton's container was using essentially all of the VM's CPU cores at the higher levels, and the client shares those same cores. So the highest levels measure the saturation of this machine and of this Triton configuration (one CPU model instance, no dynamic batching, not tuned), not a clean transport difference. 100 is therefore the ceiling for this benchmark; going higher would only measure the machine choking.

## Serving transport decision

**gRPC is the single serving path.** There is no HTTP/gRPC switch.

- At one request at a time the two transports are within a few hundredths of a millisecond of each other, and which one is faster depends on the client library: with the synchronous clients (`level5_grpc_vs_http.md`) HTTP's median was about 0.02 ms lower; with the asyncio clients used here, gRPC's p50 was lower at concurrency 1. Either way the difference is negligible in practice.
- Under concurrency gRPC is ahead where it matters for a hosted service: see the crossover finding and the run-by-run table above.
- An adaptive HTTP/gRPC switch was considered and rejected: it adds complexity for no real payoff. The hard part would be measuring live load to decide when to switch, and HTTP's advantage exists only in a single-request scenario that a production service does not see.
- The real production load test is Level 6/10. This document is a transport micro-benchmark only.
