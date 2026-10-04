# Level 5 - Triton gRPC vs HTTP inference micro-benchmark

## What this measures (and what it does not)

The same model (`fraud_xgboost`, ONNX on Triton, CPU) is called with the same single-row inputs over Triton's gRPC endpoint and over Triton's HTTP endpoint, one request at a time. This isolates the cost of the TRANSPORT. It is a micro-benchmark, not a load test: there is no concurrency, so it says nothing about throughput or behaviour under load (real performance and load work belongs to later levels). It is also not comparable with the Level 1 numbers, which measured a trivial `/health` endpoint at 100 connections.

## Method

- Both transports run from ONE throwaway container on the compose network, talking to `triton:8001` (gRPC) and `triton:8000` (HTTP) directly. Host port 8000 belongs to the FastAPI service and Triton's HTTP port is deliberately not published, so this is the only like-for-like way to reach both. Both share the same network path, so neither is handicapped by the Docker/WSL port relay.
- 2,000 timed requests per transport after 200 warm-up requests; 1,000 distinct real test-split rows (50 fraud, 950 non-fraud; seed 42), one row per request.
- For every request the transports take turns going first, so slow drift in the machine affects both equally.
- Each timing covers the full client-side request: build the input, send, receive, read the answer.
- Default client settings for both (binary tensor encoding on HTTP). The two transports returned identical probabilities on every row (largest difference 0.0e+00).
- The benchmark was run 3 times to show run-to-run spread.

### Run 1

| Transport | p50 ms | p95 ms | p99 ms | mean ms | min ms | max ms |
|---|---|---|---|---|---|---|
| grpc | 0.600 | 1.479 | 3.530 | 0.744 | 0.304 | 8.586 |
| http | 0.559 | 1.179 | 1.946 | 0.658 | 0.273 | 12.932 |

### Run 2

| Transport | p50 ms | p95 ms | p99 ms | mean ms | min ms | max ms |
|---|---|---|---|---|---|---|
| grpc | 0.479 | 0.957 | 4.102 | 0.587 | 0.294 | 6.976 |
| http | 0.456 | 0.839 | 3.267 | 0.534 | 0.282 | 5.737 |

### Run 3

| Transport | p50 ms | p95 ms | p99 ms | mean ms | min ms | max ms |
|---|---|---|---|---|---|---|
| grpc | 0.473 | 1.708 | 4.425 | 0.647 | 0.301 | 7.634 |
| http | 0.453 | 0.857 | 3.581 | 0.535 | 0.277 | 4.981 |

### Median across runs

| Transport | p50 ms | p95 ms | p99 ms | mean ms |
|---|---|---|---|---|
| grpc | 0.479 | 1.479 | 4.102 | 0.647 |
| http | 0.456 | 0.857 | 3.267 | 0.535 |

## Reading the result

At one request at a time, HTTP had the lower median latency here: median p50 gRPC 0.479 ms vs HTTP 0.456 ms (HTTP faster at p50 in 3 of 3 runs, at p95 in 3 of 3, at p99 in 3 of 3). The p50 gap is only a few hundredths of a millisecond, so the typical request is comparable; the tails differ more (median p99 gRPC 4.102 ms vs HTTP 3.267 ms). So at this concurrency gRPC shows no latency advantage over HTTP. The usual reasons to prefer gRPC (streaming, multiplexing many concurrent requests on one connection, lower per-request overhead under heavy load) are not exercised by this test. The absolute numbers move between runs (the machine is shared with other software), which is why the comparison is made within each run, with the transports alternating.

## For reference: the client path from the host

The committed client (`serving/fraud_client.py`) runs on the host and reaches Triton through the published gRPC port (`localhost:8001`), which adds Docker's port relay. Same inputs, gRPC only, 2,000 requests: p50 0.618 ms, p95 1.220 ms, p99 3.286 ms, mean 0.815 ms. This number is NOT comparable with the tables above (different network path) and is shown only to give a sense of the real client's latency.
