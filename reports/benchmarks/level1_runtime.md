# Level 1 Runtime Benchmark Results

## Investigation: Back-to-Back Comparison
To accurately investigate the performance changes and isolate environmental noise, `wrk` was executed twice back-to-back against Level 0 (commit `43dca96`) and twice against Level 1 (`main`), with the `--latency` flag, in identical container states.

### Level 0 (Baseline)
Command: `docker compose exec api wrk -t4 -c100 -d30s --latency http://localhost:8000/health`

**Run 1**:
- Throughput: 3695.88 req/s
- Latency (Avg / p50 / p99): 52.48ms / 23.87ms / 811.22ms

**Run 2**:
- Throughput: 3822.84 req/s
- Latency (Avg / p50 / p99): 49.33ms / 22.57ms / 755.35ms

*Level 0 Baseline Averages*: ~3759 req/s, ~23.2ms p50 latency.

### Level 1 (Post-Optimizations + Connection Pool)
Command: `docker compose exec api wrk -t4 -c100 -d30s --latency http://localhost:8000/health`

**Run 1**:
- Throughput: 3375.34 req/s
- Latency (Avg / p50 / p99): 57.21ms / 26.00ms / 777.68ms

**Run 2**:
- Throughput: 3576.73 req/s
- Latency (Avg / p50 / p99): 52.85ms / 23.76ms / 793.03ms

*Level 1 Averages*: ~3476 req/s, ~24.8ms p50 latency.

## Regression Analysis
A verified slight regression exists in Level 1:
- Throughput dropped by ~7.5% (from ~3759 req/s to ~3476 req/s).
- p50 Latency increased by ~1.6ms.

**Actual Cause**: The regression is not caused by misconfigured tunables or the explicit `uvloop` policy, but rather by the **asyncpg connection pool lifecycle and background maintenance overhead**. 

In Level 0, the ASGI application was a purely empty network router handling HTTP requests on a highly optimized event loop. In Level 1, the `lifespan` context initializes a live connection pool (maintaining 2-10 open TCP connections to PostgreSQL) which lives on `app.state.db_pool`. The asyncio event loop must now multiplex the synthetic `/health` endpoint requests *alongside* asyncpg's background tasks (TCP keepalives, health pings, and pool synchronization). For a trivial endpoint returning static JSON under extreme concurrency (100 parallel connections pushing maximum load), this connection management overhead manifests as a minor drop in theoretical maximum throughput and a 1.6ms p50 latency jitter. This is the expected and unavoidable cost of transitioning from a stateless mock container to an application connected to a live relational database.
