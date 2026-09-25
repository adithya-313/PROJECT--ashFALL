# PROJECT--ashFALL

PROJECT--ashFALL is an enterprise-grade, high-throughput fraud detection system designed to deliver real-time, sub-millisecond fraud scoring coupled with asynchronous, deep agentic investigations.

## Hardware & Environment Prerequisites

### GPU Acceleration Requirement (Level 4.5 Prerequisite)
The system is targeted to operate on a single NVIDIA GeForce RTX 4050 Laptop GPU (6GB VRAM, ~4–5GB usable) with 16GB system RAM.

Before executing or testing GPU-accelerated training and inference workloads (starting at Level 4.5), the host system and Docker daemon must be verified for GPU passthrough support using the provided verification script:

```bash
bash scripts/verify_gpu.sh
```

**Requirement**: `scripts/verify_gpu.sh` runs `docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi`. This script must exit with status code `0` and display the NVIDIA GPU information before attempting any Level 4.5 components.

## Local Development & Services

The application and supporting infrastructure run via Docker Compose:

- **api**: FastAPI ASGI application running under Gunicorn with Uvicorn workers and uvloop.
- **postgres**: PostgreSQL 15 with logical replication enabled (`wal_level=logical`).
- **redis**: Redis 7 (alpine) caching layer.
- **prometheus**: Metrics collection server scraping service endpoints.
- **grafana**: Observability dashboards pre-provisioned with the Prometheus datasource.

### Starting Services

```bash
docker compose up -d
```
