# File: Dockerfile
# Purpose: Defines the container environment for the Python service.
# Connects: docker-compose.yml uses this file to build the 'app' image.
#           The CMD here is what docker-compose runs when the container starts.
#
# Phase 1a: Boots uvicorn directly (baseline).
# Phase 1b: Will be switched to gunicorn+uvicorn workers.
#
# wrk (the HTTP benchmarking tool) is installed here so that benchmarks can
# be run from inside the container against localhost, avoiding any network
# overhead from benchmarking across Docker's virtual bridge from the host.

FROM python:3.11-slim

# Install wrk for HTTP benchmarking.
# wrk is available in Debian's apt repository and installs cleanly
# in python:3.11-slim without requiring compilation.
RUN apt-get update -q && apt-get install -y --no-install-recommends wrk && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first so Docker can cache the pip install layer.
# If only source code changes, Docker will skip re-running pip install.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code after pip install so code changes don't bust the package cache.
COPY src/ src/

# PYTHONPATH=/app so that "from src.core.config import config" resolves
# to /app/src/core/config.py inside the container.
ENV PYTHONPATH=/app

# Port the ASGI server will listen on inside the container.
# docker-compose.yml maps this to a host port.
EXPOSE 8000

# Phase 1a: Boot uvicorn directly, pointing at the FastAPI 'app' object
# inside src/main.py. --host 0.0.0.0 makes it reachable from outside the container.
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
