# File: Dockerfile
# Purpose: Defines the container environment for the Python service.
# Connects: docker-compose.yml uses this file to build the 'app' image.
#           The CMD here is what docker-compose runs when the container starts.
#
# Phase 1b: Boots gunicorn with uvicorn workers and uvloop event loop.
#           gunicorn manages the worker process pool; each worker runs uvicorn
#           with uvloop (instead of asyncio default) and httptools (instead of h11).
#
# wrk (the HTTP benchmarking tool) is installed so benchmarks can be run from
# inside the container against localhost, avoiding Docker bridge network overhead.

FROM python:3.11-slim

# Install wrk for HTTP benchmarking.
# wrk is in Debian's apt repository — no compilation needed, installs cleanly.
RUN apt-get update -q && apt-get install -y --no-install-recommends wrk && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first to allow Docker to cache the pip install layer.
# If only source code changes, this layer is reused from the build cache.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source after pip install so code edits don't invalidate the package layer.
COPY src/ src/

# PYTHONPATH=/app so that "from src.core.config import config" resolves
# to /app/src/core/config.py inside the container.
ENV PYTHONPATH=/app

# Port gunicorn binds to inside the container (also configured in gunicorn_conf.py).
EXPOSE 8000

# Phase 1b CMD: gunicorn reads its config from src/gunicorn_conf.py.
# That file sets worker_class=uvicorn.workers.UvicornWorker, which activates
# uvloop and httptools automatically when they are installed in the image.
CMD ["gunicorn", "-c", "src/gunicorn_conf.py", "src.main:app"]
