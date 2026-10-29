# File: Dockerfile
# Purpose: Defines the container environment for the Python service.
# Connects: docker-compose.yml uses this file to build the 'api', 'outbox-relay',
#           and 'nationalsim' services.
#           The CMD here is what docker-compose runs when the container starts.

FROM python:3.11-slim

# Install wrk for HTTP benchmarking.
RUN apt-get update -q && apt-get install -y --no-install-recommends wrk && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first to allow Docker to cache the pip install layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source and infrastructure files after pip install.
COPY api/ api/
COPY core/ core/
COPY infra/ infra/

# The nationalsim and api Compose services reuse this image and need their code and settings files.
# Copy only those two config files (nationalsim.yaml and bank_api.yaml), not the whole config/ folder.
COPY nationalsim/ nationalsim/
COPY config/nationalsim.yaml config/nationalsim.yaml
COPY config/bank_api.yaml config/bank_api.yaml

# PYTHONPATH=/app ensures "from core.config import config" and "api.main:app" resolve.
ENV PYTHONPATH=/app

# Port gunicorn binds to inside the container.
EXPOSE 8000

# Gunicorn boots reading config from infra/docker/gunicorn_conf.py targeting api.main:app
CMD ["gunicorn", "-c", "infra/docker/gunicorn_conf.py", "api.main:app"]
