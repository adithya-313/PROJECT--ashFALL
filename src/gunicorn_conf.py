"""
File: src/gunicorn_conf.py
Purpose: Configuration file for the gunicorn process manager.
Connects: Gunicorn reads this file when launched via the Dockerfile CMD.
          It pulls the worker count from the environment (GUNICORN_WORKERS),
          which is loaded by config.py. All other gunicorn settings are set here.

Why a separate config file instead of CLI flags:
Gunicorn supports a Python config file via the -c flag. This is preferable to
a long CLI string in the Dockerfile CMD because it keeps settings documented,
version-controlled, and editable without rebuilding the image (when mounted).

What this configures:
- worker_class: 'uvicorn.workers.UvicornWorker' makes each gunicorn worker
  run uvicorn's ASGI event loop. This is the recommended production pattern for
  FastAPI — gunicorn manages the process pool, uvicorn handles each HTTP connection.
- worker_connections: how many simultaneous connections each worker can maintain.
- accesslog/errorlog: route gunicorn's own logs to stdout/stderr for Docker capture.
"""

import os

# The module path of the FastAPI 'app' object gunicorn will serve.
# Gunicorn uses this to import and bind the ASGI application.
wsgi_app = "src.main:app"

# Worker class: tells gunicorn to spawn uvicorn-based workers instead of
# standard synchronous workers. Each worker runs its own asyncio event loop,
# with uvloop as the event loop implementation (activated by uvloop.install()
# in the uvicorn worker).
worker_class = "uvicorn.workers.UvicornWorker"

# Number of worker processes. Each worker is a separate OS process and
# can handle requests independently — this achieves true parallelism.
# We read this from the environment so it can be tuned without rebuilding.
# The default (2) is conservative for the RTX 4050 dev machine with 16GB RAM.
workers = int(os.environ.get("GUNICORN_WORKERS", "2"))

# Number of simultaneous connections per worker. This is the async concurrency
# limit — uvicorn can handle this many in-flight requests on one worker.
worker_connections = 1000

# Bind address. 0.0.0.0 makes the service reachable from outside the container.
bind = "0.0.0.0:8000"

# Gunicorn's own access log. "-" means stdout, which Docker captures.
# We disable this by default because our per-request structlog lines in
# FastAPI already cover request logging. Set to "-" to enable gunicorn's
# native access log in addition.
accesslog = None

# Gunicorn's own error log (worker startup/shutdown events). "-" means stderr.
errorlog = "-"

# Log level for gunicorn's own error log. "info" shows worker startup and
# any gunicorn-level errors without being excessively noisy.
loglevel = "info"
