# Makefile: thin shortcuts for the level showcases. It only forwards to `python -m showcase`; there is no logic here.
# On Windows without make installed, run the command directly instead:  python -m showcase --level N
#
#   make show LEVEL=7 [QUICK=1] [ALLOW_NOT_RUN=1] [PART=a]   (PART runs one part of a split level, e.g. 12a)
#   make show-all [QUICK=1] [ALLOW_NOT_RUN=1]
#   make check [ARGS=--no-docker]   (run the CI checks locally; see scripts/check.py)
#   make test-gpu                   (run the tests marked gpu; needs a CUDA GPU, torch and Triton)
#   make test-dvc                   (run the tests marked needs_dvc; needs `dvc pull` first; local only)

# Use python if it exists, otherwise python3 (Ubuntu and WSL often have only python3). Override: make PYTHON=...
PYTHON ?= $(shell command -v python >/dev/null 2>&1 && echo python || echo python3)
LEVEL ?=
QUICK ?=
ALLOW_NOT_RUN ?=
PART ?=

QUICK_FLAG = $(if $(filter 1 true,$(QUICK)),--quick,)
ALLOW_FLAG = $(if $(filter 1 true,$(ALLOW_NOT_RUN)),--allow-not-run,)
PART_FLAG = $(if $(PART),--part $(PART),)

.PHONY: show show-all check test-gpu test-dvc bench-latency-container

show:
	@test -n "$(LEVEL)" || { echo "Usage: make show LEVEL=7 [QUICK=1] [ALLOW_NOT_RUN=1] [PART=a]"; exit 2; }
	$(PYTHON) -m showcase --level $(LEVEL) $(QUICK_FLAG) $(ALLOW_FLAG) $(PART_FLAG)

show-all:
	$(PYTHON) -m showcase --all $(QUICK_FLAG) $(ALLOW_FLAG)

# Run the same checks as the CI jobs (lint, types, tests, the running stack). Add ARGS=--no-docker to skip the Docker steps.
check:
	$(PYTHON) scripts/check.py $(ARGS)

# The decision latency benchmark by the pre-registered method, run INSIDE the Linux container where the service runs
# (needs the compose stack). REGISTRY=fake|http, DELAY=off|default|registry-slow. Example: make bench-latency-container REGISTRY=http DELAY=default
REGISTRY ?= fake
DELAY ?= off
bench-latency-container:
	$(PYTHON) -m benchmarks.decision_latency --where container --registry $(REGISTRY) --registry-delay $(DELAY)

# CI leaves the gpu tests out (-m "not gpu"). Run them here on a machine that has the GPU.
test-gpu:
	$(PYTHON) -m pytest tests/unit -m gpu -q -rs

# Tests that need the model v1 files. CI cannot pull them, so run this after `dvc pull`.
test-dvc:
	$(PYTHON) -m pytest tests/unit -m needs_dvc -q -rs
