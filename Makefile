# Makefile: thin shortcuts for the level showcases. It only forwards to `python -m showcase`; there is no logic here.
# On Windows without make installed, run the command directly instead:  python -m showcase --level N
#
#   make show LEVEL=7 [QUICK=1] [ALLOW_NOT_RUN=1]
#   make show-all [QUICK=1] [ALLOW_NOT_RUN=1]
#   make check [ARGS=--no-docker]   (run the CI checks locally; see scripts/check.py)

PYTHON ?= python
LEVEL ?=
QUICK ?=
ALLOW_NOT_RUN ?=

QUICK_FLAG = $(if $(filter 1 true,$(QUICK)),--quick,)
ALLOW_FLAG = $(if $(filter 1 true,$(ALLOW_NOT_RUN)),--allow-not-run,)

.PHONY: show show-all check

show:
	@test -n "$(LEVEL)" || { echo "Usage: make show LEVEL=7 [QUICK=1] [ALLOW_NOT_RUN=1]"; exit 2; }
	$(PYTHON) -m showcase --level $(LEVEL) $(QUICK_FLAG) $(ALLOW_FLAG)

show-all:
	$(PYTHON) -m showcase --all $(QUICK_FLAG) $(ALLOW_FLAG)

# Run the same checks as the CI jobs (lint, types, tests, the running stack). Add ARGS=--no-docker to skip the Docker steps.
check:
	$(PYTHON) scripts/check.py $(ARGS)
