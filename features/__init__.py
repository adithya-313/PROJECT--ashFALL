# File: features/__init__.py
# Purpose: Marks the features directory as a Python package so modules can use absolute imports from the repo root.

# DATETIME CONTRACT
# All datetime objects passed into or returned from this package are naive UTC.
# Never pass local-time datetimes into these functions.
# Use datetime.utcnow() at the call site, not datetime.now().
