# File: features/__init__.py
# Purpose: Marks the features directory as a Python package so modules can use absolute imports from the repo root.

# WHAT THIS PACKAGE IS
# The "features" package turns a raw transaction into numbers a fraud model can learn from. Each module covers one
# family of signals:
#   temporal.py     - WHEN did this happen? (hour, weekday, account age, gap since the last transaction)
#   velocity.py     - HOW OFTEN is this account transacting? (counts in 1h / 6h / 24h windows, kept in Redis)
#   geo_velocity.py - HOW FAST would the user have had to travel between two transactions? (km/h)
#   entropy.py      - HOW VARIED is this account's behaviour? (merchant diversity, device diversity)
#
# Importing this package is deliberately free of side effects: nothing here connects to Redis or reads the clock.
# Callers pass in the Redis client and the current time, which keeps every function easy to unit test.
#
# Use absolute imports from the repo root, for example:
#   from features.geo_velocity import haversine_distance_km

# DATETIME CONTRACT
# All datetime objects passed into or returned from this package are naive UTC.
# Never pass local-time datetimes into these functions.
# Use datetime.utcnow() at the call site, not datetime.now().
