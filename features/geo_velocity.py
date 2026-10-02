# File: features/geo_velocity.py
# Purpose: CPU/NumPy reference implementation of Haversine distance and geo-velocity (km/h) between transactions.
#
# The fraud idea ("impossible travel"): if a card is used in Delhi and 30 minutes later in Mumbai, about 1,150 km
# away, nobody could have travelled that fast, so one of the two transactions is probably not the real owner. We
# measure this as geo-velocity: distance between the two locations divided by the time between them.
import logging
import os

import numpy as np

# The batch size above which we dispatch to the GPU kernel.
# Below this threshold, the CPU NumPy path is faster due to GPU memory transfer overhead.
# This value is set from the benchmark results in Level 4.5.
# It is also written to .env as GEO_KERNEL_CROSSOVER_BATCH_SIZE.
GEO_KERNEL_CROSSOVER_BATCH_SIZE = int(os.getenv("GEO_KERNEL_CROSSOVER_BATCH_SIZE", "500"))

logger = logging.getLogger(__name__)

# Mean radius of the Earth in kilometres. The Haversine formula works on a perfect sphere, and 6371.0 km is the
# standard average radius used to turn an angle on that sphere into a distance on the ground.
EARTH_RADIUS_KM = 6371.0
SECONDS_PER_HOUR = 3600.0


def parse_geo_coordinates(geo_str: str) -> tuple[float, float]:
    """
    Parse a "lat,lon" text string into a (latitude, longitude) pair of floats.

    Why this function exists: the API receives location as one string field, but every distance calculation needs
    two numbers. Parsing in one place, with clear errors, means bad data fails loudly here instead of producing a
    confusing wrong distance later.

    Args:
        geo_str: text such as "12.9716,77.5946", latitude first and then longitude, both in decimal degrees.

    Returns:
        A tuple (latitude, longitude) of floats.

    Edge cases:
        Raises ValueError (with the offending text in the message) if the value is empty or not a string, does not
        have exactly one comma, or either part is not a number (for example "bad_data" or "abc,def").
    """
    # The "lat,lon" string format comes from the existing API model (geo_coordinates field). The API model is
    # frozen, so we adapt to it here rather than asking for two separate fields.
    # Reject empty or non-string input first, so the later split cannot fail with an unclear error.
    if not geo_str or not isinstance(geo_str, str):
        raise ValueError(f"geo_coordinates is missing or not a string: {geo_str!r}")

    # A valid value splits into exactly two parts around the comma: latitude and longitude.
    parts = geo_str.split(",")
    if len(parts) != 2:
        raise ValueError(f"geo_coordinates must look like 'lat,lon', got: {geo_str!r}")

    # Convert both parts to numbers. float() raises ValueError on text like "abc"; we re-raise it with a message
    # that names the full bad input, which makes data problems quick to track down.
    try:
        return float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise ValueError(f"geo_coordinates has a non-numeric part: {geo_str!r}") from exc


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Return the great-circle distance in kilometres between two latitude/longitude points.

    Why this function exists: geo-velocity needs the distance between two transaction locations. We use the
    Haversine formula rather than ordinary (Euclidean) distance because the Earth is a sphere, not a flat map.
    Treating latitude and longitude as flat x/y coordinates is badly inaccurate: one degree of longitude is about
    111 km at the equator but shrinks to nothing at the poles, so "straight-line" distance on raw degrees would
    give the wrong answer, especially far from the equator or over long distances. Haversine measures the shortest
    path along the surface of the sphere (the great circle), which is how aircraft and ships really travel.

    Args:
        lat1: latitude of the first point in decimal degrees (-90 to 90).
        lon1: longitude of the first point in decimal degrees (-180 to 180).
        lat2: latitude of the second point in decimal degrees.
        lon2: longitude of the second point in decimal degrees.

    Returns:
        The distance in kilometres as a float, never negative.

    Edge cases:
        - Identical points return exactly 0.0.
        - The result is symmetric: A to B equals B to A.
        - Points on opposite sides of the Earth give about 20,015 km (half the circumference).
    """
    # Step 1: convert degrees to radians. The sin/cos functions work in radians (a full circle is 2*pi), not in
    # degrees (a full circle is 360), so skipping this step would give nonsense.
    phi1, phi2 = np.radians(lat1), np.radians(lat2)

    # Step 2: work out how far apart the points are in latitude (north-south) and longitude (east-west), also in
    # radians. These differences are the angles the two points are separated by along each direction.
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    # Step 3: apply the haversine formula. "a" is the squared half-chord length between the two points on the
    # unit sphere. Roughly: the first term measures the north-south separation, and the second measures the
    # east-west separation, shrunk by cos(latitude) because lines of longitude converge toward the poles. "a" is
    # 0 for identical points and 1 for opposite points.
    a = np.sin(delta_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2

    # Floating-point rounding can push "a" a hair outside [0, 1], which would make sqrt(1 - a) fail on a negative
    # number. Clamp it back into the valid range; this never changes a correct result.
    a = np.clip(a, 0.0, 1.0)

    # Step 4: turn "a" into the central angle "c", the angle in radians between the two points as seen from the
    # centre of the Earth. atan2 is used rather than asin because it stays accurate even for points very close
    # together or nearly opposite.
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    # Step 5: distance along a circle = angle (radians) x radius. Multiplying by the Earth's average radius
    # (6371.0 km) converts the angle into kilometres on the ground. float() turns the NumPy number into a plain
    # Python float so callers do not get surprises from NumPy types.
    return float(EARTH_RADIUS_KM * c)


def compute_geo_velocity(current_geo: str, last_geo: str | None, elapsed_seconds: float) -> float | None:
    """
    Return the apparent travel speed between the last and current transaction, in km/h, or None if unknown.

    Returns geo-velocity in km/h, or None if insufficient data.

    Why this function exists: this is the "impossible travel" fraud signal. Real people cannot be in Delhi and then
    Mumbai half an hour later, so a very high speed suggests the card or account is being used by two different
    people in two places, or through a proxy or stolen credentials.

    Args:
        current_geo: location of the current transaction as a "lat,lon" string.
        last_geo: location of the previous transaction as a "lat,lon" string, or None if there was no earlier
            transaction (it comes from Redis acct:last_loc:{account_id}).
        elapsed_seconds: seconds between the previous and current transaction.

    Returns:
        Speed in kilometres per hour as a float, or None when no speed can be computed. The unit is km/h by
        convention, because it matches how people think about travel (a car is about 100, an aircraft about 900).
        It is only a convention, so every caller and every model feature must treat the value as km/h and never
        mix it up with m/s.

    Edge cases:
        - last_geo is None: returns None. None is not the same as 0.0. A speed of 0.0 would claim "the user did
          not move", which is real information. None means "we have no data", and the caller must handle that
          separately instead of the model learning a false "stationary" signal.
        - elapsed_seconds is 0 or negative: returns None and logs a warning. Dividing by zero is impossible and a
          negative time means clocks or ordering are wrong, so any speed we computed would be meaningless.
        - Raises ValueError (from parse_geo_coordinates) if either location string is malformed.
    """
    # No earlier location means this is the account's first transaction, so there is nothing to measure against.
    if last_geo is None:
        return None

    # We divide by elapsed time below, so the time must be positive. Log a warning so data problems (duplicate
    # timestamps, clock skew) are visible in the logs, then give up on a speed for this transaction.
    if elapsed_seconds <= 0:
        logger.warning("Cannot compute geo-velocity: elapsed_seconds=%s is not positive", elapsed_seconds)
        return None

    # Convert both "lat,lon" strings into numbers. This raises ValueError with a clear message if either is bad.
    current_lat, current_lon = parse_geo_coordinates(current_geo)
    last_lat, last_lon = parse_geo_coordinates(last_geo)

    # How far apart are the two places, in kilometres?
    distance_km = haversine_distance_km(last_lat, last_lon, current_lat, current_lon)

    # Speed = distance / time. The time is converted from seconds to hours first so the answer is in km per hour.
    return distance_km / (elapsed_seconds / SECONDS_PER_HOUR)


def compute_geo_velocity_batch(
    current_geos: list[str], last_geos: list[str | None], elapsed_seconds_list: list[float]
) -> list[float | None]:
    """
    Compute geo-velocity (km/h) for many transactions at once, choosing the CPU or GPU path by batch size.

    Returns None at positions with no previous location or with elapsed time of zero or less, exactly like
    compute_geo_velocity does for a single transaction.
    """
    results: list[float | None] = [None] * len(current_geos)

    # Keep only the rows that can have a speed. Rows with no previous location or non-positive elapsed time stay
    # None in the output and never go through any compute path. We remember each kept row's original index so the
    # results can be put back in the right order.
    valid_indices = [
        i
        for i, (last_geo, elapsed) in enumerate(zip(last_geos, elapsed_seconds_list))
        if last_geo is not None and elapsed > 0
    ]
    if not valid_indices:
        return results

    # Dispatch pattern: the GPU path is only faster above the crossover batch size. Below it, the GPU memory
    # transfer overhead outweighs the parallelism gain.
    if len(valid_indices) >= GEO_KERNEL_CROSSOVER_BATCH_SIZE:
        # Lazy import: keeps this module importable when the OpenAI Triton kernel language package or a CUDA GPU
        # is not available (the CPU-only case).
        from features.gpu_geo_velocity import compute_geo_velocity_gpu

        # Parse every valid "lat,lon" string into numbers, then send the whole batch to the GPU at once.
        current = [parse_geo_coordinates(current_geos[i]) for i in valid_indices]
        last = [parse_geo_coordinates(last_geos[i]) for i in valid_indices]  # type: ignore[arg-type]
        gpu_values = compute_geo_velocity_gpu(
            np.array([p[0] for p in last]),
            np.array([p[1] for p in last]),
            np.array([p[0] for p in current]),
            np.array([p[1] for p in current]),
            np.array([elapsed_seconds_list[i] for i in valid_indices]),
        )
        for i, value in zip(valid_indices, gpu_values):
            results[i] = float(value)
    else:
        # Small batch: loop over the existing single-transaction CPU function.
        for i in valid_indices:
            results[i] = compute_geo_velocity(current_geos[i], last_geos[i], elapsed_seconds_list[i])

    return results
