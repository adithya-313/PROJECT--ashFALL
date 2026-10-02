# File: features/geo_velocity.py
# Purpose: CPU/NumPy reference implementation of Haversine distance and geo-velocity (km/h) between transactions.
import logging

import numpy as np

logger = logging.getLogger(__name__)

EARTH_RADIUS_KM = 6371.0
SECONDS_PER_HOUR = 3600.0


def parse_geo_coordinates(geo_str: str) -> tuple[float, float]:
    """Parse a "lat,lon" string into a (lat, lon) tuple of floats."""
    # The "lat,lon" string format comes from the existing API model (geo_coordinates field).
    if not geo_str or not isinstance(geo_str, str):
        raise ValueError(f"geo_coordinates is missing or not a string: {geo_str!r}")
    parts = geo_str.split(",")
    if len(parts) != 2:
        raise ValueError(f"geo_coordinates must look like 'lat,lon', got: {geo_str!r}")
    try:
        return float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise ValueError(f"geo_coordinates has a non-numeric part: {geo_str!r}") from exc


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return the great-circle distance in kilometres between two lat/lon points."""
    # Step 1: convert degrees to radians, which the trig functions expect.
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    # Step 2: compute the differences in latitude and longitude.
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    # Step 3: apply the haversine formula to get "a", the squared half-chord length between the points.
    a = np.sin(delta_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
    # Step 4: central angle c = 2 * atan2(sqrt(a), sqrt(1 - a)); multiply by Earth's radius for kilometres.
    # a is clipped to [0, 1] so floating-point rounding can never make sqrt() fail.
    a = np.clip(a, 0.0, 1.0)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return float(EARTH_RADIUS_KM * c)


def compute_geo_velocity(current_geo: str, last_geo: str | None, elapsed_seconds: float) -> float | None:
    """Returns geo-velocity in km/h, or None if insufficient data."""
    # No previous location means this is the first transaction, so no velocity exists.
    if last_geo is None:
        return None
    # Dividing by zero (or negative) time is meaningless; flag it and bail out.
    if elapsed_seconds <= 0:
        logger.warning("Cannot compute geo-velocity: elapsed_seconds=%s is not positive", elapsed_seconds)
        return None

    current_lat, current_lon = parse_geo_coordinates(current_geo)
    last_lat, last_lon = parse_geo_coordinates(last_geo)
    distance_km = haversine_distance_km(last_lat, last_lon, current_lat, current_lon)
    return distance_km / (elapsed_seconds / SECONDS_PER_HOUR)
