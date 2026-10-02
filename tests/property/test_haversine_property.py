# File: tests/property/test_haversine_property.py
# Purpose: Hypothesis property tests (identity, symmetry) and a known-value check for haversine_distance_km.
import pytest
from hypothesis import given
from hypothesis import strategies as st

from features.geo_velocity import haversine_distance_km

LATITUDES = st.floats(min_value=-90.0, max_value=90.0, allow_nan=False, allow_infinity=False)
LONGITUDES = st.floats(min_value=-180.0, max_value=180.0, allow_nan=False, allow_infinity=False)


@given(lat=LATITUDES, lon=LONGITUDES)
def test_identity_distance_to_self_is_zero(lat, lon):
    assert haversine_distance_km(lat, lon, lat, lon) == 0.0


@given(lat1=LATITUDES, lon1=LONGITUDES, lat2=LATITUDES, lon2=LONGITUDES)
def test_symmetry_a_to_b_equals_b_to_a(lat1, lon1, lat2, lon2):
    forward = haversine_distance_km(lat1, lon1, lat2, lon2)
    backward = haversine_distance_km(lat2, lon2, lat1, lon1)
    assert forward == pytest.approx(backward, abs=1e-9)


def test_known_value_delhi_to_mumbai():
    # Fixed (non-random) check: published Delhi-Mumbai distance is about 1153 km.
    assert haversine_distance_km(28.6139, 77.2090, 19.0760, 72.8777) == pytest.approx(1153, rel=0.01)
