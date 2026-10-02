# File: tests/unit/test_geo_velocity.py
# Purpose: Unit tests for coordinate parsing, Haversine distance and geo-velocity in features/geo_velocity.py.
import pytest

from features.geo_velocity import compute_geo_velocity, haversine_distance_km, parse_geo_coordinates

DELHI = (28.6139, 77.2090)
MUMBAI = (19.0760, 72.8777)
KM_PER_DEGREE_LATITUDE = 111.19492664455873


def test_parse_valid_coordinates():
    assert parse_geo_coordinates("12.9716,77.5946") == (12.9716, 77.5946)


def test_parse_bad_data_raises():
    with pytest.raises(ValueError):
        parse_geo_coordinates("bad_data")


def test_parse_non_numeric_raises():
    with pytest.raises(ValueError):
        parse_geo_coordinates("abc,def")


def test_parse_empty_raises():
    with pytest.raises(ValueError):
        parse_geo_coordinates("")


def test_haversine_delhi_to_mumbai():
    assert haversine_distance_km(*DELHI, *MUMBAI) == pytest.approx(1153, rel=0.01)


def test_haversine_same_point_is_zero():
    assert haversine_distance_km(12.0, 77.0, 12.0, 77.0) == 0.0


def test_haversine_one_degree_of_latitude():
    assert haversine_distance_km(0.0, 0.0, 1.0, 0.0) == pytest.approx(KM_PER_DEGREE_LATITUDE, abs=0.1)


def test_geo_velocity_no_last_location_returns_none():
    assert compute_geo_velocity("12.0,77.0", None, 3600.0) is None


def test_geo_velocity_zero_elapsed_returns_none():
    assert compute_geo_velocity("12.0,77.0", "13.0,77.0", 0.0) is None


def test_geo_velocity_negative_elapsed_returns_none():
    assert compute_geo_velocity("12.0,77.0", "13.0,77.0", -5.0) is None


def test_geo_velocity_100km_in_one_hour():
    # Two points exactly 100 km apart along a meridian, one hour apart.
    lat_offset = 100.0 / KM_PER_DEGREE_LATITUDE
    result = compute_geo_velocity(f"{lat_offset},0.0", "0.0,0.0", 3600.0)
    assert result == pytest.approx(100.0, abs=1.0)
