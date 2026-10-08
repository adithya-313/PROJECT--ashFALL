# File: tests/unit/test_geo_velocity.py
# Purpose: Unit tests for coordinate parsing, Haversine distance and geo-velocity in features/geo_velocity.py.
import struct

import numpy as np
import pytest

import features.geo_velocity as geo_velocity
from features.geo_velocity import (
    compute_geo_velocity,
    compute_geo_velocity_batch,
    haversine_distance_km,
    parse_geo_coordinates,
)

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


# --- Vectorized CPU batch path (Level 6) ------------------------------------------------------------------------
# The vectorized code must give EXACTLY the numbers the old row-by-row loop gave. The old loop is kept below as
# the source of truth: it is the loop that used to live inside compute_geo_velocity_batch, copied word for word.
def old_loop_batch(current_geos, last_geos, elapsed_seconds_list):
    results = [None] * len(current_geos)
    valid_indices = [
        i
        for i, (last_geo, elapsed) in enumerate(zip(last_geos, elapsed_seconds_list))
        if last_geo is not None and elapsed > 0
    ]
    for i in valid_indices:
        results[i] = compute_geo_velocity(current_geos[i], last_geos[i], elapsed_seconds_list[i])
    return results


def same_bits(left, right):
    """True when two result lists match exactly: None where None, and identical binary floats elsewhere."""
    def bits(value):
        return None if value is None else struct.pack("d", value)
    return [bits(v) for v in left] == [bits(v) for v in right]


@pytest.fixture(autouse=True)
def force_cpu_path(monkeypatch):
    """Pretend the GPU crossover is huge, so every batch in these tests takes the CPU path (no GPU needed)."""
    monkeypatch.setattr(geo_velocity, "GEO_KERNEL_CROSSOVER_BATCH_SIZE", 10**9)


def test_vectorized_batch_equals_old_loop_on_edge_cases():
    current = ["19.0760,72.8777", "19.0760,72.8777", "28.6139,77.2090", "10.0,20.0", "10.0,20.0", "0.0,0.0",
               "-33.8688,151.2093", "10.0,20.0", "12.9716,77.5946", "89.9,179.9"]
    last = ["28.6139,77.2090", None, "19.0760,72.8777", "10.0,20.0", "10.0,20.0", "0.0,180.0",
            "51.5074,-0.1278", "10.0,20.0", "12.9716,77.5946", "-89.9,-179.9"]
    elapsed = [1800.0, 1800.0, 0.0, 3600.0, 1e-9, 3600.0, 36000.0, -5.0, 60.0, float("nan")]
    # In order: normal pair, None last_geo, zero elapsed, identical coords, near-zero elapsed, antipodal points,
    # long haul, negative elapsed, identical coords in a different city, NaN elapsed.
    new = compute_geo_velocity_batch(current, last, elapsed)
    assert same_bits(new, old_loop_batch(current, last, elapsed))
    # Spot checks that the sample really covers the cases: Nones where expected, 0.0 for identical points.
    assert new[1] is None and new[2] is None and new[7] is None and new[9] is None
    assert new[3] == 0.0 and new[8] == 0.0 and new[0] > 1000


def test_vectorized_batch_equals_old_loop_on_many_random_rows():
    rng = np.random.default_rng(7)
    rows = 5_000
    lats, lons = rng.uniform(-90, 90, (2, rows)), rng.uniform(-180, 180, (2, rows))
    # tolist() turns numpy numbers into plain Python floats. Without it, numpy 2.x would write the text
    # "np.float64(22.5)" instead of "22.5", and the coordinate parser could not read it.
    current = [f"{a!r},{b!r}" for a, b in zip(lats[0].tolist(), lons[0].tolist())]
    last = [f"{a!r},{b!r}" for a, b in zip(lats[1].tolist(), lons[1].tolist())]
    last[10] = None  # sprinkle some rows that must stay None
    elapsed = rng.uniform(1e-3, 1e6, rows).tolist()
    elapsed[20] = 0.0
    assert same_bits(compute_geo_velocity_batch(current, last, elapsed), old_loop_batch(current, last, elapsed))


def test_vectorized_batch_with_no_valid_rows_returns_all_none():
    assert compute_geo_velocity_batch(["1.0,1.0"], [None], [10.0]) == [None]
    assert compute_geo_velocity_batch(["1.0,1.0"], ["2.0,2.0"], [0.0]) == [None]


def test_vectorized_batch_raises_the_same_error_as_the_old_loop_on_bad_text():
    current, last, elapsed = ["1.0,1.0", "not a location"], ["2.0,2.0", "3.0,3.0"], [10.0, 10.0]
    with pytest.raises(ValueError) as old_error:
        old_loop_batch(current, last, elapsed)
    with pytest.raises(ValueError) as new_error:
        compute_geo_velocity_batch(current, last, elapsed)
    assert str(new_error.value) == str(old_error.value)


def test_vectorized_batch_raises_zero_division_like_the_old_loop_for_a_denormal_elapsed_time():
    # An elapsed time so small that elapsed / 3600 rounds to 0.0 hours: the old loop raised ZeroDivisionError.
    current, last, elapsed = ["1.0,1.0"], ["2.0,2.0"], [5e-324]
    with pytest.raises(ZeroDivisionError):
        old_loop_batch(current, last, elapsed)
    with pytest.raises(ZeroDivisionError):
        compute_geo_velocity_batch(current, last, elapsed)


def test_array_haversine_matches_single_pair_haversine_bit_for_bit():
    rng = np.random.default_rng(3)
    lat1, lon1 = rng.uniform(-90, 90, 2_000), rng.uniform(-180, 180, 2_000)
    lat2, lon2 = rng.uniform(-90, 90, 2_000), rng.uniform(-180, 180, 2_000)
    from_array = geo_velocity.haversine_distance_km_array(lat1, lon1, lat2, lon2).tolist()
    one_by_one = [haversine_distance_km(*pair) for pair in zip(lat1, lon1, lat2, lon2)]
    assert same_bits(from_array, one_by_one)
