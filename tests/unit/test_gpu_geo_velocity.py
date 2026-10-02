# File: tests/unit/test_gpu_geo_velocity.py
# Purpose: Checks the OpenAI Triton kernel language GPU geo-velocity kernel against the CPU NumPy reference.
import numpy as np
import pytest

from features.geo_velocity import haversine_distance_km

# This test needs a CUDA GPU plus the torch and OpenAI Triton kernel language packages (WSL2 venv); skip elsewhere.
torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU not available")

N_PAIRS = 10_000


def test_gpu_matches_cpu_reference():
    from features.gpu_geo_velocity import compute_geo_velocity_gpu

    # Reproducible random inputs: coordinates anywhere on Earth and elapsed times from 1 minute to 24 hours.
    np.random.seed(42)
    lat1 = np.random.uniform(-90.0, 90.0, N_PAIRS)
    lon1 = np.random.uniform(-180.0, 180.0, N_PAIRS)
    lat2 = np.random.uniform(-90.0, 90.0, N_PAIRS)
    lon2 = np.random.uniform(-180.0, 180.0, N_PAIRS)
    elapsed = np.random.uniform(60.0, 86400.0, N_PAIRS)

    # CPU reference: Haversine distance divided by elapsed hours, one pair at a time.
    cpu_result = np.array(
        [
            haversine_distance_km(lat1[i], lon1[i], lat2[i], lon2[i]) / (elapsed[i] / 3600.0)
            for i in range(N_PAIRS)
        ]
    )

    # GPU path: the whole batch in one kernel launch.
    gpu_result = compute_geo_velocity_gpu(lat1, lon1, lat2, lon2, elapsed)

    # We use relative tolerance (rtol) not absolute (atol), because velocity values can be
    # very large in magnitude and float32's rounding error scales with the value.
    # rtol=1e-5 gives 3x headroom over the observed worst-case relative error of 3.5e-6,
    # which confirms the GPU kernel matches the CPU reference in float32 precision.
    # assert_allclose reports the mismatching indices with both values if any element fails.
    np.testing.assert_allclose(gpu_result, cpu_result, rtol=1e-5, atol=0)
