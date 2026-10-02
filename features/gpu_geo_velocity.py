# File: features/gpu_geo_velocity.py
# Purpose: Haversine geo-velocity implemented as an OpenAI Triton kernel language GPU kernel.
#          This is the GPU path; the CPU reference path lives in features/geo_velocity.py.
#          "OpenAI Triton kernel language" is always spelled out in full — never abbreviated
#          to "Triton" alone, to avoid confusion with NVIDIA Triton Inference Server.
import numpy as np
import torch
import triton
import triton.language as tl

# The atan2 function lives in the libdevice extension module of the OpenAI Triton kernel language (version 3.2 has
# no tl.atan2). libdevice maps to NVIDIA's own GPU math library, so it is accurate and runs on the GPU.
from triton.language.extra import libdevice

# Mean radius of the Earth in kilometres; identical to EARTH_RADIUS_KM in features/geo_velocity.py so the CPU and
# GPU paths agree.
EARTH_RADIUS_KM = 6371.0

# BLOCK_SIZE is the number of elements each kernel program instance handles at once. Each GPU program instance
# processes this many elements together; larger values use more GPU registers but reduce launch overhead.
BLOCK_SIZE = 1024

# Note: the kernel below writes 6371.0, 3600.0 and pi/180 as literal numbers, because the OpenAI Triton kernel
# language only lets compiled kernels read plain Python globals if they are wrapped as tl.constexpr.


@triton.jit
def haversine_triton_kernel(
    lat1_ptr,
    lon1_ptr,
    lat2_ptr,
    lon2_ptr,
    elapsed_seconds_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """GPU kernel: compute geo-velocity in km/h for BLOCK_SIZE (lat1, lon1, lat2, lon2, elapsed) rows at a time."""
    # Step 1: find out which block of the data this program instance is responsible for. The GPU launches many
    # program instances in parallel, and each one gets its own pid (0, 1, 2, ...).
    pid = tl.program_id(axis=0)

    # Step 2: work out the positions (indices) in the arrays that this instance will handle. If pid = 2 and
    # BLOCK_SIZE = 1024, this covers indices 2048 to 3071.
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Step 3: build a mask that is True only for valid positions. The last block usually extends past the end of
    # the data, and the mask stops it reading or writing memory that does not belong to us.
    mask = offsets < n_elements

    # Step 4: load the five input values from GPU memory. Masked-out positions get 0.0 as a harmless placeholder.
    lat1 = tl.load(lat1_ptr + offsets, mask=mask, other=0.0)
    lon1 = tl.load(lon1_ptr + offsets, mask=mask, other=0.0)
    lat2 = tl.load(lat2_ptr + offsets, mask=mask, other=0.0)
    lon2 = tl.load(lon2_ptr + offsets, mask=mask, other=0.0)
    elapsed_seconds = tl.load(elapsed_seconds_ptr + offsets, mask=mask, other=0.0)

    # Step 5: convert degrees to radians, because the sin/cos functions expect radians.
    phi1 = lat1 * 0.017453292519943295
    phi2 = lat2 * 0.017453292519943295
    delta_phi = (lat2 - lat1) * 0.017453292519943295
    delta_lambda = (lon2 - lon1) * 0.017453292519943295

    # Step 6: apply the haversine formula exactly as the CPU version does. "a" is the squared half-chord length
    # between the two points: a north-south term plus an east-west term scaled by cos(latitude).
    sin_half_phi = tl.sin(delta_phi / 2.0)
    sin_half_lambda = tl.sin(delta_lambda / 2.0)
    a = sin_half_phi * sin_half_phi + tl.cos(phi1) * tl.cos(phi2) * sin_half_lambda * sin_half_lambda

    # Rounding in float32 can push "a" slightly outside [0, 1], which would make sqrt(1 - a) invalid. Clamp it
    # back into range, as the CPU version does.
    a = tl.minimum(tl.maximum(a, 0.0), 1.0)

    # Step 7: turn "a" into a distance in kilometres: central angle c = 2 * atan2(sqrt(a), sqrt(1 - a)), then
    # multiply by the Earth's radius.
    distance_km = 2.0 * 6371.0 * libdevice.atan2(tl.sqrt(a), tl.sqrt(1.0 - a))

    # Step 8: speed = distance / time, with the time converted from seconds to hours so the result is km/h.
    # Elapsed time of zero or less would divide by zero, so we substitute 1.0 to keep the arithmetic safe here;
    # the result for those rows is replaced in step 9.
    safe_elapsed = tl.where(elapsed_seconds > 0.0, elapsed_seconds, 1.0)
    velocity_kmh = distance_km / (safe_elapsed / 3600.0)

    # Step 9: where elapsed time is zero or negative no real speed exists, so output 0.0 for those rows. (The
    # batch dispatcher in features/geo_velocity.py filters such rows out before they reach the GPU.)
    velocity_kmh = tl.where(elapsed_seconds <= 0.0, 0.0, velocity_kmh)

    # Step 10: write the results back to GPU memory, again only for valid positions.
    tl.store(output_ptr + offsets, velocity_kmh, mask=mask)


def compute_geo_velocity_gpu(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
    elapsed_seconds: np.ndarray,
) -> np.ndarray:
    """Compute geo-velocity in km/h for a whole batch on the GPU and return it as a NumPy array."""
    # Move each input to GPU memory as a float32 tensor. float32 is much faster than float64 on consumer GPUs.
    lat1_t, lon1_t, lat2_t, lon2_t, elapsed_t = (
        torch.as_tensor(arr, dtype=torch.float32).to("cuda").contiguous()
        for arr in (lat1, lon1, lat2, lon2, elapsed_seconds)
    )

    # Allocate the output on the GPU with the same shape.
    output_t = torch.empty_like(lat1_t)
    n = lat1_t.numel()

    # Number of program instances to launch. triton.cdiv is ceiling division, so the last block covers any
    # remaining elements (for example 10,000 elements with BLOCK_SIZE 1024 needs 10 blocks, not 9).
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)  # noqa: E731

    # Launch the kernel. The tensors' data pointers are passed along with the element count.
    haversine_triton_kernel[grid](
        lat1_t, lon1_t, lat2_t, lon2_t, elapsed_t, output_t, n, BLOCK_SIZE=BLOCK_SIZE
    )

    # Copy the result back to CPU memory as a NumPy array (this also waits for the GPU to finish).
    return output_t.cpu().numpy()


# This module requires the OpenAI Triton kernel language package and a CUDA GPU.
# It is imported conditionally by features/geo_velocity.py based on batch size.
