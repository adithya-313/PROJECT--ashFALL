# File: benchmarks/profile_one_batch.py
# Purpose: Runs the OpenAI Triton kernel language Haversine kernel on one 10,000-element batch for Nsight Compute.
import sys
from pathlib import Path

import numpy as np
import torch
import triton

# Allow running as "python benchmarks/profile_one_batch.py" from any directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from features.gpu_geo_velocity import BLOCK_SIZE, haversine_triton_kernel  # noqa: E402

BATCH_SIZE = 10_000


def main() -> None:
    """Launch the kernel exactly once so the Nsight Compute profile contains a single clean kernel."""
    rng = np.random.default_rng(7)
    arrays = [
        rng.uniform(-90, 90, BATCH_SIZE),
        rng.uniform(-180, 180, BATCH_SIZE),
        rng.uniform(-90, 90, BATCH_SIZE),
        rng.uniform(-180, 180, BATCH_SIZE),
        rng.uniform(60.0, 86400.0, BATCH_SIZE),
    ]
    tensors = [torch.as_tensor(a, dtype=torch.float32).to("cuda").contiguous() for a in arrays]
    output = torch.empty_like(tensors[0])
    grid = (triton.cdiv(BATCH_SIZE, BLOCK_SIZE),)

    # The only kernel launch. Compilation happens on the CPU before the launch, so it is not part of the profile.
    haversine_triton_kernel[grid](*tensors, output, BATCH_SIZE, BLOCK_SIZE=BLOCK_SIZE)
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
