# File: benchmarks/profile_kernel_at_size.py
# Purpose: Level 6 - launch the Level 4.5 Haversine GPU kernel (features/gpu_geo_velocity.py) on a batch of any size, so
#          Nsight Systems and Nsight Compute can profile it at several sizes. benchmarks/profile_one_batch.py does the
#          same for the single fixed size of 10,000; this one takes the size as an argument.
#
# What it launches: first ONE untimed launch (this is when the kernel is compiled and loaded), then --launches more
# identical launches. The profilers look at the kernel launches only; this script itself measures nothing.
#
# Run inside WSL2 (the GPU build lives there):
#     python benchmarks/profile_kernel_at_size.py --rows 100000 --launches 20
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import triton

# Allow running as "python benchmarks/profile_kernel_at_size.py" from any directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from features.gpu_geo_velocity import BLOCK_SIZE, haversine_triton_kernel  # noqa: E402

SEED = 7  # same seed and value ranges as benchmarks/profile_one_batch.py


def make_gpu_inputs(rows: int) -> list[torch.Tensor]:
    """Five random float32 arrays (lat1, lon1, lat2, lon2, elapsed seconds), already on the GPU."""
    rng = np.random.default_rng(SEED)
    arrays = [rng.uniform(-90, 90, rows), rng.uniform(-180, 180, rows), rng.uniform(-90, 90, rows),
              rng.uniform(-180, 180, rows), rng.uniform(60.0, 86400.0, rows)]
    return [torch.as_tensor(a, dtype=torch.float32).to("cuda").contiguous() for a in arrays]


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the Haversine GPU kernel on N rows for profiling")
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--launches", type=int, default=20, help="launches after the first (compile) launch")
    args = parser.parse_args()

    tensors = make_gpu_inputs(args.rows)
    output = torch.empty_like(tensors[0])
    grid = (triton.cdiv(args.rows, BLOCK_SIZE),)

    for _ in range(1 + args.launches):
        haversine_triton_kernel[grid](*tensors, output, args.rows, BLOCK_SIZE=BLOCK_SIZE)
    torch.cuda.synchronize()
    print(f"rows={args.rows} launches={1 + args.launches} grid_blocks={grid[0]} "
          f"gpu_memory_bytes={torch.cuda.memory_allocated()} device={torch.cuda.get_device_name(0)}")


if __name__ == "__main__":
    main()
