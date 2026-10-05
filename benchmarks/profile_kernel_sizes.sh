#!/usr/bin/env bash
# File: benchmarks/profile_kernel_sizes.sh
# Purpose: Level 6 - profile the Level 4.5 Haversine GPU kernel at three batch sizes with Nsight Systems (timeline) and,
#          if asked, Nsight Compute (hardware counters). Run it INSIDE WSL2 (the GPU build of the kernel lives there).
#
# Output goes to reports/benchmarks/level6_nsight/:
#   haversine_<rows>.nsys-rep   Nsight Systems trace (open it in the Nsight Systems app)
#   haversine_<rows>.ncu-repz   Nsight Compute report for one launch (only with RUN_NCU=1)
#   haversine_<rows>_ncu.csv    the same Nsight Compute numbers as a table (only with RUN_NCU=1)
# Then run benchmarks/summarize_kernel_profiling.py to turn them into the JSON section.
#
# Which tools: Nsight Systems 2026.1.3 (from NVIDIA's CUDA repo for WSL) is used for the kernel TIME. The older
# 2022.4 build from the Ubuntu package recorded no CUDA events for this CUDA 12.4 setup, so do not use it.
# Nsight Compute is OFF by default (set RUN_NCU=1). It replays the kernel many times, which slows the machine a lot,
# and its durations are longer than the real ones; use it only for occupancy and memory-bandwidth counters.
#
# Why three sizes: 10,000 matches Level 4.5, 100,000 and 1,000,000 show what happens when the GPU gets more work.
# Each size needs only six float32 arrays (about 24 MB at 1,000,000 rows), far inside the 6 GB of GPU memory.
set -eu

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$REPO/reports/benchmarks/level6_nsight"
WORK="${WORK:-/tmp/level6_nsight_work}"  # profiler files are written on the Linux disk first (the Windows mount fails)
NSYS="${NSYS:-/opt/nvidia/nsight-systems/2026.1.3/bin/nsys}"
NCU="${NCU:-$HOME/ncu/target/linux-desktop-glibc_2_11_3-x64/ncu}"
SIZES=(10000 100000 1000000)
LAUNCHES=20  # launches after the first one (the first launch compiles and loads the kernel)

source "$HOME/level45-venv/bin/activate"
mkdir -p "$OUT" "$WORK"
cd "$REPO"

for rows in "${SIZES[@]}"; do
  echo "=== $rows rows: Nsight Systems ==="
  # Only CUDA is traced. CPU sampling and context-switch tracing are off: WSL does not support them well and
  # they add nothing to a GPU kernel timeline.
  "$NSYS" profile --sample=none --cpuctxsw=none --trace=cuda --force-overwrite=true -o "$WORK/haversine_$rows" \
    python benchmarks/profile_kernel_at_size.py --rows "$rows" --launches "$LAUNCHES"
  cp "$WORK/haversine_$rows.nsys-rep" "$OUT/"

  if [ "${RUN_NCU:-0}" = "1" ]; then
    echo "=== $rows rows: Nsight Compute (one launch, after the first) ==="
    "$NCU" --kernel-name regex:haversine --launch-skip 1 --launch-count 1 \
      --section SpeedOfLight --section Occupancy --section MemoryWorkloadAnalysis \
      -f -o "$WORK/haversine_$rows" \
      python benchmarks/profile_kernel_at_size.py --rows "$rows" --launches 2
    cp "$WORK/haversine_$rows.ncu-repz" "$OUT/"
    "$NCU" --import "$WORK/haversine_$rows.ncu-repz" --csv --page details > "$OUT/haversine_${rows}_ncu.csv"
  fi
done
ls -l "$OUT"
