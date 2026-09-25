#!/usr/bin/env bash
# File: scripts/verify_gpu.sh
# Purpose: Verifies Docker GPU passthrough and NVIDIA container runtime functionality.
# Connects: Required prerequisite check that must pass before running GPU-accelerated
#           training/inference workloads in Level 4.5.

set -euo pipefail

echo "Verifying Docker NVIDIA GPU passthrough..."
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
