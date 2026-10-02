# Level 4.5 — Nsight Compute summary (OpenAI Triton kernel language Haversine kernel)

- Kernel: `haversine_triton_kernel`, one launch, 10,000 elements, grid (10,1,1) x block (128,1,1), GPU: RTX 4050 Laptop (CC 8.9, 20 SMs)
- Tool: Nsight Compute 2026.3.1.0, `--set full`, Linux build run inside WSL2 (the Windows `ncu.exe` cannot launch a Linux process)
- Full report: `level4_5_haversine_nsight.ncu-repz` (168 KB, committed)

## The three requested metrics

The spec named three exact counters. Only the first exists in this report under its exact name; for the other two
the report does not contain those counters (they are absent from the raw metric page), so the closest recorded
measurements are shown instead. These substitutions are stated plainly, not hidden.

| Requested metric | Result | Plain-English meaning |
|---|---|---|
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` (SM throughput) | **6.26 %** | The GPU's compute units were busy about 6% of what they could sustain over the kernel's runtime. Achieved occupancy (warps resident) was 8.37 %, against a theoretical 66.7 %. |
| `l1tex__t_bytes.sum` (memory traffic) — **not in report**; substitutes: `dram__bytes_read.sum`, `l1tex__throughput.avg.pct...` | **261.9 KB read from DRAM, 0 bytes written to DRAM; L1/texture throughput 2.31 % of peak; combined compute+memory throughput 20.49 %** | The kernel reads its five input arrays (about 200 KB of float32 data) and the output stays in cache, so memory traffic is small and far from the bandwidth limit. |
| `sm__sass_thread_inst_executed_op_fadd_pred_on.sum` (FP add instructions) — **not in report**; substitute: Nsight's instruction analysis | **15,680 fused (FMA) + 12,480 non-fused FP32 instructions** (per-warp counts); 67,434 instructions executed in total | The arithmetic work is a few tens of thousands of floating-point instructions: sin, cos, sqrt and atan2 dominate. |

Other facts from the same report: kernel duration **6.624 microseconds**; only **10 blocks** launched for **20 SMs**
(half the GPU idle); theoretical occupancy is limited by registers per thread.

## Memory-bound or compute-bound?

Neither: SM throughput is 6.3 % and combined memory throughput is 20.5 %, so both are far from their peaks. At
10,000 elements the kernel is too small to fill the GPU (10 blocks on 20 SMs, under one active warp per scheduler),
so it is **latency / launch-bound**. If it were pushed harder with a larger batch it would be expected to move toward
being compute-bound, because the sin/cos/sqrt/atan2 work per byte loaded is high (about 28,000 FP32 warp
instructions against about 262 KB read).
