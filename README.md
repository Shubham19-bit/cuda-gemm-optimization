# CUDA SGEMM Optimization: from naive to near-cuBLAS

Step-by-step optimisation of an FP32 matrix multiply (`C = A·B`, row-major) on an
**NVIDIA Tesla T4**, with every step verified against cuBLAS, timed, profiled with
**Nsight Compute**, and placed on a **roofline**.

> Status: all kernels (K1–K9) measured on a T4; K7 reaches 92.6% of cuBLAS at 4096³. Plan and progress in [`PATHWAY.md`](PATHWAY.md).

## Kernels

| id | Kernel | Key idea |
|---|---|---|
| 0 | cuBLAS FP32 | reference + correctness oracle |
| 1 | Naive | 1 thread → 1 output; uncoalesced A/C accesses |
| 2 | Coalesced | `threadIdx.x` → column, so a warp touches contiguous memory |
| 3 | Shared-memory tiling | 32×32 tiles in shared memory, 32× reuse |
| 4 | 1D block-tiling | each thread computes 8 outputs, B value reused from a register |
| 5 | 2D block-tiling | 8×8 register micro-tile per thread (outer product) |
| 6 | Vectorized | `float4` global loads/stores + transposed A tile |
| 7 | **mine** | K6 + padded A tile + conflict-free B reads (shared-memory bank-conflict fix) |
| 8 | WMMA tensor cores | FP16 in, FP32 accumulate, 16×16×16 fragments |
| 9 | cuBLAS FP16 (tensor cores) | reference for kernel 8 |

## Quick start (Colab)
Open `notebooks/run_colab.ipynb` in Colab, select a T4 GPU, and run all cells.

Or on any machine with an NVIDIA GPU + CUDA toolkit:
```bash
make                      # ARCH=sm_75 by default (T4); e.g. make ARCH=sm_86
./gemm --kernel all       # sizes 128 ... 4096, results/results.csv
python scripts/plot_perf.py results/results.csv
bash scripts/profile_all.sh        # Nsight Compute metrics -> results/ncu_summary.md
python scripts/roofline.py
```
No GPU? `make test-emu` runs every kernel on the CPU emulator to check correctness.

## Results (Tesla T4, 4096³)

Median of 10 runs, checked against cuBLAS FP32 (normalised max error ≤ 3e-6). Nsight columns
come from `scripts/profile_all.sh` (ncu locks clocks to base, so use them for ratios only).

| Kernel | GFLOPS | % cuBLAS | Speedup vs K1 | Nsight evidence | Lesson |
|---|---|---|---|---|---|
| K1 naive | 62.0 | 1.5% | 1× | DRAM 0.9%, SM 6% | stalled on uncoalesced accesses |
| K2 coalesced | 546.6 | 13.0% | 8.8× | DRAM 18.0 → 13.3 GB | coalescing alone gives ~9× |
| K3 smem tiling | 843.0 | 20.1% | 13.6× | 42 regs, 100% occupancy, 0 conflicts | fewer global load *instructions* (DRAM bytes unchanged, L2 already helped) |
| K4 1D tiling | 1721.2 | 41.1% | 27.8× | DRAM 6.5 GB, 21 FLOP/B | register reuse halves smem loads per FMA |
| K5 2D tiling | 3160.1 | 75.4% | 51.0× | 127 regs, occupancy 49%, 44 FLOP/B | lower occupancy, higher reuse: faster |
| K6 vectorized | 3351.7 | 80.0% | 54.0× | DRAM 2.35 GB, but ld conflicts 2²⁸, st 2²⁴ | float4 helped DRAM, not the real limit |
| **K7 mine** | **3880.3** | **92.6%** | **62.6×** | **conflicts 0 / 0, SM 65% → 85%** | **removing smem bank conflicts: +16% over K6** |
| cuBLAS FP32 | 4185.9 | 100% | 67.5× | | throttled by the 70 W cap (5612 at 1024³) |
| K8 WMMA FP16 | 1611.3 | 38.4% | 26.0× | DRAM 84%, SM 17.6% | tensor cores starved without smem staging |
| cuBLAS FP16 TC | 32248.6 | 769% | 520× | | the tensor-core ceiling |

K7 is 78% of cuBLAS at 1024³ and 2048³ (where cuBLAS is not yet power-throttled); the 92.6% at
4096³ is partly the throttled baseline. Run-to-run noise is about 3–4%.

_Plots: `results/perf_vs_size.png`, `results/pct_cublas.png`, `results/roofline.png`.
Per-kernel hypothesis → result → evidence: [`results/notes.md`](results/notes.md)._

## What each step fixed
- **K1 → K2 (9×):** a warp walked down a column of A (32 cache lines per load); mapping
  `threadIdx.x` to columns makes a warp's loads contiguous.
- **K2 → K3 (1.5×):** 32×32 tiles in shared memory reuse each loaded element 32 times.
- **K3 → K4 (2.1×):** each thread keeps a B value in a register for 8 outputs; DRAM bytes halve.
- **K4 → K5 (1.85×):** an 8×8 register micro-tile needs 0.25 shared loads per FMA instead of
  ~1.1. Registers rise to 127 and occupancy falls to 49%, yet the kernel is much faster.
- **K5 → K6 (+4%):** `float4` cuts DRAM traffic 25%, but ncu shows the same SM throughput:
  the loop is limited by shared-memory wavefronts, not DRAM.
- **K6 → K7 (+16% at 4096³, +26% at 1024³):** padding the transposed A tile (store conflict)
  and giving each thread two interleaved 4-wide column chunks (load conflict) so warps read
  contiguous 16-byte pieces of `Bs`. Nsight: shared load/store conflicts 2²⁸ / 2²⁴ → 0 / 0,
  SM throughput 65% → 85%, occupancy unchanged at 49%.
- **K8:** WMMA straight from global memory is DRAM-bound (84% of peak bandwidth) and ends up
  at 5% of cuBLAS FP16; staging tiles in shared memory would be the next step.

## What I would do next
Warp-level tiling, double buffering (load tile k+1 while computing tile k), shared-memory
staging for the WMMA kernel, and reading the SASS to see what separates K7 from cuBLAS.
