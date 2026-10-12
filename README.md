# CUDA SGEMM Optimization: from naive to near-cuBLAS

Step-by-step optimisation of an FP32 matrix multiply (`C = A·B`, row-major) on an
**NVIDIA Tesla T4**, with every step verified against cuBLAS, timed, profiled with
**Nsight Compute**, and placed on a **roofline**.

> Status: work in progress. Plan and progress in [`PATHWAY.md`](PATHWAY.md).

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
| 7 | *your own* | warp-tiling / double buffering / bank-conflict fix (exercise) |
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

| Kernel | GFLOPS | % cuBLAS | Speedup vs K1 | Nsight evidence | Lesson |
|---|---|---|---|---|---|
| K1 naive | _todo_ | | 1× | | |
| K2 coalesced | | | | | |
| K3 smem tiling | | | | | |
| K4 1D tiling | | | | | |
| K5 2D tiling | | | | | |
| K6 vectorized | | | | | |
| cuBLAS FP32 | | 100% | | | |

_Plots: `results/perf_vs_size.png`, `results/pct_cublas.png`, `results/roofline.png`._
