# CLAUDE.md - context for Claude Code

## Project
Step-by-step optimization of an FP32 SGEMM CUDA kernel (naive → ~80% of cuBLAS) on a
Tesla T4 (`sm_75`), with measurements, Nsight Compute profiling and a roofline.
Portfolio project for NVIDIA PS-II (Compute DevTech / Compute Architecture).
The owner is learning: **explain the reasoning behind every kernel change** (which
bottleneck it fixes and which metric should move). Small, reviewable diffs.

## Environment
- Code is edited locally (usually no GPU) and run on **Google Colab / Kaggle T4**
  via `notebooks/run_colab.ipynb` (git pull → make → run → profile).
- **CPU emulator:** `make test-emu` compiles `src/main.cu` with g++ and a CUDA shim
  (`src/emu/cuda_emu.h`) and runs every kernel on the CPU. Use it to check correctness of
  any new or changed kernel BEFORE going to the GPU. Emulator timings are meaningless.
  If a kernel uses a CUDA feature the shim lacks, extend the shim minimally.
- Build: `make` → `nvcc -O3 -std=c++17 -arch=sm_75 -lineinfo -Xptxas -v ... -lcublas`.
  Register/shared-memory usage per kernel is written to `results_ptxas.txt`.

## Commands
| Task | Command |
|---|---|
| build (GPU) | `make` (or `make ARCH=sm_86` etc.) |
| correctness on CPU | `make test-emu` |
| run all kernels | `./gemm --kernel all` |
| one kernel, chosen sizes | `./gemm --kernel 5 --sizes 1024 4096 --reps 20` |
| non-square shape | `./gemm --kernel all --mnk 4096 1024 512` |
| profile (Nsight) | `bash scripts/profile_all.sh` → `results/ncu_summary.md` |
| plots | `python scripts/plot_perf.py results/results.csv`, `python scripts/roofline.py` |

## Code structure
- `src/main.cu` - harness: init, cuBLAS reference, correctness check, timing, CSV.
  Kernel ids: 0 cuBLAS FP32, 1–6 own FP32 kernels, **7 = the owner's kernel (k7_mine, bank-conflict-free K6)**,
  8 WMMA FP16, 9 cuBLAS FP16 (tensor cores).
- `src/kernels/kN_*.cuh` - one kernel per file + a `KNCfg` struct (`supports`, `grid`, `block`)
  + a `KN_KERNEL` macro. To add a kernel: new header, then add it to `KERNELS[]`,
  `supported()` and `run_kernel()` in `main.cu`.
- Launches go through `LAUNCH(kernel, grid, block, args...)` (works on GPU and emulator).

## Rules
1. Matrices are **row-major** FP32. C = A·B with A: M×K, B: K×N.
2. Correctness metric: **normalised max error** `max|C − C_ref| / max|C_ref|` vs cuBLAS FP32.
   Tolerance 1e-4 for FP32 kernels, 2e-2 for FP16-input kernels. Kernels are only timed if they
   pass. **Never loosen a tolerance to make a kernel pass**; find the bug.
3. cuBLAS is column-major: row-major C = A·B is computed as `cublasSgemm(N, M, K, B, A)`.
4. Timing: cudaEvent per launch, warm-ups, **median** of reps. GFLOPS = 2·M·N·K / t / 1e9.
5. **Finished kernels are never edited** (the progression must stay reproducible).
   Improvements go into a new kernel (e.g. id 7, or 10+).
6. Our kernel names start with **`mygemm_`** so `ncu -k regex:mygemm` excludes cuBLAS
   (cuBLAS kernel names also contain "sgemm").
7. `CUDA_CHECK` on every API call; `cudaGetLastError()` after every launch (done in `run_kernel`).
8. Tiled kernels state their size requirements in `supports()`; the harness skips
   unsupported sizes instead of producing wrong results.
9. After each kernel: update `results/notes.md` (hypothesis → result → evidence) and the
   results table in `README.md`. Tick the boxes in `PATHWAY.md`.

## Current milestone
See the checklists in `PATHWAY.md`.
