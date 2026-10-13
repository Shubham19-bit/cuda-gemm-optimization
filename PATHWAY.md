# PATHWAY - CUDA SGEMM Optimization (≈ 3 weeks)

**Goal:** an FP32 matrix multiply (SGEMM) optimised step by step from naive to
**70–85% of cuBLAS** on a Tesla T4, with every step measured (GFLOPS, Nsight Compute,
roofline) and explained, plus **one kernel you design yourself** (kernel 7).

**Targets:** NVIDIA Project 1 (Compute DevTech) and Project 2 (Compute Architecture).
**Hardware:** free **Google Colab T4** (or Kaggle). No local GPU needed.
**Time:** ~10 h/week. Each week ends with a **Definition of Done (DoD)**.

> **What's provided vs. what's yours.** The harness, kernels K1–K6 and K8, scripts and
> notebook are ready to run (all kernels were checked for correctness on the CPU emulator,
> including non-square shapes, but **not yet run on a real GPU**, so the first Colab run is
> the real test). Your job: run, understand, measure, explain, then **beat K6 with your
> own K7**. Interviewers will ask about every kernel, so treat the provided ones as
> code you must be able to rewrite from memory.

---

## Reference numbers - NVIDIA Tesla T4 (Turing, `sm_75`)
| Spec | Value |
|---|---|
| SMs | 40 |
| FP32 peak | ~8.1 TFLOPS (at boost clock; sustained is often lower, since the T4 is power-capped at 70 W) |
| FP16 tensor-core peak | ~65 TFLOPS |
| DRAM bandwidth | 320 GB/s (GDDR6) |
| Shared memory / SM | 64 KB |
| Registers / SM | 64K × 32-bit; max 1024 threads/SM on Turing |
| Roofline ridge point | 8100 / 320 ≈ **25 FLOP/byte** |

GEMM FLOPs = `2·M·N·K`. GFLOPS = `2·M·N·K / time / 1e9`.

---

## Project layout
```
cuda-gemm-optimization/
├── CLAUDE.md / PATHWAY.md / README.md
├── Makefile                 make (GPU) · make test-emu (CPU emulator)
├── src/
│   ├── main.cu              harness: cuBLAS reference, check, timing, CSV
│   ├── common.cuh           CUDA_CHECK, LAUNCH macro, init, error metric
│   ├── kernel_common.cuh
│   ├── emu/cuda_emu.h       CPU emulation shim (no GPU needed)
│   └── kernels/             k1_naive … k6_vectorized, k8_wmma_fp16 (+ your k7)
├── scripts/                 plot_perf.py · profile_all.sh · parse_ncu.py · roofline.py
├── notebooks/run_colab.ipynb
└── results/                 notes.md (+ CSVs, plots, Nsight tables after runs)
```
**Workflow:** edit in VS Code with Claude Code → `make test-emu` (CPU check) → `git push`
→ Colab notebook: pull → `make` → run → profile → plots → download/commit results.

---

## Week 1 - First GPU run + understand K1–K3

### 1.1 Setup (day 1)
- [x] Push this folder to a **public GitHub repo** (`git init && git add . && git commit && git push`).
- [x] If the folder has `build.mk` but no `Makefile`, run `cp build.mk Makefile` once.
- [x] Locally: `make test-emu` (needs only g++). All kernels should say `ok`.
- [x] Colab: open `notebooks/run_colab.ipynb` (File → Upload notebook), *Runtime → T4 GPU*,
      set `REPO_URL`, run all cells.
- [ ] If the build or a kernel fails on the GPU, give Claude Code the exact error:
      > *"On Colab T4 this fails with: <error>. Find the root cause, explain it, propose the
      > smallest fix. Run `make test-emu` afterwards."*

### 1.2 Understand the harness (`main.cu`)
- [ ] Explain the **cuBLAS column-major trick** (row-major C = A·B ⇔ column-major Cᵀ = Bᵀ·Aᵀ).
- [ ] Why the error metric is `max|diff| / max|ref|` and not element-wise relative error.
- [ ] Why C is cleared before each correctness run, and why timing uses the **median**.
- [ ] Why the FP32 result is not bit-identical to cuBLAS (floating-point summation order).

### 1.3 Kernels K1–K3 - read, run, explain
| # | Kernel | Idea | Expect (T4, 4096) |
|---|---|---|---|
| K1 | Naive | 1 thread → 1 output, warp walks *down* a column | ~1–5% of cuBLAS |
| K2 | Coalesced | warp walks *along* a row → 128-byte transactions | several × K1 |
| K3 | Shared-memory tiling | 32×32 tiles, each element reused 32× | ~1.5–2× K2 |

For each kernel fill `results/notes.md`: **hypothesis → result → evidence**.
- [ ] Draw (on paper) which addresses one warp touches in K1 vs K2.
- [ ] Explain why K3 has no shared-memory bank conflicts.

**DoD Week 1:** first full `results/results.csv` from the T4; plots generated; you can
explain coalescing and tiling without notes.

---

## Week 2 - Register blocking, vectorisation, profiling

### 2.1 Profile everything
```bash
bash scripts/profile_all.sh          # -> results/ncu_summary.md
```
| What | Nsight metric |
|---|---|
| DRAM bytes / throughput | `dram__bytes.sum`, `dram__throughput…pct_of_peak` |
| Compute throughput | `sm__throughput…pct_of_peak` |
| Achieved occupancy | `sm__warps_active…pct_of_peak_sustained_active` |
| Registers / thread | `launch__registers_per_thread` (also `results_ptxas.txt`) |
| Smem bank conflicts | `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld/st.sum` |

> If profiling fails with `ERR_NVGPUCTRPERM`: use **Kaggle**, or skip; timing + roofline
> (estimated intensity) still work.

For a deeper look at one kernel:
```bash
ncu --set full -k regex:mygemm -c 1 -o results/k6_full ./gemm --kernel 6 --sizes 4096 --reps 1 --warmup 0 --no-check
```
(Download the `.ncu-rep` and open it in the Nsight Compute GUI on your laptop.)

### 2.2 Kernels K4–K6 - read, run, explain
| # | Kernel | Idea | Expect |
|---|---|---|---|
| K4 | 1D block-tiling | 8 outputs/thread, B value reused from a register | ~2× K3 |
| K5 | 2D block-tiling | 8×8 register micro-tile, outer product | ~1.5–2× K4 |
| K6 | Vectorised | `float4` loads/stores + transposed A tile | +10–25% |

- [ ] Compute **shared-memory loads per FMA** for K3, K4 and K5 (2, ~1.1, 0.25) and connect
      that to the speed-ups.
- [ ] Find the case where **occupancy drops but GFLOPS rises** (K5 vs K4) and explain it with
      `regs` and `occupancy_pct`.
- [ ] Confirm the **2-way bank conflict** in K6's transposed store (`smem_st_conflicts`).
- [ ] **Tile-size sweep:** copy K5 into a scratch file, try (BM,BN,BK,TM,TN) such as
      (64,64,8,8,8), (128,64,8,8,4), (128,128,16,8,8). Record GFLOPS + regs + occupancy in a table.

**DoD Week 2:** `ncu_summary.md` for K1–K6 (or the timing-only fallback) and the sweep table.

---

## Week 3 - Your kernel (K7), tensor cores, roofline, write-up

### 3.1 K7 - your own kernel (the most important part)
Create `src/kernels/k7_mine.cuh` starting from K6 and add **at least one** of:
- [x] **Fix the K6 bank conflict** (pad As: `BM + 4`, or another layout) → measure conflicts → 0.
      *Done in `k7_mine.cuh` (also fixes the Bs read conflict); conflicts 2^28/2^24 -> 0/0 on the T4, +16% over K6 at 4096.*
- [ ] **Warp-tiling:** add a warp-level tile (each warp owns e.g. 64×32 of the block tile)
      for better register-level locality.
- [ ] **Double buffering:** two shared-memory buffers; load tile k+1 while computing tile k.
- [ ] **Autotuned config** from your Week-2 sweep.

Then register it as id 7 in `main.cu` (`KERNELS[]`, `supported()`, `run_kernel()`),
`make test-emu`, run on the T4. **Goal: beat K6.** Even a small, well-explained gain counts.

> Claude Code prompt: *"I want to add K7 = K6 + padding to remove the As bank conflict.
> Explain the conflict first, then make the smallest change in a new file k7_mine.cuh and
> register it as kernel 7. Don't touch k6."*

### 3.2 Tensor cores (K8 vs K9)
- [x] Compare K8 (simple WMMA) with K9 (cuBLAS FP16 tensor cores) and K0 (FP32).
- [x] Explain why K8 is far below K9 (no shared-memory staging, tiny warp tiles).
- [ ] Optional: stage A/B tiles through shared memory in a K10 and measure the gain.
- [ ] Link to your systolic-array project: a tensor core is a small matrix-multiply unit.

### 3.3 Roofline + README
```bash
python scripts/roofline.py --results results/results.csv --size 4096
```
- [ ] The story: kernels move **right** (more reuse → higher FLOP/byte) and **up**.
- [ ] Fill the README results table, add the three plots, one paragraph per kernel
      (bottleneck → fix → evidence), and "what I'd do next".

**DoD Week 3:** K7 done and measured; README complete with numbers; repo pinned on GitHub.

---

## Learning resources
- Simon Boehm, *"How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance"*: the
  kernel order here follows the same idea. **Don't copy its code.** Your K7, your T4
  measurements and your explanations are what make this project yours.
- NVIDIA *CUDA C++ Programming Guide*: memory hierarchy, coalescing, shared-memory banks, WMMA.
- NVIDIA *CUDA C++ Best Practices Guide*: occupancy, memory optimisations.
- Nsight Compute docs: *Memory Workload Analysis*, *Warp State Statistics*, *Roofline*.

---

## Resume bullets (fill in your real numbers)
**CUDA SGEMM Optimization** | CUDA C++, cuBLAS, Nsight Compute, Python
- Optimized an FP32 GEMM through **<n> kernel versions** (coalescing, shared-memory tiling,
  2D register blocking, vectorized loads), reaching **<x>% of cuBLAS** (**<y> GFLOPS**) on a
  Tesla T4, a **<z>× speedup** over naive.
- Profiled every kernel with **Nsight Compute** and a **roofline model**, raising arithmetic
  intensity from **<a> to <b> FLOP/byte**; removed shared-memory bank conflicts in my own
  kernel for a **<w>%** gain.
- Implemented a **WMMA tensor-core** kernel (FP16/FP32) and benchmarked it against
  cuBLAS tensor-core GEMM.

## Interview questions to be ready for
1. What is memory coalescing? Show the addresses a warp touches in K1 vs K2.
2. Why does shared-memory tiling help? How much reuse does a 32×32 tile give?
3. Shared-memory loads per FMA for K3/K4/K5 and why it matters.
4. Why did occupancy go down while GFLOPS went up in K5?
5. What is a bank conflict? Where is the one in K6 and how did you fix it?
6. Where are your kernels on the roofline? Memory- or compute-bound?
7. Why isn't your result bit-identical to cuBLAS?
8. What still separates you from cuBLAS? (double buffering, warp tiling, instruction scheduling, SASS-level tuning)
9. How do tensor cores change the picture? How do they relate to a systolic array?
