# Kernel notes (hypothesis -> result -> evidence)

Platform: Colab Tesla T4 (sm_75, 40 SMs), M = N = K = 4096, median of 10 reps, 3 warm-ups.
All kernels pass the check against cuBLAS FP32 (normalised max error <= 3e-6 for FP32 kernels).
GFLOPS and % cuBLAS come from the timing run (`results/results.csv`). Counters (DRAM bytes,
occupancy, registers, bank conflicts) come from Nsight Compute (`profile_all.sh`); ncu locks
the clocks to base, so its own GFLOPS are lower (K5 1936 vs 3339) - use ncu for ratios only.

Caveat on the baseline: cuBLAS FP32 reaches 6061 GFLOPS at 1024 but 4291 at 4096 (T4 is
power-capped at 70 W and clocks down under sustained load). "% cuBLAS" is relative to that
throttled number; against the 8.1 TFLOPS nominal peak, K6 is 43%.

| id | GFLOPS | % cuBLAS | step vs previous |
|---|---|---|---|
| K1 | 61.7 | 1.5% | - |
| K2 | 570.7 | 13.4% | 9.2x |
| K3 | 861.2 | 20.3% | 1.51x |
| K4 | 1803.8 | 42.5% | 2.09x |
| K5 | 3339.4 | 78.6% | 1.85x |
| K6 | 3483.5 | 82.0% | 1.04x |
| K7 | pending T4 run | | |
| K8 (FP16) | 1677.0 | 39.5% | - |
| cuBLAS FP32 | 4291.4 | 100% | - |
| cuBLAS FP16 TC | 32538.9 | 765.9% | - |

## K1 naive
- Hypothesis: one thread per output with `threadIdx.x` along rows means a warp reads A with
  stride K (32 different 128-byte lines per load) -> bandwidth wasted, ~1-5% of cuBLAS.
- Result: 61.7 GFLOPS, 1.5% of cuBLAS.
- Evidence: ncu shows DRAM throughput 0.9% and SM throughput 6%: neither DRAM nor the SM is
  busy, the kernel is stalled on latency from uncoalesced accesses (17.9 GB DRAM traffic).

## K2 coalesced
- Hypothesis: K1 warps read A with stride K (uncoalesced); mapping threadIdx.x to columns
  makes B and C accesses contiguous (one 128-byte transaction per warp).
- Result: 570.7 GFLOPS, 9.2x K1.
- Evidence: DRAM bytes 17.9 -> 13.3 GB; SM throughput 6% -> 86%. Careful: the 86% is the
  load/store pipeline saturated (two global loads per FMA), not FMA utilisation.

## K3 shared-memory tiling
- Hypothesis: 32x32 tiles loaded once into shared memory are reused 32x, cutting global load
  instructions per FMA from 2 to 2/32; expect 1.5-2x K2.
- Result: 861.2 GFLOPS, 1.51x K2.
- Evidence: DRAM bytes are unchanged (13.25 vs 13.27 GB) because L2 already absorbed the
  reuse in K2; the gain comes from far fewer global load instructions. Registers 42, occupancy
  100%, no bank conflicts (0/0). Shared loads per FMA = 2 (one As + one Bs per FMA).

## K4 1D block-tiling
- Hypothesis: each thread computes 8 outputs in a column and keeps one B value in a register,
  so shared loads per FMA fall from 2 to (8+1)/8 ~ 1.1; expect ~2x K3.
- Result: 1803.8 GFLOPS, 2.09x K3.
- Evidence: DRAM bytes 13.25 -> 6.54 GB (block tile 64x64, twice the reuse of K3);
  arithmetic intensity 10.4 -> 21.0 FLOP/B, just at the ridge (25). 64 regs, occupancy 99.6%.

## K5 2D block-tiling
- Hypothesis: an 8x8 register micro-tile per thread (outer product) needs 16 shared loads for
  64 FMAs = 0.25 loads per FMA; accept more registers/lower occupancy for the reuse.
- Result: 3339.4 GFLOPS, 1.85x K4, 78.6% of cuBLAS.
- Evidence: registers 64 -> 127 and occupancy 99.6% -> 49.2%, yet GFLOPS nearly doubles: the
  kernel is limited by shared-memory/instruction throughput, not by latency hiding.
  Intensity 43.9 FLOP/B (right of the ridge). At 128-256 the grid has only 1-4 blocks for
  40 SMs, which is why K5/K6 are slower than K4 at small sizes.
- Not as planned: ncu reports 268,435,456 (2^28) shared-load bank conflicts for K5, i.e. even
  K5 (no transposed tile) has conflicts. See K7.

## K6 vectorized
- Hypothesis: float4 loads/stores cut load instructions 4x and the transposed A tile allows
  wide shared loads: expect +10-25%.
- Result: 3483.5 GFLOPS (82.0% of cuBLAS), only +4.3% over K5.
- Evidence: DRAM bytes 3.13 -> 2.35 GB (-25%) and intensity 43.9 -> 58.6 FLOP/B, but ncu
  GFLOPS are identical to K5 (1933 vs 1936) and SM throughput is unchanged (~65%). DRAM was
  not the limit. Conflicts: store 2^24 (the expected 2-way conflict of the transposed store)
  and load 2^28, 16x larger and the same as K5.

## K7 (mine): K6 without shared-memory bank conflicts
- Hypothesis: K5/K6 are limited by shared-memory wavefronts. Two causes:
  1. Store: `As[k][m]` written with threads `t` and `t+1` (same m, rows 4 apart = 512 words)
     hit the same bank -> 2-way. Fix: pad the As row to BM+4 floats (528 words, 528 mod 32 = 16).
  2. Load: a thread owned 8 consecutive Bs columns, so neighbouring threads read 32 B apart and
     128-bit loads of threads c and c+4 share banks -> 2-way. Fix: each thread owns two 4-wide
     column chunks [4c,4c+4) and [64+4c,64+4c+4), so a warp reads 16 consecutive 16 B pieces.
  Plus explicit float4 shared loads (arrays `__align__(16)`) so the compiler cannot emit scalar LDS.
  Expect: ld/st conflicts -> 0, GFLOPS above K6 (modest, since the FMA/issue limit remains).
- Status: written (`src/kernels/k7_mine.cuh`), registered as id 7, correct on the CPU emulator
  (square and non-square). **T4 measurement pending**: run
  `./gemm --kernel 6 7 --sizes 1024 2048 4096 --reps 20` and
  `SIZE=4096 KERNELS="6 7" bash scripts/profile_all.sh`, then fill in:
  - GFLOPS K6 vs K7 at 4096: ____ vs ____
  - smem ld/st conflicts K6 -> K7: 268435456/16777216 -> ____/____
  - registers / occupancy K7: ____ / ____
- If the conflicts do not reach 0, the first suspect is my reading of which operand conflicts
  (the counters do not say which tile): open `ncu --set full --import-source yes` and look at
  the per-line conflict counts.
- What is left to cuBLAS: warp tiling, double buffering, instruction scheduling.

## K8 WMMA tensor cores
- Hypothesis: a simple WMMA kernel reading fragments straight from global memory is starved
  by DRAM; expect far below cuBLAS FP16 tensor-core GEMM.
- Result: 1677.0 GFLOPS (FP16 in, FP32 acc), 5% of cuBLAS FP16 (32539 GFLOPS) and lower than K6.
- Evidence: ncu DRAM throughput 81.5% and 25.5 GB moved (vs 0.2 GB minimum), SM throughput
  17.6%: bandwidth-bound, sitting on the memory slope of the roofline at 5.4 FLOP/B.
  Staging tiles through shared memory (a K10) is the fix.
