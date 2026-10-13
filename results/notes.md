# Kernel notes (hypothesis -> result -> evidence)

Platform: Colab Tesla T4 (sm_75, 40 SMs), M = N = K = 4096, median of 10 reps, 3 warm-ups.
All kernels pass the check against cuBLAS FP32 (normalised max error <= 3e-6 for FP32 kernels).
GFLOPS and % cuBLAS come from the timing run (`results/results.csv`, run 2 = the full run that
includes K7). Counters (DRAM bytes, occupancy, registers, bank conflicts) come from Nsight
Compute (`profile_all.sh`); ncu locks the clocks to base, so its own GFLOPS are lower
(K6 1982 vs 3352) - use ncu for ratios only.

Run-to-run noise is ~3-4% (first run: K6 3484, cuBLAS 4291; second run: K6 3352, cuBLAS 4186),
so only compare numbers from the same run. K7 vs K6 below is from the same run.

Caveat on the baseline: cuBLAS FP32 reaches 5612 GFLOPS at 1024 but 4186 at 4096 (the T4 is
power-capped at 70 W and clocks down under sustained load). "% cuBLAS" at 4096 is relative to
that throttled number, so it flatters every kernel: K7 is 92.6% at 4096 but 78% at 1024 and
2048. Against the 8.1 TFLOPS nominal peak, K7 is 48%.

| id | GFLOPS | % cuBLAS | step vs previous |
|---|---|---|---|
| K1 | 62.0 | 1.5% | - |
| K2 | 546.6 | 13.0% | 8.8x |
| K3 | 843.0 | 20.1% | 1.54x |
| K4 | 1721.2 | 41.1% | 2.04x |
| K5 | 3160.1 | 75.4% | 1.84x |
| K6 | 3351.7 | 80.0% | 1.06x |
| **K7** | **3880.3** | **92.6%** | **1.16x** |
| K8 (FP16) | 1611.3 | 38.4% | - |
| cuBLAS FP32 | 4185.9 | 100% | - |
| cuBLAS FP16 TC | 32248.6 | 769.3% | - |

K7 vs K6 across sizes (same run): 512: 2197 vs 2179 (+1%), 1024: 4487 vs 3562 (+26%),
2048: 4722 vs 3835 (+23%), 4096: 3880 vs 3352 (+16%). At 128-512 the grid has too few blocks to
fill 40 SMs, so there is nothing to gain. K7 at 2048 (4722) is higher than at 4096 (3880) for
the same kernel, most likely power/thermal throttling during the longer 4096 runs (not measured).

## K1 naive
- Hypothesis: one thread per output with `threadIdx.x` along rows means a warp reads A with
  stride K (32 different 128-byte lines per load) -> bandwidth wasted, ~1-5% of cuBLAS.
- Result: 62.0 GFLOPS, 1.5% of cuBLAS.
- Evidence: ncu shows DRAM throughput 0.9% and SM throughput 6%: neither DRAM nor the SM is
  busy, the kernel is stalled on latency from uncoalesced accesses (18.0 GB DRAM traffic).

## K2 coalesced
- Hypothesis: K1 warps read A with stride K (uncoalesced); mapping threadIdx.x to columns
  makes B and C accesses contiguous (one 128-byte transaction per warp).
- Result: 546.6 GFLOPS, 8.8x K1.
- Evidence: DRAM bytes 18.0 -> 13.3 GB; SM throughput 6% -> 86%. Careful: the 86% is the
  load/store pipeline saturated (two global loads per FMA), not FMA utilisation.

## K3 shared-memory tiling
- Hypothesis: 32x32 tiles loaded once into shared memory are reused 32x, cutting global load
  instructions per FMA from 2 to 2/32; expect 1.5-2x K2.
- Result: 843.0 GFLOPS, 1.54x K2.
- Evidence: DRAM bytes are unchanged (13.20 vs 13.26 GB) because L2 already absorbed the
  reuse in K2; the gain comes from far fewer global load instructions. Registers 42, occupancy
  100%, no bank conflicts (0/0). Shared loads per FMA = 2 (one As + one Bs per FMA).

## K4 1D block-tiling
- Hypothesis: each thread computes 8 outputs in a column and keeps one B value in a register,
  so shared loads per FMA fall from 2 to (8+1)/8 ~ 1.1; expect ~2x K3.
- Result: 1721.2 GFLOPS, 2.04x K3.
- Evidence: DRAM bytes 13.20 -> 6.53 GB (block tile 64x64, twice the reuse of K3);
  arithmetic intensity 10.4 -> 21.0 FLOP/B, just at the ridge (25). 64 regs, occupancy 99.6%.

## K5 2D block-tiling
- Hypothesis: an 8x8 register micro-tile per thread (outer product) needs 16 shared loads for
  64 FMAs = 0.25 loads per FMA; accept more registers/lower occupancy for the reuse.
- Result: 3160.1 GFLOPS, 1.84x K4, 75.4% of cuBLAS.
- Evidence: registers 64 -> 127 and occupancy 99.6% -> 49.2%, yet GFLOPS nearly doubles: the
  kernel is limited by shared-memory/instruction throughput, not by latency hiding.
  Intensity 43.8 FLOP/B (right of the ridge). At 128-256 the grid has only 1-4 blocks for
  40 SMs, which is why K5/K6 are slower than K4 at small sizes.
- Not as planned: ncu reports 268,435,456 (2^28) shared-load bank conflicts for K5, i.e. even
  K5 (no transposed tile) has conflicts. See K7.

## K6 vectorized
- Hypothesis: float4 loads/stores cut load instructions 4x and the transposed A tile allows
  wide shared loads: expect +10-25%.
- Result: 3351.7 GFLOPS (80.0% of cuBLAS), only +6% over K5.
- Evidence: DRAM bytes 3.13 -> 2.35 GB (-25%) and intensity 43.8 -> 58.5 FLOP/B, but ncu
  GFLOPS are identical to K5 (1982 vs 1987) and SM throughput is unchanged (~65%). DRAM was
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
- Result (T4, 4096^3): **3880.3 GFLOPS = 92.6% of cuBLAS, +15.8% over K6** (3351.7), 62.6x K1.
  Larger gain at 1024 (+26%) and 2048 (+23%).
- Evidence (ncu, 4096^3):

  | | K6 | K7 |
  |---|---|---|
  | smem ld conflicts | 268,435,456 | **0** |
  | smem st conflicts | 16,777,216 | **0** |
  | SM throughput | 64.6% | **85.2%** |
  | ncu GFLOPS (base clock) | 1982 | **2607 (+31.5%)** |
  | registers / occupancy | 125 / 49.3% | 111 / 49.1% |
  | DRAM bytes / FLOP per byte | 2.35 GB / 58.5 | 2.68 GB / 51.4 |

  Both conflict counters went to exactly zero, which confirms the diagnosis: the kernel was
  limited by shared-memory wavefronts. Occupancy did not change (49%), so the gain is not
  from latency hiding; SM throughput rose from 65% to 85% because the same FMAs now need
  fewer shared-memory transactions.
- Honest limits of this result:
  - I changed three things at once (padding, interleaved column chunks, explicit float4 smem
    loads), so I cannot say how much each contributes. An ablation (padding only vs. mapping
    only) would answer that. Registers also fell 125 -> 111, which I did not predict.
  - DRAM bytes rose 14% (2.35 -> 2.68 GB) and I have not explained it. Intensity fell
    58.5 -> 51.4 FLOP/B yet the kernel got faster, so DRAM was not the limit.
  - The 92.6% at 4096 is inflated by the throttled cuBLAS baseline; at 1024 and 2048 K7 is 78%
    of cuBLAS.
- What is left to cuBLAS: warp tiling, double buffering, instruction scheduling.

## K8 WMMA tensor cores
- Hypothesis: a simple WMMA kernel reading fragments straight from global memory is starved
  by DRAM; expect far below cuBLAS FP16 tensor-core GEMM.
- Result: 1611.3 GFLOPS (FP16 in, FP32 acc), 5% of cuBLAS FP16 (32249 GFLOPS) and lower than K6.
- Evidence: ncu DRAM throughput 84.3% and 25.8 GB moved (vs 0.2 GB minimum), SM throughput
  17.6%: bandwidth-bound, sitting on the memory slope of the roofline at 5.3 FLOP/B.
  Staging tiles through shared memory (a K10) is the fix.
