# Kernel notes (fill one block per kernel)

## K1 naive
- Hypothesis:
- Result (GFLOPS @4096, % cuBLAS):
- Evidence (Nsight metric):

## K2 coalesced
- Hypothesis: K1 warps read A with stride K (uncoalesced); mapping threadIdx.x to columns makes B/C accesses contiguous.
- Result:
- Evidence: global load efficiency / DRAM bytes vs K1

## K3 shared-memory tiling
- Hypothesis:
- Result:
- Evidence:

## K4 1D block-tiling
## K5 2D block-tiling
## K6 vectorized
## K7 (yours)
## K8 WMMA tensor cores
