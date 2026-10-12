// K3 - Shared-memory tiling
// -----------------------------------------------------------------------------
// Each 32x32 thread block computes a 32x32 tile of C. It walks along K in steps of
// 32: the block cooperatively loads a 32x32 tile of A and of B into shared memory
// (one element per thread, coalesced), synchronizes, then every thread does 32 FMAs
// out of shared memory.
//
// Data reuse: each element loaded from DRAM is used 32 times from shared memory,
// so global traffic drops ~32x vs K2 (which relies on L1/L2 caches for reuse).
//
// Bank conflicts: As[ty][k] is the same address for a whole warp (broadcast) and
// Bs[k][tx] is 32 consecutive floats -> conflict-free.
// Handles any M, N, K (out-of-range elements are loaded as 0).
// -----------------------------------------------------------------------------
#pragma once
#include "../kernel_common.cuh"

template <int T>
__global__ void mygemm_k3_smem(int M, int N, int K,
                               const float* __restrict__ A,
                               const float* __restrict__ B,
                               float* __restrict__ C) {
    __shared__ float As[T][T];
    __shared__ float Bs[T][T];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * T + ty;
    const int col = blockIdx.x * T + tx;

    float acc = 0.0f;
    for (int t0 = 0; t0 < K; t0 += T) {
        As[ty][tx] = (row < M && t0 + tx < K) ? A[row * K + t0 + tx] : 0.0f;
        Bs[ty][tx] = (t0 + ty < K && col < N) ? B[(t0 + ty) * N + col] : 0.0f;
        __syncthreads();

#pragma unroll
        for (int k = 0; k < T; ++k) {
            acc += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();   // don't overwrite tiles while others still read them
    }
    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

struct K3Cfg {
    static constexpr int T = 32;
    static bool supports(int, int, int) { return true; }
    static dim3 grid(int M, int N) { return dim3(CEIL_DIV(N, T), CEIL_DIV(M, T)); }
    static dim3 block() { return dim3(T, T); }
};
#define K3_KERNEL mygemm_k3_smem<K3Cfg::T>
