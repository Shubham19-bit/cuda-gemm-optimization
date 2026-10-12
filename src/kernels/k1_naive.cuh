// K1 - Naive SGEMM
// -----------------------------------------------------------------------------
// One thread computes one element of C = A * B (row-major, A: MxK, B: KxN).
//
// Thread mapping (deliberately "wrong"):
//   row = blockIdx.x * 32 + threadIdx.x     <- consecutive threads walk DOWN a column
//   col = blockIdx.y * 32 + threadIdx.y
//
// Within a warp (32 consecutive threadIdx.x) every thread reads a DIFFERENT row of A:
// addresses A[row*K + k] are K floats apart -> 32 separate memory transactions per
// load (uncoalesced). C writes are strided the same way. K2 fixes exactly this.
// -----------------------------------------------------------------------------
#pragma once
#include "../kernel_common.cuh"

__global__ void mygemm_k1_naive(int M, int N, int K,
                                const float* __restrict__ A,
                                const float* __restrict__ B,
                                float* __restrict__ C) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            acc += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = acc;
    }
}

struct K1Cfg {
    static constexpr int T = 32;
    static bool supports(int, int, int) { return true; }
    static dim3 grid(int M, int N) { return dim3(CEIL_DIV(M, T), CEIL_DIV(N, T)); }
    static dim3 block() { return dim3(T, T); }
};
#define K1_KERNEL mygemm_k1_naive
