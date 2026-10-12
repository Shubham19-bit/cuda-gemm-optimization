// K2 - Coalesced global memory access
// -----------------------------------------------------------------------------
// Same math as K1, but threadIdx.x now indexes the COLUMN of C:
//   row = blockIdx.y * 32 + threadIdx.y
//   col = blockIdx.x * 32 + threadIdx.x     <- consecutive threads = consecutive columns
//
// In a warp: all 32 threads share the same row, so A[row*K + k] is one address
// (broadcast), and B[k*N + col] / C[row*N + col] are 32 consecutive floats = one
// 128-byte transaction (coalesced). Expect a large speed-up over K1.
// -----------------------------------------------------------------------------
#pragma once
#include "../kernel_common.cuh"

__global__ void mygemm_k2_coalesced(int M, int N, int K,
                                    const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            acc += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = acc;
    }
}

struct K2Cfg {
    static constexpr int T = 32;
    static bool supports(int, int, int) { return true; }
    static dim3 grid(int M, int N) { return dim3(CEIL_DIV(N, T), CEIL_DIV(M, T)); }
    static dim3 block() { return dim3(T, T); }
};
#define K2_KERNEL mygemm_k2_coalesced
