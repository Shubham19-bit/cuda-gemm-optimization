// K8 - Tensor cores via WMMA (FP16 inputs, FP32 accumulate)
// -----------------------------------------------------------------------------
// The WMMA API exposes Tensor Cores at warp level: a whole warp cooperatively
// loads 16x16 fragments of A and B and performs D = A*B + C on a 16x16x16 tile in
// a few instructions.
//
// This is a deliberately SIMPLE first tensor-core kernel:
//   * 128 threads = 4 warps per block, block tile 32x32 (2x2 warps, 16x16 each)
//   * fragments are loaded straight from global memory (no shared-memory staging)
// It is far from cuBLAS-FP16 speed. Improving it (shared-memory staging, bigger warp
// tiles, double buffering) is a great stretch exercise - compare against kernel 9
// (cuBLAS FP16 tensor-core GEMM).
// Requires M % 32 == 0, N % 32 == 0, K % 16 == 0.  Needs sm_70+ (T4 = sm_75).
// -----------------------------------------------------------------------------
#pragma once
#include "../kernel_common.cuh"

__global__ void __launch_bounds__(128)
mygemm_k8_wmma(int M, int N, int K,
               const half* __restrict__ A,
               const half* __restrict__ B,
               float* __restrict__ C) {
    using namespace nvcuda;
    constexpr int WM = 16, WN = 16, WK = 16;

    const int warpId = threadIdx.x / 32;          // 0..3
    const int tileRow = blockIdx.y * 32 + (warpId / 2) * WM;
    const int tileCol = blockIdx.x * 32 + (warpId % 2) * WN;

    wmma::fragment<wmma::matrix_a, WM, WN, WK, half, wmma::row_major> aFrag;
    wmma::fragment<wmma::matrix_b, WM, WN, WK, half, wmma::row_major> bFrag;
    wmma::fragment<wmma::accumulator, WM, WN, WK, float> cFrag;
    wmma::fill_fragment(cFrag, 0.0f);

    for (int k = 0; k < K; k += WK) {
        wmma::load_matrix_sync(aFrag, A + tileRow * K + k, K);      // 16x16 of A, ld = K
        wmma::load_matrix_sync(bFrag, B + k * N + tileCol, N);      // 16x16 of B, ld = N
        wmma::mma_sync(cFrag, aFrag, bFrag, cFrag);
    }
    wmma::store_matrix_sync(C + tileRow * N + tileCol, cFrag, N, wmma::mem_row_major);
}

struct K8Cfg {
    static bool supports(int M, int N, int K) { return M % 32 == 0 && N % 32 == 0 && K % 16 == 0; }
    static dim3 grid(int M, int N) { return dim3(CEIL_DIV(N, 32), CEIL_DIV(M, 32)); }
    static dim3 block() { return dim3(128); }
};
#define K8_KERNEL mygemm_k8_wmma
