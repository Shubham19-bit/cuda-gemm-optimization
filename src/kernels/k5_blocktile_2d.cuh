// K5 - 2D block-tiling (each thread computes a TM x TN micro-tile)
// -----------------------------------------------------------------------------
// Block tile BM x BN (128 x 128), K step BK (8), thread micro-tile TM x TN (8 x 8)
// -> 256 threads per block, 64 outputs per thread.
//
// Inner loop = outer product in registers:
//   regM[TM] <- column 'dot' of the As tile (this thread's rows)
//   regN[TN] <- row 'dot' of the Bs tile    (this thread's columns)
//   res[TM][TN] += regM (x) regN               -> TM*TN FMAs for TM+TN smem loads
// Shared-memory loads per FMA: (TM+TN)/(TM*TN) = 16/64 = 0.25 (K4: ~1.1, K3: 2).
//
// Trade-off to measure: more registers per thread (res = 64 floats) -> lower
// occupancy, but much higher reuse. Check with Nsight: occupancy down, GFLOPS up.
// Requires M % 128 == 0, N % 128 == 0, K % 8 == 0.
// -----------------------------------------------------------------------------
#pragma once
#include "../kernel_common.cuh"

template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN))
mygemm_k5_blocktile2d(int M, int N, int K,
                      const float* __restrict__ A,
                      const float* __restrict__ B,
                      float* __restrict__ C) {
    constexpr int NT = (BM * BN) / (TM * TN);     // 256 threads
    static_assert(NT % BK == 0 && NT % BN == 0, "load mapping assumes this");

    const int cRow = blockIdx.y;
    const int cCol = blockIdx.x;
    const int t = threadIdx.x;

    const int threadRow = t / (BN / TN);          // 0 .. BM/TM-1
    const int threadCol = t % (BN / TN);          // 0 .. BN/TN-1

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    // cooperative-load mapping: each pass loads NT elements
    const int innerRowA = t / BK, innerColA = t % BK;
    constexpr int strideA = NT / BK;              // rows of A-tile per pass
    const int innerRowB = t / BN, innerColB = t % BN;
    constexpr int strideB = NT / BN;              // rows of B-tile per pass

    float res[TM * TN];
    float regM[TM];
    float regN[TN];
#pragma unroll
    for (int i = 0; i < TM * TN; ++i) res[i] = 0.0f;

    for (int bk = 0; bk < K; bk += BK) {
#pragma unroll
        for (int off = 0; off < BM; off += strideA) {
            As[(innerRowA + off) * BK + innerColA] = A[(innerRowA + off) * K + innerColA];
        }
#pragma unroll
        for (int off = 0; off < BK; off += strideB) {
            Bs[(innerRowB + off) * BN + innerColB] = B[(innerRowB + off) * N + innerColB];
        }
        __syncthreads();

        A += BK;
        B += BK * N;

#pragma unroll
        for (int dot = 0; dot < BK; ++dot) {
#pragma unroll
            for (int i = 0; i < TM; ++i) regM[i] = As[(threadRow * TM + i) * BK + dot];
#pragma unroll
            for (int j = 0; j < TN; ++j) regN[j] = Bs[dot * BN + threadCol * TN + j];
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j)
                    res[i * TN + j] += regM[i] * regN[j];
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
            C[(threadRow * TM + i) * N + threadCol * TN + j] = res[i * TN + j];
}

struct K5Cfg {
    static constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
    static bool supports(int M, int N, int K) { return M % BM == 0 && N % BN == 0 && K % BK == 0; }
    static dim3 grid(int M, int N) { return dim3(CEIL_DIV(N, BN), CEIL_DIV(M, BM)); }
    static dim3 block() { return dim3((BM * BN) / (TM * TN)); }
};
#define K5_KERNEL mygemm_k5_blocktile2d<K5Cfg::BM, K5Cfg::BN, K5Cfg::BK, K5Cfg::TM, K5Cfg::TN>
