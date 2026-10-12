// K4 - 1D block-tiling (each thread computes TM outputs)
// -----------------------------------------------------------------------------
// Block tile: BM x BN of C, stepping K by BK. Threads per block = BM*BN/TM.
// Each thread computes a column strip of TM consecutive C elements.
//
// Why it helps: in K3 every FMA needs 2 shared-memory loads (As and Bs). Here one
// Bs value is loaded into a register and reused for TM FMAs, so shared-memory
// traffic per FMA drops from 2 loads to (1 + 1/TM) loads -> higher arithmetic
// intensity at the shared-memory level.
//
// Defaults: BM=64, BN=64, BK=8, TM=8 -> 512 threads; each thread loads exactly one
// element of A-tile (64x8) and one of B-tile (8x64) per step.
// Requires M % BM == 0, N % BN == 0, K % BK == 0 (checked by supports()).
// -----------------------------------------------------------------------------
#pragma once
#include "../kernel_common.cuh"

template <int BM, int BN, int BK, int TM>
__global__ void __launch_bounds__((BM * BN) / TM)
mygemm_k4_blocktile1d(int M, int N, int K,
                      const float* __restrict__ A,
                      const float* __restrict__ B,
                      float* __restrict__ C) {
    constexpr int NT = (BM * BN) / TM;            // threads per block
    static_assert(BM * BK == NT, "one A element per thread per step");
    static_assert(BK * BN == NT, "one B element per thread per step");

    const int cRow = blockIdx.y;
    const int cCol = blockIdx.x;
    const int t = threadIdx.x;

    const int threadCol = t % BN;                 // which column of the C tile
    const int threadRow = t / BN;                 // which TM-strip of rows

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // move pointers to the start of this block's tiles
    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    const int innerColA = t % BK, innerRowA = t / BK;
    const int innerColB = t % BN, innerRowB = t / BN;

    float res[TM];
#pragma unroll
    for (int i = 0; i < TM; ++i) res[i] = 0.0f;

    for (int bk = 0; bk < K; bk += BK) {
        As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
        Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
        __syncthreads();

        A += BK;
        B += BK * N;

#pragma unroll
        for (int dot = 0; dot < BK; ++dot) {
            const float b = Bs[dot * BN + threadCol];      // reused TM times
#pragma unroll
            for (int r = 0; r < TM; ++r) {
                res[r] += As[(threadRow * TM + r) * BK + dot] * b;
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < TM; ++r) {
        C[(threadRow * TM + r) * N + threadCol] = res[r];
    }
}

struct K4Cfg {
    static constexpr int BM = 64, BN = 64, BK = 8, TM = 8;
    static bool supports(int M, int N, int K) { return M % BM == 0 && N % BN == 0 && K % BK == 0; }
    static dim3 grid(int M, int N) { return dim3(CEIL_DIV(N, BN), CEIL_DIV(M, BM)); }
    static dim3 block() { return dim3((BM * BN) / TM); }
};
#define K4_KERNEL mygemm_k4_blocktile1d<K4Cfg::BM, K4Cfg::BN, K4Cfg::BK, K4Cfg::TM>
