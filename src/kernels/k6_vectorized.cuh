// K6 - Vectorized (float4) loads/stores + transposed A tile
// -----------------------------------------------------------------------------
// Same 128x128x8 block tile and 8x8 thread micro-tile as K5, plus:
//  1. Global loads of A and B use float4 (16-byte LDG.128): 4x fewer load instructions.
//  2. The A tile is stored TRANSPOSED in shared memory (As[k][m]). In the inner loop a
//     thread then reads its TM values of A from consecutive addresses (like Bs), which
//     the compiler can turn into wide shared-memory loads.
//  3. C is written with float4 stores.
//
// Load mapping (256 threads, exactly one float4 per thread per tile):
//   A tile 128x8  = 256 float4 -> innerRowA = t / 2, innerColA = t % 2
//   B tile 8x128  = 256 float4 -> innerRowB = t / 32, innerColB = t % 32
//
// Known remaining issue (exercise): the transposed store into As causes a 2-way
// shared-memory bank conflict. Measure it with Nsight (l1tex bank conflicts) and fix it
// (e.g. padding) in your own K7.
// Requires M % 128 == 0, N % 128 == 0, K % 8 == 0 (float4 alignment needs K, N % 4 == 0).
// -----------------------------------------------------------------------------
#pragma once
#include "../kernel_common.cuh"

template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN))
mygemm_k6_vectorized(int M, int N, int K,
                     const float* __restrict__ A,
                     const float* __restrict__ B,
                     float* __restrict__ C) {
    constexpr int NT = (BM * BN) / (TM * TN);     // 256
    static_assert(BM * BK / 4 == NT, "one float4 of A per thread per tile");
    static_assert(BK * BN / 4 == NT, "one float4 of B per thread per tile");
    static_assert(TN % 4 == 0, "float4 stores of C");

    const int cRow = blockIdx.y;
    const int cCol = blockIdx.x;
    const int t = threadIdx.x;

    const int threadRow = t / (BN / TN);
    const int threadCol = t % (BN / TN);

    __shared__ float As[BK * BM];                 // transposed: As[k * BM + m]
    __shared__ float Bs[BK * BN];

    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    const int innerRowA = t / (BK / 4);
    const int innerColA = t % (BK / 4);           // which float4 within the row
    const int innerRowB = t / (BN / 4);
    const int innerColB = t % (BN / 4);

    float res[TM * TN];
    float regM[TM];
    float regN[TN];
#pragma unroll
    for (int i = 0; i < TM * TN; ++i) res[i] = 0.0f;

    for (int bk = 0; bk < K; bk += BK) {
        // A: one float4 from global, scattered (transposed) into shared memory
        const float4 a4 = reinterpret_cast<const float4*>(&A[innerRowA * K + innerColA * 4])[0];
        As[(innerColA * 4 + 0) * BM + innerRowA] = a4.x;
        As[(innerColA * 4 + 1) * BM + innerRowA] = a4.y;
        As[(innerColA * 4 + 2) * BM + innerRowA] = a4.z;
        As[(innerColA * 4 + 3) * BM + innerRowA] = a4.w;

        // B: one float4 global -> one float4 shared
        reinterpret_cast<float4*>(&Bs[innerRowB * BN + innerColB * 4])[0] =
            reinterpret_cast<const float4*>(&B[innerRowB * N + innerColB * 4])[0];
        __syncthreads();

        A += BK;
        B += BK * N;

#pragma unroll
        for (int dot = 0; dot < BK; ++dot) {
#pragma unroll
            for (int i = 0; i < TM; ++i) regM[i] = As[dot * BM + threadRow * TM + i];
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
    for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; j += 4) {
            reinterpret_cast<float4*>(&C[(threadRow * TM + i) * N + threadCol * TN + j])[0] =
                make_float4(res[i * TN + j + 0], res[i * TN + j + 1],
                            res[i * TN + j + 2], res[i * TN + j + 3]);
        }
    }
}

struct K6Cfg {
    static constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
    static bool supports(int M, int N, int K) { return M % BM == 0 && N % BN == 0 && K % BK == 0; }
    static dim3 grid(int M, int N) { return dim3(CEIL_DIV(N, BN), CEIL_DIV(M, BM)); }
    static dim3 block() { return dim3((BM * BN) / (TM * TN)); }
};
#define K6_KERNEL mygemm_k6_vectorized<K6Cfg::BM, K6Cfg::BN, K6Cfg::BK, K6Cfg::TM, K6Cfg::TN>
