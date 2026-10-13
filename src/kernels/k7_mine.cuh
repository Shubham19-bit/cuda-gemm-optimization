// K7 - K6 without shared-memory bank conflicts
// -----------------------------------------------------------------------------
// Starting point: K6 (128x128x8 block tile, 8x8 thread micro-tile, float4 global I/O).
// Nsight on the T4 (4096^3) showed for K5/K6:
//   smem ld conflicts = 2^28 (K5 and K6), smem st conflicts = 2^24 (K6 only)
// and K6 was only ~4% faster than K5 even though float4 cut DRAM bytes by 25%.
// The inner loop was limited by shared-memory wavefronts, not by DRAM.
//
// Fix 1 - STORE conflict (As is stored transposed, As[k][m]):
//   In one warp, thread t writes row k = innerColA*4 + q, column m = t/2. The two threads
//   with the same m (innerColA = 0 and 1) are 4 rows apart = 4*128 floats = a multiple of
//   32 words -> same bank, different address -> 2-way conflict.
//   Pad each As row to BM + 4 floats: the two threads are now 4*132 = 528 words apart
//   (528 mod 32 = 16), so their banks differ by 16 and never collide with the other
//   threads' banks (m = 0..15). 132 floats = 528 B is still a multiple of 16 B, so
//   float4 reads of As stay aligned.
//
// Fix 2 - LOAD conflict (Bs, and the shape of the micro-tile):
//   K6 thread c owns 8 consecutive columns [8c, 8c+8). A warp reads Bs at a stride of
//   32 B between neighbouring threads, so 128-bit loads from threads c and c+4 hit the
//   same banks -> 2-way conflict.
//   Instead each thread owns TWO column chunks of 4: [4c, 4c+4) and [64+4c, 64+4c+4).
//   A warp's 16 distinct threads now read 16 CONSECUTIVE 16-byte chunks (256 B = the
//   2 wavefronts that 256 B costs at minimum) -> no conflict. The C stores also become
//   fully coalesced per half-tile (consecutive threads write consecutive 16 B).
//   The micro-tile is still 8x8, so FMAs per thread and registers stay the same.
//
// Fix 3 - make the wide shared loads explicit: the arrays are 16-byte aligned and
//   regM / regN are filled with float4 reads, so the compiler cannot fall back to
//   scalar LDS.32 when it cannot prove alignment.
//
// Expected effect (verify with scripts/profile_all.sh, KERNELS="6 7"):
//   smem ld/st conflicts -> 0 / 0, GFLOPS above K6.
// Requires M % 128 == 0, N % 128 == 0, K % 8 == 0.
// -----------------------------------------------------------------------------
#pragma once
#include "../kernel_common.cuh"

template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN))
mygemm_k7_mine(int M, int N, int K,
               const float* __restrict__ A,
               const float* __restrict__ B,
               float* __restrict__ C) {
    constexpr int NT = (BM * BN) / (TM * TN);     // 256
    constexpr int PAD = 4;                        // fix 1
    constexpr int LDA = BM + PAD;                 // padded row length of the transposed A tile
    constexpr int HALF = BN / 2;                  // second column chunk starts HALF columns later
    static_assert(BM * BK / 4 == NT, "one float4 of A per thread per tile");
    static_assert(BK * BN / 4 == NT, "one float4 of B per thread per tile");
    static_assert(TM == 8 && TN == 8, "micro-tile layout below assumes 8x8");
    static_assert(BN / TN == 16, "16 threads across the tile: 16 x 4 floats = HALF");

    const int cRow = blockIdx.y;
    const int cCol = blockIdx.x;
    const int t = threadIdx.x;

    const int threadRow = t / (BN / TN);          // 0..15 -> rows [8*threadRow, +8)
    const int threadCol = t % (BN / TN);          // 0..15 -> cols [4*threadCol, +4) and [HALF + 4*threadCol, +4)

    __shared__ __align__(16) float As[BK * LDA];  // transposed + padded: As[k * LDA + m]
    __shared__ __align__(16) float Bs[BK * BN];

    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    const int innerRowA = t / (BK / 4);
    const int innerColA = t % (BK / 4);
    const int innerRowB = t / (BN / 4);
    const int innerColB = t % (BN / 4);

    float res[TM * TN];
    float regM[TM];
    float regN[TN];
#pragma unroll
    for (int i = 0; i < TM * TN; ++i) res[i] = 0.0f;

    for (int bk = 0; bk < K; bk += BK) {
        const float4 a4 = reinterpret_cast<const float4*>(&A[innerRowA * K + innerColA * 4])[0];
        As[(innerColA * 4 + 0) * LDA + innerRowA] = a4.x;
        As[(innerColA * 4 + 1) * LDA + innerRowA] = a4.y;
        As[(innerColA * 4 + 2) * LDA + innerRowA] = a4.z;
        As[(innerColA * 4 + 3) * LDA + innerRowA] = a4.w;

        reinterpret_cast<float4*>(&Bs[innerRowB * BN + innerColB * 4])[0] =
            reinterpret_cast<const float4*>(&B[innerRowB * N + innerColB * 4])[0];
        __syncthreads();

        A += BK;
        B += BK * N;

#pragma unroll
        for (int dot = 0; dot < BK; ++dot) {
            // 8 consecutive A values (2 x LDS.128); the two thread-rows in a warp read
            // two different addresses, everything else is a broadcast.
            const float4 m0 = reinterpret_cast<const float4*>(&As[dot * LDA + threadRow * TM])[0];
            const float4 m1 = reinterpret_cast<const float4*>(&As[dot * LDA + threadRow * TM + 4])[0];
            regM[0] = m0.x; regM[1] = m0.y; regM[2] = m0.z; regM[3] = m0.w;
            regM[4] = m1.x; regM[5] = m1.y; regM[6] = m1.z; regM[7] = m1.w;
            // fix 2: two 4-wide chunks, a warp reads 16 consecutive 16-byte pieces each
            const float4 n0 = reinterpret_cast<const float4*>(&Bs[dot * BN + threadCol * 4])[0];
            const float4 n1 = reinterpret_cast<const float4*>(&Bs[dot * BN + HALF + threadCol * 4])[0];
            regN[0] = n0.x; regN[1] = n0.y; regN[2] = n0.z; regN[3] = n0.w;
            regN[4] = n1.x; regN[5] = n1.y; regN[6] = n1.z; regN[7] = n1.w;
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
        float* crow = &C[(threadRow * TM + i) * N];
        reinterpret_cast<float4*>(&crow[threadCol * 4])[0] =
            make_float4(res[i * TN + 0], res[i * TN + 1], res[i * TN + 2], res[i * TN + 3]);
        reinterpret_cast<float4*>(&crow[HALF + threadCol * 4])[0] =
            make_float4(res[i * TN + 4], res[i * TN + 5], res[i * TN + 6], res[i * TN + 7]);
    }
}

struct K7Cfg {
    static constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
    static bool supports(int M, int N, int K) { return M % BM == 0 && N % BN == 0 && K % BK == 0; }
    static dim3 grid(int M, int N) { return dim3(CEIL_DIV(N, BN), CEIL_DIV(M, BM)); }
    static dim3 block() { return dim3((BM * BN) / (TM * TN)); }
};
#define K7_KERNEL mygemm_k7_mine<K7Cfg::BM, K7Cfg::BN, K7Cfg::BK, K7Cfg::TM, K7Cfg::TN>
