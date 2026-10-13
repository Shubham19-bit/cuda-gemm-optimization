// cuda_emu.h - Minimal CPU emulation of the CUDA features used in this project.
// -----------------------------------------------------------------------------
// Purpose: check kernel INDEXING and harness logic on a machine with no GPU.
//   make emu && ./gemm_emu --kernel all --sizes 128 256 --reps 1
// It is NOT a performance model - timings from the emulator are meaningless.
//
// How it works:
//   * every CUDA thread of a block runs as a real host thread (pthreads)
//   * __syncthreads() is a pthread barrier across the block's threads
//   * __shared__ becomes `static` (blocks run one after another, so a static array
//     behaves like per-block shared memory)
//   * device memory is host memory; cuBLAS calls are plain CPU loops with the same
//     column-major semantics (so the row-major "swap A and B" trick is also tested)
//   * WMMA fragments hold a full 16x16 tile in every thread of the warp
// -----------------------------------------------------------------------------
#pragma once
#define EMU 1

#include <pthread.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <type_traits>
#include <vector>

// ---------------------------------------------------------------- qualifiers
#define __global__
#define __device__
#define __host__
#define __forceinline__ inline
#define __restrict__
#define __shared__ static
#define __launch_bounds__(...)

// ----------------------------------------------------------------- built-ins
struct dim3 {
    unsigned x, y, z;
    dim3(unsigned x_ = 1, unsigned y_ = 1, unsigned z_ = 1) : x(x_), y(y_), z(z_) {}
};
inline thread_local dim3 threadIdx;
inline dim3 blockIdx, blockDim, gridDim;
inline pthread_barrier_t g_emu_barrier;

inline void __syncthreads() { pthread_barrier_wait(&g_emu_barrier); }

struct alignas(16) float4 { float x, y, z, w; };
inline float4 make_float4(float x, float y, float z, float w) { return float4{x, y, z, w}; }

// FP16 is emulated as float rounded to half precision (_Float16 on x86 g++ >= 12)
typedef float half;
#ifdef __FLT16_MAX__
inline half __float2half(float f) { return (float)(_Float16)f; }
#else
// Fallback for older compilers (e.g. g++ 11): round-to-nearest-even to FP16 by bit
// manipulation, returned as a float. Handles normals, subnormals, overflow, inf/nan.
inline half __float2half(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    const uint32_t sign = u & 0x80000000u;
    const uint32_t a = u & 0x7fffffffu;
    float out;
    if (a >= 0x7f800000u) {                       // inf / nan
        out = f;
    } else if (a >= 0x477ff000u) {                // rounds to >= 65520 -> inf
        const uint32_t r = sign | 0x7f800000u;
        std::memcpy(&out, &r, 4);
    } else if (a < 0x33000001u) {                 // <= 2^-25 rounds to zero
        const uint32_t r = sign;
        std::memcpy(&out, &r, 4);
    } else {
        // Quantum of the FP16 grid at this magnitude: 2^(max(e,-14) - 10)
        int e = (int)(a >> 23) - 127;
        if (e < -14) e = -14;
        const float q = std::ldexp(1.0f, e - 10);
        float v = std::fabs(f) / q;               // exact: power-of-two scaling
        v = std::nearbyint(v);                    // ties-to-even in default rounding mode
        out = std::copysign(v * q, f);
    }
    return out;
}
#endif
inline float __half2float(half h) { return h; }

// --------------------------------------------------------------- launching
struct EmuThreadArg {
    unsigned tx, ty, tz;
    const std::function<void()>* fn;
};

inline void* emu_thread_main(void* p) {
    auto* a = static_cast<EmuThreadArg*>(p);
    threadIdx = dim3(a->tx, a->ty, a->tz);
    (*a->fn)();
    return nullptr;
}

inline void emu_launch(dim3 grid, dim3 block, const std::function<void()>& fn) {
    const unsigned nthreads = block.x * block.y * block.z;
    gridDim = grid;
    blockDim = block;
    std::vector<pthread_t> th(nthreads);
    std::vector<EmuThreadArg> args(nthreads);
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, 256 * 1024);
    for (unsigned bz = 0; bz < grid.z; ++bz)
        for (unsigned by = 0; by < grid.y; ++by)
            for (unsigned bx = 0; bx < grid.x; ++bx) {
                blockIdx = dim3(bx, by, bz);
                pthread_barrier_init(&g_emu_barrier, nullptr, nthreads);
                unsigned i = 0;
                for (unsigned z = 0; z < block.z; ++z)
                    for (unsigned y = 0; y < block.y; ++y)
                        for (unsigned x = 0; x < block.x; ++x, ++i) {
                            args[i] = EmuThreadArg{x, y, z, &fn};
                            if (pthread_create(&th[i], &attr, emu_thread_main, &args[i]) != 0) {
                                std::fprintf(stderr, "emu: pthread_create failed\n");
                                std::exit(1);
                            }
                        }
                for (unsigned k = 0; k < nthreads; ++k) pthread_join(th[k], nullptr);
                pthread_barrier_destroy(&g_emu_barrier);
            }
    pthread_attr_destroy(&attr);
}

// ------------------------------------------------------------ runtime API
typedef int cudaError_t;
constexpr cudaError_t cudaSuccess = 0;
enum cudaMemcpyKind { cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice };

template <typename T>
inline cudaError_t cudaMalloc(T** p, size_t bytes) {
    *p = static_cast<T*>(std::aligned_alloc(256, ((bytes + 255) / 256) * 256));
    return *p ? cudaSuccess : 2;
}
inline cudaError_t cudaFree(void* p) { std::free(p); return cudaSuccess; }
inline cudaError_t cudaMemcpy(void* d, const void* s, size_t n, cudaMemcpyKind) {
    std::memcpy(d, s, n); return cudaSuccess;
}
inline cudaError_t cudaMemset(void* d, int v, size_t n) { std::memset(d, v, n); return cudaSuccess; }
inline cudaError_t cudaDeviceSynchronize() { return cudaSuccess; }
inline cudaError_t cudaGetLastError() { return cudaSuccess; }
inline const char* cudaGetErrorString(cudaError_t) { return "emu error"; }

struct cudaDeviceProp {
    char name[256];
    int multiProcessorCount;
    int major, minor;
};
inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp* p, int) {
    std::snprintf(p->name, sizeof(p->name), "CPU emulator (no GPU)");
    p->multiProcessorCount = 0; p->major = 0; p->minor = 0;
    return cudaSuccess;
}

typedef std::chrono::high_resolution_clock::time_point* cudaEvent_t;
inline cudaError_t cudaEventCreate(cudaEvent_t* e) {
    *e = new std::chrono::high_resolution_clock::time_point(); return cudaSuccess;
}
inline cudaError_t cudaEventDestroy(cudaEvent_t e) { delete e; return cudaSuccess; }
inline cudaError_t cudaEventRecord(cudaEvent_t e, int = 0) {
    *e = std::chrono::high_resolution_clock::now(); return cudaSuccess;
}
inline cudaError_t cudaEventSynchronize(cudaEvent_t) { return cudaSuccess; }
inline cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t a, cudaEvent_t b) {
    *ms = std::chrono::duration<float, std::milli>(*b - *a).count(); return cudaSuccess;
}

// ------------------------------------------------------------------- cuBLAS
typedef int cublasStatus_t;
constexpr cublasStatus_t CUBLAS_STATUS_SUCCESS = 0;
typedef void* cublasHandle_t;
enum cublasOperation_t { CUBLAS_OP_N, CUBLAS_OP_T };
enum cudaDataType_t { CUDA_R_16F, CUDA_R_32F };
enum cublasComputeType_t { CUBLAS_COMPUTE_32F };
enum cublasGemmAlgo_t { CUBLAS_GEMM_DEFAULT, CUBLAS_GEMM_DEFAULT_TENSOR_OP };

inline cublasStatus_t cublasCreate(cublasHandle_t* h) { *h = nullptr; return 0; }
inline cublasStatus_t cublasDestroy(cublasHandle_t) { return 0; }

// column-major C(m x n) = alpha * A(m x k) * B(k x n) + beta * C   (only OP_N supported)
inline void emu_colmajor_gemm(int m, int n, int k, float alpha, const float* A, int lda,
                              const float* B, int ldb, float beta, float* C, int ldc) {
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i) {
            double acc = 0.0;
            for (int p = 0; p < k; ++p) acc += (double)A[i + p * lda] * B[p + j * ldb];
            C[i + j * ldc] = alpha * (float)acc + beta * C[i + j * ldc];
        }
}
inline cublasStatus_t cublasSgemm(cublasHandle_t, cublasOperation_t ta, cublasOperation_t tb,
                                  int m, int n, int k, const float* alpha,
                                  const float* A, int lda, const float* B, int ldb,
                                  const float* beta, float* C, int ldc) {
    if (ta != CUBLAS_OP_N || tb != CUBLAS_OP_N) return 7;
    emu_colmajor_gemm(m, n, k, *alpha, A, lda, B, ldb, *beta, C, ldc);
    return 0;
}
inline cublasStatus_t cublasGemmEx(cublasHandle_t, cublasOperation_t ta, cublasOperation_t tb,
                                   int m, int n, int k, const void* alpha,
                                   const void* A, cudaDataType_t, int lda,
                                   const void* B, cudaDataType_t, int ldb,
                                   const void* beta, void* C, cudaDataType_t, int ldc,
                                   cublasComputeType_t, cublasGemmAlgo_t) {
    if (ta != CUBLAS_OP_N || tb != CUBLAS_OP_N) return 7;
    emu_colmajor_gemm(m, n, k, *(const float*)alpha, (const float*)A, lda,
                      (const float*)B, ldb, *(const float*)beta, (float*)C, ldc);
    return 0;
}

// --------------------------------------------------------------------- WMMA
namespace nvcuda {
namespace wmma {
struct matrix_a {};
struct matrix_b {};
struct accumulator {};
struct row_major {};
struct col_major {};
enum layout_t { mem_row_major, mem_col_major };

template <typename Use, int M, int N, int K, typename T, typename Layout = void>
struct fragment {
    static constexpr int ROWS = std::is_same<Use, matrix_b>::value ? K : M;
    static constexpr int COLS = std::is_same<Use, matrix_a>::value ? K : N;
    T x[ROWS * COLS];
};

template <typename Use, int M, int N, int K, typename T, typename L>
inline void fill_fragment(fragment<Use, M, N, K, T, L>& f, T v) {
    for (auto& e : f.x) e = v;
}
template <typename Use, int M, int N, int K, typename T, typename L>
inline void load_matrix_sync(fragment<Use, M, N, K, T, L>& f, const T* p, unsigned ld) {
    using F = fragment<Use, M, N, K, T, L>;
    static_assert(std::is_same<L, row_major>::value, "emu supports row_major only");
    for (int r = 0; r < F::ROWS; ++r)
        for (int c = 0; c < F::COLS; ++c) f.x[r * F::COLS + c] = p[r * ld + c];
}
template <int M, int N, int K, typename TA, typename TB, typename TC>
inline void mma_sync(fragment<accumulator, M, N, K, TC>& d,
                     const fragment<matrix_a, M, N, K, TA, row_major>& a,
                     const fragment<matrix_b, M, N, K, TB, row_major>& b,
                     const fragment<accumulator, M, N, K, TC>& c) {
    TC tmp[M * N];
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            TC acc = c.x[i * N + j];
            for (int p = 0; p < K; ++p) acc += (TC)a.x[i * K + p] * (TC)b.x[p * N + j];
            tmp[i * N + j] = acc;
        }
    for (int i = 0; i < M * N; ++i) d.x[i] = tmp[i];
}
template <int M, int N, int K, typename T>
inline void store_matrix_sync(T* p, const fragment<accumulator, M, N, K, T>& f,
                              unsigned ld, layout_t layout) {
    if (threadIdx.x % 32 != 0) return;   // one lane writes (all lanes hold the same tile)
    for (int r = 0; r < M; ++r)
        for (int c = 0; c < N; ++c) {
            if (layout == mem_row_major) p[r * ld + c] = f.x[r * N + c];
            else                         p[c * ld + r] = f.x[r * N + c];
        }
}
}  // namespace wmma
}  // namespace nvcuda
