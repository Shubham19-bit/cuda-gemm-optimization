// common.cuh - host-side helpers: error checks, launch macro, init, comparison.
#pragma once

#include "kernel_common.cuh"

#ifndef EMU
#include <cublas_v2.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err__ = (call);                                               \
        if (err__ != cudaSuccess) {                                               \
            std::fprintf(stderr, "CUDA error '%s' at %s:%d\n",                    \
                         cudaGetErrorString(err__), __FILE__, __LINE__);          \
            std::exit(1);                                                         \
        }                                                                         \
    } while (0)

#define CUBLAS_CHECK(call)                                                        \
    do {                                                                          \
        cublasStatus_t st__ = (call);                                             \
        if (st__ != CUBLAS_STATUS_SUCCESS) {                                      \
            std::fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)st__,         \
                         __FILE__, __LINE__);                                     \
            std::exit(1);                                                         \
        }                                                                         \
    } while (0)

// LAUNCH(kernel, grid, block, args...) works for both the GPU and the CPU emulator.
#ifdef EMU
#define LAUNCH(kern, grid, block, ...) emu_launch((grid), (block), [&] { kern(__VA_ARGS__); })
#else
#define LAUNCH(kern, grid, block, ...) kern<<<(grid), (block)>>>(__VA_ARGS__)
#endif

// FP32 -> FP16 conversion (for the tensor-core kernels; not timed)
__global__ void convert_f32_to_f16(const float* __restrict__ in, half* __restrict__ out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(in[i]);
}

inline void fill_random(std::vector<float>& v, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : v) x = dist(gen);
}

// Normalised max error: max|c - ref| / max|ref|.
// (Element-wise relative error explodes for entries close to zero, so we normalise
//  by the largest reference magnitude instead.)
inline double normalized_max_error(const std::vector<float>& c, const std::vector<float>& ref) {
    double max_diff = 0.0, max_ref = 0.0;
    for (size_t i = 0; i < c.size(); ++i) {
        if (!std::isfinite(c[i])) return INFINITY;
        max_diff = std::max(max_diff, (double)std::fabs(c[i] - ref[i]));
        max_ref = std::max(max_ref, (double)std::fabs(ref[i]));
    }
    return max_ref > 0 ? max_diff / max_ref : max_diff;
}
