// kernel_common.cuh - shared includes/macros for every kernel header.
//
// Normal build (nvcc):   real CUDA headers.
// Emulation build (-DEMU, g++): a CPU shim that runs each CUDA thread as a host
//                                thread, so kernel indexing can be tested without a GPU.
#pragma once

#ifdef EMU
#include "emu/cuda_emu.h"
#else
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#endif

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))
