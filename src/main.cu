// main.cu - SGEMM benchmark harness
// -----------------------------------------------------------------------------
//   ./gemm                                   all kernels, sizes 128..4096
//   ./gemm --kernel 5 --sizes 1024 4096      one kernel, chosen sizes
//   ./gemm --kernel all --reps 20 --csv results/results.csv
//   ./gemm --kernel 6 --sizes 4096 --reps 1 --warmup 0 --no-check   (for Nsight)
//
// For every size: A (MxK), B (KxN) random in [-1,1], row-major FP32, M = N = K = size.
// Every kernel is checked against cuBLAS FP32 BEFORE it is timed.
// Timing: cudaEvent around each launch, `warmup` untimed runs, median of `reps`.
// -----------------------------------------------------------------------------
#include "common.cuh"
#include "kernels/k1_naive.cuh"
#include "kernels/k2_coalesced.cuh"
#include "kernels/k3_smem_tiling.cuh"
#include "kernels/k4_blocktile_1d.cuh"
#include "kernels/k5_blocktile_2d.cuh"
#include "kernels/k6_vectorized.cuh"
#include "kernels/k7_mine.cuh"
#include "kernels/k8_wmma_fp16.cuh"

#include <filesystem>
#include <string>

struct KernelInfo {
    int id;
    const char* name;
    bool fp16;        // uses half-precision inputs
    double tol;       // normalised max-error tolerance vs cuBLAS FP32
};

// id 7 = K7, the owner's kernel: K6 with the shared-memory bank conflicts removed
static const KernelInfo KERNELS[] = {
    {0, "cublas_fp32",     false, 1e-6},
    {1, "k1_naive",        false, 1e-4},
    {2, "k2_coalesced",    false, 1e-4},
    {3, "k3_smem_tiling",  false, 1e-4},
    {4, "k4_blocktile_1d", false, 1e-4},
    {5, "k5_blocktile_2d", false, 1e-4},
    {6, "k6_vectorized",   false, 1e-4},
    {7, "k7_mine",         false, 1e-4},
    {8, "k8_wmma_fp16",    true,  2e-2},
    {9, "cublas_fp16_tc",  true,  2e-2},
};

static const KernelInfo* find_kernel(int id) {
    for (const auto& k : KERNELS) if (k.id == id) return &k;
    return nullptr;
}

static bool supported(int id, int M, int N, int K) {
    switch (id) {
        case 1: return K1Cfg::supports(M, N, K);
        case 2: return K2Cfg::supports(M, N, K);
        case 3: return K3Cfg::supports(M, N, K);
        case 4: return K4Cfg::supports(M, N, K);
        case 5: return K5Cfg::supports(M, N, K);
        case 6: return K6Cfg::supports(M, N, K);
        case 7: return K7Cfg::supports(M, N, K);
        case 8: return K8Cfg::supports(M, N, K);
        default: return true;   // cuBLAS handles everything
    }
}

struct DeviceBuffers {
    float *A = nullptr, *B = nullptr, *C = nullptr;
    half *A16 = nullptr, *B16 = nullptr;
};

// Launch one kernel once (no synchronisation here).
static void run_kernel(int id, int M, int N, int K, const DeviceBuffers& d, cublasHandle_t h) {
    const float alpha = 1.0f, beta = 0.0f;
    switch (id) {
        case 0:
            // cuBLAS is column-major. Row-major C = A*B is column-major C^T = B^T * A^T,
            // so we pass B first, then A, and swap M and N.
            CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                                     d.B, N, d.A, K, &beta, d.C, N));
            break;
        case 1: LAUNCH(K1_KERNEL, K1Cfg::grid(M, N), K1Cfg::block(), M, N, K, d.A, d.B, d.C); break;
        case 2: LAUNCH(K2_KERNEL, K2Cfg::grid(M, N), K2Cfg::block(), M, N, K, d.A, d.B, d.C); break;
        case 3: LAUNCH(K3_KERNEL, K3Cfg::grid(M, N), K3Cfg::block(), M, N, K, d.A, d.B, d.C); break;
        case 4: LAUNCH(K4_KERNEL, K4Cfg::grid(M, N), K4Cfg::block(), M, N, K, d.A, d.B, d.C); break;
        case 5: LAUNCH(K5_KERNEL, K5Cfg::grid(M, N), K5Cfg::block(), M, N, K, d.A, d.B, d.C); break;
        case 6: LAUNCH(K6_KERNEL, K6Cfg::grid(M, N), K6Cfg::block(), M, N, K, d.A, d.B, d.C); break;
        case 7: LAUNCH(K7_KERNEL, K7Cfg::grid(M, N), K7Cfg::block(), M, N, K, d.A, d.B, d.C); break;
        case 8: LAUNCH(K8_KERNEL, K8Cfg::grid(M, N), K8Cfg::block(), M, N, K, d.A16, d.B16, d.C); break;
        case 9:
            // Same column-major trick, FP16 inputs, FP32 output and accumulation (tensor cores)
            CUBLAS_CHECK(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                                      d.B16, CUDA_R_16F, N, d.A16, CUDA_R_16F, K, &beta,
                                      d.C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
                                      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            break;
        default:
            std::fprintf(stderr, "kernel %d not implemented\n", id);
            std::exit(1);
    }
    CUDA_CHECK(cudaGetLastError());
}

// Median time (ms) of `reps` launches after `warmup` untimed launches.
static float time_kernel(int id, int M, int N, int K, const DeviceBuffers& d, cublasHandle_t h,
                         int warmup, int reps) {
    for (int i = 0; i < warmup; ++i) run_kernel(id, M, N, K, d, h);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    std::vector<float> times;
    for (int i = 0; i < reps; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        run_kernel(id, M, N, K, d, h);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times.push_back(ms);
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

static void usage() {
    std::printf(
        "usage: gemm [--kernel ID|all] [--sizes S1 S2 ...] [--mnk M N K] [--reps R] [--warmup W]\n"
        "            [--csv PATH] [--no-check]\n"
        "kernels:\n");
    for (const auto& k : KERNELS) std::printf("  %d  %s\n", k.id, k.name);
}

int main(int argc, char** argv) {
    std::vector<int> kernel_ids;
    std::vector<int> sizes = {128, 256, 512, 1024, 2048, 4096};
    struct Shape { int M, N, K; };
    std::vector<Shape> shapes;          // filled from --sizes (square) and/or --mnk
    int reps = 10, warmup = 3;
    bool check = true;
    std::string csv_path = "results/results.csv";

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--kernel" && i + 1 < argc) {
            std::string v = argv[++i];
            if (v == "all") { for (const auto& k : KERNELS) kernel_ids.push_back(k.id); }
            else kernel_ids.push_back(std::stoi(v));
        } else if (a == "--sizes") {
            sizes.clear();
            while (i + 1 < argc && argv[i + 1][0] != '-') sizes.push_back(std::stoi(argv[++i]));
        } else if (a == "--mnk" && i + 3 < argc) {
            const int m = std::stoi(argv[++i]), nn = std::stoi(argv[++i]), k = std::stoi(argv[++i]);
            shapes.push_back({m, nn, k});
        } else if (a == "--reps" && i + 1 < argc) { reps = std::max(1, std::stoi(argv[++i])); }
        else if (a == "--warmup" && i + 1 < argc) { warmup = std::max(0, std::stoi(argv[++i])); }
        else if (a == "--csv" && i + 1 < argc) { csv_path = argv[++i]; }
        else if (a == "--no-check") { check = false; }
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else { std::fprintf(stderr, "unknown argument: %s\n", a.c_str()); usage(); return 1; }
    }
    if (kernel_ids.empty()) for (const auto& k : KERNELS) kernel_ids.push_back(k.id);
    // --sizes gives square problems; --mnk adds arbitrary shapes. If only --mnk was
    // given, don't also run the default square sizes.
    bool sizes_given = false;
    for (int i = 1; i < argc; ++i) if (std::string(argv[i]) == "--sizes") sizes_given = true;
    if (shapes.empty() || sizes_given) {
        std::vector<Shape> sq;
        for (int n : sizes) sq.push_back({n, n, n});
        shapes.insert(shapes.begin(), sq.begin(), sq.end());
    }
    for (int id : kernel_ids) {
        if (!find_kernel(id)) {
            std::fprintf(stderr, "kernel %d does not exist\n", id);
            return 1;
        }
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("Device: %s | SMs: %d | compute capability %d.%d\n",
                prop.name, prop.multiProcessorCount, prop.major, prop.minor);
    std::printf("reps=%d warmup=%d check=%s\n\n", reps, warmup, check ? "on" : "off");

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    if (std::filesystem::path(csv_path).has_parent_path())
        std::filesystem::create_directories(std::filesystem::path(csv_path).parent_path());
    FILE* csv = std::fopen(csv_path.c_str(), "w");
    if (!csv) { std::fprintf(stderr, "cannot open %s\n", csv_path.c_str()); return 1; }
    std::fprintf(csv, "kernel_id,kernel,M,N,K,time_ms,gflops,pct_cublas_fp32,max_err,status\n");

    std::printf("%-3s %-17s %16s %11s %10s %9s %10s  %s\n",
                "id", "kernel", "MxNxK", "time_ms", "GFLOPS", "%cuBLAS", "max_err", "status");

    bool all_ok = true;
    for (const Shape& sh : shapes) {
        const int M = sh.M, N = sh.N, K = sh.K;
        char dims[48];
        std::snprintf(dims, sizeof(dims), "%dx%dx%d", M, N, K);
        const size_t szA = (size_t)M * K, szB = (size_t)K * N, szC = (size_t)M * N;
        std::vector<float> hA(szA), hB(szB), hC(szC), hRef(szC);
        fill_random(hA, 1234u + M + 7u * K);
        fill_random(hB, 5678u + N + 3u * K);

        DeviceBuffers d;
        CUDA_CHECK(cudaMalloc(&d.A, szA * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d.B, szB * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d.C, szC * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d.A16, szA * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d.B16, szB * sizeof(half)));
        CUDA_CHECK(cudaMemcpy(d.A, hA.data(), szA * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d.B, hB.data(), szB * sizeof(float), cudaMemcpyHostToDevice));
        LAUNCH(convert_f32_to_f16, dim3(CEIL_DIV((int)szA, 256)), dim3(256), d.A, d.A16, (int)szA);
        LAUNCH(convert_f32_to_f16, dim3(CEIL_DIV((int)szB, 256)), dim3(256), d.B, d.B16, (int)szB);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Reference result + baseline time from cuBLAS FP32
        float t_cublas = -1.0f;
        if (check) {
            CUDA_CHECK(cudaMemset(d.C, 0, szC * sizeof(float)));
            run_kernel(0, M, N, K, d, handle);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(hRef.data(), d.C, szC * sizeof(float), cudaMemcpyDeviceToHost));
            t_cublas = time_kernel(0, M, N, K, d, handle, warmup, reps);
        }

        for (int id : kernel_ids) {
            const KernelInfo* ki = find_kernel(id);
            if (!supported(id, M, N, K)) {
                std::printf("%-3d %-17s %16s %11s %10s %9s %10s  skipped (size not supported)\n",
                            id, ki->name, dims, "-", "-", "-", "-");
                std::fprintf(csv, "%d,%s,%d,%d,%d,,,,,unsupported\n", id, ki->name, M, N, K);
                continue;
            }

            // 1) correctness (C is cleared first so a broken kernel cannot pass by accident)
            double err = 0.0;
            const char* status = "ok";
            if (check) {
                CUDA_CHECK(cudaMemset(d.C, 0, szC * sizeof(float)));
                run_kernel(id, M, N, K, d, handle);
                CUDA_CHECK(cudaDeviceSynchronize());
                CUDA_CHECK(cudaMemcpy(hC.data(), d.C, szC * sizeof(float), cudaMemcpyDeviceToHost));
                err = normalized_max_error(hC, hRef);
                if (!(err <= ki->tol)) { status = "FAILED"; all_ok = false; }
            }

            // 2) timing (only for verified kernels)
            float t = -1.0f;
            if (std::string(status) == "ok") t = time_kernel(id, M, N, K, d, handle, warmup, reps);
            const double gflops = t > 0 ? 2.0 * M * N * (double)K / (t * 1e-3) / 1e9 : 0.0;
            const double pct = (t > 0 && t_cublas > 0) ? 100.0 * t_cublas / t : 0.0;

            std::printf("%-3d %-17s %16s %11.4f %10.1f %8.1f%% %10.2e  %s\n",
                        id, ki->name, dims, t, gflops, pct, err, status);
            std::fprintf(csv, "%d,%s,%d,%d,%d,%.6f,%.3f,%.3f,%.3e,%s\n",
                         id, ki->name, M, N, K, t, gflops, pct, err, status);
        }
        std::printf("\n");

        CUDA_CHECK(cudaFree(d.A));
        CUDA_CHECK(cudaFree(d.B));
        CUDA_CHECK(cudaFree(d.C));
        CUDA_CHECK(cudaFree(d.A16));
        CUDA_CHECK(cudaFree(d.B16));
    }

    std::fclose(csv);
    CUBLAS_CHECK(cublasDestroy(handle));
    std::printf("Results written to %s\n", csv_path.c_str());
    if (!all_ok) { std::printf("SOME KERNELS FAILED the correctness check.\n"); return 2; }
    return 0;
}
