# Build the SGEMM benchmark.
#   make            -> ./gemm      (needs nvcc + cuBLAS; default arch sm_75 = Tesla T4)
#   make ARCH=sm_80 -> e.g. A100;  sm_86 = RTX 30xx;  sm_89 = RTX 40xx;  sm_60 = P100
#   make emu        -> ./gemm_emu  (CPU emulator, no GPU needed - checks correctness only)
#   make test-emu   -> build + run the emulator on small sizes
NVCC   ?= nvcc
ARCH   ?= sm_75
NVFLAGS = -O3 -std=c++17 -arch=$(ARCH) -lineinfo -Xptxas -v
SRC     = src/main.cu
HDRS    = $(wildcard src/*.cuh src/kernels/*.cuh)

all: gemm

gemm: $(SRC) $(HDRS)
	$(NVCC) $(NVFLAGS) $(SRC) -o $@ -lcublas 2> results_ptxas.txt || (cat results_ptxas.txt; exit 1)
	@echo "Built ./gemm  (register/smem usage per kernel: results_ptxas.txt)"

# CPU emulation: compile the same .cu file as C++ with the CUDA shim
gemm_emu: $(SRC) $(HDRS) src/emu/cuda_emu.h
	g++ -O2 -std=c++17 -DEMU -x c++ -Wno-unknown-pragmas -pthread $(SRC) -o $@

emu: gemm_emu

test-emu: gemm_emu
	./gemm_emu --kernel all --sizes 128 256 --reps 1 --warmup 0 --csv results/emu_check.csv

clean:
	rm -f gemm gemm_emu results_ptxas.txt

.PHONY: all emu test-emu clean
