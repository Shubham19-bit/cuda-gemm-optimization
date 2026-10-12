#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# profile_all.sh - Nsight Compute metrics for every kernel at one size.
#
#   bash scripts/profile_all.sh            # size 4096, kernels 1 2 3 4 5 6 8
#   SIZE=2048 KERNELS="5 6" bash scripts/profile_all.sh
#
# Writes results/ncu/k<id>.csv (raw) and results/ncu_summary.csv / .md (table).
# If you see ERR_NVGPUCTRPERM, the platform blocks GPU performance counters:
# try Kaggle, or skip profiling (see PATHWAY.md, week 2.1).
# -----------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/.."

SIZE="${SIZE:-4096}"
KERNELS="${KERNELS:-1 2 3 4 5 6 8}"
OUT=results/ncu
mkdir -p "$OUT"

METRICS=(
  gpu__time_duration.sum
  dram__bytes.sum
  dram__throughput.avg.pct_of_peak_sustained_elapsed
  sm__throughput.avg.pct_of_peak_sustained_elapsed
  sm__warps_active.avg.pct_of_peak_sustained_active
  launch__registers_per_thread
  launch__occupancy_limit_registers
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum
  smsp__sass_thread_inst_executed_op_ffma_pred_on.sum
)
METRIC_LIST=$(IFS=,; echo "${METRICS[*]}")

[[ -x ./gemm ]] || make

for k in $KERNELS; do
  echo ">>> profiling kernel $k at ${SIZE}^3"
  # -k regex:mygemm  -> only our kernels (cuBLAS kernel names also contain 'sgemm')
  # -c 1             -> first matching launch only
  ncu --metrics "$METRIC_LIST" --csv --print-units base \
      -k regex:mygemm -c 1 --log-file "$OUT/k${k}.csv" \
      ./gemm --kernel "$k" --sizes "$SIZE" --reps 1 --warmup 0 --no-check \
      --csv "$OUT/run_k${k}.csv" > /dev/null
done

python3 scripts/parse_ncu.py "$OUT" --size "$SIZE"
