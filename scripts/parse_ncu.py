"""
parse_ncu.py - Turn Nsight Compute CSV logs (from profile_all.sh) into one table.

  python scripts/parse_ncu.py results/ncu --size 4096

Writes results/ncu_summary.csv (used by roofline.py) and results/ncu_summary.md.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re

SHORT = {
    "gpu__time_duration.sum": "time_ns",
    "dram__bytes.sum": "dram_bytes",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed": "dram_pct",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "sm_pct",
    "sm__warps_active.avg.pct_of_peak_sustained_active": "occupancy_pct",
    "launch__registers_per_thread": "regs",
    "launch__occupancy_limit_registers": "blocks_limit_regs",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum": "smem_ld_conflicts",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum": "smem_st_conflicts",
    "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum": "ffma_thread_inst",
}


def to_num(v: str):
    v = v.replace(",", "").strip()
    try:
        return float(v)
    except ValueError:
        return None


def parse_file(path):
    """Return {short_metric: value} from an ncu --csv log (ignores ==PROF== lines)."""
    with open(path, errors="replace") as f:
        lines = [ln for ln in f if ln.startswith('"')]
    if not lines:
        return {}
    vals = {}
    reader = csv.DictReader(lines)
    for row in reader:
        name = row.get("Metric Name", "")
        if name in SHORT:
            num = to_num(row.get("Metric Value", ""))
            if num is not None:
                vals[SHORT[name]] = num
        vals.setdefault("kernel_name", row.get("Kernel Name", ""))
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", nargs="?", default="results/ncu")
    ap.add_argument("--size", type=int, default=4096)
    args = ap.parse_args()

    rows = []
    for p in sorted(glob.glob(os.path.join(args.dir, "k*.csv"))):
        m = re.search(r"k(\d+)\.csv$", p)
        if not m:
            continue
        v = parse_file(p)
        if not v or "time_ns" not in v:
            print(f"{p}: no metrics found (counters blocked? see the file for the ncu message)")
            continue
        kid = int(m.group(1))
        n = args.size
        flops = 2.0 * n * n * n
        v["kernel_id"] = kid
        v["gflops_ncu"] = flops / v["time_ns"]                       # FLOP/ns = GFLOP/s
        v["intensity"] = flops / v["dram_bytes"] if v.get("dram_bytes") else None
        rows.append(v)

    if not rows:
        print("Nothing to summarise.")
        return

    cols = ["kernel_id", "kernel_name", "time_ns", "gflops_ncu", "dram_bytes", "intensity",
            "dram_pct", "sm_pct", "occupancy_pct", "regs", "smem_ld_conflicts",
            "smem_st_conflicts"]
    out_dir = os.path.dirname(os.path.abspath(args.dir))
    with open(os.path.join(out_dir, "ncu_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    def fmt(x, nd=1):
        return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))

    md = ["| id | GFLOPS | DRAM GB | FLOP/B | DRAM % | SM % | occupancy % | regs | smem ld/st conflicts |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(
            f"| {r['kernel_id']} | {fmt(r.get('gflops_ncu'))} | "
            f"{fmt((r.get('dram_bytes') or 0) / 1e9, 2)} | {fmt(r.get('intensity'))} | "
            f"{fmt(r.get('dram_pct'))} | {fmt(r.get('sm_pct'))} | {fmt(r.get('occupancy_pct'))} | "
            f"{fmt(r.get('regs'), 0)} | {fmt(r.get('smem_ld_conflicts'), 0)} / "
            f"{fmt(r.get('smem_st_conflicts'), 0)} |")
    text = "\n".join(md)
    with open(os.path.join(out_dir, "ncu_summary.md"), "w") as f:
        f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
