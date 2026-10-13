"""
roofline.py - Roofline plot for the SGEMM kernels.

  python scripts/roofline.py                          # uses results/ncu_summary.csv if present
  python scripts/roofline.py --results results/results.csv --size 4096
  python scripts/roofline.py --peak-gflops 8100 --bw 320   # Tesla T4 defaults

Arithmetic intensity (FLOP per DRAM byte):
  * measured  - from Nsight (dram__bytes.sum) when results/ncu_summary.csv exists
  * estimated - otherwise, from the tiling: with a BM x BN block tile, A is read
                N/BN times and B is read M/BM times from DRAM (ignores L2 hits,
                so it is a lower bound on intensity). K1/K2 have no explicit tiling
                and are only plotted when measured.
"""
from __future__ import annotations

import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# (BM, BN, bytes per input element) for the explicitly tiled kernels
TILES = {3: (32, 32, 4), 4: (64, 64, 4), 5: (128, 128, 4), 6: (128, 128, 4), 7: (128, 128, 4), 8: (16, 16, 2)}


def estimated_intensity(kid, n):
    if kid not in TILES:
        return None
    bm, bn, eb = TILES[kid]
    flops = 2.0 * n ** 3
    bytes_ = eb * (n ** 3 / bn + n ** 3 / bm) + 4 * n * n     # A, B re-reads + C write
    return flops / bytes_


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/results.csv")
    ap.add_argument("--ncu", default="results/ncu_summary.csv")
    ap.add_argument("--size", type=int, default=4096)
    ap.add_argument("--peak-gflops", type=float, default=8100.0, help="FP32 peak (T4 ~8100)")
    ap.add_argument("--bw", type=float, default=320.0, help="DRAM GB/s (T4 = 320)")
    args = ap.parse_args()

    perf = {}
    if os.path.exists(args.results):
        for r in csv.DictReader(open(args.results)):
            if r["status"] == "ok" and r["gflops"] and int(r["M"]) == int(r["N"]) == int(r["K"]) == args.size:
                perf[int(r["kernel_id"])] = (r["kernel"], float(r["gflops"]))

    measured = {}
    if os.path.exists(args.ncu):
        for r in csv.DictReader(open(args.ncu)):
            if r.get("intensity"):
                measured[int(r["kernel_id"])] = float(r["intensity"])

    if not perf:
        print(f"No rows for size {args.size} in {args.results}. Run ./gemm first.")
        return

    ridge = args.peak_gflops / args.bw
    xs = [2 ** (e / 4) for e in range(-8, 48)]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(xs, [min(args.peak_gflops, args.bw * x) for x in xs], "k-",
              label=f"FP32 roofline ({args.peak_gflops:.0f} GFLOPS, {args.bw:.0f} GB/s)")
    ax.axvline(ridge, color="gray", ls=":", lw=0.8)
    ax.text(ridge * 1.05, args.peak_gflops * 0.05, f"ridge {ridge:.1f} FLOP/B",
            fontsize=8, color="gray")

    for kid, (name, gf) in sorted(perf.items()):
        if kid in measured:
            ai, tag, marker = measured[kid], "measured", "o"
        else:
            ai, tag, marker = estimated_intensity(kid, args.size), "estimated", "s"
        if ai is None:
            continue
        ax.plot(ai, gf, marker, ms=8, label=f"{kid}: {name} ({tag})")

    ax.set_xlabel("arithmetic intensity (FLOP / DRAM byte)")
    ax.set_ylabel("achieved GFLOPS")
    ax.set_title(f"Roofline - SGEMM {args.size}^3")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(args.results)), "roofline.png")
    fig.savefig(out, dpi=150)
    print(f"Roofline written to {out}")


if __name__ == "__main__":
    main()
