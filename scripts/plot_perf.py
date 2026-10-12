"""
plot_perf.py - Plots from the benchmark CSV written by ./gemm.

  python scripts/plot_perf.py results/results.csv

Writes to results/:
  perf_vs_size.png      GFLOPS vs matrix size, one line per kernel
  pct_cublas.png        % of cuBLAS FP32 at the largest size (bar chart)
  summary.md            markdown table for the README
"""
from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["status"] != "ok" or not r["gflops"]:
                continue
            r["kernel_id"] = int(r["kernel_id"])
            for k in ("M", "N", "K"):
                r[k] = int(r[k])
            for k in ("time_ms", "gflops", "pct_cublas_fp32", "max_err"):
                r[k] = float(r[k]) if r[k] else 0.0
            rows.append(r)
    return rows


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "results/results.csv"
    rows = load(path)
    if not rows:
        print(f"No successful rows in {path}")
        return
    out = os.path.dirname(os.path.abspath(path))
    square = [r for r in rows if r["M"] == r["N"] == r["K"]]

    by_kernel = defaultdict(list)
    for r in square:
        by_kernel[(r["kernel_id"], r["kernel"])].append(r)

    # 1) GFLOPS vs size
    fig, ax = plt.subplots(figsize=(8, 5))
    for (kid, name), rs in sorted(by_kernel.items()):
        rs.sort(key=lambda r: r["M"])
        style = "--" if name.startswith("cublas") else "-"
        ax.plot([r["M"] for r in rs], [r["gflops"] for r in rs], style, marker="o",
                label=f"{kid}: {name}")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("matrix size (M = N = K)")
    ax.set_ylabel("GFLOPS")
    ax.set_title("SGEMM performance by kernel")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "perf_vs_size.png"), dpi=150)

    # 2) % of cuBLAS at the largest size
    nmax = max(r["M"] for r in square)
    big = sorted([r for r in square if r["M"] == nmax], key=lambda r: r["kernel_id"])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    names = [f"{r['kernel_id']}: {r['kernel']}" for r in big]
    vals = [r["pct_cublas_fp32"] for r in big]
    bars = ax.barh(names, vals)
    for b, v in zip(bars, vals):
        ax.text(b.get_width() + 1, b.get_y() + b.get_height() / 2, f"{v:.1f}%",
                va="center", fontsize=8)
    ax.axvline(100, color="k", lw=0.8, ls="--")
    ax.invert_yaxis()
    ax.set_xlabel("% of cuBLAS FP32 performance")
    ax.set_title(f"Relative performance at {nmax}x{nmax}x{nmax}")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "pct_cublas.png"), dpi=150)

    # 3) markdown summary
    k1 = next((r for r in big if r["kernel_id"] == 1), None)
    lines = [f"Results at {nmax}x{nmax}x{nmax}\n",
             "| Kernel | GFLOPS | % cuBLAS FP32 | Speedup vs K1 | max err |",
             "|---|---|---|---|---|"]
    for r in big:
        sp = f"{r['gflops'] / k1['gflops']:.1f}x" if k1 and k1["gflops"] > 0 else "-"
        lines.append(f"| {r['kernel_id']}: {r['kernel']} | {r['gflops']:.1f} | "
                     f"{r['pct_cublas_fp32']:.1f}% | {sp} | {r['max_err']:.1e} |")
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nPlots written to {out}")


if __name__ == "__main__":
    main()
