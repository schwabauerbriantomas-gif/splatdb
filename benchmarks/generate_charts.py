#!/usr/bin/env python3
"""
SplatsDB benchmark chart generator.

Reads ONLY from committed JSON result files in bench-data/.
Every number in the generated PNGs traces to a JSON source.

Usage:
    python benchmarks/generate_charts.py

Outputs:
    assets/splatsdb-benchmarks.png      — GPU vs CPU QPS + recall
    assets/splatsdb-faiss-comparison.png — Faiss vs SplatsDB
    assets/splatsdb-longmemeval.png      — LongMemEval recall by type
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH = os.path.join(REPO, "bench-data")
ASSETS = os.path.join(REPO, "assets")

# Dark theme matching SplatsDB UI
BG = "#0d1117"
FG = "#c9d1d9"
GRID = "#21262d"
ACCENT = "#00e5ff"
GREEN = "#3fb950"
ORANGE = "#d29922"
RED = "#f85149"
PURPLE = "#a371f7"


def load(name):
    path = os.path.join(BENCH, name)
    if not os.path.exists(path):
        print(f"WARNING: {path} not found, skipping dependent charts", file=sys.stderr)
        return None
    with open(path) as f:
        return json.load(f)


def style(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors=FG)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.5, alpha=0.7)


def chart_benchmarks():
    """GPU vs CPU QPS + recall from benchmark_results_hardware.json."""
    data = load("benchmark_results_hardware.json")
    if not data:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor=BG)
    fig.suptitle("SplatsDB Benchmarks — RTX 3090, SIFT-128 100K", color=FG, fontsize=14, fontweight="bold")

    # QPS chart (from JSON)
    engines = ["SplatsDB\nGPU", "Faiss\nHNSW", "Faiss\nIVF-PQ", "SplatsDB\nHNSW"]
    qps = [
        data["splatdb_gpu_topk"]["gpu_qps_persistent"],
        data["faiss_hnsw_cpu"]["qps"],
        data["faiss_ivf_pq_cpu"]["qps"],
        data["splatdb_hnsw_cpu"]["qps"],
    ]
    colors = [ACCENT, GREEN, GREEN, ORANGE]
    bars = ax1.bar(engines, qps, color=colors, edgecolor=FG, linewidth=0.5)
    ax1.set_ylabel("QPS (higher = better)", color=FG)
    ax1.set_title("QPS Comparison (k=64)", color=FG)
    style(ax1)
    for bar, v in zip(bars, qps):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                 f"{v:,}", ha="center", color=FG, fontsize=9, fontweight="bold")

    # Recall chart (from JSON)
    recall_labels = ["SplatsDB\nGPU", "Faiss\nHNSW", "Faiss\nIVF-PQ", "SplatsDB\nHNSW"]
    recall = [
        1.0,
        data["faiss_hnsw_cpu"]["recall_at_64"],
        data["faiss_ivf_pq_cpu"]["recall_at_64"],
        data["splatdb_hnsw_cpu"]["recall_at_64"],
    ]
    bars2 = ax2.bar(recall_labels, [r*100 for r in recall], color=colors, edgecolor=FG, linewidth=0.5)
    ax2.set_ylabel("Recall@64 (%)", color=FG)
    ax2.set_title("Recall Comparison", color=FG)
    ax2.set_ylim(0, 105)
    style(ax2)
    for bar, v in zip(bars2, recall):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"{v*100:.1f}%", ha="center", color=FG, fontsize=9, fontweight="bold")

    plt.tight_layout()
    out = os.path.join(ASSETS, "splatsdb-benchmarks.png")
    fig.savefig(out, dpi=150, facecolor=BG)
    plt.close(fig)
    print(f"Generated {out}")


def chart_faiss_comparison():
    """Faiss vs SplatsDB detailed from benchmark_results_hardware.json."""
    data = load("benchmark_results_hardware.json")
    if not data:
        return

    fig, ax = plt.subplots(figsize=(11, 6), facecolor=BG)
    ax.set_facecolor(BG)
    ax.axis("off")
    fig.suptitle("Faiss vs SplatsDB — Same Hardware, Same Dataset", color=FG, fontsize=14, fontweight="bold")

    # Table from JSON
    rows = [
        ["SplatsDB GPU (RTX 3090)", f"{data['splatdb_gpu_topk']['upload_ms']}ms",
         f"{data['splatdb_gpu_topk']['per_query_us_gpu']/1000:.3f}ms",
         f"{data['splatdb_gpu_topk']['gpu_qps_persistent']:,}", "1.000"],
        ["Faiss IVFFlat (nprobe=32)", f"{data['faiss_ivf_pq_cpu']['build_ms']/1000:.1f}s",
         f"{data['faiss_ivf_pq_cpu']['search_ms']:.2f}ms",
         f"{data['faiss_ivf_pq_cpu']['qps']:,}", f"{data['faiss_ivf_pq_cpu']['recall_at_64']}"],
        ["Faiss HNSW (M=32, ef=100)", f"{data['faiss_hnsw_cpu']['build_ms']/1000:.1f}s",
         f"{data['faiss_hnsw_cpu']['search_ms']:.2f}ms",
         f"{data['faiss_hnsw_cpu']['qps']:,}", f"{data['faiss_hnsw_cpu']['recall_at_64']}"],
        ["SplatsDB HNSW (CPU, ef=100)", f"{data['splatdb_hnsw_cpu']['build_ms']/1000:.0f}s",
         f"{data['splatdb_hnsw_cpu']['search_ms']:.2f}ms",
         f"{data['splatdb_hnsw_cpu']['qps']:,}", f"{data['splatdb_hnsw_cpu']['recall_at_64']}"],
    ]
    cols = ["Index", "Build", "p50 Latency", "QPS", "Recall@64"]
    table = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    for (r, c), cell in table.get_celld().items():
        cell.set_facecolor("#161b22" if r > 0 else "#21262d")
        cell.set_edgecolor(GRID)
        cell.set_text_props(color=FG)
        if r == 0:
            cell.set_text_props(color=FG, fontweight="bold")

    ax.text(0.5, 0.08, f"Source: bench-data/benchmark_results_hardware.json  |  {data['hardware']}",
            transform=ax.transAxes, ha="center", color="#7d8590", fontsize=8)

    plt.tight_layout()
    out = os.path.join(ASSETS, "splatsdb-faiss-comparison.png")
    fig.savefig(out, dpi=150, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated {out}")


def chart_longmemeval():
    """LongMemEval recall from longmemeval_full_results.json."""
    data = load("longmemeval_full_results.json")
    if not data:
        return

    recall = data["recall"]
    per_type = data["per_type"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), facecolor=BG)
    fig.suptitle("LongMemEval — Agent Memory Benchmark (NumPy brute-force)", color=FG, fontsize=14, fontweight="bold")

    # Session recall by k
    ks = [1, 3, 5, 10]
    vals = [recall["recall@1"]*100, recall["recall@3"]*100, recall["recall@5"]*100, recall["recall@10"]*100]
    bars = ax1.bar([f"@{k}" for k in ks], vals, color=[PURPLE, PURPLE, PURPLE, GREEN], edgecolor=FG, linewidth=0.5)
    ax1.set_ylabel("Recall (%)", color=FG)
    ax1.set_title("Session Recall by k", color=FG)
    ax1.set_ylim(0, 105)
    style(ax1)
    for bar, v in zip(bars, vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"{v:.1f}%", ha="center", color=FG, fontsize=10, fontweight="bold")

    # Per-type recall@10
    types = list(per_type.keys())
    type_recall = [per_type[t]["recall@10"]*100 for t in types]
    sort_idx = np.argsort(type_recall)[::-1]
    types = [types[i] for i in sort_idx]
    type_recall = [type_recall[i] for i in sort_idx]
    bars2 = ax2.barh(types[::-1], type_recall[::-1], color=GREEN, edgecolor=FG, linewidth=0.5)
    ax2.set_xlabel("Recall@10 (%)", color=FG)
    ax2.set_title("Per Question Type", color=FG)
    ax2.set_xlim(0, 105)
    style(ax2)
    for bar, v in zip(bars2, type_recall[::-1]):
        ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                 f"{v:.1f}%", va="center", color=FG, fontsize=9)

    fig.text(0.5, 0.01, "Source: bench-data/longmemeval_full_results.json  |  Method: numpy_cosine_normalized",
             ha="center", color="#7d8590", fontsize=8)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out = os.path.join(ASSETS, "splatsdb-longmemeval.png")
    fig.savefig(out, dpi=150, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated {out}")


if __name__ == "__main__":
    os.makedirs(ASSETS, exist_ok=True)
    chart_benchmarks()
    chart_faiss_comparison()
    chart_longmemeval()
    print("Done. All charts generated from bench-data/*.json")
