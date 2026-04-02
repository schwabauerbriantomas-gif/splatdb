#!/usr/bin/env python3
"""M2M-Rust Automated Benchmark Suite.

Runs bench-gpu and bench-gpu-ingest commands with varied configurations,
parses stdout, and saves structured JSON results.

Usage:
    python benchmarks/run_benchmarks.py                    # run all benchmarks
    python benchmarks/run_benchmarks.py --compare           # compare vs last result
    python benchmarks/run_benchmarks.py --quick             # only 1K/10K vectors
    python benchmarks/run_benchmarks.py --binary ./target/release/m2m-rust.exe
    python benchmarks/run_benchmarks.py --no-cuda           # CPU-only benchmarks

Results saved to: benchmarks/results/YYYY-MM-DD.json
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = Path(__file__).resolve().parent / "results"

DEFAULT_BINARY = str(PROJECT_ROOT / "target" / "release" / "m2m-rust.exe")

BENCH_GPU_CONFIGS = [
    # (n_vectors, dim, n_queries, top_k, metric)
    (1_000, 640, 100, 10, "l2"),
    (10_000, 640, 100, 10, "l2"),
    (100_000, 640, 100, 10, "l2"),
    (100_000, 640, 100, 10, "cosine"),
]

BENCH_GPU_INGEST_CONFIGS = [
    # (n_vectors, dim, n_clusters, n_queries)
    (10_000, 640, 50, 100),
    (100_000, 640, 100, 100),
]

QUICK_CONFIGS_GPU = [
    (1_000, 640, 100, 10, "l2"),
    (10_000, 640, 100, 10, "l2"),
]

QUICK_CONFIGS_INGEST = [
    (10_000, 640, 50, 100),
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_command(cmd: list[str], timeout: int = 600) -> tuple[str, float]:
    """Run a command and return (stdout, elapsed_seconds)."""
    t0 = time.perf_counter()
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, cwd=str(PROJECT_ROOT)
    )
    elapsed = time.perf_counter() - t0
    if result.returncode != 0:
        print(f"  WARN: command failed (rc={result.returncode})")
        print(f"  stderr: {result.stderr[:500]}")
    return result.stdout + result.stderr, elapsed


def parse_bench_gpu(output: str, wall_time: float, config: dict) -> dict:
    """Parse bench-gpu output into structured metrics."""
    metrics: dict[str, Any] = {
        "command": "bench-gpu",
        "config": config,
        "wall_time_s": round(wall_time, 3),
        "cpu": None,
        "gpu_upload": None,
        "gpu_persistent": None,
    }

    # CPU benchmark patterns
    cpu_match = re.search(r"CPU.*?(\d+(?:\.\d+)?)\s*ms.*?(\d+(?:\.\d+)?)\s*QPS", output, re.IGNORECASE | re.DOTALL)
    if cpu_match:
        metrics["cpu"] = {
            "total_ms": float(cpu_match.group(1)),
            "qps": float(cpu_match.group(2)),
        }

    # GPU upload patterns
    gpu_upload = re.search(
        r"GPU.*upload.*?(\d+(?:\.\d+)?)\s*ms.*?(\d+(?:\.\d+)?)\s*QPS",
        output, re.IGNORECASE | re.DOTALL,
    )
    if gpu_upload:
        metrics["gpu_upload"] = {
            "total_ms": float(gpu_upload.group(1)),
            "qps": float(gpu_upload.group(2)),
        }

    # GPU persistent patterns
    gpu_persist = re.search(
        r"(?:GPU.*persistent|dataset in VRAM).*?(\d+(?:\.\d+)?)\s*ms.*?(\d+(?:\.\d+)?)\s*QPS",
        output, re.IGNORECASE | re.DOTALL,
    )
    if gpu_persist:
        metrics["gpu_persistent"] = {
            "total_ms": float(gpu_persist.group(1)),
            "qps": float(gpu_persist.group(2)),
        }

    # Fallback: try to find any timing lines
    if metrics["cpu"] is None and metrics["gpu_upload"] is None:
        # Store raw output for manual inspection
        metrics["raw_output"] = output[:2000]

    return metrics


def parse_bench_gpu_ingest(output: str, wall_time: float, config: dict) -> dict:
    """Parse bench-gpu-ingest output."""
    metrics: dict[str, Any] = {
        "command": "bench-gpu-ingest",
        "config": config,
        "wall_time_s": round(wall_time, 3),
        "phases": {},
    }

    # Common phase patterns: "<Phase Name> ... XXX ms"
    phase_patterns = [
        (r"ingest.*?(\d+(?:\.\d+)?)\s*ms", "ingest_ms"),
        (r"index.*?(\d+(?:\.\d+)?)\s*ms", "index_build_ms"),
        (r"linear.*?(\d+(?:\.\d+)?)\s*ms", "linear_search_ms"),
        (r"GPU.*?(\d+(?:\.\d+)?)\s*ms", "gpu_search_ms"),
        (r"total.*?(\d+(?:\.\d+)?)\s*ms", "total_ms"),
    ]

    for pattern, key in phase_patterns:
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            metrics["phases"][key] = float(match.group(1))

    if not metrics["phases"]:
        metrics["raw_output"] = output[:2000]

    return metrics


def run_bench_gpu(binary: str, use_cuda: bool, configs: list) -> list[dict]:
    """Run bench-gpu with multiple configurations."""
    results = []
    features = "--features cuda" if use_cuda else ""
    cargo_prefix = ["cargo", "run", "--release"]
    if use_cuda:
        cargo_prefix += ["--features", "cuda"]
    cargo_prefix += ["--"]

    for n, dim, q, k, metric in configs:
        config = {"n_vectors": n, "dim": dim, "n_queries": q, "top_k": k, "metric": metric}
        cmd = cargo_prefix + ["bench-gpu", "-n", str(n), "-d", str(dim), "-q", str(q), "-k", str(k), "-m", metric]

        # Try pre-built binary first, fall back to cargo
        if binary and Path(binary).exists():
            cmd = [binary, "bench-gpu", "-n", str(n), "-d", str(dim), "-q", str(q), "-k", str(k), "-m", metric]

        print(f"  bench-gpu: n={n:,} d={dim} q={q} k={k} metric={metric}")
        output, elapsed = run_command(cmd, timeout=max(300, n // 100))
        result = parse_bench_gpu(output, elapsed, config)
        results.append(result)

    return results


def run_bench_gpu_ingest(binary: str, use_cuda: bool, configs: list) -> list[dict]:
    """Run bench-gpu-ingest with multiple configurations."""
    results = []
    cargo_prefix = ["cargo", "run", "--release"]
    if use_cuda:
        cargo_prefix += ["--features", "cuda"]
    cargo_prefix += ["--"]

    for n, dim, k, q in configs:
        config = {"n_vectors": n, "dim": dim, "n_clusters": k, "n_queries": q}
        cmd = cargo_prefix + ["bench-gpu-ingest", "-n", str(n), "-d", str(dim), "-k", str(k), "-q", str(q)]

        if binary and Path(binary).exists():
            cmd = [binary, "bench-gpu-ingest", "-n", str(n), "-d", str(dim), "-k", str(k), "-q", str(q)]

        print(f"  bench-gpu-ingest: n={n:,} d={dim} clusters={k} q={q}")
        output, elapsed = run_command(cmd, timeout=max(600, n // 50))
        result = parse_bench_gpu_ingest(output, elapsed, config)
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------

def load_previous_result() -> dict | None:
    """Load the most recent previous result file."""
    if not RESULTS_DIR.exists():
        return None
    json_files = sorted(RESULTS_DIR.glob("*.json"))
    if len(json_files) < 1:
        return None
    # Get second-to-last if today's exists, otherwise last
    with open(json_files[-1]) as f:
        return json.load(f)


def compare_results(current: dict, previous: dict) -> None:
    """Print comparison between current and previous results."""
    if not previous:
        print("\nNo previous results to compare against.")
        return

    print(f"\n{'='*60}")
    print(f"COMPARISON vs {previous.get('date', 'unknown')}")
    print(f"{'='*60}")

    cur_benchmarks = {r["command"] + str(r.get("config", "")): r for r in current.get("benchmarks", [])}
    prev_benchmarks = {r["command"] + str(r.get("config", "")): r for r in previous.get("benchmarks", [])}

    for key in cur_benchmarks:
        if key not in prev_benchmarks:
            continue
        cur = cur_benchmarks[key]
        prev = prev_benchmarks[key]
        print(f"\n  {cur['command']} {cur.get('config', {})}")

        for mode in ["cpu", "gpu_upload", "gpu_persistent"]:
            if cur.get(mode) and prev.get(mode):
                cur_ms = cur[mode]["total_ms"]
                prev_ms = prev[mode]["total_ms"]
                change = ((cur_ms - prev_ms) / prev_ms) * 100 if prev_ms else 0
                arrow = "FASTER" if change < -2 else "SLOWER" if change > 2 else "SAME"
                print(f"    {mode}: {prev_ms:.1f}ms -> {cur_ms:.1f}ms ({change:+.1f}% {arrow})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="M2M-Rust Benchmark Suite")
    parser.add_argument("--binary", default="", help="Path to pre-built m2m-rust binary")
    parser.add_argument("--compare", action="store_true", help="Compare vs previous results")
    parser.add_argument("--quick", action="store_true", help="Only run small configs")
    parser.add_argument("--no-cuda", action="store_true", help="CPU-only (no --features cuda)")
    parser.add_argument("--output", default="", help="Custom output path (default: results/YYYY-MM-DD.json)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    output_path = Path(args.output) if args.output else RESULTS_DIR / f"{today}.json"

    binary = args.binary or ""
    use_cuda = not args.no_cuda

    print(f"M2M-Rust Benchmark Suite - {today}")
    print(f"Binary: {binary or 'cargo run (build on the fly)'}")
    print(f"CUDA: {'enabled' if use_cuda else 'disabled'}")
    print()

    all_results = []

    # bench-gpu
    gpu_configs = QUICK_CONFIGS_GPU if args.quick else BENCH_GPU_CONFIGS
    print("[1/2] Running bench-gpu benchmarks...")
    gpu_results = run_bench_gpu(binary, use_cuda, gpu_configs)
    all_results.extend(gpu_results)

    # bench-gpu-ingest (requires cuda)
    if use_cuda:
        ingest_configs = QUICK_CONFIGS_INGEST if args.quick else BENCH_GPU_INGEST_CONFIGS
        print("\n[2/2] Running bench-gpu-ingest benchmarks...")
        ingest_results = run_bench_gpu_ingest(binary, use_cuda, ingest_configs)
        all_results.extend(ingest_results)
    else:
        print("\n[2/2] Skipping bench-gpu-ingest (requires CUDA)")

    # Build result object
    result = {
        "date": today,
        "timestamp": time.time(),
        "hardware": {
            "cpu": "AMD Ryzen 5 3400G (4c/8t)",
            "ram_gb": 32,
            "gpu": "NVIDIA RTX 3090 (24GB VRAM)",
            "cuda": "12.4",
        },
        "build": {
            "cuda_enabled": use_cuda,
        },
        "benchmarks": all_results,
    }

    # Save
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for b in all_results:
        cmd = b["command"]
        cfg = b.get("config", {})
        wall = b["wall_time_s"]
        print(f"  {cmd}: wall={wall:.1f}s config={cfg}")

    # Compare
    if args.compare:
        prev = load_previous_result()
        if prev and prev.get("date") != today:
            compare_results(result, prev)
        elif prev:
            # Try to find an older one
            json_files = sorted(RESULTS_DIR.glob("*.json"))
            for jf in reversed(json_files[:-1] if len(json_files) > 1 else []):
                with open(jf) as f:
                    older = json.load(f)
                compare_results(result, older)
                break
            else:
                print("\nNo previous results to compare against.")


if __name__ == "__main__":
    main()
