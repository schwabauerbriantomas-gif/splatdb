# M2M-Rust Benchmark Suite

Automated benchmarks for M2M vector search engine.

## Quick Start

```bash
# Build the binary first
cargo build --release --features cuda

# Run full benchmark suite
python benchmarks/run_benchmarks.py --binary target/release/m2m-rust.exe

# Quick run (1K + 10K vectors only)
python benchmarks/run_benchmarks.py --binary target/release/m2m-rust.exe --quick

# CPU only
python benchmarks/run_benchmarks.py --no-cuda

# Compare vs previous run
python benchmarks/run_benchmarks.py --binary target/release/m2m-rust.exe --compare
```

## What It Runs

### bench-gpu
GPU vs CPU search performance with varied dataset sizes:

| Config | Vectors | Dim | Queries | Top-K | Metric |
|--------|---------|-----|---------|-------|--------|
| Small  | 1K      | 640 | 100     | 10    | L2     |
| Medium | 10K     | 640 | 100     | 10    | L2     |
| Large  | 100K    | 640 | 100     | 10    | L2     |
| Large cosine | 100K | 640 | 100 | 10 | Cosine |

Measures: CPU QPS, GPU upload QPS, GPU persistent QPS.

### bench-gpu-ingest
Full pipeline: raw vectors → KMeans → splat centroids → indexed search.

| Config | Vectors | Clusters | Queries |
|--------|---------|----------|---------|
| Small  | 10K     | 50       | 100     |
| Large  | 100K    | 100      | 100     |

## Results Format

Results are saved to `benchmarks/results/YYYY-MM-DD.json`:

```json
{
  "date": "2026-03-30",
  "hardware": { "cpu": "...", "gpu": "..." },
  "benchmarks": [
    {
      "command": "bench-gpu",
      "config": { "n_vectors": 10000, "dim": 640, ... },
      "wall_time_s": 12.5,
      "cpu": { "total_ms": 372.0, "qps": 269.0 },
      "gpu_persistent": { "total_ms": 60.0, "qps": 1667.0 }
    }
  ]
}
```

## Flags

| Flag | Description |
|------|-------------|
| `--binary PATH` | Path to pre-built m2m-rust binary |
| `--quick` | Only run 1K and 10K configs |
| `--no-cuda` | CPU-only, skip CUDA features |
| `--compare` | Compare results vs previous run |
| `--output PATH` | Custom output file path |

## Notes

- First run compiles Rust code (slow). Use `--binary` with a pre-built release binary.
- `bench-gpu-ingest` requires CUDA features.
- Results are append-only: each run creates a new dated file.
