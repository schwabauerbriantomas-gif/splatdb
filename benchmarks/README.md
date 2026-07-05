# SplatsDB Benchmark Suite

Automated benchmarks for the SplatsDB vector search engine.

## Quick Start

```bash
# Build the binary first
cargo build --release --features cuda

# Built-in HNSW benchmark (CPU, no CUDA needed)
./target/release/splatsdb bench-hnsw \
  --train <train.bin> --queries <query.bin> --gt <gt.bin> \
  -d 128 -k 10 --metric l2 \
  --ef-search 100 --ef-construction 400 --over-fetch 2

# GPU benchmark (requires --features cuda)
./target/release/splatsdb bench-gpu --n-vectors 100000 --dim 640 --n-queries 100 --top-k 10
```

### Binary data format

`bench-hnsw` expects raw binary files (not HDF5):

| File | Format |
|------|--------|
| `--train` | `[u64 rows][u64 cols][f32 data row-major]` |
| `--queries` | `[u64 rows][u64 cols][f32 data row-major]` |
| `--gt` | `[u64 n_queries][u64 k][i64 indices]` |

To convert from ANN-Benchmarks HDF5 to this format, use `h5py` + `struct`.

## What It Runs

### bench-hnsw

HNSW graph search with persistence. Measures build time, p50/p95/p99 latency, QPS, and recall@k against ground truth.

| Config | Default |
|--------|---------|
| `--metric` | `l2` (or `cosine`) |
| `--ef-construction` | 400 |
| `--ef-search` | 100 |
| `--over-fetch` | 2 |
| `-k` | 10 |

### bench-gpu (requires `--features cuda`)

GPU vs CPU search performance with varied dataset sizes:

| Config | Vectors | Dim | Queries | Top-K | Metric |
|--------|---------|-----|---------|-------|--------|
| Small  | 1K      | 640 | 100     | 10    | L2     |
| Medium | 10K     | 640 | 100     | 10    | L2     |
| Large  | 100K    | 640 | 100     | 10    | L2     |

Measures: CPU QPS, GPU upload QPS, GPU persistent QPS.

## Results

Results from the latest runs are stored in [`bench-data/`](../bench-data/). Each JSON is self-documenting with hardware metadata, timestamps, and an integrity checklist.

## Reproducibility

All benchmark numbers published in the [main README](../README.md) trace to a JSON in `bench-data/`. If a number cannot be traced to a results file, it should not be published. See the README's **Integrity pledge** section.
