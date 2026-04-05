# SplatDB

A native Rust vector search engine powered by Gaussian Splatting embeddings and hierarchical retrieval.

[![Version](https://img.shields.io/badge/version-2.5.0-blue.svg)](https://github.com/schwabauerbriantomas-gif/splatdb)
[![License](https://img.shields.io/badge/license-AGPL--3.0-green.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-2021-orange.svg)](https://www.rust-lang.org/)
[![Tests](https://img.shields.io/badge/tests-226%20passing-brightgreen.svg)]()

---

## What Is This?

SplatDB applies **Gaussian Splatting** — a technique from 3D neural rendering — to vector search. Instead of storing raw embedding vectors, each document is represented as a probabilistic Gaussian (mean μ, opacity α, concentration κ). This enables:

- **Richer similarity semantics** via splat overlap instead of point-to-point distance
- **Natural compression** through distribution parameters
- **Uncertainty-aware retrieval** — regions with sparse coverage have high energy, guiding active learning

Combined with a two-level KMeans++ retrieval pipeline (HRM2), CUDA GPU acceleration with custom PTX kernels, and hybrid BM25+vector semantic memory, SplatDB provides a full-featured vector search engine in pure Rust.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     CLI / MCP Server                          │
│              (clap / stdio JSON-RPC 2.0)                      │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐    │
│  │ Query Layer   │  │ Semantic     │  │ GraphSplat      │    │
│  │ HRM2 Two-     │  │ Memory       │  │ Knowledge       │    │
│  │ Level         │  │ BM25+Vector  │  │ Graph Overlay   │    │
│  │ Retrieval     │  │ RRF Fusion   │  │                 │    │
│  └──────┬───────┘  └──────┬───────┘  └───────┬─────────┘    │
│         └─────────────────┼──────────────────┘               │
│                           │                                   │
│  ┌────────────────────────▼────────────────────────────────┐ │
│  │            Gaussian Splat Embeddings                     │ │
│  │         (mean μ + opacity α + concentration κ)           │ │
│  ├──────────────────────────────────────────────────────────┤ │
│  │  Compression: TurboQuant / PolarQuant (3–8 bit)          │ │
│  ├──────────────────────────────────────────────────────────┤ │
│  │  Indexes: HNSW │ LSH │ KMeans++ (coarse → fine)         │ │
│  ├──────────────────────────────────────────────────────────┤ │
│  │  Persistence: SQLite      │  GPU: CUDA PTX (optional)   │ │
│  └────────────────────────────┴─────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Rust 1.56+ (Edition 2021)
- C compiler (for SQLite bundling via `rusqlite`)
- **For GPU**: CUDA Toolkit 12.x + MSVC `cl.exe` (nvcc compiles PTX kernels at build time)

### Build & Run

```bash
# CPU-only build (no GPU dependencies)
cargo build --release
cargo test --lib

# CUDA build (requires CUDA Toolkit + MSVC)
cargo build --release --features cuda
cargo test --lib --features cuda

# Basic usage
cargo run --release -- status --verbose
cargo run --release -- search --query "0.1,0.2,0.3,...,-0.5" -k 10
```

### Rust API

```rust
use splatdb::{SplatDBConfig, SplatStore};

fn main() {
    let config = SplatDBConfig::default();  // or: simple(), mcp(), advanced(), gpu()
    let mut store = SplatStore::new(config);

    let id = store.insert(&vec![0.1f32; 640]);
    let results = store.search(&vec![0.1f32; 640], 10);

    for r in &results {
        println!("id={} dist={:.4}", r.index, r.distance);
    }
}
```

## Benchmarks

All numbers are **measured on real hardware**, not estimated.

**Hardware**: AMD Ryzen 5 3400G (4c/8t), 32GB DDR4, NVIDIA RTX 3090 24GB, CUDA 12.4
**Software**: Rust 1.94.1, cudarc 0.19.4, ndarray 0.16, rayon 1.11

### GPU Top-K Search (Custom CUDA Kernels)

Combined distance + top-k selection in one GPU pass. Dataset persists in VRAM between queries — only top-k results downloaded.

| Dataset | Dim | Queries | k | CPU QPS | GPU Persistent QPS | Speedup |
|---------|-----|---------|---|---------|--------------------|---------|
| 10K | 640 | 100 | 10 | 269 | 1,667 | **6.2x** |
| 100K | 640 | 100 | 10 | 214 | 1,667 | **7.8x** |

- Upload bandwidth: **3.9 GB/s**
- GPU results verified against CPU: identical top-k indices and order
- Per-query latency with dataset in VRAM: **0.60 ms** (constant, independent of dataset size)

### Kernel Optimizations

- `float4` vectorized loads (640D = 160 × float4 per vector)
- Shared memory query cache (avoids redundant global reads per thread)
- Thread-local sorted top-k (max K=32) with shared memory merge
- `__launch_bounds__(256)` for optimal sm_86 occupancy
- PTX compiled with `--use_fast_math -O3`

### Python Prototype

An earlier Python prototype showed **32x speedup** with HRM2 vs linear scan on 100K vectors. These numbers do not apply to the Rust implementation and are referenced for historical context only.

## Configuration Presets

Seven presets cover edge devices to GPU clusters:

| Preset | Use Case | Memory | Key Features |
|--------|----------|--------|--------------|
| `default` | General purpose | 100K splats | TurboQuant 8-bit, GraphSplat, BM25+Vector RRF |
| `simple` | Edge / IoT | 10K splats | Minimal footprint, no compression, RAM-only |
| `mcp` | AI agent memory | 100K splats | GPU auto-detect, TurboQuant, GraphSplat, semantic RRF |
| `advanced` | AI agents | 1M splats | 4-bit quant, HNSW secondary, auto-scaling, EBM |
| `training` | Model research | 500K splats | Matryoshka, distillation, noise robustness |
| `distributed` | Multi-node | 10M splats | MapReduce 32-chunk, 3–50 nodes, sharding |
| `gpu` | CUDA servers | 5M splats | All advanced features + GPU batch, HNSW M=48 |

```rust
let edge  = SplatDBConfig::simple(None);           // Raspberry Pi
let mcp   = SplatDBConfig::mcp(None);              // AI agent (auto GPU)
let agent = SplatDBConfig::advanced(Some("cuda")); // LLM tool
let gpu   = SplatDBConfig::gpu(None);              // RTX 3090+
```

## Features

- **Gaussian Splat Embeddings** — Probabilistic document representation (μ + α + κ)
- **HRM2 Two-Level Retrieval** — Coarse KMeans++ → fine KMeans++ → exact re-rank
- **TurboQuant / PolarQuant** — 3–8 bit data-oblivious vector compression (no training/codebooks)
- **CUDA GPU Acceleration** — Custom PTX kernels with persistent VRAM (optional, `--features cuda`)
- **GraphSplat Knowledge Graph** — Typed nodes, weighted edges, graph-augmented retrieval
- **Semantic Memory** — Hybrid BM25 + vector search with Reciprocal Rank Fusion
- **HNSW & LSH Indexes** — Approximate nearest neighbor search
- **DatasetTransformer** — Raw vectors → KMeans splat centroids pipeline
- **Energy-Based Model** — Exploration via Boltzmann sampling, self-organized criticality
- **SQLite Persistence** — Durable storage with no external dependencies
- **MCP Server** — Model Context Protocol for LLM agent integration

## MCP Server

SplatDB includes an MCP server for integration with AI agents (Claude, GPT, open-source LLMs).

Protocol: stdio JSON-RPC 2.0

| Tool | Description |
|------|-------------|
| `store` | Store a vector embedding with optional metadata |
| `search` | Semantic search for similar vectors |
| `status` | Engine status and statistics |
| `doc_add` | Add a document with text and metadata |
| `doc_get` | Retrieve a stored document by ID |
| `doc_del` | Delete a document (soft delete) |

```bash
# Start the MCP server (integrates with Claude Code, etc.)
cargo run --release -- mcp
```

## CLI Reference

```
splatdb <COMMAND>

Core:
  index              Add vectors from binary file (u64 rows, u64 cols, f32 data)
  search             Search by comma-separated query vector
  search-file        Search using query vector from binary file
  status             Show store statistics
  save / load        Persist / restore state
  list               List all stored shards
  backup             Backup all data to directory

Ingestion:
  ingest             Ingest via DatasetTransformer (KMeans splat centroids)
  ingest-hierarchical  Two-level hierarchical KMeans
  ingest-leader      O(n) single-pass leader clustering

Search Backends:
  fused-search       Search all enabled backends with fusion
  hnsw-search        HNSW index only
  lsh-search         LSH index only

Quantization:
  quant-index        Build TurboQuant/PolarQuant compressed index
  quant-search       Search compressed codes
  quant-status       Show quantization statistics

GPU (requires --features cuda):
  gpu-info           Show CUDA device information
  bench-gpu          Benchmark GPU vs CPU search
  bench-gpu-ingest   Benchmark full GPU ingest + search pipeline

Energy / SOC:
  soc-check          Check system criticality
  soc-avalanche      Trigger avalanche reorganization
  soc-relax          Relax toward lower energy

Documents:
  doc-add            Add document with metadata
  doc-get            Retrieve document by ID
  doc-del            Delete document

Server:
  serve              Start HTTP API server
  mcp                Start MCP server (stdio JSON-RPC)
  preset-info        Show preset subsystem details

Options:
  --data-dir <DIR>     Storage directory [default: ./splatdb_data]
  --dim <DIM>          Vector dimensionality [default: 640]
  --max-splats <N>     Maximum splat capacity
  --backend <BACKEND>  Search backend (hrm2|hnsw|lsh)
```

## Module Map

| Module | Source | Description |
|--------|--------|-------------|
| `splats` | `src/splats.rs` | Main API — insert, search, upsert, delete |
| `hrm2_engine` | `src/hrm2_engine.rs` | Two-level hierarchical retrieval |
| `engine` | `src/engine.rs` | CPU L2 distance with rayon parallelism |
| `gpu` | `src/gpu/` | CUDA PTX kernels, `GpuIndex` persistent VRAM |
| `quantization` | `src/quantization.rs` | TurboQuant / PolarQuant compression |
| `clustering` | `src/clustering.rs` | KMeans++ with ChaCha8 RNG |
| `graph_splat` | `src/graph_splat.rs` | Knowledge graph overlay |
| `semantic_memory` | `src/semantic_memory.rs` | BM25 + vector, RRF fusion |
| `hnsw_index` | `src/hnsw_index.rs` | HNSW approximate nearest neighbor |
| `lsh_index` | `src/lsh_index.rs` | Locality-sensitive hashing |
| `energy` | `src/energy.rs` | Energy landscape, E(x) = −log(Σ αᵢ·exp(−κᵢ·‖x−μᵢ‖²)) |
| `ebm` | `src/ebm/` | Boltzmann exploration, self-organized criticality |
| `storage` | `src/storage/` | SQLite persistence, WAL, JSON store |
| `mcp_server` | `src/mcp_server.rs` | MCP JSON-RPC 2.0 server |
| `config` | `src/config/` | 7 presets, device auto-detection |

## Dependencies

`ndarray` · `rayon` · `rand` · `rand_chacha` · `serde` · `serde_json` · `rusqlite` · `clap` · `cudarc` (optional, GPU)

## License

Copyright (c) 2024–2026 Brian Schwabauer

This program is free software: you can redistribute it and/or modify it under the terms of the [GNU Affero General Public License v3.0](LICENSE) as published by the Free Software Foundation.
