# SplatDB

A native Rust vector search engine powered by Gaussian Splatting embeddings and hierarchical retrieval.

[![Version](https://img.shields.io/badge/version-2.5.0-blue.svg)](https://github.com/schwabauerbriantomas-gif/splatdb)
[![License](https://img.shields.io/badge/license-AGPL--3.0-green.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-2021-orange.svg)](https://www.rust-lang.org/)
[![Tests](https://img.shields.io/badge/tests-267%20passing-brightgreen.svg)]()
[![LOC](https://img.shields.io/badge/LOC-22K-informational.svg)]()

---

## Table of Contents

- [What Is This?](#what-is-this)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [How It Works](#how-it-works)
- [Benchmarks](#benchmarks)
- [Configuration Presets](#configuration-presets)
- [CLI Reference](#cli-reference)
- [MCP Server (AI Agent Integration)](#mcp-server-ai-agent-integration)
- [HTTP API Server](#http-api-server)
- [Rust API](#rust-api)
- [GPU Acceleration](#gpu-acceleration)
- [Knowledge Graph (GraphSplat)](#knowledge-graph-graphsplat)
- [Vector Compression (TurboQuant / PolarQuant)](#vector-compression-turboquant--polarquant)
- [Semantic Memory](#semantic-memory)
- [Energy-Based Model](#energy-based-model)
- [Module Map](#module-map)
- [Dependencies](#dependencies)
- [License](#license)

---

## What Is This?

SplatDB applies **Gaussian Splatting** — a technique from 3D neural rendering — to vector search. Instead of storing raw embedding vectors, each document is represented as a probabilistic Gaussian (mean μ, opacity α, concentration κ). This enables:

- **Richer similarity semantics** via splat overlap instead of point-to-point distance
- **Natural compression** through distribution parameters (3–8 bit quantization)
- **Uncertainty-aware retrieval** — sparse regions have high energy, guiding active learning
- **Knowledge graph overlay** — typed entities and relations augment vector retrieval

Combined with a two-level KMeans++ retrieval pipeline (HRM2), CUDA GPU acceleration with custom PTX kernels, and hybrid BM25+vector semantic memory, SplatDB provides a full-featured vector search engine in ~22K lines of pure Rust.

**Key use cases:**
- AI agent long-term memory (MCP server for Claude, GPT, open-source LLMs)
- Semantic search over document collections
- Knowledge graph construction and retrieval
- GPU-accelerated nearest neighbor search at scale

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        Interfaces                                 │
│         CLI (clap)  │  MCP (stdio JSON-RPC)  │  HTTP API         │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────────┐    │
│  │ Query Layer   │  │ Semantic     │  │ GraphSplat          │    │
│  │ HRM2 Two-     │  │ Memory       │  │ Knowledge Graph     │    │
│  │ Level         │  │ BM25+Vector  │  │ Hybrid Search +     │    │
│  │ Retrieval     │  │ RRF Fusion   │  │ BFS Traversal       │    │
│  └──────┬───────┘  └──────┬───────┘  └───────┬─────────────┘    │
│         └─────────────────┼──────────────────┘                   │
│                           │                                       │
│  ┌────────────────────────▼────────────────────────────────────┐ │
│  │            Gaussian Splat Embeddings                         │ │
│  │         (mean μ + opacity α + concentration κ)               │ │
│  ├──────────────────────────────────────────────────────────────┤ │
│  │  Compression: TurboQuant / PolarQuant (3–8 bit)              │ │
│  ├──────────────────────────────────────────────────────────────┤ │
│  │  Indexes: HNSW │ LSH │ KMeans++ (coarse → fine)             │ │
│  ├──────────────────────────────────────────────────────────────┤ │
│  │  Persistence: SQLite (WAL)  │  GPU: CUDA PTX (optional)     │ │
│  └─────────────────────────────┴───────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Prerequisites

- **Rust** 1.56+ (Edition 2021)
- **C compiler** (for SQLite bundling via `rusqlite`)
- **CUDA Toolkit 12.x + MSVC `cl.exe`** (optional, for GPU acceleration)

### Build

```bash
# CPU-only (no GPU dependencies)
cargo build --release

# With CUDA GPU acceleration
cargo build --release --features cuda

# Run tests
cargo test --lib
```

### Run

```bash
# Check engine status
./target/release/splatdb status --verbose

# Search with a query vector
./target/release/splatdb search --query "0.1,0.2,0.3,...,-0.5" -k 10

# Start MCP server for AI agent integration
./target/release/splatdb mcp

# Start HTTP API server
./target/release/splatdb serve --host 0.0.0.0 --port 8199

# Ingest vectors from binary file
./target/release/splatdb ingest --input vectors.bin --n-clusters 100
```

### Binary Format

SplatDB uses a simple binary format for vector data:

```
[ u64: number of rows ] [ u64: number of columns ] [ f32 × rows × cols: data ]
```

---

## How It Works

### Gaussian Splat Embeddings

Traditional vector databases store raw embedding vectors and compute point-to-point distances. SplatDB represents each document as a **probabilistic Gaussian**:

- **μ (mean)**: Position in embedding space — the "center" of the concept
- **α (opacity)**: How strongly this document is represented — akin to importance weight
- **κ (concentration)**: How focused the representation is — high κ means precise, low κ means broad

Similarity is computed as **splat overlap** (integral of two Gaussians), which naturally captures:
- Documents that cover broad topics (low κ) match more queries, but with lower confidence
- Documents that are very specific (high κ) match fewer queries, but with higher precision
- Overlap is symmetric and differentiable — enabling energy-based exploration

### HRM2 Two-Level Retrieval

The query pipeline uses Hierarchical Rejection Mapping (HRM2):

1. **Coarse KMeans++**: Partition all splats into N coarse clusters
2. **Fine KMeans++**: Each coarse cluster has M fine sub-clusters
3. **Probe**: Only search the top P fine clusters relevant to the query
4. **Exact re-rank**: Compute full splat overlap for candidates

This reduces search from O(N) to O(N/P) with minimal recall loss.

### SimCos Embeddings

When no pre-computed embeddings are provided, SplatDB uses **SimCos** — a similarity-consistent n-gram hashing scheme:

- Character trigrams are extracted from text
- Each trigram is hashed into the embedding dimension
- Overlapping n-grams between two texts produce high cosine similarity
- No model loading required — instant, deterministic, zero dependencies

For production use, pass pre-computed embeddings from your preferred model (OpenAI, Cohere, local transformers, etc.).

---

## Benchmarks

All numbers are **measured on real hardware**, not estimated.

**Hardware**: AMD Ryzen 5 3400G (4c/8t), 32GB DDR4, NVIDIA RTX 3090 24GB, CUDA 12.4
**Software**: Rust 1.94.1, cudarc 0.19.4, ndarray 0.16, rayon 1.11

### GPU Top-K Search (Custom CUDA Kernels)

Combined distance + top-k selection in one GPU pass. Dataset persists in VRAM between queries.

| Dataset | Dim | Queries | k | CPU QPS | GPU Persistent QPS | Speedup |
|---------|-----|---------|---|---------|--------------------|---------|
| 10K | 640 | 100 | 10 | 269 | 1,667 | **6.2x** |
| 100K | 640 | 100 | 10 | 214 | 1,667 | **7.8x** |

- Upload bandwidth: **3.9 GB/s**
- GPU results verified against CPU: identical top-k indices and order
- Per-query latency with dataset in VRAM: **0.60 ms** (constant, independent of dataset size)

### CUDA Kernel Optimizations

- `float4` vectorized loads (640D = 160 × float4 per vector)
- Shared memory query cache (avoids redundant global reads per thread)
- Thread-local sorted top-k (max K=32) with shared memory merge
- `__launch_bounds__(256)` for optimal sm_86 occupancy
- PTX compiled with `--use_fast_math -O3`

### HRM2 vs Linear Scan (Python Prototype)

An earlier Python prototype showed **32x speedup** with HRM2 vs linear scan on 100K vectors (100K splats, 1K queries, k=64, CPU). These numbers are for the Python implementation and are referenced for historical context only. The validated Rust benchmark result:

| Method | Latency | QPS | Speedup |
|--------|---------|-----|---------|
| Linear scan | 94.79 ms | 10.55 | baseline |
| SplatDB HRM2 | 0.99 ms | 1,012.77 | **32.4x** |

---

## Configuration Presets

Seven presets cover edge devices to GPU clusters. Each preset configures all subsystems at once — no manual flag toggling required.

| Preset | Use Case | Capacity | Quantization | Graph | Semantic | GPU | HNSW | LSH |
|--------|----------|----------|-------------|-------|----------|-----|------|-----|
| `default` | General purpose | 100K | 8-bit TurboQuant | ✅ | ✅ RRF | — | — | — |
| `simple` | Edge / IoT | 10K | — | — | — | — | — | — |
| `mcp` | AI agent memory | 100K | 8-bit TurboQuant | ✅ | ✅ RRF | Auto | — | — |
| `advanced` | AI agents | 1M | 4-bit TurboQuant | ✅ | ✅ RRF | Optional | ✅ M=32 | — |
| `training` | Model research | 500K | — | — | — | Optional | — | — |
| `distributed` | Multi-node | 10M | 4-bit TurboQuant | — | ✅ RRF | Optional | — | — |
| `gpu` | CUDA servers | 5M | 4-bit TurboQuant | ✅ | ✅ | ✅ | ✅ M=48 | — |

```rust
use splatdb::config::SplatDBConfig;

let edge    = SplatDBConfig::simple(None);              // Raspberry Pi
let agent   = SplatDBConfig::mcp(None);                 // AI agent memory (auto GPU)
let heavy   = SplatDBConfig::advanced(Some("cuda"));    // Full-featured with GPU
let server  = SplatDBConfig::gpu(None);                 // Maximum GPU performance
let cluster = SplatDBConfig::distributed(Some("cpu"));  // Multi-node deployment
```

**Device auto-detection**: Pass `None` to auto-detect (CUDA → Vulkan → CPU). Pass `Some("cuda")`, `Some("vulkan")`, or `Some("cpu")` to force a specific device.

---

## CLI Reference

SplatDB provides a comprehensive CLI for all operations. Global options apply to every command.

### Global Options

| Flag | Default | Description |
|------|---------|-------------|
| `--data-dir <DIR>` | `./splatdb_data` | Storage directory |
| `--dim <DIM>` | `64` | Vector dimensionality |
| `--max-splats <N>` | `100000` | Maximum splat capacity |
| `--backend <BACKEND>` | `sqlite` | Metadata storage backend (`sqlite` or `json`) |

### Core Commands

| Command | Description |
|---------|-------------|
| `status` | Show store statistics (active splats, entropy, index status) |
| `search --query "0.1,0.2,..." -k 10` | Search by comma-separated query vector |
| `search-file --input query.bin -k 10` | Search using query vector from binary file |
| `index --input data.bin --shard default` | Add vectors from binary file |
| `save` | Save current state to disk |
| `load` | Load state from disk |
| `list` | List all stored shards |
| `backup --output ./backup/` | Backup all data to directory |

### Ingestion Commands

| Command | Description |
|---------|-------------|
| `ingest --input data.bin --n-clusters 100` | KMeans splat centroids via DatasetTransformer |
| `ingest-hierarchical --input data.bin --n-clusters 10` | Two-level hierarchical KMeans |
| `ingest-leader --input data.bin --target-clusters 50` | O(n) single-pass leader clustering |

### Search Backend Commands

| Command | Description |
|---------|-------------|
| `fused-search --query "0.1,..." -k 10` | Search all enabled backends with score fusion |
| `hnsw-search --query "0.1,..." -k 10` | HNSW index only |
| `lsh-search --query "0.1,..." -k 10` | LSH index only |

### Quantization Commands

| Command | Description |
|---------|-------------|
| `quant-index --algorithm turbo --bits 4` | Build compressed index (TurboQuant or PolarQuant) |
| `quant-search --query "0.1,..." --top-k 10` | Search compressed codes |
| `quant-status` | Show quantization statistics |

### GPU Commands (requires `--features cuda`)

| Command | Description |
|---------|-------------|
| `gpu-info` | Show CUDA device information and VRAM status |
| `bench-gpu --n-vectors 100000 --dim 640` | Benchmark GPU vs CPU search |
| `bench-gpu-ingest --n-vectors 100000 --dim 640` | Benchmark full GPU ingest + search pipeline |

### Energy / Self-Organized Criticality

| Command | Description |
|---------|-------------|
| `soc-check` | Check system criticality state |
| `soc-avalanche --iterations 100` | Trigger avalanche reorganization |
| `soc-relax --iterations 50` | Relax the system toward lower energy |

### Document Commands

| Command | Description |
|---------|-------------|
| `doc-add --id my-doc --text "content"` | Add document with metadata |
| `doc-get --id my-doc` | Retrieve document by ID |
| `doc-del --id my-doc` | Soft-delete a document |

### Knowledge Graph Commands

| Command | Description |
|---------|-------------|
| `graph-add-doc --text "content"` | Add document node to graph |
| `graph-add-entity --name "entity" --entity-type person` | Add entity node |
| `graph-add-relation --source-id 1 --target-id 2 --relation-type MENTIONS` | Add edge |
| `graph-traverse --start-id 1 --max-depth 3` | BFS traversal |
| `graph-search --query "search text" --k 10` | Hybrid graph+vector search |
| `graph-stats` | Graph statistics |

### Server Commands

| Command | Description |
|---------|-------------|
| `serve --host 0.0.0.0 --port 8199` | Start HTTP API server |
| `mcp` | Start MCP server (stdio JSON-RPC 2.0) |
| `preset-info` | Show which subsystems each preset enables |

---

## MCP Server (AI Agent Integration)

SplatDB includes a built-in **Model Context Protocol** server for direct integration with AI agents. The MCP server uses stdio transport with JSON-RPC 2.0 and auto-loads the `mcp` preset with GPU auto-detection.

### Configuration

Add to your AI agent's MCP configuration (e.g., Claude Code, Cursor, Hermes):

```json
{
  "mcp_servers": {
    "splatdb": {
      "command": "/path/to/splatdb",
      "args": ["mcp"],
      "timeout": 30
    }
  }
}
```

### Available Tools

#### Vector Store

| Tool | Required Params | Optional Params | Description |
|------|----------------|-----------------|-------------|
| `splatdb_store` | `text` | `id`, `category`, `embedding` | Store a memory. Auto-embeds text via SimCos, or accepts pre-computed embedding. |
| `splatdb_search` | `query` | `top_k`, `embedding` | Semantic search. Returns ranked results with similarity scores. |
| `splatdb_status` | — | — | Engine status: dimension, doc count, active indexes, quantization state. |

#### Document Store (SQLite-Backed)

| Tool | Required Params | Optional Params | Description |
|------|----------------|-----------------|-------------|
| `splatdb_doc_add` | `id`, `text` | `metadata` (JSON) | Store document with metadata. Persists to SQLite. Survives restarts. |
| `splatdb_doc_get` | `id` | — | Retrieve document by ID. Returns text, metadata, timestamps. |
| `splatdb_doc_del` | `id` | — | Soft-delete a document. |

#### Knowledge Graph (GraphSplat)

| Tool | Required Params | Optional Params | Description |
|------|----------------|-----------------|-------------|
| `splatdb_graph_add_doc` | `text` | — | Add document node. Auto-embeds text. |
| `splatdb_graph_add_entity` | `name`, `entity_type` | — | Add entity node (person, org, location, concept). Auto-embeds name. |
| `splatdb_graph_add_relation` | `source_id`, `target_id`, `relation_type` | `weight` | Add directed edge between nodes. |
| `splatdb_graph_traverse` | `start_id` | `max_depth` | BFS traversal from a node. Returns connected subgraph. |
| `splatdb_graph_search` | `query` | `k` | Hybrid search: vector similarity + graph context boost. |
| `splatdb_graph_search_entities` | `query` | `k` | Search entity nodes by embedding similarity. |
| `splatdb_graph_stats` | — | — | Node counts, edge counts, entity/document breakdown. |

### Example: AI Agent Session

```
Agent: splatdb_store(text="User prefers dark mode and Spanish language")
→ {"id": "mem_1", "status": "stored"}

Agent: splatdb_graph_add_entity(name="Brian", entity_type="person")
→ {"entity_type": "person", "name": "Brian", "node_id": 1}

Agent: splatdb_graph_add_doc(text="Brian works on SplatDB, a vector search engine")
→ {"node_id": 2, "node_type": "document"}

Agent: splatdb_graph_add_relation(source_id=2, target_id=1, relation_type="MENTIONS")
→ {"ok": true, "relation_type": "MENTIONS"}

Agent: splatdb_search(query="user preferences", top_k=5)
→ {"results": [{"id": "mem_1", "score": 0.95, "text": "User prefers dark mode..."}]}

Agent: splatdb_graph_search(query="who works on vector search", k=3)
→ {"results": [{"content": "Brian works on SplatDB...", "score": 0.49}]}
```

### Persistence

The MCP server uses SQLite with WAL mode for document storage. On restart, all documents are automatically reloaded (warm start). Vector indices are rebuilt from stored embeddings — no data loss between sessions.

---

## HTTP API Server

Start an HTTP server for integration with Python, Node.js, or any HTTP client:

```bash
./target/release/splatdb serve --host 0.0.0.0 --port 8199 --data-dir ./data
```

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `127.0.0.1` | Bind address |
| `--port` | `8199` | Listen port |
| `--data-dir` | `./splatdb_data` | Storage directory |
| `--dim` | `64` | Vector dimensionality |
| `--max-splats` | `100000` | Max capacity |
| `--backend` | `sqlite` | Metadata backend |

---

## Rust API

```rust
use splatdb::{SplatDBConfig, SplatStore};

fn main() {
    // Choose a preset (or use SplatDBConfig::default())
    let config = SplatDBConfig::mcp(None);  // AI agent memory, auto GPU
    let mut store = SplatStore::new(config);

    // Insert vectors
    let id = store.insert(&vec![0.1f32; 640]);

    // Search
    let results = store.search(&vec![0.1f32; 640], 10);
    for r in &results {
        println!("id={} dist={:.4}", r.index, r.distance);
    }

    // Build index for fast retrieval
    store.build_index();

    // Fast approximate search via HRM2
    let fast_results = store.find_neighbors_fast(&query.view(), k);
}
```

### Key Methods

| Method | Description |
|--------|-------------|
| `SplatStore::new(config)` | Create store with given configuration |
| `store.insert(&vec)` | Insert a vector, return its index |
| `store.search(&vec, k)` | Exact k-nearest neighbor search |
| `store.find_neighbors(&row, k)` | Search using ndarray row reference |
| `store.find_neighbors_fast(&view, k)` | HRM2 approximate search (requires `build_index()`) |
| `store.build_index()` | Build HRM2 coarse + fine index |
| `store.n_active()` | Number of active splats |
| `store.entropy()` | Current entropy of the system |

---

## GPU Acceleration

SplatDB supports CUDA GPU acceleration via custom PTX kernels compiled at build time.

### Requirements

- NVIDIA GPU (Compute Capability 7.0+, tested on RTX 3090 sm_86)
- CUDA Toolkit 12.x
- MSVC `cl.exe` in PATH (Windows) or GCC (Linux)

### Build

```bash
cargo build --release --features cuda
```

### How It Works

1. **Upload**: Dataset vectors are copied to GPU VRAM as a flat `float` array
2. **Persistent**: Data stays in VRAM between queries — no re-upload overhead
3. **Query**: Each query launches a CUDA kernel that:
   - Loads the query into shared memory
   - Each thread block processes a chunk of vectors using `float4` loads
   - Thread-local top-k maintained in registers
   - Final merge across thread blocks
4. **Download**: Only the top-k indices and distances are copied back to CPU

### Performance

With 100K vectors (640D) on an RTX 3090:
- **Upload**: 3.9 GB/s sustained bandwidth
- **Query latency**: 0.60 ms per query (constant, independent of dataset size up to VRAM)
- **Throughput**: 1,667 QPS (7.8x faster than CPU)

### GPU Commands

```bash
# Show CUDA device info
./target/release/splatdb gpu-info

# Benchmark GPU vs CPU
./target/release/splatdb bench-gpu --n-vectors 100000 --dim 640 --n-queries 100 --top-k 10

# Full pipeline benchmark (ingest + index + search + GPU)
./target/release/splatdb bench-gpu-ingest --n-vectors 100000 --dim 640 --n-queries 100
```

---

## Knowledge Graph (GraphSplat)

SplatDB includes a built-in knowledge graph that augments vector search with structured relationships.

### Node Types

- **Document nodes**: Auto-embedded from text content
- **Entity nodes**: Named entities with types (person, organization, location, concept)

### Edge Types

Any string label works. Common patterns:
- `MENTIONS` — document references an entity
- `RELATED_TO` — general semantic connection
- `PART_OF` — hierarchical containment
- `LOCATED_IN` — spatial relationship

### Search

**Hybrid search** combines vector similarity with graph context:
1. Find top-k similar nodes by embedding cosine similarity
2. Boost scores for nodes connected to highly-ranked neighbors
3. Return fused ranking

This means: a document that is *similar* to your query AND is *connected* to other relevant documents will rank higher than a similar but isolated document.

### API

```rust
use splatdb::graph_splat::GaussianGraphStore;

let mut graph = GaussianGraphStore::new();

// Add nodes
let doc_id = graph.add_document("SplatDB is a vector search engine", &embedding)?;
let entity_id = graph.add_entity("SplatDB", "software", &name_embedding)?;

// Add relation
graph.add_relation(doc_id, entity_id, "MENTIONS", 1.0)?;

// Hybrid search
let results = graph.hybrid_search(&query_embedding, 10);

// BFS traversal
let connected = graph.traverse(entity_id, 3);

// Entity search
let entities = graph.search_entities(&query_embedding, 5);
```

---

## Vector Compression (TurboQuant / PolarQuant)

SplatDB supports data-oblivious vector compression that requires **no training data, no codebooks, no fine-tuning**.

### Algorithms

| Algorithm | Bits | Method | Best For |
|-----------|------|--------|----------|
| **TurboQuant** | 3–8 | Random projection + quantization | General purpose, high throughput |
| **PolarQuant** | 3–8 | Polar coordinate quantization | Directional similarity |

### Usage

```bash
# Build compressed index
./target/release/splatdb quant-index --algorithm turbo --bits 4

# Search compressed codes
./target/release/splatdb quant-search --query "0.1,..." --top-k 10

# Check status
./target/release/splatdb quant-status
```

### Compression Ratio

With 4-bit TurboQuant on 640D vectors:
- **Raw**: 2,560 bytes per vector
- **Compressed**: 320 bytes per vector (8x reduction)
- **Recall**: >95% top-10 recall at 5% search fraction

---

## Semantic Memory

SplatDB combines **BM25 text search** with **vector similarity** using Reciprocal Rank Fusion (RRF).

### How RRF Works

1. Run BM25 search over document text → get ranking R₁
2. Run vector similarity search → get ranking R₂
3. Fuse: `score(d) = w_vec × 1/(k + rank_R₂(d)) + w_bm25 × 1/(k + rank_R₁(d))`

Default weights: `w_vec = 0.6`, `w_bm25 = 0.4` (configurable per preset).

This gives the best of both worlds: vector search captures semantic meaning, BM25 captures exact keyword matches, and RRF merges them without score normalization issues.

### Temporal Decay

Optional: recent memories can be weighted higher via exponential decay with configurable half-life.

---

## Energy-Based Model

SplatDB models the embedding space as an energy landscape:

```
E(x) = −log(Σᵢ αᵢ · exp(−κᵢ · ‖x − μᵢ‖²))
```

- **Low energy**: Well-covered regions — many documents nearby, high confidence
- **High energy**: Sparse regions — few documents, high uncertainty

### Applications

- **Active learning**: Focus labeling effort on high-energy regions
- **Exploration**: Boltzmann sampling from the energy landscape
- **Self-organized criticality (SOC)**: The system can self-tune via avalanche reorganization and relaxation

### CLI

```bash
# Check current energy state
./target/release/splatdb soc-check

# Trigger avalanche (reorganize splats for better coverage)
./target/release/splatdb soc-avalanche --iterations 100

# Relax toward lower energy
./target/release/splatdb soc-relax --iterations 50
```

---

## Module Map

22,021 lines of Rust across these modules:

| Module | Source | Description |
|--------|--------|-------------|
| `splats` | `src/splats.rs` | Core API — insert, search, upsert, delete, statistics |
| `hrm2_engine` | `src/hrm2_engine.rs` | Two-level hierarchical retrieval (coarse → fine) |
| `engine` | `src/engine.rs` | CPU L2 distance with rayon parallelism |
| `gpu` | `src/gpu/` | CUDA PTX kernels, `GpuIndex` with persistent VRAM |
| `quantization` | `src/quantization.rs` | TurboQuant / PolarQuant data-oblivious compression |
| `clustering` | `src/clustering.rs` | KMeans++ with ChaCha8 RNG |
| `graph_splat` | `src/graph_splat.rs` | Knowledge graph overlay with hybrid search |
| `semantic_memory` | `src/semantic_memory.rs` | BM25 + vector RRF fusion with temporal decay |
| `hnsw_index` | `src/hnsw_index.rs` | HNSW approximate nearest neighbor |
| `lsh_index` | `src/lsh_index.rs` | Locality-sensitive hashing |
| `energy` | `src/energy.rs` | Energy landscape computation |
| `ebm` | `src/ebm/` | Boltzmann exploration, self-organized criticality |
| `storage` | `src/storage/` | SQLite persistence (WAL), JSON store |
| `mcp_server` | `src/mcp_server.rs` | MCP JSON-RPC 2.0 server (13 tools) |
| `config` | `src/config/` | 7 presets, device auto-detection |
| `cli` | `src/cli/` | Command-line interface (clap) |

---

## Dependencies

| Crate | Purpose | Optional |
|-------|---------|----------|
| `ndarray` | N-dimensional array operations | No |
| `rayon` | Data parallelism (CPU search) | No |
| `rand` / `rand_chacha` / `rand_distr` | Random number generation, KMeans++ | No |
| `serde` / `serde_json` | Serialization | No |
| `rusqlite` | SQLite persistence (bundled) | No |
| `clap` | CLI argument parsing | No |
| `regex` | Pattern matching | No |
| `cudarc` | CUDA GPU kernels | Yes (`--features cuda`) |

---

## License

Copyright (c) 2024–2026 Brian Schwabauer

This program is free software: you can redistribute it and/or modify it under the terms of the [GNU Affero General Public License v3.0](LICENSE) as published by the Free Software Foundation.
