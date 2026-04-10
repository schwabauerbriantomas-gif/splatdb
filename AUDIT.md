# SplatsDB v2.1.0 — Complete Codebase Audit

**Date**: 2026-03-28
**Auditor**: Alfred
**Project**: splatsdb (Rust native implementation)

---

## Folder Structure

```
splatsdb/
├── Cargo.toml
├── README.md
├── AUDIT.md
└── src/
    ├── lib.rs                      (47)   — Crate root, module declarations
    ├── main.rs                     (678)  — CLI entry point
    │
    ├── config.rs                   (284)  — Global configuration
    ├── splat_types.rs              (398)  — GaussianSplat, Hrm2Node, MemoryPartition
    │
    ├── ── CORE ENGINE ──
    ├── splats.rs                   (339)  — SplatStore: high-level vector operations
    ├── hrm2_engine.rs              (345)  — HRM2: hierarchical retrieval engine
    ├── engine.rs                   (219)  — CPU-optimized expert distance computation
    ├── energy.rs                   (226)  — Energy-based model functions
    ├── geometry.rs                 (200)  — Manifold geometry operations
    ├── clustering.rs               (238)  — KMeans++ with Lloyd's algorithm
    │
    ├── ── SEARCH INDEXES ──
    ├── bm25_index.rs               (272)  — BM25 text search (Okapi BM25)
    ├── hnsw_index.rs               (315)  — HNSW approximate nearest neighbor
    ├── lsh_index.rs                (239)  — Locality-sensitive hashing
    ├── quantization.rs             (288)  — TurboQuant/PolarQuant compression
    │
    ├── ── GRAPH & MEMORY ──
    ├── graph_splat.rs              (307)  — Knowledge graph over Gaussian splats
    ├── semantic_memory.rs          (482)  — Hybrid BM25+vector semantic memory
    ├── memory.rs                   (266)  — Memory management utilities
    ├── entity_extractor.rs         (304)  — Named entity recognition + n-grams
    │
    ├── ── EMBEDDINGS & TRAINING ──
    ├── embedding_model.rs          (316)  — Matryoshka truncation, distillation loss
    ├── embedding_config.rs         (104)  — Embedding configuration types
    ├── encoding.rs                 (545)  — Full/embedding builders, encoding pipeline
    ├── train_embeddings.rs         (348)  — Training loop, LR scheduling, SyntheticDataset
    ├── evaluate_embeddings.rs      (360)  — Benchmark texts, latency, teacher/student
    ├── dataset_transformer.rs      (578)  — Data → GaussianSplat conversion, KMeans
    │
    ├── ── GPU MODULES (CPU fallback) ──
    ├── cuda_search.rs              (334)  — CUDA-accelerated search (CPU fallback)
    ├── gpu_vector_index.rs         (302)  — GPU vector index (CPU fallback)
    ├── gpu_hierarchical_search.rs  (301)  — GPU hierarchical search (CPU fallback)
    ├── gpu_auto_tune.rs            (246)  — Auto-tuning GPU parameters
    │
    ├── ── DISTRIBUTED CLUSTER ──
    ├── cluster/
    │   ├── mod.rs                  (14)   — Cluster module root
    │   ├── health.rs               (106)  — Health monitoring & heartbeats
    │   ├── balancer.rs             (93)   — Load balancing strategies
    │   ├── energy_router.rs        (311)  — Energy-based query routing
    │   ├── router.rs               (211)  — Cluster routing logic
    │   ├── aggregator.rs           (119)  — Result aggregation from nodes
    │   ├── client.rs               (105)  — Cluster client connector
    │   ├── sync.rs                 (125)  — Node synchronization
    │   ├── sharding.rs             (127)  — Data sharding (FNV-1a hash, haversine)
    │   ├── edge_node.rs            (88)   — Edge node management
    │   └── protocol.rs             (35)   — Wire protocol types
    │
    ├── ── INFRASTRUCTURE ──
    ├── auto_scaling.rs             (339)  — Auto-scaling with trend detection
    ├── backend_comm.rs             (419)  — Priority message bus with DLQ
    ├── mapreduce_indexer.rs        (267)  — MapReduce KMeans indexing
    ├── data_lake.rs                (120)  — Data lake storage abstraction
    ├── quality_reflector.rs        (383)  — Quality-based search reflection
    ├── query_optimizer.rs          (475)  — Query plan optimization
    ├── query_router.rs             (290)  — Multi-strategy query routing
    ├── search_supervisor.rs        (295)  — Multi-backend search supervision
    ├── optimized_api.rs            (203)  — Optimized API layer
    ├── interfaces.rs               (172)  — Shared trait definitions
    │
    ├── ── API LAYER ──
    ├── api/
    │   ├── mod.rs                  (6)    — API module root
    │   ├── edge_api.rs             (424)  — Edge REST API (CRUD + search)
    │   └── coordinator_api.rs      (358)  — Coordinator REST API (node mgmt)
    │
    ├── ── ENERGY-BASED MODEL ──
    ├── ebm/
    │   ├── mod.rs                  (7)    — EBM module root
    │   ├── energy_api.rs           (315)  — Energy API endpoints
    │   ├── exploration.rs          (351)  — Boltzmann sampling, region detection
    │   └── soc.rs                  (308)  — Self-organizing criticality
    │
    ├── ── STORAGE ──
    ├── storage/
    │   ├── mod.rs                  (8)    — Storage module root
    │   ├── json_store.rs           (104)  — JSON file persistence
    │   ├── metadata_store.rs       (70)   — Metadata key-value store
    │   ├── persistence.rs          (256)  — WAL-based persistence layer
    │   ├── sqlite_store.rs         (133)  — SQLite document storage
    │   └── wal.rs                  (186)  — Write-ahead log implementation
    │
    └── ── LOADERS ──
        ├── loaders/
        │   ├── mod.rs              (3)    — Loaders module root
        │   └── optimized_loader.rs (52)   — Optimized data loader
```

---

## Summary Statistics

| Metric | Value |
|--------|-------|
| Source files | 62 |
| Total lines | 14,978 |
| Directories | 6 (api, cluster, ebm, storage, loaders) |
| Test count | 184 |
| Dependencies | 9 (7 runtime + 2 dev/path) |
| License | AGPL-3.0 |

### Lines by Category

| Category | Lines | Files |
|----------|-------|-------|
| Core Engine | 1,367 | 6 |
| Search Indexes | 1,114 | 4 |
| Graph & Memory | 1,359 | 4 |
| Embeddings & Training | 2,251 | 7 |
| GPU Modules | 1,183 | 4 |
| Cluster | 1,234 | 10 |
| Infrastructure | 2,641 | 9 |
| API Layer | 788 | 3 |
| EBM | 981 | 4 |
| Storage | 757 | 6 |
| Other (config, types, lib, main, loaders) | 1,503 | 5 |

---

## Module Details

### Core Engine

**splats.rs** (339 lines)
- `SplatStore`: Main API wrapping HRM2Engine
- `add_splat()`: Add Gaussian splat with mu/alpha/kappa
- `build_index()`: Trigger HRM2 hierarchical index build
- `find_neighbors()`: k-NN search with LOD levels
- `find_neighbors_batch()`: Batch query support
- `entropy()`: Shannon entropy of splat distribution
- `compact()`: Remove low-frequency splats
- `get_statistics()`: Query stats (n_splats, mean_alpha, etc.)

**hrm2_engine.rs** (345 lines)
- `HRM2Engine`: Two-level hierarchical index (coarse + fine KMeans)
- `index()`: Build hierarchy with optional precomputed embeddings
- `query()`: Hierarchical search at LOD 1 (coarse only), 2 (fine), 3 (exact)
- `batch_query()`: Multiple queries
- `get_statistics()`: Latency tracking, index metrics
- Uses `KMeans++` for both levels

**engine.rs** (219 lines)
- `expert_distances()`: Batch L2 distance computation via ndarray
- `expert_distances_cosine()`: Cosine distance variant
- `expert_distances_dot()`: Dot product variant
- Parallel via Rayon `par_bridge()`

**energy.rs** (226 lines)
- Energy function computation for EBM
- `compute_energy()`: Negative log-density of Gaussian mixture
- Geometric penalty, compactness terms

**geometry.rs** (200 lines)
- Manifold geometry: tangent spaces, projections
- `euclidean_distance()`, `cosine_similarity()`

**clustering.rs** (238 lines)
- `KMeans` with KMeans++ initialization
- Lloyd's algorithm convergence
- Parallel via Rayon `into_par_iter()`

### Search Indexes

**bm25_index.rs** (272 lines)
- `BM25Index`: Okapi BM25 text ranking
- Term frequency, IDF, document length normalization
- `add()`, `search()`, `save()`, `load()`

**hnsw_index.rs** (315 lines)
- `HNSWIndex`: Hierarchical navigable small world graph
- Multi-layer graph construction, beam search
- Configurable M, ef_construction, ef_search

**lsh_index.rs** (239 lines)
- `LSHIndex`: Locality-sensitive hashing
- Random hyperplane projections
- Bucket-based candidate generation

**quantization.rs** (288 lines)
- `QuantizedStore`: Vector compression via TurboQuant
- `TurboQuantizer`: Polar/Turbo codes (3-8 bits)
- `PolarQuantizer`: Angle + magnitude quantization
- Parallel encode/search with Rayon
- Inner product estimation in compressed space

**src/quant/** (5 files, ~650 lines) — Integrated turbo-quant (MIT)
- `mod.rs`: Public re-exports (`TurboQuantizer`, `PolarQuantizer`, `PolarCode`, `TurboCode`)
- `turbo.rs`: TurboQuant — two-stage PolarQuant + QJL residual compression
- `polar.rs`: PolarQuant — polar coordinate encoding (angle + magnitude bins)
- `qjl.rs`: Quantized Johnson-Lindenstrauss — random projection + unbiased inner product estimation
- `rotation.rs`: Modified Gram-Schmidt orthogonal rotation (replaces nalgebra QR)
- `error.rs`: Error types (no thiserror dependency)
- Originally from github.com/RecursiveIntell/turbo-quant (MIT license)

**config.rs** (467 lines) — Fully wired subsystem configuration
- `SplatsDBConfig`: 60+ fields covering all subsystems
- Enums: `QuantAlgorithm`, `SearchBackend`, `FusionMethod`, `Dtype`
- `default()`: Balanced — TurboQuant 8-bit, GraphSplat, Semantic Memory ON
- `simple(device)`: Edge — all heavy features OFF, 10K splats, RAM-only
- `advanced(device)`: Agentic — TurboQuant 4-bit, HNSW secondary, auto-scaling 2-20 nodes, 1M splats
- `finalize()`: Device auto-detection (CUDA > Vulkan > CPU)
- `training(device)`: Embedding research — noise augmentation, Matryoshka dims, distillation, data lake
- `distributed(device)`: Multi-node cluster — auto-scaling 3-50 nodes, MapReduce 32 chunks, 10M splats
- `gpu(device)`: CUDA-accelerated — batch 4096, HNSW M=48, 5M splats, auto-tune

### Graph & Memory

**graph_splat.rs** (307 lines)
- `GaussianGraphStore`: Knowledge graph over splats
- `NodeType`: Document, Entity, Concept
- `add_document()`, `add_entity()`: Node creation with dedup
- `add_relation()`: Directed edges with weight
- `search_entities()`: Embedding similarity search
- `traverse()`: BFS graph traversal
- `hybrid_search()`: Vector + graph context boost

**semantic_memory.rs** (482 lines)
- `SemanticMemory`: Hybrid BM25 + vector search
- `FusionMethod`: RRFRank, Weighted, BM25Only, VectorOnly
- `store()`: Add memory with metadata + categories
- `search()`: Hybrid scoring with configurable fusion
- `batch_store()`: Batch insertion
- `apply_temporal_decay()`: Exponential decay by half-life
- `search_with_decay()`: Time-weighted search
- Temporal decay formula: `score * exp(-elapsed / half_life)`

**memory.rs** (266 lines)
- Memory persistence, category management
- Bridge between semantic_memory and storage backends

**entity_extractor.rs** (304 lines)
- `EntityExtractor`: Named entity recognition
- Structural patterns (CAPS, CamelCase, URLs, emails)
- N-gram analysis with capitalized sequences
- `validate_semantic()`: Cosine matching with splat centers

### Embeddings & Training

**embedding_model.rs** (316 lines)
- `ProjectionHead`: Linear projection + L2 normalization
- Matryoshka truncation at multiple dimensions
- `DistillationLoss`: MSE + cosine + matryoshka combined
- `EncoderRegistry`: Multi-model encoder management

**train_embeddings.rs** (348 lines)
- `TrainingConfig`: Epochs, batch size, LR schedule
- `LRScheduler`: Linear warmup + cosine decay
- `SyntheticDataset`: Programmable training data
- `evaluate_embeddings()`: Recall@k metric computation

**evaluate_embeddings.rs** (360 lines)
- Benchmark text generation
- Latency measurement per encoding
- Teacher vs student comparison
- `evaluate()`: Comprehensive evaluation pipeline

**encoding.rs** (545 lines)
- `FullEmbeddingBuilder`: Dense + sparse + colbert embeddings
- `EmbeddingBuilder`: Simplified single-type builder
- Multi-vector encoding pipeline
- Quantization-aware encoding

**dataset_transformer.rs** (578 lines)
- `to_splats()`: Data → GaussianSplat with flat KMeans
- `to_splats_hierarchical()`: Two-level KMeans conversion
- Small cluster merging
- Memory tier partitioning
- `TransformStats`: Conversion metrics

### GPU Modules (CPU Fallback)

All GPU modules implement full algorithmic logic with CPU fallback via ndarray.

**cuda_search.rs** (334 lines)
- `CudaSearchEngine`: CUDA batch search interface
- CPU fallback: brute-force L2 with ndarray
- Batch query support

**gpu_vector_index.rs** (302 lines)
- `GpuVectorIndex`: GPU-accelerated vector indexing
- IVF-like partitioning on CPU
- Add/search/merge operations

**gpu_hierarchical_search.rs** (301 lines)
- `GpuHierarchicalSearch`: GPU hierarchical index
- Coarse-to-fine search pipeline
- Configurable n_probe, n_clusters

**gpu_auto_tune.rs** (246 lines)
- `GpuAutoTune`: Auto-tune GPU parameters
- Benchmark-based parameter selection
- Block size, tile size optimization

### Distributed Cluster

**cluster/health.rs** (106 lines)
- `HealthMonitor`: Node health via heartbeat tracking
- `get_status()`: Healthy/Unhealthy/Unknown
- Configurable timeout thresholds

**cluster/balancer.rs** (93 lines)
- `LoadBalancer`: Round-robin, least-loaded, weighted strategies
- Node selection with load tracking

**cluster/energy_router.rs** (311 lines)
- `EnergyRouter`: Energy-based deterministic query routing
- Hash-based node selection (reproducible)
- Energy zone management, FNV-1a hashing

**cluster/router.rs** (211 lines)
- `ClusterRouter`: Multi-strategy routing
- Centroid-based routing, capacity-aware

**cluster/aggregator.rs** (119 lines)
- `ResultAggregator`: Merge results from multiple nodes
- Top-k merging, score normalization

**cluster/client.rs** (105 lines)
- `ClusterClient`: Connect to cluster nodes
- Request/response handling

**cluster/sync.rs** (125 lines)
- `ClusterSync`: Node synchronization protocol
- Merge operations, conflict resolution

**cluster/sharding.rs** (127 lines)
- `ShardingManager`: Data distribution across nodes
- FNV-1a hash sharding
- Haversine distance for geo-sharding

**cluster/edge_node.rs** (88 lines)
- `EdgeNode`: Lightweight edge node representation
- Capacity tracking, region assignment

**cluster/protocol.rs** (35 lines)
- Wire protocol types for inter-node communication

### Infrastructure

**auto_scaling.rs** (339 lines)
- `AutoScaler`: Trend detection + predictive scaling
- Linear regression on metrics history
- Cooldown periods to prevent oscillation

**backend_comm.rs** (419 lines)
- `BackendComm`: Priority message bus
- Dead letter queue (DLQ) for failed messages
- Latency metrics: p50, p95, p99

**mapreduce_indexer.rs** (267 lines)
- `MapReduceIndexer`: Chunk-based KMeans via map/reduce
- Local KMeans on chunks, global centroid merge

**quality_reflector.rs** (383 lines)
- `QualityReflector`: Quality-based search reflection
- Precision/recall estimation
- Adaptive search parameter tuning

**query_optimizer.rs** (475 lines)
- `QueryOptimizer`: Query plan optimization
- Cost estimation, index selection
- Multi-stage plan generation

**query_router.rs** (290 lines)
- `QueryRouter`: Multi-strategy query routing
- Backend selection: GPU, CPU, quantized

**search_supervisor.rs** (295 lines)
- `SearchSupervisor`: Multi-backend search coordination
- Latency budget management
- Auto-fallback between backends

**optimized_api.rs** (203 lines)
- `OptimizedApi`: High-performance API layer
- Batch operation support

**interfaces.rs** (172 lines)
- Shared trait definitions
- Common type aliases

### API Layer

**api/edge_api.rs** (424 lines)
- `EdgeApi`: Full REST CRUD for edge nodes
- Search, health check, bulk operations
- Request routing to edge nodes

**api/coordinator_api.rs** (358 lines)
- `CoordinatorApi`: Node management + routing
- Cluster coordination, node registration
- Statistics aggregation

### Energy-Based Model

**ebm/energy_api.rs** (315 lines)
- Energy landscape API
- Energy computation endpoints

**ebm/exploration.rs** (351 lines)
- `EbmExploration`: Boltzmann sampling, high-energy region detection
- `EnergyRegion`: Region tracking with visit counts
- Knowledge gap finder, greedy clustering
- `EnergyFn` trait for custom energy functions

**ebm/soc.rs** (308 lines)
- Self-organizing criticality implementation
- Avalanche detection, power-law fitting

### Storage

**storage/persistence.rs** (256 lines)
- `PersistenceManager`: WAL-based persistence
- Snapshot + WAL recovery
- `save()`, `load()`, `checkpoint()`

**storage/wal.rs** (186 lines)
- `WriteAheadLog`: Append-only log
- Entry serialization, replay

**storage/sqlite_store.rs** (133 lines)
- `SqliteStore`: SQLite document storage
- CRUD operations via rusqlite

**storage/json_store.rs** (104 lines)
- `JsonStore`: JSON file persistence
- Serialization/deserialization

**storage/metadata_store.rs** (70 lines)
- `MetadataStore`: Key-value metadata
- Type-safe storage

### Loaders

**loaders/optimized_loader.rs** (52 lines)
- `OptimizedLoader`: Efficient data loading
- Batch-oriented, memory-mapped

---

## Dependencies

| Crate | Version | Purpose |
|-------|---------|---------|
| ndarray | 0.16 | Core array/vector operations |
| rayon | 1 | Parallel iterators |
| rand | 0.8 | Random number generation |
| rand_chacha | 0.3 | Deterministic RNG (ChaCha8) |
| regex | 1 | Entity extraction patterns |
| serde | 1 | Serialization framework |
| serde_json | 1 | JSON serialization |
| rusqlite | 0.31 | SQLite storage backend |
| turbo-quant | local | Polar/Turbo code quantization |
| approx | 0.5 (dev) | Floating-point test assertions |

---

## Architecture

```
                    ┌─────────────────┐
                    │   API Layer      │
                    │ edge_api         │
                    │ coordinator_api  │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Search Layer    │
                    │ query_optimizer  │
                    │ query_router     │
                    │ search_supervisor│
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
     ┌────────▼──┐  ┌───────▼──────┐  ┌───▼────────┐
     │ SplatStore │  │ SemanticMem  │  │ GraphSplat │
     │ (main API) │  │ (BM25+Vec)   │  │ (KG)       │
     └────────┬──┘  └──────────────┘  └────────────┘
              │
     ┌────────▼──────────────┐
     │     HRM2Engine        │
     │  (hierarchical index) │
     └────────┬──────────────┘
              │
     ┌────────▼──────────────┐
     │  Indexes + Quantizer  │
     │ KMeans HNSW LSH TurboQ│
     └───────────────────────┘
              │
     ┌────────▼──────────────┐
     │    Storage Layer      │
     │ SQLite WAL JSON Meta  │
     └───────────────────────┘
```

---

*Audit completed 2026-03-28 by Alfred. 62 files, 14,978 lines, 184 tests, all passing.*
