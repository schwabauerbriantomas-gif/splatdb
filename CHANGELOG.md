# Changelog

## [v2.7.1] - 2026-10-25

### Fixed
- **storage**: atomic tmp+rename writes for metadata JSON, vector shards, energy state and `field/` members.idx/radii.bin — a crash or disk-full mid-write no longer truncates shards or silently discards metadata; a corrupt metadata file is quarantined to `*.corrupt-<timestamp>` with a loud stderr note instead of being dropped
- **storage**: `WriteAheadLog::truncate` rewritten crash-safe — the old error path pointed the file handle at `/dev/null`, silently losing every later `log_operation` (and always failing on Windows); it now builds a fsynced sibling temp and swaps the handle only after every fallible step succeeds
- **cli**: `bench-hnsw` panicked (index out of bounds) with fewer than 100 query samples — percentiles clamp to `n-1` and zero samples exits with an error
- **cli**: `field-search` panicked on query-dim mismatch (ndarray shape panic); clean error naming query/splats/raw dims now (also validated in `search`/`search-file`)
- **cli**: `cluster-reset` removed the wrong file (`cluster_state.bin` vs the real `cluster_state.json`) and panicked on IO errors
- **cli**: `quant-index --bits` validated against the library contract (1..=16; TurboQuant 2..=16) instead of panicking; `doc-add` warns instead of silently discarding invalid metadata JSON
- **cli**: `verbatim store` never inserted the embedded vector — searches attributed neighbor vectors to documents whose metadata pointed at indices no vector occupied, defeating the anti-hallucination purpose; vectors are now really inserted and persisted with their real index, and `decompress` no longer panics on non-ASCII hex (invalid hex is an error, not silently dropped digits)
- **cli**: `graph-*` commands were in-memory only — `graph-add-relation` always failed with `NodeNotFound`, `graph-stats` always printed zeros, and `graph-search` injected a fake demo document; `GaussianGraphStore` now persists to `graph_state.json` (atomic write, duplicate-id/dangling-edge rejection on load) and every subcommand loads and saves; the fake injection is gone
- **ebm**: `soc-avalanche` panicked on NaN energies (`partial_cmp().unwrap()`); Ordering fallback now
- **text_compression**: `semantic_compress` panicked or silently corrupted output on Unicode whose lowercase changes byte length (`İ` U+0130 → 3 bytes); single-string char-wise matching now, with regression tests
- **hnsw**: `HNSWIndex::new` enforces the same `m` range as `load` ([2, 4096]) — an index saved by `new` can always be reloaded
- **cli**: `.bin` loaders cap header-driven allocations (1e9 elements) and validate real file size before allocating — a hostile header can no longer OOM the ingest/search/bench paths

### Removed
- Dead code (zero callers verified by search): `src/energy.rs` (239 LOC), `src/ebm/exploration.rs` (447 LOC)

### Docs
- Consistency pass: measured test counts, every benchmark table cites its persisted artifact in `bench-data/`, BEIR v2 labeled NOT VALIDATED, broken anchors fixed, TOC completed, MCP tool tables completed (18/18), stale PDFs with superseded claims removed

## [v2.7.0] - 2026-10-25

### Added
- Process-level VRAM dataset cache (`gpu::cached_dataset_index`): the dataset uploads to GPU once per dataset version instead of per query/batch; `SplatStore::data_version()` (global epoch-seeded) keys the single-entry cache; alpha/kappa ride along; sub-threshold datasets (≤10K floats) bypass it
- `HNSWIndex::len()`/`dim()`/`is_empty()` accessors; `SyncAction::seq` monotonic identity
- Regression test `test_stale_persisted_hnsw_rejected_and_rebuilt`: a stale persisted HNSW (fewer vectors than the store) is rejected on load and rebuilt — vectors added after the snapshot stay findable via the fused path (covers the `384e205` fix)

### Fixed
- `semantic_memory`: hybrid search fabricated `vec_{n}` ids that never matched real doc ids — fusion never joined vector+BM25 results (CRITICAL); honest capacity/dim errors on store; `with_encoder()` documented + warns (drops stored vectors); `stats()` counted deletes twice
- `splats`: panics when the HNSW backend returns <k candidates; stale or dim-mismatched persisted HNSW now rejected and rebuilt (was silently accepted → permanent recall loss); `hnsw_indexed_count` no longer advances past failed backend inserts (vectors were lost forever); per-vector fallback stops at first failure (positional shift = silent corruption)
- `hnsw_index`: `m`/`m_max0` from file headers bounded (DoS via ~100-byte hostile index); `m ≥ 2` keeps `level_mult` finite
- HTTP/MCP: embeddings with NaN/inf rejected at every entry point (poisoned the whole index); hex decompress no longer panics the MCP process on odd/multibyte input; oversized-line reader preserves bytes after the newline (stream no longer desyncs); `/field-search` runs its O(n·d) scan on `spawn_blocking` (was freezing the whole runtime); `load_field` validates file size before header-driven allocations
- cluster: rendezvous placement unified between CLI ingest and library (split-brain mapping); `mark_flushed` no longer drops re-queued failed actions; sharding stable under edge reordering
- GPU: no more global panic-hook swap (fallback causes now logged once); extended-kernel PTX cached per process (was JIT-loaded from disk per call, even inside the k-means loop); l2/cosine top-k honors >48KB shared-memory opt-in (silent failures for K≥24); sentinel −1 no longer surfaces as vector 0; CPU fallbacks log once instead of silently degrading
- CI: `cargo-audit` job; `Cargo.lock` versioned; release binaries smoke-tested before publish; dead Python tests (m2m-memory inheritance) removed; pentest setups fail loudly instead of passing empty
- MCP stdio dispatcher runs behind `catch_unwind`: a panic in a tool handler returns JSON-RPC `-32603` instead of killing the server process; GPU clamp test asserts exact result length and sentinel/NaN-free outputs under CUDA
- docs: badge/tests count measured (393), GPU microbenchmark re-measured with persisted artifact (`bench-data/gpu_kernel_microbench_2026-10-25.json`: 6 runs, median 30.6×, GPU 20ms @ 10K×640D×100q), longmemeval NumPy-vs-native results separated (native: 0.304 recall / 671.63 QPS), unsourced HRM2/TurboQuant claims marked NOT VALIDATED, 15 missing CLI commands documented, verbatim/compression status corrected, inert config keys (`semantic_fusion`, `semantic_vector_weight`, `semantic_bm25_*`, `semantic_decay_halflife`, `autoscale_*`) documented as unused in code (doc-comments in `src/config/mod.rs`), `build_single` input preconditions documented in code (doc-comment in `src/encoding.rs`), LOC badge recount (34,461)

### Removed
- Dead code, wave-4 audit (zero callers verified by search before each deletion): `error::SplatsDBError`, `engine::SplatsDBEngine` + `compute_expert_distances*`, geometry helpers (`normalize_sphere_1d`, `geodesic_distance_batch`, `exp_map`, `log_map`, `project_to_tangent`, `knn_geodesic`), `BM25Index::search_ids`, `SplatStore::decay_frequencies`, cluster `protocol`/`edge_node`/`sync` modules + `HealthChecker` (`EdgeNodeInfo`/`LoadMetrics`/`GeoLocation` kept — router/sharding use them), `backend_comm`, `auto_scaling`, `api::coordinator_api` + `api::edge_api` — ~2.8K lines

### Verified
- 393 tests green (317 lib + 76 integration), `clippy -D warnings` clean on cuda + cpu-only, GPU kernel benchmark median 30.6× across 6 runs (range 25.45×–34.32×, GPU median 20ms @ 10K×640D×100q, same-run CPU control)

## [v2.6.1] - 2026-10-25

### Added
- Two-phase IVF over the splat field: `field-search --nprobe N` (CLI) and `/field-search {"mode":"two-phase","nprobe":N}` (HTTP) — phase 1 selects top-N clusters by d(q,μ), phase 2 re-ranks their members exactly; `--gpu` runs phase 1 on the PTX kernel (`f32bc87`)
- `/search` results now carry `topic` + `confidence` per hit (1 − d(x,μ_topic)/R_topic) when a `field/` is persisted; dead-weight splats pruned during ingest (`5cf7e09`)
- `semantic-hub/faststart.sh`: one-command stack — embeddinggemma-2 embedder (:8860) + exact GPU hub (:8871) + end-to-end smoke (`5cf7e09`)

### Fixed
- `serve --gpu`: the `/search` GPU index now searches the full chunk corpus (`field.raw`) instead of the cluster centroids μ, so `topic`/`confidence` always refer to real chunks (`cf38316`)
- CI: `gpu_info()` no longer panics when `cudarc` finds no driver (`5c12ee5`); cudarc-without-driver panic guarded in the rest of the GPU path and clippy 1.99 lints fixed (`96ddbd2`)

### Removed
- `training` preset and its subsystems (data lake, `train_embeddings`, training config fields) as unvalidated dead weight — ~1,250 lines deleted (`5cf7e09`)

## [v2.6.0] - 2026-10-20

### Added
- `field-search` CLI: búsqueda exacta por cotas sobre el campo splat (celdas con radio + desigualdad triangular); radios y membresías CSR persistidos por cada ingest (`field/`, shard `raw`) — recall 0.99 vs GT exacto
- `serve --gpu`: índice exacto en VRAM (kernel PTX `l2_topk_kernel` fused, herencia m2m); `/search` con embedding corre GPU y marca `"backend":"gpu"` — p50 11.4 ms E2E en 26k×768, recall 1.0
- Endpoint `/field-search`: sweep exacto server-side con métricas de poda reales
- Feature `cuda` ON por defecto (build.rs degrada sin nvcc; runtime sin driver cae a CPU)
- `bench-latent/`: GT exacto 30 queries + comparativa de modos — HNSW 0.997, campo exacto 0.99, scoring de energía gaussiana descartado (recall 0.00-0.05, alpha domina = popularidad)

### Fixed
- `ingest-hierarchical` no persistía splats: calculaba, imprimía éxito y descartaba todo
- `simple_kmeans`: asignación por GEMM (init LCG + update f64 intactos) — flat 30-60+ min → 63 s, hierarchical 38,5 → 9,2 s
- Poda por cotas: `kth` solo es cota válida con el top-k completo (recall 0.837 → 0.99)

### Changed
- CI: matriz de features (cpu-only + default-cuda), runtime smoke sin GPU, clippy en ambas configs

## [v2.5.0] - 2026-07-04

### Added
- GitHub Actions CI workflow (test + lint)
- PyPI Trusted Publisher auto-release workflow
- 256 tests passing (CPU mode)

### Fixed
- Various CI/CD improvements
