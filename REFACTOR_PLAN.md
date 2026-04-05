# SplatDB Security Refactoring Plan — 98 Findings

## Group 1: Replace unsafe transmutes with bytemuck (14 instances, 6 files)
- [ ] Add `bytemuck` to Cargo.toml
- [ ] storage/persistence.rs: lines 110, 130, 202, 208, 213 (5 unsafe blocks)
- [ ] cli/helpers.rs: lines 75, 95 (2 unsafe blocks)
- [ ] loaders/optimized_loader.rs: lines 21, 37 (2 unsafe blocks)
- [ ] gpu/cuda_kernel.rs: replace `as i32` casts with checked conversions (lines 138-139, 204-206, 273-274, 368-370)
- [ ] dataset_transformer.rs: lines 251, 326, 410 — checked arithmetic

## Group 2: Checked arithmetic — OOM protection (6 files)
- [ ] loaders/optimized_loader.rs: rows*cols checked_mul
- [ ] cli/helpers.rs: rows*cols checked_mul (2 functions)
- [ ] storage/persistence.rs: rows*cols checked_mul + u64->usize try_into
- [ ] gpu/cuda_kernel.rs: n_queries*n checked_mul, n_queries*k checked_mul
- [ ] splats.rs: next_id checked_add
- [ ] storage/wal.rs: serialized.len() check before u32 cast

## Group 3: GPU input validation (cuda_kernel.rs)
- [ ] upload_dataset: assert dataset.len() == n_vectors * dim
- [ ] l2_distances: validate query.len() == dim
- [ ] batch_l2_distances: validate queries.len() == n_queries * dim
- [ ] topk_search: validate queries.len() == n_queries * dim
- [ ] All kernel launches: assert n <= i32::MAX, dim <= i32::MAX
- [ ] shared_mem_bytes: validate <= 48KB GPU limit
- [ ] Assert contiguity for Array2 in optimized_loader

## Group 4: Capacity limits — unbounded growth (7 files)
- [ ] hnsw_index.rs: max_elements cap, removed set compaction, saturating_sub in n_items
- [ ] semantic_memory.rs: max_memories limit, check add_splat return value
- [ ] bm25_index.rs: max_document_length limit
- [ ] quantization.rs: max_vectors cap
- [ ] hrm2_engine.rs: capacity limit
- [ ] lsh_index.rs: upper bound validation on n_tables, n_bits, dim
- [ ] storage/wal.rs: max total entries/bytes limit

## Group 5: Path traversal fixes (persistence.rs)
- [ ] save_vectors: validate shard_name (no /, \, ..)
- [ ] backup: replace contains("..") with canonicalization + prefix check
- [ ] validate_path: use Component::ParentDir rejection
- [ ] copy_dir_recursive: check for symlinks before recursing

## Group 6: API Server hardening (api_server.rs)
- [ ] Add authentication middleware (bearer token via env var)
- [ ] Replace CorsLayer::permissive() with configurable origins
- [ ] Add rate limiting (tower-governor or manual)
- [ ] Validate StoreRequest: embedding dims, text length, id format
- [ ] Validate SearchRequest: top_k max 10000
- [ ] Generic error messages (no internal details)
- [ ] Fix TOCTOU: acquire lock once in store_memory
- [ ] Fix next_id atomicity: increment inside store lock
- [ ] Remove version from /health endpoint
- [ ] Add persistence config to run_server

## Group 7: MCP Server hardening (mcp_server.rs)
- [ ] Validate text length, embedding dims, id format on store
- [ ] Cap top_k in search
- [ ] Validate metadata size before JSON parse
- [ ] Add serde recursion limit for JSON deserialization
- [ ] Move embedding computation outside lock
- [ ] Make SQLite path configurable
- [ ] Replace .expect() with error recovery
- [ ] Reset EMBED_SERVICE_AVAILABLE on retry

## Group 8: Concurrency fixes (4 files)
- [ ] storage/wal.rs: replace .expect() with lock recovery or parking_lot::Mutex
- [ ] storage/json_store.rs: replace .expect() with proper error handling
- [ ] api_server.rs: single lock acquisition in store_memory
- [ ] mcp_server.rs: compute embeddings outside mutex

## Group 9: Cluster security (cluster/*.rs, api/coordinator_api.rs)
- [ ] Add shared cluster secret to ClusterMessage
- [ ] Authenticate coordinator API endpoints
- [ ] Validate heartbeat metric ranges
- [ ] Add HMAC/signature to cluster protocol messages
- [ ] Document LoadBalancer thread safety requirements
- [ ] Add cache eviction to energy_router.rs

## Group 10: Misc fixes
- [ ] config/mod.rs: use full path for nvidia-smi
- [ ] config/mod.rs: randomize quant_seed from OsRng
- [ ] hnsw_index.rs: guard random_level against r=0.0
- [ ] data_lake.rs: validate deserialized entries
- [ ] storage/sqlite_store.rs: propagate deserialization errors
- [ ] storage/json_store.rs: file size check before reading
- [ ] engine.rs: cap results in compute_expert_distances
- [ ] dataset_transformer.rs: MAX_HIERARCHY_DEPTH guard
- [ ] mapreduce_indexer.rs: add actual parallelism with rayon
- [ ] storage/persistence.rs: proper error on non-contiguous array
