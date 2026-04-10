# SplatsDB Security Audit Report

**Date:** 2026-04-05
**Scope:** Full codebase — 82 source files, all dependencies
**Commit:** 44967ea (CI green)
**Committee:** 5 specialized reviews run in parallel

---

## Executive Summary

| Severity | Count |
|----------|-------|
| Critical | 2     |
| High     | 9     |
| Medium   | 17    |
| Low      | 17    |
| **Total** | **45** |

**No Critical memory safety issues.** No use-after-free, no transmute, no SQL injection, no path traversal. The codebase is fundamentally sound.

The 2 Critical findings are **network security** gaps (no auth + permissive CORS on HTTP API). These are acceptable for a localhost-only development tool but must be addressed before any public deployment.

**cargo audit:** Passing (CI run #4, all jobs green).

---

## Critical Findings (2)

### C1. No Authentication on HTTP API
- **File:** `src/api_server.rs:206-227`
- **Risk:** All 4 endpoints (/health, /status, /store, /search) have zero authentication. Any network client can read/write all data.
- **Fix:** Add API key middleware on non-health routes.

### C2. Permissive CORS
- **File:** `src/api_server.rs:220`
- **Risk:** `CorsLayer::permissive()` allows any origin. Any website can make cross-origin requests.
- **Fix:** Restrict to localhost or specific origins.

---

## High Findings (9)

### H1. Unchecked `rows * cols` overflow in binary loader
- **File:** `src/loaders/optimized_loader.rs:19`
- **Risk:** Crafted binary file can cause OOM or buffer overrun via integer wrap.
- **Fix:** `rows.checked_mul(cols)` + cap at 1B elements.

### H2. Same issue in CLI helpers
- **File:** `src/cli/helpers.rs:120`
- **Fix:** Same as H1.

### H3. No TLS on HTTP server
- **File:** `src/api_server.rs:206-227`
- **Risk:** All data transmitted in plaintext.
- **Fix:** Add rustls support, configurable cert paths.

### H4. Plaintext HTTP to embedding service
- **File:** `src/mcp_server.rs:26`
- **Risk:** Query text sent unencrypted to localhost embedding service.
- **Fix:** Support https:// scheme, make URL configurable.

### H5. Sensitive data in stderr logs
- **Files:** `src/mcp_server.rs:487, 741, 770, 979`
- **Risk:** Query text, entity names, full JSON-RPC bodies logged.
- **Fix:** Log metadata only (lengths, counts), not content.

### H6. TOCTOU + nested lock in api_server store_memory
- **File:** `src/api_server.rs:107-134`
- **Risk:** Store lock released then reacquired; nested lock (store + next_id) creates deadlock potential.
- **Fix:** Single lock scope, or move next_id into store.

### H7. TOCTOU in WAL truncate
- **File:** `src/storage/wal.rs:128-158`
- **Risk:** Entries read before lock acquired; new entries lost during rewrite.
- **Fix:** Hold lock during entire truncate operation.

### H8. Unchecked `rows * cols` in CLI helpers (supply chain angle)
- **File:** `src/cli/helpers.rs:88-91`
- **Risk:** u64-to-usize truncation on 32-bit targets.
- **Fix:** Use `usize::try_from()`.

### H9. usize-to-i32 truncation in CUDA kernel args
- **File:** `src/gpu/cuda_kernel.rs:159,237,313,430`
- **Risk:** Large n_vectors casts to negative i32, kernel misinterprets.
- **Fix:** Bounds check before cast.

---

## Medium Findings (17)

| # | File | Issue |
|---|------|-------|
| M1 | `mcp_server.rs:360` | No input length validation on store text |
| M2 | `mcp_server.rs:686` | No upper bound on graph traverse depth |
| M3 | `api_server.rs:103-143` | No size validation on HTTP store endpoint |
| M4 | `wal.rs:174` | Unbounded WAL recovery into memory |
| M5 | `wal.rs:183` | 100MB per-entry from untrusted file |
| M6 | `persistence.rs:361` | Recursive directory copy without depth limit |
| M7 | `splats.rs:52` | Integer overflow in n_coarse/n_fine computation |
| M8 | `dataset_transformer.rs:310` | Unchecked n*dim*4 multiplication |
| M9 | `memory.rs:76` | Unbounded add_splats bypasses ram_limit |
| M10 | `mcp_server.rs:27` | AtomicBool with Relaxed ordering |
| M11 | `sqlite_store.rs:34-36` | New connection per operation (no pooling) |
| M12 | `wal.rs:167-199` | read_entries bypasses inner mutex |
| M13 | `gpu/cuda_kernel.rs:22-29` | GpuIndex Send/Sync not verified |
| M14 | `optimized_loader.rs:42-43` | ndarray contiguity assumption in unsafe cast |
| M15 | `cli/helpers.rs:102,125` | Unsafe f32-to-u8 cast — use bytemuck instead |
| M16 | `gpu/cuda_kernel.rs:136,295` | shared_mem_bytes can exceed GPU hardware limits |
| M17 | `cli/mod.rs:206-209` | Bind to 0.0.0.0 possible with no warning |

---

## Low Findings (17)

| # | File | Issue |
|---|------|-------|
| L1 | `gpu/cuda_kernel.rs:145` | shared_mem_bytes cast to u32 may truncate |
| L2 | `gpu/cuda_kernel.rs:217` | Unchecked n_queries * n for GPU buffer |
| L3 | `hnsw_index.rs:218` | Hardcoded 10M limit without config |
| L4 | `splats.rs:112` | Pre-allocation of max_splats * latent_dim array |
| L5 | `json_store.rs:24` | Unbounded serde_json::from_str |
| L6 | `sqlite_store.rs:44` | Unbounded serde_json::from_str on metadata |
| L7 | `mcp_server.rs:540` | Unbounded serde_json::from_str on user metadata |
| L8 | `mcp_server.rs:362-366` | No dimension validation on user embedding |
| L9 | `mcp_server.rs:898` | Hardcoded SQLite database path |
| L10 | `api_server.rs:24-28` | Unnecessary Arc<Mutex<usize>> for counter |
| L11 | `cluster/sync.rs` | SyncQueue has no synchronization docs |
| L12 | `auto_scaling.rs` | AutoScaler has no synchronization docs |
| L13 | `encoding.rs:83` | d as i32 frequency exponent truncation |
| L14 | `build.rs:44` | expect() panics on nvcc execution failure |
| L15 | `Cargo.toml` | tokio "full" feature — unnecessary bloat |
| L16 | `Cargo.toml` | rusqlite "bundled" — supply chain concern |
| L17 | `cluster/protocol.rs:27-35` | Cluster messages lack auth/integrity fields |

---

## What's Already Good ✅

- **Zero SQL injection** — All rusqlite queries use parameterized `params![]`
- **Zero path traversal** — `validate_shard_name()` and `validate_path()` properly check `..`, `/`, `\`
- **Zero transmute** — No `std::mem::transmute` anywhere in the codebase
- **Zero hardcoded secrets** — No API keys, passwords, or tokens in source
- **No git/path dependencies** — All from crates.io with pinned versions
- **CUDA unsafe blocks documented** — All have SAFETY comments
- **bytemuck already in Cargo.toml** — Ready to replace unsafe casts
- **persistence.rs has good guards** — `checked_mul`, 1B element cap
- **graph_splat.rs has explicit caps** — MAX_TRAVERSE_RESULTS, MAX_EMBEDDING_DIM

---

## Recommended Fix Priority

### Phase 1 — Network Security (Critical, ~1 day)
1. Add API key auth middleware to HTTP routes (C1)
2. Restrict CORS to localhost (C2)
3. Remove sensitive data from logs (H5)

### Phase 2 — Memory Safety Hardening (High, ~2 days)
4. Add `checked_mul` + caps in binary loaders (H1, H2)
5. Add bounds checks on CUDA kernel args (H9)
6. Replace unsafe pointer casts with bytemuck (M15)

### Phase 3 — Concurrency Fixes (High, ~1 day)
7. Fix TOCTOU + nested lock in api_server (H6)
8. Fix WAL truncate race condition (H7)

### Phase 4 — Input Validation (Medium, ~2 days)
9. Add text length limits in MCP handlers (M1)
10. Clamp graph traverse depth (M2)
11. Validate embedding dimensions (L8)
12. Add serde_json recursion limits (L5-L7)

### Phase 5 — Polish (Low, ~1 day)
13. Make SQLite path configurable (L9, M10)
14. Reduce tokio features (L15)
15. Add Send/Sync assertions for GpuIndex (M13)
