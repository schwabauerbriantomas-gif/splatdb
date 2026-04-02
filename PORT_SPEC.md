# M2M Rust Port — Full Specification

## Goal
Complete the Rust port of m2m-vector-search from Python. Fix all bugs, optimize with rayon/BLAS, and port missing modules.

## Location
- Rust project: `C:\Users\Brian\Desktop\m2m-rust`
- Python source (reference): `C:\Users\Brian\Desktop\m2m-vector-search-main\src\m2m\`
- Shell: PowerShell (NOT bash). Use `& "$env:USERPROFILE\.cargo\bin\cargo.exe"` for cargo.

## Current State
- 15 Rust modules ported, 26/26 tests passing
- Compiles clean (0 errors)
- Has benchmark suite (criterion)

## Tasks (in priority order)

### Task 1: Fix KMeans Performance (CRITICAL)
The Rust KMeans is 5x SLOWER than Python for N=10,000. Python uses numpy/BLAS.

**Fix**: Add `ndarray-linalg` or `blas-src` + `openblas-src` to Cargo.toml and use BLAS-backed matrix operations in KMeans. The key bottleneck is distance computation in the assignment step.

Alternative: Use `rayon` parallel iterators for the assignment step instead of BLAS.

**Benchmark target**: Rust KMeans N=10K should be at least 2x faster than Python (110ms).

### Task 2: Fix e_geom Performance
Python e_geom (38us) beats Rust (46us). The Rust implementation should use vectorized operations.

**Fix**: Replace loops with ndarray vectorized operations or rayon parallel reduce.

### Task 3: Fix Ownership Bugs in hrm2_engine.rs
Lines 206 and 237 use `q.insert_axis(Axis(0))` which moves `q`, then `q` is borrowed again at lines 223 and 252.

**Fix**: Use `q.view().insert_axis(Axis(0))` instead (uses view, doesn't move).

### Task 4: Port SplatStore (HIGH PRIORITY)
Port `C:\Users\Brian\Desktop\m2m-vector-search-main\src\m2m\splats.py` to Rust as `src/splats.rs`.

This is the MAIN API that users interact with. It wraps:
- HRM2Engine for indexing
- FullEmbeddingBuilder for encoding
- SplatStore.add_splat(), build_index(), find_neighbors(), delete_splat()

### Task 5: Port Memory Module
Port `memory.py` and `semantic_memory.py` to `src/memory.rs`.
- VectorMemoryStore: store/recall with HRM2 backend
- SemanticMemory: concept graph with energy-based consolidation

### Task 6: Port Query System
Port `query_optimizer.py` and `query_router.py` to `src/query.rs`.
- QueryOptimizer: adaptive LOD selection
- QueryRouter: route queries to best index (HNSW vs HRM2 vs linear)

### Task 7: Port EBM Exploration
Port `ebm/exploration.py` to `src/ebm/exploration.rs`.
- Langevin dynamics sampler
- Temperature annealing schedule

### Task 8: Optimize All Modules with rayon
Add parallel iterators to:
- encoding.rs: batch encoding (already has rayon in Cargo.toml, not used)
- clustering.rs: assignment step
- engine.rs: distance computation
- energy.rs: e_splats_vectorized

### Task 9: Clean Up Warnings
Remove all unused imports across all modules (15 warnings).

### Task 10: Add Integration Tests
Add comprehensive tests in `tests/` that mirror the Python test suite.
- test_splat_store_workflow
- test_hrm2_end_to_end
- test_memory_store_recall
- test_encoding_roundtrip

## Constraints
- Windows 10, PowerShell only
- No GPU (CPU-only)
- No flash-attn, no torch.compile
- f32 throughout
- Use `ndarray` + `rayon` as numerical stack
- All code must pass `cargo test` and `cargo clippy` 
- Use `serde` for serialization
- Each new module must have at least 2 tests

## Verification
After each task:
1. `cargo build` must succeed with 0 errors
2. `cargo test` must pass all tests
3. `cargo clippy` should have 0 errors (warnings OK temporarily)
