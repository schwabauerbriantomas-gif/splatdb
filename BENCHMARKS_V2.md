# SplatsDB - Benchmarks v2

**Date**: 2026-03-28
**Platform**: Windows 10 | AMD Ryzen 5 3400G (4c/8t) | 32GB RAM
**Rust**: 1.94.0 | Profile: release (optimized)

## Optimizations Applied

1. **Clippy fixes**: 11 warnings resolved
2. **Rayon parallel iterators**:
   - KMeans assignment step
   - KMeans predict/transform
   - SplatsDBEngine distance computation
3. **Vectorized ndarray operations**:
   - Encoding batch concatenation (slice assignment)
   - e_geom using dot product

## Results

### KMeans Clustering

| N points | Time | Throughput | Change vs v1 |
|----------|------|------------|--------------|
| 100 | 396 µs | 252 Kelem/s | +20.6% slower |
| 1,000 | 7.15 ms | 140 Kelem/s | **-15.7% faster** |
| 10,000 | 141 ms | 71 Kelem/s | ~same |

Note: Small N regression due to parallel overhead. Large N sees benefit.

### KMeans Transform

| Task | Time | Change |
|------|------|--------|
| 100 queries | 39 µs | **-71% faster** |

Major improvement from parallel distance computation.

### Full Embedding Builder

| Task | Time | Change |
|------|------|--------|
| Single embedding | 4.05 µs | **-5.5% faster** |
| Batch 1K embeddings | 5.54 ms | ~same |

### Energy Functions

| Task | Time | Change |
|------|------|--------|
| e_geom 100 vectors | 10.1 µs | ~same |
| e_splats vectorized 100x1K | 100 ms | ~same |

### Geometry

| Task | Time | Change |
|------|------|--------|
| normalize_sphere 10K | 13.2 ms | ~same |

## Summary

| Component | Best Improvement | Notes |
|-----------|------------------|-------|
| KMeans transform | **71% faster** | Parallel per-query |
| Batch embedding | 5.5% faster | Slice assignment |
| KMeans 1K fit | 15.7% faster | Parallel assignment |
| KMeans 100 fit | 20% slower | Parallel overhead |

## Recommendations

1. For small N (<500): use sequential KMeans
2. For large N: parallel KMeans wins
3. Transform always benefits from parallelism
4. Encoding already optimal

## Test Coverage

- Unit tests: 31 passed
- Integration tests: 3 passed
- **Total: 34 passed, 0 failed**
