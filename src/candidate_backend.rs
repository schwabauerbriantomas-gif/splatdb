//! Pluggable candidate-generation backends for `SplatStore`.
//!
//! Frozen contract (swarm O2, father → B2):
//! ```text
//! trait CandidateBackend {
//!     fn search(&self, q: &[f32], k_fetch: usize) -> Vec<(usize, f32)>;
//!     fn add_batch(&mut self, vectors: &[Vec<f32>]) -> Result<(), String>;
//!     fn len(&self) -> usize;
//!     fn capabilities(&self) -> BackendCaps;
//! }
//! struct BackendCaps { supports_delete: bool, incremental: bool }
//! ```
//! The extended surface (`name` / `build` / `is_built` / `save`, plus a
//! default `is_empty`) follows the R8 interface report §2, adapted to what
//! `SplatStore` actually needs behind a `Box<dyn CandidateBackend>`
//! (build-from-scratch, persistence and built-check cannot be downcast).
//!
//! Design principle (R8 §1.5): the backend is ONLY a candidate generator.
//! Every distance shown to the user is recomputed exactly against `mu` by
//! the store, so backend distances are hints and need not be exact.

use crate::hnsw_index::HNSWIndex;
use crate::interfaces::VectorIndex;
use ndarray::Array2;
use std::path::Path;

/// Capabilities advertised by a [`CandidateBackend`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BackendCaps {
    /// Backend supports (tombstone-based) removal of indexed vectors.
    pub supports_delete: bool,
    /// Backend supports incremental insertion without a full rebuild.
    pub incremental: bool,
}

/// Candidate-generation backend for `SplatStore`.
///
/// `search` returns `(index, distance_hint)` pairs ordered best-first by the
/// backend's internal metric; the store re-ranks every candidate with exact
/// distances against `mu`, so hints may be approximate.
pub trait CandidateBackend: Send + Sync {
    /// Approximate top-`k_fetch` candidates for `q` as `(index, hint)`.
    fn search(&self, q: &[f32], k_fetch: usize) -> Vec<(usize, f32)>;

    /// Append vectors incrementally. Only called when
    /// `capabilities().incremental` is true.
    fn add_batch(&mut self, vectors: &[Vec<f32>]) -> Result<(), String>;

    /// Number of indexed vectors (net of tombstones).
    fn len(&self) -> usize;

    /// Whether nothing is indexed.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Advertised capabilities.
    fn capabilities(&self) -> BackendCaps;

    // ── Extended surface required by SplatStore today (R8 §2 adaptation) ──

    /// Backend identifier ("hnsw", "cagra", ...).
    fn name(&self) -> &'static str;

    /// (Re)build the index from scratch over `vectors` (n×dim, row-major).
    fn build(&mut self, vectors: &Array2<f32>);

    /// Whether the backend holds a usable (non-empty) index.
    fn is_built(&self) -> bool;

    /// Persist the index to `path` using the backend's own format.
    fn save(&self, path: &Path) -> Result<(), String>;
}

/// Adapter that puts the existing homegrown HNSW behind [`CandidateBackend`].
///
/// Behavior-identical to the pre-trait code paths: `search` returns exactly
/// the indices `HNSWIndex::search` returned (via its `VectorIndex` impl),
/// paired with the graph's distance hints.
pub struct HnswBackend {
    dim: usize,
    index: HNSWIndex,
}

impl HnswBackend {
    /// New, empty backend (same parameters as `HNSWIndex::new`).
    pub fn new(
        dim: usize,
        m: usize,
        ef_construction: usize,
        ef_search: usize,
        metric: &str,
        seed: u64,
    ) -> Self {
        Self {
            dim,
            index: HNSWIndex::new(dim, m, ef_construction, ef_search, metric, seed),
        }
    }

    /// Wrap an already-built (or loaded from disk) `HNSWIndex`.
    ///
    /// `dim` is the store's `latent_dim` (the index's own dim field is
    /// private and must agree — same trust level as the pre-trait code).
    pub fn from_index(index: HNSWIndex, dim: usize) -> Self {
        Self { dim, index }
    }

    /// Access the wrapped index (e.g. for legacy direct calls).
    pub fn index(&self) -> &HNSWIndex {
        &self.index
    }

    /// Mutable access to the wrapped index.
    pub fn index_mut(&mut self) -> &mut HNSWIndex {
        &mut self.index
    }
}

impl CandidateBackend for HnswBackend {
    fn search(&self, q: &[f32], k_fetch: usize) -> Vec<(usize, f32)> {
        if q.is_empty() || !self.index.is_built() {
            return Vec::new();
        }
        let view = ndarray::ArrayView1::from(q);
        let res = self.index.search(view, k_fetch);
        res.indices.into_iter().zip(res.distances).collect()
    }

    fn add_batch(&mut self, vectors: &[Vec<f32>]) -> Result<(), String> {
        if vectors.is_empty() {
            return Ok(());
        }
        for (i, v) in vectors.iter().enumerate() {
            if v.len() != self.dim {
                return Err(format!(
                    "hnsw add_batch: vector {} has dim {}, expected {}",
                    i,
                    v.len(),
                    self.dim
                ));
            }
            self.index.insert(v);
        }
        Ok(())
    }

    fn len(&self) -> usize {
        self.index.n_items()
    }

    fn capabilities(&self) -> BackendCaps {
        BackendCaps {
            supports_delete: true,
            incremental: true,
        }
    }

    fn name(&self) -> &'static str {
        "hnsw"
    }

    fn build(&mut self, vectors: &Array2<f32>) {
        self.index.build(vectors.clone());
    }

    fn is_built(&self) -> bool {
        self.index.is_built()
    }

    fn save(&self, path: &Path) -> Result<(), String> {
        self.index.save(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_vectors(n: usize, dim: usize) -> Array2<f32> {
        Array2::from_shape_fn((n, dim), |(i, j)| ((i * 31 + j * 7) as f32 * 0.37).sin())
    }

    #[test]
    fn test_hnsw_backend_parity_with_vector_index() {
        // The backend must return exactly what HNSWIndex::search returns.
        let dim = 8usize;
        let mut direct = HNSWIndex::new(dim, 8, 40, 20, "l2", 42);
        let data = sample_vectors(50, dim);
        direct.build(data.clone());

        let mut backend = HnswBackend::new(dim, 8, 40, 20, "l2", 42);
        backend.build(&data);

        assert!(backend.is_built());
        assert_eq!(backend.len(), direct.n_items());
        assert_eq!(backend.name(), "hnsw");
        assert_eq!(
            backend.capabilities(),
            BackendCaps {
                supports_delete: true,
                incremental: true
            }
        );

        let q: Vec<f32> = data.row(7).to_vec();
        let view = ndarray::ArrayView1::from(&q[..]);
        let expect = crate::interfaces::VectorIndex::search(&direct, view, 5);
        let got = backend.search(&q, 5);

        assert_eq!(got.len(), expect.indices.len());
        for (g, idx) in got.iter().zip(expect.indices.iter()) {
            assert_eq!(g.0, *idx, "same index order as HNSWIndex::search");
        }
        // Self-query: nearest candidate must be the query itself.
        assert_eq!(got[0].0, 7);
    }

    #[test]
    fn test_hnsw_backend_add_batch_incremental() {
        let dim = 8usize;
        let mut backend = HnswBackend::new(dim, 8, 40, 20, "l2", 42);
        let data = sample_vectors(30, dim);
        backend.build(&data);
        assert_eq!(backend.len(), 30);

        let extra: Vec<Vec<f32>> = (0..5)
            .map(|i| {
                (0..dim)
                    .map(|j| ((i * 13 + j) as f32 * 0.11).cos())
                    .collect()
            })
            .collect();
        backend.add_batch(&extra).expect("add_batch should succeed");
        assert_eq!(backend.len(), 35);

        // Inconsistent dimensions must error, not corrupt the index.
        let bad = vec![vec![0.0f32; dim + 1]];
        assert!(backend.add_batch(&bad).is_err());
        assert_eq!(backend.len(), 35);

        // Empty batch is a no-op.
        backend.add_batch(&[]).expect("empty batch ok");
        assert_eq!(backend.len(), 35);

        // New vectors must be searchable.
        let q = extra[2].clone();
        let res = backend.search(&q, 1);
        assert_eq!(res[0].0, 32, "incrementally added vector 32 must be found");
    }
}
