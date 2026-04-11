//! SplatStore - Main API for SplatsDB Vector Search.
//! Ported from Python splats.py. CPU-only.

use crate::dataset_transformer;
use crate::interfaces::VectorIndex;
use ndarray::{Array1, Array2, ArrayView1};

use crate::config::SplatsDBConfig;
use crate::hrm2_engine::HRM2Engine;

/// Result of a neighbor search.
#[derive(Clone, Debug)]
pub struct NeighborResult {
    pub mu: Vec<f32>,
    pub alpha: f32,
    pub kappa: f32,
    pub index: usize,
    pub distance: f32,
}

/// Main SplatStore - wraps HRM2Engine for high-level vector operations.
///
/// Subsystems are initialized based on preset config:
/// - `enable_quantization` → QuantizedStore for compressed search
/// - `enable_hnsw` → HNSWIndex as secondary index
/// - `enable_lsh` → CrossPolytopeLSH for approximate search
/// - `enable_semantic_memory` → SemanticMemoryDB for hybrid text+vector search
pub struct SplatStore {
    config: SplatsDBConfig,
    max_splats: usize,
    n_active: usize,
    mu: Array2<f32>,
    alpha: Array1<f32>,
    kappa: Array1<f32>,
    frequency: Array1<f32>,
    engine: HRM2Engine,
    next_id: u64,
    // Preset-driven subsystems (None when disabled)
    quant_store: Option<crate::quantization::QuantizedStore>,
    hnsw: Option<crate::hnsw_index::HNSWIndex>,
    /// Number of vectors currently indexed in HNSW. When this < n_active,
    /// there are unindexed vectors that need incremental insertion.
    hnsw_indexed_count: usize,
    lsh: Option<crate::lsh_index::CrossPolytopeLSH>,
    // Semantic memory is a separate layer (it contains its own SplatStore internally,
    // so we can't embed it here without creating a cycle). Use has_semantic_memory()
    // to check if the preset enables it, then manage SemanticMemoryDB externally.
}

impl SplatStore {
    /// New.
    pub fn new(config: SplatsDBConfig) -> Self {
        let max_splats = config.max_splats;
        let dim = config.latent_dim;
        let n_coarse = (((max_splats as f64).sqrt() / 10.0) as usize).max(10);
        let n_fine = (max_splats / n_coarse).max(100);
        let engine = HRM2Engine::new(n_coarse, n_fine, dim, n_coarse.min(5), "cosine");

        // Initialize subsystems based on preset config
        let quant_store = if config.enable_quantization {
            let qcfg = crate::quantization::QuantConfig {
                bits: config.quant_bits,
                projections: config.quant_projections,
                seed: 42,
                algorithm: match config.quant_algorithm {
                    crate::config::QuantAlgorithm::TurboQuant => {
                        crate::quantization::QuantAlgorithm::TurboQuant
                    }
                    crate::config::QuantAlgorithm::PolarQuant => {
                        crate::quantization::QuantAlgorithm::PolarQuant
                    }
                    crate::config::QuantAlgorithm::None => {
                        crate::quantization::QuantAlgorithm::TurboQuant
                    }
                },
            };
            crate::quantization::QuantizedStore::new(dim, qcfg).ok()
        } else {
            None
        };

        let hnsw = if config.enable_hnsw {
            Some(crate::hnsw_index::HNSWIndex::new(
                dim,
                config.hnsw_m,
                config.hnsw_ef_construction,
                config.hnsw_ef_search,
                &config.hnsw_metric,
                42,
            ))
        } else {
            None
        };

        let lsh = if config.enable_lsh {
            let lsh_cfg = crate::lsh_index::LSHConfig {
                dim,
                n_tables: 16,
                n_bits: 8,
                n_probes: 3,
                n_candidates: 100,
                seed: 42,
            };
            Some(crate::lsh_index::CrossPolytopeLSH::new(lsh_cfg))
        } else {
            None
        };

        let _semantic_enabled = config.enable_semantic_memory;

        Self {
            config,
            max_splats,
            n_active: 0,
            mu: Array2::zeros((max_splats, dim)),
            alpha: Array1::ones(max_splats),
            kappa: Array1::ones(max_splats),
            frequency: Array1::zeros(max_splats),
            engine,
            next_id: 0,
            quant_store,
            hnsw,
            hnsw_indexed_count: 0,
            lsh,
        }
    }

    /// Add a batch of splat vectors. Returns false if capacity exceeded.
    pub fn add_splat(&mut self, x: &Array2<f32>) -> bool {
        let n_new = x.nrows();
        if self.n_active + n_new > self.max_splats {
            return false;
        }
        for i in 0..n_new {
            self.mu.row_mut(self.n_active + i).assign(&x.row(i));
            self.alpha[self.n_active + i] = self.config.init_alpha as f32;
            self.kappa[self.n_active + i] = self.config.init_kappa as f32;
            self.frequency[self.n_active + i] = 1.0;
        }
        self.n_active += n_new;
        self.next_id += n_new as u64;
        true
    }

    /// Insert a single vector. Returns its index in the store.
    ///
    /// Convenience wrapper around `add_splat` for single vectors.
    /// Returns `None` if capacity is exceeded or dimension mismatches.
    pub fn insert(&mut self, vec: &[f32]) -> Option<usize> {
        let dim = self.config.latent_dim;
        if vec.len() != dim || self.n_active >= self.max_splats {
            return None;
        }
        let arr = Array2::from_shape_vec((1, dim), vec.to_vec()).ok()?;
        let idx = self.n_active;
        if !self.add_splat(&arr) {
            return None;
        }
        Some(idx)
    }

    /// Search for k nearest neighbors by raw vector.
    ///
    /// Convenience wrapper around `find_neighbors` that accepts a slice.
    /// Returns results sorted by distance (closest first).
    #[inline]
    pub fn search(&self, vec: &[f32], k: usize) -> Vec<NeighborResult> {
        let query = Array1::from_vec(vec.to_vec());
        self.find_neighbors(&query.view(), k)
    }

    /// Build the search index from active vectors.
    ///
    /// Feeds active vectors to all enabled subsystems:
    /// - HRM2 engine (always)
    /// - QuantizedStore (if enable_quantization)
    /// - HNSW index (if enable_hnsw)
    /// - LSH index (if enable_lsh)
    ///
    /// Optionally auto-saves HNSW to disk if `data_dir` is provided.
    pub fn build_index(&mut self) {
        self.build_index_with_save(None);
    }

    /// Build the search index and optionally auto-save HNSW to `data_dir`.
    pub fn build_index_with_save(&mut self, data_dir: Option<&str>) {
        if self.n_active == 0 {
            return;
        }
        let embeddings = self.mu.slice(ndarray::s![..self.n_active, ..]).to_owned();

        // Primary HRM2 index
        self.engine.index(Some(embeddings.clone()));

        // Quantized store
        if let Some(ref mut qs) = self.quant_store {
            qs.add_batch(&embeddings, 0);
        }

        // HNSW secondary index
        if let Some(ref mut hnsw) = self.hnsw {
            hnsw.build(embeddings.clone());
            self.hnsw_indexed_count = self.n_active;
        }

        // LSH index
        if let Some(ref mut lsh) = self.lsh {
            lsh.index(embeddings);
        }

        // Auto-save HNSW if data_dir provided
        if let Some(dir) = data_dir {
            if let Err(e) = self.save_hnsw(dir) {
                eprintln!("[splatsdb] Warning: failed to save HNSW index: {}", e);
            }
        }
    }

    /// Build index only if not already built. Tries to load persisted HNSW first.
    ///
    /// Returns true if the index was built or loaded, false if nothing needed doing.
    pub fn build_index_if_needed(&mut self, data_dir: &str) -> bool {
        if self.n_active == 0 {
            return false;
        }

        // Try to load persisted HNSW first
        if self.hnsw.is_some() && self.try_load_hnsw(data_dir) {
            eprintln!("[splatsdb] Loaded persisted HNSW index from disk");
            self.hnsw_indexed_count = self.n_active;

            // Still need to build other indexes (HRM2, quant, LSH) if not done
            let embeddings = self.mu.slice(ndarray::s![..self.n_active, ..]).to_owned();
            self.engine.index(Some(embeddings.clone()));
            if let Some(ref mut qs) = self.quant_store {
                qs.add_batch(&embeddings, 0);
            }
            if let Some(ref mut lsh) = self.lsh {
                lsh.index(embeddings);
            }
            return true;
        }

        // No persisted index — build from scratch and save
        self.build_index_with_save(Some(data_dir));
        true
    }

    /// Try to load persisted HNSW from `{data_dir}/hnsw_index.bin`.
    /// Returns true if loaded successfully.
    pub fn try_load_hnsw(&mut self, data_dir: &str) -> bool {
        let path = std::path::Path::new(data_dir).join("hnsw_index.bin");
        if !path.exists() {
            return false;
        }

        let m = self.config.hnsw_m;
        let ef_construction = self.config.hnsw_ef_construction;
        let ef_search = self.config.hnsw_ef_search;
        let metric = &self.config.hnsw_metric;

        match crate::hnsw_index::HNSWIndex::load(&path, m, ef_construction, ef_search, metric, 42) {
            Ok(loaded) => {
                self.hnsw = Some(loaded);
                true
            }
            Err(e) => {
                eprintln!("[splatsdb] Warning: failed to load HNSW index: {}", e);
                false
            }
        }
    }

    /// Save the HNSW index to `{data_dir}/hnsw_index.bin`.
    pub fn save_hnsw(&self, data_dir: &str) -> Result<(), String> {
        if let Some(ref hnsw) = self.hnsw {
            if !hnsw.is_built() {
                return Ok(());
            }
            let dir = std::path::Path::new(data_dir);
            std::fs::create_dir_all(dir).map_err(|e| format!("create dir {}: {}", data_dir, e))?;
            let path = dir.join("hnsw_index.bin");
            hnsw.save(&path)?;
            eprintln!(
                "[splatsdb] Saved HNSW index ({} vectors) to {}",
                hnsw.n_items(),
                path.display()
            );
        }
        Ok(())
    }

    // ── Incremental HNSW insertion ──────────────────────────────────

    /// Check if HNSW has unindexed vectors (new vectors since last build/insert).
    pub fn hnsw_needs_sync(&self) -> bool {
        self.hnsw.is_some() && self.hnsw_indexed_count < self.n_active
    }

    /// Get the number of vectors currently indexed in HNSW.
    pub fn hnsw_indexed_count(&self) -> usize {
        self.hnsw_indexed_count
    }

    /// Insert all new (unindexed) vectors into the HNSW graph incrementally.
    ///
    /// Unlike `build_index()` which reconstructs everything, this only inserts
    /// the vectors added since the last build/sync. O(M) per vector where M
    /// is the HNSW connectivity parameter — typically M=48, so very fast.
    ///
    /// Returns the number of newly inserted vectors, or 0 if nothing to do.
    pub fn hnsw_sync_incremental(&mut self) -> usize {
        let start = self.hnsw_indexed_count;
        let end = self.n_active;
        if start >= end || self.hnsw.is_none() {
            return 0;
        }

        let count = end - start;
        if let Some(ref mut hnsw) = self.hnsw {
            for i in start..end {
                let vec: Vec<f32> = self.mu.row(i).to_vec();
                hnsw.insert(&vec);
            }
        }
        self.hnsw_indexed_count = end;
        count
    }

    /// Insert all new (unindexed) vectors incrementally, then save to disk.
    ///
    /// This is the recommended method for production use — insert new vectors,
    /// sync to HNSW, persist the updated graph. No full rebuild needed.
    ///
    /// Returns the number of newly inserted vectors.
    pub fn hnsw_sync_and_save(&mut self, data_dir: &str) -> usize {
        let count = self.hnsw_sync_incremental();
        if count > 0 {
            if let Err(e) = self.save_hnsw(data_dir) {
                eprintln!(
                    "[splatsdb] Warning: failed to save HNSW after incremental sync: {}",
                    e
                );
            }
        }
        count
    }

    /// Add a single vector and incrementally insert it into HNSW.
    ///
    /// This is an all-in-one method that:
    /// 1. Adds the vector to the SplatStore
    /// 2. Incrementally inserts it into the HNSW graph (if enabled)
    /// 3. Optionally saves the updated HNSW to disk
    ///
    /// Returns the index of the new vector, or None if capacity exceeded.
    pub fn insert_with_hnsw(&mut self, vec: &[f32], data_dir: Option<&str>) -> Option<usize> {
        let idx = self.insert(vec)?;
        if let Some(ref mut hnsw) = self.hnsw {
            hnsw.insert(vec);
            self.hnsw_indexed_count = self.n_active;
            if let Some(dir) = data_dir {
                if let Err(e) = self.save_hnsw(dir) {
                    eprintln!(
                        "[splatsdb] Warning: failed to save HNSW after insert: {}",
                        e
                    );
                }
            }
        }
        Some(idx)
    }

    /// Add a batch of vectors and incrementally insert them into HNSW.
    ///
    /// More efficient than calling `insert_with_hnsw` in a loop because
    /// it only saves to disk once after all insertions.
    pub fn add_batch_with_hnsw(&mut self, vectors: &Array2<f32>, data_dir: Option<&str>) -> bool {
        let n_before = self.n_active;
        if !self.add_splat(vectors) {
            return false;
        }

        if let Some(ref mut hnsw) = self.hnsw {
            for i in n_before..self.n_active {
                let vec: Vec<f32> = self.mu.row(i).to_vec();
                hnsw.insert(&vec);
            }
            self.hnsw_indexed_count = self.n_active;
            if let Some(dir) = data_dir {
                if let Err(e) = self.save_hnsw(dir) {
                    eprintln!("[splatsdb] Warning: failed to save HNSW after batch: {}", e);
                }
            }
        }
        true
    }

    /// Find k nearest neighbors for a single query.
    ///
    /// Performs a brute-force linear scan over all active vectors using
    /// squared L2 distance with partial-sort top-k selection.
    #[inline]
    pub fn find_neighbors(&self, query: &ArrayView1<f32>, k: usize) -> Vec<NeighborResult> {
        let n = self.n_active;
        if n == 0 || k == 0 {
            return Vec::new();
        }
        let k = k.min(n);
        let index_data = self.mu.slice(ndarray::s![..n, ..]);

        // Squared L2 distances
        let dists_sq: Vec<f32> = index_data
            .outer_iter()
            .map(|row| {
                row.iter()
                    .zip(query.iter())
                    .map(|(a, b)| {
                        let d = a - b;
                        d * d
                    })
                    .sum()
            })
            .collect();

        // Partial sort top-k
        let mut indexed: Vec<(f32, usize)> =
            dists_sq.iter().enumerate().map(|(i, &d)| (d, i)).collect();
        if k < indexed.len() {
            indexed.select_nth_unstable_by(k, |a, b| {
                a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        indexed[..k].sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        indexed[..k]
            .iter()
            .map(|&(dist_sq, idx)| NeighborResult {
                mu: index_data.row(idx).to_vec(),
                alpha: self.alpha[idx],
                kappa: self.kappa[idx],
                index: idx,
                distance: dist_sq.sqrt(),
            })
            .collect()
    }

    /// Find k nearest neighbors using ALL enabled search backends.
    ///
    /// Strategy (priority order):
    /// 1. If HNSW is built → use as primary (sub-millisecond), re-rank with exact distances
    /// 2. If only quantized or LSH → use as pre-filter, then exact re-rank
    /// 3. Fallback: linear scan (O(N), exact but slow)
    ///
    /// Returns results ranked by exact distance.
    pub fn find_neighbors_fused(&self, query: &ArrayView1<f32>, k: usize) -> Vec<NeighborResult> {
        let n = self.n_active;
        if n == 0 || k == 0 {
            return Vec::new();
        }
        let k = k.min(n);
        let index_data = self.mu.slice(ndarray::s![..n, ..]);

        // Helper: compute exact L2 distance for re-ranking
        let exact_dist = |idx: usize| -> f32 {
            index_data
                .row(idx)
                .iter()
                .zip(query.iter())
                .map(|(a, b)| {
                    let d = a - b;
                    d * d
                })
                .sum::<f32>()
                .sqrt()
        };

        // Strategy 1: HNSW primary (sub-linear, high recall)
        if let Some(ref hnsw) = self.hnsw {
            if hnsw.is_built() {
                // Over-fetch from HNSW for better recall, then exact re-rank
                let fetch_k = (k * self.config.over_fetch).min(n);
                let hresults = hnsw.search(*query, fetch_k);

                // Re-rank with exact distances
                let mut ranked: Vec<(usize, f32)> = hresults
                    .indices
                    .iter()
                    .filter(|&&idx| idx < n)
                    .map(|&idx| (idx, exact_dist(idx)))
                    .collect();

                ranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

                return ranked[..k]
                    .iter()
                    .map(|&(idx, dist)| NeighborResult {
                        mu: index_data.row(idx).to_vec(),
                        alpha: self.alpha[idx],
                        kappa: self.kappa[idx],
                        index: idx,
                        distance: dist,
                    })
                    .collect();
            }
        }

        // Strategy 2: Collect candidates from approximate backends
        let mut candidate_set: std::collections::HashSet<usize> = std::collections::HashSet::new();

        // Quantized search
        if let Some(ref qs) = self.quant_store {
            let qresults = qs.search(query, k * 5);
            for (id, _) in qresults {
                let idx = id as usize;
                if idx < n {
                    candidate_set.insert(idx);
                }
            }
        }

        // LSH search
        if let Some(ref lsh) = self.lsh {
            let (indices, _) = lsh.query(query, k * 5);
            for &idx in &indices {
                if idx < n {
                    candidate_set.insert(idx);
                }
            }
        }

        // If we have candidates from approximate backends, re-rank them
        if !candidate_set.is_empty() {
            let mut ranked: Vec<(usize, f32)> = candidate_set
                .into_iter()
                .map(|idx| (idx, exact_dist(idx)))
                .collect();
            ranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            if ranked.len() > k {
                ranked.truncate(k);
            }
            return ranked
                .iter()
                .map(|&(idx, dist)| NeighborResult {
                    mu: index_data.row(idx).to_vec(),
                    alpha: self.alpha[idx],
                    kappa: self.kappa[idx],
                    index: idx,
                    distance: dist,
                })
                .collect();
        }

        // Strategy 3: Fallback — linear scan (always correct, O(N))
        self.find_neighbors(query, k)
    }

    /// Check if quantization subsystem is active.
    pub fn has_quantization(&self) -> bool {
        self.quant_store.is_some()
    }
    /// Check if HNSW index is active.
    pub fn has_hnsw(&self) -> bool {
        self.hnsw.is_some()
    }
    /// Check if HNSW index is active AND has been built (has vectors).
    pub fn hnsw_is_built(&self) -> bool {
        self.hnsw.as_ref().is_some_and(|h| h.is_built())
    }
    /// Check if LSH index is active.
    pub fn has_lsh(&self) -> bool {
        self.lsh.is_some()
    }
    /// Check if semantic memory is enabled in this preset.
    /// Note: SemanticMemoryDB must be managed externally (it wraps its own SplatStore).
    pub fn has_semantic_memory(&self) -> bool {
        self.config.enable_semantic_memory
    }

    /// Batch find neighbors for multiple queries.
    pub fn find_neighbors_batch(
        &self,
        queries: &Array2<f32>,
        k: usize,
    ) -> Vec<Vec<NeighborResult>> {
        (0..queries.nrows())
            .map(|i| self.find_neighbors(&queries.row(i), k))
            .collect()
    }

    /// Find k nearest neighbors using HRM2 engine (fast path).
    ///
    /// Uses the HRM2 hierarchical index with LOD 2 (exact within probed clusters)
    /// for approximate search. Falls back to linear scan if:
    /// - Dataset has fewer than 100 active vectors (not worth the overhead)
    /// - HRM2 engine is not indexed
    /// - HRM2 query returns an error
    pub fn find_neighbors_fast(
        &mut self,
        query: &ArrayView1<f32>,
        k: usize,
    ) -> Vec<NeighborResult> {
        let n = self.n_active;
        if n == 0 || k == 0 {
            return Vec::new();
        }
        let k = k.min(n);

        // For small datasets, linear scan is cheaper than HRM2 overhead
        if n < 100 {
            return self.find_neighbors(query, k);
        }

        // Try HRM2 fast path with LOD 2 (exact within probed clusters)
        match self.engine.query(query, k, 2) {
            Ok(candidates) if !candidates.is_empty() => {
                let index_data = self.mu.slice(ndarray::s![..n, ..]);
                candidates
                    .into_iter()
                    .map(|(idx, dist)| NeighborResult {
                        mu: index_data.row(idx).to_vec(),
                        alpha: self.alpha[idx],
                        kappa: self.kappa[idx],
                        index: idx,
                        distance: dist,
                    })
                    .collect()
            }
            _ => {
                // HRM2 not indexed or query failed — fall back to linear scan
                self.find_neighbors(query, k)
            }
        }
    }

    /// Find neighbors with validation. Returns error on invalid input.
    pub fn find_neighbors_validated(
        &self,
        query: &ArrayView1<f32>,
        k: usize,
    ) -> Result<Vec<NeighborResult>, String> {
        if k < 1 {
            return Err("k must be >= 1".into());
        }
        if query.len() != self.config.latent_dim {
            return Err(format!(
                "Query dimension mismatch: expected {}, got {}",
                self.config.latent_dim,
                query.len()
            ));
        }
        if !query.iter().all(|v| v.is_finite()) {
            return Err("Query vector contains NaN or Inf".into());
        }
        Ok(self.find_neighbors(query, k))
    }

    /// Load pre-computed splats (mu, alpha, kappa) into the store.
    pub fn load_splats(
        &mut self,
        mu_data: &Array2<f32>,
        alpha_data: &[f32],
        kappa_data: &[f32],
    ) -> Result<(), String> {
        let n = mu_data.nrows();
        if n > self.max_splats {
            return Err(format!("Too many splats: {} > {}", n, self.max_splats));
        }
        if alpha_data.len() < n || kappa_data.len() < n {
            return Err("alpha/kappa arrays too short".into());
        }
        for i in 0..n {
            self.mu.row_mut(i).assign(&mu_data.row(i));
            self.alpha[i] = alpha_data[i];
            self.kappa[i] = kappa_data[i];
            self.frequency[i] = 1.0;
        }
        // Zero out remainder
        for i in n..self.max_splats {
            self.mu.row_mut(i).fill(0.0);
            self.alpha[i] = 0.0;
            self.kappa[i] = 0.0;
            self.frequency[i] = 0.0;
        }
        self.n_active = n;
        self.next_id = n as u64;
        Ok(())
    }

    /// Update frequency for accessed splats (SOC boost).
    pub fn touch(&mut self, indices: &[usize]) {
        for &idx in indices {
            if idx < self.n_active {
                self.frequency[idx] += 1.0;
            }
        }
    }

    /// Decay all frequencies (call periodically).
    pub fn decay_frequencies(&mut self, factor: f32) {
        for i in 0..self.n_active {
            self.frequency[i] *= factor;
        }
    }

    /// Get active frequency values.
    pub fn get_frequency(&self) -> Option<Array1<f32>> {
        if self.n_active == 0 {
            return None;
        }
        Some(
            self.frequency
                .slice(ndarray::s![..self.n_active])
                .to_owned(),
        )
    }

    /// Shannon entropy of the kappa distribution.
    pub fn entropy(&self) -> f32 {
        if self.n_active == 0 {
            return 0.0;
        }
        let kappa = self.kappa.slice(ndarray::s![..self.n_active]);
        let total: f32 = kappa.sum();
        if total <= 0.0 {
            return 0.0;
        }
        let h: f32 = kappa
            .iter()
            .filter(|&&k| k > 0.0)
            .map(|&k| {
                let p = k / total;
                -p * p.ln()
            })
            .sum();
        let max_h = (self.n_active as f32).ln();
        if max_h <= 0.0 {
            0.0
        } else {
            (h / max_h).clamp(0.0, 1.0)
        }
    }

    /// Remove invalid splats (alpha ~0, NaN/Inf in mu).
    pub fn compact(&mut self) {
        if self.n_active == 0 {
            return;
        }
        let n = self.n_active;
        let mut write_idx = 0;
        for read_idx in 0..n {
            let alpha_ok = self.alpha[read_idx] >= 1e-6;
            let mu_ok = self.mu.row(read_idx).iter().all(|v| v.is_finite());
            if alpha_ok && mu_ok {
                if write_idx != read_idx {
                    let row_data = self.mu.row(read_idx).to_owned();
                    self.mu.row_mut(write_idx).assign(&row_data);
                    self.alpha[write_idx] = self.alpha[read_idx];
                    self.kappa[write_idx] = self.kappa[read_idx];
                    self.frequency[write_idx] = self.frequency[read_idx];
                }
                write_idx += 1;
            }
        }
        for i in write_idx..n {
            self.mu.row_mut(i).fill(0.0);
            self.alpha[i] = 0.0;
            self.kappa[i] = 0.0;
            self.frequency[i] = 0.0;
        }
        self.n_active = write_idx;
    }

    /// Get store statistics.
    pub fn get_statistics(&self) -> SplatStoreStats {
        SplatStoreStats {
            n_active: self.n_active,
            max_splats: self.max_splats,
            embedding_dim: self.config.latent_dim,
            entropy: self.entropy(),
        }
    }

    /// Number of active (non-empty) splats.
    #[inline]
    pub fn n_active(&self) -> usize {
        self.n_active
    }

    /// Find k nearest neighbors among a **subset** of vector indices.
    ///
    /// This is the core of spatial pre-filtering: instead of computing distances
    /// against ALL N vectors, we only compute against the candidate set.
    /// Complexity: O(|candidates| × dim) instead of O(N × dim).
    ///
    /// Returns results sorted by distance (ascending).
    pub fn find_neighbors_filtered(
        &self,
        query: &ArrayView1<f32>,
        candidate_indices: &[usize],
        k: usize,
    ) -> Vec<NeighborResult> {
        if candidate_indices.is_empty() || k == 0 {
            return Vec::new();
        }
        let k = k.min(candidate_indices.len());
        let index_data = self.mu.slice(ndarray::s![..self.n_active, ..]);

        // Compute distances only for candidates
        let mut scored: Vec<(f32, usize)> = candidate_indices
            .iter()
            .filter(|&&idx| idx < self.n_active)
            .map(|&idx| {
                let row = index_data.row(idx);
                let dist_sq: f32 = row
                    .iter()
                    .zip(query.iter())
                    .map(|(a, b)| {
                        let d = a - b;
                        d * d
                    })
                    .sum();
                (dist_sq, idx)
            })
            .collect();

        // Partial sort top-k
        if k < scored.len() {
            scored.select_nth_unstable_by(k, |a, b| {
                a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        scored[..k].sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        scored[..k]
            .iter()
            .map(|&(dist_sq, idx)| NeighborResult {
                mu: index_data.row(idx).to_vec(),
                alpha: self.alpha[idx],
                kappa: self.kappa[idx],
                index: idx,
                distance: dist_sq.sqrt(),
            })
            .collect()
    }
    /// Maximum splat capacity.
    #[inline]
    pub fn max_splats(&self) -> usize {
        self.max_splats
    }

    /// Get active mu vectors (n_active × dim slice).
    pub fn get_mu(&self) -> Option<Array2<f32>> {
        if self.n_active == 0 {
            return None;
        }
        Some(self.mu.slice(ndarray::s![..self.n_active, ..]).to_owned())
    }

    /// Get active alpha values.
    pub fn get_alpha(&self) -> Option<Array1<f32>> {
        if self.n_active == 0 {
            return None;
        }
        Some(self.alpha.slice(ndarray::s![..self.n_active]).to_owned())
    }

    /// Get active kappa values.
    pub fn get_kappa(&self) -> Option<Array1<f32>> {
        if self.n_active == 0 {
            return None;
        }
        Some(self.kappa.slice(ndarray::s![..self.n_active]).to_owned())
    }

    /// Ingest raw vectors via DatasetTransformer pipeline.
    ///
    /// Normalizes data using fit_transform, clusters into Gaussian Splats
    /// via KMeans, and loads resulting splat centroids into the store.
    /// Returns (splat count, compression ratio, transform stats).
    pub fn ingest_with_transformer(
        &mut self,
        data: &Array2<f32>,
        n_clusters: usize,
        seed: u64,
    ) -> Result<(usize, f64, dataset_transformer::TransformStats), String> {
        let (_, dim) = data.dim();
        if dim != self.config.latent_dim {
            return Err(format!(
                "Data dimension {} != config latent_dim {}",
                dim, self.config.latent_dim
            ));
        }

        let tconfig = dataset_transformer::TransformConfig {
            normalize: true,
            center: true,
            unit_sphere: true,
            augment_noise: 0.0,
            augment_flip: false,
            ..Default::default()
        };
        let mut transformer = dataset_transformer::DatasetTransformer::new(tconfig);
        let transformed = transformer.fit_transform(data);

        let min_cluster_size = 1;
        let (splats, _hierarchy, _partition, stats) =
            transformer.to_splats(&transformed, n_clusters, min_cluster_size, seed);

        if splats.is_empty() {
            return Err("Transformer produced 0 splats".into());
        }
        if splats.len() > self.max_splats {
            return Err(format!(
                "Too many splats: {} > max {}",
                splats.len(),
                self.max_splats
            ));
        }

        // Load splat centroids as mu
        let n = splats.len();
        let mu_arr =
            Array2::from_shape_vec((n, dim), splats.iter().flat_map(|s| s.mu.clone()).collect())
                .expect("mu shape should match (n, dim)");
        let alpha_arr: Vec<f32> = splats.iter().map(|s| s.alpha as f32).collect();
        let kappa_arr: Vec<f32> = splats.iter().map(|s| s.kappa as f32).collect();

        self.load_splats(&mu_arr, &alpha_arr, &kappa_arr)?;
        Ok((n, stats.compression_ratio, stats))
    }

    /// Ingest raw vectors with hierarchical clustering pipeline.
    ///
    /// Uses two-level KMeans (coarse then fine) for better cluster quality
    /// on large datasets. Returns transform stats.
    pub fn ingest_hierarchical(
        &mut self,
        data: &Array2<f32>,
        n_clusters: usize,
        min_cluster_size: usize,
        seed: u64,
    ) -> Result<(usize, f64, dataset_transformer::TransformStats), String> {
        let (_, dim) = data.dim();
        if dim != self.config.latent_dim {
            return Err(format!(
                "Data dimension {} != config latent_dim {}",
                dim, self.config.latent_dim
            ));
        }

        let tconfig = dataset_transformer::TransformConfig {
            normalize: true,
            center: true,
            unit_sphere: true,
            augment_noise: 0.0,
            augment_flip: false,
            ..Default::default()
        };
        let mut transformer = dataset_transformer::DatasetTransformer::new(tconfig);
        let transformed = transformer.fit_transform(data);

        let (splats, _hierarchy, _partition, stats) =
            transformer.to_splats_hierarchical(&transformed, n_clusters, min_cluster_size, seed);

        if splats.is_empty() {
            return Err("Hierarchical transformer produced 0 splats".into());
        }
        if splats.len() > self.max_splats {
            return Err(format!(
                "Too many splats: {} > max {}",
                splats.len(),
                self.max_splats
            ));
        }

        let n = splats.len();
        let mu_arr =
            Array2::from_shape_vec((n, dim), splats.iter().flat_map(|s| s.mu.clone()).collect())
                .expect("mu shape should match (n, dim)");
        let alpha_arr: Vec<f32> = splats.iter().map(|s| s.alpha as f32).collect();
        let kappa_arr: Vec<f32> = splats.iter().map(|s| s.kappa as f32).collect();

        self.load_splats(&mu_arr, &alpha_arr, &kappa_arr)?;
        Ok((n, stats.compression_ratio, stats))
    }

    /// Ingest raw vectors via Leader Clustering pipeline.
    ///
    /// O(n) single-pass — no KMeans iterations.
    /// Normalizes data, then uses Leader Clustering to create splats.
    /// `threshold` controls cluster radius (None = auto-compute).
    /// Returns (splat count, compression ratio, transform stats).
    pub fn ingest_leader(
        &mut self,
        data: &Array2<f32>,
        target_clusters: usize,
        seed: u64,
        threshold: Option<f64>,
    ) -> Result<(usize, f64, dataset_transformer::TransformStats), String> {
        let (_, dim) = data.dim();
        if dim != self.config.latent_dim {
            return Err(format!(
                "Data dimension {} != config latent_dim {}",
                dim, self.config.latent_dim
            ));
        }

        let tconfig = dataset_transformer::TransformConfig {
            normalize: true,
            center: true,
            unit_sphere: true,
            augment_noise: 0.0,
            augment_flip: false,
            ..Default::default()
        };
        let mut transformer = dataset_transformer::DatasetTransformer::new(tconfig);
        let transformed = transformer.fit_transform(data);

        let (splats, _hierarchy, _partition, stats) =
            transformer.to_splats_leader(&transformed, target_clusters, 1, seed, threshold);

        if splats.is_empty() {
            return Err("Leader clustering produced 0 splats".into());
        }
        if splats.len() > self.max_splats {
            return Err(format!(
                "Too many splats: {} > max {}",
                splats.len(),
                self.max_splats
            ));
        }

        let n = splats.len();
        let mu_arr =
            Array2::from_shape_vec((n, dim), splats.iter().flat_map(|s| s.mu.clone()).collect())
                .expect("mu shape should match (n, dim)");
        let alpha_arr: Vec<f32> = splats.iter().map(|s| s.alpha as f32).collect();
        let kappa_arr: Vec<f32> = splats.iter().map(|s| s.kappa as f32).collect();

        self.load_splats(&mu_arr, &alpha_arr, &kappa_arr)?;
        Ok((n, stats.compression_ratio, stats))
    }
}

/// Statistics about a SplatStore.
#[derive(Debug, Clone)]
pub struct SplatStoreStats {
    pub n_active: usize,
    pub max_splats: usize,
    pub embedding_dim: usize,
    pub entropy: f32,
}

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::*;

    fn test_config() -> SplatsDBConfig {
        let mut c = SplatsDBConfig::default();
        c.max_splats = 1000;
        c.latent_dim = 64;
        c
    }

    fn test_config_with_hnsw() -> SplatsDBConfig {
        let mut c = SplatsDBConfig::default();
        c.max_splats = 2000;
        c.latent_dim = 32;
        c.enable_hnsw = true;
        c.hnsw_m = 16;
        c.hnsw_ef_construction = 100;
        c.hnsw_ef_search = 50;
        c
    }

    fn random_vectors(n: usize, dim: usize, seed: u64) -> Array2<f32> {
        use rand::Rng;
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let data: Vec<f32> = (0..n * dim).map(|_| rng.gen::<f32>()).collect();
        Array2::from_shape_vec((n, dim), data).unwrap()
    }

    #[test]
    fn test_add_and_query() {
        let mut store = SplatStore::new(test_config());
        let data = Array2::from_shape_fn((100, 64), |(i, j)| ((i * 64 + j) as f32 * 0.01).sin());
        let mut normalized = data.clone();
        for mut row in normalized.rows_mut() {
            let norm = row.dot(&row).sqrt().max(1e-10);
            row.mapv_inplace(|v| v / norm);
        }
        assert!(store.add_splat(&normalized));
        assert_eq!(store.n_active(), 100);
        store.build_index();
        let query = normalized.row(0).to_owned();
        let results = store.find_neighbors(&query.view(), 5);
        assert_eq!(results.len(), 5);
        assert!(
            results[0].distance < 0.01,
            "First neighbor should be near-identical, got dist={}",
            results[0].distance
        );
        assert_eq!(results[0].index, 0);
    }

    #[test]
    fn test_capacity_limit() {
        let mut store = SplatStore::new(test_config());
        let data = Array2::zeros((1000, 64));
        assert!(store.add_splat(&data));
        let more = Array2::zeros((1, 64));
        assert!(!store.add_splat(&more));
    }

    #[test]
    fn test_entropy() {
        let mut store = SplatStore::new(test_config());
        assert_eq!(store.entropy(), 0.0);
        let data = Array2::ones((10, 64));
        store.add_splat(&data);
        let e = store.entropy();
        assert!(
            e > 0.0 && e <= 1.0,
            "Entropy should be in (0, 1], got {}",
            e
        );
    }

    #[test]
    fn test_compact() {
        let mut store = SplatStore::new(test_config());
        let data = Array2::from_shape_fn((10, 64), |(i, _)| (i as f32).cos());
        store.add_splat(&data);
        store.alpha[0] = 0.0;
        store.alpha[5] = 0.0;
        store.mu.row_mut(7).fill(f32::NAN);
        store.compact();
        assert_eq!(store.n_active(), 7);
    }

    #[test]
    fn test_empty_query() {
        let store = SplatStore::new(test_config());
        let q = Array1::zeros(64);
        let results = store.find_neighbors(&q.view(), 5);
        assert!(results.is_empty());
    }

    // ── Incremental HNSW insertion tests ──────────────────────────────

    #[test]
    fn test_incremental_hnsw_basic() {
        let mut store = SplatStore::new(test_config_with_hnsw());

        // Build initial index with 100 vectors
        let data = random_vectors(100, 32, 42);
        store.add_splat(&data);
        store.build_index();
        assert_eq!(store.hnsw_indexed_count(), 100);
        assert!(!store.hnsw_needs_sync());

        // Add 50 more vectors
        let more = random_vectors(50, 32, 99);
        store.add_splat(&more);
        assert_eq!(store.n_active(), 150);
        assert!(store.hnsw_needs_sync()); // 100 indexed, 150 active

        // Sync incrementally
        let inserted = store.hnsw_sync_incremental();
        assert_eq!(inserted, 50);
        assert_eq!(store.hnsw_indexed_count(), 150);
        assert!(!store.hnsw_needs_sync());
    }

    #[test]
    fn test_incremental_hnsw_recall() {
        let mut store = SplatStore::new(test_config_with_hnsw());

        // Build initial index
        let data = random_vectors(200, 32, 42);
        store.add_splat(&data);
        store.build_index();

        // Query a known vector — should find itself
        let query = data.row(0).to_owned();
        let results = store.find_neighbors_fused(&query.view(), 10);
        assert!(
            results.iter().any(|r| r.index == 0),
            "Should find vector 0 before incremental insert"
        );

        // Add 100 more vectors incrementally
        let more = random_vectors(100, 32, 77);
        store.add_splat(&more);
        store.hnsw_sync_incremental();

        // Query the SAME vector — should still find itself
        let results2 = store.find_neighbors_fused(&query.view(), 10);
        assert!(
            results2.iter().any(|r| r.index == 0),
            "Should find vector 0 after incremental insert"
        );

        // Query a NEW vector — should find itself
        let query2 = more.row(50).to_owned();
        let results3 = store.find_neighbors_fused(&query2.view(), 10);
        assert!(
            results3.iter().any(|r| r.index == 250),
            "Should find newly inserted vector 250"
        );
    }

    #[test]
    fn test_insert_with_hnsw_single() {
        let mut store = SplatStore::new(test_config_with_hnsw());

        // Build initial index
        let data = random_vectors(50, 32, 42);
        store.add_splat(&data);
        store.build_index();

        // Insert single vector via insert_with_hnsw
        let new_vec: Vec<f32> = (0..32).map(|i| (i as f32 * 0.1).sin()).collect();
        let idx = store
            .insert_with_hnsw(&new_vec, None)
            .expect("insert should succeed");
        assert_eq!(idx, 50);
        assert_eq!(store.hnsw_indexed_count(), 51);
        assert!(!store.hnsw_needs_sync());

        // Should be findable via search
        let query = Array1::from_vec(new_vec);
        let results = store.find_neighbors_fused(&query.view(), 5);
        assert!(
            results.iter().any(|r| r.index == 50),
            "Should find newly inserted vector via fused search"
        );
    }

    #[test]
    fn test_add_batch_with_hnsw() {
        let mut store = SplatStore::new(test_config_with_hnsw());

        // Initial batch
        let data = random_vectors(50, 32, 42);
        store.add_splat(&data);
        store.build_index();

        // Add batch incrementally
        let batch = random_vectors(30, 32, 88);
        assert!(store.add_batch_with_hnsw(&batch, None));
        assert_eq!(store.n_active(), 80);
        assert_eq!(store.hnsw_indexed_count(), 80);

        // Query a vector from the new batch
        let query = batch.row(15).to_owned();
        let results = store.find_neighbors_fused(&query.view(), 5);
        assert!(
            results.iter().any(|r| r.index == 65),
            "Should find vector from batch at index 65"
        );
    }

    #[test]
    fn test_incremental_hnsw_persistence() {
        let dir = std::env::temp_dir().join("splatsdb_incremental_test");
        std::fs::create_dir_all(&dir).unwrap();
        let data_dir = dir.to_str().unwrap();

        // Phase 1: Build initial index and save
        {
            let mut store = SplatStore::new(test_config_with_hnsw());
            let data = random_vectors(100, 32, 42);
            store.add_splat(&data);
            store.build_index_with_save(Some(data_dir));

            // Add 50 more incrementally and save
            let more = random_vectors(50, 32, 99);
            store.add_splat(&more);
            let inserted = store.hnsw_sync_and_save(data_dir);
            assert_eq!(inserted, 50);
        }

        // Phase 2: Load from disk — should have 150 vectors in HNSW
        {
            let mut store = SplatStore::new(test_config_with_hnsw());
            let data = random_vectors(100, 32, 42);
            let more = random_vectors(50, 32, 99);
            store.add_splat(&data);
            store.add_splat(&more);

            assert!(store.build_index_if_needed(data_dir));
            assert_eq!(store.hnsw_indexed_count(), 150);
            assert!(!store.hnsw_needs_sync());

            // Verify search works on all vectors
            let query = more.row(25).to_owned();
            let results = store.find_neighbors_fused(&query.view(), 5);
            assert!(
                results.iter().any(|r| r.index == 125),
                "Should find vector 125 after reload"
            );
        }

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_incremental_multiple_rounds() {
        let mut store = SplatStore::new(test_config_with_hnsw());

        // Round 1: 50 vectors
        let d1 = random_vectors(50, 32, 1);
        store.add_splat(&d1);
        store.build_index();
        assert_eq!(store.hnsw_indexed_count(), 50);

        // Round 2: +30
        let d2 = random_vectors(30, 32, 2);
        store.add_splat(&d2);
        let n = store.hnsw_sync_incremental();
        assert_eq!(n, 30);
        assert_eq!(store.hnsw_indexed_count(), 80);

        // Round 3: +20
        let d3 = random_vectors(20, 32, 3);
        store.add_splat(&d3);
        let n2 = store.hnsw_sync_incremental();
        assert_eq!(n2, 20);
        assert_eq!(store.hnsw_indexed_count(), 100);

        // Round 4: +40
        let d4 = random_vectors(40, 32, 4);
        store.add_batch_with_hnsw(&d4, None);
        assert_eq!(store.hnsw_indexed_count(), 140);

        // Verify all vectors are searchable
        assert_eq!(store.n_active(), 140);
        for (seed, start_idx) in [(1, 0), (2, 50), (3, 80), (4, 100)] {
            let data = random_vectors(
                if seed == 1 {
                    50
                } else if seed == 2 {
                    30
                } else if seed == 3 {
                    20
                } else {
                    40
                },
                32,
                seed,
            );
            let query = data.row(0).to_owned();
            let results = store.find_neighbors_fused(&query.view(), 5);
            assert!(
                results.iter().any(|r| r.index == start_idx),
                "Round {}: should find vector {} (seed={})",
                seed,
                start_idx,
                seed
            );
        }
    }

    #[test]
    fn test_incremental_noop_when_no_new_vectors() {
        let mut store = SplatStore::new(test_config_with_hnsw());
        let data = random_vectors(50, 32, 42);
        store.add_splat(&data);
        store.build_index();

        // Sync with no new vectors
        let n = store.hnsw_sync_incremental();
        assert_eq!(n, 0);
        assert_eq!(store.hnsw_indexed_count(), 50);
    }

    #[test]
    fn test_incremental_hnsw_cosine_metric() {
        let mut c = test_config_with_hnsw();
        c.latent_dim = 16;
        c.max_splats = 500;
        let mut store = SplatStore::new(c);

        let data = random_vectors(50, 16, 42);
        store.add_splat(&data);
        store.build_index();

        // Add incrementally
        let more = random_vectors(20, 16, 77);
        store.add_splat(&more);
        store.hnsw_sync_incremental();

        // Search should work
        let query = data.row(10).to_owned();
        let results = store.find_neighbors_fused(&query.view(), 5);
        assert!(!results.is_empty());
        assert!(
            results[0].distance < 0.1,
            "Self-query should be near-zero distance, got {}",
            results[0].distance
        );
    }

    #[test]
    fn test_find_neighbors_filtered() {
        let mut store = SplatStore::new(SplatsDBConfig::default());
        // Insert 100 vectors in 2 distinct clusters
        let dim = store.get_statistics().embedding_dim;
        let mut all_data = Vec::new();

        // Cluster A: centered around [1, 0, 0, ...]
        for i in 0..50 {
            let mut v = vec![0.0f32; dim];
            v[0] = 1.0 + (i as f32) * 0.001;
            all_data.extend_from_slice(&v);
        }
        // Cluster B: centered around [-1, 0, 0, ...]
        for i in 0..50 {
            let mut v = vec![0.0f32; dim];
            v[0] = -1.0 - (i as f32) * 0.001;
            all_data.extend_from_slice(&v);
        }
        let arr = Array2::from_shape_vec((100, dim), all_data).unwrap();
        assert!(store.add_splat(&arr));
        assert_eq!(store.n_active(), 100);

        // Query near cluster A
        let mut query = vec![0.0f32; dim];
        query[0] = 1.0;
        let q = Array1::from_vec(query);

        // Filter to only cluster A indices (0..50)
        let cluster_a: Vec<usize> = (0..50).collect();
        let results_a = store.find_neighbors_filtered(&q.view(), &cluster_a, 5);
        assert_eq!(results_a.len(), 5);
        // All results should be from cluster A
        for r in &results_a {
            assert!(r.index < 50, "Expected cluster A index, got {}", r.index);
            assert!(r.distance < 0.2, "Distance too high: {}", r.distance);
        }

        // Filter to only cluster B indices (50..100) — should return farther results
        let cluster_b: Vec<usize> = (50..100).collect();
        let results_b = store.find_neighbors_filtered(&q.view(), &cluster_b, 5);
        assert_eq!(results_b.len(), 5);
        // All results from cluster B but distances should be ~2.0+
        for r in &results_b {
            assert!(r.index >= 50, "Expected cluster B index, got {}", r.index);
            assert!(
                r.distance > 1.5,
                "Expected large distance, got {}",
                r.distance
            );
        }

        // Empty candidates
        let empty: Vec<usize> = vec![];
        let results_empty = store.find_neighbors_filtered(&q.view(), &empty, 5);
        assert!(results_empty.is_empty());
    }
}
