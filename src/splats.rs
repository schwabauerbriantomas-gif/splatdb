//! SplatStore - Main API for SplatDB Vector Search.
//! Ported from Python splats.py. CPU-only.

use crate::dataset_transformer;
use crate::interfaces::VectorIndex;
use ndarray::{Array1, Array2, ArrayView1};

use crate::config::SplatDBConfig;
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
    config: SplatDBConfig,
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
    lsh: Option<crate::lsh_index::CrossPolytopeLSH>,
    // Semantic memory is a separate layer (it contains its own SplatStore internally,
    // so we can't embed it here without creating a cycle). Use has_semantic_memory()
    // to check if the preset enables it, then manage SemanticMemoryDB externally.
}

impl SplatStore {
    /// New.
    pub fn new(config: SplatDBConfig) -> Self {
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
                    crate::config::QuantAlgorithm::TurboQuant => crate::quantization::QuantAlgorithm::TurboQuant,
                    crate::config::QuantAlgorithm::PolarQuant => crate::quantization::QuantAlgorithm::PolarQuant,
                    crate::config::QuantAlgorithm::None => crate::quantization::QuantAlgorithm::TurboQuant,
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
                "cosine",
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
    pub fn build_index(&mut self) {
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
        }

        // LSH index
        if let Some(ref mut lsh) = self.lsh {
            lsh.index(embeddings);
        }
    }

    /// Find k nearest neighbors for a single query.
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
            .map(|row| row.iter().zip(query.iter()).map(|(a, b)| { let d = a - b; d * d }).sum())
            .collect();

        // Partial sort top-k
        let mut indexed: Vec<(f32, usize)> = dists_sq.iter().enumerate().map(|(i, &d)| (d, i)).collect();
        if k < indexed.len() {
            indexed.select_nth_unstable_by(k, |a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
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
    /// Fuses results from:
    /// - Linear scan (primary, always)
    /// - Quantized search (if enable_quantization)
    /// - HNSW search (if enable_hnsw)
    /// - LSH search (if enable_lsh)
    ///
    /// Returns fused results ranked by distance.
    pub fn find_neighbors_fused(&self, query: &ArrayView1<f32>, k: usize) -> Vec<NeighborResult> {
        let n = self.n_active;
        if n == 0 || k == 0 {
            return Vec::new();
        }
        let k = k.min(n);

        // Collect candidate sets from each enabled backend
        let mut candidates: std::collections::HashMap<usize, f32> = std::collections::HashMap::new();

        // 1. Primary linear scan (always)
        let index_data = self.mu.slice(ndarray::s![..n, ..]);
        for i in 0..n {
            let dist: f32 = index_data.row(i).iter().zip(query.iter())
                .map(|(a, b)| { let d = a - b; d * d }).sum::<f32>().sqrt();
            candidates.insert(i, dist);
        }

        // 2. Quantized search (pre-filter candidates)
        if let Some(ref qs) = self.quant_store {
            let qresults = qs.search(query, k * 2);
            for (id, dist) in qresults {
                let idx = id as usize;
                if idx < n {
                    candidates.entry(idx).and_modify(|d| *d = (*d).min(dist)).or_insert(dist);
                }
            }
        }

        // 3. HNSW search
        if let Some(ref hnsw) = self.hnsw {
            let hresults = hnsw.search(*query, k * 2);
            for (idx, dist) in hresults.indices.iter().zip(hresults.distances.iter()) {
                if *idx < n {
                    candidates.entry(*idx).and_modify(|d| *d = (*d).min(*dist)).or_insert(*dist);
                }
            }
        }

        // 4. LSH search
        if let Some(ref lsh) = self.lsh {
            let (indices, dists) = lsh.query(query, k * 2);
            for (idx, dist) in indices.iter().zip(dists.iter()) {
                if *idx < n {
                    candidates.entry(*idx).and_modify(|d| *d = (*d).min(*dist)).or_insert(*dist);
                }
            }
        }

        // Sort by distance and return top-k
        let mut ranked: Vec<(usize, f32)> = candidates.into_iter().collect();
        if k < ranked.len() {
            ranked.select_nth_unstable_by(k, |a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        }
        ranked[..k].sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        ranked[..k]
            .iter()
            .map(|&(idx, dist)| NeighborResult {
                mu: index_data.row(idx).to_vec(),
                alpha: self.alpha[idx],
                kappa: self.kappa[idx],
                index: idx,
                distance: dist,
            })
            .collect()
    }

    /// Check if quantization subsystem is active.
    pub fn has_quantization(&self) -> bool { self.quant_store.is_some() }
    /// Check if HNSW index is active.
    pub fn has_hnsw(&self) -> bool { self.hnsw.is_some() }
    /// Check if LSH index is active.
    pub fn has_lsh(&self) -> bool { self.lsh.is_some() }
    /// Check if semantic memory is enabled in this preset.
    /// Note: SemanticMemoryDB must be managed externally (it wraps its own SplatStore).
    pub fn has_semantic_memory(&self) -> bool { self.config.enable_semantic_memory }

    /// Batch find neighbors for multiple queries.
    pub fn find_neighbors_batch(&self, queries: &Array2<f32>, k: usize) -> Vec<Vec<NeighborResult>> {
        (0..queries.nrows()).map(|i| self.find_neighbors(&queries.row(i), k)).collect()
    }

    /// Find k nearest neighbors using HRM2 engine (fast path).
    ///
    /// Uses the HRM2 hierarchical index with LOD 2 (exact within probed clusters)
    /// for approximate search. Falls back to linear scan if:
    /// - Dataset has fewer than 100 active vectors (not worth the overhead)
    /// - HRM2 engine is not indexed
    /// - HRM2 query returns an error
    pub fn find_neighbors_fast(&mut self, query: &ArrayView1<f32>, k: usize) -> Vec<NeighborResult> {
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
        if k < 1 { return Err("k must be >= 1".into()); }
        if query.len() != self.config.latent_dim {
            return Err(format!("Query dimension mismatch: expected {}, got {}", self.config.latent_dim, query.len()));
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
        if self.n_active == 0 { return None; }
        Some(self.frequency.slice(ndarray::s![..self.n_active]).to_owned())
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
        let h: f32 = kappa.iter().filter(|&&k| k > 0.0).map(|&k| { let p = k / total; -p * p.ln() }).sum();
        let max_h = (self.n_active as f32).ln();
        if max_h <= 0.0 { 0.0 } else { (h / max_h).clamp(0.0, 1.0) }
    }

    /// Remove invalid splats (alpha ~0, NaN/Inf in mu).
    pub fn compact(&mut self) {
        if self.n_active == 0 { return; }
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
    pub fn n_active(&self) -> usize { self.n_active }
    /// Maximum splat capacity.
    pub fn max_splats(&self) -> usize { self.max_splats }

    /// Get active mu vectors (n_active × dim slice).
    pub fn get_mu(&self) -> Option<Array2<f32>> {
        if self.n_active == 0 { return None; }
        Some(self.mu.slice(ndarray::s![..self.n_active, ..]).to_owned())
    }

    /// Get active alpha values.
    pub fn get_alpha(&self) -> Option<Array1<f32>> {
        if self.n_active == 0 { return None; }
        Some(self.alpha.slice(ndarray::s![..self.n_active]).to_owned())
    }

    /// Get active kappa values.
    pub fn get_kappa(&self) -> Option<Array1<f32>> {
        if self.n_active == 0 { return None; }
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
        let mu_arr = Array2::from_shape_vec(
            (n, dim),
            splats.iter().flat_map(|s| s.mu.clone()).collect(),
        )
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
        let mu_arr = Array2::from_shape_vec(
            (n, dim),
            splats.iter().flat_map(|s| s.mu.clone()).collect(),
        )
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
        let mu_arr = Array2::from_shape_vec(
            (n, dim),
            splats.iter().flat_map(|s| s.mu.clone()).collect(),
        ).expect("mu shape should match (n, dim)");
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
mod tests {
    use super::*;

    fn test_config() -> SplatDBConfig {
        let mut c = SplatDBConfig::default();
        c.max_splats = 1000;
        c.latent_dim = 64;
        c
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
        assert!(results[0].distance < 0.01, "First neighbor should be near-identical, got dist={}", results[0].distance);
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
        assert!(e > 0.0 && e <= 1.0, "Entropy should be in (0, 1], got {}", e);
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
}
