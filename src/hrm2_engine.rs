//! HRM2 Engine — Hierarchical Retrieval Model 2.
//! Two-level hierarchical index for fast similarity search.

use std::collections::HashMap;
use std::time::Instant;

use ndarray::{Array1, Array2, ArrayView1, Axis};

use crate::clustering::KMeans;
use crate::encoding::FullEmbeddingBuilder;
use crate::splat_types::GaussianSplat;

/// Result of a similarity search.
#[derive(Clone, Debug)]
pub struct SearchResult {
    pub splat_id: usize,
    pub distance: f32,
    pub coarse_cluster: usize,
    pub fine_cluster: usize,
}

/// Statistics for HRM2 Engine.
#[derive(Clone, Debug, Default)]
pub struct HRM2Stats {
    pub n_splats: usize,
    pub n_coarse_clusters: usize,
    pub n_fine_clusters: usize,
    pub build_time_ms: f64,
    pub avg_query_time_us: f64,
    pub total_queries: usize,
}

/// HRM2 Engine — two-level hierarchical retrieval.
pub struct HRM2Engine {
    pub n_coarse: usize,
    pub n_fine: usize,
    pub embedding_dim: usize,
    pub n_probe: usize,
    pub batch_size: usize,
    pub metric: String,

    splats: Vec<GaussianSplat>,
    embeddings: Option<Array2<f32>>,

    coarse_model: Option<KMeans>,
    coarse_assignments: Option<Array1<usize>>,
    fine_models: HashMap<usize, KMeans>,
    fine_assignments: HashMap<usize, Array1<usize>>,

    cluster_indices: HashMap<usize, Vec<usize>>,
    cluster_embeddings: HashMap<usize, Array2<f32>>,
    cluster_masks: HashMap<usize, Vec<bool>>,

    encoder: FullEmbeddingBuilder,
    is_indexed: bool,
    stats: HRM2Stats,
}

impl HRM2Engine {
    /// Create a new HRM2 engine with the given clustering parameters.
    pub fn new(
        n_coarse: usize,
        n_fine: usize,
        embedding_dim: usize,
        n_probe: usize,
        metric: &str,
    ) -> Self {
        Self {
            n_coarse,
            n_fine,
            embedding_dim,
            n_probe,
            batch_size: 10000,
            metric: metric.to_string(),
            splats: Vec::new(),
            embeddings: None,
            coarse_model: None,
            coarse_assignments: None,
            fine_models: HashMap::new(),
            fine_assignments: HashMap::new(),
            cluster_indices: HashMap::new(),
            cluster_embeddings: HashMap::new(),
            cluster_masks: HashMap::new(),
            encoder: FullEmbeddingBuilder::new(),
            is_indexed: false,
            stats: HRM2Stats::default(),
        }
    }

    /// Add Gaussian splats to the engine. Invalidates the index.
    pub fn add_splats(&mut self, splats: Vec<GaussianSplat>) {
        self.splats.extend(splats);
        self.is_indexed = false;
    }

    /// Build the hierarchical index. Optionally accept precomputed embeddings.
    pub fn index(&mut self, precomputed: Option<Array2<f32>>) -> f64 {
        let start = Instant::now();

        let n_samples = if let Some(embs) = precomputed {
            let mut e = embs;
            if self.metric == "cosine" {
                Self::normalize_rows(&mut e);
            }
            self.embeddings = Some(e);
            self.embeddings
                .as_ref()
                .expect("embeddings just set above")
                .nrows()
        } else if !self.splats.is_empty() {
            let n = self.splats.len();
            let dim = 3;
            let mut pos_data = Vec::with_capacity(n * dim);
            let mut col_data = Vec::with_capacity(n * dim);
            let mut sca_data = Vec::with_capacity(n * dim);
            let mut rot_data = Vec::with_capacity(n * 4);
            let mut opa_data = Vec::with_capacity(n);
            for s in &self.splats {
                pos_data.extend_from_slice(&s.position);
                col_data.extend_from_slice(&s.color);
                sca_data.extend_from_slice(&s.scale);
                rot_data.extend_from_slice(&s.rotation);
                opa_data.push(s.opacity);
            }
            let positions =
                Array2::from_shape_vec((n, dim), pos_data).expect("position shape mismatch");
            let colors = Array2::from_shape_vec((n, dim), col_data).expect("color shape mismatch");
            let opacities = Array1::from(opa_data);
            let scales = Array2::from_shape_vec((n, dim), sca_data).expect("scale shape mismatch");
            let rotations =
                Array2::from_shape_vec((n, 4), rot_data).expect("rotation shape mismatch");

            let mut e = self
                .encoder
                .build(&positions, &colors, &opacities, &scales, &rotations);
            if self.metric == "cosine" {
                Self::normalize_rows(&mut e);
            }
            self.embeddings = Some(e);
            self.embeddings
                .as_ref()
                .expect("embeddings just set above")
                .nrows()
        } else {
            return 0.0;
        };

        let embeddings = self
            .embeddings
            .as_ref()
            .expect("embeddings must exist after index()");

        // Level 1: Coarse clustering
        let n_coarse = self.n_coarse.min(n_samples / 10).max(1);
        let mut coarse = KMeans::new(n_coarse, 100, 42);
        coarse.fit(embeddings);
        let coarse_labels = coarse
            .labels
            .clone()
            .expect("coarse labels must exist after fit");

        self.coarse_model = Some(coarse);
        self.coarse_assignments = Some(coarse_labels.clone());

        // Level 2: Fine clustering per coarse cluster
        self.fine_models.clear();
        self.fine_assignments.clear();
        self.cluster_indices.clear();
        self.cluster_embeddings.clear();
        self.cluster_masks.clear();

        for c in 0..n_coarse {
            let indices: Vec<usize> = coarse_labels
                .iter()
                .enumerate()
                .filter(|(_, &label)| label == c)
                .map(|(i, _)| i)
                .collect();

            self.cluster_indices.insert(c, indices.clone());

            if indices.len() < 2 {
                self.fine_models.insert(c, KMeans::new(1, 10, 42));
                self.fine_assignments
                    .insert(c, Array1::zeros(indices.len()));
                self.cluster_embeddings
                    .insert(c, Array2::zeros((0, self.embedding_dim)));
                continue;
            }

            // Gather cluster embeddings
            let mut cluster_embs = Array2::zeros((indices.len(), self.embedding_dim));
            for (j, &idx) in indices.iter().enumerate() {
                cluster_embs.row_mut(j).assign(&embeddings.row(idx));
            }
            self.cluster_embeddings.insert(c, cluster_embs.clone());

            let n_fine = self.n_fine.min(indices.len() / 5).max(1);
            let mut fine = KMeans::new(n_fine, 50, 42 + c as u64);
            fine.fit(&cluster_embs);
            self.fine_models.insert(c, fine.clone());
            self.fine_assignments.insert(
                c,
                fine.labels
                    .clone()
                    .expect("fine labels must exist after fit"),
            );
        }

        self.is_indexed = true;
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;

        self.stats.n_splats = n_samples;
        self.stats.n_coarse_clusters = n_coarse;
        self.stats.n_fine_clusters = self.fine_models.values().map(|m| m.n_clusters).sum();
        self.stats.build_time_ms = elapsed;

        elapsed
    }

    /// Query for k nearest neighbors at given level of detail.
    pub fn query(
        &mut self,
        query: &ArrayView1<f32>,
        k: usize,
        lod: usize,
    ) -> anyhow::Result<Vec<(usize, f32)>> {
        if !self.is_indexed {
            return Err(anyhow::anyhow!("Index not built. Call index() first."));
        }

        let start = Instant::now();
        let embeddings = self
            .embeddings
            .as_ref()
            .expect("embeddings must exist after index()");
        let coarse = self
            .coarse_model
            .as_ref()
            .expect("coarse model must exist after index()");

        let q = if self.metric == "cosine" {
            let norm = query.dot(query).sqrt().max(1e-10);
            query.mapv(|v| v / norm)
        } else {
            query.to_owned()
        };

        // Find nearest coarse clusters
        let q_2d = q.view().insert_axis(Axis(0));
        let coarse_dists = coarse.transform(&q_2d);
        let coarse_dists = coarse_dists.row(0);
        let mut coarse_order: Vec<usize> = (0..coarse_dists.len()).collect();
        coarse_order.sort_by(|a, b| {
            coarse_dists[*a]
                .partial_cmp(&coarse_dists[*b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let probe_count = self.n_probe.min(coarse_order.len());
        let closest: Vec<usize> = coarse_order.iter().take(probe_count).copied().collect();

        let mut candidates: Vec<(usize, f32)> = Vec::new();

        match lod {
            0 => {
                // LOD 0: Coarse approximation
                for &c in &closest {
                    if let Some(indices) = self.cluster_indices.get(&c) {
                        for &idx in indices.iter().take(k) {
                            let dist = Self::l2_distance(embeddings.row(idx), &q);
                            candidates.push((idx, dist));
                        }
                    }
                    if candidates.len() >= k {
                        break;
                    }
                }
            }
            1 => {
                // LOD 1: Fine approximation
                for &c in &closest {
                    let fine = match self.fine_models.get(&c) {
                        Some(f) if f.centroids.is_some() => f,
                        _ => continue,
                    };
                    let q_fine = q.view().insert_axis(Axis(0));
                    let fine_dists = fine.transform(&q_fine);
                    let fine_dists = fine_dists.row(0);
                    let best_fine = (0..fine_dists.len())
                        .min_by(|a, b| {
                            fine_dists[*a]
                                .partial_cmp(&fine_dists[*b])
                                .unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .expect("fine_dists must not be empty");

                    let indices = self
                        .cluster_indices
                        .get(&c)
                        .map(|v| v.as_slice())
                        .unwrap_or(&[]);
                    let fine_assigns: &[usize] = match self.fine_assignments.get(&c) {
                        Some(a) => a.as_slice().unwrap_or(&[]),
                        None => &[],
                    };

                    for (local_i, &global_i) in indices.iter().enumerate() {
                        if local_i < fine_assigns.len() && fine_assigns[local_i] == best_fine {
                            let dist = Self::l2_distance(embeddings.row(global_i), &q);
                            candidates.push((global_i, dist));
                        }
                    }
                    if candidates.len() >= k {
                        break;
                    }
                }
            }
            _ => {
                // LOD 2: Exact search within probed clusters
                let mut all_indices: Vec<usize> = Vec::new();
                for &c in &closest {
                    if let Some(indices) = self.cluster_indices.get(&c) {
                        all_indices.extend_from_slice(indices);
                    }
                }

                // Vectorized distance computation
                for &idx in &all_indices {
                    let dist = Self::l2_distance(embeddings.row(idx), &q);
                    candidates.push((idx, dist));
                }
            }
        }

        // Sort and take top-k
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(k);

        // Update stats
        self.stats.total_queries += 1;
        let query_us = start.elapsed().as_micros() as f64;
        let n = self.stats.total_queries;
        self.stats.avg_query_time_us =
            self.stats.avg_query_time_us * ((n - 1) as f64) / (n as f64) + query_us / (n as f64);

        Ok(candidates)
    }

    /// Batch query for multiple queries.
    pub fn query_batch(
        &mut self,
        queries: &Array2<f32>,
        k: usize,
        lod: usize,
    ) -> anyhow::Result<Vec<Vec<(usize, f32)>>> {
        let n = queries.nrows();
        (0..n)
            .map(|i| self.query(&queries.row(i), k, lod))
            .collect()
    }

    /// Get engine statistics.
    pub fn get_stats(&self) -> &HRM2Stats {
        &self.stats
    }

    /// Check if the index has been built.
    pub fn is_indexed(&self) -> bool {
        self.is_indexed
    }

    /// Reset the engine, removing all data and index state.
    pub fn clear(&mut self) {
        self.splats.clear();
        self.embeddings = None;
        self.coarse_model = None;
        self.coarse_assignments = None;
        self.fine_models.clear();
        self.fine_assignments.clear();
        self.cluster_indices.clear();
        self.cluster_embeddings.clear();
        self.cluster_masks.clear();
        self.is_indexed = false;
        self.stats = HRM2Stats::default();
    }

    fn l2_distance(a: ndarray::ArrayView1<f32>, b: &ndarray::Array1<f32>) -> f32 {
        let diff = &a - b;
        diff.dot(&diff).sqrt()
    }

    fn normalize_rows(arr: &mut Array2<f32>) {
        for mut row in arr.rows_mut() {
            let norm = row.dot(&row).sqrt().max(1e-10);
            row.mapv_inplace(|v| v / norm);
        }
    }
}

/// Generate synthetic splats for testing.
pub fn generate_test_splats(n: usize, seed: u64) -> Vec<GaussianSplat> {
    use rand::Rng;
    use rand::SeedableRng;
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    (0..n)
        .map(|i| {
            let mut position = [0.0f32; 3];
            let mut color = [0.0f32; 3];
            let mut scale = [0.0f32; 3];
            let mut rotation = [0.0f32; 4];
            for p in position.iter_mut() {
                *p = rng.gen::<f32>() * 10.0 - 5.0;
            }
            for c in color.iter_mut() {
                *c = rng.gen::<f32>();
            }
            for s in scale.iter_mut() {
                *s = (-rng.gen::<f32>() * 2.0).exp();
            }
            for r in rotation.iter_mut() {
                *r = rng.gen::<f32>();
            }
            let norm: f32 = rotation
                .iter()
                .map(|x| x * x)
                .sum::<f32>()
                .sqrt()
                .max(1e-10);
            for r in rotation.iter_mut() {
                *r /= norm;
            }

            GaussianSplat {
                id: i as u64,
                position,
                color,
                opacity: rng.gen::<f32>(),
                scale,
                rotation,
            }
        })
        .collect()
}
