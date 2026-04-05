//! Dataset Transformer — data transformations for SplatDB datasets.
//!
//! Handles normalization, splitting, augmentation, format conversion,
//! and Gaussian Splat generation via KMeans clustering.
//! Ported from splatdb Python.

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

/// Gaussian Splat representing a vector cluster.
#[derive(Debug, Clone)]
pub struct GaussianSplat {
    pub mu: Vec<f32>,        // Centroid
    pub alpha: f64,          // Weight
    pub kappa: f64,          // Concentration
    pub n_vectors: usize,    // Count
    pub indices: Vec<usize>, // Original indices
}

/// HRM2 hierarchy node.
#[derive(Debug, Clone)]
pub struct Hrm2Node {
    pub splat: GaussianSplat,
    pub children: Vec<Hrm2Node>,
    pub level: usize,
}

/// Memory tier partition result.
#[derive(Debug, Clone)]
pub struct MemoryPartition {
    pub hot: Vec<usize>,  // Top 20% — vram
    pub warm: Vec<usize>, // 20-50% — ram
    pub cold: Vec<usize>, // 50-100% — ssd
}

/// Transform statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformStats {
    pub original_count: usize,
    pub splat_count: usize,
    pub compression_ratio: f64,
    pub original_size_mb: f64,
    pub compressed_size_mb: f64,
    pub memory_savings_pct: f64,
    pub transform_time_s: f64,
}

/// Dataset split for train/val/test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetSplit {
    pub train: Array2<f32>,
    pub val: Option<Array2<f32>>,
    pub test: Option<Array2<f32>>,
    pub train_labels: Option<Vec<String>>,
    pub val_labels: Option<Vec<String>>,
    pub test_labels: Option<Vec<String>>,
}

/// Transformer configuration.
#[derive(Debug, Clone)]
pub struct TransformConfig {
    pub normalize: bool,
    pub center: bool,
    pub unit_sphere: bool,
    pub augment_noise: f64,
    pub augment_flip: bool,
    pub train_ratio: f64,
    pub val_ratio: f64,
}

impl Default for TransformConfig {
    fn default() -> Self {
        Self {
            normalize: true,
            center: true,
            unit_sphere: false,
            augment_noise: 0.0,
            augment_flip: true,
            train_ratio: 0.8,
            val_ratio: 0.1,
        }
    }
}

/// Dataset transformer for SplatDB.
pub struct DatasetTransformer {
    config: TransformConfig,
    mean: Option<Array1<f32>>,
    std: Option<Array1<f32>>,
}

impl DatasetTransformer {
    /// New.
    pub fn new(config: TransformConfig) -> Self {
        Self {
            config,
            mean: None,
            std: None,
        }
    }

    /// Compute and store normalization statistics from data.
    pub fn fit(&mut self, data: &Array2<f32>) {
        let (n, _d) = data.dim();
        if n == 0 {
            return;
        }

        if self.config.center {
            let mean = data
                .mean_axis(ndarray::Axis(0))
                .expect("mean_axis should succeed when n > 0");
            self.mean = Some(mean);
        }

        if self.config.normalize {
            let centered = match &self.mean {
                Some(m) => data - m,
                None => data.clone(),
            };
            let std = centered.std_axis(ndarray::Axis(0), 0.0);
            // Avoid division by zero
            let std = std.mapv(|v| if v < 1e-8 { 1.0 } else { v });
            self.std = Some(std);
        }
    }

    /// Apply stored transformations to data.
    pub fn transform(&self, data: &Array2<f32>) -> Array2<f32> {
        let mut result = data.clone();

        if let Some(ref mean) = self.mean {
            result = &result - mean;
        }

        if let Some(ref std) = self.std {
            result /= std;
        }

        if self.config.unit_sphere {
            // Normalize each row to unit length
            for mut row in result.rows_mut() {
                let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 1e-8 {
                    row.mapv_inplace(|x| x / norm);
                }
            }
        }

        if self.config.augment_noise > 0.0 {
            use rand::Rng;
            let mut rng = rand::thread_rng();
            for mut row in result.rows_mut() {
                for val in row.iter_mut() {
                    let noise: f32 =
                        rng.gen_range(-self.config.augment_noise..self.config.augment_noise) as f32;
                    *val += noise;
                }
            }
        }

        if self.config.augment_flip {
            use rand::Rng;
            let mut rng = rand::thread_rng();
            for mut row in result.rows_mut() {
                if rng.gen_bool(0.5) {
                    row.mapv_inplace(|x| -x);
                }
            }
        }

        result
    }

    /// Fit and transform in one step.
    pub fn fit_transform(&mut self, data: &Array2<f32>) -> Array2<f32> {
        self.fit(data);
        self.transform(data)
    }

    /// Split dataset into train/val/test.
    pub fn split(&self, data: &Array2<f32>, labels: Option<&[String]>) -> DatasetSplit {
        let n = data.nrows();
        let train_n = (n as f64 * self.config.train_ratio) as usize;
        let val_n = (n as f64 * self.config.val_ratio) as usize;
        let test_n = n - train_n - val_n;

        let train = data.slice(ndarray::s![..train_n, ..]).to_owned();
        let val = if val_n > 0 {
            Some(
                data.slice(ndarray::s![train_n..train_n + val_n, ..])
                    .to_owned(),
            )
        } else {
            None
        };
        let test = if test_n > 0 {
            Some(data.slice(ndarray::s![train_n + val_n.., ..]).to_owned())
        } else {
            None
        };

        let (train_labels, val_labels, test_labels) = match labels {
            Some(l) => {
                let tl = l[..train_n].to_vec();
                let vl = if val_n > 0 {
                    Some(l[train_n..train_n + val_n].to_vec())
                } else {
                    None
                };
                let tsl = if test_n > 0 {
                    Some(l[train_n + val_n..].to_vec())
                } else {
                    None
                };
                (Some(tl), vl, tsl)
            }
            None => (None, None, None),
        };

        DatasetSplit {
            train,
            val,
            test,
            train_labels,
            val_labels,
            test_labels,
        }
    }

    /// Get stored mean.
    pub fn get_mean(&self) -> Option<&Array1<f32>> {
        self.mean.as_ref()
    }

    /// Get stored std.
    pub fn get_std(&self) -> Option<&Array1<f32>> {
        self.std.as_ref()
    }

    /// Convert vectors to Gaussian Splats via flat KMeans.
    /// Returns (splats, hierarchy, partition, stats).
    pub fn to_splats(
        &self,
        data: &Array2<f32>,
        n_clusters: usize,
        min_cluster_size: usize,
        seed: u64,
    ) -> (
        Vec<GaussianSplat>,
        Hrm2Node,
        MemoryPartition,
        TransformStats,
    ) {
        let t0 = std::time::Instant::now();
        let (n, dim) = data.dim();

        let mut splats = if n_clusters <= 1 || n <= 1 {
            let mu = data
                .mean_axis(ndarray::Axis(0))
                .expect("mean_axis should succeed when n > 0")
                .to_vec();
            vec![GaussianSplat {
                mu,
                alpha: 1.0,
                kappa: 10.0,
                n_vectors: n,
                indices: (0..n).collect(),
            }]
        } else {
            let k = n_clusters.min(n);
            self.kmeans_to_splats(data, k, seed)
        };

        // Merge small clusters
        if min_cluster_size > 1 && splats.len() > 1 {
            splats = self.merge_small_clusters(splats, min_cluster_size);
        }

        // Build hierarchy
        let root_mu = data
            .mean_axis(ndarray::Axis(0))
            .expect("mean_axis should succeed when n > 0")
            .to_vec();
        let root_splat = GaussianSplat {
            mu: root_mu,
            alpha: 1.0,
            kappa: 1.0,
            n_vectors: n,
            indices: (0..n).collect(),
        };
        let children: Vec<Hrm2Node> = splats
            .iter()
            .map(|s| Hrm2Node {
                splat: s.clone(),
                children: Vec::new(),
                level: 1,
            })
            .collect();
        let hierarchy = Hrm2Node {
            splat: root_splat,
            children,
            level: 0,
        };

        // Partition
        let partition = self.partition_splats(&splats, data, seed);

        // Stats
        let original_bytes = n * dim * 4;
        let compressed_bytes: usize = splats
            .iter()
            .map(|s| s.mu.len() * 4 + 16 + s.indices.len() * 4)
            .sum();
        let elapsed = t0.elapsed().as_secs_f64();

        let stats = TransformStats {
            original_count: n,
            splat_count: splats.len(),
            compression_ratio: n as f64 / splats.len().max(1) as f64,
            original_size_mb: original_bytes as f64 / (1024.0 * 1024.0),
            compressed_size_mb: compressed_bytes as f64 / (1024.0 * 1024.0),
            memory_savings_pct: if original_bytes > 0 {
                (1.0 - compressed_bytes as f64 / original_bytes as f64) * 100.0
            } else {
                0.0
            },
            transform_time_s: elapsed,
        };

        (splats, hierarchy, partition, stats)
    }

    /// Hierarchical clustering: coarse then fine KMeans.
    pub fn to_splats_hierarchical(
        &self,
        data: &Array2<f32>,
        n_clusters: usize,
        min_cluster_size: usize,
        seed: u64,
    ) -> (
        Vec<GaussianSplat>,
        Hrm2Node,
        MemoryPartition,
        TransformStats,
    ) {
        let t0 = std::time::Instant::now();
        let (n, dim) = data.dim();
        let n_coarse = ((n_clusters as f64).sqrt() as usize).max(10).min(n);
        let n_fine = (n_clusters / n_coarse.max(1)).max(2);

        // Coarse KMeans
        let (_coarse_centers, coarse_labels) = simple_kmeans(data, n_coarse, 20, seed);

        let mut splats = Vec::new();
        for c in 0..n_coarse {
            let members: Vec<usize> = coarse_labels
                .iter()
                .enumerate()
                .filter(|(_, &l)| l == c)
                .map(|(i, _)| i)
                .collect();
            if members.is_empty() {
                continue;
            }

            if members.len() <= n_fine {
                // Too small for sub-clustering
                let mu = vec_mean_indices(data, &members);
                let variance = cluster_variance(data, &members, &mu);
                let kappa = (1.0 / variance).clamp(0.1, 100.0);
                splats.push(GaussianSplat {
                    mu,
                    alpha: members.len() as f64 / n as f64,
                    kappa,
                    n_vectors: members.len(),
                    indices: members,
                });
            } else {
                // Fine KMeans within coarse cluster
                let sub_data = extract_rows(data, &members);
                let (_, fine_labels) =
                    simple_kmeans(&sub_data, n_fine.min(members.len()), 15, seed + c as u64);
                for f in 0..n_fine {
                    let fine_members: Vec<usize> = fine_labels
                        .iter()
                        .enumerate()
                        .filter(|(_, &l)| l == f)
                        .map(|(i, _)| members[i])
                        .collect();
                    if fine_members.is_empty() {
                        continue;
                    }
                    let mu = vec_mean_indices(data, &fine_members);
                    let variance = cluster_variance(data, &fine_members, &mu);
                    let kappa = (1.0 / variance).clamp(0.1, 100.0);
                    splats.push(GaussianSplat {
                        mu,
                        alpha: fine_members.len() as f64 / n as f64,
                        kappa,
                        n_vectors: fine_members.len(),
                        indices: fine_members,
                    });
                }
            }
        }

        if min_cluster_size > 1 && splats.len() > 1 {
            splats = self.merge_small_clusters(splats, min_cluster_size);
        }

        let root_mu = data
            .mean_axis(ndarray::Axis(0))
            .expect("mean_axis should succeed when n > 0")
            .to_vec();
        let root_splat = GaussianSplat {
            mu: root_mu,
            alpha: 1.0,
            kappa: 1.0,
            n_vectors: n,
            indices: (0..n).collect(),
        };
        let children: Vec<Hrm2Node> = splats
            .iter()
            .map(|s| Hrm2Node {
                splat: s.clone(),
                children: Vec::new(),
                level: 1,
            })
            .collect();
        let hierarchy = Hrm2Node {
            splat: root_splat,
            children,
            level: 0,
        };
        let partition = self.partition_splats(&splats, data, seed);

        let original_bytes = n * dim * 4;
        let compressed_bytes: usize = splats
            .iter()
            .map(|s| s.mu.len() * 4 + 16 + s.indices.len() * 4)
            .sum();
        let stats = TransformStats {
            original_count: n,
            splat_count: splats.len(),
            compression_ratio: n as f64 / splats.len().max(1) as f64,
            original_size_mb: original_bytes as f64 / (1024.0 * 1024.0),
            compressed_size_mb: compressed_bytes as f64 / (1024.0 * 1024.0),
            memory_savings_pct: if original_bytes > 0 {
                (1.0 - compressed_bytes as f64 / original_bytes as f64) * 100.0
            } else {
                0.0
            },
            transform_time_s: t0.elapsed().as_secs_f64(),
        };

        (splats, hierarchy, partition, stats)
    }

    /// Convert vectors to Gaussian Splats via Leader Clustering.
    ///
    /// O(n) single-pass algorithm — no iterations, no convergence loops.
    /// Each vector is either assigned to its nearest leader (within threshold)
    /// or becomes a new leader. Centroids are updated incrementally.
    ///
    /// `target_clusters` is a soft target — actual count depends on data density.
    /// `threshold` controls the radius: lower = more clusters, higher = fewer.
    /// When threshold = None, it's auto-computed from data statistics.
    ///
    /// Returns (splats, hierarchy, partition, stats).
    pub fn to_splats_leader(
        &self,
        data: &Array2<f32>,
        target_clusters: usize,
        min_cluster_size: usize,
        seed: u64,
        threshold: Option<f64>,
    ) -> (
        Vec<GaussianSplat>,
        Hrm2Node,
        MemoryPartition,
        TransformStats,
    ) {
        let t0 = std::time::Instant::now();
        let (n, dim) = data.dim();

        // Auto-compute threshold: estimate average nearest-neighbor distance
        let effective_threshold = threshold.unwrap_or_else(|| {
            if n < 2 {
                return 1.0;
            }
            // Sample up to 200 points, compute avg nn distance, scale by target density
            let sample_size = 200.min(n);
            let step = n / sample_size;
            let mut total_nn = 0.0f64;
            let mut count = 0usize;
            for i in (0..n).step_by(step) {
                let row = data.row(i);
                let mut min_d = f64::MAX;
                for j in (0..n).step_by(step) {
                    if i == j {
                        continue;
                    }
                    let d: f64 = row
                        .iter()
                        .zip(data.row(j).iter())
                        .map(|(&a, &b)| {
                            let d = a as f64 - b as f64;
                            d * d
                        })
                        .sum();
                    if d < min_d {
                        min_d = d;
                    }
                }
                if min_d < f64::MAX {
                    total_nn += min_d.sqrt();
                    count += 1;
                }
            }
            let avg_nn = if count > 0 {
                total_nn / count as f64
            } else {
                1.0
            };
            // Scale: we want ~target_clusters, each covering ~n/target vectors
            // Higher target_clusters -> lower threshold -> more leaders
            let density_factor = (n as f64 / target_clusters.max(1) as f64).sqrt() / n as f64;
            let auto_t = avg_nn * density_factor * (target_clusters as f64).sqrt().max(1.0) * 0.8;
            auto_t.clamp(0.01, 10.0)
        });

        let mut splats = leader_cluster(data, effective_threshold, seed);

        // Merge small clusters
        if min_cluster_size > 1 && splats.len() > 1 {
            splats = self.merge_small_clusters(splats, min_cluster_size);
        }

        // Build hierarchy
        let root_mu = data
            .mean_axis(ndarray::Axis(0))
            .expect("mean_axis should succeed when n > 0")
            .to_vec();
        let root_splat = GaussianSplat {
            mu: root_mu,
            alpha: 1.0,
            kappa: 1.0,
            n_vectors: n,
            indices: (0..n).collect(),
        };
        let children: Vec<Hrm2Node> = splats
            .iter()
            .map(|s| Hrm2Node {
                splat: s.clone(),
                children: Vec::new(),
                level: 1,
            })
            .collect();
        let hierarchy = Hrm2Node {
            splat: root_splat,
            children,
            level: 0,
        };

        let partition = self.partition_splats(&splats, data, seed);

        let original_bytes = n * dim * 4;
        let compressed_bytes: usize = splats
            .iter()
            .map(|s| s.mu.len() * 4 + 16 + s.indices.len() * 4)
            .sum();
        let elapsed = t0.elapsed().as_secs_f64();

        let stats = TransformStats {
            original_count: n,
            splat_count: splats.len(),
            compression_ratio: n as f64 / splats.len().max(1) as f64,
            original_size_mb: original_bytes as f64 / (1024.0 * 1024.0),
            compressed_size_mb: compressed_bytes as f64 / (1024.0 * 1024.0),
            memory_savings_pct: if original_bytes > 0 {
                (1.0 - compressed_bytes as f64 / original_bytes as f64) * 100.0
            } else {
                0.0
            },
            transform_time_s: elapsed,
        };

        (splats, hierarchy, partition, stats)
    }

    fn kmeans_to_splats(&self, data: &Array2<f32>, k: usize, seed: u64) -> Vec<GaussianSplat> {
        let (n, _) = data.dim();
        let (_, labels) = simple_kmeans(data, k, 20, seed);

        let mut splats = Vec::new();
        for c in 0..k {
            let members: Vec<usize> = labels
                .iter()
                .enumerate()
                .filter(|(_, &l)| l == c)
                .map(|(i, _)| i)
                .collect();
            if members.is_empty() {
                continue;
            }

            let mu = vec_mean_indices(data, &members);
            let variance = cluster_variance(data, &members, &mu);
            let kappa = (1.0 / variance).clamp(0.1, 100.0);

            splats.push(GaussianSplat {
                mu,
                alpha: members.len() as f64 / n as f64,
                kappa,
                n_vectors: members.len(),
                indices: members,
            });
        }
        splats
    }

    fn merge_small_clusters(
        &self,
        splats: Vec<GaussianSplat>,
        min_size: usize,
    ) -> Vec<GaussianSplat> {
        let mut splats = splats;
        let small: Vec<usize> = splats
            .iter()
            .enumerate()
            .filter(|(_, s)| s.n_vectors < min_size)
            .map(|(i, _)| i)
            .collect();

        let mut to_remove = std::collections::HashSet::new();
        for si in small {
            if to_remove.contains(&si) {
                continue;
            }
            // Find nearest larger cluster
            let mut best_j = None;
            let mut best_dist = f64::MAX;
            for (j, s) in splats.iter().enumerate() {
                if j == si || to_remove.contains(&j) || s.n_vectors < min_size {
                    continue;
                }
                let dist = euclidean(&splats[si].mu, &s.mu);
                if dist < best_dist {
                    best_dist = dist;
                    best_j = Some(j);
                }
            }
            if let Some(j) = best_j {
                let si_indices = splats[si].indices.clone();
                let si_mu = splats[si].mu.clone();
                let si_n = splats[si].n_vectors;
                let si_alpha = splats[si].alpha;
                let total = splats[j].n_vectors + si_n;
                // Weighted centroid merge
                #[allow(clippy::needless_range_loop)]
                for k in 0..splats[j].mu.len() {
                    splats[j].mu[k] = (splats[j].mu[k] * splats[j].n_vectors as f32
                        + si_mu[k] * si_n as f32)
                        / total as f32;
                }
                splats[j].alpha += si_alpha;
                splats[j].indices.extend_from_slice(&si_indices);
                splats[j].n_vectors = total;
                to_remove.insert(si);
            }
        }
        splats
            .into_iter()
            .enumerate()
            .filter(|(i, _)| !to_remove.contains(i))
            .map(|(_, s)| s)
            .collect()
    }

    fn partition_splats(
        &self,
        splats: &[GaussianSplat],
        _data: &Array2<f32>,
        _seed: u64,
    ) -> MemoryPartition {
        if splats.is_empty() {
            return MemoryPartition {
                hot: Vec::new(),
                warm: Vec::new(),
                cold: Vec::new(),
            };
        }
        let n = splats.len();
        let mut scored: Vec<(usize, f64)> = splats
            .iter()
            .enumerate()
            .map(|(i, s)| {
                // Score = weighted combination of access, size, concentration
                let score = 0.4 * s.alpha + 0.3 * s.n_vectors as f64 + 0.3 * s.kappa;
                (i, score)
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let hot_end = (n as f64 * 0.2).ceil() as usize;
        let warm_end = (n as f64 * 0.5).ceil() as usize;

        MemoryPartition {
            hot: scored[..hot_end].iter().map(|(i, _)| *i).collect(),
            warm: scored[hot_end..warm_end].iter().map(|(i, _)| *i).collect(),
            cold: scored[warm_end..].iter().map(|(i, _)| *i).collect(),
        }
    }
}

// ─── Leader Clustering — O(n) single-pass, no iterations ───

/// Leader Clustering algorithm.
///
/// Single-pass O(n): each vector is assigned to nearest leader within `threshold`,
/// or becomes a new leader if no match found. Centroids updated incrementally.
///
/// This replaces KMeans for the memory ingestion pipeline — no convergence loops,
/// no iteration count tuning, deterministic with seed.
fn leader_cluster(data: &Array2<f32>, threshold: f64, seed: u64) -> Vec<GaussianSplat> {
    let (n, dim) = data.dim();
    if n == 0 {
        return Vec::new();
    }

    let threshold_sq = threshold * threshold;
    let mut leaders: Vec<LeaderState> = Vec::new();
    let mut rng_state = seed;

    // Shuffle-assisted pass: process in a seeded random order for better quality
    let mut order: Vec<usize> = (0..n).collect();
    // Fisher-Yates shuffle with deterministic seed
    for i in (1..n).rev() {
        rng_state = rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let j = (rng_state as usize) % (i + 1);
        order.swap(i, j);
    }

    for &idx in &order {
        let row = data.row(idx);

        // Find nearest leader within threshold
        let mut best_l = None;
        let mut best_dist_sq = f64::MAX;

        for (li, leader) in leaders.iter().enumerate() {
            let dist_sq: f64 = row
                .iter()
                .zip(leader.centroid.iter())
                .map(|(&a, &b)| {
                    let d = a as f64 - b as f64;
                    d * d
                })
                .sum();
            if dist_sq < best_dist_sq {
                best_dist_sq = dist_sq;
                best_l = Some(li);
            }
        }

        if let Some(li) = best_l {
            if best_dist_sq <= threshold_sq {
                // Assign to existing leader — incremental centroid update
                let count = leaders[li].count as f64;
                let inv = 1.0 / (count + 1.0);
                for j in 0..dim {
                    let old = leaders[li].centroid[j] as f64;
                    leaders[li].centroid[j] = (((old * count) + row[j] as f64) * inv) as f32;
                }
                leaders[li].count += 1;
                leaders[li].indices.push(idx);
                leaders[li].sum_dist += best_dist_sq.sqrt();
                continue;
            }
        }

        // No match within threshold — create new leader
        leaders.push(LeaderState {
            centroid: row.to_vec(),
            count: 1,
            indices: vec![idx],
            sum_dist: 0.0,
        });
    }

    // Convert to GaussianSplats
    let total = n as f64;
    leaders
        .into_iter()
        .map(|l| {
            let avg_dist = if l.count > 1 {
                l.sum_dist / l.count as f64
            } else {
                1.0
            };
            let variance = avg_dist + 1e-8;
            GaussianSplat {
                mu: l.centroid,
                alpha: l.count as f64 / total,
                kappa: (1.0 / variance).clamp(0.1, 100.0),
                n_vectors: l.count,
                indices: l.indices,
            }
        })
        .collect()
}

/// Internal state for leader clustering.
struct LeaderState {
    centroid: Vec<f32>,
    count: usize,
    indices: Vec<usize>,
    sum_dist: f64,
}

// ─── Helper functions ───

fn simple_kmeans(
    data: &Array2<f32>,
    k: usize,
    max_iter: usize,
    seed: u64,
) -> (Vec<Vec<f32>>, Vec<usize>) {
    let (n, dim) = data.dim();
    let k = k.min(n);
    if k == 0 || dim == 0 {
        return (Vec::new(), vec![0; n]);
    }

    let mut rng_state = seed;
    let mut centroids: Vec<Vec<f32>> = (0..k)
        .map(|_| {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let idx = (rng_state as usize) % n;
            data.row(idx).to_vec()
        })
        .collect();

    let mut labels = vec![0usize; n];

    for _ in 0..max_iter {
        let mut changed = false;
        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            let row = data.row(i);
            let mut best_c = 0;
            let mut best_dist = f64::MAX;
            for (c, center) in centroids.iter().enumerate() {
                let dist: f64 = row
                    .iter()
                    .zip(center.iter())
                    .map(|(&x, &y)| {
                        let d = x as f64 - y as f64;
                        d * d
                    })
                    .sum();
                if dist < best_dist {
                    best_dist = dist;
                    best_c = c;
                }
            }
            if labels[i] != best_c {
                labels[i] = best_c;
                changed = true;
            }
        }
        if !changed {
            break;
        }

        let mut counts = vec![0usize; k];
        let mut sums = vec![vec![0.0f64; dim]; k];
        for i in 0..n {
            let c = labels[i];
            counts[c] += 1;
            for j in 0..dim {
                sums[c][j] += data[[i, j]] as f64;
            }
        }
        for c in 0..k {
            if counts[c] > 0 {
                centroids[c] = sums[c]
                    .iter()
                    .map(|s| (s / counts[c] as f64) as f32)
                    .collect();
            }
        }
    }
    (centroids, labels)
}

fn vec_mean_indices(data: &Array2<f32>, indices: &[usize]) -> Vec<f32> {
    let dim = data.ncols();
    let n = indices.len() as f32;
    let mut sum = vec![0.0f32; dim];
    for &i in indices {
        for j in 0..dim {
            sum[j] += data[[i, j]];
        }
    }
    sum.iter().map(|s| s / n).collect()
}

fn cluster_variance(data: &Array2<f32>, indices: &[usize], center: &[f32]) -> f64 {
    let dim = data.ncols();
    let mut total_dist = 0.0f64;
    for &i in indices {
        for j in 0..dim {
            let d = data[[i, j]] as f64 - center[j] as f64;
            total_dist += d * d;
        }
    }
    (total_dist.sqrt() / indices.len().max(1) as f64) + 1e-8
}

fn extract_rows(data: &Array2<f32>, indices: &[usize]) -> Array2<f32> {
    let dim = data.ncols();
    let flat: Vec<f32> = indices.iter().flat_map(|&i| data.row(i).to_vec()).collect();
    Array2::from_shape_vec((indices.len(), dim), flat).unwrap_or_else(|_| Array2::zeros((0, dim)))
}

fn euclidean(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = x as f64 - y as f64;
            d * d
        })
        .sum::<f64>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_data() -> Array2<f32> {
        Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap()
    }

    #[test]
    fn test_fit_transform() {
        let config = TransformConfig::default();
        let mut transformer = DatasetTransformer::new(config);
        let data = make_data();
        let transformed = transformer.fit_transform(&data);
        assert_eq!(transformed.dim(), (4, 2));
        assert!(transformer.get_mean().is_some());
        assert!(transformer.get_std().is_some());
    }

    #[test]
    fn test_unit_sphere() {
        let config = TransformConfig {
            unit_sphere: true,
            center: false,
            normalize: false,
            ..Default::default()
        };
        let mut transformer = DatasetTransformer::new(config);
        let data = make_data();
        let transformed = transformer.fit_transform(&data);
        // Check each row is unit length
        for i in 0..transformed.nrows() {
            let row = transformed.row(i);
            let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-5, "Row {} norm = {}", i, norm);
        }
    }

    #[test]
    fn test_split() {
        let config = TransformConfig {
            train_ratio: 0.5,
            val_ratio: 0.25,
            ..Default::default()
        };
        let transformer = DatasetTransformer::new(config);
        let data = make_data();
        let split = transformer.split(&data, None);
        assert_eq!(split.train.nrows(), 2);
        assert!(split.val.is_some());
        assert!(split.test.is_some());
    }

    #[test]
    fn test_no_center() {
        let config = TransformConfig {
            center: false,
            normalize: false,
            augment_flip: false,
            ..Default::default()
        };
        let mut transformer = DatasetTransformer::new(config);
        let data = make_data();
        let transformed = transformer.fit_transform(&data);
        assert_eq!(&transformed, &data);
    }

    #[test]
    fn test_to_splats() {
        let config = TransformConfig {
            augment_flip: false,
            ..Default::default()
        };
        let transformer = DatasetTransformer::new(config);
        let data = Array2::from_shape_vec((20, 3), (0..60).map(|i| i as f32).collect()).unwrap();
        let (splats, hierarchy, partition, stats) = transformer.to_splats(&data, 4, 1, 42);
        assert!(!splats.is_empty());
        assert!(splats.len() <= 4);
        assert_eq!(hierarchy.level, 0);
        assert!(!hierarchy.children.is_empty());
        assert!(stats.compression_ratio > 0.0);
        assert!(stats.original_count == 20);
        assert!(!partition.hot.is_empty() || !partition.warm.is_empty());
    }

    #[test]
    fn test_hierarchical_splats() {
        let config = TransformConfig {
            augment_flip: false,
            ..Default::default()
        };
        let transformer = DatasetTransformer::new(config);
        let data =
            Array2::from_shape_vec((50, 4), (0..200).map(|i| (i as f32 * 0.1).sin()).collect())
                .unwrap();
        let (splats, _hierarchy, _partition, stats) =
            transformer.to_splats_hierarchical(&data, 16, 2, 42);
        assert!(!splats.is_empty());
        assert!(stats.splat_count > 0);
    }

    #[test]
    fn test_leader_clustering_basic() {
        let data = Array2::from_shape_vec(
            (100, 4),
            (0..400).map(|i| (i as f32 * 0.05).cos()).collect(),
        )
        .unwrap();
        let splats = leader_cluster(&data, 0.5, 42);
        assert!(!splats.is_empty());
        // Each splat should have at least 1 member
        for s in &splats {
            assert!(s.n_vectors >= 1);
            assert_eq!(s.indices.len(), s.n_vectors);
        }
        // Total members should equal total vectors
        let total: usize = splats.iter().map(|s| s.n_vectors).sum();
        assert_eq!(total, 100);
    }

    #[test]
    fn test_leader_clustering_deterministic() {
        let data = Array2::from_shape_vec((50, 3), (0..150).map(|i| i as f32).collect()).unwrap();
        let splats1 = leader_cluster(&data, 0.5, 42);
        let splats2 = leader_cluster(&data, 0.5, 42);
        assert_eq!(splats1.len(), splats2.len());
        for (a, b) in splats1.iter().zip(splats2.iter()) {
            assert_eq!(a.n_vectors, b.n_vectors);
        }
    }

    #[test]
    fn test_leader_to_splats() {
        let config = TransformConfig {
            augment_flip: false,
            ..Default::default()
        };
        let transformer = DatasetTransformer::new(config);
        let data =
            Array2::from_shape_vec((60, 4), (0..240).map(|i| (i as f32 * 0.1).sin()).collect())
                .unwrap();
        let (splats, _hierarchy, _partition, stats) =
            transformer.to_splats_leader(&data, 10, 1, 42, Some(0.3));
        assert!(!splats.is_empty());
        assert!(stats.compression_ratio > 1.0);
        assert_eq!(stats.original_count, 60);
        // All vectors accounted for
        let total: usize = splats.iter().map(|s| s.n_vectors).sum();
        assert_eq!(total, 60);
    }

    #[test]
    fn test_leader_auto_threshold() {
        let config = TransformConfig {
            augment_flip: false,
            ..Default::default()
        };
        let transformer = DatasetTransformer::new(config);
        // 3 clear clusters
        let data = Array2::from_shape_vec((90, 4), {
            let mut v = Vec::new();
            for _ in 0..30 {
                v.extend_from_slice(&[1.0, 0.0, 0.0, 0.0]);
            }
            for _ in 0..30 {
                v.extend_from_slice(&[0.0, 1.0, 0.0, 0.0]);
            }
            for _ in 0..30 {
                v.extend_from_slice(&[0.0, 0.0, 1.0, 0.0]);
            }
            v
        })
        .unwrap();
        let (splats, _, _, _stats) = transformer.to_splats_leader(&data, 3, 1, 42, None);
        assert!(!splats.is_empty());
        assert!(splats.len() <= 10); // auto-threshold shouldn't explode
        let total: usize = splats.iter().map(|s| s.n_vectors).sum();
        assert_eq!(total, 90);
    }
}
