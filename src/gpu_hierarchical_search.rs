//! Hierarchical GPU Search — Two-stage ANN search.
//! Stage 1 (Coarse): Query vs cluster centroids.
//! Stage 2 (Fine): Query vs candidate cluster members.
//! Ported from splatdb Python.

use ndarray::{Array1, Array2, ArrayView2};

use crate::cuda_search::{BruteForceSearcher, Metric};

/// Hierarchical search configuration.
#[derive(Debug, Clone)]
pub struct HierarchicalConfig {
    pub n_clusters: usize,
    pub n_probe: usize,
    pub metric: Metric,
}

impl Default for HierarchicalConfig {
    fn default() -> Self {
        Self { n_clusters: 100, n_probe: 5, metric: Metric::L2 }
    }
}

/// A single cluster's data.
struct ClusterData {
    _members: Array2<f32>,
    original_ids: Vec<usize>,
    searcher: BruteForceSearcher,
}

/// Two-stage hierarchical vector search.
///
/// Architecture:
///   Stage 1 (Coarse): Q queries vs C centroids — O(Q * C)
///   Stage 2 (Fine):   Q queries vs n_probe cluster members — O(Q * n_probe * M)
pub struct HierarchicalSearch {
    config: HierarchicalConfig,
    built: bool,
    centroid_searcher: Option<BruteForceSearcher>,
    clusters: Vec<ClusterData>,
    dim: usize,
}

impl HierarchicalSearch {
    /// New.
    pub fn new(config: HierarchicalConfig) -> Self {
        let n_probe = config.n_probe.min(config.n_clusters);
        let config = HierarchicalConfig { n_probe, ..config };
        Self { config, built: false, centroid_searcher: None, clusters: Vec::new(), dim: 0 }
    }

    /// Build the hierarchical index from vectors.
    pub fn build(&mut self, vectors: ArrayView2<f32>) {
        let (n, d) = (vectors.nrows(), vectors.ncols());
        self.dim = d;
        let n_clusters = self.config.n_clusters.min(n);

        // KMeans clustering (CPU, done once at build time)
        let (centroids, labels) = kmeans(vectors, n_clusters, 30);

        // Group vectors by cluster
        self.clusters = Vec::new();
        for c in 0..n_clusters {
            let mut members_indices = Vec::new();
            let mut members_data = Vec::new();
            for (i, &label) in labels.iter().enumerate() {
                if label == c {
                    members_indices.push(i);
                    members_data.push(vectors.row(i).to_vec());
                }
            }
            if members_data.is_empty() {
                continue;
            }
            let members = Array2::from_shape_vec((members_data.len(), d), members_data.into_iter().flatten().collect())
                .unwrap_or_else(|_| Array2::zeros((0, d)));
            let searcher = BruteForceSearcher::new(members.view(), self.config.metric);
            self.clusters.push(ClusterData { _members: members, original_ids: members_indices, searcher });
        }

        // Build centroid index
        self.centroid_searcher = Some(BruteForceSearcher::new(centroids.view(), self.config.metric));
        self.built = true;
    }

    /// Search for k nearest neighbors using two-stage approach.
    pub fn search(&self, query: &[f32], k: usize) -> HierarchicalResult {
        if !self.built {
            return HierarchicalResult { indices: vec![], distances: vec![] };
        }

        let centroid_searcher = self.centroid_searcher.as_ref().unwrap();
        let n_probe = self.clusters.len().min(self.config.n_probe);

        // Stage 1: Coarse — find n_probe closest clusters
        let coarse = centroid_searcher.search(
            Array1::from_vec(query.to_vec()).view(),
            n_probe,
        );

        // Stage 2: Fine — search within probed clusters
        let mut candidates: std::collections::HashMap<usize, f32> = std::collections::HashMap::new();

        for &cluster_idx in &coarse.indices {
            if cluster_idx >= self.clusters.len() {
                continue;
            }
            let cluster = &self.clusters[cluster_idx];
            let local_result = cluster.searcher.search(
                Array1::from_vec(query.to_vec()).view(),
                k,
            );

            for (i, &local_id) in local_result.indices.iter().enumerate() {
                let global_id = cluster.original_ids[local_id];
                let dist = local_result.distances[i];
                let entry = candidates.entry(global_id).or_insert(f32::MAX);
                if dist < *entry {
                    *entry = dist;
                }
            }
        }

        // Sort and take top-k
        let mut sorted: Vec<(usize, f32)> = candidates.into_iter().collect();
        sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted.truncate(k);

        HierarchicalResult {
            indices: sorted.iter().map(|(i, _)| *i).collect(),
            distances: sorted.iter().map(|(_, d)| *d).collect(),
        }
    }

    /// Batch search for multiple queries.
    pub fn search_batch(&self, queries: ArrayView2<f32>, k: usize) -> Vec<HierarchicalResult> {
        (0..queries.nrows())
            .map(|i| self.search(queries.row(i).as_slice().unwrap_or(&[]), k))
            .collect()
    }

    /// Is built.
    pub fn is_built(&self) -> bool { self.built }
    /// N clusters.
    pub fn n_clusters(&self) -> usize { self.clusters.len() }
    /// Dim.
    pub fn dim(&self) -> usize { self.dim }
}

/// Hierarchical search result.
#[derive(Debug, Clone)]
pub struct HierarchicalResult {
    pub indices: Vec<usize>,
    pub distances: Vec<f32>,
}

/// Simple KMeans clustering (CPU, for build time).
fn kmeans(vectors: ArrayView2<f32>, n_clusters: usize, max_iter: usize) -> (Array2<f32>, Vec<usize>) {
    let (n, d) = (vectors.nrows(), vectors.ncols());
    let n_clusters = n_clusters.min(n);

    // Random init (deterministic)
    let mut rng_seed: u64 = 42;
    let mut init_indices = Vec::new();
    for _ in 0..n_clusters {
        rng_seed = rng_seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        init_indices.push((rng_seed as usize) % n);
    }

    let mut centroids: Array2<f32> = Array2::zeros((n_clusters, d));
    for (c, &idx) in init_indices.iter().enumerate() {
        for j in 0..d {
            centroids[[c, j]] = vectors[[idx, j]];
        }
    }

    let mut labels = vec![0usize; n];

    for _ in 0..max_iter {
        // Assign: nearest centroid
        let mut changed = false;
        for i in 0..n {
            let mut best_c = 0;
            let mut best_dist = f64::MAX;
            for c in 0..n_clusters {
                let dist: f64 = (0..d)
                    .map(|j| {
                        let diff = vectors[[i, j]] as f64 - centroids[[c, j]] as f64;
                        diff * diff
                    })
                    .sum::<f64>()
                    .sqrt();
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

        // Update centroids
        let mut counts = vec![0usize; n_clusters];
        let mut sums = Array2::<f64>::zeros((n_clusters, d));
        for i in 0..n {
            let c = labels[i];
            counts[c] += 1;
            for j in 0..d {
                sums[[c, j]] += vectors[[i, j]] as f64;
            }
        }
        for c in 0..n_clusters {
            if counts[c] > 0 {
                for j in 0..d {
                    centroids[[c, j]] = (sums[[c, j]] / counts[c] as f64) as f32;
                }
            }
        }
    }

    (centroids, labels)
}

/// GPU hierarchical search facade (same interface as Python).
pub struct GpuHierarchicalSearch {
    inner: HierarchicalSearch,
}

impl GpuHierarchicalSearch {
    /// New.
    pub fn new(n_clusters: usize, n_probe: usize) -> Self {
        Self {
            inner: HierarchicalSearch::new(HierarchicalConfig {
                n_clusters,
                n_probe,
                metric: Metric::L2,
            }),
        }
    }

    /// Build.
    pub fn build(&mut self, vectors: ArrayView2<f32>) {
        self.inner.build(vectors);
    }

    /// Search.
    pub fn search(&self, query: &[f32], k: usize) -> HierarchicalResult {
        self.inner.search(query, k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};

    fn make_data() -> Array2<f32> {
        array![
            [1.0, 0.0], [1.1, 0.1], [0.9, -0.1],
            [0.0, 1.0], [0.1, 1.1], [-0.1, 0.9],
            [5.0, 5.0], [5.1, 4.9], [4.9, 5.1],
        ].into_shape_with_order((9, 2)).unwrap()
    }

    #[test]
    fn test_hierarchical_search() {
        let data = make_data();
        let mut hs = HierarchicalSearch::new(HierarchicalConfig {
            n_clusters: 3, n_probe: 2, metric: Metric::L2,
        });
        hs.build(data.view());
        assert!(hs.is_built());
        assert_eq!(hs.n_clusters(), 3);

        let result = hs.search(&[1.0, 0.0], 2);
        assert_eq!(result.indices.len(), 2);
        // Should find index 0 or 1 or 2 as nearest
        assert!(result.indices[0] <= 2);
    }

    #[test]
    fn test_batch_search() {
        let data = make_data();
        let mut hs = HierarchicalSearch::new(HierarchicalConfig {
            n_clusters: 3, n_probe: 2, metric: Metric::L2,
        });
        hs.build(data.view());
        let queries = array![[1.0, 0.0], [5.0, 5.0]];
        let results = hs.search_batch(queries.view(), 1);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_gpu_facade() {
        let data = make_data();
        let mut ghs = GpuHierarchicalSearch::new(3, 2);
        ghs.build(data.view());
        let result = ghs.search(&[0.0, 1.0], 1);
        assert_eq!(result.indices.len(), 1);
    }
}


