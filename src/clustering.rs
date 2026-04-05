//! KMeans clustering — native Rust implementation (no sklearn).
//! Optimized with KMeans++ init and Lloyd's algorithm.

use ndarray::{Array1, Array2, ArrayView2};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

/// KMeans clustering result.
#[derive(Clone, Debug)]
pub struct KMeans {
    pub n_clusters: usize,
    pub centroids: Option<Array2<f32>>,
    pub labels: Option<Array1<usize>>,
    pub inertia: f32,
    pub n_iter: usize,
    max_iter: usize,
    seed: u64,
}

impl KMeans {
    /// New.
    pub fn new(n_clusters: usize, max_iter: usize, seed: u64) -> Self {
        Self {
            n_clusters,
            centroids: None,
            labels: None,
            inertia: 0.0,
            n_iter: 0,
            max_iter,
            seed,
        }
    }

    /// KMeans++ initialization: pick first centroid randomly, then pick
    /// subsequent centroids proportional to squared distance.
    fn kmeans_plus_plus_init(&self, data: &Array2<f32>, mut rng: ChaCha8Rng) -> Array2<f32> {
        let (n, dim) = (data.nrows(), data.ncols());
        let k = self.n_clusters.min(n);
        let mut centroids = Array2::zeros((k, dim));

        // First centroid: random
        let first = rng.gen_range(0..n);
        centroids.row_mut(0).assign(&data.row(first));

        let mut min_dists: Vec<f32> = vec![f32::MAX; n];

        for c in 1..k {
            // Update min distances to nearest existing centroid
            let cent_row = centroids.row(c - 1);
            for (i, min_d) in min_dists.iter_mut().enumerate().take(n) {
                let row = data.row(i);
                let dist = (&row - &cent_row).mapv(|v| v * v).sum();
                if dist < *min_d {
                    *min_d = dist;
                }
            }

            // Weighted sampling proportional to min_dists
            let total: f32 = min_dists.iter().sum();
            if total < 1e-10 {
                // All points are the same, pick randomly
                let idx = rng.gen_range(0..n);
                centroids.row_mut(c).assign(&data.row(idx));
                continue;
            }

            let mut threshold = rng.gen::<f32>() * total;
            let mut chosen = n - 1;
            for (i, &d) in min_dists.iter().enumerate().take(n) {
                threshold -= d;
                if threshold <= 0.0 {
                    chosen = i;
                    break;
                }
            }
            centroids.row_mut(c).assign(&data.row(chosen));
        }

        centroids
    }

    /// Fit KMeans to data using Lloyd's algorithm.
    pub fn fit(&mut self, data: &Array2<f32>) -> &Self {
        let (n, dim) = (data.nrows(), data.ncols());
        let k = self.n_clusters.min(n);

        if n == 0 || k == 0 {
            return self;
        }

        let rng = ChaCha8Rng::seed_from_u64(self.seed);
        let mut centroids = self.kmeans_plus_plus_init(data, rng);

        let mut labels = Array1::zeros(n);
        let mut counts = vec![0usize; k];
        let mut new_centroids = Array2::zeros((k, dim));

        for iteration in 0..self.max_iter {
            // Assignment step: assign each point to nearest centroid (parallel)
            let assignments: Vec<(usize, usize, f32)> = (0..n)
                .into_par_iter()
                .map(|i| {
                    let row = data.row(i);
                    let (best_k, best_dist) = (0..k)
                        .map(|c| {
                            let diff = &row - &centroids.row(c);
                            let d = diff.dot(&diff);
                            (c, d)
                        })
                        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                        .expect("clusters must not be empty");
                    (i, best_k, best_dist)
                })
                .collect();

            new_centroids.fill(0.0);
            counts.iter_mut().for_each(|c| *c = 0);
            let mut changed = 0usize;
            let mut total_inertia = 0.0f32;

            for &(i, best_k, best_dist) in &assignments {
                if labels[i] != best_k {
                    changed += 1;
                }
                labels[i] = best_k;
                total_inertia += best_dist;
                counts[best_k] += 1;
                let mut cent_row = new_centroids.row_mut(best_k);
                cent_row.scaled_add(1.0, &data.row(i));
            }

            self.inertia = total_inertia;

            // Update step: recompute centroids
            for (c, count) in counts.iter().enumerate().take(k) {
                if *count > 0 {
                    let mut row = centroids.row_mut(c);
                    row.assign(&new_centroids.row(c));
                    row.mapv_inplace(|v| v / *count as f32);
                }
                // else: keep old centroid
            }

            self.n_iter = iteration + 1;

            // Convergence check
            if changed == 0 {
                break;
            }
        }

        // Handle clusters with 0 members by re-assigning their centroids
        // to the point furthest from its centroid
        for (c, count) in counts.iter().enumerate().take(k) {
            if *count == 0 {
                // Find point with max distance to its assigned centroid
                let mut best_dist = 0.0f32;
                let mut best_idx = 0;
                for i in 0..n {
                    let assigned = labels[i];
                    let diff = &data.row(i) - &centroids.row(assigned);
                    let d = diff.dot(&diff);
                    if d > best_dist {
                        best_dist = d;
                        best_idx = i;
                    }
                }
                centroids.row_mut(c).assign(&data.row(best_idx));
                labels[best_idx] = c;
            }
        }

        self.centroids = Some(centroids);
        self.labels = Some(labels);
        self
    }

    /// Predict cluster assignments for new data.
    pub fn predict(&self, data: &ArrayView2<f32>) -> Array1<usize> {
        let centroids = self.centroids.as_ref().expect("Must call fit() first");
        let n = data.nrows();
        let labels: Vec<usize> = (0..n)
            .into_par_iter()
            .map(|i| {
                let row = data.row(i);
                (0..centroids.nrows())
                    .map(|c| {
                        let diff = &row - &centroids.row(c);
                        diff.dot(&diff)
                    })
                    .enumerate()
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx)
                    .expect("centroids must not be empty")
            })
            .collect();

        Array1::from(labels)
    }

    /// Fit and return labels.
    pub fn fit_predict(&mut self, data: &Array2<f32>) -> Array1<usize> {
        self.fit(data);
        self.labels.clone().expect("labels must exist after fit()")
    }

    /// Transform: compute distances from each point to each centroid.
    pub fn transform<S2>(
        &self,
        data: &ndarray::ArrayBase<S2, ndarray::Dim<[usize; 2]>>,
    ) -> Array2<f32>
    where
        S2: ndarray::Data<Elem = f32> + std::marker::Sync,
    {
        let centroids = self.centroids.as_ref().expect("Must call fit() first");
        let n = data.nrows();
        let k = centroids.nrows();
        let rows: Vec<Vec<f32>> = (0..n)
            .into_par_iter()
            .map(|i| {
                let row = data.row(i);
                (0..k)
                    .map(|c| {
                        let diff = &row - &centroids.row(c);
                        diff.dot(&diff).sqrt()
                    })
                    .collect()
            })
            .collect();

        let mut distances = Array2::zeros((n, k));
        for (i, row_dists) in rows.iter().enumerate() {
            for (c, &d) in row_dists.iter().enumerate() {
                distances[[i, c]] = d;
            }
        }

        distances
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Create simple 2D data with 2 clear clusters.
    fn two_cluster_data() -> Array2<f32> {
        // 5 points near (0,0), 5 points near (10,10)
        let mut data = Vec::new();
        for i in 0..5 {
            data.push([i as f32 * 0.1, i as f32 * 0.1]);
        }
        for i in 0..5 {
            data.push([10.0 + i as f32 * 0.1, 10.0 + i as f32 * 0.1]);
        }
        Array2::from_shape_vec((10, 2), data.into_iter().flatten().collect()).unwrap()
    }

    #[test]
    fn test_kmeans_creation() {
        let km = KMeans::new(3, 100, 42);
        assert_eq!(km.n_clusters, 3);
        assert_eq!(km.max_iter, 100);
        assert!(km.centroids.is_none());
        assert!(km.labels.is_none());
        assert_eq!(km.n_iter, 0);
    }

    #[test]
    fn test_fit_and_predict() {
        let data = two_cluster_data();
        let mut km = KMeans::new(2, 50, 42);
        km.fit(&data);

        assert!(km.centroids.is_some());
        assert!(km.labels.is_some());
        assert!(km.n_iter > 0);

        // Each point should be assigned to one of 2 clusters
        let labels = km.labels.as_ref().unwrap();
        assert_eq!(labels.len(), 10);
        for &l in labels.iter() {
            assert!(l < 2);
        }

        // Predict on same data should match labels
        let pred = km.predict(&data.view());
        assert_eq!(pred.len(), 10);
    }

    #[test]
    fn test_fit_empty_data() {
        let data = Array2::<f32>::zeros((0, 3));
        let mut km = KMeans::new(3, 50, 42);
        km.fit(&data);
        assert!(km.centroids.is_none());
    }

    #[test]
    fn test_fit_predict_convenience() {
        let data = two_cluster_data();
        let mut km = KMeans::new(2, 50, 42);
        let labels = km.fit_predict(&data);
        assert_eq!(labels.len(), 10);
    }

    #[test]
    fn test_transform_distances() {
        let data = two_cluster_data();
        let mut km = KMeans::new(2, 50, 42);
        km.fit(&data);
        let dists = km.transform(&data.view());
        assert_eq!(dists.shape(), &[10, 2]);
        // All distances should be non-negative
        for v in dists.iter() {
            assert!(*v >= 0.0);
        }
    }
}
