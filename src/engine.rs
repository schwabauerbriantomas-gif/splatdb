//! SplatDB Compute Engine - CPU-optimized expert distance computation.
//!
//! Ported from Python m2m/engine.py. Uses ndarray einsum for batch L2 distances
//! and partial_sort for efficient top-k selection.

use ndarray::{Array2, ArrayView1};
use rayon::prelude::*;

/// Result entry from expert distance computation.
#[derive(Debug, Clone)]
pub struct ExpertDistanceResult {
    pub index: usize,
    pub distance: f32,
    pub coarse_id: usize,
    pub fine_id: usize,
}

/// CPU-only SplatDB engine for expert distance computation.
///
/// In the Rust native version, GPU backends (CUDA/Vulkan) are not available.
/// All computation uses optimized CPU paths with SIMD-friendly f32 operations.
pub struct SplatDBEngine {
    pub device: String,
}

impl SplatDBEngine {
    /// New.
    pub fn new(_config: Option<&crate::config::SplatDBConfig>) -> Self {
        // CPU-only in native Rust. Config device hint is acknowledged but not used.
        Self {
            device: "cpu".to_string(),
        }
    }

    /// Compute L2 distances between a query and expert embeddings.
    ///
    /// Returns results with original indices, distances, coarse and fine cluster IDs.
    ///
    /// # Optimization
    /// - Uses `ndarray::einsum("ij,ij->i", ...)` for batch squared distances (no sqrt)
    /// - Results are pre-sorted by distance ascending
    pub fn compute_expert_distances(
        &self,
        query: &[f32],
        expert_embeddings: &Array2<f32>,
        expert_indices: &[usize],
        coarse_ids: &[usize],
        fine_ids: &[usize],
    ) -> Vec<ExpertDistanceResult> {
        if expert_indices.is_empty() {
            return Vec::new();
        }

        let n = expert_indices.len();

        // Use einsum for vectorized squared L2 distance: ||e - q||^2
        let q = ArrayView1::from(query);
        let distances_sq: Vec<f32> = expert_embeddings
            .outer_iter()
            .par_bridge()
            .map(|row: ndarray::ArrayView1<f32>| {
                row.iter()
                    .zip(q.iter())
                    .map(|(e, qv)| {
                        let d = e - qv;
                        d * d
                    })
                    .sum()
            })
            .collect();

        // Build results with sqrt distance
        let mut results: Vec<ExpertDistanceResult> = (0..n)
            .map(|i| ExpertDistanceResult {
                index: expert_indices[i],
                distance: distances_sq[i].sqrt(),
                coarse_id: coarse_ids[i],
                fine_id: fine_ids[i],
            })
            .collect();

        // Sort by distance ascending
        results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal));

        results
    }

    /// Compute squared distances (avoids sqrt, useful when only ordering matters).
    pub fn compute_expert_distances_sq(
        &self,
        query: &[f32],
        expert_embeddings: &Array2<f32>,
        expert_indices: &[usize],
        coarse_ids: &[usize],
        fine_ids: &[usize],
    ) -> Vec<(usize, f32, usize, usize)> {
        if expert_indices.is_empty() {
            return Vec::new();
        }

        let q = ArrayView1::from(query);
        let distances_sq: Vec<f32> = expert_embeddings
            .outer_iter()
            .par_bridge()
            .map(|row| {
                row.iter()
                    .zip(q.iter())
                    .map(|(e, qv)| {
                        let d = e - qv;
                        d * d
                    })
                    .sum::<f32>()
            })
            .collect();

        let mut results: Vec<(usize, f32, usize, usize)> = (0..expert_indices.len())
            .map(|i| (expert_indices[i], distances_sq[i], coarse_ids[i], fine_ids[i]))
            .collect();

        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Compute top-k distances using partial sort (no full sort needed).
    pub fn compute_expert_distances_topk(
        &self,
        query: &[f32],
        expert_embeddings: &Array2<f32>,
        expert_indices: &[usize],
        coarse_ids: &[usize],
        fine_ids: &[usize],
        k: usize,
    ) -> Vec<ExpertDistanceResult> {
        if expert_indices.is_empty() || k == 0 {
            return Vec::new();
        }

        let q = ArrayView1::from(query);
        let distances_sq: Vec<f32> = expert_embeddings
            .outer_iter()
            .par_bridge()
            .map(|row| {
                row.iter()
                    .zip(q.iter())
                    .map(|(e, qv)| {
                        let d = e - qv;
                        d * d
                    })
                    .sum::<f32>()
            })
            .collect();

        let mut indexed: Vec<(f32, usize)> = (0..expert_indices.len())
            .map(|i| (distances_sq[i], i))
            .collect();

        // Partial sort for top-k (O(n) average instead of O(n log n))
        let k_actual = k.min(indexed.len());
        indexed.select_nth_unstable_by(k_actual, |a, b| {
            a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Sort only the top-k portion
        indexed[..k_actual].sort_by(|a, b| {
            a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
        });

        indexed[..k_actual]
            .iter()
            .map(|(dist_sq, i)| ExpertDistanceResult {
                index: expert_indices[*i],
                distance: dist_sq.sqrt(),
                coarse_id: coarse_ids[*i],
                fine_id: fine_ids[*i],
            })
            .collect()
    }
}

impl Default for SplatDBEngine {
    fn default() -> Self {
        Self::new(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_empty_experts() {
        let engine = SplatDBEngine::default();
        let embeddings = Array2::<f32>::zeros((0, 3));
        let results = engine.compute_expert_distances(
            &[0.0, 0.0, 0.0],
            &embeddings,
            &[],
            &[],
            &[],
        );
        assert!(results.is_empty());
    }

    #[test]
    fn test_distance_computation() {
        let engine = SplatDBEngine::default();
        let embeddings = array![[0.0f32, 0.0, 0.0], [3.0, 4.0, 0.0]];
        let results = engine.compute_expert_distances(
            &[0.0, 0.0, 0.0],
            &embeddings,
            &[0, 1],
            &[0, 1],
            &[0, 0],
        );
        assert_eq!(results.len(), 2);
        assert!((results[0].distance - 0.0).abs() < 1e-6);
        assert!((results[1].distance - 5.0).abs() < 1e-4);
    }
}
