//! Geometry functions for SplatsDB hypersphere operations.
//!
//! Optimized implementations using ndarray with iterators and zero-copy where possible.

use ndarray::{Array2, ArrayView1};

const EPSILON: f32 = 1e-8;
const DOT_CLIP_MARGIN: f32 = 1e-7;

/// Normalize vectors to unit hypersphere.
///
/// Divides each row by its L2 norm + epsilon to avoid division by zero.
///
/// # Arguments
/// * `x` - Input array of shape (N, D) or (D,)
///
/// # Returns
/// Normalized array with same shape
pub fn normalize_sphere(x: &Array2<f32>) -> Array2<f32> {
    let n_rows = x.nrows();
    let mut result = x.clone();

    for i in 0..n_rows {
        let row = result.row(i);
        let norm: f32 = row.iter().map(|&v| v * v).sum::<f32>().sqrt() + EPSILON;
        let mut row_mut = result.row_mut(i);
        row_mut.mapv_inplace(|v| v / norm);
    }

    result
}

/// Calculate geodesic distance between vectors.
///
/// Returns arccos of the clipped dot product, ensuring numerical stability.
///
/// # Arguments
/// * `x` - First vector
/// * `y` - Second vector
///
/// # Returns
/// Geodesic distance in radians
pub fn geodesic_distance(x: &ArrayView1<f32>, y: &ArrayView1<f32>) -> f32 {
    let dot: f32 = x.iter().zip(y.iter()).map(|(&a, &b)| a * b).sum();

    // Early exit for near-identical vectors
    if dot > 1.0 - 1e-5 {
        return 0.0;
    }

    // Clip to avoid numerical issues with arccos
    let clipped = dot.clamp(-1.0 + DOT_CLIP_MARGIN, 1.0 - DOT_CLIP_MARGIN);

    clipped.acos()
}

/// Compute pairwise cosine similarity matrix.
///
/// # Arguments
/// * `x` - Input array of shape (N, D), assumed to be normalized
///
/// # Returns
/// Similarity matrix of shape (N, N)
pub fn cosine_similarity_matrix(x: &Array2<f32>) -> Array2<f32> {
    let n = x.nrows();
    let dim = x.ncols();

    // Try GPU path
    let flat: Vec<f32> = x.iter().copied().collect();
    if let Some(gpu_result) = crate::gpu::cosine_similarity_matrix(&flat, n, dim) {
        return Array2::from_shape_vec((n, n), gpu_result).unwrap_or_else(|_| {
            // Fallback on shape error
            cpu_cosine_similarity_matrix(x)
        });
    }
    static GPU_FALLBACK: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    if GPU_FALLBACK.set(()).is_ok() {
        eprintln!(
            "[splatsdb] warning: GPU cosine_similarity_matrix kernel unavailable — CPU fallback (logged once)"
        );
    }

    cpu_cosine_similarity_matrix(x)
}

fn cpu_cosine_similarity_matrix(x: &Array2<f32>) -> Array2<f32> {
    let n = x.nrows();
    let mut result = Array2::<f32>::zeros((n, n));

    // Compute upper triangle and mirror
    for i in 0..n {
        for j in i..n {
            let dot: f32 = x
                .row(i)
                .iter()
                .zip(x.row(j).iter())
                .map(|(&a, &b)| a * b)
                .sum();

            result[[i, j]] = dot;
            if i != j {
                result[[j, i]] = dot;
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_normalize_sphere() {
        let x = array![[3.0_f32, 4.0_f32], [0.0_f32, 0.0_f32]];
        let normalized = normalize_sphere(&x);

        // First row: [3, 4] -> [3/5, 4/5]
        assert_relative_eq!(normalized[[0, 0]], 0.6_f32, epsilon = 1e-5);
        assert_relative_eq!(normalized[[0, 1]], 0.8_f32, epsilon = 1e-5);

        // Second row: [0, 0] -> [0, 0] (divided by epsilon)
        assert_relative_eq!(normalized[[1, 0]], 0.0_f32, epsilon = 1e-5);
        assert_relative_eq!(normalized[[1, 1]], 0.0_f32, epsilon = 1e-5);
    }

    #[test]
    fn test_geodesic_distance() {
        let x = array![1.0_f32, 0.0_f32, 0.0_f32];
        let y = array![0.0_f32, 1.0_f32, 0.0_f32];

        let dist = geodesic_distance(&x.view(), &y.view());
        assert_relative_eq!(dist, std::f32::consts::FRAC_PI_2, epsilon = 1e-5);

        // Same vector should have distance 0
        let dist_same = geodesic_distance(&x.view(), &x.view());
        assert_relative_eq!(dist_same, 0.0_f32, epsilon = 1e-5);
    }
}
