//! Data types for Gaussian Splatting.
//!
//! Core data structures for representing Gaussian splats and their embeddings.

use serde::{Deserialize, Serialize};
use ndarray::Array1;

/// Type alias for Splat ID
pub type SplatID = u64;

/// Type alias for Cluster ID  
pub type ClusterID = u64;

/// Type alias for embedding vectors
pub type EmbeddingVector = Array1<f32>;

/// Represents a single Gaussian Splat.
///
/// A Gaussian splat is a 3D ellipsoid with position, color, opacity,
/// scale, and rotation that represents a small piece of a 3D scene.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GaussianSplat {
    /// Unique identifier for the splat
    pub id: u64,
    /// 3D position (x, y, z) in world coordinates
    pub position: [f32; 3],
    /// RGB color values, typically in [0, 1]
    pub color: [f32; 3],
    /// Transparency value in [0, 1]
    pub opacity: f32,
    /// Scale factors (sx, sy, sz) for the ellipsoid axes
    pub scale: [f32; 3],
    /// Quaternion rotation (w, x, y, z) - normalized
    pub rotation: [f32; 4],
}

impl Default for GaussianSplat {
    fn default() -> Self {
        Self {
            id: 0,
            position: [0.0; 3],
            color: [1.0; 3],
            opacity: 1.0,
            scale: [1.0; 3],
            rotation: [1.0, 0.0, 0.0, 0.0], // Identity quaternion
        }
    }
}

impl GaussianSplat {
    /// Create a new GaussianSplat with default values
    pub fn new(id: u64) -> Self {
        Self { id, ..Default::default() }
    }

    /// Create a GaussianSplat with all fields specified
    pub fn with_fields(
        id: u64,
        position: [f32; 3],
        color: [f32; 3],
        opacity: f32,
        scale: [f32; 3],
        rotation: [f32; 4],
    ) -> Self {
        let mut splat = Self {
            id,
            position,
            color,
            opacity,
            scale,
            rotation,
        };
        splat.normalize_quaternion();
        splat
    }

    /// Normalize the quaternion to unit length
    pub fn normalize_quaternion(&mut self) {
        let norm = (self.rotation[0] * self.rotation[0]
            + self.rotation[1] * self.rotation[1]
            + self.rotation[2] * self.rotation[2]
            + self.rotation[3] * self.rotation[3])
            .sqrt();
        
        if norm > 0.0 {
            for r in &mut self.rotation {
                *r /= norm;
            }
        }
    }

    /// Convert quaternion to rotation matrix.
    /// 
    /// Returns a 3x3 rotation matrix from quaternion (w, x, y, z).
    #[inline(always)]
    pub fn quaternion_to_matrix(q: &[f32; 4]) -> [[f32; 3]; 3] {
        let (w, x, y, z) = (q[0], q[1], q[2], q[3]);
        
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y + w * z),
                2.0 * (x * z - w * y),
            ],
            [
                2.0 * (x * y - w * z),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z + w * x),
            ],
            [
                2.0 * (x * z + w * y),
                2.0 * (y * z - w * x),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ]
    }

    /// Compute the 3D covariance matrix from scale and rotation.
    /// 
    /// Returns: Σ = R * S * S^T * R^T = M * M^T where M = R * S
    pub fn covariance_3d(&self) -> [[f32; 3]; 3] {
        let r = Self::quaternion_to_matrix(&self.rotation);
        let s = self.scale;
        
        // M = R * S (scale columns of R)
        let m = [
            [r[0][0] * s[0], r[0][1] * s[1], r[0][2] * s[2]],
            [r[1][0] * s[0], r[1][1] * s[1], r[1][2] * s[2]],
            [r[2][0] * s[0], r[2][1] * s[1], r[2][2] * s[2]],
        ];
        
        // Σ = M * M^T
        [
            [
                m[0][0] * m[0][0] + m[0][1] * m[0][1] + m[0][2] * m[0][2],
                m[0][0] * m[1][0] + m[0][1] * m[1][1] + m[0][2] * m[1][2],
                m[0][0] * m[2][0] + m[0][1] * m[2][1] + m[0][2] * m[2][2],
            ],
            [
                m[1][0] * m[0][0] + m[1][1] * m[0][1] + m[1][2] * m[0][2],
                m[1][0] * m[1][0] + m[1][1] * m[1][1] + m[1][2] * m[1][2],
                m[1][0] * m[2][0] + m[1][1] * m[2][1] + m[1][2] * m[2][2],
            ],
            [
                m[2][0] * m[0][0] + m[2][1] * m[0][1] + m[2][2] * m[0][2],
                m[2][0] * m[1][0] + m[2][1] * m[1][1] + m[2][2] * m[1][2],
                m[2][0] * m[2][0] + m[2][1] * m[2][1] + m[2][2] * m[2][2],
            ],
        ]
    }

    /// Convert to ndarray position
    pub fn position_ndarray(&self) -> Array1<f32> {
        Array1::from_vec(self.position.to_vec())
    }

    /// Convert to ndarray color
    pub fn color_ndarray(&self) -> Array1<f32> {
        Array1::from_vec(self.color.to_vec())
    }

    /// Convert to ndarray scale
    pub fn scale_ndarray(&self) -> Array1<f32> {
        Array1::from_vec(self.scale.to_vec())
    }

    /// Convert to ndarray rotation
    pub fn rotation_ndarray(&self) -> Array1<f32> {
        Array1::from_vec(self.rotation.to_vec())
    }
}

/// Embedding vector for a Gaussian Splat.
///
/// The embedding is a 640-dimensional vector composed of:
/// - Position encoding (64 dims): Sinusoidal encoding of 3D position
/// - Color encoding (512 dims): Histogram-based color representation  
/// - Attribute encoding (64 dims): Opacity, scale, rotation features
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SplatEmbedding {
    /// ID of the corresponding splat
    pub splat_id: u64,
    /// 64D positional embedding
    pub position_encoding: Vec<f32>,
    /// 512D color histogram embedding
    pub color_encoding: Vec<f32>,
    /// 64D attribute embedding
    pub attribute_encoding: Vec<f32>,
}

impl Default for SplatEmbedding {
    fn default() -> Self {
        Self {
            splat_id: 0,
            position_encoding: vec![0.0; 64],
            color_encoding: vec![0.0; 512],
            attribute_encoding: vec![0.0; 64],
        }
    }
}

impl SplatEmbedding {
    /// Create a new embedding for a splat
    pub fn new(splat_id: u64) -> Self {
        Self {
            splat_id,
            ..Default::default()
        }
    }

    /// Concatenate all encodings into a single 640D vector.
    /// 
    /// Returns a heap-allocated array with all three encodings concatenated.
    pub fn full_embedding(&self) -> Vec<f32> {
        let mut result = Vec::with_capacity(640);
        result.extend_from_slice(&self.position_encoding);
        result.extend_from_slice(&self.color_encoding);
        result.extend_from_slice(&self.attribute_encoding);
        result
    }

    /// Get full embedding as ndarray
    pub fn full_embedding_ndarray(&self) -> Array1<f32> {
        Array1::from_vec(self.full_embedding())
    }

    /// Total embedding dimension
    pub const fn embedding_dim() -> usize {
        640
    }

    /// Get a view of position encoding
    pub fn position_view(&self) -> &[f32] {
        &self.position_encoding
    }

    /// Get a view of color encoding
    pub fn color_view(&self) -> &[f32] {
        &self.color_encoding
    }

    /// Get a view of attribute encoding
    pub fn attribute_view(&self) -> &[f32] {
        &self.attribute_encoding
    }
}

/// A cluster of similar splats.
///
/// Used in HRM2 for hierarchical organization.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SplatCluster {
    /// Cluster identifier
    pub id: u64,
    /// Center of the cluster in embedding space
    pub centroid: Vec<f32>,
    /// List of splat IDs in this cluster
    pub splat_ids: Vec<u64>,
    /// Bounding box (min, max) in 3D space
    pub bounds: ([f32; 3], [f32; 3]),
}

impl Default for SplatCluster {
    fn default() -> Self {
        Self {
            id: 0,
            centroid: Vec::new(),
            splat_ids: Vec::new(),
            bounds: ([0.0; 3], [0.0; 3]),
        }
    }
}

impl SplatCluster {
    /// Create a new cluster
    pub fn new(id: u64) -> Self {
        Self {
            id,
            ..Default::default()
        }
    }

    /// Create a cluster with centroid and bounds
    pub fn with_centroid(id: u64, centroid: Vec<f32>, bounds: ([f32; 3], [f32; 3])) -> Self {
        Self {
            id,
            centroid,
            splat_ids: Vec::new(),
            bounds,
        }
    }

    /// Number of splats in cluster
    pub fn size(&self) -> usize {
        self.splat_ids.len()
    }

    /// Check if a point is within the cluster bounds.
    pub fn contains_point(&self, point: &[f32; 3]) -> bool {
        let (min_b, max_b) = &self.bounds;
        
        point[0] >= min_b[0] && point[0] <= max_b[0]
            && point[1] >= min_b[1] && point[1] <= max_b[1]
            && point[2] >= min_b[2] && point[2] <= max_b[2]
    }

    /// Add a splat to this cluster
    pub fn add_splat(&mut self, splat_id: u64) {
        self.splat_ids.push(splat_id);
    }

    /// Get centroid as ndarray
    pub fn centroid_ndarray(&self) -> Array1<f32> {
        Array1::from_vec(self.centroid.clone())
    }

    /// Update centroid from splat embeddings (simple average)
    pub fn update_centroid(&mut self, embeddings: &[SplatEmbedding]) {
        if embeddings.is_empty() {
            return;
        }
        
        let dim = SplatEmbedding::embedding_dim();
        self.centroid = vec![0.0; dim];
        
        for emb in embeddings {
            let full = emb.full_embedding();
            for (i, val) in full.iter().enumerate() {
                self.centroid[i] += val;
            }
        }
        
        let n = embeddings.len() as f32;
        for val in &mut self.centroid {
            *val /= n;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_splat_default() {
        let splat = GaussianSplat::default();
        assert_eq!(splat.id, 0);
        assert_eq!(splat.position, [0.0; 3]);
        assert_eq!(splat.rotation, [1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_quaternion_to_matrix() {
        // Identity quaternion should give identity matrix
        let q = [1.0, 0.0, 0.0, 0.0];
        let r = GaussianSplat::quaternion_to_matrix(&q);
        
        assert!((r[0][0] - 1.0).abs() < 1e-5);
        assert!((r[1][1] - 1.0).abs() < 1e-5);
        assert!((r[2][2] - 1.0).abs() < 1e-5);
        assert!(r[0][1].abs() < 1e-5);
        assert!(r[0][2].abs() < 1e-5);
    }

    #[test]
    fn test_covariance_3d() {
        let mut splat = GaussianSplat::default();
        splat.scale = [1.0, 2.0, 3.0];
        splat.rotation = [1.0, 0.0, 0.0, 0.0]; // Identity
        
        let cov = splat.covariance_3d();
        
        // With identity rotation, diagonal should be scale^2
        assert!((cov[0][0] - 1.0).abs() < 1e-5);
        assert!((cov[1][1] - 4.0).abs() < 1e-5);
        assert!((cov[2][2] - 9.0).abs() < 1e-5);
    }

    #[test]
    fn test_splat_embedding() {
        let emb = SplatEmbedding::new(42);
        assert_eq!(emb.splat_id, 42);
        assert_eq!(emb.full_embedding().len(), 640);
    }

    #[test]
    fn test_splat_cluster() {
        let mut cluster = SplatCluster::new(1);
        cluster.bounds = ([0.0, 0.0, 0.0], [10.0, 10.0, 10.0]);
        
        assert!(cluster.contains_point(&[5.0, 5.0, 5.0]));
        assert!(!cluster.contains_point(&[15.0, 5.0, 5.0]));
        
        cluster.add_splat(1);
        cluster.add_splat(2);
        assert_eq!(cluster.size(), 2);
    }
}
