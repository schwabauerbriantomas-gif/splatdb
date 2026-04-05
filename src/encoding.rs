//! Encoding functions for Gaussian Splat embeddings.
//!
//! Optimized implementations for converting Gaussian splat attributes
//! into embedding vectors for indexing.

use ndarray::{s, Array1, Array2, Axis};

const EPSILON: f32 = 1e-8;

// ==================== POSITION ENCODING ====================

/// Sinusoidal Position Encoder for 3D coordinates.
///
/// Uses multi-frequency sinusoids similar to NeRF positional encoding.
/// Output dimension is adjusted to be divisible by 6 (for x, y, z sin/cos pairs).
pub struct SinusoidalPositionEncoder {
    /// Output dimension (divisible by 6)
    dim: usize,
    /// Target dimension (may include padding)
    target_dim: usize,
    /// Number of frequency bands (dim / 6)
    n_freq: usize,
}

impl SinusoidalPositionEncoder {
    /// Create a new encoder with specified output dimension.
    ///
    /// Dimension will be adjusted to be divisible by 6, then padded to target.
    pub fn new(dim: usize) -> Self {
        let target_dim = dim;
        let dim = ((dim / 6).max(1)) * 6;
        let n_freq = dim / 6;
        Self {
            dim,
            target_dim,
            n_freq,
        }
    }

    /// Default 64D encoder
    pub fn default_64() -> Self {
        Self::new(64)
    }

    /// Get output dimension
    #[allow(clippy::misnamed_getters)]
    pub fn dim(&self) -> usize {
        self.target_dim
    }

    /// Encode a single 3D position.
    ///
    /// # Arguments
    /// * `position` - 3D position [x, y, z]
    /// * `bounds` - Optional bounds for normalization [(min_x, min_y, min_z), (max_x, max_y, max_z)]
    ///
    /// # Returns
    /// Encoding of dimension `self.dim`
    pub fn encode_single(
        &self,
        position: &[f32; 3],
        bounds: Option<(&[f32; 3], &[f32; 3])>,
    ) -> Array1<f32> {
        let mut encoding = Array1::<f32>::zeros(self.dim);

        // Normalize to [0, 1] using bounds or assume already normalized
        let (x, y, z) = if let Some((min, max)) = bounds {
            let range_x = (max[0] - min[0]).max(EPSILON);
            let range_y = (max[1] - min[1]).max(EPSILON);
            let range_z = (max[2] - min[2]).max(EPSILON);

            (
                (position[0] - min[0]) / range_x,
                (position[1] - min[1]) / range_y,
                (position[2] - min[2]) / range_z,
            )
        } else {
            (position[0], position[1], position[2])
        };

        // Multi-frequency encoding
        for d in 0..self.n_freq {
            let freq = 2.0_f32.powi(d as i32);
            let idx = d * 6;

            encoding[idx] = (x * freq).sin();
            encoding[idx + 1] = (x * freq).cos();
            encoding[idx + 2] = (y * freq).sin();
            encoding[idx + 3] = (y * freq).cos();
            encoding[idx + 4] = (z * freq).sin();
            encoding[idx + 5] = (z * freq).cos();
        }

        // Pad to target dimension if needed
        if self.target_dim > self.dim {
            let mut padded = Array1::<f32>::zeros(self.target_dim);
            padded.slice_mut(s![..self.dim]).assign(&encoding);
            padded
        } else {
            encoding
        }
    }

    /// Encode batch of 3D positions.
    ///
    /// # Arguments
    /// * `positions` - Array of shape (N, 3)
    ///
    /// # Returns
    /// Array of shape (N, dim)
    pub fn encode(&self, positions: &Array2<f32>) -> Array2<f32> {
        let n = positions.nrows();
        let mut result = Array2::<f32>::zeros((n, self.target_dim));

        // Compute bounds from data
        let mut min_vals = [f32::MAX, f32::MAX, f32::MAX];
        let mut max_vals = [f32::MIN, f32::MIN, f32::MIN];

        for row in positions.axis_iter(Axis(0)) {
            for (i, &val) in row.iter().enumerate() {
                min_vals[i] = min_vals[i].min(val);
                max_vals[i] = max_vals[i].max(val);
            }
        }

        let bounds = Some((&min_vals, &max_vals));

        // Encode each position
        for (i, row) in positions.axis_iter(Axis(0)).enumerate() {
            let pos: [f32; 3] = [row[0], row[1], row[2]];
            let enc = self.encode_single(&pos, bounds);
            result.row_mut(i).assign(&enc);
        }

        result
    }
}

// ==================== COLOR ENCODING ====================

/// Color Histogram Encoder using Gaussian-smoothed bins.
///
/// Creates an 8^3 = 512 dimensional histogram with Gaussian kernel smoothing.
pub struct ColorHistogramEncoder {
    /// Number of bins per channel
    n_bins: usize,
    /// Output dimension (n_bins^3)
    dim: usize,
}

impl ColorHistogramEncoder {
    /// Create a new encoder with 8 bins per channel (512D output).
    pub fn new() -> Self {
        Self::with_bins(8)
    }

    /// Create encoder with custom bin count.
    pub fn with_bins(n_bins: usize) -> Self {
        Self {
            n_bins,
            dim: n_bins * n_bins * n_bins,
        }
    }

    /// Get output dimension
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Encode a single RGB color.
    ///
    /// # Arguments
    /// * `color` - RGB color in [0, 1] range
    ///
    /// # Returns
    /// Histogram of dimension n_bins^3
    pub fn encode_single(&self, color: &[f32; 3]) -> Array1<f32> {
        let mut encoding = Array1::<f32>::zeros(self.dim);

        let r = color[0].clamp(0.0, 1.0);
        let g = color[1].clamp(0.0, 1.0);
        let b = color[2].clamp(0.0, 1.0);

        // Quantize to nearest bin
        let bin_r = ((r * self.n_bins as f32) as usize).min(self.n_bins - 1);
        let bin_g = ((g * self.n_bins as f32) as usize).min(self.n_bins - 1);
        let bin_b = ((b * self.n_bins as f32) as usize).min(self.n_bins - 1);

        // Gaussian kernel smoothing
        let mut idx = 0;
        for br in 0..self.n_bins {
            for bg in 0..self.n_bins {
                for bb in 0..self.n_bins {
                    let dr = (br as f32) - (bin_r as f32);
                    let dg = (bg as f32) - (bin_g as f32);
                    let db = (bb as f32) - (bin_b as f32);

                    // Gaussian kernel: exp(-dist^2 / 4)
                    let dist_sq = dr * dr + dg * dg + db * db;
                    encoding[idx] = (-dist_sq / 4.0).exp();

                    idx += 1;
                }
            }
        }

        encoding
    }

    /// Encode batch of RGB colors.
    ///
    /// # Arguments
    /// * `colors` - Array of shape (N, 3) in [0, 1] or [0, 255] range
    ///
    /// # Returns
    /// Array of shape (N, dim)
    pub fn encode(&self, colors: &Array2<f32>) -> Array2<f32> {
        let n = colors.nrows();
        let mut result = Array2::<f32>::zeros((n, self.dim));

        // Detect if colors are in [0, 255] range
        let max_val = colors.iter().cloned().fold(0.0_f32, f32::max);
        let needs_normalize = max_val > 1.0;

        for (i, row) in colors.axis_iter(Axis(0)).enumerate() {
            let color: [f32; 3] = if needs_normalize {
                [row[0] / 255.0, row[1] / 255.0, row[2] / 255.0]
            } else {
                [row[0], row[1], row[2]]
            };

            let enc = self.encode_single(&color);
            result.row_mut(i).assign(&enc);
        }

        result
    }
}

impl Default for ColorHistogramEncoder {
    fn default() -> Self {
        Self::new()
    }
}

// ==================== ATTRIBUTE ENCODING ====================

/// Attribute Encoder for opacity, scale, and rotation.
///
/// Creates 64D hand-crafted features from splat attributes:
/// - Opacity features (8 dims)
/// - Scale features (24 dims)
/// - Rotation features (32 dims)
pub struct AttributeEncoder {
    dim: usize,
}

impl AttributeEncoder {
    /// Create a new 64D attribute encoder.
    pub fn new() -> Self {
        Self { dim: 64 }
    }

    /// Get output dimension
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Encode single splat attributes.
    ///
    /// # Arguments
    /// * `opacity` - Opacity value in [0, 1]
    /// * `scale` - Scale factors [sx, sy, sz]
    /// * `rotation` - Quaternion [w, x, y, z]
    ///
    /// # Returns
    /// 64D attribute encoding
    pub fn encode_single(
        &self,
        opacity: f32,
        scale: &[f32; 3],
        rotation: &[f32; 4],
    ) -> Array1<f32> {
        let mut encoding = Array1::<f32>::zeros(self.dim);

        let o = opacity;
        let (sx, sy, sz) = (scale[0], scale[1], scale[2]);
        let (qw, qx, qy, qz) = (rotation[0], rotation[1], rotation[2], rotation[3]);

        // Opacity features (8 dims) - indices 0-7
        encoding[0] = o;
        encoding[1] = o * o;
        encoding[2] = o * o * o;
        encoding[3] = (o + EPSILON).sqrt();
        encoding[4] = (o + EPSILON).ln();
        encoding[5] = 1.0 - o;
        encoding[6] = if o > 0.5 { 1.0 } else { 0.0 };
        encoding[7] = if o < 0.5 { 1.0 } else { 0.0 };

        // Scale features (24 dims) - indices 8-31
        encoding[8] = sx;
        encoding[9] = sy;
        encoding[10] = sz;
        encoding[11] = sx * sy * sz;
        encoding[12] = (sx + sy + sz) / 3.0;
        encoding[13] = (sx * sx + sy * sy + sz * sz).sqrt();
        encoding[14] = sx / (sy + EPSILON);
        encoding[15] = sx / (sz + EPSILON);
        encoding[16] = sy / (sz + EPSILON);
        encoding[17] = sx * sx;
        encoding[18] = sy * sy;
        encoding[19] = sz * sz;
        encoding[20] = (sx + EPSILON).ln();
        encoding[21] = (sy + EPSILON).ln();
        encoding[22] = (sz + EPSILON).ln();
        // indices 23-31: zero padding

        // Rotation features (32 dims) - indices 32-63
        encoding[32] = qw;
        encoding[33] = qx;
        encoding[34] = qy;
        encoding[35] = qz;
        encoding[36] = qw * qw;
        encoding[37] = qx * qx;
        encoding[38] = qy * qy;
        encoding[39] = qz * qz;
        encoding[40] = qw * qx;
        encoding[41] = qw * qy;
        encoding[42] = qw * qz;
        encoding[43] = qx * qy;
        encoding[44] = qx * qz;
        encoding[45] = qy * qz;
        // indices 46-63: zero padding

        encoding
    }

    /// Encode batch of splat attributes.
    ///
    /// # Arguments
    /// * `opacities` - Array of shape (N,)
    /// * `scales` - Array of shape (N, 3)
    /// * `rotations` - Array of shape (N, 4)
    ///
    /// # Returns
    /// Array of shape (N, 64)
    pub fn encode(
        &self,
        opacities: &Array1<f32>,
        scales: &Array2<f32>,
        rotations: &Array2<f32>,
    ) -> Array2<f32> {
        let n = opacities.len();
        let mut result = Array2::<f32>::zeros((n, self.dim));

        for i in 0..n {
            let scale: [f32; 3] = [scales[[i, 0]], scales[[i, 1]], scales[[i, 2]]];
            let rotation: [f32; 4] = [
                rotations[[i, 0]],
                rotations[[i, 1]],
                rotations[[i, 2]],
                rotations[[i, 3]],
            ];

            let enc = self.encode_single(opacities[i], &scale, &rotation);
            result.row_mut(i).assign(&enc);
        }

        result
    }
}

impl Default for AttributeEncoder {
    fn default() -> Self {
        Self::new()
    }
}

// ==================== FULL EMBEDDING ====================

/// Builder for complete 640D splat embeddings.
///
/// Combines position, color, and attribute encodings.
pub struct FullEmbeddingBuilder {
    pos_encoder: SinusoidalPositionEncoder,
    color_encoder: ColorHistogramEncoder,
    attr_encoder: AttributeEncoder,
}

impl FullEmbeddingBuilder {
    /// Create a new embedding builder with default encoders.
    pub fn new() -> Self {
        Self {
            pos_encoder: SinusoidalPositionEncoder::default_64(),
            color_encoder: ColorHistogramEncoder::new(),
            attr_encoder: AttributeEncoder::new(),
        }
    }

    /// Total embedding dimension (640)
    pub fn embedding_dim(&self) -> usize {
        640
    }

    /// Build full 640D embeddings for batch of splats.
    ///
    /// # Arguments
    /// * `positions` - (N, 3) positions
    /// * `colors` - (N, 3) colors in [0, 1]
    /// * `opacities` - (N,) opacities
    /// * `scales` - (N, 3) scales
    /// * `rotations` - (N, 4) quaternions
    ///
    /// # Returns
    /// (N, 640) embeddings
    pub fn build(
        &self,
        positions: &Array2<f32>,
        colors: &Array2<f32>,
        opacities: &Array1<f32>,
        scales: &Array2<f32>,
        rotations: &Array2<f32>,
    ) -> Array2<f32> {
        let n = positions.nrows();

        let pos_enc = self.pos_encoder.encode(positions);
        let color_enc = self.color_encoder.encode(colors);
        let attr_enc = self.attr_encoder.encode(opacities, scales, rotations);

        // Concatenate along axis 1 using slice assignment
        let mut result = Array2::<f32>::zeros((n, 640));
        result
            .slice_mut(s![.., 0..64])
            .assign(&pos_enc.slice(s![.., 0..64]));
        result.slice_mut(s![.., 64..576]).assign(&color_enc);
        result.slice_mut(s![.., 576..640]).assign(&attr_enc);

        result
    }

    /// Build embedding for a single splat.
    ///
    /// # Returns
    /// 640D embedding vector
    pub fn build_single(
        &self,
        position: &[f32; 3],
        color: &[f32; 3],
        opacity: f32,
        scale: &[f32; 3],
        rotation: &[f32; 4],
    ) -> Array1<f32> {
        let mut result = Array1::<f32>::zeros(640);

        // Position encoding (64 dims)
        let pos_enc = self.pos_encoder.encode_single(position, None);
        result.slice_mut(s![0..64]).assign(&pos_enc);

        // Color encoding (512 dims)
        let color_enc = self.color_encoder.encode_single(color);
        result.slice_mut(s![64..576]).assign(&color_enc);

        // Attribute encoding (64 dims)
        let attr_enc = self.attr_encoder.encode_single(opacity, scale, rotation);
        result.slice_mut(s![576..640]).assign(&attr_enc);

        result
    }
}

impl Default for FullEmbeddingBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience function to build full 640D embedding for splats.
pub fn build_full_embedding(
    positions: &Array2<f32>,
    colors: &Array2<f32>,
    opacities: &Array1<f32>,
    scales: &Array2<f32>,
    rotations: &Array2<f32>,
) -> Array2<f32> {
    let builder = FullEmbeddingBuilder::new();
    builder.build(positions, colors, opacities, scales, rotations)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_sinusoidal_position_encoder() {
        let encoder = SinusoidalPositionEncoder::new(64);
        assert_eq!(encoder.dim(), 64); // Padded to 64 (10 freq bands * 6 + 4 padding)

        let pos = [0.5, 0.5, 0.5];
        let enc = encoder.encode_single(&pos, None);
        assert_eq!(enc.len(), 64);

        // First frequency should give sin(0.5), cos(0.5)
        assert_relative_eq!(enc[0], 0.5_f32.sin(), epsilon = 1e-5);
        assert_relative_eq!(enc[1], 0.5_f32.cos(), epsilon = 1e-5);
    }

    #[test]
    fn test_color_histogram_encoder() {
        let encoder = ColorHistogramEncoder::new();
        assert_eq!(encoder.dim(), 512);

        // Pure red should have peak at bin (7, 0, 0) for [1, 0, 0]
        let color = [1.0, 0.0, 0.0];
        let enc = encoder.encode_single(&color);
        assert_eq!(enc.len(), 512);

        // Should have non-zero values (Gaussian smoothing spreads)
        assert!(enc.iter().any(|&v| v > 0.0));
    }

    #[test]
    fn test_attribute_encoder() {
        let encoder = AttributeEncoder::new();
        assert_eq!(encoder.dim(), 64);

        let enc = encoder.encode_single(0.8, &[0.1, 0.2, 0.3], &[1.0, 0.0, 0.0, 0.0]);
        assert_eq!(enc.len(), 64);

        // Check opacity features
        assert_relative_eq!(enc[0], 0.8, epsilon = 1e-5);
        assert_relative_eq!(enc[1], 0.64, epsilon = 1e-5);

        // Check rotation features
        assert_relative_eq!(enc[32], 1.0, epsilon = 1e-5); // qw
        assert_relative_eq!(enc[36], 1.0, epsilon = 1e-5); // qw^2
    }

    #[test]
    fn test_full_embedding_builder() {
        let builder = FullEmbeddingBuilder::new();
        assert_eq!(builder.embedding_dim(), 640);

        let embedding = builder.build_single(
            &[1.0, 2.0, 3.0],
            &[0.5, 0.5, 0.5],
            0.9,
            &[0.1, 0.1, 0.1],
            &[1.0, 0.0, 0.0, 0.0],
        );

        assert_eq!(embedding.len(), 640);
        assert!(embedding.iter().any(|&v| v != 0.0));
    }

    #[test]
    fn test_batch_encoding() {
        let positions = array![[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]];
        let colors = array![[0.5, 0.5, 0.5], [1.0, 0.0, 0.0]];
        let opacities = array![0.5, 1.0];
        let scales = array![[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]];
        let rotations = array![[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]];

        let embeddings = build_full_embedding(&positions, &colors, &opacities, &scales, &rotations);

        assert_eq!(embeddings.nrows(), 2);
        assert_eq!(embeddings.ncols(), 640);
    }
}
