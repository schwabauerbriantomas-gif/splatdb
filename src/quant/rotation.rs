//! Random rotation matrices for whitening vectors before quantization.
//!
//! Generates a d×d orthogonal matrix via Modified Gram-Schmidt on a random
//! Gaussian matrix. Same purpose as turbo-quant's rotation.rs but uses ndarray
//! instead of nalgebra.
//!
//! License: MIT (adapted from github.com/RecursiveIntell/turbo-quant)

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Serialize};

use super::error::{Result, QuantError};

/// A full d×d orthogonal rotation matrix generated via Modified Gram-Schmidt.
///
/// Seeded deterministically so that quantizer state can be reconstructed
/// without storing the matrix — only the seed and dimension need to be persisted.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredRotation {
    dim: usize,
    seed: u64,
    /// Row-major storage of the d×d orthogonal matrix.
    matrix: Vec<f32>,
}

impl StoredRotation {
    /// Generate a new rotation for vectors of dimension `dim` using `seed`.
    pub fn new(dim: usize, seed: u64) -> Result<Self> {
        if dim == 0 {
            return Err(QuantError::ZeroDimension);
        }
        let matrix = generate_orthogonal(dim, seed)?;
        Ok(Self { dim, seed, matrix })
    }

    /// Apply rotation: y = R · x.
    pub fn apply(&self, input: &[f32], output: &mut [f32]) -> Result<()> {
        check_dim(input.len(), self.dim)?;
        check_dim(output.len(), self.dim)?;
        for (i, out) in output.iter_mut().enumerate() {
            let row_start = i * self.dim;
            let row = &self.matrix[row_start..row_start + self.dim];
            *out = row.iter().zip(input.iter()).map(|(r, x)| r * x).sum();
        }
        Ok(())
    }

    /// Apply inverse (transpose) rotation: x = R^T · y.
    pub fn apply_inverse(&self, input: &[f32], output: &mut [f32]) -> Result<()> {
        check_dim(input.len(), self.dim)?;
        check_dim(output.len(), self.dim)?;
        // R^T is column-major access of row-major matrix
        for (i, out) in output.iter_mut().enumerate() {
            let mut sum = 0.0f32;
            #[allow(clippy::needless_range_loop)]
            for j in 0..self.dim {
                sum += self.matrix[j * self.dim + i] * input[j];
            }
            *out = sum;
        }
        Ok(())
    }

    pub fn dim(&self) -> usize { self.dim }
    pub fn seed(&self) -> u64 { self.seed }
    pub fn memory_bytes(&self) -> usize { self.dim * self.dim * 4 }
}

/// Generate d×d orthogonal matrix via Modified Gram-Schmidt on random Gaussian.
fn generate_orthogonal(dim: usize, seed: u64) -> Result<Vec<f32>> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // Generate random Gaussian matrix (d columns, stored as column-major Vec<Vec<f32>>)
    let mut columns: Vec<Vec<f32>> = (0..dim)
        .map(|_| (0..dim).map(|_| StandardNormal.sample(&mut rng)).collect())
        .collect();

    // Modified Gram-Schmidt orthogonalization
    for i in 0..dim {
        // Normalize column i
        let norm: f32 = columns[i].iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm < 1e-10 {
            return Err(QuantError::RotationFailed(format!("zero norm at column {}", i)));
        }
        for v in columns[i].iter_mut() {
            *v /= norm;
        }

        // Remove projection from remaining columns
        for j in (i + 1)..dim {
            let dot: f32 = columns[i].iter().zip(columns[j].iter()).map(|(a, b)| a * b).sum();
            #[allow(clippy::needless_range_loop)]
            for k in 0..dim {
                columns[j][k] -= dot * columns[i][k];
            }
        }
    }

    // Fix signs from implicit R diagonal (ensure det = +1)
    // With MGS, the R diagonal is always positive, so Q is already a proper rotation.

    // Convert column-major to row-major
    let mut matrix = vec![0.0f32; dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            matrix[i * dim + j] = columns[j][i];
        }
    }

    Ok(matrix)
}

fn check_dim(got: usize, expected: usize) -> Result<()> {
    if got != expected {
        Err(QuantError::DimensionMismatch { expected, got })
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotation_is_deterministic() {
        let r1 = StoredRotation::new(8, 42).unwrap();
        let r2 = StoredRotation::new(8, 42).unwrap();
        assert_eq!(r1.matrix, r2.matrix);
    }

    #[test]
    fn rotation_is_orthogonal() {
        let r = StoredRotation::new(16, 7).unwrap();
        let d = r.dim;
        // Check R^T R = I
        for i in 0..d {
            for j in 0..d {
                let mut dot = 0.0f32;
                for k in 0..d {
                    dot += r.matrix[k * d + i] * r.matrix[k * d + j];
                }
                let expected = if i == j { 1.0f32 } else { 0.0f32 };
                assert!((dot - expected).abs() < 1e-4, "R^TR[{i},{j}] = {dot}, expected {expected}");
            }
        }
    }

    #[test]
    fn apply_inverse_recovers_input() {
        let r = StoredRotation::new(8, 99).unwrap();
        let x = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut y = vec![0.0f32; 8];
        let mut recovered = vec![0.0f32; 8];
        r.apply(&x, &mut y).unwrap();
        r.apply_inverse(&y, &mut recovered).unwrap();
        for (orig, rec) in x.iter().zip(recovered.iter()) {
            assert!((orig - rec).abs() < 1e-4, "orig={orig}, recovered={rec}");
        }
    }

    #[test]
    fn rotation_preserves_inner_products() {
        let r = StoredRotation::new(8, 13).unwrap();
        let x = vec![1.0f32, 0.5, -1.0, 2.0, 0.1, -0.3, 1.5, 0.8];
        let y = vec![0.2f32, -1.0, 0.5, 1.0, -0.5, 0.3, 0.9, -0.7];
        let mut rx = vec![0.0f32; 8];
        let mut ry = vec![0.0f32; 8];
        r.apply(&x, &mut rx).unwrap();
        r.apply(&y, &mut ry).unwrap();
        let ip_orig: f32 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
        let ip_rot: f32 = rx.iter().zip(ry.iter()).map(|(a, b)| a * b).sum();
        assert!((ip_orig - ip_rot).abs() < 1e-3, "orig={ip_orig}, rotated={ip_rot}");
    }
}
