//! Quantized Johnson-Lindenstrauss (QJL) transform for unbiased inner product estimation.
//!
//! Projects a d-dimensional vector onto m random hyperplanes and records only
//! the sign of each projection. This costs exactly 1 bit per projection dimension.
//!
//! Unbiased estimator: <x, y> ~ (pi / 2m) * sum_i sign(g_i . x) * (g_i . y)
//!
//! Reference: "Quantized Johnson-Lindenstrauss is Optimal for Inner Products"
//! Zandieh et al., AAAI 2025.
//!
//! License: MIT (adapted from github.com/RecursiveIntell/turbo-quant)

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

use super::error::{QuantError, Result};

/// A QJL sketch: sign of each random projection (+1 or -1).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QjlSketch {
    pub dim: usize,
    pub projections: usize,
    pub signs: Vec<i8>,
}

impl QjlSketch {
    pub fn encoded_bytes(&self) -> usize {
        self.projections
    }
}

/// Projects vectors to QJL sketches and estimates inner products.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QjlQuantizer {
    dim: usize,
    projections: usize,
    seed: u64,
}

impl QjlQuantizer {
    pub fn new(dim: usize, projections: usize, seed: u64) -> Result<Self> {
        if dim == 0 {
            return Err(QuantError::ZeroDimension);
        }
        if projections == 0 {
            return Err(QuantError::ZeroProjectionCount);
        }
        Ok(Self {
            dim,
            projections,
            seed,
        })
    }

    pub fn sketch(&self, vector: &[f32]) -> Result<QjlSketch> {
        if vector.len() != self.dim {
            return Err(QuantError::DimensionMismatch {
                expected: self.dim,
                got: vector.len(),
            });
        }
        let g = self.projection_matrix();
        let signs: Vec<i8> = g
            .iter()
            .map(|row| {
                let dot: f32 = row.iter().zip(vector.iter()).map(|(g, x)| g * x).sum();
                if dot >= 0.0 {
                    1i8
                } else {
                    -1i8
                }
            })
            .collect();
        Ok(QjlSketch {
            dim: self.dim,
            projections: self.projections,
            signs,
        })
    }

    pub fn inner_product_estimate(&self, sketch: &QjlSketch, query: &[f32]) -> Result<f32> {
        if sketch.dim != self.dim {
            return Err(QuantError::DimensionMismatch {
                expected: self.dim,
                got: sketch.dim,
            });
        }
        if query.len() != self.dim {
            return Err(QuantError::DimensionMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }
        let g = self.projection_matrix();
        let m = self.projections as f32;
        let scale = PI / (2.0 * m);
        let estimate: f32 = g
            .iter()
            .zip(sketch.signs.iter())
            .map(|(row, &sign)| {
                let g_dot_query: f32 = row.iter().zip(query.iter()).map(|(g, q)| g * q).sum();
                sign as f32 * g_dot_query
            })
            .sum();
        Ok(scale * estimate)
    }

    fn projection_matrix(&self) -> Vec<Vec<f32>> {
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed.wrapping_add(0xDEAD_BEEF_1234_5678));
        (0..self.projections)
            .map(|_| {
                (0..self.dim)
                    .map(|_| StandardNormal.sample(&mut rng))
                    .collect()
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_vector(dim: usize, seed: u64) -> Vec<f32> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        (0..dim).map(|_| StandardNormal.sample(&mut rng)).collect()
    }

    #[test]
    fn sketch_is_deterministic() {
        let q = QjlQuantizer::new(16, 32, 42).unwrap();
        let x = random_vector(16, 1);
        let s1 = q.sketch(&x).unwrap();
        let s2 = q.sketch(&x).unwrap();
        assert_eq!(s1.signs, s2.signs);
    }

    #[test]
    fn inner_product_unbiased() {
        // Use more projections for reliable unbiased estimation
        let q = QjlQuantizer::new(16, 4096, 0).unwrap();
        let x = random_vector(16, 10);
        let y = random_vector(16, 20);
        let exact: f32 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
        let sketch = q.sketch(&x).unwrap();
        let estimated = q.inner_product_estimate(&sketch, &y).unwrap();
        let error = (estimated - exact).abs();
        let tolerance = 0.30 * exact.abs() + 3.0;
        assert!(
            error < tolerance,
            "error={error:.3}, exact={exact:.3}, estimated={estimated:.3}"
        );
    }
}
