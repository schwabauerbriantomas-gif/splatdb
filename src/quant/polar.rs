//! PolarQuant: vector compression via polar coordinate encoding.
//!
//! Converts Cartesian pairs into polar form (radius, angle), then uniformly
//! quantizes angles. After rotation, angles are uniformly distributed on [-pi, pi],
//! making uniform quantization near-optimal with no calibration needed.
//!
//! Algorithm: Given rotated vector y = R·x, group into d/2 pairs,
//! convert to polar (r, theta), quantize theta to `bits` levels.
//!
//! License: MIT (adapted from github.com/RecursiveIntell/turbo-quant)

use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

use super::error::{QuantError, Result};
use super::rotation::StoredRotation;

/// A compressed vector in polar form.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PolarCode {
    pub dim: usize,
    pub bits: u8,
    /// Per-pair radii (f32, lossless).
    pub radii: Vec<f32>,
    /// Quantized angle indices in [0, 2^bits).
    pub angle_indices: Vec<u16>,
}

impl PolarCode {
    pub fn pair_count(&self) -> usize {
        self.dim / 2
    }

    pub fn dequantize_angle(&self, i: usize) -> f32 {
        let levels = 1u32 << self.bits;
        (self.angle_indices[i] as f32 / levels as f32) * (2.0 * PI) - PI
    }

    pub fn encoded_bytes(&self) -> usize {
        self.pair_count() * 4 + self.pair_count() * 2
    }
}

/// Encodes/decodes vectors using PolarQuant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolarQuantizer {
    dim: usize,
    bits: u8,
    rotation: StoredRotation,
}

impl PolarQuantizer {
    pub fn new(dim: usize, bits: u8, seed: u64) -> Result<Self> {
        if dim == 0 {
            return Err(QuantError::ZeroDimension);
        }
        if !dim.is_multiple_of(2) {
            return Err(QuantError::OddDimension { got: dim });
        }
        if bits == 0 || bits > 16 {
            return Err(QuantError::InvalidBitWidth { got: bits });
        }
        let rotation = StoredRotation::new(dim, seed)?;
        Ok(Self {
            dim,
            bits,
            rotation,
        })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }
    pub fn bits(&self) -> u8 {
        self.bits
    }

    pub fn encode(&self, vector: &[f32]) -> Result<PolarCode> {
        if vector.len() != self.dim {
            return Err(QuantError::DimensionMismatch {
                expected: self.dim,
                got: vector.len(),
            });
        }
        let mut rotated = vec![0.0f32; self.dim];
        self.rotation.apply(vector, &mut rotated)?;

        let pairs = self.dim / 2;
        let mut radii = Vec::with_capacity(pairs);
        let mut angle_indices = Vec::with_capacity(pairs);
        for i in 0..pairs {
            let a = rotated[2 * i];
            let b = rotated[2 * i + 1];
            let (r, idx) = encode_pair(a, b, self.bits);
            radii.push(r);
            angle_indices.push(idx);
        }
        Ok(PolarCode {
            dim: self.dim,
            bits: self.bits,
            radii,
            angle_indices,
        })
    }

    pub fn decode(&self, code: &PolarCode) -> Result<Vec<f32>> {
        self.validate_code(code)?;
        let mut rotated = vec![0.0f32; self.dim];
        for i in 0..self.dim / 2 {
            let theta = code.dequantize_angle(i);
            let r = code.radii[i];
            rotated[2 * i] = r * theta.cos();
            rotated[2 * i + 1] = r * theta.sin();
        }
        let mut output = vec![0.0f32; self.dim];
        self.rotation.apply_inverse(&rotated, &mut output)?;
        Ok(output)
    }

    pub fn inner_product_estimate(&self, code: &PolarCode, query: &[f32]) -> Result<f32> {
        if query.len() != self.dim {
            return Err(QuantError::DimensionMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }
        self.validate_code(code)?;
        let mut rotated_query = vec![0.0f32; self.dim];
        self.rotation.apply(query, &mut rotated_query)?;

        let pairs = self.dim / 2;
        let mut estimate = 0.0f32;
        for i in 0..pairs {
            let theta = code.dequantize_angle(i);
            let r = code.radii[i];
            let q_a = rotated_query[2 * i];
            let q_b = rotated_query[2 * i + 1];
            estimate += r * (q_a * theta.cos() + q_b * theta.sin());
        }
        Ok(estimate)
    }

    fn validate_code(&self, code: &PolarCode) -> Result<()> {
        if code.dim != self.dim {
            return Err(QuantError::DimensionMismatch {
                expected: self.dim,
                got: code.dim,
            });
        }
        if code.bits != self.bits {
            return Err(QuantError::MalformedCode(format!(
                "code bits={}, quantizer bits={}",
                code.bits, self.bits
            )));
        }
        Ok(())
    }
}

fn encode_pair(a: f32, b: f32, bits: u8) -> (f32, u16) {
    let r = (a * a + b * b).sqrt();
    let theta = b.atan2(a);
    let levels = 1u32 << bits;
    let normalized = (theta + PI) / (2.0 * PI);
    let idx = (normalized * levels as f32).floor() as u32 % levels;
    (r, idx as u16)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_vector(dim: usize, seed: u64) -> Vec<f32> {
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;
        use rand_distr::{Distribution, StandardNormal};
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        (0..dim).map(|_| StandardNormal.sample(&mut rng)).collect()
    }

    #[test]
    fn encode_decode_roundtrip() {
        let q = PolarQuantizer::new(8, 16, 42).unwrap();
        let x = vec![1.0f32, 2.0, -1.5, 0.5, 3.0, -2.0, 0.1, -0.8];
        let code = q.encode(&x).unwrap();
        let decoded = q.decode(&code).unwrap();
        for (orig, dec) in x.iter().zip(decoded.iter()) {
            assert!(
                (orig - dec).abs() < 0.01,
                "orig={orig:.4}, decoded={dec:.4}"
            );
        }
    }

    #[test]
    fn inner_product_accurate_at_high_bits() {
        let q = PolarQuantizer::new(16, 16, 7).unwrap();
        let x = random_vector(16, 1);
        let y = random_vector(16, 2);
        let code = q.encode(&x).unwrap();
        let estimated = q.inner_product_estimate(&code, &y).unwrap();
        let exact: f32 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
        let rel_error = (estimated - exact).abs() / (exact.abs() + 1e-6);
        assert!(
            rel_error < 0.02,
            "rel_error={rel_error:.4}, est={estimated:.4}, exact={exact:.4}"
        );
    }

    #[test]
    fn deterministic() {
        let q = PolarQuantizer::new(8, 8, 0).unwrap();
        let x = vec![1.0f32; 8];
        let c1 = q.encode(&x).unwrap();
        let c2 = q.encode(&x).unwrap();
        assert_eq!(c1.angle_indices, c2.angle_indices);
        assert_eq!(c1.radii, c2.radii);
    }

    #[test]
    fn odd_dim_rejected() {
        assert!(PolarQuantizer::new(7, 8, 0).is_err());
    }
}
