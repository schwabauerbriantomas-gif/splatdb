//! TurboQuant: two-stage vector compression combining PolarQuant and QJL.
//!
//! 1. PolarQuant stage (bits-1): Compress vector via polar encoding.
//! 2. QJL stage (1 bit per projection): Sketch the residual for unbiased correction.
//!
//! Combined estimator: <x, y> ~ IP_polar + IP_qjl(residual)
//! This is provably unbiased (TurboQuant paper, ICLR 2026).
//!
//! License: MIT (adapted from github.com/RecursiveIntell/turbo-quant)

use serde::{Deserialize, Serialize};
use rayon::prelude::*;

use super::error::{Result, QuantError};
use super::polar::{PolarCode, PolarQuantizer};
use super::qjl::{QjlQuantizer, QjlSketch};

/// A TurboQuant-compressed vector: polar code + QJL residual sketch.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TurboCode {
    pub polar_code: PolarCode,
    pub residual_sketch: QjlSketch,
}

impl TurboCode {
    pub fn encoded_bytes(&self) -> usize {
        self.polar_code.encoded_bytes() + self.residual_sketch.encoded_bytes()
    }

    pub fn compression_ratio(&self) -> f32 {
        let original = self.polar_code.dim * std::mem::size_of::<f32>();
        original as f32 / self.encoded_bytes().max(1) as f32
    }
}

/// TurboQuant compressor: encodes vectors and estimates inner products.
///
/// Configuration `(dim, bits, projections, seed)` fully determines state.
/// Only these four values need to be persisted.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurboQuantizer {
    dim: usize,
    bits: u8,
    projections: usize,
    seed: u64,
    polar: PolarQuantizer,
    qjl: QjlQuantizer,
}

impl TurboQuantizer {
    pub fn new(dim: usize, bits: u8, projections: usize, seed: u64) -> Result<Self> {
        if dim == 0 { return Err(QuantError::ZeroDimension); }
        if !dim.is_multiple_of(2) { return Err(QuantError::OddDimension { got: dim }); }
        if !(2..=16).contains(&bits) { return Err(QuantError::InvalidBitWidth { got: bits }); }
        if projections == 0 { return Err(QuantError::ZeroProjectionCount); }

        let polar_seed = seed;
        let qjl_seed = seed.wrapping_add(0xCAFE_BABE_0000_0001);

        let polar = PolarQuantizer::new(dim, bits - 1, polar_seed)?;
        let qjl = QjlQuantizer::new(dim, projections, qjl_seed)?;

        Ok(Self { dim, bits, projections, seed, polar, qjl })
    }

    pub fn dim(&self) -> usize { self.dim }
    pub fn bits(&self) -> u8 { self.bits }
    pub fn projections(&self) -> usize { self.projections }

    /// Encode a vector into a TurboCode.
    pub fn encode(&self, vector: &[f32]) -> Result<TurboCode> {
        if vector.len() != self.dim {
            return Err(QuantError::DimensionMismatch { expected: self.dim, got: vector.len() });
        }

        let polar_code = self.polar.encode(vector)?;
        let reconstruction = self.polar.decode(&polar_code)?;
        let residual: Vec<f32> = vector.iter().zip(reconstruction.iter()).map(|(o, r)| o - r).collect();
        let residual_sketch = self.qjl.sketch(&residual)?;

        Ok(TurboCode { polar_code, residual_sketch })
    }

    /// Encode a batch of vectors in parallel.
    pub fn encode_batch(&self, vectors: &[Vec<f32>]) -> Vec<Result<TurboCode>> {
        vectors.par_iter().map(|v| self.encode(v)).collect()
    }

    /// Estimate <original_vector, query> from a TurboCode and raw query.
    /// Provably unbiased.
    pub fn inner_product_estimate(&self, code: &TurboCode, query: &[f32]) -> Result<f32> {
        if query.len() != self.dim {
            return Err(QuantError::DimensionMismatch { expected: self.dim, got: query.len() });
        }
        let polar_est = self.polar.inner_product_estimate(&code.polar_code, query)?;
        let qjl_correction = self.qjl.inner_product_estimate(&code.residual_sketch, query)?;
        Ok(polar_est + qjl_correction)
    }

    /// Estimate squared L2 distance.
    pub fn l2_distance_estimate(&self, code: &TurboCode, query: &[f32]) -> Result<f32> {
        let ip = self.inner_product_estimate(code, query)?;
        let query_norm_sq: f32 = query.iter().map(|x| x * x).sum();
        let code_norm_sq: f32 = code.polar_code.radii.iter().map(|r| r * r).sum();
        Ok((query_norm_sq + code_norm_sq - 2.0 * ip).max(0.0))
    }

    /// Decode to approximate reconstruction (PolarQuant component only).
    pub fn decode_approximate(&self, code: &TurboCode) -> Result<Vec<f32>> {
        self.polar.decode(&code.polar_code)
    }
}

/// Compression statistics for a batch.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BatchStats {
    pub count: usize,
    pub total_encoded_bytes: usize,
    pub total_original_bytes: usize,
    pub compression_ratio: f32,
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
    fn encode_is_deterministic() {
        let q = TurboQuantizer::new(16, 8, 16, 42).unwrap();
        let x = random_vector(16, 1);
        let c1 = q.encode(&x).unwrap();
        let c2 = q.encode(&x).unwrap();
        assert_eq!(c1.polar_code, c2.polar_code);
        assert_eq!(c1.residual_sketch, c2.residual_sketch);
    }

    #[test]
    fn nearest_neighbor_ordering_preserved() {
        let q = TurboQuantizer::new(16, 8, 16, 7).unwrap();
        let query = random_vector(16, 99);
        let close: Vec<f32> = query.iter().map(|x| x + 0.05).collect();
        let far1 = random_vector(16, 200);
        let far2 = random_vector(16, 201);

        let ip_close = q.inner_product_estimate(&q.encode(&close).unwrap(), &query).unwrap();
        let ip_far1 = q.inner_product_estimate(&q.encode(&far1).unwrap(), &query).unwrap();
        let ip_far2 = q.inner_product_estimate(&q.encode(&far2).unwrap(), &query).unwrap();

        assert!(ip_close > ip_far1 && ip_close > ip_far2,
            "close={ip_close:.3}, far1={ip_far1:.3}, far2={ip_far2:.3}");
    }

    #[test]
    fn compression_ratio_positive() {
        let q = TurboQuantizer::new(64, 8, 32, 0).unwrap();
        let x = random_vector(64, 1);
        let code = q.encode(&x).unwrap();
        assert!(code.compression_ratio() > 1.0);
    }

    #[test]
    fn invalid_config_rejected() {
        assert!(TurboQuantizer::new(0, 8, 16, 0).is_err());
        assert!(TurboQuantizer::new(7, 8, 16, 0).is_err());
        assert!(TurboQuantizer::new(8, 1, 16, 0).is_err());
        assert!(TurboQuantizer::new(8, 8, 0, 0).is_err());
    }
}
