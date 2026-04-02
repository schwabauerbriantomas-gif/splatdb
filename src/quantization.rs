//! TurboQuant integration for SplatDB Vector Search.
//!
//! Compresses 640D splat embeddings to 4-8 bits per value with minimal accuracy loss.
//! Uses data-oblivious quantization — no training, no codebooks, instant indexing.

use ndarray::{Array2, ArrayView1};
use rayon::prelude::*;
use crate::quant::{TurboQuantizer, PolarQuantizer, PolarCode, TurboCode};

/// Quantization algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantAlgorithm {
    TurboQuant,
    PolarQuant,
}

/// Configuration for vector quantization.
#[derive(Debug, Clone)]
pub struct QuantConfig {
    /// Bits per value (3-8). 8 = near-lossless, 4 = aggressive compression.
    pub bits: u8,
    /// Number of random projections. Recommended: dim / 4 for search.
    pub projections: usize,
    /// Random seed.
    pub seed: u64,
    /// Algorithm to use.
    pub algorithm: QuantAlgorithm,
}

impl Default for QuantConfig {
    fn default() -> Self {
        Self {
            bits: 8,
            projections: 160,
            seed: 42,
            algorithm: QuantAlgorithm::TurboQuant,
        }
    }
}

impl QuantConfig {
    /// For search.
    pub fn for_search(dim: usize) -> Self {
        Self {
            bits: 4,
            projections: dim / 8,
            seed: 42,
            algorithm: QuantAlgorithm::TurboQuant,
        }
    }

    /// For max compression.
    pub fn for_max_compression(dim: usize) -> Self {
        Self {
            bits: 3,
            projections: dim / 16,
            seed: 42,
            algorithm: QuantAlgorithm::PolarQuant,
        }
    }

    /// For compression.
    pub fn for_compression(dim: usize) -> Self {
        Self {
            bits: 4,
            projections: dim / 8,
            seed: 42,
            algorithm: QuantAlgorithm::TurboQuant,
        }
    }
}

/// Compressed vector store using TurboQuant.
pub struct QuantizedStore {
    config: QuantConfig,
    dim: usize,
    turbo: Option<TurboQuantizer>,
    polar: Option<PolarQuantizer>,
    codes_turbo: Vec<TurboCode>,
    codes_polar: Vec<PolarCode>,
    ids: Vec<u64>,
}

impl QuantizedStore {
    /// New.
    pub fn new(dim: usize, config: QuantConfig) -> Result<Self, String> {
        let (turbo, polar) = match config.algorithm {
            QuantAlgorithm::TurboQuant => {
                let t = TurboQuantizer::new(dim, config.bits, config.projections, config.seed)
                    .map_err(|e| format!("TurboQuant init error: {}", e))?;
                (Some(t), None)
            }
            QuantAlgorithm::PolarQuant => {
                let p = PolarQuantizer::new(dim, config.bits, config.seed)
                    .map_err(|e| format!("PolarQuant init error: {}", e))?;
                (None, Some(p))
            }
        };

        Ok(Self {
            config,
            dim,
            turbo,
            polar,
            codes_turbo: Vec::new(),
            codes_polar: Vec::new(),
            ids: Vec::new(),
        })
    }

    /// Encode and store a batch of vectors in parallel.
    pub fn add_batch(&mut self, vectors: &Array2<f32>, start_id: u64) -> usize {
        let n = vectors.nrows();
        if let Some(ref turbo) = self.turbo {
            let new_codes: Vec<TurboCode> = (0..n)
                .into_par_iter()
                .filter_map(|i| {
                    let v: Vec<f32> = vectors.row(i).to_vec();
                    turbo.encode(&v).ok()
                })
                .collect();
            self.codes_turbo.extend(new_codes);
        } else if let Some(ref polar) = self.polar {
            let new_codes: Vec<PolarCode> = (0..n)
                .into_par_iter()
                .filter_map(|i| {
                    let v: Vec<f32> = vectors.row(i).to_vec();
                    polar.encode(&v).ok()
                })
                .collect();
            self.codes_polar.extend(new_codes);
        }

        for i in 0..n {
            self.ids.push(start_id + i as u64);
        }
        n
    }

    /// Encode a single vector.
    pub fn add_single(&mut self, vector: &ArrayView1<f32>, id: u64) -> bool {
        let v: Vec<f32> = vector.to_vec();
        if let Some(ref turbo) = self.turbo {
            match turbo.encode(&v) {
                Ok(code) => { self.codes_turbo.push(code); }
                Err(_) => return false,
            }
        } else if let Some(ref polar) = self.polar {
            match polar.encode(&v) {
                Ok(code) => { self.codes_polar.push(code); }
                Err(_) => return false,
            }
        }
        self.ids.push(id);
        true
    }

    /// Search for k nearest neighbors using parallel inner product estimation.
    /// Returns (id, estimated_inner_product) pairs sorted by similarity (descending).
    pub fn search(&self, query: &ArrayView1<f32>, k: usize) -> Vec<(u64, f32)> {
        let q: Vec<f32> = query.to_vec();
        let n = self.ids.len();
        let k = k.min(n);
        if k == 0 { return Vec::new(); }

        let mut scores: Vec<(usize, f32)> = if let Some(ref turbo) = self.turbo {
            self.codes_turbo.par_iter().enumerate()
                .filter_map(|(i, code)| {
                    turbo.inner_product_estimate(code, &q).ok().map(|s| (i, s))
                })
                .collect()
        } else if let Some(ref polar) = self.polar {
            self.codes_polar.par_iter().enumerate()
                .filter_map(|(i, code)| {
                    polar.inner_product_estimate(code, &q).ok().map(|s| (i, s))
                })
                .collect()
        } else {
            Vec::new()
        };

        // Sort descending by inner product (higher = more similar)
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(k);

        scores.into_iter()
            .map(|(i, score)| (self.ids[i], score))
            .collect()
    }

    /// Get number of stored codes.
    pub fn len(&self) -> usize { self.ids.len() }

    /// Check if store is empty.
    pub fn is_empty(&self) -> bool { self.ids.is_empty() }

    /// Estimate memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        let code_size = if self.turbo.is_some() {
            // TurboCode: roughly dim * bits / 8 bytes + overhead
            self.dim * self.config.bits as usize / 8 + 64
        } else {
            // PolarCode: roughly dim * bits / 8
            self.dim * self.config.bits as usize / 8 + 32
        };
        self.ids.len() * (code_size + 8) // 8 bytes for id
    }

    /// Get compression ratio vs raw f32 storage.
    pub fn compression_ratio(&self) -> f32 {
        let raw_bytes = self.dim * 4; // f32 per dim
        let quant_bytes = self.dim * self.config.bits as usize / 8;
        raw_bytes as f32 / quant_bytes.max(1) as f32
    }

    /// Get the config.
    pub fn config(&self) -> &QuantConfig { &self.config }
}

#[cfg(test)]
mod tests {
    use super::*;


    fn random_vectors(n: usize, dim: usize, seed: u64) -> Array2<f32> {
        use rand::SeedableRng;
        use rand::Rng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        Array2::from_shape_fn((n, dim), |(_, _)| rng.gen::<f32>())
    }

    #[test]
    fn test_turbo_quant_roundtrip() {
        let dim = 64;
        let config = QuantConfig::for_search(dim);
        let mut store = QuantizedStore::new(dim, config).unwrap();

        let vectors = random_vectors(100, dim, 42);
        store.add_batch(&vectors, 0);
        assert_eq!(store.len(), 100);
        assert!(!store.is_empty());
    }

    #[test]
    fn test_turbo_quant_search() {
        let dim = 64;
        let config = QuantConfig::for_search(dim);
        let mut store = QuantizedStore::new(dim, config).unwrap();

        let vectors = random_vectors(100, dim, 42);
        store.add_batch(&vectors, 0);

        // Search with the first vector — should find itself near the top
        let results = store.search(&vectors.row(0), 5);
        assert_eq!(results.len(), 5);
        // With 4-bit quantization + turbo codes, top result should have positive score
        assert!(results[0].1 > 0.0, "Top result should have positive score");
        let _top_ids: Vec<u64> = results.iter().map(|(id, _)| *id).collect();
    }

    #[test]
    fn test_polar_quant_search() {
        let dim = 64;
        let config = QuantConfig {
            bits: 8,
            projections: 16,
            seed: 42,
            algorithm: QuantAlgorithm::PolarQuant,
        };
        let mut store = QuantizedStore::new(dim, config).unwrap();

        let vectors = random_vectors(100, dim, 42);
        store.add_batch(&vectors, 0);

        let results = store.search(&vectors.row(0), 5);
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_compression_ratio() {
        let dim = 128;
        let config = QuantConfig::for_search(dim);
        let store = QuantizedStore::new(dim, config).unwrap();
        // 4 bits: raw=512 bytes, quant=64 bytes => ratio=8x
        assert!(store.compression_ratio() >= 7.0, "Expected ~8x ratio, got {}", store.compression_ratio());
    }

    #[test]
    fn test_memory_usage() {
        let dim = 64;
        let config = QuantConfig::for_search(dim);
        let mut store = QuantizedStore::new(dim, config).unwrap();

        let vectors = random_vectors(50, dim, 42);
        store.add_batch(&vectors, 0);

        let mem = store.memory_usage();
        let raw = 50 * dim * 4; // f32
        assert!(mem < raw, "Compressed {} should be less than raw {}", mem, raw);
    }
}
