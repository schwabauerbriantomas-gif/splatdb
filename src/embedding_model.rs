//! Embedding model wrapper for SplatDB.
//!
//! Supports external encoders via trait. Includes Matryoshka representations,
//! projection head, and distillation loss utilities.
//! Ported from splatdb Python.

use ndarray::{Array1, Array2};

/// Encoder trait — implement for any embedding model.
pub trait Encoder: Send + Sync {
    /// Encode a single text, returning embedding vector.
    fn encode_one(&self, text: &str) -> Result<Array1<f32>, Box<dyn std::error::Error + Send + Sync>>;

    /// Encode multiple texts, returning embedding matrix [N, D].
    fn encode_batch(&self, texts: &[&str]) -> Result<Array2<f32>, Box<dyn std::error::Error + Send + Sync>> {
        let mut rows = Vec::with_capacity(texts.len());
        for text in texts {
            rows.push(self.encode_one(text)?);
        }
        let dim = rows[0].len();
        let mut data = Vec::with_capacity(texts.len() * dim);
        for row in &rows {
            data.extend_from_slice(row.as_slice().expect("ndarray Array1 should be contiguous"));
        }
        Array2::from_shape_vec((texts.len(), dim), data)
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)
    }

    /// Get the embedding dimension.
    fn dim(&self) -> usize;

    /// Get the model name.
    fn model_name(&self) -> &str;
}

/// Matryoshka embedding dimensions (default: 64, 128, 256, 640).
pub const DEFAULT_MATRYOSHKA_DIMS: &[usize] = &[64, 128, 256, 640];

/// Truncate embedding to a Matryoshka dimension and normalize.
pub fn matryoshka_truncate(embedding: &[f32], dim: usize) -> Vec<f32> {
    let d = dim.min(embedding.len());
    let mut out = embedding[..d].to_vec();
    // Normalize
    let norm: f32 = out.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 1e-8 { for v in out.iter_mut() { *v /= norm; } }
    out
}

/// Generate all Matryoshka representations for an embedding.
pub fn matryoshka_all(embedding: &[f32], dims: &[usize]) -> Vec<Vec<f32>> {
    dims.iter().map(|&d| matryoshka_truncate(embedding, d)).collect()
}

/// Compute cosine similarity between two embeddings.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let na: f32 = a.iter().map(|v| v * v).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|v| v * v).sum::<f32>().sqrt();
    let denom = na * nb;
    if denom < 1e-8 { 0.0 } else { dot / denom }
}

/// Simple projection head: linear projection + normalization.
pub struct ProjectionHead {
    weight: Vec<Vec<f32>>,  // [output_dim x input_dim]
    bias: Vec<f32>,         // [output_dim]
    input_dim: usize,
    output_dim: usize,
}

impl ProjectionHead {
    /// Create a new projection head with random initialization.
    pub fn new(input_dim: usize, output_dim: usize, seed: u64) -> Self {
        let mut rng = seed;
        let scale = (2.0 / input_dim as f64).sqrt() as f32;
        let weight: Vec<Vec<f32>> = (0..output_dim).map(|_| {
            (0..input_dim).map(|_| {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                let val = ((rng >> 33) as f64 / (1u64 << 31) as f64) as f32;
                val * 2.0 * scale - scale
            }).collect()
        }).collect();
        let bias = vec![0.0f32; output_dim];
        Self { weight, bias, input_dim, output_dim }
    }

    /// Project input vector to output space and normalize.
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; self.output_dim];
        #[allow(clippy::needless_range_loop)]
        for j in 0..self.output_dim {
            let dot: f32 = input.iter().zip(self.weight[j].iter()).map(|(&x, &w)| x * w).sum();
            out[j] = dot + self.bias[j];
        }
        // Normalize
        let norm: f32 = out.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 1e-8 { for v in out.iter_mut() { *v /= norm; } }
        out
    }

    /// Project and return Matryoshka representations.
    pub fn forward_matryoshka(&self, input: &[f32], dims: &[usize]) -> Vec<Vec<f32>> {
        let full = self.forward(input);
        dims.iter().map(|&d| matryoshka_truncate(&full, d)).collect()
    }

    /// Output dim.
    pub fn output_dim(&self) -> usize { self.output_dim }
    /// Input dim.
    pub fn input_dim(&self) -> usize { self.input_dim }
}

/// Distillation loss components.
#[derive(Debug, Clone)]
pub struct DistillationLoss {
    pub mse: f32,
    pub cosine: f32,
    pub matryoshka: f32,
    pub total: f32,
}

/// Compute distillation loss between student and teacher embeddings.
pub fn compute_distillation_loss(
    student: &[f32],
    teacher: &[f32],
    dims: &[usize],
    mse_weight: f32,
    cosine_weight: f32,
    matryoshka_weight: f32,
) -> DistillationLoss {
    // Cosine loss: 1 - cosine_similarity
    let cos_sim = cosine_similarity(student, teacher);
    let cosine = 1.0 - cos_sim;

    // MSE loss
    let min_dim = student.len().min(teacher.len());
    let mse: f32 = student[..min_dim].iter().zip(teacher[..min_dim].iter())
        .map(|(&s, &t)| { let d = s - t; d * d })
        .sum::<f32>() / min_dim.max(1) as f32;

    // Matryoshka loss: average cosine loss at sub-dimensions
    let matryoshka: f32 = if dims.is_empty() {
        0.0
    } else {
        dims.iter().map(|&d| {
            let s = matryoshka_truncate(student, d);
            let t = matryoshka_truncate(teacher, d);
            1.0 - cosine_similarity(&s, &t)
        }).sum::<f32>() / dims.len() as f32
    };

    let total = mse_weight * mse + cosine_weight * cosine + matryoshka_weight * matryoshka;

    DistillationLoss { mse, cosine, matryoshka, total }
}

/// Simple hash-based encoder for testing (deterministic, no model needed).
pub struct HashEncoder {
    dim: usize,
}

impl HashEncoder {
    /// New.
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl Encoder for HashEncoder {
    fn encode_one(&self, text: &str) -> Result<Array1<f32>, Box<dyn std::error::Error + Send + Sync>> {
        let mut data = vec![0.0f32; self.dim];
        let bytes = text.as_bytes();

        #[allow(clippy::needless_range_loop)]
        for i in 0..self.dim {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            let mut hasher = DefaultHasher::new();
            (i, bytes).hash(&mut hasher);
            let h = hasher.finish();
            data[i] = ((h % 10000) as f32) / 5000.0 - 1.0;
        }

        let norm: f32 = data.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 0.0 { for v in data.iter_mut() { *v /= norm; } }

        Ok(Array1::from_vec(data))
    }

    fn dim(&self) -> usize { self.dim }
    fn model_name(&self) -> &str { "hash-encoder" }
}

/// Encoder registry for managing multiple models.
pub struct EncoderRegistry {
    encoders: Vec<(String, Box<dyn Encoder>)>,
    default: Option<String>,
}

impl EncoderRegistry {
    /// New.
    pub fn new() -> Self {
        Self { encoders: Vec::new(), default: None }
    }

    /// Register.
    pub fn register(&mut self, name: &str, encoder: Box<dyn Encoder>, set_default: bool) {
        if set_default { self.default = Some(name.to_string()); }
        self.encoders.push((name.to_string(), encoder));
    }

    /// Get.
    pub fn get(&self, name: &str) -> Option<&dyn Encoder> {
        self.encoders.iter().find(|(n, _)| n == name).map(|(_, e)| e.as_ref())
    }

    /// Get default.
    pub fn get_default(&self) -> Option<&dyn Encoder> {
        self.default.as_ref().and_then(|name| self.get(name))
    }

    /// List.
    pub fn list(&self) -> Vec<&str> {
        self.encoders.iter().map(|(n, _)| n.as_str()).collect()
    }
}

impl Default for EncoderRegistry { fn default() -> Self { Self::new() } }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_encoder() {
        let enc = HashEncoder::new(64);
        let v = enc.encode_one("hello world").unwrap();
        assert_eq!(v.len(), 64);
        // Verify normalized
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_hash_encoder_deterministic() {
        let enc = HashEncoder::new(32);
        let v1 = enc.encode_one("test").unwrap();
        let v2 = enc.encode_one("test").unwrap();
        assert_eq!(v1, v2);
    }

    #[test]
    fn test_hash_encoder_different() {
        let enc = HashEncoder::new(32);
        let v1 = enc.encode_one("hello").unwrap();
        let v2 = enc.encode_one("world").unwrap();
        assert_ne!(v1, v2);
    }

    #[test]
    fn test_encode_batch() {
        let enc = HashEncoder::new(16);
        let texts = vec!["hello", "world", "test"];
        let batch = enc.encode_batch(&texts).unwrap();
        assert_eq!(batch.nrows(), 3);
        assert_eq!(batch.ncols(), 16);
    }

    #[test]
    fn test_registry() {
        let mut reg = EncoderRegistry::new();
        reg.register("hash64", Box::new(HashEncoder::new(64)), true);
        reg.register("hash128", Box::new(HashEncoder::new(128)), false);
        assert_eq!(reg.list().len(), 2);
        let default = reg.get_default().unwrap();
        assert_eq!(default.dim(), 64);
    }

    #[test]
    fn test_matryoshka_truncate() {
        let emb: Vec<f32> = (0..640).map(|i| (i as f32) / 640.0).collect();
        let trunc = matryoshka_truncate(&emb, 128);
        assert_eq!(trunc.len(), 128);
        let norm: f32 = trunc.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_matryoshka_all() {
        let emb: Vec<f32> = (0..640).map(|i| (i as f32) / 640.0).collect();
        let all = matryoshka_all(&emb, DEFAULT_MATRYOSHKA_DIMS);
        assert_eq!(all.len(), 4);
        assert_eq!(all[0].len(), 64);
        assert_eq!(all[3].len(), 640);
    }

    #[test]
    fn test_projection_head() {
        let head = ProjectionHead::new(384, 640, 42);
        assert_eq!(head.input_dim(), 384);
        assert_eq!(head.output_dim(), 640);
        let input = vec![1.0f32; 384];
        let output = head.forward(&input);
        assert_eq!(output.len(), 640);
        let norm: f32 = output.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "Output should be normalized");
    }

    #[test]
    fn test_distillation_loss() {
        let student = vec![0.5f32, 0.5, 0.5, 0.5];
        let teacher = vec![0.5f32, 0.5, 0.5, 0.5];
        let loss = compute_distillation_loss(&student, &teacher, &[2], 1.0, 1.0, 0.3);
        assert!(loss.cosine < 0.01, "Identical vectors should have near-zero cosine loss");
        assert!(loss.mse < 0.01, "Identical vectors should have near-zero MSE");
    }

    #[test]
    fn test_distillation_loss_different() {
        let student = vec![1.0f32, 0.0, 0.0, 0.0];
        let teacher = vec![0.0f32, 1.0, 0.0, 0.0];
        let loss = compute_distillation_loss(&student, &teacher, &[2], 1.0, 1.0, 0.3);
        assert!(loss.cosine > 0.5, "Orthogonal vectors should have high cosine loss");
        assert!(loss.total > 0.0);
    }
}
