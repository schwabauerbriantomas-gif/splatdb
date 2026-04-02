//! Embedding configuration for SplatDB.
//!
//! Defines supported embedding models and their parameters.
//! Ported from splatdb Python.

use serde::{Deserialize, Serialize};

/// Supported embedding models.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EmbeddingModel {
    /// all-MiniLM-L6-v2 (384D, fast, good quality)
    AllMiniLML6V2,
    /// bge-small-en-v1.5 (384D, best all-rounder)
    BgeSmallEnV15,
    /// gte-small (384D, best accuracy)
    GteSmall,
    /// Custom model with specified dimension
    Custom { name: String, dim: usize },
}

impl EmbeddingModel {
    /// Get the output dimension of this model.
    pub fn dim(&self) -> usize {
        match self {
            Self::AllMiniLML6V2 => 384,
            Self::BgeSmallEnV15 => 384,
            Self::GteSmall => 384,
            Self::Custom { dim, .. } => *dim,
        }
    }

    /// Get the model name string.
    pub fn name(&self) -> &str {
        match self {
            Self::AllMiniLML6V2 => "all-MiniLM-L6-v2",
            Self::BgeSmallEnV15 => "bge-small-en-v1.5",
            Self::GteSmall => "gte-small",
            Self::Custom { name, .. } => name.as_str(),
        }
    }
}

/// Configuration for embedding generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Which model to use
    pub model: EmbeddingModel,
    /// Batch size for encoding
    pub batch_size: usize,
    /// Normalize embeddings to unit length
    pub normalize: bool,
    /// Compute device: "cpu" or "cuda"
    pub device: String,
    /// Quantization: None, "int8", "binary"
    pub quantization: Option<String>,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            model: EmbeddingModel::AllMiniLML6V2,
            batch_size: 32,
            normalize: true,
            device: "cpu".to_string(),
            quantization: None,
        }
    }
}

impl EmbeddingConfig {
    /// With model.
    pub fn with_model(model: EmbeddingModel) -> Self {
        Self {
            model,
            ..Default::default()
        }
    }

    /// Dim.
    pub fn dim(&self) -> usize {
        self.model.dim()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let cfg = EmbeddingConfig::default();
        assert_eq!(cfg.dim(), 384);
        assert_eq!(cfg.model.name(), "all-MiniLM-L6-v2");
        assert!(cfg.normalize);
    }

    #[test]
    fn test_custom_model() {
        let model = EmbeddingModel::Custom {
            name: "my-model".to_string(),
            dim: 768,
        };
        assert_eq!(model.dim(), 768);
        assert_eq!(model.name(), "my-model");
    }
}
