//! SplatDB Configuration Types
//!
//! Enums and data types used by SplatDBConfig.

use serde::{Deserialize, Serialize};
use std::any::Any;

/// Quantization algorithm selector.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
pub enum QuantAlgorithm {
    #[default]
    TurboQuant,
    PolarQuant,
    None,
}

/// Search backend selector.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
pub enum SearchBackend {
    #[default]
    Hrm2,
    Hnsw,
    Lsh,
    Quantized,
}

/// Semantic search fusion method.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
pub enum FusionMethod {
    #[default]
    Rrf,
    Weighted,
    VectorOnly,
    Bm25Only,
}

/// Dtype enum to represent numpy-style data types
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
pub enum Dtype {
    #[default]
    Float32,
    Float64,
    Int32,
    Int64,
}

impl Dtype {
    /// Downcast to `dyn Any` for runtime type inspection.
    pub fn as_any(&self) -> &dyn Any {
        self
    }
}
