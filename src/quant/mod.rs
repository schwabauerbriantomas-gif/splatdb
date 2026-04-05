//! Integrated vector quantization module.
//!
//! TurboQuant, PolarQuant, and QJL — adapted from turbo-quant (MIT license).
//! Original: github.com/RecursiveIntell/turbo-quant
//!
//! These algorithms compress high-dimensional vectors to 3-8 bits per value
//! with minimal accuracy loss, using data-oblivious quantization (no training).

pub mod error;
pub mod polar;
pub mod qjl;
pub mod rotation;
pub mod turbo;

pub use error::{QuantError, Result};
pub use polar::{PolarCode, PolarQuantizer};
pub use qjl::{QjlQuantizer, QjlSketch};
pub use rotation::StoredRotation;
pub use turbo::{BatchStats, TurboCode, TurboQuantizer};
