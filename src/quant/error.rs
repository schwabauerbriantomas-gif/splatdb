//! Quantization error types.

#[derive(Debug)]
pub enum QuantError {
    DimensionMismatch { expected: usize, got: usize },
    OddDimension { got: usize },
    ZeroDimension,
    InvalidBitWidth { got: u8 },
    ZeroProjectionCount,
    RotationFailed(String),
    MalformedCode(String),
}

impl std::fmt::Display for QuantError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {}, got {}", expected, got)
            }
            Self::OddDimension { got } => write!(f, "dimension must be even, got {}", got),
            Self::ZeroDimension => write!(f, "dimension must be non-zero"),
            Self::InvalidBitWidth { got } => write!(f, "bits must be 1-16, got {}", got),
            Self::ZeroProjectionCount => write!(f, "projection count must be non-zero"),
            Self::RotationFailed(r) => write!(f, "rotation failed: {}", r),
            Self::MalformedCode(r) => write!(f, "malformed code: {}", r),
        }
    }
}

impl std::error::Error for QuantError {}

pub type Result<T> = std::result::Result<T, QuantError>;
