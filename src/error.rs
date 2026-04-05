use std::fmt;

/// Unified error type for SplatDB operations.
#[derive(Debug)]
pub enum SplatDBError {
    IndexNotBuilt,
    NoBackends,
    InvalidConfig(String),
    StorageError(String),
    SearchError(String),
    DimensionMismatch { expected: usize, got: usize },
}

impl fmt::Display for SplatDBError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SplatDBError::IndexNotBuilt => write!(f, "index has not been built"),
            SplatDBError::NoBackends => write!(f, "no search backends available"),
            SplatDBError::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
            SplatDBError::StorageError(msg) => write!(f, "storage error: {msg}"),
            SplatDBError::SearchError(msg) => write!(f, "search error: {msg}"),
            SplatDBError::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
        }
    }
}

impl std::error::Error for SplatDBError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_display_messages() {
        let cases = vec![
            (SplatDBError::IndexNotBuilt, "index has not been built"),
            (SplatDBError::NoBackends, "no search backends available"),
            (
                SplatDBError::InvalidConfig("bad param".into()),
                "invalid config: bad param",
            ),
            (
                SplatDBError::StorageError("disk full".into()),
                "storage error: disk full",
            ),
            (
                SplatDBError::SearchError("timeout".into()),
                "search error: timeout",
            ),
            (
                SplatDBError::DimensionMismatch {
                    expected: 128,
                    got: 64,
                },
                "dimension mismatch: expected 128, got 64",
            ),
        ];
        for (err, expected_substring) in cases {
            let msg = format!("{}", err);
            assert!(
                msg.contains(expected_substring),
                "Expected '{}' in '{}'",
                expected_substring,
                msg
            );
        }
    }

    #[test]
    fn test_error_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<SplatDBError>();
    }

    #[test]
    fn test_error_debug_format() {
        let err = SplatDBError::DimensionMismatch {
            expected: 64,
            got: 32,
        };
        let debug = format!("{:?}", err);
        assert!(debug.contains("DimensionMismatch"));
        assert!(debug.contains("expected: 64"));
        assert!(debug.contains("got: 32"));
    }
}
