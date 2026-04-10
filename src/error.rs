use std::fmt;

/// Unified error type for SplatsDB operations.
#[derive(Debug)]
pub enum SplatsDBError {
    IndexNotBuilt,
    NoBackends,
    InvalidConfig(String),
    StorageError(String),
    SearchError(String),
    DimensionMismatch { expected: usize, got: usize },
}

impl fmt::Display for SplatsDBError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SplatsDBError::IndexNotBuilt => write!(f, "index has not been built"),
            SplatsDBError::NoBackends => write!(f, "no search backends available"),
            SplatsDBError::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
            SplatsDBError::StorageError(msg) => write!(f, "storage error: {msg}"),
            SplatsDBError::SearchError(msg) => write!(f, "search error: {msg}"),
            SplatsDBError::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
        }
    }
}

impl std::error::Error for SplatsDBError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_display_messages() {
        let cases = vec![
            (SplatsDBError::IndexNotBuilt, "index has not been built"),
            (SplatsDBError::NoBackends, "no search backends available"),
            (
                SplatsDBError::InvalidConfig("bad param".into()),
                "invalid config: bad param",
            ),
            (
                SplatsDBError::StorageError("disk full".into()),
                "storage error: disk full",
            ),
            (
                SplatsDBError::SearchError("timeout".into()),
                "search error: timeout",
            ),
            (
                SplatsDBError::DimensionMismatch {
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
        assert_send_sync::<SplatsDBError>();
    }

    #[test]
    fn test_error_debug_format() {
        let err = SplatsDBError::DimensionMismatch {
            expected: 64,
            got: 32,
        };
        let debug = format!("{:?}", err);
        assert!(debug.contains("DimensionMismatch"));
        assert!(debug.contains("expected: 64"));
        assert!(debug.contains("got: 32"));
    }
}
