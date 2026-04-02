use std::fmt;

/// Unified error type for M2M operations.
#[derive(Debug)]
pub enum M2MError {
    IndexNotBuilt,
    NoBackends,
    InvalidConfig(String),
    StorageError(String),
    SearchError(String),
    DimensionMismatch { expected: usize, got: usize },
}

impl fmt::Display for M2MError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            M2MError::IndexNotBuilt => write!(f, "index has not been built"),
            M2MError::NoBackends => write!(f, "no search backends available"),
            M2MError::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
            M2MError::StorageError(msg) => write!(f, "storage error: {msg}"),
            M2MError::SearchError(msg) => write!(f, "search error: {msg}"),
            M2MError::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
        }
    }
}

impl std::error::Error for M2MError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_display_messages() {
        let cases = vec![
            (M2MError::IndexNotBuilt, "index has not been built"),
            (M2MError::NoBackends, "no search backends available"),
            (M2MError::InvalidConfig("bad param".into()), "invalid config: bad param"),
            (M2MError::StorageError("disk full".into()), "storage error: disk full"),
            (M2MError::SearchError("timeout".into()), "search error: timeout"),
            (M2MError::DimensionMismatch { expected: 128, got: 64 }, "dimension mismatch: expected 128, got 64"),
        ];
        for (err, expected_substring) in cases {
            let msg = format!("{}", err);
            assert!(msg.contains(expected_substring), "Expected '{}' in '{}'", expected_substring, msg);
        }
    }

    #[test]
    fn test_error_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<M2MError>();
    }

    #[test]
    fn test_error_debug_format() {
        let err = M2MError::DimensionMismatch { expected: 64, got: 32 };
        let debug = format!("{:?}", err);
        assert!(debug.contains("DimensionMismatch"));
        assert!(debug.contains("expected: 64"));
        assert!(debug.contains("got: 32"));
    }
}
