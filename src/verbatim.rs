//! Verbatim Storage — Store and retrieve original document text alongside splats.
//!
//! When a vector search returns results, the confidence score tells you how reliable
//! the match is. Verbatim storage ensures you always have the original source text
//! to verify against, preventing hallucination.
//!
//! Confidence levels:
//! - HIGH (distance < 0.3): Very reliable match, source text is directly relevant
//! - MEDIUM (0.3 <= distance < 0.6): Good match, verify context before relying on it
//! - LOW (distance >= 0.6): Weak match, treat as suggestive — always verify against source

use serde::{Deserialize, Serialize};

/// Confidence level based on vector distance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Confidence {
    /// distance < 0.3 — very reliable match
    High,
    /// 0.3 <= distance < 0.6 — good match, verify context
    Medium,
    /// distance >= 0.6 — weak match, treat as suggestive
    Low,
}

impl std::fmt::Display for Confidence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Confidence::High => write!(f, "HIGH"),
            Confidence::Medium => write!(f, "MEDIUM"),
            Confidence::Low => write!(f, "LOW"),
        }
    }
}

impl Confidence {
    /// Classify a distance into a confidence level.
    pub fn from_distance(distance: f32) -> Self {
        if distance < 0.3 {
            Confidence::High
        } else if distance < 0.6 {
            Confidence::Medium
        } else {
            Confidence::Low
        }
    }

    /// Human-readable explanation of the confidence level.
    pub fn explanation(&self) -> &'static str {
        match self {
            Confidence::High => "Very reliable match — source text is directly relevant",
            Confidence::Medium => "Good match — verify context before relying on it",
            Confidence::Low => "Weak match — treat as suggestive, always verify against source",
        }
    }

    /// Emoji indicator.
    pub fn emoji(&self) -> &'static str {
        match self {
            Confidence::High => "🟢",
            Confidence::Medium => "🟡",
            Confidence::Low => "🔴",
        }
    }
}

/// A verbatim search result with confidence scoring.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerbatimResult {
    /// Document ID
    pub id: String,
    /// Vector index in the store
    pub index: usize,
    /// Raw distance from query vector
    pub distance: f32,
    /// Confidence classification
    pub confidence: Confidence,
    /// Original document text (verbatim)
    pub text: String,
    /// Optional metadata
    pub metadata: Option<serde_json::Value>,
    /// Similarity score (0.0 - 1.0, where 1.0 is perfect match)
    pub similarity: f64,
}

impl VerbatimResult {
    /// Create a new verbatim result from search components.
    pub fn new(
        id: String,
        index: usize,
        distance: f32,
        text: String,
        metadata: Option<serde_json::Value>,
    ) -> Self {
        let confidence = Confidence::from_distance(distance);
        // Convert L2 distance to similarity score (cosine-like normalization)
        let similarity = (1.0 / (1.0 + distance as f64)).min(1.0);
        Self {
            id,
            index,
            distance,
            confidence,
            text,
            metadata,
            similarity,
        }
    }

    /// Format as a human-readable string with confidence indicator.
    pub fn to_display(&self) -> String {
        let meta_str = self
            .metadata
            .as_ref()
            .map(|m| format!(" | meta: {}", m))
            .unwrap_or_default();
        format!(
            "{} [{}] sim={:.3} dist={:.4}\n  {}{}\n  {}",
            self.confidence.emoji(),
            self.confidence,
            self.similarity,
            self.distance,
            self.text,
            self.confidence.explanation(),
            meta_str,
        )
    }

    /// Format as JSON for API output.
    pub fn to_json_value(&self) -> serde_json::Value {
        serde_json::json!({
            "id": self.id,
            "index": self.index,
            "distance": (self.distance * 10000.0).round() / 10000.0,
            "similarity": (self.similarity * 10000.0).round() / 10000.0,
            "confidence": self.confidence.to_string(),
            "confidence_emoji": self.confidence.emoji(),
            "text": self.text,
            "metadata": self.metadata,
            "warning": if self.confidence == Confidence::Low {
                Some("LOW CONFIDENCE — verify against original source before using")
            } else {
                None
            },
        })
    }
}

/// Verbatim storage configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerbatimConfig {
    /// Whether to store original text alongside vectors
    pub enabled: bool,
    /// Maximum text length to store (bytes)
    pub max_text_length: usize,
    /// Whether to include confidence scores in search results
    pub show_confidence: bool,
    /// Whether to warn on LOW confidence results
    pub warn_low_confidence: bool,
}

impl Default for VerbatimConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_text_length: 100_000,
            show_confidence: true,
            warn_low_confidence: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_confidence_from_distance() {
        assert_eq!(Confidence::from_distance(0.1), Confidence::High);
        assert_eq!(Confidence::from_distance(0.25), Confidence::High);
        assert_eq!(Confidence::from_distance(0.3), Confidence::Medium);
        assert_eq!(Confidence::from_distance(0.5), Confidence::Medium);
        assert_eq!(Confidence::from_distance(0.6), Confidence::Low);
        assert_eq!(Confidence::from_distance(1.0), Confidence::Low);
    }

    #[test]
    fn test_verbatim_result_creation() {
        let result = VerbatimResult::new(
            "doc_1".to_string(),
            0,
            0.15,
            "Original document text here".to_string(),
            None,
        );
        assert_eq!(result.confidence, Confidence::High);
        assert!(result.similarity > 0.8);
    }

    #[test]
    fn test_verbatim_result_low_confidence() {
        let result = VerbatimResult::new(
            "doc_2".to_string(),
            1,
            0.75,
            "Weakly related text".to_string(),
            Some(serde_json::json!({"source": "web"})),
        );
        assert_eq!(result.confidence, Confidence::Low);
        let json = result.to_json_value();
        assert!(json["warning"].is_string());
    }

    #[test]
    fn test_display_format() {
        let result = VerbatimResult::new(
            "doc_3".to_string(),
            2,
            0.45,
            "Some document text".to_string(),
            None,
        );
        let display = result.to_display();
        assert!(display.contains("MEDIUM"));
        assert!(display.contains("verify context"));
    }

    #[test]
    fn test_similarity_score_range() {
        for dist in [0.0, 0.1, 0.3, 0.5, 1.0, 5.0, 100.0] {
            let result = VerbatimResult::new(
                "test".to_string(),
                0,
                dist,
                "text".to_string(),
                None,
            );
            assert!(
                result.similarity >= 0.0 && result.similarity <= 1.0,
                "similarity out of range for distance={}: {}",
                dist,
                result.similarity
            );
        }
    }
}
