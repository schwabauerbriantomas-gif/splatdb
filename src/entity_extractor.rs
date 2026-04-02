//! Entity extractor for SplatDB.
//! Extracts entities via structural patterns (regex), n-gram analysis,
//! and optional semantic clustering.
//! Ported from splatdb Python.

use regex::Regex;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
pub struct EntityCandidate {
    pub text: String,
    pub entity_type: String,
    pub score: f64,
    pub count: usize,
    pub embedding: Option<Vec<f32>>,
    pub cluster_id: i32,
    pub start_positions: Vec<usize>,
}

/// Configuration for the entity extractor.
#[derive(Debug, Clone)]
pub struct ExtractorConfig {
    pub use_structural_patterns: bool,
    pub use_ngram_analysis: bool,
    pub min_ngram_score: f64,
}

impl Default for ExtractorConfig {
    fn default() -> Self {
        Self { use_structural_patterns: true, use_ngram_analysis: true, min_ngram_score: 0.3 }
    }
}

pub struct SplatDBEntityExtractor {
    email_re: Regex,
    url_re: Regex,
    phone_re: Regex,
    date_re: Regex,
    money_re: Regex,
    word_re: Regex,
    stopwords: HashSet<String>,
    config: ExtractorConfig,
}

impl SplatDBEntityExtractor {
    /// New.
    pub fn new() -> Self {
        Self::with_config(ExtractorConfig::default())
    }

    /// With config.
    pub fn with_config(config: ExtractorConfig) -> Self {
        Self {
            email_re: Regex::new(r"(?i)[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}").expect("invalid email regex"),
            url_re: Regex::new(r#"https?://[^\s<>"]+|www\.[^\s<>"]+"#).expect("invalid url regex"),
            phone_re: Regex::new(r"\d{3}[-.]?\d{3}[-.]?\d{4}").expect("invalid phone regex"),
            date_re: Regex::new(r"\d{4}-\d{2}-\d{2}").expect("invalid date regex"),
            money_re: Regex::new(r"\$\d+(?:,\d{3})*(?:\.\d{2})?").expect("invalid money regex"),
            word_re: Regex::new(r"(?u)\b[A-Za-z0-9]+\b").expect("invalid word regex"),
            stopwords: [
                "el","la","los","las","un","una","unos","unas","de","del","a","ante",
                "con","en","para","por","y","o","u","e","que","es","son","se","su","lo",
                "the","a","an","and","or","in","on","at","to","for","with","is","are",
                "of","from","by","this","that","it","as","was","were","be","been","has","have",
            ].iter().map(|s| s.to_string()).collect(),
            config,
        }
    }

    /// Extract entities from text using all enabled methods.
    pub fn extract(&self, text: &str) -> Vec<EntityCandidate> {
        let mut candidates: HashMap<String, EntityCandidate> = HashMap::new();

        // 1. Structural patterns (regex)
        if self.config.use_structural_patterns {
            self.extract_structural(text, &mut candidates);
        }

        // 2. N-gram analysis (capitalized sequences)
        if self.config.use_ngram_analysis {
            self.extract_ngrams(text, &mut candidates);
        }

        candidates.into_values()
            .filter(|c| c.score >= self.config.min_ngram_score)
            .collect()
    }

    /// Extract using structural regex patterns.
    fn extract_structural(&self, text: &str, candidates: &mut HashMap<String, EntityCandidate>) {
        let patterns: &[(&Regex, &str)] = &[
            (&self.email_re, "contact"),
            (&self.url_re, "url"),
            (&self.phone_re, "contact"),
            (&self.date_re, "date"),
            (&self.money_re, "money"),
        ];
        for (re, etype) in patterns {
            for m in re.find_iter(text) {
                let key = m.as_str().to_lowercase();
                candidates.entry(key).and_modify(|c| {
                    c.count += 1;
                    c.start_positions.push(m.start());
                    c.score = (c.score + 0.1).min(1.0);
                }).or_insert_with(|| EntityCandidate {
                    text: m.as_str().to_string(),
                    entity_type: etype.to_string(),
                    score: 1.0,
                    count: 1,
                    embedding: None,
                    cluster_id: -1,
                    start_positions: vec![m.start()],
                });
            }
        }
    }

    /// Extract proper nouns via capitalized n-gram sequences.
    fn extract_ngrams(&self, text: &str, candidates: &mut HashMap<String, EntityCandidate>) {
        let tokens: Vec<(&str, usize)> = self.word_re.find_iter(text)
            .map(|m| (m.as_str(), m.start()))
            .collect();

        let connectors: HashSet<&str> = ["de","del","la","el","y","e","u"].iter().copied().collect();
        let mut i = 0;

        while i < tokens.len() {
            let (word, start_pos) = tokens[i];

            if is_capitalized(word) && !self.stopwords.contains(&word.to_lowercase()) {
                let mut j = i + 1;
                let mut entity_tokens = vec![word];

                // Consume following capitalized words
                while j < tokens.len() && j - i < 4 {
                    let next_word = tokens[j].0;
                    if is_capitalized(next_word) && !self.stopwords.contains(&next_word.to_lowercase()) {
                        entity_tokens.push(next_word);
                        j += 1;
                    } else if connectors.contains(next_word.to_lowercase().as_str())
                        && j + 1 < tokens.len()
                        && is_capitalized(tokens[j + 1].0)
                    {
                        // Allow connectors if next is capitalized (e.g., "Ministerio de Hacienda")
                        entity_tokens.push(next_word);
                        entity_tokens.push(tokens[j + 1].0);
                        j += 2;
                    } else {
                        break;
                    }
                }

                let entity_text = entity_tokens.join(" ");
                let key = entity_text.to_lowercase();

                candidates.entry(key).and_modify(|c| {
                    c.count += 1;
                    c.start_positions.push(start_pos);
                    c.score = (c.score + 0.1).min(1.0);
                }).or_insert_with(|| EntityCandidate {
                    text: entity_text,
                    entity_type: "proper_noun".into(),
                    score: 0.8,
                    count: 1,
                    embedding: None,
                    cluster_id: -1,
                    start_positions: vec![start_pos],
                });

                i = j;
            } else {
                i += 1;
            }
        }
    }

    /// Validate entities against splat centers (semantic clustering).
    pub fn validate_semantic(
        &self,
        candidates: &mut [EntityCandidate],
        splat_centers: &[Vec<f32>],
    ) {
        if splat_centers.is_empty() { return; }

        for c in candidates.iter_mut() {
            if let Some(ref emb) = c.embedding {
                // Find closest splat center via cosine similarity
                let mut best_idx = 0;
                let mut best_sim = f64::NEG_INFINITY;
                for (i, center) in splat_centers.iter().enumerate() {
                    let sim = cosine_sim(emb, center);
                    if sim > best_sim { best_sim = sim; best_idx = i; }
                }
                c.cluster_id = best_idx as i32;
                if best_sim > 0.8 {
                    c.score = (c.score + 0.2).min(1.0);
                }
            }
        }
    }

    /// Count unique entity types found.
    pub fn count_by_type(&self, candidates: &[EntityCandidate]) -> HashMap<String, usize> {
        let mut counts = HashMap::new();
        for c in candidates {
            *counts.entry(c.entity_type.clone()).or_insert(0) += 1;
        }
        counts
    }
}

fn is_capitalized(s: &str) -> bool {
    s.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) && s.len() > 1
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(&x, &y)| (x * y) as f64).sum();
    let na: f64 = a.iter().map(|&v| (v * v) as f64).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|&v| (v * v) as f64).sum::<f64>().sqrt();
    let denom = na * nb;
    if denom < 1e-10 { 0.0 } else { dot / denom }
}

impl Default for SplatDBEntityExtractor {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_url() {
        let ext = SplatDBEntityExtractor::new();
        let results = ext.extract("Visit https://example.com for info");
        assert!(results.iter().any(|c| c.entity_type == "url"));
    }

    #[test]
    fn test_extract_date() {
        let ext = SplatDBEntityExtractor::new();
        let results = ext.extract("The event is on 2026-03-28");
        assert!(results.iter().any(|c| c.entity_type == "date" && c.text == "2026-03-28"));
    }

    #[test]
    fn test_extract_money() {
        let ext = SplatDBEntityExtractor::new();
        let results = ext.extract("Price is $1234");
        assert!(results.iter().any(|c| c.entity_type == "money"));
    }

    #[test]
    fn test_extract_email() {
        let ext = SplatDBEntityExtractor::new();
        let results = ext.extract("Contact user@example.com for info");
        assert!(results.iter().any(|c| c.entity_type == "contact" && c.text.contains("@")));
    }

    #[test]
    fn test_extract_proper_nouns() {
        let ext = SplatDBEntityExtractor::new();
        let results = ext.extract("Apple Inc. announced Steve Jobs as CEO");
        let proper: Vec<_> = results.iter().filter(|c| c.entity_type == "proper_noun").collect();
        assert!(!proper.is_empty(), "Should detect proper nouns");
    }

    #[test]
    fn test_ngram_with_connectors() {
        let ext = SplatDBEntityExtractor::new();
        let results = ext.extract("El Ministerio de Hacienda firmo el acuerdo");
        let proper: Vec<_> = results.iter().filter(|c| c.entity_type == "proper_noun").collect();
        assert!(proper.iter().any(|c| c.text.contains("Ministerio")));
    }

    #[test]
    fn test_empty() {
        let ext = SplatDBEntityExtractor::new();
        let results = ext.extract("");
        assert!(results.is_empty());
    }

    #[test]
    fn test_validate_semantic() {
        let ext = SplatDBEntityExtractor::new();
        let mut candidates = vec![
            EntityCandidate {
                text: "test".into(), entity_type: "proper_noun".into(), score: 0.5,
                count: 1, embedding: Some(vec![1.0, 0.0]), cluster_id: -1, start_positions: vec![0],
            },
        ];
        let centers = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        ext.validate_semantic(&mut candidates, &centers);
        assert_eq!(candidates[0].cluster_id, 0);
        assert!(candidates[0].score > 0.5);
    }

    #[test]
    fn test_count_by_type() {
        let ext = SplatDBEntityExtractor::new();
        let results = ext.extract("Email test@a.com and visit https://example.com on 2026-01-01");
        let counts = ext.count_by_type(&results);
        assert!(counts.contains_key("contact"));
        assert!(counts.contains_key("url"));
    }
}
