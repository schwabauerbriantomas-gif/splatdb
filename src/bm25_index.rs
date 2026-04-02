//! BM25 (Okapi BM25) full-text search index.
//!
//! Lightweight implementation using term frequency / inverse document frequency.
//! Designed for small-to-medium collections (~1-10K docs).
//! Ported from splatdb Python.

use std::collections::{HashMap, HashSet};

use regex::Regex;

/// BM25 full-text search index.
pub struct BM25Index {
    k1: f64,
    b: f64,
    docs: HashMap<String, String>,
    doc_tokens: HashMap<String, Vec<String>>,
    doc_lengths: HashMap<String, usize>,
    doc_freq: HashMap<String, usize>,
    term_freqs: HashMap<String, HashMap<String, usize>>,
    avg_dl: f64,
    n_docs: usize,
    tokenizer: fn(&str) -> Vec<String>,
}

fn default_tokenizer(text: &str) -> Vec<String> {
    let re = Regex::new(r"[a-záéíóúñü0-9]+").unwrap();
    re.find_iter(&text.to_lowercase())
        .map(|m| m.as_str().to_string())
        .collect()
}

impl BM25Index {
    /// New.
    pub fn new() -> Self {
        Self::with_params(1.5, 0.75, None)
    }

    /// With params.
    pub fn with_params(k1: f64, b: f64, tokenizer: Option<fn(&str) -> Vec<String>>) -> Self {
        Self {
            k1,
            b,
            docs: HashMap::new(),
            doc_tokens: HashMap::new(),
            doc_lengths: HashMap::new(),
            doc_freq: HashMap::new(),
            term_freqs: HashMap::new(),
            avg_dl: 0.0,
            n_docs: 0,
            tokenizer: tokenizer.unwrap_or(default_tokenizer),
        }
    }

    /// Add a document to the index.
    pub fn add(&mut self, doc_id: &str, text: &str) {
        // Remove old if re-adding
        if self.docs.contains_key(doc_id) {
            self.remove(doc_id);
        }

        let tokens = (self.tokenizer)(text);
        let token_count = tokens.len();
        let mut tf: HashMap<String, usize> = HashMap::new();
        let mut unique_terms: HashSet<String> = HashSet::new();

        for token in &tokens {
            *tf.entry(token.clone()).or_insert(0) += 1;
            unique_terms.insert(token.clone());
        }

        self.docs.insert(doc_id.to_string(), text.to_string());
        self.doc_tokens.insert(doc_id.to_string(), tokens);
        self.doc_lengths.insert(doc_id.to_string(), token_count);
        self.term_freqs.insert(doc_id.to_string(), tf);

        for term in &unique_terms {
            *self.doc_freq.entry(term.clone()).or_insert(0) += 1;
        }

        self.n_docs += 1;
        self.update_avg_dl();
    }

    /// Remove a document from the index.
    pub fn remove(&mut self, doc_id: &str) -> bool {
        if !self.docs.contains_key(doc_id) {
            return false;
        }

        if let Some(tokens) = self.doc_tokens.get(doc_id) {
            let unique: HashSet<&String> = tokens.iter().collect();
            for term in unique {
                if let Some(count) = self.doc_freq.get_mut(term) {
                    *count = count.saturating_sub(1);
                    if *count == 0 {
                        self.doc_freq.remove(term);
                    }
                }
            }
        }

        self.docs.remove(doc_id);
        self.doc_tokens.remove(doc_id);
        self.doc_lengths.remove(doc_id);
        self.term_freqs.remove(doc_id);
        self.n_docs = self.n_docs.saturating_sub(1);
        self.update_avg_dl();
        true
    }

    /// Search for documents matching the query.
    pub fn search(&self, query: &str, k: usize) -> Vec<(String, f64)> {
        self.search_filtered(query, k, None)
    }

    /// Search with optional document filter.
    pub fn search_filtered(
        &self,
        query: &str,
        k: usize,
        doc_filter: Option<&HashSet<String>>,
    ) -> Vec<(String, f64)> {
        if self.n_docs == 0 {
            return vec![];
        }

        let query_tokens = (self.tokenizer)(query);
        let mut scores: HashMap<String, f64> = HashMap::new();

        let n = self.n_docs as f64;
        let avg_dl = if self.avg_dl > 0.0 { self.avg_dl } else { 1.0 };

        for term in &query_tokens {
            let df = match self.doc_freq.get(term) {
                Some(&d) => d as f64,
                None => continue,
            };

            let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();

            for (doc_id, tf_counter) in &self.term_freqs {
                if let Some(filter) = doc_filter {
                    if !filter.contains(doc_id) {
                        continue;
                    }
                }

                let tf = match tf_counter.get(term) {
                    Some(&c) => c as f64,
                    None => continue,
                };

                let dl = *self.doc_lengths.get(doc_id).unwrap_or(&1) as f64;
                let numerator = tf * (self.k1 + 1.0);
                let denominator = tf + self.k1 * (1.0 - self.b + self.b * dl / avg_dl);
                *scores.entry(doc_id.clone()).or_insert(0.0) += idf * numerator / denominator;
            }
        }

        let mut ranked: Vec<(String, f64)> = scores.into_iter().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked.truncate(k);
        ranked
    }

    /// Search returning only doc IDs.
    pub fn search_ids(&self, query: &str, k: usize) -> Vec<String> {
        self.search(query, k).into_iter().map(|(id, _)| id).collect()
    }

    /// Number of indexed documents.
    pub fn n_docs(&self) -> usize {
        self.n_docs
    }

    /// Get document text by ID.
    pub fn get_doc(&self, doc_id: &str) -> Option<&str> {
        self.docs.get(doc_id).map(|s| s.as_str())
    }

    /// Clear all documents.
    pub fn clear(&mut self) {
        self.docs.clear();
        self.doc_tokens.clear();
        self.doc_lengths.clear();
        self.doc_freq.clear();
        self.term_freqs.clear();
        self.avg_dl = 0.0;
        self.n_docs = 0;
    }

    fn update_avg_dl(&mut self) {
        if self.n_docs > 0 {
            let total: usize = self.doc_lengths.values().sum();
            self.avg_dl = total as f64 / self.n_docs as f64;
        } else {
            self.avg_dl = 0.0;
        }
    }
}

impl Default for BM25Index {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_search() {
        let mut bm25 = BM25Index::new();
        bm25.add("doc_1", "SplatDB vector search for semantic memory");
        bm25.add("doc_2", "Brian decided to use SplatDB for semantic memory");

        let results = bm25.search("SplatDB semantic", 5);
        assert!(!results.is_empty());
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_remove() {
        let mut bm25 = BM25Index::new();
        bm25.add("doc_1", "hello world");
        assert_eq!(bm25.n_docs(), 1);
        assert!(bm25.remove("doc_1"));
        assert_eq!(bm25.n_docs(), 0);
        assert!(!bm25.remove("doc_1"));
    }

    #[test]
    fn test_re_add() {
        let mut bm25 = BM25Index::new();
        bm25.add("doc_1", "hello world");
        bm25.add("doc_1", "updated text here");
        assert_eq!(bm25.n_docs(), 1);
        let results = bm25.search("updated", 5);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "doc_1");
    }

    #[test]
    fn test_search_ids() {
        let mut bm25 = BM25Index::new();
        bm25.add("a", "test document");
        let ids = bm25.search_ids("test", 5);
        assert_eq!(ids, vec!["a".to_string()]);
    }

    #[test]
    fn test_empty_search() {
        let bm25 = BM25Index::new();
        let results = bm25.search("anything", 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_clear() {
        let mut bm25 = BM25Index::new();
        bm25.add("doc_1", "hello");
        bm25.clear();
        assert_eq!(bm25.n_docs(), 0);
    }

    #[test]
    fn test_unicode_tokenizer() {
        let tokens = default_tokenizer("El niño está aquí 123");
        assert!(tokens.contains(&"niño".to_string()));
        assert!(tokens.contains(&"está".to_string()));
        assert!(tokens.contains(&"123".to_string()));
    }
}
