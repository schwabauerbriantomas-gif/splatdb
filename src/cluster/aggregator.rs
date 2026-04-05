//! Result aggregator for distributed search.
//! Supports RRF (Reciprocal Rank Fusion) and score-based merge.
//! Ported from splatdb Python.

use std::collections::HashMap;

/// Merges search results from multiple cluster nodes.
pub struct ResultAggregator {
    rrf_k: usize,
}

impl ResultAggregator {
    pub fn new(rrf_k: usize) -> Self {
        Self { rrf_k }
    }

    /// Merge results from multiple nodes using specified strategy.
    /// Input: map of edge_id -> list of (doc_id, distance)
    /// Output: top-k merged results as (doc_id, distance)
    pub fn merge_results(
        &self,
        results: &HashMap<String, Vec<(usize, f64)>>,
        k: usize,
        strategy: &str,
    ) -> Vec<(usize, f64)> {
        if results.is_empty() {
            return Vec::new();
        }
        match strategy {
            "rrf" => self.rrf_merge(results, k),
            "score" => self.score_merge(results, k),
            _ => self.score_merge(results, k),
        }
    }

    /// Reciprocal Rank Fusion.
    /// RRF_score = sum(1 / (k + rank_in_list))
    fn rrf_merge(
        &self,
        results: &HashMap<String, Vec<(usize, f64)>>,
        k: usize,
    ) -> Vec<(usize, f64)> {
        let mut rrf_scores: HashMap<usize, f64> = HashMap::new();
        let mut best_distances: HashMap<usize, f64> = HashMap::new();

        for edge_results in results.values() {
            for (rank, &(doc_id, distance)) in edge_results.iter().enumerate() {
                *rrf_scores.entry(doc_id).or_insert(0.0) += 1.0 / (self.rrf_k + rank + 1) as f64;
                let best = best_distances.entry(doc_id).or_insert(f64::MAX);
                if distance < *best {
                    *best = distance;
                }
            }
        }

        let mut sorted: Vec<(usize, f64)> = rrf_scores.into_iter().collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        sorted.truncate(k);
        sorted
            .into_iter()
            .map(|(doc_id, _score)| (doc_id, best_distances[&doc_id]))
            .collect()
    }

    /// Simple score/distance-based merge. Sort by distance ascending.
    fn score_merge(
        &self,
        results: &HashMap<String, Vec<(usize, f64)>>,
        k: usize,
    ) -> Vec<(usize, f64)> {
        let mut best: HashMap<usize, f64> = HashMap::new();
        for edge_results in results.values() {
            for &(doc_id, distance) in edge_results {
                let entry = best.entry(doc_id).or_insert(f64::MAX);
                if distance < *entry {
                    *entry = distance;
                }
            }
        }
        let mut all: Vec<(usize, f64)> = best.into_iter().collect();
        all.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        all.truncate(k);
        all
    }
}

impl Default for ResultAggregator {
    fn default() -> Self {
        Self::new(60)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_score_merge() {
        let mut results = HashMap::new();
        results.insert("e1".into(), vec![(1, 0.9), (2, 0.8)]);
        results.insert("e2".into(), vec![(3, 0.95), (1, 0.85)]);

        let agg = ResultAggregator::new(60);
        let merged = agg.merge_results(&results, 5, "score");
        assert_eq!(merged[0].0, 2); // lowest distance
        assert_eq!(merged.len(), 3);
    }

    #[test]
    fn test_rrf_merge() {
        let mut results = HashMap::new();
        results.insert("e1".into(), vec![(1, 0.5), (2, 0.6)]);
        results.insert("e2".into(), vec![(2, 0.55), (3, 0.7)]);

        let agg = ResultAggregator::new(60);
        let merged = agg.merge_results(&results, 3, "rrf");
        // Doc 2 appears in both lists, should rank higher by RRF
        assert_eq!(merged.len(), 3);
    }
}
