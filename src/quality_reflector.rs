//! Quality Reflector — evaluates search result quality.
//!
//! Metrics: precision@k, recall@k, anomaly detection, cross-backend comparison.
//! Pattern: MASFactory ReflectionAgent.
//! Ported from splatdb Python.

use serde::Serialize;
use std::collections::HashMap;

/// Quality levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum QualityLevel {
    Excellent, // score >= 0.9
    Good,      // score >= 0.7
    Acceptable, // score >= 0.5
    Poor,      // score >= 0.3
    Critical,  // score < 0.3
}

/// Quality report for a search evaluation.
#[derive(Debug, Clone, Serialize)]
pub struct QualityReport {
    pub precision_at_k: f64,
    pub recall_at_k: f64,
    pub quality_level: QualityLevel,
    pub anomalies: Vec<String>,
    pub suggestions: Vec<String>,
    pub backend_used: String,
}

/// Reflector statistics.
#[derive(Debug, Clone, Default, Serialize)]
pub struct ReflectorStats {
    pub total_evaluations: u64,
    pub avg_precision: f64,
    pub avg_recall: f64,
    pub anomalies_detected: u64,
    pub reindex_suggestions: u64,
}

const MAX_HISTORY: usize = 200;
const DEGRADATION_WINDOW: usize = 10;

/// Quality reflector for SplatDB search results.
pub struct QualityReflector {
    precision_warning: f64,
    precision_critical: f64,
    recall_warning: f64,
    _recall_critical: f64,
    enable_cross_backend: bool,
    stats: ReflectorStats,
    quality_history: Vec<QualityReport>,
    backend_results: HashMap<String, Vec<(Vec<String>, f64)>>,
}

impl QualityReflector {
    /// New.
    pub fn new() -> Self {
        Self {
            precision_warning: 0.5,
            precision_critical: 0.3,
            recall_warning: 0.6,
            _recall_critical: 0.4,
            enable_cross_backend: true,
            stats: ReflectorStats::default(),
            quality_history: Vec::new(),
            backend_results: HashMap::new(),
        }
    }

    /// Evaluate quality of search results.
    pub fn evaluate(
        &mut self,
        result_ids: &[String],
        ground_truth: Option<&[String]>,
        k: usize,
        backend: &str,
        distances: Option<&[f64]>,
    ) -> QualityReport {
        let mut anomalies = Vec::new();
        let mut suggestions = Vec::new();

        let precision = Self::compute_precision(result_ids, ground_truth, k);
        let recall = Self::compute_recall(result_ids, ground_truth);

        // Distance anomalies
        if let Some(dists) = distances {
            anomalies.extend(Self::detect_distance_anomalies(dists));
        }

        // Duplicate detection
        let unique: std::collections::HashSet<&String> = result_ids.iter().collect();
        if unique.len() < result_ids.len() {
            anomalies.push(format!("{} duplicate IDs found", result_ids.len() - unique.len()));
        }

        // Fewer results than k
        if result_ids.len() < k {
            anomalies.push(format!("Only {}/{} results returned", result_ids.len(), k));
        }

        // Degradation detection
        if let Some(deg) = self.detect_degradation(precision, recall) {
            anomalies.push(deg);
            suggestions.push("Consider re-indexing the dataset".into());
        }

        let quality_level = Self::classify_quality(precision, recall);
        suggestions.extend(self.generate_suggestions(quality_level, precision, recall));

        let report = QualityReport {
            precision_at_k: precision,
            recall_at_k: recall,
            quality_level,
            anomalies,
            suggestions,
            backend_used: backend.to_string(),
        };

        self.update_stats(&report);

        if self.enable_cross_backend {
            self.backend_results
                .entry(backend.to_string())
                .or_default()
                .push((result_ids.to_vec(), precision));
        }

        report
    }

    /// Evaluate results from multiple backends.
    pub fn evaluate_cross_backend(
        &mut self,
        results_map: &HashMap<String, Vec<String>>,
        ground_truth: Option<&[String]>,
        k: usize,
    ) -> HashMap<String, QualityReport> {
        let mut reports = HashMap::new();
        for (backend, ids) in results_map {
            let report = self.evaluate(ids, ground_truth, k, backend, None);
            reports.insert(backend.clone(), report);
        }
        reports
    }

    fn compute_precision(result_ids: &[String], ground_truth: Option<&[String]>, k: usize) -> f64 {
        let gt = match ground_truth {
            Some(g) if !g.is_empty() => g.iter().cloned().collect::<std::collections::HashSet<_>>(),
            _ => return 1.0,
        };
        let hits = result_ids.iter().filter(|id| gt.contains(*id)).count();
        hits as f64 / (result_ids.len().min(k) as f64)
    }

    fn compute_recall(result_ids: &[String], ground_truth: Option<&[String]>) -> f64 {
        let gt = match ground_truth {
            Some(g) if !g.is_empty() => g.iter().cloned().collect::<std::collections::HashSet<_>>(),
            _ => return 1.0,
        };
        let found = result_ids.iter().filter(|id| gt.contains(*id)).count();
        found as f64 / gt.len() as f64
    }

    fn detect_distance_anomalies(distances: &[f64]) -> Vec<String> {
        if distances.len() < 3 {
            return vec![];
        }
        let mut anomalies = Vec::new();

        // Check ascending order
        for i in 1..distances.len() {
            if distances[i] < distances[i - 1] - 1e-6 {
                anomalies.push("Distances not in ascending order".into());
                break;
            }
        }

        // Detect big jumps
        if distances.len() > 2 {
            let mut diffs: Vec<f64> = Vec::new();
            for i in 1..distances.len() {
                diffs.push(distances[i] - distances[i - 1]);
            }
            diffs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let median = diffs[(diffs.len() - 1) / 2];
            if median > 0.0 {
                for (i, &d) in diffs.iter().enumerate() {
                    if d > 5.0 * median {
                        anomalies.push(format!("Anomalous distance jump at position {}", i));
                    }
                }
            }
        }

        // Too uniform distances
        if distances.len() > 5 {
            let mean = distances.iter().sum::<f64>() / distances.len() as f64;
            let variance = distances.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / distances.len() as f64;
            let std = variance.sqrt();
            let cv = if mean > 0.0 { std / mean } else { 0.0 };
            if cv < 0.01 {
                anomalies.push("Distances too uniform (possible degenerate index)".into());
            }
        }

        anomalies
    }

    fn detect_degradation(&self, precision: f64, recall: f64) -> Option<String> {
        if self.quality_history.len() < DEGRADATION_WINDOW {
            return None;
        }

        let recent = &self.quality_history[self.quality_history.len() - DEGRADATION_WINDOW..];
        let avg_prec: f64 = recent.iter().map(|r| r.precision_at_k).sum::<f64>() / recent.len() as f64;
        let avg_rec: f64 = recent.iter().map(|r| r.recall_at_k).sum::<f64>() / recent.len() as f64;

        if precision < avg_prec * 0.7 && avg_prec > 0.5 {
            return Some(format!("Precision degradation: current={:.2} vs recent_avg={:.2}", precision, avg_prec));
        }
        if recall < avg_rec * 0.7 && avg_rec > 0.5 {
            return Some(format!("Recall degradation: current={:.2} vs recent_avg={:.2}", recall, avg_rec));
        }

        None
    }

    fn classify_quality(precision: f64, recall: f64) -> QualityLevel {
        let score = (precision + recall) / 2.0;
        if score >= 0.9 { QualityLevel::Excellent }
        else if score >= 0.7 { QualityLevel::Good }
        else if score >= 0.5 { QualityLevel::Acceptable }
        else if score >= 0.3 { QualityLevel::Poor }
        else { QualityLevel::Critical }
    }

    fn generate_suggestions(&mut self, level: QualityLevel, precision: f64, recall: f64) -> Vec<String> {
        let mut suggestions = Vec::new();
        match level {
            QualityLevel::Critical => {
                suggestions.push("[!] CRITICAL quality: re-index immediately".into());
                self.stats.reindex_suggestions += 1;
            }
            QualityLevel::Poor => {
                suggestions.push("[!] Low quality: consider re-indexing".into());
                self.stats.reindex_suggestions += 1;
            }
            QualityLevel::Acceptable => {
                if precision < self.precision_warning {
                    suggestions.push("Precision below warning threshold".into());
                }
                if recall < self.recall_warning {
                    suggestions.push("Recall below warning threshold".into());
                }
            }
            _ => {}
        }
        suggestions
    }

    fn update_stats(&mut self, report: &QualityReport) {
        self.stats.total_evaluations += 1;
        let n = self.stats.total_evaluations as f64;
        self.stats.avg_precision = (self.stats.avg_precision * (n - 1.0) + report.precision_at_k) / n;
        self.stats.avg_recall = (self.stats.avg_recall * (n - 1.0) + report.recall_at_k) / n;
        self.stats.anomalies_detected += report.anomalies.len() as u64;

        self.quality_history.push(report.clone());
        if self.quality_history.len() > MAX_HISTORY {
            let start = self.quality_history.len() - MAX_HISTORY;
            self.quality_history = self.quality_history.split_off(start);
        }
    }

    /// Check if re-indexing is recommended based on history.
    pub fn should_reindex(&self) -> bool {
        if self.quality_history.len() < DEGRADATION_WINDOW {
            return false;
        }
        let recent = &self.quality_history[self.quality_history.len() - DEGRADATION_WINDOW..];
        let avg: f64 = recent.iter().map(|r| r.precision_at_k).sum::<f64>() / recent.len() as f64;
        avg < self.precision_critical
    }

    /// Get reflector statistics.
    pub fn get_stats(&self) -> &ReflectorStats {
        &self.stats
    }

    /// Get quality history.
    pub fn get_history(&self) -> &[QualityReport] {
        &self.quality_history
    }
}

impl Default for QualityReflector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_excellent_quality() {
        let mut r = QualityReflector::new();
        let report = r.evaluate(
            &["1".into(), "2".into(), "3".into(), "4".into(), "5".into()],
            Some(&["1".into(), "2".into(), "3".into(), "4".into(), "5".into()]),
            5, "cpu", None,
        );
        assert_eq!(report.quality_level, QualityLevel::Excellent);
        assert_eq!(report.precision_at_k, 1.0);
        assert_eq!(report.recall_at_k, 1.0);
    }

    #[test]
    fn test_poor_quality() {
        let mut r = QualityReflector::new();
        let report = r.evaluate(
            &["1".into(), "2".into(), "x".into(), "y".into(), "z".into()],
            Some(&["1".into(), "2".into(), "3".into(), "4".into(), "5".into()]),
            5, "cpu", None,
        );
        assert_eq!(report.quality_level, QualityLevel::Poor);
    }

    #[test]
    fn test_distance_anomaly() {
        let mut r = QualityReflector::new();
        let distances = vec![0.1, 0.2, 100.0]; // big jump
        let report = r.evaluate(
            &["a".into(), "b".into(), "c".into()],
            Some(&["a".into(), "b".into(), "c".into()]),
            3, "cpu", Some(&distances),
        );
        assert!(!report.anomalies.is_empty());
    }

    #[test]
    fn test_duplicate_detection() {
        let mut r = QualityReflector::new();
        let report = r.evaluate(
            &["a".into(), "a".into(), "b".into()],
            Some(&["a".into(), "b".into()]),
            3, "cpu", None,
        );
        assert!(report.anomalies.iter().any(|a| a.contains("duplicate")));
    }

    #[test]
    fn test_no_ground_truth() {
        let mut r = QualityReflector::new();
        let report = r.evaluate(
            &["1".into(), "2".into()],
            None, 2, "cpu", None,
        );
        assert_eq!(report.precision_at_k, 1.0);
        assert_eq!(report.quality_level, QualityLevel::Excellent);
    }

    #[test]
    fn test_cross_backend() {
        let mut r = QualityReflector::new();
        let mut map = HashMap::new();
        map.insert("cpu".into(), vec!["1".into(), "2".into(), "3".into()]);
        map.insert("cuda".into(), vec!["1".into(), "2".into()]);
        let gt = vec!["1".into(), "2".into(), "3".into()];
        let reports = r.evaluate_cross_backend(&map, Some(&gt), 3);
        assert_eq!(reports.len(), 2);
    }

    #[test]
    fn test_should_reindex() {
        let mut r = QualityReflector::new();
        for _ in 0..15 {
            r.evaluate(&["x".into()], Some(&["1".into(), "2".into(), "3".into(), "4".into(), "5".into()]), 5, "cpu", None);
        }
        assert!(r.should_reindex());
    }
}


