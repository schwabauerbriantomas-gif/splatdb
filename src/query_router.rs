//! Query Router — classifies queries and routes to optimal search strategy.
//!
//! Patterns: Exact k-NN, HRM2 approximate, Range, Batch parallel, LSH.
//! Auto-learning from latency history.
//! Ported from m2m-vector-search Python.

use serde::Serialize;
use std::collections::HashMap;

/// Available search strategies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub enum SearchStrategy {
    Exact,
    ApproximateHrm2,
    Range,
    BatchParallel,
    Lsh,
}

impl SearchStrategy {
    /// As str.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Exact => "exact",
            Self::ApproximateHrm2 => "hrm2",
            Self::Range => "range",
            Self::BatchParallel => "batch",
            Self::Lsh => "lsh",
        }
    }
}

/// Query profile for classification.
#[derive(Debug, Clone)]
pub struct QueryProfile {
    pub k: usize,
    pub batch_size: usize,
    pub query_dim: usize,
    pub has_range_filter: bool,
    pub range_radius: f64,
    pub dataset_size: usize,
    pub priority: u8, // 0=normal, 1=low-latency, 2=high-recall
}

impl Default for QueryProfile {
    fn default() -> Self {
        Self {
            k: 10,
            batch_size: 1,
            query_dim: 0,
            has_range_filter: false,
            range_radius: 0.0,
            dataset_size: 0,
            priority: 0,
        }
    }
}

/// Routing decision.
#[derive(Debug, Clone, Serialize)]
pub struct RouteDecision {
    pub strategy: SearchStrategy,
    pub confidence: f64,
    pub reason: String,
    pub estimated_latency_ms: f64,
}

/// Router statistics.
#[derive(Debug, Clone, Default, Serialize)]
pub struct RouterStats {
    pub total_routes: u64,
    pub routes_by_strategy: HashMap<String, u64>,
    pub avg_latency_by_strategy: HashMap<String, f64>,
    pub auto_adjustments: u64,
}

const HRM2_MIN_DATASET: usize = 1000;
const HRM2_MIN_K: usize = 5;
const BATCH_MIN_SIZE: usize = 4;
const EXACT_MAX_DATASET: usize = 5000;

/// Query router for M2M.
pub struct QueryRouter {
    _default_strategy: SearchStrategy,
    enable_auto_learning: bool,
    learning_window: usize,
    strategies: HashMap<SearchStrategy, String>, // strategy name -> registered
    stats: RouterStats,
    latency_history: HashMap<SearchStrategy, Vec<f64>>,
}

impl QueryRouter {
    /// New.
    pub fn new(default_strategy: SearchStrategy, enable_auto_learning: bool) -> Self {
        let mut latency_history = HashMap::new();
        for s in [SearchStrategy::Exact, SearchStrategy::ApproximateHrm2, SearchStrategy::Range, SearchStrategy::BatchParallel, SearchStrategy::Lsh] {
            latency_history.insert(s, Vec::new());
        }
        Self {
            _default_strategy: default_strategy,
            enable_auto_learning,
            learning_window: 100,
            strategies: HashMap::new(),
            stats: RouterStats::default(),
            latency_history,
        }
    }

    /// Register a search strategy as available.
    pub fn register_strategy(&mut self, strategy: SearchStrategy) {
        self.strategies.insert(strategy, strategy.as_str().to_string());
    }

    /// Unregister a strategy.
    pub fn unregister_strategy(&mut self, strategy: SearchStrategy) {
        self.strategies.remove(&strategy);
    }

    /// Classify a query profile and select the best strategy.
    pub fn classify(&self, profile: &QueryProfile) -> RouteDecision {
        // 1. Range search priority
        if profile.has_range_filter {
            return RouteDecision {
                strategy: SearchStrategy::Range,
                confidence: 0.95,
                reason: "Range filter active".into(),
                estimated_latency_ms: self.estimate_latency(SearchStrategy::Range),
            };
        }

        // 2. Batch search
        if profile.batch_size >= BATCH_MIN_SIZE {
            let (strategy, reason) = if profile.dataset_size >= HRM2_MIN_DATASET {
                (SearchStrategy::ApproximateHrm2, format!("Batch of {} with large dataset", profile.batch_size))
            } else {
                (SearchStrategy::BatchParallel, format!("Batch of {} queries", profile.batch_size))
            };
            return RouteDecision {
                strategy,
                confidence: 0.85,
                reason,
                estimated_latency_ms: self.estimate_latency(strategy),
            };
        }

        // 3. Auto-learning adjustment
        if let Some(adjusted) = self.auto_adjust() {
            return adjusted;
        }

        // 4. Decision based on k and dataset size
        if profile.dataset_size < HRM2_MIN_DATASET || profile.k < HRM2_MIN_K {
            RouteDecision {
                strategy: SearchStrategy::Exact,
                confidence: 0.9,
                reason: format!("Dataset={}, k={} -> exact", profile.dataset_size, profile.k),
                estimated_latency_ms: self.estimate_latency(SearchStrategy::Exact),
            }
        } else if profile.dataset_size <= EXACT_MAX_DATASET && profile.k <= 20 {
            RouteDecision {
                strategy: SearchStrategy::Exact,
                confidence: 0.75,
                reason: format!("Dataset={}, k={} -> exact viable", profile.dataset_size, profile.k),
                estimated_latency_ms: self.estimate_latency(SearchStrategy::Exact),
            }
        } else {
            RouteDecision {
                strategy: SearchStrategy::ApproximateHrm2,
                confidence: 0.85,
                reason: format!("Dataset={}, k={} -> HRM2", profile.dataset_size, profile.k),
                estimated_latency_ms: self.estimate_latency(SearchStrategy::ApproximateHrm2),
            }
        }
    }

    /// Alias for classify (Router pattern compatibility).
    pub fn route(&self, profile: &QueryProfile) -> RouteDecision {
        self.classify(profile)
    }

    /// Record latency for a strategy.
    pub fn record_latency(&mut self, strategy: SearchStrategy, latency_ms: f64) {
        self.stats.total_routes += 1;
        *self.stats.routes_by_strategy.entry(strategy.as_str().to_string()).or_insert(0) += 1;

        let history = self.latency_history.entry(strategy).or_default();
        history.push(latency_ms);
        if history.len() > self.learning_window {
            let start = history.len() - self.learning_window;
            *history = history.split_off(start);
        }

        let avg: f64 = history.iter().sum::<f64>() / history.len() as f64;
        self.stats.avg_latency_by_strategy.insert(strategy.as_str().to_string(), avg);
    }

    fn auto_adjust(&self) -> Option<RouteDecision> {
        if !self.enable_auto_learning {
            return None;
        }

        let relevant = [SearchStrategy::Exact, SearchStrategy::ApproximateHrm2];
        let mut latencies = HashMap::new();

        for &s in &relevant {
            let history = self.latency_history.get(&s)?;
            if history.len() >= 10 {
                let avg: f64 = history.iter().sum::<f64>() / history.len() as f64;
                latencies.insert(s, avg);
            }
        }

        if latencies.len() < 2 {
            return None;
        }

        let best = latencies.iter().min_by(|a, b| {
            a.1.partial_cmp(b.1).expect("latency values should be finite and comparable")
        })?;
        Some(RouteDecision {
            strategy: *best.0,
            confidence: 0.7,
            reason: format!("Auto-learning: {} is faster ({:.1}ms)", best.0.as_str(), best.1),
            estimated_latency_ms: *best.1,
        })
    }

    fn estimate_latency(&self, strategy: SearchStrategy) -> f64 {
        self.latency_history
            .get(&strategy)
            .filter(|h| !h.is_empty())
            .map(|h| h.iter().sum::<f64>() / h.len() as f64)
            .unwrap_or(0.0)
    }

    /// Get router statistics.
    pub fn get_stats(&self) -> &RouterStats {
        &self.stats
    }

    /// Check if a strategy is registered.
    pub fn has_strategy(&self, strategy: SearchStrategy) -> bool {
        self.strategies.contains_key(&strategy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_small_dataset_exact() {
        let router = QueryRouter::new(SearchStrategy::Exact, false);
        let profile = QueryProfile { dataset_size: 500, k: 5, ..Default::default() };
        let decision = router.classify(&profile);
        assert_eq!(decision.strategy, SearchStrategy::Exact);
    }

    #[test]
    fn test_large_dataset_hrm2() {
        let router = QueryRouter::new(SearchStrategy::Exact, false);
        let profile = QueryProfile { dataset_size: 50000, k: 20, ..Default::default() };
        let decision = router.classify(&profile);
        assert_eq!(decision.strategy, SearchStrategy::ApproximateHrm2);
    }

    #[test]
    fn test_range_filter() {
        let router = QueryRouter::new(SearchStrategy::Exact, false);
        let profile = QueryProfile { has_range_filter: true, ..Default::default() };
        let decision = router.classify(&profile);
        assert_eq!(decision.strategy, SearchStrategy::Range);
        assert!(decision.confidence > 0.9);
    }

    #[test]
    fn test_batch_parallel() {
        let router = QueryRouter::new(SearchStrategy::Exact, false);
        let profile = QueryProfile { batch_size: 10, dataset_size: 500, ..Default::default() };
        let decision = router.classify(&profile);
        assert_eq!(decision.strategy, SearchStrategy::BatchParallel);
    }

    #[test]
    fn test_record_latency() {
        let mut router = QueryRouter::new(SearchStrategy::Exact, false);
        router.record_latency(SearchStrategy::Exact, 10.0);
        router.record_latency(SearchStrategy::Exact, 20.0);
        let stats = router.get_stats();
        assert_eq!(stats.total_routes, 2);
        let avg = stats.avg_latency_by_strategy.get("exact").unwrap();
        assert!(*avg > 14.0 && *avg < 16.0);
    }
}


