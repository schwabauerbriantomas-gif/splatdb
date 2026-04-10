//! Search Supervisor — orchestrates multi-backend search (CPU, CUDA, Vulkan).
//!
//! Pattern: MASFactory Supervisor.
//! Decides backend based on query complexity, hardware, latency budget.
//! Automatic fallback on errors.
//! Ported from splatsdb Python.

use serde::Serialize;
use std::collections::HashMap;

/// Backend types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub enum BackendType {
    Cpu,
    Cuda,
    Vulkan,
}

impl BackendType {
    /// As str.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::Vulkan => "vulkan",
        }
    }
}

/// Query complexity levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum QueryComplexity {
    Simple,
    Moderate,
    Complex,
}

/// Information about a registered backend.
#[derive(Debug, Clone)]
pub struct BackendInfo {
    pub backend: BackendType,
    pub available: bool,
    pub avg_latency_ms: f64,
    pub total_queries: u64,
    pub total_errors: u64,
    pub max_dimension: usize,
}

/// Supervisor decision.
#[derive(Debug, Clone, Serialize)]
pub struct SupervisorDecision {
    pub backend: BackendType,
    pub reason: String,
    pub fallback_chain: Vec<BackendType>,
}

/// Supervisor statistics.
#[derive(Debug, Clone, Default, Serialize)]
pub struct SupervisorStats {
    pub total_decisions: u64,
    pub decisions_by_backend: HashMap<String, u64>,
    pub fallbacks_triggered: u64,
    pub total_errors: u64,
    pub avg_decision_time_ms: f64,
}

const SIMPLE_K_THRESHOLD: usize = 20;
const SIMPLE_DATASET_THRESHOLD: usize = 10_000;
const MODERATE_K_THRESHOLD: usize = 100;
const MODERATE_DATASET_THRESHOLD: usize = 100_000;

/// Search supervisor for multi-backend orchestration.
pub struct SearchSupervisor {
    _default_backend: BackendType,
    latency_budget_ms: Option<f64>,
    _enable_auto_fallback: bool,
    backends: HashMap<BackendType, BackendInfo>,
    stats: SupervisorStats,
    latency_history: HashMap<BackendType, Vec<f64>>,
}

impl SearchSupervisor {
    /// New.
    pub fn new(default_backend: BackendType, enable_auto_fallback: bool) -> Self {
        let mut backends = HashMap::new();
        for bt in [BackendType::Cpu, BackendType::Cuda, BackendType::Vulkan] {
            backends.insert(
                bt,
                BackendInfo {
                    backend: bt,
                    available: false,
                    avg_latency_ms: 0.0,
                    total_queries: 0,
                    total_errors: 0,
                    max_dimension: 0,
                },
            );
        }
        let mut latency_history = HashMap::new();
        for bt in [BackendType::Cpu, BackendType::Cuda, BackendType::Vulkan] {
            latency_history.insert(bt, Vec::new());
        }

        Self {
            _default_backend: default_backend,
            latency_budget_ms: None,
            _enable_auto_fallback: enable_auto_fallback,
            backends,
            stats: SupervisorStats::default(),
            latency_history,
        }
    }

    /// Register a backend as available.
    pub fn register_backend(&mut self, backend_type: BackendType, max_dimension: usize) {
        if let Some(info) = self.backends.get_mut(&backend_type) {
            info.available = true;
            info.max_dimension = max_dimension;
        }
    }

    /// Unregister a backend.
    pub fn unregister_backend(&mut self, backend_type: BackendType) {
        if let Some(info) = self.backends.get_mut(&backend_type) {
            info.available = false;
        }
    }

    /// Classify query complexity.
    pub fn classify_complexity(
        &self,
        k: usize,
        dataset_size: usize,
        batch_size: usize,
    ) -> QueryComplexity {
        if batch_size > 1 {
            return QueryComplexity::Complex;
        }
        if k <= SIMPLE_K_THRESHOLD && dataset_size <= SIMPLE_DATASET_THRESHOLD {
            QueryComplexity::Simple
        } else if k <= MODERATE_K_THRESHOLD && dataset_size <= MODERATE_DATASET_THRESHOLD {
            QueryComplexity::Moderate
        } else {
            QueryComplexity::Complex
        }
    }

    /// Decide which backend to use.
    pub fn decide_backend(
        &mut self,
        k: usize,
        dataset_size: usize,
        query_dim: usize,
    ) -> anyhow::Result<SupervisorDecision> {
        let start = std::time::Instant::now();
        let complexity = self.classify_complexity(k, dataset_size, 1);

        let available: Vec<BackendType> = self
            .backends
            .iter()
            .filter(|(_, info)| info.available)
            .map(|(bt, _)| *bt)
            .collect();

        if available.is_empty() {
            self.stats.total_errors += 1;
            return Err(anyhow::anyhow!("No backends available"));
        }

        let preference_order = self.get_preference_order(complexity, &available, query_dim);
        let primary = preference_order[0];
        let fallback_chain = preference_order[1..].to_vec();

        self.stats.total_decisions += 1;
        *self
            .stats
            .decisions_by_backend
            .entry(primary.as_str().to_string())
            .or_insert(0) += 1;

        let elapsed_us = start.elapsed().as_micros() as f64 / 1000.0;
        self.stats.avg_decision_time_ms = (self.stats.avg_decision_time_ms
            * (self.stats.total_decisions - 1) as f64
            + elapsed_us)
            / self.stats.total_decisions as f64;

        Ok(SupervisorDecision {
            backend: primary,
            reason: format!("complexity={:?}, available={}", complexity, available.len()),
            fallback_chain,
        })
    }

    fn get_preference_order(
        &self,
        complexity: QueryComplexity,
        available: &[BackendType],
        query_dim: usize,
    ) -> Vec<BackendType> {
        let order = match complexity {
            QueryComplexity::Simple => [BackendType::Cpu, BackendType::Vulkan, BackendType::Cuda],
            QueryComplexity::Moderate => [BackendType::Vulkan, BackendType::Cuda, BackendType::Cpu],
            QueryComplexity::Complex => [BackendType::Cuda, BackendType::Vulkan, BackendType::Cpu],
        };

        let valid: Vec<BackendType> = order
            .iter()
            .filter(|bt| {
                available.contains(bt) && {
                    let info = self.backends.get(bt).expect("backend not found in map");
                    info.max_dimension == 0 || query_dim <= info.max_dimension
                }
            })
            .copied()
            .collect();

        if valid.is_empty() {
            available.to_vec()
        } else {
            valid
        }
    }

    /// Record that a backend completed a query.
    pub fn record_query_result(&mut self, backend: BackendType, elapsed_ms: f64, success: bool) {
        if let Some(info) = self.backends.get_mut(&backend) {
            info.total_queries += 1;
            if success {
                info.avg_latency_ms = (info.avg_latency_ms * (info.total_queries - 1) as f64
                    + elapsed_ms)
                    / info.total_queries as f64;
                let history = self.latency_history.entry(backend).or_default();
                history.push(elapsed_ms);
                if history.len() > 100 {
                    let start = history.len() - 100;
                    *history = history.split_off(start);
                }
            } else {
                info.total_errors += 1;
            }
        }
    }

    /// Record a fallback event.
    pub fn record_fallback(&mut self) {
        self.stats.fallbacks_triggered += 1;
    }

    /// Health check for all backends.
    pub fn health_check(&self) -> serde_json::Value {
        let mut status = serde_json::Map::new();
        for (bt, info) in &self.backends {
            let error_rate = if info.total_queries > 0 {
                info.total_errors as f64 / info.total_queries as f64
            } else {
                0.0
            };
            status.insert(
                bt.as_str().to_string(),
                serde_json::json!({
                    "available": info.available,
                    "total_queries": info.total_queries,
                    "total_errors": info.total_errors,
                    "avg_latency_ms": (info.avg_latency_ms * 100.0).round() / 100.0,
                    "error_rate": (error_rate * 10000.0).round() / 10000.0,
                }),
            );
        }
        serde_json::Value::Object(status)
    }

    /// Get supervisor statistics.
    pub fn get_stats(&self) -> &SupervisorStats {
        &self.stats
    }

    /// Set latency budget.
    pub fn set_latency_budget(&mut self, budget_ms: f64) {
        self.latency_budget_ms = Some(budget_ms);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_complexity() {
        let sup = SearchSupervisor::new(BackendType::Cpu, true);
        assert_eq!(sup.classify_complexity(5, 1000, 1), QueryComplexity::Simple);
        assert_eq!(
            sup.classify_complexity(50, 50000, 1),
            QueryComplexity::Moderate
        );
        assert_eq!(
            sup.classify_complexity(200, 200000, 1),
            QueryComplexity::Complex
        );
        assert_eq!(
            sup.classify_complexity(5, 100, 10),
            QueryComplexity::Complex
        );
    }

    #[test]
    fn test_decide_backend_cpu_only() {
        let mut sup = SearchSupervisor::new(BackendType::Cpu, true);
        sup.register_backend(BackendType::Cpu, 0);
        let decision = sup.decide_backend(10, 500, 64).unwrap();
        assert_eq!(decision.backend, BackendType::Cpu);
    }

    #[test]
    fn test_decide_backend_with_cuda() {
        let mut sup = SearchSupervisor::new(BackendType::Cpu, true);
        sup.register_backend(BackendType::Cpu, 0);
        sup.register_backend(BackendType::Cuda, 0);
        // Complex query should prefer CUDA
        let decision = sup.decide_backend(200, 200000, 64).unwrap();
        assert_eq!(decision.backend, BackendType::Cuda);
    }

    #[test]
    fn test_record_and_health() {
        let mut sup = SearchSupervisor::new(BackendType::Cpu, true);
        sup.register_backend(BackendType::Cpu, 0);
        sup.record_query_result(BackendType::Cpu, 5.0, true);
        sup.record_query_result(BackendType::Cpu, 10.0, true);
        sup.record_query_result(BackendType::Cpu, 0.0, false);
        let health = sup.health_check();
        assert_eq!(health["cpu"]["total_queries"], 3);
    }
}
