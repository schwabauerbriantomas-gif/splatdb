//! Optimized SplatDB API with caching, GPU config, and auto-scaling hooks.
//!
//! Integrates query caching, prefetching, and metrics tracking.
//! Ported from splatdb Python.

use ndarray::Array1;
use serde::Serialize;

use crate::query_optimizer::QueryOptimizer;

/// Optimization statistics.
#[derive(Debug, Clone, Default, Serialize)]
pub struct OptimizationMetrics {
    pub total_queries: u64,
    pub total_adds: u64,
    pub gpu_queries: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
}

/// GPU configuration detected by auto-tuner.
#[derive(Debug, Clone, Serialize)]
pub struct GpuConfig {
    pub device_name: String,
    pub vram_mb: u64,
    pub optimal_batch_size: usize,
    pub compute_units: u32,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            device_name: "cpu".into(),
            vram_mb: 0,
            optimal_batch_size: 100,
            compute_units: 0,
        }
    }
}

/// Optimized SplatDB database wrapper.
pub struct M2MOptimized {
    pub enable_gpu: bool,
    pub enable_cache: bool,
    pub enable_autoscale: bool,
    pub gpu_config: Option<GpuConfig>,
    pub optimizer: Option<QueryOptimizer>,
    pub latent_dim: usize,
    metrics: OptimizationMetrics,
}

impl M2MOptimized {
    /// New.
    pub fn new(
        latent_dim: usize,
        enable_gpu: bool,
        enable_cache: bool,
        cache_entries: usize,
        cache_memory_mb: usize,
        enable_autoscale: bool,
    ) -> Self {
        let gpu_config = if enable_gpu { Self::detect_gpu() } else { None };

        let optimizer = if enable_cache {
            Some(QueryOptimizer::new(cache_entries, cache_memory_mb, true))
        } else {
            None
        };

        Self {
            enable_gpu,
            enable_cache,
            enable_autoscale,
            gpu_config,
            optimizer,
            latent_dim,
            metrics: OptimizationMetrics::default(),
        }
    }

    /// Create with default settings.
    pub fn with_dim(latent_dim: usize) -> Self {
        Self::new(latent_dim, true, true, 1000, 100, false)
    }

    fn detect_gpu() -> Option<GpuConfig> {
        // In Rust, GPU detection would use wgpu or cudarc
        // For now, return None (CPU only)
        None
    }

    /// Record a query.
    pub fn record_query(&mut self) {
        self.metrics.total_queries += 1;
    }

    /// Record an add operation.
    pub fn record_add(&mut self) {
        self.metrics.total_adds += 1;
    }

    /// Search with caching if enabled.
    pub fn search_cached<F>(&mut self, query: &Array1<f32>, k: usize, search_fn: F) -> String
    where
        F: FnOnce(&Array1<f32>, usize) -> String,
    {
        self.metrics.total_queries += 1;

        if let Some(ref mut optimizer) = self.optimizer {
            let result = optimizer.execute_with_cache(query, k, None, search_fn);
            let stats = optimizer.get_stats();
            if let Some(cache) = stats.get("cache") {
                self.metrics.cache_hits = cache["hits"].as_u64().unwrap_or(0);
                self.metrics.cache_misses = cache["misses"].as_u64().unwrap_or(0);
            }
            result
        } else {
            search_fn(query, k)
        }
    }

    /// Update cluster metrics for auto-scaling.
    pub fn update_cluster_metrics(
        &mut self,
        cpu_percent: f64,
        memory_percent: f64,
        qps: f64,
        latency_ms: f64,
    ) {
        // Placeholder for auto-scaling integration
        let _ = (cpu_percent, memory_percent, qps, latency_ms);
    }

    /// Get optimization statistics.
    pub fn get_optimization_stats(&self) -> serde_json::Value {
        serde_json::json!({
            "optimization": &self.metrics,
            "gpu": self.gpu_config.as_ref().map(|g| serde_json::to_value(g).unwrap_or(serde_json::json!({}))).unwrap_or(serde_json::json!({})),
            "cache": self.optimizer.as_ref().map(|o| o.get_stats()).unwrap_or(serde_json::json!({"enabled": false})),
        })
    }

    /// Get prefetch suggestions.
    pub fn get_prefetch_suggestions(&self, n: usize) -> Vec<String> {
        self.optimizer
            .as_ref()
            .map(|o| o.get_prefetch_suggestions(n))
            .unwrap_or_default()
    }

    /// Clear query cache.
    pub fn clear_cache(&mut self) {
        if let Some(ref mut opt) = self.optimizer {
            opt.clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_optimized() {
        let opt = M2MOptimized::with_dim(640);
        assert_eq!(opt.latent_dim, 640);
        assert!(opt.enable_gpu);
        assert!(opt.enable_cache);
    }

    #[test]
    fn test_search_cached() {
        let mut opt = M2MOptimized::new(64, false, true, 100, 10, false);
        let q = Array1::from_vec(vec![1.0, 2.0]);
        let r1 = opt.search_cached(&q, 5, |_, _| "result1".into());
        assert_eq!(r1, "result1");
        assert_eq!(opt.metrics.total_queries, 1);

        let r2 = opt.search_cached(&q, 5, |_, _| "result2".into());
        assert_eq!(r2, "result1"); // cached
        assert_eq!(opt.metrics.total_queries, 2);
    }

    #[test]
    fn test_stats() {
        let opt = M2MOptimized::with_dim(384);
        let stats = opt.get_optimization_stats();
        assert!(stats.is_object());
    }

    #[test]
    fn test_clear_cache() {
        let mut opt = M2MOptimized::new(64, false, true, 100, 10, false);
        let q = Array1::from_vec(vec![1.0]);
        opt.search_cached(&q, 5, |_, _| "r".into());
        opt.clear_cache();
        assert!(opt.optimizer.is_some());
    }
}
