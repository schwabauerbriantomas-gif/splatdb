//! Cluster load balancer.
//! Ported from splatdb Python.

use std::collections::HashMap;

use crate::cluster::health::LoadMetrics;

/// Load balancer for distributing queries across cluster nodes.
pub struct LoadBalancer {
    current_rr_index: usize,
}

impl LoadBalancer {
    pub fn new() -> Self {
        Self {
            current_rr_index: 0,
        }
    }

    /// Round-robin: returns a single edge sequentially.
    pub fn route_round_robin(&mut self, online_edges: &[String]) -> Option<String> {
        if online_edges.is_empty() {
            return None;
        }
        let idx = self.current_rr_index % online_edges.len();
        self.current_rr_index = (self.current_rr_index + 1) % online_edges.len();
        Some(online_edges[idx].clone())
    }

    /// Returns edges sorted by least active queries and lowest latency.
    pub fn route_least_loaded(
        &self,
        online_edges: &[String],
        load_metrics: &HashMap<String, LoadMetrics>,
    ) -> Vec<String> {
        let mut scored: Vec<(String, f64)> = online_edges
            .iter()
            .map(|eid| {
                let m = load_metrics.get(eid);
                let score = match m {
                    Some(metrics) => {
                        (metrics.active_queries as f64) * 100.0 + metrics.query_latency_ms
                    }
                    None => 0.0,
                };
                (eid.clone(), score)
            })
            .collect();
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().map(|(id, _)| id).collect()
    }

    /// Main entrypoint: map strategy names to routing logic.
    pub fn select_best_edges(
        &mut self,
        online_edges: &[String],
        load_metrics: &HashMap<String, LoadMetrics>,
        strategy: &str,
    ) -> Vec<String> {
        match strategy {
            "broadcast" => online_edges.to_vec(),
            "round_robin" => self.route_round_robin(online_edges).into_iter().collect(),
            "least_loaded" => self.route_least_loaded(online_edges, load_metrics),
            _ => online_edges.to_vec(),
        }
    }
}

impl Default for LoadBalancer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_round_robin() {
        let mut lb = LoadBalancer::new();
        let edges = vec!["a".into(), "b".into(), "c".into()];
        assert_eq!(lb.route_round_robin(&edges), Some("a".into()));
        assert_eq!(lb.route_round_robin(&edges), Some("b".into()));
        assert_eq!(lb.route_round_robin(&edges), Some("c".into()));
        assert_eq!(lb.route_round_robin(&edges), Some("a".into()));
    }

    #[test]
    fn test_least_loaded() {
        let lb = LoadBalancer::new();
        let edges = vec!["a".into(), "b".into()];
        let mut metrics = HashMap::new();
        metrics.insert(
            "a".into(),
            LoadMetrics {
                active_queries: 10,
                query_latency_ms: 50.0,
                ..Default::default()
            },
        );
        metrics.insert(
            "b".into(),
            LoadMetrics {
                active_queries: 2,
                query_latency_ms: 10.0,
                ..Default::default()
            },
        );
        let result = lb.route_least_loaded(&edges, &metrics);
        assert_eq!(result[0], "b");
    }
}
