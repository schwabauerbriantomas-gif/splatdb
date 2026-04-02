//! Cluster router with energy-aware routing support.
//! Wraps EnergyRouter and LoadBalancer for unified routing.
//! Ported from m2m-vector-search Python.

use std::collections::{HashMap, HashSet};

use crate::cluster::balancer::LoadBalancer;
use crate::cluster::energy_router::{EnergyRouter, EnergyRouterConfig};
use crate::cluster::health::{EdgeNodeInfo, LoadMetrics};

/// Unified cluster router.
pub struct ClusterRouter {
    edge_nodes: HashMap<String, EdgeNodeInfo>,
    metadata_index: HashMap<String, HashSet<String>>,
    load_metrics: HashMap<String, LoadMetrics>,
    _centroids: HashMap<String, Vec<f32>>,
    balancer: LoadBalancer,
    energy_router: EnergyRouter,
    heartbeat_timeout: f64,
}

impl ClusterRouter {
    pub fn new(energy_config: Option<EnergyRouterConfig>) -> Self {
        Self {
            edge_nodes: HashMap::new(),
            metadata_index: HashMap::new(),
            load_metrics: HashMap::new(),
            _centroids: HashMap::new(),
            balancer: LoadBalancer::new(),
            energy_router: EnergyRouter::new(energy_config.unwrap_or_default()),
            heartbeat_timeout: 30.0,
        }
    }

    /// Register an edge node.
    pub fn register_edge(&mut self, edge_id: &str, url: &str, weight: f64) {
        let info = EdgeNodeInfo::new(edge_id, url);
        self.edge_nodes.insert(edge_id.to_string(), info);
        self.load_metrics.insert(edge_id.to_string(), LoadMetrics::default());
        self.energy_router.register_node(edge_id, weight);
    }

    /// Remove an edge node.
    pub fn remove_edge(&mut self, edge_id: &str) {
        self.edge_nodes.remove(edge_id);
        self.load_metrics.remove(edge_id);
        self.energy_router.remove_node(edge_id);

        // Clean metadata index
        let empty_docs: Vec<String> = self.metadata_index
            .iter()
            .filter_map(|(doc_id, edges)| {
                if edges.contains(edge_id) {
                    Some(doc_id.clone())
                } else {
                    None
                }
            })
            .collect();
        for doc_id in empty_docs {
            if let Some(edges) = self.metadata_index.get_mut(&doc_id) {
                edges.remove(edge_id);
                if edges.is_empty() {
                    self.metadata_index.remove(&doc_id);
                }
            }
        }
    }

    /// Update heartbeat and metrics for an edge node.
    pub fn heartbeat(&mut self, edge_id: &str, metrics: LoadMetrics) {
        if let Some(node) = self.edge_nodes.get_mut(edge_id) {
            node.last_heartbeat = now_secs();
            node.status = "online".to_string();
            self.load_metrics.insert(edge_id.to_string(), metrics);
        }
    }

    /// Get list of online and responsive edge IDs.
    pub fn get_online_edges(&self) -> Vec<String> {
        let now = now_secs();
        self.edge_nodes
            .iter()
            .filter(|(_, node)| {
                node.status == "online" && (now - node.last_heartbeat) < self.heartbeat_timeout
            })
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// Mark stale nodes as offline.
    pub fn check_stale_nodes(&mut self) -> Vec<String> {
        let now = now_secs();
        let mut stale = Vec::new();
        for (id, node) in &mut self.edge_nodes {
            if node.status == "online" && (now - node.last_heartbeat) >= self.heartbeat_timeout {
                node.status = "offline".to_string();
                stale.push(id.clone());
            }
        }
        stale
    }

    /// Route a query to appropriate edge nodes.
    pub fn route_query(&mut self, query: &[f32], _k: usize, strategy: &str) -> Vec<String> {
        let online_edges = self.get_online_edges();
        if online_edges.is_empty() {
            return Vec::new();
        }

        let query_hash = hash_query(query);
        if self.energy_router_enabled() {
            return self.energy_router.route(query_hash, &online_edges, Some(&self.load_metrics));
        }

        self.balancer.select_best_edges(&online_edges, &self.load_metrics, strategy)
    }

    /// Register a document location.
    pub fn register_document(&mut self, doc_id: &str, edge_id: &str) {
        self.metadata_index
            .entry(doc_id.to_string())
            .or_default()
            .insert(edge_id.to_string());
    }

    /// Find which edges hold a document.
    pub fn locate_document(&self, doc_id: &str) -> Vec<String> {
        self.metadata_index
            .get(doc_id)
            .map(|edges| edges.iter().cloned().collect())
            .unwrap_or_default()
    }

    fn energy_router_enabled(&self) -> bool {
        // Check if energy router is enabled by looking at stats
        self.energy_router.get_routing_stats().enabled
    }

    /// Get routing statistics.
    pub fn get_routing_stats(&self) -> RouterStats {
        RouterStats {
            online_nodes: self.get_online_edges().len(),
            total_nodes: self.edge_nodes.len(),
            documents_indexed: self.metadata_index.len(),
            energy_router: self.energy_router.get_routing_stats(),
        }
    }

    pub fn update_node_energy(&mut self, edge_id: &str, energy: f64) {
        self.energy_router.update_node_energy(edge_id, energy);
    }
}

/// Overall router statistics.
#[derive(Debug)]
pub struct RouterStats {
    pub online_nodes: usize,
    pub total_nodes: usize,
    pub documents_indexed: usize,
    pub energy_router: crate::cluster::energy_router::EnergyRouterStats,
}

fn now_secs() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

fn hash_query(query: &[f32]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    for &v in query {
        v.to_bits().hash(&mut hasher);
    }
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_and_route() {
        let mut router = ClusterRouter::new(None);
        router.register_edge("e1", "localhost:8001", 1.0);
        router.register_edge("e2", "localhost:8002", 1.0);

        let query = vec![0.1, 0.2, 0.3];
        let targets = router.route_query(&query, 10, "broadcast");
        assert_eq!(targets.len(), 2);
    }

    #[test]
    fn test_remove_edge() {
        let mut router = ClusterRouter::new(None);
        router.register_edge("e1", "localhost:8001", 1.0);
        router.remove_edge("e1");
        assert!(router.get_online_edges().is_empty());
    }

    #[test]
    fn test_document_location() {
        let mut router = ClusterRouter::new(None);
        router.register_edge("e1", "localhost:8001", 1.0);
        router.register_document("doc1", "e1");
        let locs = router.locate_document("doc1");
        assert_eq!(locs, vec!["e1"]);
    }
}


