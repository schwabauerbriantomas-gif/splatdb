//! Coordinator API — cluster coordinator REST endpoints.
//! Request/response types and handler logic (framework-agnostic).
//! Ready to plug into actix-web, axum, or warp.
//! Ported from splatdb Python.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ─── Request Types ───

/// Register node request.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RegisterNodeRequest {
    pub node_id: String,
    pub role: String,
    pub address: String,
    pub capabilities: HashMap<String, bool>,
}

/// Heartbeat request.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct HeartbeatRequest {
    pub node_id: String,
    pub n_vectors: usize,
    pub cpu_percent: f64,
    pub memory_percent: f64,
    pub qps: f64,
    pub latency_ms: f64,
}

/// Route query request.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RouteQueryRequest {
    pub query: Vec<f32>,
    pub k: usize,
    pub strategy: String,
}

/// Assign shard request.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AssignShardRequest {
    pub shard_id: String,
    pub node_id: String,
}

// ─── Response Types ───

/// Node info.
#[derive(Debug, Clone, Serialize)]
pub struct NodeInfo {
    pub node_id: String,
    pub role: String,
    pub address: String,
    pub healthy: bool,
    pub n_vectors: usize,
    pub last_heartbeat: f64,
}

/// Cluster stats response.
#[derive(Debug, Clone, Serialize)]
pub struct ClusterStatsResponse {
    pub total_nodes: usize,
    pub healthy_nodes: usize,
    pub total_vectors: usize,
    pub shards: usize,
    pub avg_latency_ms: f64,
}

/// Route query response.
#[derive(Debug, Clone, Serialize)]
pub struct RouteQueryResponse {
    pub target_nodes: Vec<String>,
    pub strategy: String,
}

// ─── Coordinator API Handler ───

const MAX_REGISTRATIONS: usize = 10_000;

/// Coordinator API: manages cluster nodes, routing, and shard assignment.
pub struct CoordinatorApi {
    nodes: HashMap<String, NodeState>,
    shard_assignments: HashMap<String, String>, // shard_id -> node_id
    total_queries: usize,
    _total_latency_ms: f64,
}

#[derive(Debug)]
struct NodeState {
    node_id: String,
    role: String,
    address: String,
    healthy: bool,
    n_vectors: usize,
    last_heartbeat: f64,
    cpu_percent: f64,
    memory_percent: f64,
    qps: f64,
    latency_ms: f64,
}

impl CoordinatorApi {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            shard_assignments: HashMap::new(),
            total_queries: 0,
            _total_latency_ms: 0.0,
        }
    }

    // ─── Node Management ───

    pub fn register_node(&mut self, req: &RegisterNodeRequest) -> Result<NodeInfo, String> {
        if self.nodes.contains_key(&req.node_id) {
            return Err(format!("Node '{}' already registered", req.node_id));
        }
        if self.nodes.len() >= MAX_REGISTRATIONS {
            return Err("maximum node registrations reached".into());
        }
        let state = NodeState {
            node_id: req.node_id.clone(),
            role: req.role.clone(),
            address: req.address.clone(),
            healthy: true,
            n_vectors: 0,
            last_heartbeat: now_secs(),
            cpu_percent: 0.0,
            memory_percent: 0.0,
            qps: 0.0,
            latency_ms: 0.0,
        };
        self.nodes.insert(req.node_id.clone(), state);
        self.get_node(&req.node_id)
            .ok_or("Node registered but not found".into())
    }

    pub fn unregister_node(&mut self, node_id: &str) -> Result<(), String> {
        self.nodes
            .remove(node_id)
            .map(|_| ())
            .ok_or_else(|| format!("Node '{}' not found", node_id))
    }

    pub fn heartbeat(&mut self, req: &HeartbeatRequest) -> Result<(), String> {
        let node = self
            .nodes
            .get_mut(&req.node_id)
            .ok_or_else(|| format!("Node '{}' not found", req.node_id))?;
        node.healthy = true;
        node.n_vectors = req.n_vectors;
        node.cpu_percent = req.cpu_percent;
        node.memory_percent = req.memory_percent;
        node.qps = req.qps;
        node.latency_ms = req.latency_ms;
        node.last_heartbeat = now_secs();
        Ok(())
    }

    pub fn get_node(&self, node_id: &str) -> Option<NodeInfo> {
        self.nodes.get(node_id).map(|n| NodeInfo {
            node_id: n.node_id.clone(),
            role: n.role.clone(),
            address: n.address.clone(),
            healthy: n.healthy,
            n_vectors: n.n_vectors,
            last_heartbeat: n.last_heartbeat,
        })
    }

    pub fn list_nodes(&self) -> Vec<NodeInfo> {
        self.nodes
            .values()
            .map(|n| NodeInfo {
                node_id: n.node_id.clone(),
                role: n.role.clone(),
                address: n.address.clone(),
                healthy: n.healthy,
                n_vectors: n.n_vectors,
                last_heartbeat: n.last_heartbeat,
            })
            .collect()
    }

    // ─── Routing ───

    pub fn route_query(&mut self, req: &RouteQueryRequest) -> RouteQueryResponse {
        let healthy_nodes: Vec<&NodeState> = self.nodes.values().filter(|n| n.healthy).collect();

        let targets = match req.strategy.as_str() {
            "broadcast" => healthy_nodes.iter().map(|n| n.node_id.clone()).collect(),
            "round_robin" => {
                // Simple: pick first healthy node
                healthy_nodes
                    .first()
                    .map(|n| vec![n.node_id.clone()])
                    .unwrap_or_default()
            }
            "least_loaded" => {
                // Pick node with lowest CPU
                healthy_nodes
                    .iter()
                    .min_by(|a, b| {
                        a.cpu_percent
                            .partial_cmp(&b.cpu_percent)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|n| vec![n.node_id.clone()])
                    .unwrap_or_default()
            }
            _ => healthy_nodes.iter().map(|n| n.node_id.clone()).collect(),
        };

        self.total_queries += 1;
        RouteQueryResponse {
            target_nodes: targets,
            strategy: req.strategy.clone(),
        }
    }

    // ─── Shard Management ───

    pub fn assign_shard(&mut self, req: &AssignShardRequest) -> Result<(), String> {
        if !self.nodes.contains_key(&req.node_id) {
            return Err(format!("Node '{}' not found", req.node_id));
        }
        self.shard_assignments
            .insert(req.shard_id.clone(), req.node_id.clone());
        Ok(())
    }

    pub fn get_shard_assignment(&self, shard_id: &str) -> Option<&str> {
        self.shard_assignments.get(shard_id).map(|s| s.as_str())
    }

    pub fn list_shards(&self) -> Vec<(String, String)> {
        self.shard_assignments
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    // ─── Stats ───

    pub fn cluster_stats(&self) -> ClusterStatsResponse {
        let healthy_count = self.nodes.values().filter(|n| n.healthy).count();
        let total_vectors: usize = self.nodes.values().map(|n| n.n_vectors).sum();
        let avg_latency = if !self.nodes.is_empty() {
            self.nodes.values().map(|n| n.latency_ms).sum::<f64>() / self.nodes.len() as f64
        } else {
            0.0
        };

        ClusterStatsResponse {
            total_nodes: self.nodes.len(),
            healthy_nodes: healthy_count,
            total_vectors,
            shards: self.shard_assignments.len(),
            avg_latency_ms: avg_latency,
        }
    }

    // ─── Health Checks ───

    /// Mark stale nodes as unhealthy (no heartbeat for 30s).
    pub fn check_stale_nodes(&mut self, timeout_secs: f64) -> Vec<String> {
        let now = now_secs();
        let mut stale = Vec::new();
        for node in self.nodes.values_mut() {
            if node.healthy && (now - node.last_heartbeat) > timeout_secs {
                node.healthy = false;
                stale.push(node.node_id.clone());
            }
        }
        stale
    }
}

impl Default for CoordinatorApi {
    fn default() -> Self {
        Self::new()
    }
}

fn now_secs() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_and_list() {
        let mut api = CoordinatorApi::new();
        api.register_node(&RegisterNodeRequest {
            node_id: "n1".into(),
            role: "worker".into(),
            address: "localhost:8001".into(),
            capabilities: HashMap::new(),
        })
        .unwrap();
        let nodes = api.list_nodes();
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].node_id, "n1");
    }

    #[test]
    fn test_heartbeat() {
        let mut api = CoordinatorApi::new();
        api.register_node(&RegisterNodeRequest {
            node_id: "n1".into(),
            role: "worker".into(),
            address: "localhost:8001".into(),
            capabilities: HashMap::new(),
        })
        .unwrap();
        api.heartbeat(&HeartbeatRequest {
            node_id: "n1".into(),
            n_vectors: 100,
            cpu_percent: 50.0,
            memory_percent: 30.0,
            qps: 10.0,
            latency_ms: 5.0,
        })
        .unwrap();
        let node = api.get_node("n1").unwrap();
        assert_eq!(node.n_vectors, 100);
    }

    #[test]
    fn test_route_broadcast() {
        let mut api = CoordinatorApi::new();
        api.register_node(&RegisterNodeRequest {
            node_id: "n1".into(),
            role: "worker".into(),
            address: "localhost:8001".into(),
            capabilities: HashMap::new(),
        })
        .unwrap();
        api.register_node(&RegisterNodeRequest {
            node_id: "n2".into(),
            role: "worker".into(),
            address: "localhost:8002".into(),
            capabilities: HashMap::new(),
        })
        .unwrap();
        let resp = api.route_query(&RouteQueryRequest {
            query: vec![0.0; 64],
            k: 10,
            strategy: "broadcast".into(),
        });
        assert_eq!(resp.target_nodes.len(), 2);
    }

    #[test]
    fn test_route_least_loaded() {
        let mut api = CoordinatorApi::new();
        api.register_node(&RegisterNodeRequest {
            node_id: "n1".into(),
            role: "worker".into(),
            address: "localhost:8001".into(),
            capabilities: HashMap::new(),
        })
        .unwrap();
        api.register_node(&RegisterNodeRequest {
            node_id: "n2".into(),
            role: "worker".into(),
            address: "localhost:8002".into(),
            capabilities: HashMap::new(),
        })
        .unwrap();
        api.heartbeat(&HeartbeatRequest {
            node_id: "n1".into(),
            n_vectors: 100,
            cpu_percent: 80.0,
            memory_percent: 30.0,
            qps: 10.0,
            latency_ms: 5.0,
        })
        .unwrap();
        api.heartbeat(&HeartbeatRequest {
            node_id: "n2".into(),
            n_vectors: 50,
            cpu_percent: 20.0,
            memory_percent: 30.0,
            qps: 5.0,
            latency_ms: 3.0,
        })
        .unwrap();
        let resp = api.route_query(&RouteQueryRequest {
            query: vec![0.0; 64],
            k: 10,
            strategy: "least_loaded".into(),
        });
        assert_eq!(resp.target_nodes, vec!["n2"]);
    }

    #[test]
    fn test_shard_assignment() {
        let mut api = CoordinatorApi::new();
        api.register_node(&RegisterNodeRequest {
            node_id: "n1".into(),
            role: "worker".into(),
            address: "localhost:8001".into(),
            capabilities: HashMap::new(),
        })
        .unwrap();
        api.assign_shard(&AssignShardRequest {
            shard_id: "shard_0".into(),
            node_id: "n1".into(),
        })
        .unwrap();
        assert_eq!(api.get_shard_assignment("shard_0"), Some("n1"));
    }

    #[test]
    fn test_stale_detection() {
        let mut api = CoordinatorApi::new();
        api.register_node(&RegisterNodeRequest {
            node_id: "n1".into(),
            role: "worker".into(),
            address: "localhost:8001".into(),
            capabilities: HashMap::new(),
        })
        .unwrap();
        // Manually age the heartbeat
        if let Some(node) = api.nodes.get_mut("n1") {
            node.last_heartbeat = now_secs() - 60.0;
        }
        let stale = api.check_stale_nodes(30.0);
        assert_eq!(stale, vec!["n1"]);
        assert!(!api.get_node("n1").unwrap().healthy);
    }
}
