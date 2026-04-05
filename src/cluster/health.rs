//! Cluster health monitoring and types.
//! Ported from splatdb Python.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Load metrics for a node.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LoadMetrics {
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub active_queries: usize,
    pub query_latency_ms: f64,
}

/// Geographic location of an edge node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeoLocation {
    pub latitude: f64,
    pub longitude: f64,
}

/// Information about an edge node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeNodeInfo {
    pub edge_id: String,
    pub url: String,
    pub status: String, // "online", "offline", "degraded"
    pub last_heartbeat: f64,
    pub document_count: usize,
    pub location: Option<GeoLocation>,
    pub capabilities: HashMap<String, bool>,
}

impl EdgeNodeInfo {
    pub fn new(edge_id: &str, url: &str) -> Self {
        Self {
            edge_id: edge_id.to_string(),
            url: url.to_string(),
            status: "online".to_string(),
            last_heartbeat: now_secs(),
            document_count: 0,
            location: None,
            capabilities: HashMap::new(),
        }
    }
}

/// Health status of a cluster node.
#[derive(Debug, Clone, Serialize)]
pub struct HealthStatus {
    pub node_id: String,
    pub healthy: bool,
    pub n_vectors: usize,
    pub last_check: f64,
}

/// Health checker for cluster nodes.
pub struct HealthChecker {
    statuses: Vec<HealthStatus>,
    timeout_secs: f64,
}

impl HealthChecker {
    pub fn new(timeout_secs: f64) -> Self {
        Self {
            statuses: Vec::new(),
            timeout_secs,
        }
    }

    pub fn update(&mut self, node_id: &str, healthy: bool, n_vectors: usize) {
        self.statuses.retain(|s| s.node_id != node_id);
        self.statuses.push(HealthStatus {
            node_id: node_id.to_string(),
            healthy,
            n_vectors,
            last_check: now_secs(),
        });
    }

    pub fn healthy_nodes(&self) -> Vec<&HealthStatus> {
        self.statuses.iter().filter(|s| s.healthy).collect()
    }

    pub fn all_healthy(&self) -> bool {
        self.statuses.iter().all(|s| s.healthy)
    }

    /// Check for stale nodes and mark them unhealthy.
    pub fn check_timeouts(&mut self) -> Vec<String> {
        let now = now_secs();
        let mut timed_out = Vec::new();
        for s in &mut self.statuses {
            if s.healthy && now - s.last_check > self.timeout_secs {
                s.healthy = false;
                timed_out.push(s.node_id.clone());
            }
        }
        timed_out
    }
}

fn now_secs() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}
