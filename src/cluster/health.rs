//! Cluster health monitoring and types.
//! Ported from splatsdb Python.

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

fn now_secs() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}
