//! Cluster protocol definitions.
//! Ported from splatsdb Python.

use serde::{Deserialize, Serialize};

/// Cluster node role.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeRole {
    Coordinator,
    Worker,
    Edge,
}

/// Node status.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeStatus {
    pub id: String,
    pub role: NodeRole,
    pub address: String,
    pub healthy: bool,
    pub n_vectors: usize,
    pub last_heartbeat: f64,
}

/// Cluster protocol message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ClusterMessage {
    Heartbeat { node_id: String, n_vectors: usize },
    ShardAssign { shard_id: String, node_id: String },
    ShardSync { shard_id: String, version: u64 },
    SearchRequest { query: Vec<f32>, k: usize },
    SearchResponse { results: Vec<(usize, f32)> },
    Join { node_id: String, role: NodeRole },
    Leave { node_id: String },
}
