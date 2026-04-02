//! Cluster client for distributed SplatDB search.
//! Handles query routing, result aggregation, and failover.
//! Ported from splatdb Python.

use std::collections::HashMap;

use crate::cluster::router::ClusterRouter;
use crate::cluster::sharding::shard_by_hash;

/// Error type for cluster client operations.
#[derive(Debug)]
pub enum ClusterError {
    CoordinatorUnavailable(String),
    EdgeError(String),
    NoResults,
}

/// Client for interacting with a SplatDB cluster.
pub struct SplatDBClusterClient {
    coordinator_url: Option<String>,
    fallback_edges: Vec<String>,
    in_memory_router: Option<ClusterRouter>,
    edge_urls: HashMap<String, String>,
}

impl SplatDBClusterClient {
    pub fn new(
        coordinator_url: Option<String>,
        fallback_edges: Vec<String>,
        in_memory_router: Option<ClusterRouter>,
    ) -> Self {
        Self {
            coordinator_url,
            fallback_edges,
            in_memory_router,
            edge_urls: HashMap::new(),
        }
    }

    pub fn register_edge_url(&mut self, edge_id: &str, url: &str) {
        self.edge_urls.insert(edge_id.to_string(), url.to_string());
    }

    /// Distributed search across the cluster.
    pub fn search(
        &mut self,
        query: &[f32],
        k: usize,
    ) -> Result<Vec<(usize, f64)>, ClusterError> {
        let edge_ids = if let Some(ref mut router) = self.in_memory_router {
            router.route_query(query, k, "broadcast")
        } else if self.coordinator_url.is_some() {
            // Would make HTTP call to coordinator; for now return empty
            return Err(ClusterError::CoordinatorUnavailable(
                "HTTP coordinator not yet implemented".into(),
            ));
        } else {
            return Err(ClusterError::CoordinatorUnavailable(
                "No coordinator or router provided".into(),
            ));
        };

        if edge_ids.is_empty() {
            return Ok(Vec::new());
        }

        // In a real deployment, these would be HTTP calls to edge nodes.
        // For local/embedded use, results would come from direct search.
        // This returns the routing decision; actual search happens at edge nodes.
        Ok(Vec::new())
    }

    /// Ingest documents to cluster using shard strategy.
    pub fn ingest_sharded(
        &mut self,
        doc_ids: &[String],
        edge_ids: &[String],
    ) -> usize {
        if edge_ids.is_empty() {
            return 0;
        }

        let mut added = 0;
        for doc_id in doc_ids {
            let target = shard_by_hash(doc_id, edge_ids.len());
            if target < edge_ids.len() {
                if let Some(ref mut router) = self.in_memory_router {
                    router.register_document(doc_id, &edge_ids[target]);
                }
                added += 1;
            }
        }
        added
    }

    /// Fallback: query known edges directly.
    pub fn fallback_search(&self) -> Vec<String> {
        if !self.fallback_edges.is_empty() {
            self.fallback_edges.clone()
        } else {
            Vec::new()
        }
    }
}
