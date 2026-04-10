//! Edge node: local SplatsDB instance that can operate offline.
//! Ported from splatsdb Python.

use crate::cluster::health::LoadMetrics;
use crate::cluster::sync::SyncQueue;

/// An SplatsDB edge node with local storage and optional coordinator sync.
pub struct EdgeNode {
    pub edge_id: String,
    coordinator_url: Option<String>,
    sync_queue: SyncQueue,
    n_vectors: usize,
    active_queries: usize,
    total_queries: usize,
    total_search_time_ms: f64,
}

impl EdgeNode {
    pub fn new(edge_id: &str, coordinator_url: Option<String>) -> Self {
        Self {
            edge_id: edge_id.to_string(),
            coordinator_url,
            sync_queue: SyncQueue::new(5.0),
            n_vectors: 0,
            active_queries: 0,
            total_queries: 0,
            total_search_time_ms: 0.0,
        }
    }

    /// Ingest vectors locally.
    pub fn ingest(&mut self, n_new: usize, doc_ids: &[&str]) -> usize {
        self.n_vectors += n_new;

        if self.coordinator_url.is_some() {
            for doc_id in doc_ids {
                self.sync_queue.add_action(
                    "register",
                    &format!(r#"{{"doc_id":"{}","edge_id":"{}"}}"#, doc_id, self.edge_id),
                );
            }
        }
        n_new
    }

    /// Record a search operation.
    pub fn record_search(&mut self, duration_ms: f64) {
        self.total_queries += 1;
        self.total_search_time_ms += duration_ms;
    }

    /// Get current performance metrics.
    pub fn get_metrics(&self) -> LoadMetrics {
        let avg_latency = if self.total_queries > 0 {
            self.total_search_time_ms / self.total_queries as f64
        } else {
            0.0
        };
        LoadMetrics {
            active_queries: self.active_queries,
            query_latency_ms: avg_latency,
            ..Default::default()
        }
    }

    pub fn n_vectors(&self) -> usize {
        self.n_vectors
    }

    pub fn sync_queue(&mut self) -> &mut SyncQueue {
        &mut self.sync_queue
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ingest_and_metrics() {
        let mut node = EdgeNode::new("e1", None);
        node.ingest(100, &["d1", "d2"]);
        assert_eq!(node.n_vectors(), 100);
        node.record_search(5.0);
        let m = node.get_metrics();
        assert_eq!(m.query_latency_ms, 5.0);
    }
}
