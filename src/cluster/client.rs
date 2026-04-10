//! Cluster client for distributed SplatsDB search.
//! Handles query routing, result aggregation, and failover.
//! Ported from splatsdb Python.

use std::collections::HashMap;

use crate::cluster::aggregator::ResultAggregator;
use crate::cluster::router::ClusterRouter;
use crate::cluster::sharding::shard_by_hash;

/// Error type for cluster client operations.
#[derive(Debug)]
pub enum ClusterError {
    CoordinatorUnavailable(String),
    EdgeError(String),
    NoResults,
}

/// Type alias for local search function: takes (query, k) → Vec<(doc_id, distance)>
pub type LocalSearchFn = Box<dyn FnMut(&[f32], usize) -> Vec<(usize, f64)>>;

/// Client for interacting with a SplatsDB cluster.
pub struct SplatsDBClusterClient {
    coordinator_url: Option<String>,
    fallback_edges: Vec<String>,
    in_memory_router: Option<ClusterRouter>,
    edge_urls: HashMap<String, String>,
    /// Per-edge local search functions (for embedded mode)
    edge_search_fns: HashMap<String, LocalSearchFn>,
    aggregator: ResultAggregator,
}

impl SplatsDBClusterClient {
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
            edge_search_fns: HashMap::new(),
            aggregator: ResultAggregator::new(60),
        }
    }

    /// Create a client for embedded (in-process) cluster mode.
    pub fn new_embedded(router: ClusterRouter) -> Self {
        Self {
            coordinator_url: None,
            fallback_edges: Vec::new(),
            in_memory_router: Some(router),
            edge_urls: HashMap::new(),
            edge_search_fns: HashMap::new(),
            aggregator: ResultAggregator::new(60),
        }
    }

    pub fn register_edge_url(&mut self, edge_id: &str, url: &str) {
        self.edge_urls.insert(edge_id.to_string(), url.to_string());
    }

    /// Register a local search function for an edge node (embedded mode).
    pub fn register_edge_search_fn(&mut self, edge_id: &str, search_fn: LocalSearchFn) {
        self.edge_search_fns.insert(edge_id.to_string(), search_fn);
    }

    /// Distributed search across the cluster.
    /// Routes query to appropriate edge nodes, runs local searches, merges via RRF.
    pub fn search(
        &mut self,
        query: &[f32],
        k: usize,
        strategy: &str,
    ) -> Result<Vec<(usize, f64)>, ClusterError> {
        let edge_ids = if let Some(ref mut router) = self.in_memory_router {
            router.route_query(query, k, strategy)
        } else if self.coordinator_url.is_some() {
            return Err(ClusterError::CoordinatorUnavailable(
                "HTTP coordinator not yet implemented".into(),
            ));
        } else {
            return Err(ClusterError::CoordinatorUnavailable(
                "No coordinator or router provided".into(),
            ));
        };

        if edge_ids.is_empty() {
            return Err(ClusterError::NoResults);
        }

        // Collect results from each edge node
        let mut all_results: HashMap<String, Vec<(usize, f64)>> = HashMap::new();

        for edge_id in &edge_ids {
            if let Some(search_fn) = self.edge_search_fns.get_mut(edge_id) {
                let results = search_fn(query, k);
                if !results.is_empty() {
                    all_results.insert(edge_id.clone(), results);
                }
            }
            // For HTTP mode (future): would make HTTP call to edge_url here
        }

        if all_results.is_empty() {
            // No local search fns registered — return routing decision only
            return Ok(Vec::new());
        }

        // Merge results from all edges using RRF
        let merged = self.aggregator.merge_results(&all_results, k, "rrf");
        Ok(merged)
    }

    /// Ingest documents to cluster using shard strategy.
    pub fn ingest_sharded(&mut self, doc_ids: &[String], edge_ids: &[String]) -> usize {
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

    /// Get the list of online edges from the router.
    pub fn online_edges(&self) -> Vec<String> {
        match &self.in_memory_router {
            Some(router) => router.get_online_edges(),
            None => Vec::new(),
        }
    }

    /// Get routing stats from the router.
    pub fn routing_stats(&self) -> Option<crate::cluster::router::RouterStats> {
        self.in_memory_router
            .as_ref()
            .map(|r| r.get_routing_stats())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cluster::router::ClusterRouter;

    #[test]
    fn test_search_with_local_fns() {
        let mut router = ClusterRouter::new(None);
        router.register_edge("e1", "localhost:8001", 1.0);
        router.register_edge("e2", "localhost:8002", 1.0);

        let mut client = SplatsDBClusterClient::new_embedded(router);

        // Register mock search functions
        client.register_edge_search_fn(
            "e1",
            Box::new(|_q, k| {
                vec![(1, 0.5), (2, 0.8), (3, 1.0)]
                    .into_iter()
                    .take(k)
                    .collect()
            }),
        );
        client.register_edge_search_fn(
            "e2",
            Box::new(|_q, k| vec![(2, 0.55), (4, 0.9)].into_iter().take(k).collect()),
        );

        let results = client.search(&[0.1, 0.2, 0.3], 5, "broadcast").unwrap();
        assert!(!results.is_empty());
        // Doc 2 appears in both edges → should rank high via RRF
        assert!(results.iter().any(|(id, _)| *id == 2));
    }

    #[test]
    fn test_ingest_sharded_distributes() {
        let mut router = ClusterRouter::new(None);
        router.register_edge("e1", "localhost:8001", 1.0);
        router.register_edge("e2", "localhost:8002", 1.0);

        let mut client = SplatsDBClusterClient::new_embedded(router);
        let edges = vec!["e1".to_string(), "e2".to_string()];

        let docs: Vec<String> = (0..100).map(|i| format!("doc_{}", i)).collect();
        let added = client.ingest_sharded(&docs, &edges);
        assert_eq!(added, 100);
    }

    #[test]
    fn test_no_router_returns_error() {
        let mut client = SplatsDBClusterClient::new(None, Vec::new(), None);
        let result = client.search(&[0.1], 10, "broadcast");
        assert!(result.is_err());
    }

    /// End-to-end integration test: register edges → ingest → route → search → aggregate
    #[test]
    fn test_cluster_e2e_workflow() {
        use crate::cluster::aggregator::ResultAggregator;
        use crate::cluster::sharding::shard_by_hash;

        // 1. Setup: 3 edge nodes with 100 docs each
        let mut router = ClusterRouter::new(None);
        router.register_edge("edge-us", "us.example.com:8001", 1.0);
        router.register_edge("edge-eu", "eu.example.com:8002", 1.0);
        router.register_edge("edge-asia", "asia.example.com:8003", 0.8);

        let edge_ids = vec![
            "edge-us".to_string(),
            "edge-eu".to_string(),
            "edge-asia".to_string(),
        ];

        // 2. Ingest: shard 500 docs across edges
        let mut per_edge_docs: std::collections::HashMap<String, Vec<usize>> =
            std::collections::HashMap::new();
        for i in 0..500 {
            let doc_id = format!("doc_{}", i);
            let shard = shard_by_hash(&doc_id, edge_ids.len());
            let target = &edge_ids[shard];
            router.register_document(&doc_id, target);
            per_edge_docs.entry(target.clone()).or_default().push(i);
        }

        // Verify all docs distributed
        let total: usize = per_edge_docs.values().map(|v| v.len()).sum();
        assert_eq!(total, 500);
        // Each edge should have some docs (hash distribution)
        for edge in &edge_ids {
            assert!(
                per_edge_docs.contains_key(edge),
                "Edge {} has no docs",
                edge
            );
        }

        // 3. Build client with mock search functions simulating per-edge data
        let mut client = SplatsDBClusterClient::new_embedded(router);

        for edge_id in &edge_ids {
            let docs = per_edge_docs.get(edge_id).cloned().unwrap_or_default();
            client.register_edge_search_fn(
                edge_id,
                Box::new(move |_q, top_k| {
                    docs.iter()
                        .take(top_k)
                        .enumerate()
                        .map(|(rank, doc_id)| (*doc_id, (rank as f64 + 1.0) * 0.1))
                        .collect()
                }),
            );
        }

        // 4. Search with broadcast strategy (hits all edges)
        let query = vec![0.5f32; 64];
        let results = client
            .search(&query, 10, "broadcast")
            .expect("search should succeed with registered edges");

        assert!(
            !results.is_empty(),
            "Should get results from cluster search"
        );
        assert!(results.len() <= 10, "Should not return more than k results");

        // 5. Verify routing stats
        let stats = client.routing_stats().expect("should have stats");
        assert_eq!(stats.total_nodes, 3);
        assert!(stats.documents_indexed > 0);

        // 6. Test round-robin strategy
        let results_rr = client
            .search(&query, 5, "round_robin")
            .expect("round_robin should work");
        assert!(!results_rr.is_empty());

        // 7. Test ingest_sharded
        let new_docs: Vec<String> = (500..600).map(|i| format!("doc_{}", i)).collect();
        let added = client.ingest_sharded(&new_docs, &edge_ids);
        assert_eq!(added, 100);

        // 8. Verify online edges
        let online = client.online_edges();
        assert_eq!(online.len(), 3);
    }
}
