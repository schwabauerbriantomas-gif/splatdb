//! Cluster CLI commands — manage distributed SplatsDB nodes.

use splatsdb::cluster::client::SplatsDBClusterClient;
use splatsdb::cluster::energy_router::EnergyRouterConfig;
use splatsdb::cluster::router::ClusterRouter;
use splatsdb::cluster::sharding::shard_by_hash;

// ─── Global cluster state (persisted as JSON for CLI session) ──────

const STATE_FILE: &str = "cluster_state.json";

fn load_cluster_state() -> Option<ClusterState> {
    let path = std::path::Path::new(STATE_FILE);
    if !path.exists() {
        return None;
    }
    let data = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&data).ok()
}

fn save_cluster_state(state: &ClusterState) {
    let path = std::path::Path::new(STATE_FILE);
    if let Ok(data) = serde_json::to_string_pretty(state) {
        let _ = std::fs::write(path, data);
    }
}

#[derive(serde::Serialize, serde::Deserialize, Default)]
struct ClusterState {
    nodes: Vec<EdgeEntry>,
    sharding_strategy: String,
    routing_strategy: String,
    energy_router_enabled: bool,
}

#[derive(serde::Serialize, serde::Deserialize, Clone)]
struct EdgeEntry {
    id: String,
    url: String,
    weight: f64,
    n_vectors: usize,
    role: String,
}

// ─── Commands ─────────────────────────────────────────────────────────

/// Join a node to the cluster (register an edge node).
pub fn cmd_cluster_join(edge_id: &str, url: &str, role: &str, weight: f64) {
    let mut state = load_cluster_state().unwrap_or_default();

    // Check duplicate
    if state.nodes.iter().any(|n| n.id == edge_id) {
        eprintln!("Error: node '{}' already registered", edge_id);
        std::process::exit(1);
    }

    state.nodes.push(EdgeEntry {
        id: edge_id.to_string(),
        url: url.to_string(),
        weight,
        n_vectors: 0,
        role: role.to_string(),
    });

    save_cluster_state(&state);
    println!(
        "✓ Node '{}' joined cluster as {} (url={})",
        edge_id, role, url
    );
    println!("  Cluster size: {} nodes", state.nodes.len());
}

/// Remove a node from the cluster.
pub fn cmd_cluster_leave(edge_id: &str) {
    let mut state = load_cluster_state().unwrap_or_default();

    let before = state.nodes.len();
    state.nodes.retain(|n| n.id != edge_id);

    if state.nodes.len() == before {
        eprintln!("Error: node '{}' not found in cluster", edge_id);
        std::process::exit(1);
    }

    save_cluster_state(&state);
    println!("✓ Node '{}' left cluster", edge_id);
    println!("  Cluster size: {} nodes", state.nodes.len());
}

/// Show cluster status — nodes, routing, shard assignments.
pub fn cmd_cluster_status(verbose: bool) {
    let state = load_cluster_state();

    match state {
        None => {
            println!("No cluster state found. Use 'cluster-join' to add nodes.");
        }
        Some(s) if s.nodes.is_empty() => {
            println!("Cluster is empty. Use 'cluster-join' to add nodes.");
        }
        Some(s) => {
            println!("SplatsDB Cluster Status");
            println!("══════════════════════");
            println!("Nodes: {}", s.nodes.len());
            println!("Sharding: {}", s.sharding_strategy);
            println!("Routing: {}", s.routing_strategy);
            println!(
                "Energy-aware: {}",
                if s.energy_router_enabled { "ON" } else { "OFF" }
            );
            println!();

            let total_vectors: usize = s.nodes.iter().map(|n| n.n_vectors).sum();
            println!("Total vectors: {}", total_vectors);
            println!();

            if verbose {
                println!(
                    "{:<15} {:<25} {:<10} {:<10} {:<8}",
                    "ID", "URL", "Role", "Vectors", "Weight"
                );
                println!("{}", "─".repeat(70));
                for n in &s.nodes {
                    println!(
                        "{:<15} {:<25} {:<10} {:<10} {:<8.1}",
                        n.id, n.url, n.role, n.n_vectors, n.weight
                    );
                }
            } else {
                for n in &s.nodes {
                    println!("  • {} ({}) — {} vectors", n.id, n.role, n.n_vectors);
                }
            }

            // Show shard distribution if there are docs
            println!();
            println!(
                "Shard strategy: {} (hash / cluster / geo)",
                s.sharding_strategy
            );
        }
    }
}

/// Ingest documents to the cluster using sharding.
pub fn cmd_cluster_ingest(input: &str, n_docs: usize, strategy: &str) {
    let mut state = load_cluster_state().unwrap_or_default();

    if state.nodes.is_empty() {
        eprintln!("Error: no nodes in cluster. Use 'cluster-join' first.");
        std::process::exit(1);
    }

    state.sharding_strategy = strategy.to_string();

    let edge_ids: Vec<String> = state.nodes.iter().map(|n| n.id.clone()).collect();
    let mut router = build_router(&state);

    let mut per_edge_counts: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();

    for i in 0..n_docs {
        let doc_id = format!("{}_{}", input, i);
        let target = shard_by_hash(&doc_id, edge_ids.len());
        let target_edge = &edge_ids[target];
        router.register_document(&doc_id, target_edge);
        *per_edge_counts.entry(target_edge.clone()).or_insert(0) += 1;
    }

    // Update node vector counts
    for node in &mut state.nodes {
        if let Some(count) = per_edge_counts.get(&node.id) {
            node.n_vectors += count;
        }
    }

    save_cluster_state(&state);

    println!(
        "✓ Ingested {} documents to {} nodes",
        n_docs,
        edge_ids.len()
    );
    for (edge, count) in &per_edge_counts {
        println!("  • {} → {} docs", edge, count);
    }
}

/// Search the cluster — demonstrates routing + RRF aggregation.
pub fn cmd_cluster_search(query: &str, k: usize, strategy: &str) {
    let state = match load_cluster_state() {
        Some(s) if !s.nodes.is_empty() => s,
        _ => {
            eprintln!("Error: no nodes in cluster. Use 'cluster-join' first.");
            std::process::exit(1);
        }
    };

    {
        let router = build_router(&state);
        let mut client = SplatsDBClusterClient::new_embedded(router);

        let _edge_ids: Vec<String> = state.nodes.iter().map(|n| n.id.clone()).collect();

        // Register mock search functions for demonstration
        // (In production, these would be HTTP calls to edge nodes)
        for node in &state.nodes {
            let node_vectors = node.n_vectors;
            client.register_edge_search_fn(
                &node.id,
                Box::new(move |_q, top_k| {
                    // Simulated local search — returns top_k from this node's data
                    (0..node_vectors.min(top_k))
                        .map(|i| (i, (i as f64 + 1.0) * 0.1))
                        .collect()
                }),
            );
        }

        // Parse query
        let query_vec: Vec<f32> = query
            .split(',')
            .filter_map(|v| v.trim().parse().ok())
            .collect();

        if query_vec.is_empty() {
            eprintln!("Error: query must be comma-separated floats (e.g. '0.1,0.2,0.3')");
            std::process::exit(1);
        }

        match client.search(&query_vec, k, strategy) {
            Ok(results) => {
                if results.is_empty() {
                    println!("No results (edge nodes have no data). Ingest documents first.");
                } else {
                    println!("Cluster search results (strategy={}, k={}):", strategy, k);
                    println!("{}", "─".repeat(40));
                    for (rank, (doc_id, dist)) in results.iter().enumerate() {
                        println!("  #{} doc_id={} distance={:.4}", rank + 1, doc_id, dist);
                    }
                }
            }
            Err(e) => {
                eprintln!("Search error: {:?}", e);
            }
        }

        // Show routing stats
        if let Some(stats) = client.routing_stats() {
            println!();
            println!(
                "Routing: {} online / {} total, {} docs indexed",
                stats.online_nodes, stats.total_nodes, stats.documents_indexed
            );
        }
    }
}

/// Benchmark the cluster — run N searches and report QPS.
pub fn cmd_cluster_bench(n_queries: usize, k: usize, strategy: &str) {
    let state = match load_cluster_state() {
        Some(s) if !s.nodes.is_empty() => s,
        _ => {
            eprintln!("Error: no nodes in cluster. Use 'cluster-join' first.");
            std::process::exit(1);
        }
    };

    let dim = 64;
    let query = vec![0.1f32; dim];

    // Build client with mock search fns
    let router = build_router(&state);
    let mut client = SplatsDBClusterClient::new_embedded(router);

    for node in &state.nodes {
        let nv = node.n_vectors;
        client.register_edge_search_fn(
            &node.id,
            Box::new(move |_q, top_k| {
                (0..nv.min(top_k))
                    .map(|i| (i, (i as f64 + 1.0) * 0.1))
                    .collect()
            }),
        );
    }

    let start = std::time::Instant::now();
    let mut completed = 0usize;

    for _ in 0..n_queries {
        if client.search(&query, k, strategy).is_ok() {
            completed += 1;
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    let qps = if elapsed > 0.0 {
        completed as f64 / elapsed
    } else {
        0.0
    };

    println!("Cluster Benchmark Results");
    println!("═════════════════════════");
    println!("Queries:    {}", n_queries);
    println!("Completed:  {}", completed);
    println!("k:          {}", k);
    println!("Strategy:   {}", strategy);
    println!("Nodes:      {}", state.nodes.len());
    println!("Time:       {:.3}s", elapsed);
    println!("QPS:        {:.1}", qps);
    println!(
        "p50:        {:.3}ms",
        elapsed / completed.max(1) as f64 * 1000.0
    );
}

/// Remove cluster state file.
pub fn cmd_cluster_reset() {
    let path = std::path::Path::new("cluster_state.bin");
    if path.exists() {
        std::fs::remove_file(path).expect("Failed to remove cluster state");
        println!("✓ Cluster state reset");
    } else {
        println!("No cluster state to reset");
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────

fn build_router(state: &ClusterState) -> ClusterRouter {
    let energy_config = if state.energy_router_enabled {
        Some(EnergyRouterConfig {
            enabled: true,
            ..Default::default()
        })
    } else {
        None
    };

    let mut router = ClusterRouter::new(energy_config);
    for node in &state.nodes {
        router.register_edge(&node.id, &node.url, node.weight);
    }
    router
}
