//! Graph Splat CLI command handlers.
//!
//! Wraps the `graph_splat::GaussianGraphStore` operations for CLI use.

use splatsdb::graph_splat::GaussianGraphStore;

use super::helpers::*;

/// Graph-Add-Doc: add a document node to the graph.
///
/// Embedding is provided as a comma-separated f32 vector.
pub fn cmd_graph_add_doc(text: String, embedding: String) {
    let emb = match parse_query(&embedding) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Error parsing embedding: {}", e);
            std::process::exit(1);
        }
    };

    let emb_vec: Vec<f32> = emb.to_vec();
    let mut store = GaussianGraphStore::new();
    match store.add_document(&text, &emb_vec) {
        Ok(id) => {
            println!(
                "{}",
                serde_json::json!({
                    "status": "ok",
                    "node_id": id,
                    "node_type": "document",
                })
            );
        }
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    }
}

/// Graph-Add-Entity: add an entity node to the graph.
pub fn cmd_graph_add_entity(name: String, embedding: String, entity_type: String) {
    let emb = match parse_query(&embedding) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Error parsing embedding: {}", e);
            std::process::exit(1);
        }
    };

    let emb_vec: Vec<f32> = emb.to_vec();
    let mut store = GaussianGraphStore::new();
    match store.add_entity(&name, &emb_vec, &entity_type) {
        Ok(id) => {
            println!(
                "{}",
                serde_json::json!({
                    "status": "ok",
                    "node_id": id,
                    "node_type": "entity",
                })
            );
        }
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    }
}

/// Graph-Add-Relation: add a directed edge between two nodes.
pub fn cmd_graph_add_relation(
    source_id: usize,
    target_id: usize,
    relation_type: String,
    weight: f64,
) {
    let mut store = GaussianGraphStore::new();
    match store.add_relation(source_id, target_id, &relation_type, weight) {
        Ok(()) => {
            println!(
                "{}",
                serde_json::json!({
                    "status": "ok",
                    "source_id": source_id,
                    "target_id": target_id,
                    "relation_type": relation_type,
                    "weight": weight,
                })
            );
        }
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    }
}

/// Graph-Traverse: BFS traversal from a start node.
pub fn cmd_graph_traverse(text: String, embedding: String, max_depth: usize, add_doc: bool) {
    let mut store = GaussianGraphStore::new();

    // If --add-doc is set, first insert a document node then traverse from it.
    let start_id = if add_doc {
        let emb = match parse_query(&embedding) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("Error parsing embedding: {}", e);
                std::process::exit(1);
            }
        };
        let emb_vec: Vec<f32> = emb.to_vec();
        match store.add_document(&text, &emb_vec) {
            Ok(id) => id,
            Err(e) => {
                eprintln!("Error adding document: {}", e);
                std::process::exit(1);
            }
        }
    } else {
        // Parse start_id from text
        match text.parse::<usize>() {
            Ok(id) => id,
            Err(_) => {
                eprintln!("Error: provide a numeric node ID or use --add-doc");
                std::process::exit(1);
            }
        }
    };

    let result = store.traverse(start_id, max_depth);
    println!(
        "{}",
        serde_json::json!({
            "start_id": start_id,
            "max_depth": max_depth,
            "visited": result,
            "n_visited": result.len(),
        })
    );
}

/// Graph-Search: hybrid vector + graph search.
pub fn cmd_graph_search(query: String, k: usize, search_type: String) {
    let emb = match parse_query(&query) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Error parsing query embedding: {}", e);
            std::process::exit(1);
        }
    };

    let emb_vec: Vec<f32> = emb.to_vec();

    let mut store = GaussianGraphStore::new();

    // Add a demo document so search has something to return.
    // In real usage the store would be pre-loaded.
    let _ = store.add_document("demo", &emb_vec);

    match search_type.as_str() {
        "hybrid" => {
            let results = store.hybrid_search(&emb_vec, k);
            let entries: Vec<serde_json::Value> = results
                .iter()
                .map(|r| {
                    serde_json::json!({
                        "node_id": r.node_id,
                        "score": (r.score * 10000.0).round() / 10000.0,
                        "content": r.content,
                    })
                })
                .collect();
            println!(
                "{}",
                serde_json::json!({
                    "search_type": "hybrid",
                    "n_results": entries.len(),
                    "results": entries,
                })
            );
        }
        "entities" => {
            let results = store.search_entities(&emb_vec, k);
            let entries: Vec<serde_json::Value> = results
                .iter()
                .map(|n| {
                    serde_json::json!({
                        "node_id": n.id,
                        "content": n.content,
                    })
                })
                .collect();
            println!(
                "{}",
                serde_json::json!({
                    "search_type": "entities",
                    "n_results": entries.len(),
                    "results": entries,
                })
            );
        }
        _ => {
            eprintln!(
                "Unknown search type '{}'. Use 'hybrid' or 'entities'.",
                search_type
            );
            std::process::exit(1);
        }
    }
}

/// Graph-Stats: print graph statistics.
pub fn cmd_graph_stats() {
    let store = GaussianGraphStore::new();
    let stats = store.get_stats();
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "total_nodes": stats.total_nodes,
            "total_edges": stats.total_edges,
            "documents": stats.documents,
            "entities": stats.entities,
            "concepts": stats.concepts,
        }))
        .expect("Failed to serialize graph stats")
    );
}
