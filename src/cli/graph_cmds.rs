//! Graph Splat CLI command handlers.
//!
//! Wraps the `graph_splat::GaussianGraphStore` operations for CLI use.
//!
//! Persistencia: el grafo vive en `graph_state.json` (CWD, misma convención
//! que `cluster_state.json` — el dispatch no pasa data_dir a los subcomandos
//! graph). Cada subcomando carga el estado al inicio; los mutadores guardan
//! al final, así el node_id reportado sobrevive al proceso y
//! graph-add-relation encuentra los nodos de sesiones anteriores.

use splatsdb::graph_splat::{GaussianGraphStore, GRAPH_STATE_FILENAME};
use std::path::{Path, PathBuf};

use super::helpers::*;

/// Dir de estado del grafo (CWD).
const GRAPH_STATE_DIR: &str = ".";

fn graph_state_path() -> PathBuf {
    Path::new(GRAPH_STATE_DIR).join(GRAPH_STATE_FILENAME)
}

/// Cargar el grafo persistido; store vacío si todavía no existe estado.
fn load_graph() -> GaussianGraphStore {
    if !graph_state_path().exists() {
        return GaussianGraphStore::new();
    }
    match GaussianGraphStore::load_from_dir(GRAPH_STATE_DIR) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error loading graph state: {}", e);
            std::process::exit(1);
        }
    }
}

/// Persistir el grafo (falla ruidosamente: perder mutaciones en silencio no).
fn save_graph(store: &GaussianGraphStore) {
    if let Err(e) = store.save_to_dir(GRAPH_STATE_DIR) {
        eprintln!("Error saving graph state: {}", e);
        std::process::exit(1);
    }
}

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
    let mut store = load_graph();
    match store.add_document(&text, &emb_vec) {
        Ok(id) => {
            save_graph(&store);
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
    let mut store = load_graph();
    match store.add_entity(&name, &emb_vec, &entity_type) {
        Ok(id) => {
            save_graph(&store);
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
    let mut store = load_graph();
    match store.add_relation(source_id, target_id, &relation_type, weight) {
        Ok(()) => {
            save_graph(&store);
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
    let mut store = load_graph();

    // If --add-doc is set, first insert a document node (persisted) then
    // traverse from it.
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
            Ok(id) => {
                save_graph(&store);
                id
            }
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

/// Graph-Search: hybrid vector + graph search over the persisted graph.
pub fn cmd_graph_search(query: String, k: usize, search_type: String) {
    let emb = match parse_query(&query) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Error parsing query embedding: {}", e);
            std::process::exit(1);
        }
    };

    let emb_vec: Vec<f32> = emb.to_vec();
    let store = load_graph();

    if store.n_nodes() == 0 {
        // Sin grafo: resultados vacíos honestos (antes se inyectaba un doc
        // demo falso que "encontraba" siempre lo buscado).
        println!(
            "{}",
            serde_json::json!({
                "search_type": search_type,
                "n_results": 0,
                "results": [],
                "message": "graph is empty — add nodes with graph-add-doc / graph-add-entity",
            })
        );
        return;
    }

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

/// Graph-Stats: print real graph statistics from the persisted state.
pub fn cmd_graph_stats() {
    let store = load_graph();
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
