//! MCP (Model Context Protocol) Server — stdio transport
//!
//! Exposes SplatsDB Vector Search as an MCP server for AI agent integration.
//! Uses JSON-RPC 2.0 over stdin/stdout (stdio transport).
//! All logs go to stderr to keep the protocol channel clean.
//!
//! v2.5: Graph Splat tools — knowledge graph over Gaussian splats.

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::{self, BufRead, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use crate::config::SplatsDBConfig;
use crate::splats::SplatStore;
use crate::storage::metadata_store::{DocumentRecord, MetadataStore};
use crate::storage::sqlite_store::SqliteMetadataStore;

// ============================================================================
// Embedding client — real model via HTTP, SimCos fallback
// ============================================================================

const EMBEDDING_SERVICE_URL_DEFAULT: &str = "http://127.0.0.1:8788/embed";
static EMBED_SERVICE_AVAILABLE: AtomicBool = AtomicBool::new(true);
static SHUTDOWN_REQUESTED: AtomicBool = AtomicBool::new(false);
const MAX_DOC_TEXT_CACHE: usize = 10_000;
const MAX_WARM_START_DOCS: usize = 50_000;

fn embedding_service_url() -> String {
    std::env::var("SPLATSDB_EMBED_URL").unwrap_or_else(|_| EMBEDDING_SERVICE_URL_DEFAULT.to_string())
}

#[cfg(unix)]
extern "C" fn handle_sigint(_: i32) {
    SHUTDOWN_REQUESTED.store(true, Ordering::Relaxed);
}

fn get_embedding(text: &str, dim: usize) -> Vec<f32> {
    if EMBED_SERVICE_AVAILABLE.load(Ordering::Relaxed) {
        if let Some(vec) = fetch_real_embedding(text, dim) {
            return vec;
        }
    }
    simcos_embedding(text, dim)
}

fn fetch_real_embedding(text: &str, target_dim: usize) -> Option<Vec<f32>> {
    use std::time::Duration;
    let body = serde_json::json!({ "texts": [text] }).to_string();
    let response = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(5)))
        .build()
        .new_agent()
        .post(&embedding_service_url())
        .header("Content-Type", "application/json")
        .send(&body)
        .ok()?;
    let resp: serde_json::Value = response.into_body().read_json().ok()?;
    let embeddings = resp.get("embeddings")?.as_array()?.first()?.as_array()?;
    let mut vec: Vec<f32> = embeddings
        .iter()
        .filter_map(|v: &Value| v.as_f64().map(|f| f as f32))
        .collect();
    if vec.len() < target_dim {
        vec.resize(target_dim, 0.0f32);
    } else {
        vec.truncate(target_dim);
    }
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-8 {
        for x in vec.iter_mut() {
            *x /= norm;
        }
    }
    Some(vec)
}

// ============================================================================
// Types
// ============================================================================

struct McpState {
    store: SplatStore,
    doc_store: SqliteMetadataStore,
    /// Maps vector index → doc_id for retrieval
    vector_to_doc: HashMap<usize, String>,
    /// Maps doc_id → text content (in-memory cache with eviction)
    doc_texts: HashMap<String, String>,
    /// Ordered insertion keys for LRU eviction
    doc_text_order: Vec<String>,
    next_id: usize,
    /// Knowledge graph store (Gaussian splat graph)
    graph: crate::graph_splat::GaussianGraphStore,
    /// Spatial memory: maps (wing, room, hall) → vector indices for pre-filter search.
    /// Updated when docs are stored/loaded with spatial metadata.
    spatial_index: crate::spatial::SpatialIndex,
}

impl McpState {
    /// Insert text into cache, evicting oldest entries if over limit.
    fn cache_doc_text(&mut self, id: String, text: String) {
        use std::collections::hash_map::Entry;
        match self.doc_texts.entry(id.clone()) {
            Entry::Occupied(mut e) => {
                e.insert(text);
                return;
            }
            Entry::Vacant(_) => {}
        }
        // Evict oldest entries if at capacity
        while self.doc_texts.len() >= MAX_DOC_TEXT_CACHE && !self.doc_text_order.is_empty() {
            if let Some(old_id) = self.doc_text_order.first().cloned() {
                self.doc_texts.remove(&old_id);
                self.doc_text_order.remove(0);
            }
        }
        self.doc_text_order.push(id.clone());
        self.doc_texts.insert(id, text);
    }

    /// Remove text from cache.
    fn uncache_doc_text(&mut self, id: &str) {
        self.doc_texts.remove(id);
        self.doc_text_order.retain(|k| k != id);
    }
}

#[derive(Deserialize)]
#[allow(dead_code)]
struct JsonRpcRequest {
    jsonrpc: String,
    id: Option<Value>,
    method: String,
    params: Option<Value>,
}

#[derive(Serialize)]
struct JsonRpcResponse {
    jsonrpc: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<JsonRpcError>,
}

#[derive(Serialize)]
struct JsonRpcError {
    code: i32,
    message: String,
}

// ============================================================================
// Tool definitions
// ============================================================================

fn tool_definitions() -> Vec<Value> {
    vec![
        json!({
            "name": "splatsdb_store",
            "description": "Store a memory in the SplatsDB vector search engine. Returns the memory ID.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "text": { "type": "string", "description": "The text content to store" },
                    "category": { "type": "string", "description": "Optional category tag" },
                    "id": { "type": "string", "description": "Optional custom ID (auto-generated if omitted)" },
                    "embedding": { "type": "array", "items": { "type": "number" }, "description": "Optional pre-computed embedding vector" },
                    "wing": { "type": "string", "description": "Spatial: top-level scope (project, persona, domain)" },
                    "room": { "type": "string", "description": "Spatial: semantic grouping within wing" },
                    "hall": { "type": "string", "description": "Spatial: memory type filter (fact, decision, event, error)" }
                },
                "required": ["text"]
            }
        }),
        json!({
            "name": "splatsdb_search",
            "description": "Search for similar memories in the SplatsDB vector store. Returns ranked results with similarity scores.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "Search query text" },
                    "top_k": { "type": "number", "description": "Number of results to return (default: 10)" },
                    "embedding": { "type": "array", "items": { "type": "number" }, "description": "Optional pre-computed query embedding" }
                },
                "required": ["query"]
            }
        }),
        json!({
            "name": "splatsdb_status",
            "description": "Get the current status of the SplatsDB vector store (number of memories, dimensions, active indexes).",
            "inputSchema": {
                "type": "object",
                "properties": {}
            }
        }),
        json!({
            "name": "splatsdb_doc_add",
            "description": "Add a document with metadata to the SplatsDB store. Persists to SQLite.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "id": { "type": "string", "description": "Document ID" },
                    "text": { "type": "string", "description": "Document text content" },
                    "metadata": { "type": "string", "description": "Optional JSON metadata string" }
                },
                "required": ["id", "text"]
            }
        }),
        json!({
            "name": "splatsdb_doc_get",
            "description": "Retrieve a document by ID from the SplatsDB store (SQLite-backed).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "id": { "type": "string", "description": "Document ID to retrieve" }
                },
                "required": ["id"]
            }
        }),
        json!({
            "name": "splatsdb_doc_del",
            "description": "Delete a document from the SplatsDB store (SQLite-backed, soft delete).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "id": { "type": "string", "description": "Document ID to delete" }
                },
                "required": ["id"]
            }
        }),
        // ── Graph Splat tools ──────────────────────────────────────────────
        json!({
            "name": "splatsdb_graph_add_doc",
            "description": "Add a document node to the knowledge graph. Auto-embeds the text.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "text": { "type": "string", "description": "Document text content" }
                },
                "required": ["text"]
            }
        }),
        json!({
            "name": "splatsdb_graph_add_entity",
            "description": "Add an entity node to the knowledge graph. Auto-embeds the name.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "name": { "type": "string", "description": "Entity name" },
                    "entity_type": { "type": "string", "description": "Entity type (e.g. person, organization, location, concept)" }
                },
                "required": ["name", "entity_type"]
            }
        }),
        json!({
            "name": "splatsdb_graph_add_relation",
            "description": "Add a directed edge between two nodes in the knowledge graph.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "source_id": { "type": "number", "description": "Source node ID" },
                    "target_id": { "type": "number", "description": "Target node ID" },
                    "relation_type": { "type": "string", "description": "Relation type (e.g. MENTIONS, RELATED_TO)" },
                    "weight": { "type": "number", "description": "Edge weight (default: 1.0)" }
                },
                "required": ["source_id", "target_id", "relation_type"]
            }
        }),
        json!({
            "name": "splatsdb_graph_traverse",
            "description": "BFS traversal from a start node in the knowledge graph.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "start_id": { "type": "number", "description": "Start node ID" },
                    "max_depth": { "type": "number", "description": "Maximum traversal depth (default: 3)" }
                },
                "required": ["start_id"]
            }
        }),
        json!({
            "name": "splatsdb_graph_search",
            "description": "Hybrid search over the knowledge graph (vector similarity + graph context boost). Auto-embeds query.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "Search query text" },
                    "k": { "type": "number", "description": "Number of results (default: 10)" }
                },
                "required": ["query"]
            }
        }),
        json!({
            "name": "splatsdb_graph_search_entities",
            "description": "Search entity nodes by embedding similarity. Auto-embeds query.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "Search query text" },
                    "k": { "type": "number", "description": "Number of results (default: 10)" }
                },
                "required": ["query"]
            }
        }),
        json!({
            "name": "splatsdb_graph_stats",
            "description": "Get knowledge graph statistics (node counts, edge counts, etc.).",
            "inputSchema": {
                "type": "object",
                "properties": {}
            }
        }),
        // ── Spatial Memory tools ──
        serde_json::json!({
            "name": "splatsdb_spatial_search",
            "description": "Search memories with spatial filters. Organize memory like physical space: Wings (projects), Rooms (topics), Halls (types). Filters reduce the search space before vector distance computation for higher recall with less noise.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query text"
                    },
                    "wing": {
                        "type": "string",
                        "description": "Filter by wing (project/persona/domain). E.g. 'project-x', 'personal'"
                    },
                    "room": {
                        "type": "string",
                        "description": "Filter by room (semantic cluster/topic). E.g. 'auth', 'billing', 'migration'"
                    },
                    "hall": {
                        "type": "string",
                        "description": "Filter by hall (memory type). E.g. 'decision', 'fact', 'error', 'event'"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results (default: 10)"
                    }
                },
                "required": ["query"]
            }
        }),
        serde_json::json!({
            "name": "splatsdb_spatial_info",
            "description": "Show the spatial memory structure: all Wings (projects), Rooms (topics), Halls (memory types), and Tunnels (cross-wing connections). Use this to understand the organization before querying with splatsdb_spatial_search.",
            "inputSchema": {
                "type": "object",
                "properties": {}
            }
        }),
        // ── Verbatim Storage tools ──────────────────────────────────────
        json!({
            "name": "splatsdb_verbatim_search",
            "description": "Search with confidence scoring to prevent hallucination. Returns results with HIGH/MEDIUM/LOW confidence based on vector distance, plus original verbatim text for verification.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "Search query text" },
                    "top_k": { "type": "number", "description": "Number of results (default: 10)" }
                },
                "required": ["query"]
            }
        }),
        json!({
            "name": "splatsdb_compress",
            "description": "Compress text using AAAK (semantic + binary compression). Returns semantically compressed text that LLMs can read natively, plus binary compressed data for storage.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "text": { "type": "string", "description": "Text to compress" }
                },
                "required": ["text"]
            }
        }),
        json!({
            "name": "splatsdb_decompress",
            "description": "Decompress AAAK binary data back to LLM-readable semantic text.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "data": { "type": "string", "description": "Hex-encoded compressed data" }
                },
                "required": ["data"]
            }
        }),
    ]
}

// ============================================================================
// Embedding: SimCos — similarity-consistent n-gram hashing
// ============================================================================

/// Produce a similarity-consistent embedding from text.
///
/// Unlike simple hash (where "hello world" and "hello earth" have 0 similarity),
/// SimCos uses character n-gram overlap to produce embeddings where:
/// - Similar texts have high cosine similarity
/// - Different texts have low similarity
/// - It's fast and requires no model loading
///
/// For production use, pass pre-computed embeddings via the `embedding` parameter.
fn simcos_embedding(text: &str, dim: usize) -> Vec<f32> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    // Generate character trigrams from the text
    let text_lower = text.to_lowercase();
    let chars: Vec<char> = text_lower.chars().collect();
    let mut trigram_set: Vec<String> = Vec::new();

    // Character trigrams
    if chars.len() >= 3 {
        for i in 0..chars.len() - 2 {
            let tg: String = chars[i..i + 3].iter().collect();
            trigram_set.push(tg);
        }
    } else {
        // Short text: use the whole thing
        trigram_set.push(text_lower.clone());
    }

    // Word unigrams and bigrams
    let words: Vec<&str> = text_lower.split_whitespace().collect();
    for w in &words {
        trigram_set.push(format!("w:{}", w));
    }
    if words.len() >= 2 {
        for i in 0..words.len() - 1 {
            trigram_set.push(format!("wb:{}:{}", words[i], words[i + 1]));
        }
    }

    // Hash trigrams into the embedding vector using random projection
    let mut result = vec![0.0f32; dim];
    for tg in &trigram_set {
        // Each trigram contributes to multiple dimensions
        for band in 0..3 {
            let mut hasher = DefaultHasher::new();
            tg.hash(&mut hasher);
            band.hash(&mut hasher);
            let hash_val = hasher.finish();

            // Map hash to dimension index
            let idx = (hash_val as usize) % dim;
            // Use sign from a second hash
            let mut hasher2 = DefaultHasher::new();
            tg.hash(&mut hasher2);
            band.hash(&mut hasher2);
            (idx as u64).hash(&mut hasher2);
            let sign_hash = hasher2.finish();

            let sign = if sign_hash.is_multiple_of(2) {
                1.0f32
            } else {
                -1.0f32
            };
            result[idx] += sign;
        }
    }

    // L2 normalize
    let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in result.iter_mut() {
            *x /= norm;
        }
    }

    result
}

// ============================================================================
// Tool handlers
// ============================================================================

fn handle_store(state: &Mutex<McpState>, params: &Value) -> Result<Value, String> {
    let text = params["text"].as_str().ok_or("missing 'text' field")?;
    const MAX_TEXT_LEN: usize = 100_000;
    if text.len() > MAX_TEXT_LEN {
        return Err(format!("Text exceeds max length ({MAX_TEXT_LEN})"));
    }
    let category = params["category"].as_str();
    let embedding_opt = params["embedding"].as_array().map(|arr| {
        arr.iter()
            .filter_map(|v| v.as_f64().map(|f| f as f32))
            .collect::<Vec<_>>()
    });

    let mut s = state.lock().map_err(|e| {
        eprintln!("[mcp] mutex poisoned: {}", e);
        "internal server error".to_string()
    })?;
    let dim = s.store.get_statistics().embedding_dim;

    // Use provided embedding, or get real embedding (MiniLM or SimCos fallback)
    let embedding = embedding_opt.unwrap_or_else(|| get_embedding(text, dim));

    if embedding.len() != dim {
        return Err(format!(
            "Embedding dimension mismatch: expected {dim}, got {}",
            embedding.len()
        ));
    }

    let arr = Array2::from_shape_vec((1, dim), embedding)
        .map_err(|e| format!("bad embedding shape: {}", e))?;

    let vector_idx = s.store.n_active();
    let added = s.store.add_splat(&arr);
    if !added {
        return Err("store is full".into());
    }
    s.store.hnsw_sync_incremental();

    s.next_id += 1;
    let id = params["id"]
        .as_str()
        .map(|s| s.to_string())
        .unwrap_or_else(|| format!("mem_{}", s.next_id));

    // Store text and mapping
    s.vector_to_doc.insert(vector_idx, id.clone());
    s.cache_doc_text(id.clone(), text.to_string());

    // Persist to SQLite
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64();

    let mut metadata_json = serde_json::Map::new();
    metadata_json.insert("text".into(), json!(text));
    if let Some(cat) = category {
        metadata_json.insert("category".into(), json!(cat));
    }

    let record = DocumentRecord {
        id: id.clone(),
        shard_idx: 0,
        vector_idx: vector_idx as i64,
        metadata: Some(serde_json::Value::Object(metadata_json)),
        document: Some(text.to_string()),
        deleted: false,
        created_at: now,
        updated_at: now,
    };

    if let Err(e) = s.doc_store.upsert(&record) {
        eprintln!("[mcp] warning: SQLite upsert failed: {}", e);
    }

    // Register spatial metadata if present
    let wing = params["wing"].as_str();
    let room = params["room"].as_str();
    let hall = params["hall"].as_str();
    if wing.is_some() || room.is_some() || hall.is_some() {
        s.spatial_index.register_doc(&id, wing, room, hall);
    }

    eprintln!(
        "[mcp] store: id={}, vec_idx={}, text_len={}",
        id,
        vector_idx,
        text.len()
    );
    Ok(json!({ "id": id, "status": "stored" }))
}

fn handle_search(state: &Mutex<McpState>, params: &Value) -> Result<Value, String> {
    let query = params["query"].as_str().ok_or("missing 'query' field")?;
    if query.len() > 10_000 {
        return Err("query text exceeds maximum length of 10000 characters".into());
    }
    let top_k = params["top_k"].as_u64().unwrap_or(10) as usize;
    if top_k > 1000 {
        return Err("top_k exceeds maximum value of 1000".into());
    }

    let mut s = state.lock().map_err(|e| {
        eprintln!("[mcp] mutex poisoned: {}", e);
        "internal server error".to_string()
    })?;
    let dim = s.store.get_statistics().embedding_dim;
    let n_active = s.store.n_active();

    if n_active == 0 {
        return Ok(json!({ "results": [] }));
    }

    let k = top_k.min(n_active);
    let embedding_opt = params["embedding"].as_array().map(|arr| {
        arr.iter()
            .filter_map(|v| v.as_f64().map(|f| f as f32))
            .collect::<Vec<_>>()
    });
    let embedding = embedding_opt.unwrap_or_else(|| get_embedding(query, dim));

    if embedding.len() != dim {
        return Err(format!(
            "Embedding dimension mismatch: expected {dim}, got {}",
            embedding.len()
        ));
    }

    let query_vec = Array1::from_vec(embedding);
    let neighbors = s.store.find_neighbors_fast(&query_vec.view(), k);

    let results: Vec<Value> = neighbors
        .into_iter()
        .map(|n| {
            // Look up doc_id from vector index
            let doc_id = s
                .vector_to_doc
                .get(&n.index)
                .cloned()
                .unwrap_or_else(|| format!("vec_{}", n.index));
            let text = s.doc_texts.get(&doc_id).cloned().unwrap_or_default();

            // Try to get metadata from SQLite
            let metadata: Value = match s.doc_store.get(&doc_id) {
                Ok(Some(record)) => record.metadata.clone().unwrap_or(json!(null)),
                _ => json!(null),
            };

            json!({
                "id": doc_id,
                "index": n.index,
                "score": n.distance,
                "text": text,
                "metadata": metadata,
            })
        })
        .collect();

    eprintln!(
        "[mcp] search: query_len={}, results={}",
        query.len(),
        results.len()
    );
    Ok(json!({ "results": results }))
}

fn handle_status(state: &Mutex<McpState>) -> Result<Value, String> {
    let s = state.lock().map_err(|e| {
        eprintln!("[mcp] mutex poisoned: {}", e);
        "internal server error".to_string()
    })?;
    let stats = s.store.get_statistics();

    let doc_count = s.doc_store.count(false).unwrap_or(0);

    Ok(json!({
        "n_active": s.store.n_active(),
        "max_splats": s.store.max_splats(),
        "dimension": stats.embedding_dim,
        "doc_count": doc_count,
        "has_hnsw": s.store.has_hnsw(),
        "has_lsh": s.store.has_lsh(),
        "has_quantization": s.store.has_quantization(),
        "has_semantic_memory": s.store.has_semantic_memory(),
    }))
}

fn handle_doc_add(state: &Mutex<McpState>, params: &Value) -> Result<Value, String> {
    let id = params["id"].as_str().ok_or("missing 'id'")?;
    let text = params["text"].as_str().ok_or("missing 'text'")?;
    let metadata = params["metadata"].as_str();
    const MAX_TEXT_LEN: usize = 100_000;
    if text.len() > MAX_TEXT_LEN {
        return Err(format!("Text exceeds max length ({MAX_TEXT_LEN})"));
    }

    let mut s = state.lock().map_err(|e| {
        eprintln!("[mcp] mutex poisoned: {}", e);
        "internal server error".to_string()
    })?;
    let dim = s.store.get_statistics().embedding_dim;

    // Compute embedding (real model or SimCos fallback)
    let embedding = get_embedding(text, dim);
    let arr = Array2::from_shape_vec((1, dim), embedding)
        .map_err(|e| format!("embedding error: {}", e))?;

    let vector_idx = s.store.n_active();
    s.store.add_splat(&arr);
    s.store.hnsw_sync_incremental();

    // Update mappings
    s.vector_to_doc.insert(vector_idx, id.to_string());
    s.cache_doc_text(id.to_string(), text.to_string());

    // Persist to SQLite
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64();

    let meta_val: Option<serde_json::Value> = metadata
        .and_then(|m| serde_json::from_str(m).ok())
        .or_else(|| Some(json!({})));

    let mut full_meta = meta_val.unwrap_or(json!({}));
    if let Some(obj) = full_meta.as_object_mut() {
        obj.insert("text".into(), json!(text));
    }

    let record = DocumentRecord {
        id: id.to_string(),
        shard_idx: 0,
        vector_idx: vector_idx as i64,
        metadata: Some(full_meta),
        document: Some(text.to_string()),
        deleted: false,
        created_at: now,
        updated_at: now,
    };

    s.doc_store
        .upsert(&record)
        .map_err(|e| format!("SQLite error: {}", e))?;

    eprintln!(
        "[mcp] doc_add: id={}, vec_idx={}, text_len={}, has_meta={}",
        id,
        vector_idx,
        text.len(),
        metadata.is_some()
    );
    Ok(json!({ "ok": true, "id": id }))
}

fn handle_doc_get(state: &Mutex<McpState>, params: &Value) -> Result<Value, String> {
    let id = params["id"].as_str().ok_or("missing 'id'")?;

    let s = state.lock().map_err(|e| {
        eprintln!("[mcp] mutex poisoned: {}", e);
        "internal server error".to_string()
    })?;

    match s.doc_store.get(id) {
        Ok(Some(record)) => {
            let text = record.document.unwrap_or_default();
            Ok(json!({
                "id": record.id,
                "text": text,
                "metadata": record.metadata,
                "vector_idx": record.vector_idx,
                "created_at": record.created_at,
            }))
        }
        Ok(None) => Err(format!("document not found: {}", id)),
        Err(e) => Err(format!("SQLite error: {}", e)),
    }
}

fn handle_doc_del(state: &Mutex<McpState>, params: &Value) -> Result<Value, String> {
    let id = params["id"].as_str().ok_or("missing 'id'")?;

    let mut s = state.lock().map_err(|e| {
        eprintln!("[mcp] mutex poisoned: {}", e);
        "internal server error".to_string()
    })?;

    // Soft delete in SQLite
    match s.doc_store.soft_delete(id) {
        Ok(true) => {
            // Remove from in-memory caches
            s.uncache_doc_text(id);
            // Find and remove the vector_to_doc mapping for this doc id
            if let Some(idx) = s
                .vector_to_doc
                .iter()
                .find(|(_, v)| *v == id)
                .map(|(k, _)| *k)
            {
                s.vector_to_doc.remove(&idx);
            }
            eprintln!("[mcp] doc_del: id={} (soft deleted)", id);
            Ok(json!({ "ok": true, "id": id }))
        }
        Ok(false) => Err(format!("document not found: {}", id)),
        Err(e) => Err(format!("SQLite error: {}", e)),
    }
}

// ============================================================================
// Graph Splat tool handlers
// ============================================================================

fn handle_graph_add_doc(state: &Mutex<McpState>, params: &Value) -> Result<Value, String> {
    let text = params["text"].as_str().ok_or("missing 'text' field")?;
    const MAX_TEXT_LEN: usize = 100_000;
    if text.len() > MAX_TEXT_LEN {
        return Err(format!("Text exceeds max length ({MAX_TEXT_LEN})"));
    }

    let mut s = state.lock().map_err(|e| {
        eprintln!("[mcp] mutex poisoned: {}", e);
        "internal server error".to_string()
    })?;
    let dim = s.store.get_statistics().embedding_dim;
    let embedding = get_embedding(text, dim);

    let node_id = s
        .graph
        .add_document(text, &embedding)
        .map_err(|e| e.to_string())?;

    eprintln!(
        "[mcp] graph_add_doc: node_id={}, text_len={}",
        node_id,
        text.len()
    );
    Ok(json!({ "node_id": node_id, "node_type": "document" }))
}

fn handle_graph_add_entity(state: &Mutex<McpState>, params: &Value) -> Result<Value, String> {
    let name = params["name"].as_str().ok_or("missing 'name' field")?;
    let entity_type = params["entity_type"]
        .as_str()
        .ok_or("missing 'entity_type' field")?;
    const MAX_NAME_LEN: usize = 10_000;
    if name.len() > MAX_NAME_LEN {
        return Err(format!("Name exceeds max length ({MAX_NAME_LEN})"));
    }

    let mut s = state.lock().map_err(|e| {
        eprintln!("[mcp] mutex poisoned: {}", e);
        "internal server error".to_string()
    })?;
    let dim = s.store.get_statistics().embedding_dim;
    let embedding = get_embedding(name, dim);

    let node_id = s
        .graph
        .add_entity(name, &embedding, entity_type)
        .map_err(|e| e.to_string())?;

    eprintln!(
        "[mcp] graph_add_entity: node_id={}, name_len={}, type='{}'",
        node_id,
        name.len(),
        entity_type
    );
    Ok(json!({ "node_id": node_id, "name": name, "entity_type": entity_type }))
}

fn handle_graph_add_relation(state: &Mutex<McpState>, params: &Value) -> Result<Value, String> {
    let source_id = params["source_id"].as_u64().ok_or("missing 'source_id'")? as usize;
    let target_id = params["target_id"].as_u64().ok_or("missing 'target_id'")? as usize;
    let relation_type = params["relation_type"]
        .as_str()
        .ok_or("missing 'relation_type'")?;
    let weight = params["weight"].as_f64().unwrap_or(1.0);

    let mut s = state.lock().map_err(|e| {
        eprintln!("[mcp] mutex poisoned: {}", e);
        "internal server error".to_string()
    })?;
    s.graph
        .add_relation(source_id, target_id, relation_type, weight)
        .map_err(|e| e.to_string())?;

    eprintln!(
        "[mcp] graph_add_relation: {} --{}--> {} (w={})",
        source_id, relation_type, target_id, weight
    );
    Ok(json!({
        "ok": true,
        "source_id": source_id,
        "target_id": target_id,
        "relation_type": relation_type,
        "weight": weight
    }))
}

fn handle_graph_traverse(state: &Mutex<McpState>, params: &Value) -> Result<Value, String> {
    let start_id = params["start_id"].as_u64().ok_or("missing 'start_id'")? as usize;
    let max_depth = params["max_depth"].as_u64().unwrap_or(3).min(20) as usize;

    let s = state.lock().map_err(|e| {
        eprintln!("[mcp] mutex poisoned: {}", e);
        "internal server error".to_string()
    })?;

    if s.graph.get_node(start_id).is_none() {
        return Err(format!("node not found: {}", start_id));
    }

    let visited = s.graph.traverse(start_id, max_depth);

    let nodes: Vec<Value> = visited
        .iter()
        .filter_map(|&id| {
            s.graph.get_node(id).map(|n| {
                json!({
                    "id": n.id,
                    "node_type": format!("{:?}", n.node_type).to_lowercase(),
                    "content": n.content,
                })
            })
        })
        .collect();

    eprintln!(
        "[mcp] graph_traverse: start={}, depth={}, found={}",
        start_id,
        max_depth,
        nodes.len()
    );
    Ok(json!({ "nodes": nodes, "count": nodes.len() }))
}

fn handle_graph_search(state: &Mutex<McpState>, params: &Value) -> Result<Value, String> {
    let query = params["query"].as_str().ok_or("missing 'query' field")?;
    let k = params["k"].as_u64().unwrap_or(10).min(1000) as usize;

    let s = state.lock().map_err(|e| {
        eprintln!("[mcp] mutex poisoned: {}", e);
        "internal server error".to_string()
    })?;
    let dim = s.store.get_statistics().embedding_dim;
    let embedding = get_embedding(query, dim);

    let results = s.graph.hybrid_search(&embedding, k);

    let out: Vec<Value> = results
        .iter()
        .map(|r| {
            json!({
                "node_id": r.node_id,
                "content": r.content,
                "score": r.score,
                "node_type": format!("{:?}", r.node_type).to_lowercase(),
            })
        })
        .collect();

    eprintln!(
        "[mcp] graph_search: query_len={}, results={}",
        query.len(),
        out.len()
    );
    Ok(json!({ "results": out }))
}

fn handle_graph_search_entities(state: &Mutex<McpState>, params: &Value) -> Result<Value, String> {
    let query = params["query"].as_str().ok_or("missing 'query' field")?;
    let k = params["k"].as_u64().unwrap_or(10).min(1000) as usize;

    let s = state.lock().map_err(|e| {
        eprintln!("[mcp] mutex poisoned: {}", e);
        "internal server error".to_string()
    })?;
    let dim = s.store.get_statistics().embedding_dim;
    let embedding = get_embedding(query, dim);

    let results = s.graph.search_entities(&embedding, k);

    let out: Vec<Value> = results
        .iter()
        .map(|n| {
            json!({
                "id": n.id,
                "content": n.content,
                "node_type": format!("{:?}", n.node_type).to_lowercase(),
            })
        })
        .collect();

    eprintln!(
        "[mcp] graph_search_entities: query_len={}, results={}",
        query.len(),
        out.len()
    );
    Ok(json!({ "results": out }))
}

fn handle_graph_stats(state: &Mutex<McpState>) -> Result<Value, String> {
    let s = state.lock().map_err(|e| {
        eprintln!("[mcp] mutex poisoned: {}", e);
        "internal server error".to_string()
    })?;
    let stats = s.graph.get_stats();

    Ok(json!({
        "total_nodes": stats.total_nodes,
        "total_edges": stats.total_edges,
        "documents": stats.documents,
        "entities": stats.entities,
        "concepts": stats.concepts,
    }))
}

// ============================================================================
// Spatial Memory handlers
// ============================================================================

/// Spatial search: uses SpatialIndex for O(|candidates|×dim) pre-filter search.
///
/// Pipeline:
///   1. SpatialIndex.filter() → candidate doc IDs (in-memory, no SQLite)
///   2. doc_id → vector_idx mapping (in-memory HashMap)
///   3. SplatStore.find_neighbors_filtered() → distance only against candidates
///
/// This avoids the old post-filter approach that computed distances against ALL N vectors.
fn handle_spatial_search(state: &Mutex<McpState>, params: &Value) -> Result<Value, String> {
    let query = params["query"].as_str().ok_or("missing 'query' field")?;
    if query.len() > 10_000 {
        return Err("query text exceeds maximum length of 10000 characters".into());
    }
    let top_k = params["top_k"].as_u64().unwrap_or(10) as usize;
    if top_k > 1000 {
        return Err("top_k exceeds maximum value of 1000".into());
    }

    let wing_filter = params["wing"].as_str().map(|s| s.to_string());
    let room_filter = params["room"].as_str().map(|s| s.to_string());
    let hall_filter = params["hall"].as_str().map(|s| s.to_string());

    let has_filter = wing_filter.is_some() || room_filter.is_some() || hall_filter.is_some();

    let mut s = state.lock().map_err(|e| {
        eprintln!("[mcp] mutex poisoned: {}", e);
        "internal server error".to_string()
    })?;

    let dim = s.store.get_statistics().embedding_dim;
    let n_active = s.store.n_active();
    if n_active == 0 {
        return Ok(json!({ "results": [], "total": 0, "filter": "no documents" }));
    }

    // Build inverted index: doc_id → vector_idx
    let mut doc_to_vec: HashMap<String, usize> = HashMap::new();
    for (&vec_idx, doc_id) in &s.vector_to_doc {
        doc_to_vec.insert(doc_id.clone(), vec_idx);
    }

    // Get candidate vector indices via SpatialIndex (pre-filter)
    let candidate_vec_indices: Vec<usize> = if has_filter {
        let spatial_filter = crate::spatial::SpatialFilter {
            wing: wing_filter.clone(),
            room: room_filter.clone(),
            hall: hall_filter.clone(),
        };
        let candidate_ids = s.spatial_index.filter(&spatial_filter);
        candidate_ids
            .iter()
            .filter_map(|id| doc_to_vec.get(id).copied())
            .collect()
    } else {
        // No spatial filter — search all vectors
        (0..n_active).collect()
    };

    if candidate_vec_indices.is_empty() {
        let filter_desc = describe_filter(&wing_filter, &room_filter, &hall_filter);
        return Ok(json!({
            "results": [],
            "total": 0,
            "filter": filter_desc,
            "candidates_scanned": 0,
        }));
    }

    // Get embedding for query
    let embedding = get_embedding(query, dim);
    if embedding.len() != dim {
        return Err(format!(
            "Embedding dimension mismatch: expected {dim}, got {}",
            embedding.len()
        ));
    }
    let query_vec = Array1::from_vec(embedding);

    // Pre-filter search: only compute distances against candidates
    let neighbors = if has_filter {
        s.store
            .find_neighbors_filtered(&query_vec.view(), &candidate_vec_indices, top_k)
    } else {
        let k = top_k.min(n_active);
        s.store.find_neighbors_fast(&query_vec.view(), k)
    };

    let mut out = Vec::new();
    for n in neighbors {
        let doc_id = s
            .vector_to_doc
            .get(&n.index)
            .cloned()
            .unwrap_or_else(|| format!("vec_{}", n.index));
        let text = s.doc_texts.get(&doc_id).cloned().unwrap_or_default();
        let metadata: Value = match s.doc_store.get(&doc_id) {
            Ok(Some(record)) => record.metadata.clone().unwrap_or(json!(null)),
            _ => json!(null),
        };
        out.push(json!({
            "id": doc_id,
            "index": n.index,
            "score": n.distance,
            "text": text,
            "metadata": metadata,
        }));
        if out.len() >= top_k {
            break;
        }
    }

    let filter_desc = describe_filter(&wing_filter, &room_filter, &hall_filter);
    eprintln!(
        "[mcp] spatial_search: query_len={}, filter={}, candidates={}, results={}",
        query.len(),
        filter_desc,
        candidate_vec_indices.len(),
        out.len()
    );

    Ok(json!({
        "results": out,
        "total": out.len(),
        "filter": filter_desc,
        "candidates_scanned": candidate_vec_indices.len(),
    }))
}

fn describe_filter(wing: &Option<String>, room: &Option<String>, hall: &Option<String>) -> String {
    match (wing, room, hall) {
        (Some(w), Some(r), Some(h)) => format!("wing={w}, room={r}, hall={h}"),
        (Some(w), Some(r), None) => format!("wing={w}, room={r}"),
        (Some(w), None, Some(h)) => format!("wing={w}, hall={h}"),
        (None, Some(r), Some(h)) => format!("room={r}, hall={h}"),
        (Some(w), None, None) => format!("wing={w}"),
        (None, Some(r), None) => format!("room={r}"),
        (None, None, Some(h)) => format!("hall={h}"),
        _ => "no spatial filter".into(),
    }
}

/// Show spatial memory structure from metadata and in-memory SpatialIndex.
fn handle_spatial_info(state: &Mutex<McpState>) -> Result<Value, String> {
    let s = state.lock().map_err(|e| {
        eprintln!("[mcp] mutex poisoned: {}", e);
        "internal server error".to_string()
    })?;

    let all_ids = s.doc_store.list_ids(false).map_err(|e| e.to_string())?;

    // Scan metadata to build spatial structure
    let mut wings: std::collections::HashMap<String, std::collections::HashSet<String>> =
        std::collections::HashMap::new();
    let mut halls: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut total_with_spatial = 0usize;

    for id in &all_ids {
        if let Ok(Some(rec)) = s.doc_store.get(id) {
            if let Some(ref meta) = rec.metadata {
                let has_wing = meta.get("wing").and_then(|v| v.as_str()).is_some();
                let has_room = meta.get("room").and_then(|v| v.as_str()).is_some();
                let has_hall = meta.get("hall").and_then(|v| v.as_str());

                if has_wing || has_room || has_hall.is_some() {
                    total_with_spatial += 1;
                }

                if let (Some(w), Some(r)) = (
                    meta.get("wing").and_then(|v| v.as_str()),
                    meta.get("room").and_then(|v| v.as_str()),
                ) {
                    wings
                        .entry(w.to_string())
                        .or_default()
                        .insert(r.to_string());
                }

                if let Some(h) = has_hall {
                    halls.insert(h.to_string());
                }
            }
        }
    }

    // Detect tunnels: rooms shared across wings
    let mut room_wings: std::collections::HashMap<String, Vec<String>> =
        std::collections::HashMap::new();
    for (wing, rooms) in &wings {
        for room in rooms {
            room_wings
                .entry(room.clone())
                .or_default()
                .push(wing.clone());
        }
    }
    let tunnels: Vec<Value> = room_wings
        .iter()
        .filter(|(_, ws)| ws.len() >= 2)
        .map(|(room, ws)| {
            json!({
                "room": room,
                "wings": ws,
            })
        })
        .collect();

    let wings_json: Value = wings
        .iter()
        .map(|(w, rooms)| {
            let rooms_vec: Vec<&String> = rooms.iter().collect();
            (w.clone(), json!(rooms_vec))
        })
        .collect();

    Ok(json!({
        "total_documents": all_ids.len(),
        "documents_with_spatial_metadata": total_with_spatial,
        "wings": wings_json,
        "halls": halls.iter().collect::<Vec<_>>(),
        "tunnels": tunnels,
        "tunnel_count": tunnels.len(),
    }))
}

// ============================================================================
// Verbatim Storage + AAAK Compression handlers
// ============================================================================

fn handle_verbatim_search(state: &Mutex<McpState>, params: &Value) -> Result<Value, String> {
    let query = params["query"].as_str().ok_or("missing 'query' field")?;
    let top_k = params["top_k"].as_u64().unwrap_or(10) as usize;

    let mut s = state.lock().map_err(|e| {
        eprintln!("[mcp] mutex poisoned: {}", e);
        "internal server error".to_string()
    })?;

    let n_active = s.store.n_active();
    if n_active == 0 {
        return Ok(json!({ "results": [], "message": "No documents in store" }));
    }

    let dim = s.store.get_statistics().embedding_dim;
    let k = top_k.min(n_active);
    let embedding = get_embedding(query, dim);
    let query_vec = Array1::from_vec(embedding);
    let neighbors = s.store.find_neighbors_fast(&query_vec.view(), k);

    let results: Vec<Value> = neighbors
        .into_iter()
        .map(|n| {
            let doc_id = s.vector_to_doc.get(&n.index).cloned().unwrap_or_default();
            let text = s.doc_texts.get(&doc_id).cloned().unwrap_or_default();

            // Confidence classification
            let (confidence, emoji, explanation) = if n.distance < 0.3 {
                ("HIGH", "🟢", "Very reliable match")
            } else if n.distance < 0.6 {
                ("MEDIUM", "🟡", "Good match — verify context")
            } else {
                ("LOW", "🔴", "Weak match — verify against source")
            };

            let similarity = (1.0 / (1.0 + n.distance as f64)).min(1.0);

            json!({
                "id": doc_id,
                "index": n.index,
                "distance": (n.distance * 10000.0).round() / 10000.0,
                "similarity": (similarity * 10000.0).round() / 10000.0,
                "confidence": confidence,
                "confidence_emoji": emoji,
                "explanation": explanation,
                "text": text,
                "warning": if confidence == "LOW" {
                    Some("LOW CONFIDENCE — may be hallucinated, verify against original source")
                } else {
                    None
                },
            })
        })
        .collect();

    let high_count = results.iter().filter(|r| r["confidence"].as_str() == Some("HIGH")).count();
    let medium_count = results.iter().filter(|r| r["confidence"].as_str() == Some("MEDIUM")).count();
    let low_count = results.iter().filter(|r| r["confidence"].as_str() == Some("LOW")).count();

    Ok(json!({
        "query": query,
        "results": results,
        "summary": {
            "total": results.len(),
            "high": high_count,
            "medium": medium_count,
            "low": low_count,
            "reliable": high_count + medium_count,
        },
        "disclaimer": "Always verify LOW confidence results against original source text to prevent hallucination",
    }))
}

fn handle_compress(params: &Value) -> Result<Value, String> {
    let text = params["text"].as_str().ok_or("missing 'text' field")?;
    let result = crate::text_compression::compress(text);

    let hex_data: String = result.binary_data.iter()
        .map(|b| format!("{:02x}", b))
        .collect();

    Ok(json!({
        "semantic_text": result.semantic_text,
        "original_size": result.original_size,
        "semantic_size": result.semantic_size,
        "binary_size": result.binary_size,
        "semantic_ratio": format!("{:.2}×", result.semantic_ratio),
        "total_ratio": format!("{:.2}×", result.compression_ratio),
        "data": hex_data,
    }))
}

fn handle_decompress(params: &Value) -> Result<Value, String> {
    let hex_data = params["data"].as_str().ok_or("missing 'data' field")?;
    let binary_data: Vec<u8> = (0..hex_data.len())
        .step_by(2)
        .filter_map(|i| u8::from_str_radix(&hex_data[i..i + 2], 16).ok())
        .collect();

    let text = crate::text_compression::decompress(&binary_data)
        .map_err(|e| format!("Decompression error: {}", e))?;

    Ok(json!({
        "text": text,
        "binary_size": binary_data.len(),
        "text_size": text.len(),
    }))
}

// ============================================================================
// JSON-RPC dispatcher
// ============================================================================

fn dispatch_request(state: &Mutex<McpState>, req: &JsonRpcRequest) -> JsonRpcResponse {
    match req.method.as_str() {
        "initialize" => JsonRpcResponse {
            jsonrpc: "2.0".into(),
            id: req.id.clone(),
            result: Some(json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": { "listChanged": false }
                },
                "serverInfo": {
                    "name": "splatsdb",
                    "version": "2.5.0"
                }
            })),
            error: None,
        },

        "notifications/initialized" => JsonRpcResponse {
            jsonrpc: "2.0".into(),
            id: None,
            result: None,
            error: None,
        },

        "tools/list" => JsonRpcResponse {
            jsonrpc: "2.0".into(),
            id: req.id.clone(),
            result: Some(json!({ "tools": tool_definitions() })),
            error: None,
        },

        "tools/call" => {
            let params = req.params.as_ref();
            let tool_name = params.and_then(|p| p["name"].as_str()).unwrap_or("");

            let arguments = params
                .and_then(|p| p.get("arguments"))
                .cloned()
                .unwrap_or(json!({}));

            let result = match tool_name {
                "splatsdb_store" => handle_store(state, &arguments),
                "splatsdb_search" => handle_search(state, &arguments),
                "splatsdb_status" => handle_status(state),
                "splatsdb_doc_add" => handle_doc_add(state, &arguments),
                "splatsdb_doc_get" => handle_doc_get(state, &arguments),
                "splatsdb_doc_del" => handle_doc_del(state, &arguments),
                "splatsdb_graph_add_doc" => handle_graph_add_doc(state, &arguments),
                "splatsdb_graph_add_entity" => handle_graph_add_entity(state, &arguments),
                "splatsdb_graph_add_relation" => handle_graph_add_relation(state, &arguments),
                "splatsdb_graph_traverse" => handle_graph_traverse(state, &arguments),
                "splatsdb_graph_search" => handle_graph_search(state, &arguments),
                "splatsdb_graph_search_entities" => handle_graph_search_entities(state, &arguments),
                "splatsdb_graph_stats" => handle_graph_stats(state),
                "splatsdb_spatial_search" => handle_spatial_search(state, &arguments),
                "splatsdb_spatial_info" => handle_spatial_info(state),
                "splatsdb_verbatim_search" => handle_verbatim_search(state, &arguments),
                "splatsdb_compress" => handle_compress(&arguments),
                "splatsdb_decompress" => handle_decompress(&arguments),
                _ => Err(format!("unknown tool: {}", tool_name)),
            };

            match result {
                Ok(val) => JsonRpcResponse {
                    jsonrpc: "2.0".into(),
                    id: req.id.clone(),
                    result: Some(json!({
                        "content": [{ "type": "text", "text": val.to_string() }],
                        "isError": false
                    })),
                    error: None,
                },
                Err(msg) => JsonRpcResponse {
                    jsonrpc: "2.0".into(),
                    id: req.id.clone(),
                    result: Some(json!({
                        "content": [{ "type": "text", "text": format!("Error: {}", msg) }],
                        "isError": true
                    })),
                    error: None,
                },
            }
        }

        _ => JsonRpcResponse {
            jsonrpc: "2.0".into(),
            id: req.id.clone(),
            result: None,
            error: Some(JsonRpcError {
                code: -32601,
                message: format!("Method not found: {}", req.method),
            }),
        },
    }
}

// ============================================================================
// Main entry point
// ============================================================================

pub fn run_mcp_server() {
    eprintln!("[splatsdb] MCP server v2.5 starting (stdio transport)...");

    let config = SplatsDBConfig::mcp(None);
    let mut store = SplatStore::new(config);

    // Initialize SQLite doc store
    let db_path = std::env::var("SPLATSDB_DOC_PATH")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
            std::path::PathBuf::from(format!("{home}/.hermes/splatsdb_docs.db"))
        });
    let doc_store = match SqliteMetadataStore::new(db_path.clone()) {
        Ok(ds) => {
            eprintln!("[splatsdb] SQLite doc store: {:?}", db_path);
            ds
        }
        Err(e) => {
            eprintln!(
                "[splatsdb] WARNING: SQLite init failed ({}), docs won't persist",
                e
            );
            SqliteMetadataStore::new(std::path::PathBuf::from(":memory:")).unwrap_or_else(|e| {
                eprintln!("[splatsdb] FATAL: in-memory SQLite failed: {}", e);
                std::process::exit(1);
            })
        }
    };

    // Warm start: reload documents from SQLite into vector store
    let mut vector_to_doc: HashMap<usize, String> = HashMap::new();
    let mut doc_texts: HashMap<String, String> = HashMap::new();
    let mut doc_text_order: Vec<String> = Vec::new();
    let mut next_id: usize = 0;
    let dim = store.get_statistics().embedding_dim;

    match doc_store.list_ids(false) {
        Ok(ids) => {
            let total = ids.len();
            if total > MAX_WARM_START_DOCS {
                eprintln!(
                    "[splatsdb] Warm start: {} docs found, capping at {} for performance",
                    total, MAX_WARM_START_DOCS
                );
            }
            for (i, doc_id) in ids.iter().enumerate().take(MAX_WARM_START_DOCS) {
                if SHUTDOWN_REQUESTED.load(Ordering::Relaxed) {
                    eprintln!("[splatsdb] Warm start interrupted by shutdown signal");
                    break;
                }
                if let Ok(Some(record)) = doc_store.get(doc_id) {
                    let text = record.document.clone().unwrap_or_default();
                    if text.is_empty() {
                        continue;
                    }

                    // Re-compute embedding and add to vector store
                    let embedding = get_embedding(&text, dim);
                    if let Ok(arr) = Array2::from_shape_vec((1, dim), embedding) {
                        let vector_idx = store.n_active();
                        if store.add_splat(&arr) {
                            vector_to_doc.insert(vector_idx, doc_id.clone());
                            doc_texts.insert(doc_id.clone(), text);
                            doc_text_order.push(doc_id.clone());
                            next_id = next_id.max(
                                doc_id
                                    .strip_prefix("mem_")
                                    .and_then(|s| s.parse::<usize>().ok())
                                    .unwrap_or(0),
                            );
                        }
                    }
                }
                if i > 0 && i % 100 == 0 {
                    eprintln!(
                        "[splatsdb] Warm start: {}/{} docs loaded",
                        i,
                        total.min(MAX_WARM_START_DOCS)
                    );
                }
            }
            if !ids.is_empty() {
                store.hnsw_sync_incremental();
                eprintln!(
                    "[splatsdb] Warm start: reloaded {}/{} documents from SQLite",
                    doc_texts.len().min(MAX_WARM_START_DOCS),
                    total
                );
            }
        }
        Err(e) => eprintln!("[splatsdb] Warm start: could not list docs: {}", e),
    }

    let state = Mutex::new(McpState {
        store,
        doc_store,
        vector_to_doc,
        doc_texts,
        doc_text_order,
        next_id,
        graph: crate::graph_splat::GaussianGraphStore::new(),
        spatial_index: crate::spatial::SpatialIndex::new(),
    });

    let stdin = io::stdin();
    let mut stdout = io::stdout();
    let reader = stdin.lock();

    // Set up signal handler for graceful shutdown
    #[cfg(unix)]
    {
        unsafe {
            libc::signal(libc::SIGINT, handle_sigint as *const () as usize);
        }
    }

    eprintln!("[splatsdb] MCP server ready. Waiting for JSON-RPC on stdin...");
    eprintln!(
        "[splatsdb] Env vars: SPLATSDB_EMBED_URL={}, SPLATSDB_DOC_PATH={}",
        std::env::var("SPLATSDB_EMBED_URL").unwrap_or_else(|_| "(default)".into()),
        std::env::var("SPLATSDB_DOC_PATH").unwrap_or_else(|_| "(default)".into())
    );

    for line in reader.lines() {
        if SHUTDOWN_REQUESTED.load(Ordering::Relaxed) {
            eprintln!("[splatsdb] Shutdown requested, exiting main loop.");
            break;
        }
        match line {
            Ok(line) => {
                let line = line.trim().to_string();
                if line.is_empty() {
                    continue;
                }

                let req: JsonRpcRequest = match serde_json::from_str::<JsonRpcRequest>(&line) {
                    Ok(r) => {
                        // Validate jsonrpc version
                        if r.jsonrpc != "2.0" {
                            eprintln!("[splatsdb] invalid jsonrpc version: {}", r.jsonrpc);
                            let resp = JsonRpcResponse {
                                jsonrpc: "2.0".into(),
                                id: r.id.clone(),
                                result: None,
                                error: Some(JsonRpcError {
                                    code: -32600,
                                    message: format!(
                                        "Invalid jsonrpc version: expected '2.0', got '{}'",
                                        r.jsonrpc
                                    ),
                                }),
                            };
                            let out = serde_json::to_string(&resp)
                                .unwrap_or_else(|_| r#"{"jsonrpc":"2.0","id":null,"error":{"code":-32600,"message":"Invalid request"}}"#.to_string());
                            writeln!(stdout, "{}", out).ok();
                            stdout.flush().ok();
                            continue;
                        }
                        eprintln!("[splatsdb] <- method={}", r.method);
                        r
                    }
                    Err(e) => {
                        eprintln!("[splatsdb] parse error: {}", e);
                        let resp = JsonRpcResponse {
                            jsonrpc: "2.0".into(),
                            id: None,
                            result: None,
                            error: Some(JsonRpcError {
                                code: -32700,
                                message: format!("Parse error: {}", e),
                            }),
                        };
                        let out = serde_json::to_string(&resp)
                            .unwrap_or_else(|_| r#"{"jsonrpc":"2.0","id":null,"error":{"code":-32700,"message":"Parse error"}}"#.to_string());
                        writeln!(stdout, "{}", out).ok();
                        stdout.flush().ok();
                        continue;
                    }
                };

                let is_notification = req.id.is_none() && req.method != "initialize";

                let resp = dispatch_request(&state, &req);

                if !is_notification && resp.id.is_some() {
                    let out = match serde_json::to_string(&resp) {
                        Ok(s) => s,
                        Err(e) => {
                            eprintln!("[splatsdb] serialization error: {}", e);
                            r#"{"jsonrpc":"2.0","id":null,"error":{"code":-32603,"message":"Internal error"}}"#.to_string()
                        }
                    };
                    let preview: String = out.chars().take(200).collect();
                    eprintln!("[splatsdb] -> {}...", preview);
                    writeln!(stdout, "{}", out).ok();
                    stdout.flush().ok();
                }
            }
            Err(e) => {
                eprintln!("[splatsdb] stdin error: {}", e);
                break;
            }
        }
    }

    eprintln!("[splatsdb] MCP server shutting down (stdin closed).");
}
