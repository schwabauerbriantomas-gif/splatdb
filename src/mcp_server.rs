//! MCP (Model Context Protocol) Server — stdio transport
//!
//! Exposes SplatDB Vector Search as an MCP server for AI agent integration.
//! Uses JSON-RPC 2.0 over stdin/stdout (stdio transport).
//! All logs go to stderr to keep the protocol channel clean.
//!
//! v2.4: Real embeddings via HTTP embedding service (MiniLM-L6-v2) with SimCos fallback.

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::{self, BufRead, Write};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::config::SplatDBConfig;
use crate::splats::SplatStore;
use crate::storage::metadata_store::{DocumentRecord, MetadataStore};
use crate::storage::sqlite_store::SqliteMetadataStore;

// ============================================================================
// Embedding client — real model via HTTP, SimCos fallback
// ============================================================================

const EMBEDDING_SERVICE_URL: &str = "http://127.0.0.1:8788/embed";
static EMBED_SERVICE_AVAILABLE: AtomicBool = AtomicBool::new(true);

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
        .post(EMBEDDING_SERVICE_URL)
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
    /// Maps doc_id → text content (in-memory cache of document text)
    doc_texts: HashMap<String, String>,
    next_id: usize,
}

#[derive(Deserialize)]
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
            "name": "splatdb_store",
            "description": "Store a memory in the SplatDB vector search engine. Returns the memory ID.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "text": { "type": "string", "description": "The text content to store" },
                    "category": { "type": "string", "description": "Optional category tag" },
                    "id": { "type": "string", "description": "Optional custom ID (auto-generated if omitted)" },
                    "embedding": { "type": "array", "items": { "type": "number" }, "description": "Optional pre-computed embedding vector" }
                },
                "required": ["text"]
            }
        }),
        json!({
            "name": "splatdb_search",
            "description": "Search for similar memories in the SplatDB vector store. Returns ranked results with similarity scores.",
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
            "name": "splatdb_status",
            "description": "Get the current status of the SplatDB vector store (number of memories, dimensions, active indexes).",
            "inputSchema": {
                "type": "object",
                "properties": {}
            }
        }),
        json!({
            "name": "splatdb_doc_add",
            "description": "Add a document with metadata to the SplatDB store. Persists to SQLite.",
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
            "name": "splatdb_doc_get",
            "description": "Retrieve a document by ID from the SplatDB store (SQLite-backed).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "id": { "type": "string", "description": "Document ID to retrieve" }
                },
                "required": ["id"]
            }
        }),
        json!({
            "name": "splatdb_doc_del",
            "description": "Delete a document from the SplatDB store (SQLite-backed, soft delete).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "id": { "type": "string", "description": "Document ID to delete" }
                },
                "required": ["id"]
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

            let sign = if sign_hash % 2 == 0 { 1.0f32 } else { -1.0f32 };
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
    let category = params["category"].as_str();
    let embedding_opt = params["embedding"].as_array().map(|arr| {
        arr.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect::<Vec<_>>()
    });

    let mut s = state.lock().map_err(|e| format!("lock error: {}", e))?;
    let dim = s.store.get_statistics().embedding_dim;

    // Use provided embedding, or get real embedding (MiniLM or SimCos fallback)
    let embedding = embedding_opt.unwrap_or_else(|| get_embedding(text, dim));

    let arr = Array2::from_shape_vec((1, dim), embedding)
        .map_err(|e| format!("bad embedding shape: {}", e))?;

    let vector_idx = s.store.n_active();
    let added = s.store.add_splat(&arr);
    if !added {
        return Err("store is full".into());
    }
    s.store.build_index();

    s.next_id += 1;
    let id = params["id"].as_str()
        .map(|s| s.to_string())
        .unwrap_or_else(|| format!("mem_{}", s.next_id));

    // Store text and mapping
    s.vector_to_doc.insert(vector_idx, id.clone());
    s.doc_texts.insert(id.clone(), text.to_string());

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

    eprintln!("[mcp] store: id={}, vec_idx={}, text_len={}", id, vector_idx, text.len());
    Ok(json!({ "id": id, "status": "stored" }))
}

fn handle_search(state: &Mutex<McpState>, params: &Value) -> Result<Value, String> {
    let query = params["query"].as_str().ok_or("missing 'query' field")?;
    let top_k = params["top_k"].as_u64().unwrap_or(10) as usize;

    let s = state.lock().map_err(|e| format!("lock error: {}", e))?;
    let dim = s.store.get_statistics().embedding_dim;
    let n_active = s.store.n_active();

    if n_active == 0 {
        return Ok(json!({ "results": [] }));
    }

    let k = top_k.min(n_active);
    let embedding_opt = params["embedding"].as_array().map(|arr| {
        arr.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect::<Vec<_>>()
    });
    let embedding = embedding_opt.unwrap_or_else(|| get_embedding(query, dim));

    let query_vec = Array1::from_vec(embedding);
    let neighbors = s.store.find_neighbors(&query_vec.view(), k);

    let results: Vec<Value> = neighbors.into_iter().map(|n| {
        // Look up doc_id from vector index
        let doc_id = s.vector_to_doc.get(&n.index)
            .cloned()
            .unwrap_or_else(|| format!("vec_{}", n.index));
        let text = s.doc_texts.get(&doc_id)
            .cloned()
            .unwrap_or_default();

        // Try to get metadata from SQLite
        let metadata: Value = match s.doc_store.get(&doc_id) {
            Ok(Some(record)) => {
                record.metadata.clone().unwrap_or(json!(null))
            },
            _ => json!(null),
        };

        json!({
            "id": doc_id,
            "index": n.index,
            "score": n.distance,
            "text": text,
            "metadata": metadata,
        })
    }).collect();

    eprintln!("[mcp] search: query='{}', results={}", &query[..query.len().min(50)], results.len());
    Ok(json!({ "results": results }))
}

fn handle_status(state: &Mutex<McpState>) -> Result<Value, String> {
    let s = state.lock().map_err(|e| format!("lock error: {}", e))?;
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

    let mut s = state.lock().map_err(|e| format!("lock error: {}", e))?;
    let dim = s.store.get_statistics().embedding_dim;

    // Compute embedding (real model or SimCos fallback)
    let embedding = get_embedding(text, dim);
    let arr = Array2::from_shape_vec((1, dim), embedding)
        .map_err(|e| format!("embedding error: {}", e))?;

    let vector_idx = s.store.n_active();
    s.store.add_splat(&arr);
    s.store.build_index();

    // Update mappings
    s.vector_to_doc.insert(vector_idx, id.to_string());
    s.doc_texts.insert(id.to_string(), text.to_string());

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

    s.doc_store.upsert(&record)
        .map_err(|e| format!("SQLite error: {}", e))?;

    eprintln!("[mcp] doc_add: id={}, vec_idx={}, text_len={}, has_meta={}", id, vector_idx, text.len(), metadata.is_some());
    Ok(json!({ "ok": true, "id": id }))
}

fn handle_doc_get(state: &Mutex<McpState>, params: &Value) -> Result<Value, String> {
    let id = params["id"].as_str().ok_or("missing 'id'")?;

    let s = state.lock().map_err(|e| format!("lock error: {}", e))?;

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
        },
        Ok(None) => Err(format!("document not found: {}", id)),
        Err(e) => Err(format!("SQLite error: {}", e)),
    }
}

fn handle_doc_del(state: &Mutex<McpState>, params: &Value) -> Result<Value, String> {
    let id = params["id"].as_str().ok_or("missing 'id'")?;

    let mut s = state.lock().map_err(|e| format!("lock error: {}", e))?;

    // Soft delete in SQLite
    match s.doc_store.soft_delete(id) {
        Ok(true) => {
            // Remove from in-memory caches
            s.doc_texts.remove(id);
            eprintln!("[mcp] doc_del: id={} (soft deleted)", id);
            Ok(json!({ "ok": true, "id": id }))
        },
        Ok(false) => Err(format!("document not found: {}", id)),
        Err(e) => Err(format!("SQLite error: {}", e)),
    }
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
                    "name": "splatdb",
                    "version": "2.2.0"
                }
            })),
            error: None,
        },

        "notifications/initialized" => {
            return JsonRpcResponse {
                jsonrpc: "2.0".into(),
                id: None,
                result: None,
                error: None,
            };
        }

        "tools/list" => JsonRpcResponse {
            jsonrpc: "2.0".into(),
            id: req.id.clone(),
            result: Some(json!({ "tools": tool_definitions() })),
            error: None,
        },

        "tools/call" => {
            let params = req.params.as_ref();
            let tool_name = params
                .and_then(|p| p["name"].as_str())
                .unwrap_or("");

            let arguments = params
                .and_then(|p| p.get("arguments"))
                .cloned()
                .unwrap_or(json!({}));

            let result = match tool_name {
                "splatdb_store" => handle_store(state, &arguments),
                "splatdb_search" => handle_search(state, &arguments),
                "splatdb_status" => handle_status(state),
                "splatdb_doc_add" => handle_doc_add(state, &arguments),
                "splatdb_doc_get" => handle_doc_get(state, &arguments),
                "splatdb_doc_del" => handle_doc_del(state, &arguments),
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
    eprintln!("[splatdb] MCP server v2.3 starting (stdio transport)...");

    let config = SplatDBConfig::default();
    let mut store = SplatStore::new(config);

    // Initialize SQLite doc store
    let db_path = std::path::PathBuf::from("/root/.hermes/splatdb_docs.db");
    let doc_store = match SqliteMetadataStore::new(db_path.clone()) {
        Ok(ds) => {
            eprintln!("[splatdb] SQLite doc store: {:?}", db_path);
            ds
        }
        Err(e) => {
            eprintln!("[splatdb] WARNING: SQLite init failed ({}), docs won't persist", e);
            SqliteMetadataStore::new(std::path::PathBuf::from(":memory:"))
                .expect("in-memory SQLite should always work")
        }
    };

    // Warm start: reload documents from SQLite into vector store
    let mut vector_to_doc: HashMap<usize, String> = HashMap::new();
    let mut doc_texts: HashMap<String, String> = HashMap::new();
    let mut next_id: usize = 0;
    let dim = store.get_statistics().embedding_dim;

    match doc_store.list_ids(false) {
        Ok(ids) => {
            for doc_id in &ids {
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
                            next_id = next_id.max(
                                doc_id.strip_prefix("mem_")
                                    .and_then(|s| s.parse::<usize>().ok())
                                    .unwrap_or(0)
                            );
                        }
                    }
                }
            }
            if !ids.is_empty() {
                store.build_index();
                eprintln!("[splatdb] Warm start: reloaded {} documents from SQLite", ids.len());
            }
        }
        Err(e) => eprintln!("[splatdb] Warm start: could not list docs: {}", e),
    }

    let state = Mutex::new(McpState {
        store,
        doc_store,
        vector_to_doc,
        doc_texts,
        next_id,
    });

    let stdin = io::stdin();
    let mut stdout = io::stdout();
    let reader = stdin.lock();

    eprintln!("[splatdb] MCP server ready. Waiting for JSON-RPC on stdin...");

    for line in reader.lines() {
        match line {
            Ok(line) => {
                let line = line.trim().to_string();
                if line.is_empty() {
                    continue;
                }

                eprintln!("[splatdb] <- {}", if line.len() > 200 { &line[..200] } else { &line });

                let req: JsonRpcRequest = match serde_json::from_str(&line) {
                    Ok(r) => r,
                    Err(e) => {
                        eprintln!("[splatdb] parse error: {}", e);
                        let resp = JsonRpcResponse {
                            jsonrpc: "2.0".into(),
                            id: None,
                            result: None,
                            error: Some(JsonRpcError {
                                code: -32700,
                                message: format!("Parse error: {}", e),
                            }),
                        };
                        let out = serde_json::to_string(&resp).unwrap();
                        writeln!(stdout, "{}", out).ok();
                        stdout.flush().ok();
                        continue;
                    }
                };

                let is_notification = req.id.is_none() && req.method != "initialize";

                let resp = dispatch_request(&state, &req);

                if !is_notification && resp.id.is_some() {
                    let out = serde_json::to_string(&resp).unwrap();
                    eprintln!("[splatdb] -> {}", if out.len() > 200 { &out[..200] } else { &out });
                    writeln!(stdout, "{}", out).ok();
                    stdout.flush().ok();
                }
            }
            Err(e) => {
                eprintln!("[splatdb] stdin error: {}", e);
                break;
            }
        }
    }

    eprintln!("[splatdb] MCP server shutting down (stdin closed).");
}
