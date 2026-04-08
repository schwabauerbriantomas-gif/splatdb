//! Minimal HTTP API server for Python bindings
//!
//! Endpoints:
//!   POST /store   - Store a memory (text + optional metadata)
//!   POST /search  - Search memories (query text, top-k)
//!   POST /status  - Get store stats
//!   GET  /health  - Health check
//!
//! Security:
//!   - API key authentication via SPLATDB_API_KEY env var (optional)
//!   - CORS restricted to localhost by default
//!   - Input size limits enforced

use axum::{
    extract::{Request, State},
    http::{HeaderMap, StatusCode},
    middleware::{self, Next},
    routing::{get, post},
    Json, Router,
};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::sync::Mutex;
use tower_http::cors::CorsLayer;

use crate::config::SplatDBConfig;
use crate::splats::SplatStore;

// ─── Security constants ──────────────────────────────────
const MAX_TEXT_LEN: usize = 100_000; // 100KB max text input
const MAX_EMBEDDING_DIM: usize = 8192; // Max embedding dimension
const MAX_TOP_K: usize = 1000;

#[derive(Clone)]
pub struct AppState {
    pub store: Arc<Mutex<SplatStore>>,
    pub next_id: Arc<AtomicUsize>,
}

#[derive(Deserialize)]
pub struct StoreRequest {
    pub text: String,
    pub category: Option<String>,
    pub id: Option<String>,
    #[serde(default)]
    pub embedding: Option<Vec<f32>>,
}

#[derive(Deserialize)]
pub struct SearchRequest {
    pub query: String,
    #[serde(default)]
    pub embedding: Option<Vec<f32>>,
    pub top_k: Option<usize>,
}

#[derive(Serialize)]
pub struct StoreResponse {
    pub id: String,
    pub status: String,
}

#[derive(Serialize)]
pub struct SearchResponse {
    pub results: Vec<SearchResult>,
}

#[derive(Serialize)]
pub struct SearchResult {
    pub index: usize,
    pub score: f32,
    pub metadata: Option<String>,
}

#[derive(Serialize)]
pub struct StatusResponse {
    pub n_active: usize,
    pub max_splats: usize,
    pub dimension: usize,
    pub has_hnsw: bool,
    pub has_lsh: bool,
    pub has_quantization: bool,
    pub has_semantic_memory: bool,
}

#[derive(Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
}

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".into(),
        version: env!("CARGO_PKG_VERSION").into(),
    })
}

async fn status(State(state): State<AppState>) -> Json<StatusResponse> {
    let store = state.store.lock().await;
    let stats = store.get_statistics();
    Json(StatusResponse {
        n_active: store.n_active(),
        max_splats: store.max_splats(),
        dimension: stats.embedding_dim,
        has_hnsw: store.has_hnsw(),
        has_lsh: store.has_lsh(),
        has_quantization: store.has_quantization(),
        has_semantic_memory: store.has_semantic_memory(),
    })
}

async fn store_memory(
    State(state): State<AppState>,
    Json(req): Json<StoreRequest>,
) -> Result<(StatusCode, Json<StoreResponse>), (StatusCode, String)> {
    // Input validation
    if req.text.len() > MAX_TEXT_LEN {
        return Err((
            StatusCode::PAYLOAD_TOO_LARGE,
            format!("Text exceeds max length ({MAX_TEXT_LEN})"),
        ));
    }

    let mut store = state.store.lock().await;
    let dim = store.get_statistics().embedding_dim;

    // Validate user-supplied embedding dimension
    if let Some(ref emb) = req.embedding {
        if emb.len() != dim {
            return Err((
                StatusCode::BAD_REQUEST,
                format!(
                    "Embedding dimension mismatch: expected {dim}, got {}",
                    emb.len()
                ),
            ));
        }
        if emb.len() > MAX_EMBEDDING_DIM {
            return Err((
                StatusCode::BAD_REQUEST,
                format!("Embedding dimension exceeds max ({MAX_EMBEDDING_DIM})"),
            ));
        }
    }

    let embedding = req
        .embedding
        .unwrap_or_else(|| simple_hash_embedding(&req.text, dim));

    let arr = Array2::from_shape_vec((1, dim), embedding)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Bad embedding shape: {e}")))?;

    let added = store.add_splat(&arr);
    if !added {
        return Err((StatusCode::INSUFFICIENT_STORAGE, "Store full".into()));
    }

    // Incrementally insert the new vector into HNSW (no full rebuild)
    store.hnsw_sync_incremental();

    let id_num = state.next_id.fetch_add(1, Ordering::SeqCst);
    let id = req.id.unwrap_or_else(|| format!("mem_{id_num}"));

    Ok((
        StatusCode::OK,
        Json(StoreResponse {
            id,
            status: "stored".into(),
        }),
    ))
}

async fn search_memories(
    State(state): State<AppState>,
    Json(req): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, (StatusCode, String)> {
    // Input validation
    if req.query.len() > MAX_TEXT_LEN {
        return Err((
            StatusCode::PAYLOAD_TOO_LARGE,
            format!("Query exceeds max length ({MAX_TEXT_LEN})"),
        ));
    }

    let store = state.store.lock().await;
    let dim = store.get_statistics().embedding_dim;

    if store.n_active() == 0 {
        return Ok(Json(SearchResponse { results: vec![] }));
    }

    let k = req.top_k.unwrap_or(10).min(store.n_active()).min(MAX_TOP_K);

    // Validate user-supplied embedding dimension
    if let Some(ref emb) = req.embedding {
        if emb.len() != dim {
            return Err((
                StatusCode::BAD_REQUEST,
                format!(
                    "Embedding dimension mismatch: expected {dim}, got {}",
                    emb.len()
                ),
            ));
        }
    }

    let embedding = req
        .embedding
        .unwrap_or_else(|| simple_hash_embedding(&req.query, dim));
    let query = Array1::from_vec(embedding);
    let neighbors = store.find_neighbors(&query.view(), k);

    let results: Vec<SearchResult> = neighbors
        .into_iter()
        .map(|n| SearchResult {
            index: n.index,
            score: n.distance,
            metadata: None,
        })
        .collect();

    Ok(Json(SearchResponse { results }))
}

/// Deterministic hash-based pseudo-embedding for text.
/// NOT a real embedding - placeholder for structure.
/// Replace with actual model (DINOv2, BGE, etc.) in production.
fn simple_hash_embedding(text: &str, dim: usize) -> Vec<f32> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut result = Vec::with_capacity(dim);
    for i in 0..dim {
        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        i.hash(&mut hasher);
        let raw = hasher.finish() as f32;
        let val = (raw / u32::MAX as f32) * 2.0 - 1.0;
        result.push(val);
    }
    // Normalize
    let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in result.iter_mut() {
            *x /= norm;
        }
    }
    result
}

// ─── API Key Authentication Middleware ────────────────────
async fn api_key_auth(
    headers: HeaderMap,
    request: Request,
    next: Next,
) -> Result<axum::response::Response, StatusCode> {
    // Skip auth for health endpoint
    if request.uri().path() == "/health" {
        return Ok(next.run(request).await);
    }

    // If no SPLATDB_API_KEY env var set, skip auth (localhost dev mode)
    let Ok(expected_key) = std::env::var("SPLATDB_API_KEY") else {
        return Ok(next.run(request).await);
    };

    let provided = headers
        .get("Authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "));

    match provided {
        Some(key) if key == expected_key => Ok(next.run(request).await),
        _ => Err(StatusCode::UNAUTHORIZED),
    }
}

pub async fn run_server(addr: &str, port: u16) -> anyhow::Result<()> {
    let config = SplatDBConfig::default();
    let store = SplatStore::new(config);

    let state = AppState {
        store: Arc::new(Mutex::new(store)),
        next_id: Arc::new(AtomicUsize::new(0)),
    };

    // CORS: restrict to localhost by default
    let cors = CorsLayer::new()
        .allow_origin([
            "http://localhost".parse().unwrap(),
            "http://127.0.0.1".parse().unwrap(),
        ])
        .allow_methods([axum::http::Method::GET, axum::http::Method::POST])
        .allow_headers([
            axum::http::header::CONTENT_TYPE,
            axum::http::header::AUTHORIZATION,
        ]);

    let app = Router::new()
        .route("/health", get(health))
        .route("/status", post(status))
        .route("/store", post(store_memory))
        .route("/search", post(search_memories))
        .layer(cors)
        .layer(middleware::from_fn(api_key_auth))
        .with_state(state);

    let bind_addr = format!("{addr}:{port}");
    println!("SplatDB server listening on {bind_addr}");
    let listener = tokio::net::TcpListener::bind(&bind_addr).await?;
    Ok(axum::serve(listener, app).await?)
}
