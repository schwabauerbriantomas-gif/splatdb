//! Minimal HTTP API server for Python bindings
//!
//! Endpoints:
//!   POST /store   - Store a memory (text + optional metadata)
//!   POST /search  - Search memories (query text, top-k)
//!   POST /status  - Get store stats
//!   GET  /health  - Health check

use axum::{
    extract::State,
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;
use tower_http::cors::CorsLayer;

use crate::splats::SplatStore;
use crate::config::SplatDBConfig;

#[derive(Clone)]
pub struct AppState {
    pub store: Arc<Mutex<SplatStore>>,
    pub next_id: Arc<Mutex<usize>>,
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
        version: "2.1.0".into(),
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
    let dim = {
        let store = state.store.lock().await;
        store.get_statistics().embedding_dim
    };
    
    let embedding = req.embedding.unwrap_or_else(|| {
        simple_hash_embedding(&req.text, dim)
    });
    
    let mut store = state.store.lock().await;
    let arr = Array2::from_shape_vec((1, dim), embedding.clone())
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Bad embedding shape: {}", e)))?;
    
    let added = store.add_splat(&arr);
    if !added {
        return Err((StatusCode::INSUFFICIENT_STORAGE, "Store full".into()));
    }
    
    // Build index after insert for consistency
    store.build_index();
    
    let mut id_counter = state.next_id.lock().await;
    *id_counter += 1;
    let id = req.id.unwrap_or_else(|| format!("mem_{}", *id_counter));
    
    Ok((StatusCode::OK, Json(StoreResponse {
        id,
        status: "stored".into(),
    })))
}

async fn search_memories(
    State(state): State<AppState>,
    Json(req): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, (StatusCode, String)> {
    let (_store, results) = {
        let store = state.store.lock().await;
        let dim = store.get_statistics().embedding_dim;
        let k = req.top_k.unwrap_or(10).min(store.n_active());
        
        if store.n_active() == 0 {
            return Ok(Json(SearchResponse { results: vec![] }));
        }
        
        let embedding = req.embedding.unwrap_or_else(|| {
            simple_hash_embedding(&req.query, dim)
        });
        
        let query = Array1::from_vec(embedding);
        let neighbors = store.find_neighbors(&query.view(), k);
        
        let search_results: Vec<SearchResult> = neighbors
            .into_iter()
            .map(|n| SearchResult {
                index: n.index,
                score: n.distance,
                metadata: None,
            })
            .collect();
        
        (store, search_results)
    };
    
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

pub async fn run_server(port: u16) -> anyhow::Result<()> {
    let config = SplatDBConfig::default();
    let store = SplatStore::new(config);
    
    let state = AppState {
        store: Arc::new(Mutex::new(store)),
        next_id: Arc::new(Mutex::new(0)),
    };

    let app = Router::new()
        .route("/health", get(health))
        .route("/status", post(status))
        .route("/store", post(store_memory))
        .route("/search", post(search_memories))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let addr = format!("127.0.0.1:{}", port);
    println!("SplatDB server listening on {}", addr);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    Ok(axum::serve(listener, app).await?)
}
