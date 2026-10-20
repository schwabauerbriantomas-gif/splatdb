//! Minimal HTTP API server for Python bindings
//!
//! Endpoints:
//!   POST /store   - Store a memory (text + optional metadata)
//!   POST /search  - Search memories (query text, top-k, `?rescore=gaussian`)
//!   POST /status  - Get store stats
//!   GET  /health  - Health check
//!
//! Architecture (R9 audit):
//!   - `/search` NEVER takes the global store mutex: it enqueues into an mpsc
//!     channel and a single `batch_worker` task coalesces requests within a
//!     BATCH_WINDOW_MS window (up to MAX_BATCH) and takes the mutex ONCE per
//!     batch (R9 §1 issue 1: the global lock no longer serializes searches).
//!   - CPU path: `find_neighbors_fused` (splats.rs) — HNSW-first sub-linear
//!     search with exact re-rank. The server config enables HNSW (the old
//!     `SplatsDBConfig::default()` shipped it disabled → every query was a
//!     linear O(N·D) scan).
//!   - GPU path (when built with `--features cuda` and a device is present):
//!     one `gpu::batch_search` per batch over a cached flat snapshot of the
//!     dataset, refreshed only when `n_active` changes (R9 §2).
//!   - `simple_hash_embedding` hashes the text ONCE and derives the D
//!     dimensions from a SplitMix64 PRNG (O(1) hashes instead of O(D)).
//!
//! Security:
//!   - API key authentication via SPLATSDB_API_KEY env var (optional)
//!   - CORS restricted to localhost by default
//!   - Input size limits enforced

use axum::{
    extract::{Query, Request, State},
    http::{HeaderMap, StatusCode},
    middleware::{self, Next},
    routing::{get, post},
    Json, Router,
};
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, oneshot, Mutex};
use tower_http::cors::CorsLayer;

use crate::config::SplatsDBConfig;
use crate::splats::SplatStore;

// ─── Security constants ──────────────────────────────────
const MAX_TEXT_LEN: usize = 100_000; // 100KB max text input
const MAX_EMBEDDING_DIM: usize = 8192; // Max embedding dimension
const MAX_TOP_K: usize = 1000;

// ─── Micro-batching constants (R9 §2) ────────────────────
/// Coalescing window: requests arriving within this window are served by a
/// single batched search instead of one lock acquisition + scan per request.
const BATCH_WINDOW_MS: u64 = 2;
/// Hard cap on queries coalesced into one batch.
const MAX_BATCH: usize = 64;
/// Channel capacity for pending search requests (backpressure beyond this).
const BATCH_CHANNEL_CAP: usize = 256;

#[derive(Clone)]
pub struct AppState {
    pub store: Arc<Mutex<SplatStore>>,
    pub next_id: Arc<AtomicUsize>,
    /// Sender side of the search micro-batching channel. The `/search` hot
    /// path pushes here and never takes the global store mutex; the single
    /// `batch_worker` task takes it once per batch.
    pub batch_tx: mpsc::Sender<BatchRequest>,
    /// Number of batches served by the worker (observability / tests).
    pub batches_served: Arc<AtomicUsize>,
    /// Índice exacto en GPU (kernel PTX l2_topk). topk_search es &self → se
    /// comparte sin Mutex. Presente solo con serve --gpu y feature cuda.
    #[cfg(feature = "cuda")]
    pub gpu: Option<Arc<crate::gpu::cuda_kernel::GpuIndex>>,
    /// Campo splat (celdas con radio) para /field-search exacto. None si no
    /// hay field/ persistido en el data-dir.
    pub field: Option<Arc<FieldData>>,
}

/// A single search request queued for batch execution. The reply is delivered
/// on a private oneshot channel.
pub struct BatchRequest {
    query: Vec<f32>,
    k: usize,
    /// `?rescore=gaussian` was requested.
    rescore_gaussian: bool,
    resp: oneshot::Sender<Result<Vec<SearchResult>, String>>,
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

/// Optional query-string parameters for `/search`.
///
/// `?rescore=gaussian` re-ranks the top-k under the exact Gaussian splat
/// score `alpha_i · exp(-kappa_i · ‖q − mu_i‖²)` (higher is better — the
/// response `score` field carries THIS value instead of L2 distance; see
/// `SplatStore::find_neighbors_gaussian`). Any other value is ignored.
#[derive(Deserialize)]
pub struct SearchQuery {
    pub rescore: Option<String>,
}

#[derive(Serialize)]
pub struct StoreResponse {
    pub id: String,
    pub status: String,
}

#[derive(Serialize)]
pub struct SearchResponse {
    pub results: Vec<SearchResult>,
    /// "gpu" cuando la query corrió por el kernel PTX exacto (serve --gpu).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub backend: Option<String>,
}

#[derive(Serialize, Clone)]
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

    let dim = {
        let store = state.store.lock().await;
        store.get_statistics().embedding_dim
    }; // lock dropped here — embedding computation doesn't need the mutex

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

    let id_num = state.next_id.fetch_add(1, Ordering::SeqCst);
    let id = req.id.unwrap_or_else(|| format!("mem_{id_num}"));

    // Wrap CPU-intensive work in spawn_blocking to avoid blocking the tokio runtime
    let store = state.store.clone();
    let added = tokio::task::spawn_blocking(move || {
        let mut store = store.blocking_lock();
        let added = store.add_splat(&arr);
        if added {
            // Incrementally insert the new vector into HNSW (no full rebuild)
            store.hnsw_sync_incremental();
        }
        added
    })
    .await
    .map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("task error: {e}"),
        )
    })?;

    if !added {
        return Err((StatusCode::INSUFFICIENT_STORAGE, "Store full".into()));
    }

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
    Query(params): Query<SearchQuery>,
    Json(req): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, (StatusCode, String)> {
    // Input validation
    if req.query.len() > MAX_TEXT_LEN {
        return Err((
            StatusCode::PAYLOAD_TOO_LARGE,
            format!("Query exceeds max length ({MAX_TEXT_LEN})"),
        ));
    }

    let (dim, n_active) = {
        let store = state.store.lock().await;
        let dim = store.get_statistics().embedding_dim;
        let n = store.n_active();
        (dim, n)
    }; // lock dropped — the hot path below never re-acquires it

    // Validate user-supplied embedding dimension BEFORE the empty-store
    // early-return so malformed requests are always rejected with 400.
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

    if n_active == 0 {
        return Ok(Json(SearchResponse { results: vec![], backend: None }));
    }

    let k = req.top_k.unwrap_or(10).max(1).min(n_active).min(MAX_TOP_K);

    let had_embedding = req.embedding.is_some();
    let embedding = req
        .embedding
        .unwrap_or_else(|| simple_hash_embedding(&req.query, dim));

    let rescore_gaussian = matches!(params.rescore.as_deref(), Some("gaussian"));

    // ── GPU exact fast path (serve --gpu): kernel PTX fused l2+topk.
    // Exacto, sin store mutex (índice compartido &self). Rescore gaussiano y
    // query de texto (hash embedding) siguen por el camino CPU.
    #[cfg(feature = "cuda")]
    if let Some(gpu) = &state.gpu {
        if had_embedding && !rescore_gaussian {
            if let Some(mut per_query) = gpu.topk_search(&embedding, 1, k, "l2") {
                if let Some((idxs, dists)) = per_query.pop() {
                    let mut pairs: Vec<(usize, f32)> =
                        idxs.into_iter().zip(dists).collect();
                    pairs.sort_by(|a, b| {
                        a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                    });
                    let results = pairs
                        .into_iter()
                        .map(|(i, d)| SearchResult { index: i, score: d, metadata: None })
                        .collect();
                    return Ok(Json(SearchResponse {
                        results,
                        backend: Some("gpu".into()),
                    }));
                }
            }
            eprintln!("[splatsdb] GPU topk devolvió None; query cae a CPU");
        }
    }

    // ── Micro-batched path (R9 §2): enqueue and wait on a private oneshot.
    // The store mutex is taken only by the batch worker, once per batch.
    let (tx, rx) = oneshot::channel();
    state
        .batch_tx
        .send(BatchRequest {
            query: embedding,
            k,
            rescore_gaussian,
            resp: tx,
        })
        .await
        .map_err(|_| {
            (
                StatusCode::SERVICE_UNAVAILABLE,
                "search worker unavailable".into(),
            )
        })?;

    let results = rx
        .await
        .map_err(|_| {
            (
                StatusCode::SERVICE_UNAVAILABLE,
                "search worker dropped request".into(),
            )
        })?
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;

    Ok(Json(SearchResponse { results, backend: None }))
}

// ─── Campo splat (celdas con radio): /field-search exacto ────────────────
// Geometría correcta del campo (veredicto bench-latent): la energía
// alpha·exp(−kappa·d²) NO es una cota y destruye el ranking en 768d; la
// desigualdad triangular d(q,x) ≥ d(q,mu_c) − R_c SÍ lo es → sweep exacto.

/// Crudos + splats + membresías CSR + radios, persistidos por
/// `field_cmds::persist_field` tras cada ingest.
pub struct FieldData {
    pub raw: Array2<f32>,
    pub splats: Array2<f32>,
    pub offsets: Vec<u64>,
    pub ids: Vec<u32>,
    pub radii: Vec<f32>,
}

fn load_field(data_dir: &str) -> Option<FieldData> {
    use std::io::Read;
    let dir = std::path::Path::new(data_dir).join("field");
    if !dir.join("members.idx").exists() || !dir.join("radii.bin").exists() {
        return None;
    }
    let mut f = std::fs::File::open(dir.join("members.idx")).ok()?;
    let mut hdr = [0u8; 16];
    f.read_exact(&mut hdr).ok()?;
    let n = u64::from_le_bytes(hdr[..8].try_into().ok()?) as usize;
    let k = u64::from_le_bytes(hdr[8..].try_into().ok()?) as usize;
    let mut ob = vec![0u8; (k + 1) * 8];
    f.read_exact(&mut ob).ok()?;
    let mut offsets = vec![0u64; k + 1];
    for (i, c) in ob.chunks_exact(8).enumerate() {
        offsets[i] = u64::from_le_bytes(c.try_into().ok()?);
    }
    let mut ib = vec![0u8; n * 4];
    f.read_exact(&mut ib).ok()?;
    let mut ids = vec![0u32; n];
    for (i, c) in ib.chunks_exact(4).enumerate() {
        ids[i] = u32::from_le_bytes(c.try_into().ok()?);
    }
    let mut f = std::fs::File::open(dir.join("radii.bin")).ok()?;
    let mut k8 = [0u8; 8];
    f.read_exact(&mut k8).ok()?;
    let kr = u64::from_le_bytes(k8) as usize;
    let mut rb = vec![0u8; kr * 4];
    f.read_exact(&mut rb).ok()?;
    let mut radii = vec![0f32; kr];
    for (i, c) in rb.chunks_exact(4).enumerate() {
        radii[i] = f32::from_le_bytes(c.try_into().ok()?);
    }
    let persist = crate::storage::persistence::SplatsDBPersistence::with_backend(
        data_dir,
        crate::storage::persistence::StorageBackend::Sqlite,
        false,
    )
    .ok()?;
    let raw = persist.load_vectors("raw").ok()??;
    let splats = persist.load_vectors("default").ok()??;
    if raw.nrows() != n || splats.nrows() != k || kr != k {
        eprintln!("[splatsdb] field/ desincronizado; /field-search deshabilitado");
        return None;
    }
    Some(FieldData { raw, splats, offsets, ids, radii })
}

fn field_sweep(field: &FieldData, q: &[f32], kq: usize) -> serde_json::Value {
    let t0 = std::time::Instant::now();
    let qv = ndarray::arr1(q);
    let k = field.splats.nrows();
    let n = field.raw.nrows();
    let d_mu: ndarray::Array1<f32> = field
        .splats
        .map_axis(ndarray::Axis(1), |r| {
            let d = r.to_owned() - &qv;
            d.dot(&d).max(0.0).sqrt()
        });
    // cota por cluster: un cluster entero es podable si (d(q,mu)−R)² > kth
    let lb: Vec<f32> = (0..k).map(|c| (d_mu[c] - field.radii[c]).max(0.0)).collect();
    let mut order: Vec<usize> = (0..k).collect();
    order.sort_by(|a, b| d_mu[*a].partial_cmp(&d_mu[*b]).unwrap_or(std::cmp::Ordering::Equal));

    let mut top: Vec<(f32, u32)> = Vec::with_capacity(kq + 8);
    let mut kth = f32::INFINITY;
    // kth SOLO acota cuando el top-k está completo (cota inválida con top parcial)
    let mut visited = 0usize;
    let mut scanned = 0usize;
    for &c in &order {
        let lb_c = lb[c] * lb[c];
        if !(lb_c < kth || top.len() < kq) {
            continue;
        }
        let (lo, hi) = (field.offsets[c] as usize, field.offsets[c + 1] as usize);
        if hi == lo {
            continue;
        }
        visited += 1;
        scanned += hi - lo;
        for &id in &field.ids[lo..hi] {
            let row = field.raw.row(id as usize);
            let diff = row.to_owned() - &qv;
            let d2 = diff.dot(&diff);
            if top.len() < kq || d2 < kth {
                top.push((d2, id));
                top.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
                if top.len() > kq {
                    top.truncate(kq);
                }
                kth = if top.len() >= kq { top[top.len() - 1].0 } else { f32::INFINITY };
            }
        }
    }
    let pct = |x: f64| (x * 10000.0).round() / 10000.0;
    serde_json::json!({
        "backend": "field_exact_bounds",
        "k": kq,
        "n_vectors": n,
        "n_splats": k,
        "clusters_visited": visited,
        "cluster_frac": pct(visited as f64 / k as f64),
        "vectors_scanned": scanned,
        "scan_frac": pct(scanned as f64 / n as f64),
        "time_ms": t0.elapsed().as_millis() as u64,
        "results": top.iter()
            .map(|(d, i)| serde_json::json!({"index": *i as usize, "score": *d}))
            .collect::<Vec<_>>(),
    })
}

#[derive(Deserialize)]
pub struct FieldSearchRequest {
    pub embedding: Vec<f32>,
    pub top_k: Option<usize>,
}

async fn field_search_handler(
    State(state): State<AppState>,
    Json(req): Json<FieldSearchRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let Some(field) = &state.field else {
        return Err((
            StatusCode::NOT_FOUND,
            "no hay field/ en este data-dir (correr ingest con splatdb >= 2.6.1)".into(),
        ));
    };
    if req.embedding.len() != field.splats.ncols() {
        return Err((
            StatusCode::BAD_REQUEST,
            format!(
                "dim esperada {}, llegada {}",
                field.splats.ncols(),
                req.embedding.len()
            ),
        ));
    }
    let k = req.top_k.unwrap_or(10).max(1).min(1000);
    Ok(Json(field_sweep(field, &req.embedding, k)))
}

// ─── Batch worker (R9 §2) ─────────────────────────────────

/// Cached flat copy of the active dataset for the GPU batch path.
/// Refreshed only when `n_active` changes, so the store mutex is held just
/// long enough to copy — never during the search itself.
struct DatasetSnapshot {
    /// Dataset version (= n_active at snapshot time; the server only appends
    /// via /store, so this is a sufficient staleness check).
    version: usize,
    n: usize,
    dim: usize,
    /// Row-major [n, dim] flat data.
    data: Vec<f32>,
    /// Per-splat amplitudes [n] (gaussian rescore input).
    alpha: Vec<f32>,
    /// Per-splat concentrations [n] (gaussian rescore input).
    kappa: Vec<f32>,
}

/// Coalescing worker: receives search requests, groups everything that
/// arrives within `BATCH_WINDOW_MS` (up to `MAX_BATCH`) and serves it as a
/// single batch — one lock acquisition and one (GPU) or a few (CPU/HNSW)
/// searches instead of one lock + scan per request.
async fn batch_worker(
    mut rx: mpsc::Receiver<BatchRequest>,
    store: Arc<Mutex<SplatStore>>,
    batches_served: Arc<AtomicUsize>,
) {
    // Host-side dataset snapshot cache for the GPU path.
    let mut snapshot: Option<DatasetSnapshot> = None;

    while let Some(first) = rx.recv().await {
        let mut batch = Vec::with_capacity(MAX_BATCH.min(16));
        batch.push(first);

        // Coalescing window: drain until the deadline or MAX_BATCH.
        let deadline = tokio::time::Instant::now() + Duration::from_millis(BATCH_WINDOW_MS);
        while batch.len() < MAX_BATCH {
            match tokio::time::timeout_at(deadline, rx.recv()).await {
                Ok(Some(r)) => batch.push(r),
                Ok(None) | Err(_) => break, // channel closed or window elapsed
            }
        }

        batches_served.fetch_add(1, Ordering::Relaxed);
        serve_batch(batch, &store, &mut snapshot).await;
    }
}

/// Execute one coalesced batch and deliver each request's slice of the
/// results on its oneshot channel. Any request that arrives on a channel
/// that was already dropped is silently discarded.
async fn serve_batch(
    batch: Vec<BatchRequest>,
    store: &Arc<Mutex<SplatStore>>,
    snapshot: &mut Option<DatasetSnapshot>,
) {
    let queries: Vec<Vec<f32>> = batch.iter().map(|b| b.query.clone()).collect();
    let ks: Vec<usize> = batch.iter().map(|b| b.k).collect();
    let rescore: Vec<bool> = batch.iter().map(|b| b.rescore_gaussian).collect();
    let use_gpu = crate::gpu::is_cuda_available();

    let store = store.clone();
    // The snapshot cache must survive the 'static spawn_blocking boundary:
    // take it in, get it back with the results.
    let mut snap = snapshot.take();

    let outcome = tokio::task::spawn_blocking(move || {
        let result = if use_gpu {
            gpu_batch_path(&store, &mut snap, &queries, &ks, &rescore)
        } else {
            cpu_fused_batch(&store, &queries, &ks, &rescore)
        };
        (result, snap)
    })
    .await;

    match outcome {
        Ok((Ok(per_query), snap_back)) => {
            *snapshot = snap_back;
            for (b, r) in batch.into_iter().zip(per_query) {
                let _ = b.resp.send(Ok(r));
            }
        }
        Ok((Err(e), snap_back)) => {
            *snapshot = snap_back;
            for b in batch {
                let _ = b.resp.send(Err(e.clone()));
            }
        }
        Err(join_err) => {
            // spawn_blocking panicked — worker state (snapshot) is lost, which
            // is safe: it is only a cache.
            for b in batch {
                let _ = b
                    .resp
                    .send(Err(format!("batch worker task failed: {join_err}")));
            }
        }
    }
}

/// CPU path: single lock acquisition for the whole batch, then
/// `find_neighbors_fused` per query — HNSW-first sub-linear search with
/// exact re-rank (falls back to linear scan internally when no approximate
/// backend is available).
fn cpu_fused_batch(
    store: &Arc<Mutex<SplatStore>>,
    queries: &[Vec<f32>],
    ks: &[usize],
    rescore: &[bool],
) -> Result<Vec<Vec<SearchResult>>, String> {
    // The ONE lock acquisition per batch (R9 §1 issue 1).
    let mut store = store.blocking_lock();

    let per_query = queries
        .iter()
        .zip(ks.iter())
        .zip(rescore.iter())
        .map(|((q, &k), &want_rescore)| {
            let view = ndarray::ArrayView1::from(&q[..]);
            // ?rescore=gaussian → exact Gaussian re-rank of the top-k (B2);
            // otherwise the fused HNSW path returns L2 distances.
            let neighbors = if want_rescore {
                store.find_neighbors_gaussian(&view, k)
            } else {
                store.find_neighbors_fused(&view, k)
            };
            neighbors
                .into_iter()
                .map(|n| SearchResult {
                    index: n.index,
                    // rescore=gaussian → `score` is the Gaussian splat score
                    // (higher is better); otherwise it is the exact L2
                    // distance (lower is better).
                    score: if want_rescore { n.score } else { n.distance },
                    metadata: None,
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    // Feed access frequencies for tiering (R9 §5.3): O(Q·k), near-free, and
    // we already hold the lock.
    let touched: Vec<usize> = per_query
        .iter()
        .flat_map(|r| r.iter().map(|s| s.index))
        .collect();
    store.touch(&touched);

    Ok(per_query)
}

/// GPU path: refresh the flat snapshot only when the dataset changed, then a
/// single `gpu::batch_search` for the whole batch (GPU top-k kernel with CPU
/// fallback inside). The store mutex is never held during the search.
fn gpu_batch_path(
    store: &Arc<Mutex<SplatStore>>,
    snapshot: &mut Option<DatasetSnapshot>,
    queries: &[Vec<f32>],
    ks: &[usize],
    rescore: &[bool],
) -> Result<Vec<Vec<SearchResult>>, String> {
    // ── Snapshot refresh under a brief lock ──
    {
        let s = store.blocking_lock();
        let version = s.n_active();
        let stale = match snapshot.as_ref() {
            Some(snap) => snap.version != version,
            None => true,
        };
        if version == 0 {
            *snapshot = None;
        } else if stale {
            if let Some(mu) = s.get_mu() {
                let dim = mu.ncols();
                let data = mu.iter().copied().collect::<Vec<f32>>();
                let alpha = s
                    .get_alpha()
                    .map(|a| a.iter().copied().collect::<Vec<f32>>())
                    .unwrap_or_else(|| vec![1.0f32; version]);
                let kappa = s
                    .get_kappa()
                    .map(|k| k.iter().copied().collect::<Vec<f32>>())
                    .unwrap_or_else(|| vec![10.0f32; version]);
                *snapshot = Some(DatasetSnapshot {
                    version,
                    n: version,
                    dim,
                    data,
                    alpha,
                    kappa,
                });
            }
        }
    } // lock released — the search itself never holds the store mutex

    let snap = snapshot
        .as_ref()
        .ok_or_else(|| "store is empty".to_string())?;

    let n_queries = queries.len();

    // ── Split the batch: gaussian-rescored queries go through
    // `gpu::gaussian_topk_scores` (B1 wrapper: GPU kernel + exact CPU
    // fallback, returns score-sorted top-k directly), plain L2 queries
    // through `gpu::batch_search` (top-k kernel, squared-L2 → sqrt).
    let gauss_flat: Vec<f32> = queries
        .iter()
        .zip(rescore.iter())
        .filter(|(_, &r)| r)
        .flat_map(|(q, _)| q.iter().copied())
        .collect();
    let gauss_ks: Vec<usize> = ks
        .iter()
        .zip(rescore.iter())
        .filter(|(_, &r)| r)
        .map(|(&k, _)| k)
        .collect();
    let n_gauss = gauss_ks.len();

    let l2_flat: Vec<f32> = queries
        .iter()
        .zip(rescore.iter())
        .filter(|(_, &r)| !r)
        .flat_map(|(q, _)| q.iter().copied())
        .collect();
    let l2_ks: Vec<usize> = ks
        .iter()
        .zip(rescore.iter())
        .filter(|(_, &r)| !r)
        .map(|(&k, _)| k)
        .collect();
    let n_l2 = l2_ks.len();

    let gauss_out = if n_gauss > 0 {
        let kmax_g = gauss_ks.iter().copied().max().unwrap_or(1).min(snap.n);
        crate::gpu::gaussian_topk_scores(
            &gauss_flat,
            &snap.data,
            &snap.alpha,
            &snap.kappa,
            snap.n,
            snap.dim,
            n_gauss,
            kmax_g,
        )
        .ok_or_else(|| "gaussian_topk_scores failed".to_string())?
    } else {
        Vec::new()
    };

    let l2_out = if n_l2 > 0 {
        let kmax_l = l2_ks.iter().copied().max().unwrap_or(1).min(snap.n);
        crate::gpu::batch_search(&l2_flat, n_l2, &snap.data, snap.n, snap.dim, kmax_l, "l2")
    } else {
        Vec::new()
    };

    // ── Reassemble rows in original request order.
    let mut per_query = Vec::with_capacity(n_queries);
    let mut g = 0usize; // cursor into gauss_out
    let mut l = 0usize; // cursor into l2_out
    for (i, (&k, &want_rescore)) in ks.iter().zip(rescore.iter()).enumerate() {
        let row: Vec<SearchResult> = if want_rescore {
            let src = &gauss_out[g];
            g += 1;
            src.iter()
                .filter(|&&(ix, _)| ix < snap.n)
                .take(k)
                .map(|&(ix, score)| SearchResult {
                    index: ix,
                    // Gaussian score: higher is better.
                    score,
                    metadata: None,
                })
                .collect()
        } else {
            let (idxs, dists) = &l2_out[l];
            l += 1;
            idxs.iter()
                .zip(dists.iter())
                .filter(|(&ix, _)| ix < snap.n)
                .take(k)
                .map(|(&ix, &d)| SearchResult {
                    index: ix,
                    // batch_search reports squared L2; expose true L2 to
                    // match the CPU path (find_neighbors sqrt'd distances).
                    score: d.sqrt(),
                    metadata: None,
                })
                .collect()
        };
        per_query.push(row);
        let _ = i; // index kept for clarity in debug
    }

    // Feed access frequencies for tiering under a second brief lock.
    let touched: Vec<usize> = per_query
        .iter()
        .flat_map(|r| r.iter().map(|s| s.index))
        .collect();
    store.blocking_lock().touch(&touched);

    Ok(per_query)
}

// ─── Embedding ────────────────────────────────────────────

/// Deterministic hash-based pseudo-embedding for text.
/// NOT a real embedding - placeholder for structure.
/// Replace with actual model (DINOv2, BGE, etc.) in production.
///
/// O(1) hashes: the text is hashed ONCE and the D dimensions are derived
/// from a SplitMix64 PRNG seeded with that hash (previously the text was
/// re-hashed D times — O(D) hasher passes — R9 §5.1).
fn simple_hash_embedding(text: &str, dim: usize) -> Vec<f32> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    text.hash(&mut hasher);
    // Odd, non-zero seed so SplitMix64 output is well distributed from step 0.
    let mut state = hasher.finish() | 1;

    let mut result = Vec::with_capacity(dim);
    for _ in 0..dim {
        // SplitMix64 finalizer step
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        result.push((z as f32 / u64::MAX as f32) * 2.0 - 1.0);
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

    // If no SPLATSDB_API_KEY env var set, skip auth (localhost dev mode)
    let Ok(expected_key) = std::env::var("SPLATSDB_API_KEY") else {
        return Ok(next.run(request).await);
    };

    let provided = headers
        .get("Authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "));

    match provided {
        Some(key) if constant_time_eq(key, &expected_key) => Ok(next.run(request).await),
        _ => Err(StatusCode::UNAUTHORIZED),
    }
}

/// Constant-time string comparison to prevent timing attacks on API keys.
fn constant_time_eq(a: &str, b: &str) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut result: u8 = 0;
    for (x, y) in a.as_bytes().iter().zip(b.as_bytes().iter()) {
        result |= x ^ y;
    }
    result == 0
}

/// Load vectors from the `default` shard of `data_dir` into `store`.
///
/// Returns the number of vectors loaded. Mirrors the CLI's
/// `load_or_create_store` behaviour (src/cli/helpers.rs), keeping `serve`
/// consistent with `status`/`search` over the same data-dir.
fn load_default_shard(store: &mut SplatStore, data_dir: &str) -> anyhow::Result<usize> {
    let persist = crate::storage::persistence::SplatsDBPersistence::with_backend(
        data_dir,
        crate::storage::persistence::StorageBackend::Sqlite,
        false,
    )
    .map_err(|e| anyhow::anyhow!("persistence init failed: {e}"))?;
    match persist.load_vectors("default")? {
        Some(vectors) => {
            if !store.add_splat(&vectors) {
                anyhow::bail!(
                    "store capacity exceeded (max_splats={}) while loading {} vectors",
                    store.max_splats(),
                    vectors.nrows()
                );
            }
            Ok(vectors.nrows())
        }
        None => Ok(0),
    }
}

pub async fn run_server(
    addr: &str,
    port: u16,
    config: SplatsDBConfig,
    data_dir: &str,
    gpu_enabled: bool,
) -> anyhow::Result<()> {
    let config = server_config(config);
    let hnsw_on = config.enable_hnsw;
    let mut store = SplatStore::new(config);

    // Auto-load the `default` shard from `data_dir` (if present) so `serve`
    // boots with the vectors indexed by the CLI instead of an empty store,
    // and honours the CLI --dim flag instead of SplatsDBConfig::default() (issue #3).
    match load_default_shard(&mut store, data_dir) {
        Ok(n) if n > 0 => eprintln!(
            "[splatsdb] serve: loaded {n} vectors (dim {}) from {data_dir}",
            store.get_statistics().embedding_dim
        ),
        Ok(_) => eprintln!("[splatsdb] serve: no existing data in {data_dir}, starting empty"),
        Err(e) => eprintln!("[splatsdb] serve: failed to load {data_dir}: {e}"),
    }

    // ── Índice exacto en GPU (--gpu, feature cuda): kernel PTX l2+topk sobre
    // los vectores cargados. Fallback silencioso a CPU si no hay GPU.
    #[cfg(feature = "cuda")]
    let gpu_index: Option<Arc<crate::gpu::cuda_kernel::GpuIndex>> = if gpu_enabled {
        match crate::gpu::cuda_kernel::GpuIndex::new() {
            Some(mut g) => match store.get_mu() {
                Some(mu) if mu.nrows() > 0 && mu.ncols() > 0 => {
                    let (n, d) = (mu.nrows(), mu.ncols());
                    let slice = mu
                        .as_slice()
                        .map(<[f32]>::to_vec)
                        .unwrap_or_else(|| mu.iter().copied().collect());
                    if g.upload_dataset(&slice, n, d) {
                        eprintln!("[splatsdb] serve: GPU exact index listo ({n} vectores, dim {d})");
                        Some(Arc::new(g))
                    } else {
                        eprintln!("[splatsdb] serve: upload GPU falló; búsqueda en CPU");
                        None
                    }
                }
                _ => {
                    eprintln!("[splatsdb] serve: store vacío, GPU sin dataset");
                    None
                }
            },
            None => {
                eprintln!("[splatsdb] serve: GPU no disponible (--gpu ignorado); búsqueda en CPU");
                None
            }
        }
    } else {
        None
    };
    #[cfg(not(feature = "cuda"))]
    let _ = gpu_enabled;

    let field_data = load_field(data_dir);
    if field_data.is_some() {
        eprintln!("[splatsdb] serve: campo splat cargado (/field-search disponible)");
    }

    let store = Arc::new(Mutex::new(store));
    let (batch_tx, batch_rx) = mpsc::channel(BATCH_CHANNEL_CAP);
    let batches_served = Arc::new(AtomicUsize::new(0));
    tokio::spawn(batch_worker(
        batch_rx,
        store.clone(),
        batches_served.clone(),
    ));

    let state = AppState {
        store,
        next_id: Arc::new(AtomicUsize::new(0)),
        batch_tx,
        batches_served,
        #[cfg(feature = "cuda")]
        gpu: gpu_index,
        field: field_data.map(Arc::new),
    };

    let app = build_router(state);

    let bind_addr = format!("{addr}:{port}");
    println!(
        "SplatsDB server listening on {bind_addr} \
         (hnsw={}, batching: window={BATCH_WINDOW_MS}ms max={MAX_BATCH}, gpu={})",
        hnsw_on,
        crate::gpu::is_cuda_available()
    );
    let listener = tokio::net::TcpListener::bind(&bind_addr).await?;
    Ok(axum::serve(listener, app).await?)
}

/// Server preset applied on top of the CLI-provided config: HNSW ENABLED with
/// a reasonable over-fetch.
///
/// The server previously used `SplatsDBConfig::default()`, which ships HNSW
/// disabled (config/mod.rs: `enable_hnsw: false`) — every `/search` degraded
/// to a linear O(N·D) scan even through `find_neighbors_fused` (R9 §5.3).
/// The CLI config (dim, data_dir plumbing from issue #3) is preserved; only
/// the HNSW tuning knobs are overridden here.
fn server_config(mut base: SplatsDBConfig) -> SplatsDBConfig {
    base.enable_hnsw = true;
    base.hnsw_m = 32;
    base.hnsw_ef_construction = 200;
    base.hnsw_ef_search = 100;
    // HNSW candidate over-fetch factor before exact re-rank.
    base.over_fetch = 5;
    base
}

/// Build the application router (shared by `run_server` and tests).
fn build_router(state: AppState) -> Router {
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

    Router::new()
        .route("/health", get(health))
        .route("/status", post(status))
        .route("/store", post(store_memory))
        .route("/search", post(search_memories))
        .route("/field-search", post(field_search_handler))
        .layer(cors)
        .layer(middleware::from_fn(api_key_auth))
        .with_state(state)
}

// ─── Tests ────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request as HttpRequest;
    use serde_json::{json, Value};
    use tower::ServiceExt; // oneshot

    fn test_config() -> SplatsDBConfig {
        SplatsDBConfig {
            max_splats: 500,
            latent_dim: 32,
            enable_hnsw: true,
            hnsw_m: 8,
            hnsw_ef_construction: 50,
            hnsw_ef_search: 32,
            over_fetch: 5,
            ..SplatsDBConfig::default()
        }
    }

    /// App instance with a fresh store, a live batch worker and a readable
    /// batches-served counter for verifying coalescing.
    async fn test_app() -> (Router, Arc<AtomicUsize>) {
        let store = Arc::new(Mutex::new(SplatStore::new(test_config())));
        let (batch_tx, batch_rx) = mpsc::channel(BATCH_CHANNEL_CAP);
        let batches_served = Arc::new(AtomicUsize::new(0));
        tokio::spawn(batch_worker(
            batch_rx,
            store.clone(),
            batches_served.clone(),
        ));
        let state = AppState {
            store,
            next_id: Arc::new(AtomicUsize::new(0)),
            batch_tx,
            batches_served: batches_served.clone(),
            #[cfg(feature = "cuda")]
            gpu: None,
            field: None,
        };
        (build_router(state), batches_served)
    }

    async fn send(
        app: Router,
        method: &str,
        uri: &str,
        body: Option<Value>,
    ) -> (StatusCode, Value) {
        let body = match body {
            Some(v) => Body::from(v.to_string()),
            None => Body::empty(),
        };
        let req = HttpRequest::builder()
            .method(method)
            .uri(uri)
            .header("content-type", "application/json")
            .body(body)
            .unwrap();
        let resp = app.oneshot(req).await.expect("oneshot request failed");
        let status = resp.status();
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .expect("read body failed");
        // Error responses (e.g. 400/413) carry a plain-text body — fall back
        // to a string Value so callers can still inspect status + message.
        let json: Value = if bytes.is_empty() {
            Value::Null
        } else {
            serde_json::from_slice(&bytes)
                .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&bytes).into_owned()))
        };
        (status, json)
    }

    fn unit_vec(dim: usize, axis: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; dim];
        v[axis] = 1.0;
        v
    }

    // ── health ──────────────────────────────────────────────

    #[tokio::test]
    async fn test_health_endpoint() {
        let (app, _) = test_app().await;
        let (status, body) = send(app, "GET", "/health", None).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["status"], "ok");
        assert!(body["version"].is_string());
    }

    // ── store + search roundtrip ────────────────────────────

    #[tokio::test]
    async fn test_store_and_search_roundtrip() {
        let (app, _) = test_app().await;

        // Store 3 orthogonal unit vectors (dim 32).
        for axis in 0..3 {
            let (status, body) = send(
                app.clone(),
                "POST",
                "/store",
                Some(json!({ "text": format!("vec {axis}"), "embedding": unit_vec(32, axis) })),
            )
            .await;
            assert_eq!(status, StatusCode::OK, "store axis {axis} failed: {body}");
            assert_eq!(body["status"], "stored");
            assert!(body["id"].is_string());
        }

        // Status reflects the 3 stored splats.
        let (status, body) = send(app.clone(), "POST", "/status", Some(json!({}))).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["n_active"], 3);
        assert_eq!(body["dimension"], 32);
        assert_eq!(body["has_hnsw"], true);

        // Search with the exact embedding of splat 1 → nearest is index 1.
        let (status, body) = send(
            app.clone(),
            "POST",
            "/search",
            Some(json!({ "query": "exact", "embedding": unit_vec(32, 1), "top_k": 2 })),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let results = body["results"].as_array().expect("results array");
        assert_eq!(results.len(), 2, "top_k=2 must return 2 results");
        assert_eq!(results[0]["index"], 1, "nearest neighbor must be splat 1");
        let score = results[0]["score"].as_f64().expect("score is number");
        assert!(score < 1e-4, "exact match distance must be ~0, got {score}");
        assert!(results[0]["metadata"].is_null());

        // Text-only query (hash embedding path) still returns k results.
        let (status, body) = send(
            app,
            "POST",
            "/search",
            Some(json!({ "query": "hello world", "top_k": 3 })),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["results"].as_array().unwrap().len(), 3);
    }

    #[tokio::test]
    async fn test_search_empty_store() {
        let (app, _) = test_app().await;
        let (status, body) =
            send(app, "POST", "/search", Some(json!({ "query": "anything" }))).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            body["results"].as_array().unwrap().len(),
            0,
            "empty store must return empty results"
        );
    }

    // ── ?rescore=gaussian accepted ──────────────────────────

    #[tokio::test]
    async fn test_search_rescore_gaussian_param_accepted() {
        let (app, _) = test_app().await;
        send(
            app.clone(),
            "POST",
            "/store",
            Some(json!({ "text": "a", "embedding": unit_vec(32, 0) })),
        )
        .await;

        let (status, body) = send(
            app.clone(),
            "POST",
            "/search?rescore=gaussian",
            Some(json!({ "query": "a", "embedding": unit_vec(32, 0), "top_k": 1 })),
        )
        .await;
        assert_eq!(
            status,
            StatusCode::OK,
            "?rescore=gaussian must be accepted (200), got {status}"
        );
        let results = body["results"].as_array().expect("results array");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0]["index"], 0);

        // Unknown rescore values are tolerated (ignored).
        let (status, _) = send(
            app,
            "POST",
            "/search?rescore=bogus",
            Some(json!({ "query": "a" })),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
    }

    // ── ?rescore=gaussian semantics ─────────────────────────

    #[tokio::test]
    async fn test_rescore_gaussian_changes_score_semantics() {
        let (app, _) = test_app().await;

        // Store 5 splats: dense axes 0..5 (dim 32).
        for axis in 0..5 {
            let (status, _) = send(
                app.clone(),
                "POST",
                "/store",
                Some(json!({ "text": format!("g{axis}"), "embedding": unit_vec(32, axis) })),
            )
            .await;
            assert_eq!(status, StatusCode::OK);
        }

        let query = {
            // Blend of axes → non-trivial distance profile.
            let mut v = vec![0.0f32; 32];
            v[0] = 0.9;
            v[1] = 0.43589;
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            v.iter_mut().for_each(|x| *x /= norm);
            v
        };

        // Plain L2 search: scores are distances, ascending.
        let (status, plain) = send(
            app.clone(),
            "POST",
            "/search",
            Some(json!({ "query": "blend", "embedding": query.clone(), "top_k": 5 })),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let plain_scores: Vec<f64> = plain["results"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r["score"].as_f64().unwrap())
            .collect();
        assert_eq!(plain_scores.len(), 5);
        assert!(
            plain_scores.windows(2).all(|w| w[0] <= w[1]),
            "L2 scores must be ascending distances: {plain_scores:?}"
        );

        // Gaussian rescore: scores are alpha·exp(-kappa·d²), descending, in (0,1].
        let (status, rescored) = send(
            app,
            "POST",
            "/search?rescore=gaussian",
            Some(json!({ "query": "blend", "embedding": query.clone(), "top_k": 5 })),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let rescored_scores: Vec<f64> = rescored["results"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r["score"].as_f64().unwrap())
            .collect();
        assert_eq!(rescored_scores.len(), 5);
        assert!(
            rescored_scores.windows(2).all(|w| w[0] >= w[1]),
            "gaussian scores must be descending: {rescored_scores:?}"
        );
        assert!(
            rescored_scores.iter().all(|s| *s > 0.0 && *s <= 1.000001),
            "gaussian scores must be in (0,1] (alpha=1): {rescored_scores:?}"
        );

        // Same winner (uniform kappa ⇒ same argmax as min-distance).
        assert_eq!(
            plain["results"][0]["index"], rescored["results"][0]["index"],
            "uniform alpha/kappa ⇒ gaussian argmax must equal L2 argmin"
        );
    }

    // ── GPU-path logic (runs on CPU fallback when no CUDA device) ──

    /// Runs `gpu_batch_path` exactly like `serve_batch` does: inside
    /// `spawn_blocking` (it takes a blocking store lock), round-tripping the
    /// snapshot cache.
    async fn run_gpu_batch(
        store: &Arc<Mutex<SplatStore>>,
        snapshot: &mut Option<DatasetSnapshot>,
        queries: &[Vec<f32>],
        ks: &[usize],
        rescore: &[bool],
    ) -> Result<Vec<Vec<SearchResult>>, String> {
        let store2 = store.clone();
        let queries = queries.to_vec();
        let ks = ks.to_vec();
        let rescore = rescore.to_vec();
        let mut snap = snapshot.take();
        let (result, snap_back) = tokio::task::spawn_blocking(move || {
            let r = gpu_batch_path(&store2, &mut snap, &queries, &ks, &rescore);
            (r, snap)
        })
        .await
        .expect("spawn_blocking panicked");
        *snapshot = snap_back;
        result
    }

    #[tokio::test]
    async fn test_gpu_batch_path_snapshot_and_rescore_split() {
        let store = Arc::new(Mutex::new(SplatStore::new(test_config())));
        // Seed 6 splats directly.
        {
            let mut s = store.lock().await;
            let data = Array2::from_shape_fn((6, 32), |(i, j)| if i % 32 == j { 1.0 } else { 0.0 });
            assert!(s.add_splat(&data));
        }

        let mut snapshot = None;
        let q0: Vec<f32> = unit_vec(32, 0);
        let q1: Vec<f32> = unit_vec(32, 3);

        // Mixed batch: one plain L2 query (k=2), one gaussian-rescored (k=2).
        let queries = vec![q0.clone(), q1.clone()];
        let ks = vec![2usize, 2];
        let rescore = vec![false, true];

        let out = run_gpu_batch(&store, &mut snapshot, &queries, &ks, &rescore)
            .await
            .expect("gpu_batch_path must succeed via CPU fallback");

        assert_eq!(out.len(), 2);
        // Row 0: plain L2 — nearest is splat 0 at distance ~0.
        assert_eq!(out[0][0].index, 0);
        assert!(
            out[0][0].score < 1e-4,
            "exact match L2 ~0, got {}",
            out[0][0].score
        );
        assert!(out[0][0].score <= out[0][1].score, "L2 ascending");
        // Row 1: gaussian — nearest is splat 3, score = alpha·exp(0) ≈ 1.
        assert_eq!(out[1][0].index, 3);
        assert!(
            (out[1][0].score - 1.0).abs() < 1e-4,
            "gaussian score of exact match ≈ 1 (alpha=1), got {}",
            out[1][0].score
        );
        assert!(out[1][0].score >= out[1][1].score, "gaussian descending");

        // Snapshot cached: same version → not rebuilt (data ptr identity as a
        // cheap proxy is unreliable; instead verify search still correct and
        // version unchanged after a second call).
        let out2 = run_gpu_batch(&store, &mut snapshot, &queries, &ks, &rescore)
            .await
            .expect("second call must succeed with cached snapshot");
        assert_eq!(out2[0][0].index, 0);
        assert_eq!(out2[1][0].index, 3);
        assert_eq!(snapshot.as_ref().unwrap().version, 6);

        // Store grows → snapshot invalidated and refreshed.
        {
            let mut s = store.lock().await;
            assert!(s.add_splat(&Array2::from_shape_vec((1, 32), unit_vec(32, 5)).unwrap()));
        }
        let out3 = run_gpu_batch(&store, &mut snapshot, &queries, &ks, &rescore)
            .await
            .expect("third call must refresh snapshot");
        assert_eq!(snapshot.as_ref().unwrap().version, 7);
        assert_eq!(out3[0][0].index, 0);
    }

    // ── validation ──────────────────────────────────────────

    #[tokio::test]
    async fn test_embedding_dim_mismatch_rejected() {
        let (app, _) = test_app().await;
        let (status, body) = send(
            app.clone(),
            "POST",
            "/search",
            Some(json!({ "query": "x", "embedding": unit_vec(31, 0) })),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(
            body.to_string().contains("Embedding dimension mismatch"),
            "error message should mention the mismatch, got: {body}"
        );

        // /store rejects mismatched dims too.
        let (status, _) = send(
            app,
            "POST",
            "/store",
            Some(json!({ "text": "x", "embedding": unit_vec(31, 0) })),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    // ── micro-batching actually coalesces ───────────────────

    #[tokio::test]
    async fn test_concurrent_searches_coalesce_into_batches() {
        let (app, batches) = test_app().await;

        // Seed the store with 20 splats.
        for axis in 0..20 {
            let mut emb = vec![0.0f32; 32];
            emb[axis % 32] = 1.0;
            if axis >= 32 {
                emb[axis - 32] = 0.5;
            }
            let (status, _) = send(
                app.clone(),
                "POST",
                "/store",
                Some(json!({ "text": format!("v{axis}"), "embedding": emb })),
            )
            .await;
            assert_eq!(status, StatusCode::OK);
        }

        let before = batches.load(Ordering::Relaxed);

        // 8 concurrent searches on a current-thread runtime: all sends are
        // enqueued before the worker is polled, so they must land in fewer
        // than 8 batches (in practice exactly 1).
        let mut futs: Vec<BoxFut> = Vec::new();
        for i in 0..8 {
            let app_i = app.clone();
            futs.push(Box::pin(async move {
                send(
                    app_i,
                    "POST",
                    "/search",
                    Some(json!({ "query": format!("q{i}"), "top_k": 4 })),
                )
                .await
            }));
        }
        let replies = futures_join_all(futs).await; // see helper below

        for (i, (status, body)) in replies.iter().enumerate() {
            assert_eq!(*status, StatusCode::OK, "concurrent search {i} failed");
            assert_eq!(body["results"].as_array().unwrap().len(), 4);
        }

        let served = batches.load(Ordering::Relaxed) - before;
        assert!(
            served < 8,
            "8 concurrent searches must coalesce into fewer than 8 batches (got {served})",
        );
        eprintln!("[test] 8 concurrent searches served in {served} batch(es)");
    }

    /// Poor-man's join_all without pulling in the `futures` crate: spawn each
    /// future as a tokio task and collect. On a current-thread runtime this
    /// still enqueues all sends before the worker runs.
    type BoxFut = std::pin::Pin<Box<dyn std::future::Future<Output = (StatusCode, Value)> + Send>>;
    async fn futures_join_all(futs: Vec<BoxFut>) -> Vec<(StatusCode, Value)> {
        let mut handles = Vec::with_capacity(futs.len());
        for f in futs {
            handles.push(tokio::spawn(f));
        }
        let mut out = Vec::with_capacity(handles.len());
        for h in handles {
            out.push(h.await.expect("task panicked"));
        }
        out
    }

    // ── hash embedding ──────────────────────────────────────

    #[test]
    fn test_hash_embedding_properties() {
        let dim = 64;
        let emb = simple_hash_embedding("hello splats", dim);
        assert_eq!(emb.len(), dim);
        // Deterministic
        assert_eq!(emb, simple_hash_embedding("hello splats", dim));
        // Unit norm
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "expected unit norm, got {norm}");
        // Bounded
        assert!(emb.iter().all(|x| x.abs() <= 1.0));
        // Different text → different embedding
        let other = simple_hash_embedding("hello splats!", dim);
        assert_ne!(emb, other);
        // O(1)-hash construction must still cover the full dim
        assert!(emb.iter().filter(|x| **x != 0.0).count() > dim / 2);
    }
}
