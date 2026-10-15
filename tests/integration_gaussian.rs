//! Integration tests for the Gaussian re-scoring path (Olafe 2 / B6).
//!
//! Covers four areas:
//! 1. E2E: 5k random vectors (fixed seed) → build index → `find_neighbors_fused`
//!    must return the exact same top-3 as the brute-force linear scan, for 50 queries.
//! 2. `find_neighbors_gaussian`: top-k by Gaussian score must match an independent
//!    brute-force reference (`alpha_i * exp(-kappa_i * ||q - mu_i||^2)`).
//! 3. GPU/CPU parity of the Gaussian top-k rescore (`gaussian_topk_scores` /
//!    `gpu_gaussian_topk`) — compiled only with `--features cuda`, silently
//!    skipped when no GPU device is present.
//! 4. API server smoke test: boot `run_server` on an ephemeral port, then
//!    /health → /store ×2 → /search over raw HTTP (no extra dependencies).
//!
//! NOTE (B6): `find_neighbors_gaussian`, `gaussian_score` and the Gaussian
//! top-k CUDA kernel came from B1/B2's in-flight work; these tests exercise
//! them exactly as they exist in this clone.

use ndarray::{Array2, ArrayView1};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use splatsdb::config::SplatsDBConfig;
use splatsdb::splats::SplatStore;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

const N_VECTORS: usize = 5_000;
const DIM: usize = 64;
const N_QUERIES: usize = 50;
const TOP_K: usize = 3;
const SEED: u64 = 20261015;

fn random_normalized(n: usize, dim: usize, seed: u64) -> Array2<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut data = Array2::from_shape_fn((n, dim), |(_, _)| rng.gen::<f32>());
    for mut row in data.rows_mut() {
        let norm = row.dot(&row).sqrt().max(1e-10);
        row.mapv_inplace(|v: f32| v / norm);
    }
    data
}

/// Config for the E2E tests: HNSW enabled (generous ef for near-perfect
/// candidate recall), quantization/LSH off so strategy 1 (HNSW + exact
/// re-rank) is the path under test.
#[allow(clippy::field_reassign_with_default)]
fn e2e_config() -> SplatsDBConfig {
    let mut c = SplatsDBConfig::default();
    c.max_splats = 2 * N_VECTORS;
    c.latent_dim = DIM;
    c.enable_hnsw = true;
    c.enable_quantization = false;
    c.enable_lsh = false;
    c.hnsw_m = 24;
    c.hnsw_ef_construction = 200;
    c.hnsw_ef_search = 256;
    c.over_fetch = 10;
    c
}

/// Independent brute-force top-k by Gaussian score.
/// score_i = alpha_i * exp(-kappa_i * ||q - mu_i||^2), higher is better,
/// ties broken by lower index (mirrors the documented convention).
fn gaussian_reference_topk(
    store: &SplatStore,
    query: &ArrayView1<f32>,
    k: usize,
) -> Vec<(usize, f32)> {
    let mu = store.get_mu().expect("mu");
    let alpha = store.get_alpha().expect("alpha");
    let kappa = store.get_kappa().expect("kappa");
    let n = store.n_active();
    let mut scored: Vec<(usize, f32)> = (0..n)
        .map(|i| {
            let row = mu.row(i);
            let d2: f32 = row
                .iter()
                .zip(query.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum();
            (i, alpha[i] * (-(kappa[i] * d2)).exp())
        })
        .collect();
    // Stable descending sort by score → ties keep the lower index first.
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(k);
    scored
}

fn l2_sq(q: &[f32], v: &[f32]) -> f32 {
    q.iter().zip(v.iter()).map(|(a, b)| (a - b) * (a - b)).sum()
}

// ─────────────────────────────────────────────────────────────────────────────
// 1) E2E: fused search == linear scan (top-3, 50 queries)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn e2e_fused_matches_linear_top3_5k_vectors() {
    let data = random_normalized(N_VECTORS, DIM, SEED);
    let queries = random_normalized(N_QUERIES, DIM, SEED + 1);

    let mut store = SplatStore::new(e2e_config());
    assert!(
        store.add_splat(&data),
        "batch insert of {N_VECTORS} vectors"
    );
    assert_eq!(store.n_active(), N_VECTORS);

    store.build_index();
    assert!(store.hnsw_is_built(), "HNSW backend must be built");

    let mut mismatches = 0usize;
    for q in queries.rows() {
        let linear = store.find_neighbors(&q, TOP_K);
        let fused = store.find_neighbors_fused(&q, TOP_K);

        assert_eq!(fused.len(), TOP_K, "fused must return k results");
        assert_eq!(linear.len(), TOP_K, "linear must return k results");

        for (lin, fus) in linear.iter().zip(fused.iter()) {
            if lin.index != fus.index {
                mismatches += 1;
                eprintln!(
                    "query mismatch: linear idx={} d={:.6} vs fused idx={} d={:.6}",
                    lin.index, lin.distance, fus.index, fus.distance
                );
                continue;
            }
            // Same index → exact distance must match to the last bits of fp32.
            assert!(
                (lin.distance - fus.distance).abs() < 1e-5,
                "distance drift on idx {}: {} vs {}",
                lin.index,
                lin.distance,
                fus.distance
            );
        }
    }
    assert_eq!(
        mismatches,
        0,
        "{mismatches}/{} fused results differ from linear",
        N_QUERIES * TOP_K
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// 2) find_neighbors_gaussian vs independent brute-force reference
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn gaussian_search_matches_bruteforce_reference() {
    let data = random_normalized(N_VECTORS, DIM, SEED + 7);
    let queries = random_normalized(16, DIM, SEED + 8);

    let mut store = SplatStore::new(e2e_config());
    assert!(store.add_splat(&data));
    store.build_index();

    let k = 5;
    for q in queries.rows() {
        let got = store.find_neighbors_gaussian(&q, k);
        assert_eq!(got.len(), k, "gaussian search must return k results");

        let want = gaussian_reference_topk(&store, &q, k);

        // Identical indices, in identical order.
        for (j, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert_eq!(
                g.index, w.0,
                "rank {j}: gaussian search returned idx {} but reference says {}",
                g.index, w.0
            );
            // Score must match the reference formula (f32 vs f64-free path).
            assert!(
                (g.score - w.1).abs() < 1e-5,
                "rank {j}: score {} vs reference {}",
                g.score,
                w.1
            );
        }

        // Scores must be descending.
        for w in got.windows(2) {
            assert!(
                w[0].score >= w[1].score,
                "scores must be descending: {} then {}",
                w[0].score,
                w[1].score
            );
        }

        // Distance field stays consistent with mu (spot-check rank 0).
        let d2 = l2_sq(q.as_slice().unwrap(), &got[0].mu);
        assert!((d2.sqrt() - got[0].distance).abs() < 1e-4);
    }
}

/// With default uniform alpha/kappa the Gaussian score is monotonically
/// decreasing in squared distance, so the Gaussian ranking must equal the
/// plain distance ranking — a cheap cross-check tying both paths together.
#[test]
fn gaussian_search_ranking_equals_distance_ranking_with_uniform_params() {
    let data = random_normalized(2_000, DIM, SEED + 9);

    let mut store = SplatStore::new(e2e_config());
    assert!(store.add_splat(&data));
    store.build_index();

    let queries = random_normalized(10, DIM, SEED + 10);
    for q in queries.rows() {
        let by_dist = store.find_neighbors_fused(&q, 5);
        let by_score = store.find_neighbors_gaussian(&q, 5);
        for (d, s) in by_dist.iter().zip(by_score.iter()) {
            assert_eq!(d.index, s.index);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3) GPU/CPU parity of the Gaussian top-k rescore (feature "cuda" only)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "cuda")]
mod gpu_parity {
    use super::*;
    use splatsdb::gpu;
    use splatsdb::gpu::cuda_kernel::gpu_gaussian_topk;

    const GPU_N: usize = 4_096;
    const GPU_Q: usize = 64;

    fn dataset_and_params(seed: u64) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut data = Vec::with_capacity(GPU_N * DIM);
        let mut alpha = Vec::with_capacity(GPU_N);
        let mut kappa = Vec::with_capacity(GPU_N);
        for _ in 0..GPU_N {
            let mut v: Vec<f32> = (0..DIM).map(|_| rng.gen::<f32>()).collect();
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-10);
            for x in &mut v {
                *x /= norm;
            }
            data.extend(v);
            alpha.push(0.5 + rng.gen::<f32>() * 1.5); // alpha ∈ [0.5, 2.0]
            kappa.push(0.1 + rng.gen::<f32>() * 9.9); // kappa ∈ [0.1, 10.0]
        }
        (data, alpha, kappa)
    }

    /// Independent CPU reference for the Gaussian top-k (same convention as
    /// `cpu_gaussian_topk`, implemented here so the test never just compares
    /// the crate against itself).
    fn reference_topk(
        queries: &[f32],
        dataset: &[f32],
        alpha: &[f32],
        kappa: &[f32],
        n_queries: usize,
        k: usize,
    ) -> Vec<Vec<(usize, f32)>> {
        (0..n_queries)
            .map(|q| {
                let qs = &queries[q * DIM..(q + 1) * DIM];
                let mut scored: Vec<(usize, f32)> = (0..GPU_N)
                    .map(|i| {
                        let row = &dataset[i * DIM..(i + 1) * DIM];
                        let d2: f32 = qs
                            .iter()
                            .zip(row.iter())
                            .map(|(a, b)| (a - b) * (a - b))
                            .sum();
                        (i, alpha[i] * (-(kappa[i] * d2)).exp())
                    })
                    .collect();
                scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                scored.truncate(k);
                scored
            })
            .collect()
    }

    fn assert_parity(gpu_out: &[Vec<(usize, f32)>], cpu_ref: &[Vec<(usize, f32)>], ctx: &str) {
        assert_eq!(gpu_out.len(), cpu_ref.len(), "{ctx}: query count");
        for (q, (gq, cq)) in gpu_out.iter().zip(cpu_ref.iter()).enumerate() {
            assert_eq!(gq.len(), cq.len(), "{ctx}: k mismatch on query {q}");
            for (rank, ((gi, gs), (ci, cs))) in gq.iter().zip(cq.iter()).enumerate() {
                assert_eq!(
                    gi, ci,
                    "{ctx}: query {q} rank {rank}: GPU idx {gi} vs CPU idx {ci}"
                );
                let diff = (gs - cs).abs();
                assert!(
                    diff <= 1e-5,
                    "{ctx}: query {q} rank {rank} idx {gi}: GPU score {gs} vs CPU {cs} (diff {diff})"
                );
            }
        }
    }

    #[test]
    fn gaussian_topk_gpu_cpu_parity() {
        if !gpu::is_cuda_available() {
            return; // silent skip: no GPU device
        }

        let (dataset, alpha, kappa) = dataset_and_params(SEED + 100);
        let queries: Vec<f32> = random_normalized(GPU_Q, DIM, SEED + 101)
            .as_slice()
            .unwrap()
            .to_vec();

        let mut gpu_verified = 0usize;
        for k in [1usize, 5, 32] {
            // Direct GPU path (gaussian_topk_kernel PTX, 1 <= K <= 32).
            // The smem opt-in fix means K=32 (65.5KB dynamic smem on D=64)
            // must launch on sm_86 — None here is a hard failure.
            let gpu_out =
                gpu_gaussian_topk(&queries, GPU_Q, &dataset, &alpha, &kappa, GPU_N, DIM, k)
                    .unwrap_or_else(|| {
                        panic!(
                            "gpu_gaussian_topk returned None for k={k} — kernel must support \
                     1<=K<=32 (extended smem opt-in failed?)"
                        )
                    });

            let cpu_ref = reference_topk(&queries, &dataset, &alpha, &kappa, GPU_Q, k);
            assert_parity(&gpu_out, &cpu_ref, &format!("direct kernel k={k}"));

            // Public wrapper: same result, whichever internal path it takes.
            let wrapped =
                gpu::gaussian_topk_scores(&queries, &dataset, &alpha, &kappa, GPU_N, DIM, GPU_Q, k)
                    .expect("wrapper must return Some with valid inputs");
            assert_parity(&wrapped, &cpu_ref, &format!("wrapper k={k}"));
            gpu_verified += 1;
        }
        eprintln!("gaussian GPU kernel parity verified for {gpu_verified}/3 k values");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 4) API server smoke test (oneshot-style over real HTTP)
// ─────────────────────────────────────────────────────────────────────────────

mod api_smoke {
    use std::io::{Read, Write};
    use std::net::{TcpListener, TcpStream};
    use std::time::Duration;

    /// Find a free TCP port (bind :0, read the port, drop the listener).
    fn free_port() -> u16 {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind :0");
        listener.local_addr().expect("local addr").port()
    }

    /// Minimal HTTP/1.1 client: one request per connection (Connection: close).
    fn http_request(port: u16, method: &str, path: &str, body: Option<&str>) -> (u16, String) {
        let mut stream = TcpStream::connect(("127.0.0.1", port)).expect("connect");
        stream
            .set_read_timeout(Some(Duration::from_secs(10)))
            .unwrap();
        let body = body.unwrap_or("");
        let request = format!(
            "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        stream.write_all(request.as_bytes()).expect("write");
        let mut response = String::new();
        stream.read_to_string(&mut response).expect("read");
        let status: u16 = response
            .lines()
            .next()
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|s| s.parse().ok())
            .expect("status line");
        let body = response.split("\r\n\r\n").nth(1).unwrap_or("").to_string();
        (status, body)
    }

    #[test]
    fn api_server_smoke_search_oneshot() {
        let port = free_port();

        // Boot the real server (run_server) in a dedicated thread + runtime.
        // The thread lives until the test process exits — acceptable for a
        // smoke test; there is no shutdown hook in run_server.
        std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("tokio runtime");
            let _ = rt.block_on(splatsdb::api_server::run_server(
                "127.0.0.1",
                port,
                splatsdb::config::SplatsDBConfig::default(),
                std::env::temp_dir()
                    .join("splatsdb_smoke")
                    .to_str()
                    .unwrap(),
            ));
        });

        // Wait for /health with retries (server boot is asynchronous; under
        // CI's debug-build CPU contention boot can take tens of seconds).
        let mut healthy = false;
        for _ in 0..600 {
            if let Ok(s) = TcpStream::connect(("127.0.0.1", port)) {
                drop(s);
                let (status, body) = http_request(port, "GET", "/health", None);
                if status == 200 && body.contains("\"ok\"") {
                    healthy = true;
                    break;
                }
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        assert!(healthy, "server did not become healthy on port {port}");

        // Store two memories (hash-embedding path, deterministic).
        let (s1, b1) = http_request(
            port,
            "POST",
            "/store",
            Some(r#"{"text": "gaussian splat memory one"}"#),
        );
        assert_eq!(s1, 200, "POST /store #1 failed: {b1}");
        assert!(b1.contains("stored"), "unexpected body: {b1}");

        let (s2, b2) = http_request(
            port,
            "POST",
            "/store",
            Some(r#"{"text": "gaussian splat memory two"}"#),
        );
        assert_eq!(s2, 200, "POST /store #2 failed: {b2}");

        // Search must return results ranked by distance.
        let (s3, b3) = http_request(
            port,
            "POST",
            "/search",
            Some(r#"{"query": "gaussian splat memory one", "top_k": 2}"#),
        );
        assert_eq!(s3, 200, "POST /search failed: {b3}");

        let parsed: serde_json::Value = serde_json::from_str(&b3).expect("search JSON");
        let results = parsed["results"].as_array().expect("results array");
        assert!(!results.is_empty(), "search returned no results: {b3}");
        for r in results {
            assert!(r["index"].is_u64(), "result missing index: {r}");
            assert!(r["score"].is_f64(), "result missing score: {r}");
        }
        let scores: Vec<f64> = results.iter().filter_map(|r| r["score"].as_f64()).collect();
        let mut sorted = scores.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(scores, sorted, "search results must be distance-sorted");
    }
}
