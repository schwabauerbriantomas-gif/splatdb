//! Index, ingest, and benchmark command handlers.

use std::path::{Path, PathBuf};

use splatsdb::{SplatStore, SplatsDBConfig};

use super::helpers::*;

pub fn cmd_index(
    data_dir: String,
    backend: String,
    config: SplatsDBConfig,
    input: PathBuf,
    shard: String,
) {
    let vectors = match load_vectors_bin(&input) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    };
    eprintln!(
        "[splatsdb] Loaded {} vectors ({}D) from {}",
        vectors.nrows(),
        vectors.ncols(),
        input.display()
    );

    let persist = make_persistence(&data_dir, &backend, true).unwrap_or_else(|e| {
        eprintln!("Storage error: {}", e);
        std::process::exit(1);
    });

    persist.save_vectors(&vectors, &shard).unwrap_or_else(|e| {
        eprintln!("Save error: {}", e);
        std::process::exit(1);
    });

    let mut store = SplatStore::new(config);
    store.add_splat(&vectors);
    store.build_index_with_save(Some(&data_dir));

    println!(
        "{}",
        serde_json::json!({
            "status": "indexed",
            "shard": shard,
            "n_vectors": vectors.nrows(),
            "dim": vectors.ncols(),
            "n_active": store.n_active(),
            "entropy": (store.entropy() * 10000.0).round() / 10000.0,
        })
    );
}

/// Append vectors to an existing store using incremental HNSW insertion.
///
/// Unlike `cmd_index` which creates a fresh store, this loads the existing
/// store (including persisted HNSW), adds new vectors, incrementally inserts
/// them into the HNSW graph (no full rebuild), and saves.
pub fn cmd_append(
    data_dir: String,
    _backend: String,
    config: SplatsDBConfig,
    input: PathBuf,
    shard: String,
) {
    let vectors = match load_vectors_bin(&input) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    };
    eprintln!(
        "[splatsdb] Appending {} vectors ({}D) from {}",
        vectors.nrows(),
        vectors.ncols(),
        input.display()
    );

    // Load existing store
    let (mut store, persist) = load_or_create_store(&data_dir, &config);
    let n_before = store.n_active();
    let hnsw_before = store.hnsw_indexed_count();

    eprintln!(
        "[splatsdb] Existing store: {} vectors, {} HNSW indexed",
        n_before, hnsw_before
    );

    // Add new vectors
    let t0 = std::time::Instant::now();
    let added = store.add_batch_with_hnsw(&vectors, Some(&data_dir));
    let elapsed = t0.elapsed();

    if !added {
        eprintln!("Error: store full, cannot append");
        std::process::exit(1);
    }

    // Persist vectors to storage backend
    if let Some(p) = persist {
        if let Some(vectors) = store.get_mu() {
            if let Err(e) = p.save_vectors(&vectors, &shard) {
                eprintln!("Warning: save failed: {}", e);
            }
        }
    }

    println!(
        "{}",
        serde_json::json!({
            "status": "appended",
            "shard": shard,
            "vectors_added": vectors.nrows(),
            "n_before": n_before,
            "n_after": store.n_active(),
            "hnsw_indexed": store.hnsw_indexed_count(),
            "append_time_ms": elapsed.as_millis() as u64,
        })
    );
}

pub fn cmd_ingest(config: SplatsDBConfig, input: PathBuf, n_clusters: Option<usize>, seed: u64) {
    let vectors = match load_vectors_bin(&input) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    };
    eprintln!(
        "[splatsdb] Ingesting {} vectors ({}D) via DatasetTransformer",
        vectors.nrows(),
        vectors.ncols()
    );

    let mut store = SplatStore::new(config);
    let k = n_clusters.unwrap_or_else(|| (vectors.nrows() as f64).sqrt() as usize);
    let t0 = std::time::Instant::now();

    match store.ingest_with_transformer(&vectors, k, seed) {
        Ok((n_splats, compression, stats)) => {
            let elapsed = t0.elapsed();
            store.build_index();
            println!(
                "{}",
                serde_json::json!({
                    "status": "ingested",
                    "method": "kmeans",
                    "original_count": stats.original_count,
                    "n_splats": n_splats,
                    "compression_ratio": (compression * 10000.0).round() / 10000.0,
                    "transform_time_s": (stats.transform_time_s * 10000.0).round() / 10000.0,
                    "total_ingest_ms": elapsed.as_millis() as u64,
                    "dim": vectors.ncols(),
                })
            );
        }
        Err(e) => {
            eprintln!("Ingest error: {}", e);
            std::process::exit(1);
        }
    }
}

pub fn cmd_ingest_hierarchical(
    config: SplatsDBConfig,
    input: PathBuf,
    n_clusters: usize,
    min_cluster_size: usize,
    seed: u64,
) {
    let vectors = match load_vectors_bin(&input) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    };
    eprintln!(
        "[splatsdb] Hierarchical ingest {} vectors ({}D)",
        vectors.nrows(),
        vectors.ncols()
    );

    let mut store = SplatStore::new(config);
    let t0 = std::time::Instant::now();

    match store.ingest_hierarchical(&vectors, n_clusters, min_cluster_size, seed) {
        Ok((n_splats, compression, stats)) => {
            let elapsed = t0.elapsed();
            store.build_index();
            println!(
                "{}",
                serde_json::json!({
                    "status": "ingested",
                    "method": "hierarchical_kmeans",
                    "original_count": stats.original_count,
                    "n_splats": n_splats,
                    "compression_ratio": (compression * 10000.0).round() / 10000.0,
                    "total_ingest_ms": elapsed.as_millis() as u64,
                    "dim": vectors.ncols(),
                })
            );
        }
        Err(e) => {
            eprintln!("Ingest error: {}", e);
            std::process::exit(1);
        }
    }
}

pub fn cmd_ingest_leader(
    data_dir: String,
    backend: String,
    config: SplatsDBConfig,
    input: PathBuf,
    target_clusters: usize,
    threshold: Option<f64>,
    seed: u64,
) {
    let vectors = match load_vectors_bin(&input) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    };
    eprintln!(
        "[splatsdb] Leader clustering ingest {} vectors ({}D)",
        vectors.nrows(),
        vectors.ncols()
    );

    let mut store = SplatStore::new(config);
    let t0 = std::time::Instant::now();

    match store.ingest_leader(&vectors, target_clusters, seed, threshold) {
        Ok((n_splats, compression, stats)) => {
            let elapsed = t0.elapsed();
            store.build_index();
            let persist = make_persistence(&data_dir, &backend, false).ok();
            if let Some(p) = persist {
                if let (Some(mu), Some(alpha), Some(kappa)) =
                    (store.get_mu(), store.get_alpha(), store.get_kappa())
                {
                    let n = store.n_active();
                    let vectors = mu.slice(ndarray::s![..n, ..]).to_owned();
                    if let Err(e) = p.save_vectors(&vectors, "default") {
                        eprintln!("Warning: save failed: {}", e);
                    }
                    if let Err(e) = p.save_energy_state(&vectors, &alpha.to_vec(), &kappa.to_vec())
                    {
                        eprintln!("Warning: energy save failed: {}", e);
                    }
                }
            }
            println!(
                "{}",
                serde_json::json!({
                    "status": "ingested",
                    "method": "leader_clustering",
                    "original_count": stats.original_count,
                    "n_splats": n_splats,
                    "compression_ratio": (compression * 10000.0).round() / 10000.0,
                    "total_ingest_ms": elapsed.as_millis() as u64,
                    "dim": vectors.ncols(),
                    "threshold": threshold,
                    "saved": true,
                })
            );
        }
        Err(e) => {
            eprintln!("Ingest error: {}", e);
            std::process::exit(1);
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn cmd_bench_gpu(
    n_vectors: usize,
    dim: usize,
    n_queries: usize,
    top_k: usize,
    metric: String,
    seed: Option<u64>,
    gt: Option<PathBuf>,
    gt_out: Option<PathBuf>,
    out: Option<PathBuf>,
) {
    #[cfg(not(feature = "cuda"))]
    {
        eprintln!("[splatsdb] bench-gpu requires --features cuda");
        std::process::exit(1);
    }
    #[cfg(feature = "cuda")]
    {
        use splatsdb::gpu::cuda_kernel::GpuIndex;
        use std::time::Instant;

        eprintln!(
            "[splatsdb] Generating {} vectors ({}D), seed: {}...",
            n_vectors,
            dim,
            seed
                .map(|s| s.to_string())
                .unwrap_or_else(|| "<random>".to_string())
        );
        // Seeded RNG (ChaCha8) for reproducible data, or thread RNG when no seed.
        let mut thread_rng = rand::thread_rng();
        let mut seeded = seed
            .map(|s| <rand_chacha::ChaCha8Rng as rand::SeedableRng>::seed_from_u64(s));
        let mut genf = |lo: f32, hi: f32| -> f32 {
            match seeded.as_mut() {
                Some(r) => rand::Rng::gen_range(r, lo..hi),
                None => rand::Rng::gen_range(&mut thread_rng, lo..hi),
            }
        };
        let dataset: Vec<f32> = (0..n_vectors * dim).map(|_| genf(-1.0, 1.0)).collect();
        let queries: Vec<f32> = (0..n_queries * dim).map(|_| genf(-1.0, 1.0)).collect();

        // Ground truth: load from --gt (same parser as bench-hnsw), else compute CPU-exact.
        let gt_data: Option<(usize, usize, Vec<i64>)> = match gt
            .as_ref()
            .map(|p| cli_load_ground_truth_bin(p))
        {
            Some(Ok(g)) => {
                eprintln!(
                    "[splatsdb] Loaded GT from {}: {} queries, k={}",
                    gt.as_ref().unwrap().display(),
                    g.0,
                    g.1
                );
                Some(g)
            }
            Some(Err(e)) => {
                eprintln!("Error loading ground truth: {}", e);
                std::process::exit(1);
            }
            None => None,
        };

        eprintln!("[splatsdb] CPU benchmark (exact brute-force, rayon)...");
        let t0 = Instant::now();
        let cpu_results =
            cli_cpu_topk(&queries, n_queries, &dataset, n_vectors, dim, top_k, &metric);
        let cpu_ms = t0.elapsed().as_millis() as f64;

        // Optionally persist the CPU-exact top-k as GT (same bin format bench-hnsw reads).
        if gt_data.is_none() {
            if let Some(ref p) = gt_out {
                match cli_save_ground_truth_bin(p, &cpu_results) {
                    Ok(()) => eprintln!("[splatsdb] Saved CPU-exact GT to {}", p.display()),
                    Err(e) => eprintln!("[splatsdb] Warning: could not save GT: {}", e),
                }
            }
        }

        let gpu_result = if let Some(mut gpu_idx) = GpuIndex::new() {
            eprintln!("[splatsdb] Uploading {} vectors to GPU...", n_vectors);
            let t_upload = Instant::now();
            gpu_idx.upload_dataset(&dataset, n_vectors, dim);
            let upload_ms = t_upload.elapsed().as_millis() as f64;

            eprintln!("[splatsdb] GPU query benchmark (dataset in VRAM)...");
            let t1 = Instant::now();
            let gpu_search = gpu_idx.topk_search(&queries, n_queries, top_k, &metric);
            let query_ms = t1.elapsed().as_millis() as f64;

            let (gpu_indices, fell_back): (Vec<Vec<usize>>, bool) = match gpu_search {
                Some(r) => (r.into_iter().map(|(idxs, _)| idxs).collect(), false),
                None => {
                    eprintln!("[splatsdb] GPU topk_search returned None; GPU metrics unavailable");
                    (cpu_results.clone(), true)
                }
            };

            let matches = cli_topk_sets_equal(&cpu_results, &gpu_indices);

            let recall_json = if let Some((gt_nq, gt_k, ref gt_idx)) = gt_data {
                serde_json::json!(
                    (cli_recall_at_k(&gpu_indices, gt_nq, gt_k, gt_idx, top_k) * 10000.0).round()
                        / 10000.0
                )
            } else {
                serde_json::Value::Null
            };

            eprintln!("[splatsdb] GPU/CPU results match: {}", matches);
            Some((upload_ms, query_ms, matches, recall_json, fell_back))
        } else {
            eprintln!("[splatsdb] No CUDA device available; GPU section skipped");
            None
        };

        let qps_gpu_persistent = gpu_result
            .as_ref()
            .map(|(_, query_ms, _, _, _)| n_queries as f64 / (query_ms / 1000.0));
        let results_match = gpu_result
            .as_ref()
            .map(|(_, _, matches, _, _)| *matches);
        let recall_at_k = gpu_result
            .as_ref()
            .map(|(_, _, _, r, _)| r.clone())
            .unwrap_or(serde_json::Value::Null);

        let payload = serde_json::json!({
            "bench": "gpu",
            "config": {
                "n_vectors": n_vectors,
                "dim": dim,
                "n_queries": n_queries,
                "top_k": top_k,
                "metric": metric,
                "seed": seed,
                "gt_path": gt.as_ref().map(|p| p.display().to_string()),
                "gt_loaded": gt_data.as_ref().map(|(nq, k, _)| serde_json::json!({"n_queries": nq, "k": k})),
                "dataset_mb": (n_vectors * dim * 4) as f64 / 1_048_576.0,
            },
            "qps_cpu": n_queries as f64 / (cpu_ms / 1000.0),
            "qps_gpu_persistent": qps_gpu_persistent.map(|v| (v * 100.0).round() / 100.0),
            "recall_at_k_vs_gt": recall_at_k,
            "results_match": results_match,
            "timestamp": cli_timestamp_iso8601(),
            "git_hash": cli_git_hash(),
            "cpu": {
                "total_ms": cpu_ms,
                "per_query_us": cpu_ms / n_queries as f64 * 1000.0,
                "qps": n_queries as f64 / (cpu_ms / 1000.0),
            },
            "gpu": gpu_result.map(|(upload_ms, query_ms, _matches, _recall, fell_back)| {
                let total_gpu = upload_ms + query_ms;
                serde_json::json!({
                    "upload_ms": upload_ms,
                    "query_ms": query_ms,
                    "total_ms": total_gpu,
                    "per_query_us": query_ms / n_queries as f64 * 1000.0,
                    "qps_persistent": n_queries as f64 / (query_ms / 1000.0),
                    "qps_with_upload": n_queries as f64 / (total_gpu / 1000.0),
                    "speedup_persistent": cpu_ms / query_ms,
                    "speedup_with_upload": cpu_ms / total_gpu,
                    "fallback_cpu": fell_back,
                })
            }),
        });

        cli_emit_json(&payload, out.as_deref());
    } // cfg cuda
}

pub fn cmd_bench_gpu_ingest(
    n_vectors: usize,
    dim: usize,
    n_clusters: usize,
    n_queries: usize,
    seed: Option<u64>,
    out: Option<PathBuf>,
) {
    use std::time::Instant;

    eprintln!(
        "[splatsdb] Generating {} vectors ({}D), seed: {}...",
        n_vectors,
        dim,
        seed.map(|s| s.to_string())
            .unwrap_or_else(|| "<random>".to_string())
    );
    let mut thread_rng = rand::thread_rng();
    let mut seeded = seed
        .map(|s| <rand_chacha::ChaCha8Rng as rand::SeedableRng>::seed_from_u64(s));
    let mut genf = |lo: f32, hi: f32| -> f32 {
        match seeded.as_mut() {
            Some(r) => rand::Rng::gen_range(r, lo..hi),
            None => rand::Rng::gen_range(&mut thread_rng, lo..hi),
        }
    };
    let raw: Vec<f32> = (0..n_vectors * dim).map(|_| genf(-1.0, 1.0)).collect();
    let dataset = ndarray::Array2::from_shape_vec((n_vectors, dim), raw).unwrap();
    let queries: Vec<f32> = (0..n_queries * dim).map(|_| genf(-1.0, 1.0)).collect();

    let mut ingest_cfg = SplatsDBConfig::gpu(None);
    ingest_cfg.latent_dim = dim;
    ingest_cfg.max_splats = n_clusters * 2;
    ingest_cfg.device = "cpu".to_string();
    ingest_cfg.enable_cuda = false;
    ingest_cfg.enable_gpu_search = false;
    ingest_cfg.finalize();

    let mut store = SplatStore::new(ingest_cfg);
    eprintln!("[splatsdb] Phase 1: DatasetTransformer ingest...");
    let t_ingest = Instant::now();
    let bench_seed = seed.unwrap_or(42);
    let (n_splats, compression, stats) = store
        .ingest_with_transformer(&dataset, n_clusters, bench_seed)
        .unwrap_or_else(|e| {
            eprintln!("Ingest error: {}", e);
            std::process::exit(1);
        });
    let ingest_ms = t_ingest.elapsed().as_millis() as f64;

    eprintln!("[splatsdb] Phase 2: Building index...");
    let t_build = Instant::now();
    store.build_index();
    let build_ms = t_build.elapsed().as_millis() as f64;

    eprintln!(
        "[splatsdb] Phase 3: Search {} queries against {} splats...",
        n_queries, n_splats
    );
    let query_arr = ndarray::Array2::from_shape_vec((n_queries, dim), queries.clone())
        .expect("valid shape from args");
    let t_search = Instant::now();
    for i in 0..n_queries {
        let _ = store.find_neighbors(&query_arr.row(i), 10);
    }
    let search_ms = t_search.elapsed().as_millis() as f64;

    #[cfg(feature = "cuda")]
    let gpu_search: Option<serde_json::Value> = {
        if let Some(mut gpu_idx) = splatsdb::gpu::cuda_kernel::GpuIndex::new() {
            let raw_flat: Vec<f32> = dataset.clone().into_raw_vec_and_offset().0;
            eprintln!(
                "[splatsdb] Phase 4: GPU search on raw {} vectors...",
                n_vectors
            );
            let t_gpu_upload = Instant::now();
            gpu_idx.upload_dataset(&raw_flat, n_vectors, dim);
            let gpu_upload_ms = t_gpu_upload.elapsed().as_millis() as f64;

            let t_gpu_q = Instant::now();
            let _ = gpu_idx.topk_search(&queries, n_queries, 10, "l2");
            let gpu_query_ms = t_gpu_q.elapsed().as_millis() as f64;
            Some(serde_json::json!({
                "upload_ms": gpu_upload_ms,
                "query_ms": gpu_query_ms,
                "total_ms": gpu_upload_ms + gpu_query_ms,
                "qps": n_queries as f64 / (gpu_query_ms / 1000.0),
            }))
        } else {
            None
        }
    };
    #[cfg(not(feature = "cuda"))]
    let gpu_search: Option<serde_json::Value> = None;

    eprintln!("[splatsdb] Phase 5: Fused search...");
    let n_fused = n_queries.min(10);
    let t_fused = Instant::now();
    for i in 0..n_fused {
        let _ = store.find_neighbors_fused(&query_arr.row(i), 10);
    }
    let fused_ms = t_fused.elapsed().as_millis() as f64;

    let total_pipeline = ingest_ms + build_ms + search_ms;

    let payload = serde_json::json!({
            "pipeline": "gpu_transformer",
            "config": {
                "n_vectors": n_vectors,
                "dim": dim,
                "n_clusters": n_clusters,
                "n_queries": n_queries,
                "seed": seed,
                "dataset_mb": (n_vectors * dim * 4) as f64 / 1_048_576.0,
            },
            "transformer": {
                "original_count": stats.original_count,
                "n_splats": n_splats,
                "compression_ratio": (compression * 10000.0).round() / 10000.0,
                "transform_time_s": stats.transform_time_s,
                "total_ingest_ms": ingest_ms as u64,
            },
            "index_build_ms": build_ms as u64,
            "linear_search": {
                "total_ms": search_ms,
                "per_query_us": search_ms / n_queries as f64 * 1000.0,
                "qps": n_queries as f64 / (search_ms / 1000.0),
            },
            "fused_search": {
                "n_queries": n_fused,
                "total_ms": fused_ms,
                "per_query_us": fused_ms / n_fused as f64 * 1000.0,
            },
            "gpu_raw_search": gpu_search,
            "total_pipeline_ms": total_pipeline as u64,
            "timestamp": cli_timestamp_iso8601(),
            "git_hash": cli_git_hash(),
    });

    cli_emit_json(&payload, out.as_deref());
}

// ============================================================================
// bench-gaussian: A/B ranking — pure L2 vs kappa-weighted Gaussian score
// ============================================================================

/// Benchmark synthetic Gaussian splats: exact CPU linear scan (GT) vs
/// HNSW-candidate + Gaussian-rescore (CPU, and GPU-rescore when CUDA is
/// available). Reports recall@k, QPS, speedup and the % agreement between
/// the Gaussian-weighted top-k and the pure-L2 top-k.
#[allow(clippy::too_many_arguments)]
pub fn cmd_bench_gaussian(
    n_vectors: usize,
    n_queries: usize,
    top_k: usize,
    seed: u64,
    out: Option<PathBuf>,
    gt_out: Option<PathBuf>,
    hnsw_m: usize,
    hnsw_ef_construction: usize,
    hnsw_ef_search: usize,
    candidate_mult: usize,
) {
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    use std::time::Instant;

    let dim: usize = 128;
    let n_candidates = (top_k * candidate_mult).min(n_vectors).max(top_k);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    eprintln!(
        "[bench-gaussian] Generating {} synthetic splats (D={}, seed={}, kappa log-uniform [0.1,100])...",
        n_vectors, dim, seed
    );

    // Synthetic splats: mu uniform in [-1,1]^D, alpha uniform (0.05, 1.0],
    // kappa log-uniform in [0.1, 100] (matches dataset_transformer clamp).
    let mut mu = vec![0.0f32; n_vectors * dim];
    for v in mu.iter_mut() {
        *v = rng.gen_range(-1.0..1.0);
    }
    let alpha: Vec<f32> = (0..n_vectors)
        .map(|_| rng.gen_range(0.05..=1.0))
        .collect();
    let kappa: Vec<f32> = (0..n_vectors)
        .map(|_| (rng.gen_range(0.1f64.ln()..=100.0f64.ln())).exp() as f32)
        .collect();

    let mut queries = vec![0.0f32; n_queries * dim];
    for v in queries.iter_mut() {
        *v = rng.gen_range(-1.0..1.0);
    }

    // Gaussian score: higher = better. score_i = alpha_i * exp(-kappa_i * ||q - mu_i||^2).
    // Equivalent ranking to energy E = -log(sum ...) per query (monotone transform
    // per query since argmax_i of the sum's term dominates... actually argmin energy
    // != argmax single term; here we rank by individual splat affinity, the standard
    // SplatsDB "gaussian neighbor" semantics used by e_splats terms).
    let gaussian_score = |qi: usize, si: usize| -> f32 {
        let q = &queries[qi * dim..(qi + 1) * dim];
        let m = &mu[si * dim..(si + 1) * dim];
        let d2: f32 = q.iter().zip(m.iter()).map(|(a, b)| (a - b) * (a - b)).sum();
        alpha[si] * (-kappa[si] * d2).exp()
    };

    // ---- Path A: exact CPU linear scan over ALL splats (ground truth) ----
    eprintln!(
        "[bench-gaussian] Path A: exact linear CPU scan (GT), {} queries...",
        n_queries
    );
    let mut gt_ranking: Vec<Vec<usize>> = Vec::with_capacity(n_queries);
    let t_gt = Instant::now();
    for qi in 0..n_queries {
        let mut scored: Vec<(f32, usize)> = (0..n_vectors)
            .map(|si| (gaussian_score(qi, si), si))
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        gt_ranking.push(scored.into_iter().take(top_k).map(|(_, i)| i).collect());
    }
    let gt_ms = t_gt.elapsed().as_millis() as f64;

    // Optionally persist GT in the same bin format as bench-gpu/bench-hnsw.
    if let Some(ref p) = gt_out {
        let as_pairs: Vec<Vec<usize>> = gt_ranking.clone();
        match cli_save_ground_truth_bin(p, &as_pairs) {
            Ok(()) => eprintln!("[bench-gaussian] Saved GT to {}", p.display()),
            Err(e) => eprintln!("[bench-gaussian] Warning: could not save GT: {}", e),
        }
    }

    // ---- Path B: HNSW candidates (L2) + exact Gaussian rescore (CPU) ----
    eprintln!(
        "[bench-gaussian] Path B: building HNSW (m={}, ef_c={}) over splat centroids...",
        hnsw_m, hnsw_ef_construction
    );
    let mu_arr = ndarray::Array2::from_shape_vec((n_vectors, dim), mu.clone())
        .expect("mu shape");
    let t_build = Instant::now();
    let mut hnsw = splatsdb::hnsw_index::HNSWIndex::new(
        dim,
        hnsw_m,
        hnsw_ef_construction,
        hnsw_ef_search,
        "l2",
        seed,
    );
    use splatsdb::interfaces::VectorIndex;
    hnsw.build(mu_arr);
    let hnsw_build_ms = t_build.elapsed().as_millis() as f64;
    eprintln!(
        "[bench-gaussian] HNSW built in {:.0} ms ({} nodes)",
        hnsw_build_ms,
        hnsw.n_items()
    );

    eprintln!(
        "[bench-gaussian] Path B: HNSW top-{} candidates + Gaussian rescore (CPU)...",
        n_candidates
    );
    let mut hnsw_ranking: Vec<Vec<usize>> = Vec::with_capacity(n_queries);
    let t_hnsw = Instant::now();
    for qi in 0..n_queries {
        let q = ndarray::ArrayView1::from(&queries[qi * dim..(qi + 1) * dim]);
        let cand = hnsw.search(q, n_candidates);
        let mut scored: Vec<(f32, usize)> = cand
            .indices
            .iter()
            .map(|&si| (gaussian_score(qi, si), si))
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        hnsw_ranking.push(scored.into_iter().take(top_k).map(|(_, i)| i).collect());
    }
    let hnsw_ms = t_hnsw.elapsed().as_millis() as f64;

    // ---- Path C (optional): GPU L2 candidate distances + CPU Gaussian rescore ----
    #[cfg(feature = "cuda")]
    let gpu_json: Option<serde_json::Value> = {
        use splatsdb::gpu::cuda_kernel::GpuIndex;
        if let Some(mut gpu_idx) = GpuIndex::new() {
            eprintln!("[bench-gaussian] Path C: uploading splat centroids to GPU...");
            let t_up = Instant::now();
            gpu_idx.upload_dataset(&mu, n_vectors, dim);
            let upload_ms = t_up.elapsed().as_millis() as f64;

            eprintln!(
                "[bench-gaussian] Path C: GPU L2 distances for {} candidates + CPU rescore...",
                n_candidates
            );
            let mut gpu_ranking: Vec<Vec<usize>> = Vec::with_capacity(n_queries);
            let t_gpu = Instant::now();
            for qi in 0..n_queries {
                let q = &queries[qi * dim..(qi + 1) * dim];
                // Full L2 distance row from GPU (dataset persistent in VRAM).
                let dists = gpu_idx.l2_distances(q).unwrap_or_else(|| {
                    eprintln!("[bench-gaussian] GPU l2_distances failed for query {}", qi);
                    vec![f32::INFINITY; n_vectors]
                });
                // Candidates by pure L2, rescored by Gaussian affinity on CPU.
                let mut cand: Vec<(f32, usize)> =
                    dists.iter().enumerate().map(|(i, &d)| (d, i)).collect();
                cand.select_nth_unstable_by(n_candidates.min(n_vectors - 1), |a, b| {
                    a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
                });
                cand.truncate(n_candidates);
                let mut scored: Vec<(f32, usize)> = cand
                    .into_iter()
                    .map(|(_, si)| (gaussian_score(qi, si), si))
                    .collect();
                scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                gpu_ranking.push(scored.into_iter().take(top_k).map(|(_, i)| i).collect());
            }
            let gpu_ms = t_gpu.elapsed().as_millis() as f64;

            let recall_gpu = cli_recall_ranking(&gpu_ranking, &gt_ranking);
            let agreement_gpu_vs_l2 = cli_agreement_ranking(
                &gpu_ranking,
                &cli_pure_l2_topk(&queries, n_queries, &mu, n_vectors, dim, top_k),
            );
            Some(serde_json::json!({
                "upload_ms": upload_ms,
                "rescore_query_ms": gpu_ms,
                "qps": n_queries as f64 / (gpu_ms / 1000.0),
                "recall_at_k": round4(recall_gpu),
                "agreement_vs_l2_pct": round4(agreement_gpu_vs_l2),
                "speedup_vs_exact_cpu": gt_ms / gpu_ms.max(0.001),
            }))
        } else {
            eprintln!("[bench-gaussian] CUDA not available; Path C skipped");
            None
        }
    };
    #[cfg(not(feature = "cuda"))]
    let gpu_json: Option<serde_json::Value> = None;

    // ---- Pure-L2 top-k (for the A/B agreement metric) ----
    eprintln!(
        "[bench-gaussian] Computing pure-L2 top-{} for agreement metric...",
        top_k
    );
    let l2_ranking = cli_pure_l2_topk(&queries, n_queries, &mu, n_vectors, dim, top_k);

    // ---- Metrics ----
    let recall_hnsw = cli_recall_ranking(&hnsw_ranking, &gt_ranking);
    let agreement_hnsw_vs_l2 = cli_agreement_ranking(&hnsw_ranking, &l2_ranking);
    // Baseline: how much the EXACT gaussian ranking differs from pure L2 ranking.
    let agreement_gt_vs_l2 = cli_agreement_ranking(&gt_ranking, &l2_ranking);

    eprintln!(
        "[bench-gaussian] recall@{} HNSW+rescore = {:.4}, agreement(gauss vs L2) = {:.2}%",
        top_k, recall_hnsw, agreement_hnsw_vs_l2 * 100.0
    );

    let payload = serde_json::json!({
        "bench": "gaussian",
        "config": {
            "n_vectors": n_vectors,
            "dim": dim,
            "n_queries": n_queries,
            "top_k": top_k,
            "seed": seed,
            "n_candidates": n_candidates,
            "candidate_mult": candidate_mult,
            "kappa_range": [0.1, 100.0],
            "kappa_dist": "log_uniform",
            "alpha_range": [0.05, 1.0],
            "hnsw": {"m": hnsw_m, "ef_construction": hnsw_ef_construction, "ef_search": hnsw_ef_search},
        },
        "timestamp": cli_timestamp_iso8601(),
        "git_hash": cli_git_hash(),
        "gt_exact_cpu": {
            "total_ms": gt_ms,
            "per_query_us": gt_ms / n_queries as f64 * 1000.0,
            "qps": n_queries as f64 / (gt_ms / 1000.0),
        },
        "hnsw_gaussian_cpu": {
            "hnsw_build_ms": hnsw_build_ms,
            "total_ms": hnsw_ms,
            "per_query_us": hnsw_ms / n_queries as f64 * 1000.0,
            "qps": n_queries as f64 / (hnsw_ms / 1000.0),
            "recall_at_k": round4(recall_hnsw),
            "speedup_vs_exact_cpu": round4(gt_ms / hnsw_ms.max(0.001)),
            "agreement_vs_l2_pct": round4(agreement_hnsw_vs_l2),
        },
        "gpu_l2_gaussian_rescore": gpu_json,
        "ab_ranking": {
            "metric": "pct of top-k slots where kappa-weighted Gaussian ranking equals pure-L2 ranking",
            "agreement_gt_gaussian_vs_l2_pct": round4(agreement_gt_vs_l2),
            "agreement_hnsw_gaussian_vs_l2_pct": round4(agreement_hnsw_vs_l2),
        },
    });

    cli_emit_json(&payload, out.as_deref());
}

fn round4(v: f64) -> f64 {
    (v * 10000.0).round() / 10000.0
}

/// Exact top-k by pure squared-L2 distance over mu (parallel, deterministic).
fn cli_pure_l2_topk(
    queries: &[f32],
    n_queries: usize,
    mu: &[f32],
    n_vectors: usize,
    dim: usize,
    top_k: usize,
) -> Vec<Vec<usize>> {
    use rayon::prelude::*;
    (0..n_queries)
        .into_par_iter()
        .map(|qi| {
            let q = &queries[qi * dim..(qi + 1) * dim];
            let mut scored: Vec<(f32, usize)> = (0..n_vectors)
                .map(|si| {
                    let m = &mu[si * dim..(si + 1) * dim];
                    let d2: f32 =
                        q.iter().zip(m.iter()).map(|(a, b)| (a - b) * (a - b)).sum();
                    (d2, si)
                })
                .collect();
            scored.select_nth_unstable_by(top_k.min(n_vectors - 1), |a, b| {
                a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
            });
            scored.truncate(top_k);
            scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            scored.into_iter().map(|(_, i)| i).collect()
        })
        .collect()
}

/// Recall@k of an approximate ranking vs the exact ranking, averaged over queries.
fn cli_recall_ranking(approx: &[Vec<usize>], gt: &[Vec<usize>]) -> f64 {
    let n = approx.len().min(gt.len());
    if n == 0 {
        return 0.0;
    }
    let mut hits = 0usize;
    let mut total = 0usize;
    for i in 0..n {
        let k = approx[i].len().min(gt[i].len());
        let gt_set: std::collections::HashSet<usize> =
            gt[i].iter().take(k).copied().collect();
        total += k;
        for &idx in approx[i].iter().take(k) {
            if gt_set.contains(&idx) {
                hits += 1;
            }
        }
    }
    if total == 0 {
        0.0
    } else {
        hits as f64 / total as f64
    }
}

/// Fraction of top-k slots where two rankings share the same id (position-wise
/// is too strict for ties; we use set-overlap, same convention as recall).
fn cli_agreement_ranking(a: &[Vec<usize>], b: &[Vec<usize>]) -> f64 {
    cli_recall_ranking(a, b)
}

// ============================================================================
// Shared helpers for bench commands
// ============================================================================

/// Emit JSON to stdout (single line) and optionally to --out <file> (pretty).
fn cli_emit_json(payload: &serde_json::Value, out: Option<&Path>) {
    match out {
        Some(p) => {
            let pretty = serde_pretty(payload);
            if let Err(e) = std::fs::write(p, &pretty) {
                eprintln!("[splatsdb] Warning: failed to write {}: {}", p.display(), e);
            } else {
                eprintln!("[splatsdb] Results written to {}", p.display());
            }
            println!("{}", payload);
        }
        None => println!("{}", payload),
    }
}

fn serde_pretty(v: &serde_json::Value) -> String {
    serde_json::to_string_pretty(v).unwrap_or_else(|_| v.to_string())
}

/// RFC3339/ISO8601 UTC timestamp from the system clock (no extra deps).
fn cli_timestamp_iso8601() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs() as i64;
    let millis = now.subsec_millis();
    let (y, mo, d, h, mi, s) = civil_from_unix(secs);
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:03}Z",
        y, mo, d, h, mi, s, millis
    )
}

/// Days-from-civil inverse (Howard Hinnant's algorithm) — no chrono dep.
fn civil_from_unix(secs: i64) -> (i64, u32, u32, u32, u32, u32) {
    let days = secs.div_euclid(86_400);
    let sod = secs.rem_euclid(86_400);
    let h = (sod / 3600) as u32;
    let mi = ((sod % 3600) / 60) as u32;
    let s = (sod % 60) as u32;

    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d, h, mi, s)
}

/// Short git hash + branch read from .git (HEAD + packed-refs/refs), no git dep.
fn cli_git_hash() -> String {
    let read = |p: &Path| std::fs::read_to_string(p).ok().map(|s| s.trim().to_string());
    let head = match read(Path::new(".git/HEAD")) {
        Some(h) => h,
        None => return "unknown".to_string(),
    };
    let branch = head.strip_prefix("ref: ").map(|r| r.to_string());
    let sha = match branch {
        Some(ref r) => read(Path::new(&format!(".git/{}", r)))
            .or_else(|| {
                // packed-refs fallback
                let packed = read(Path::new(".git/packed-refs")).unwrap_or_default();
                packed
                    .lines()
                    .find(|l| l.ends_with(r))
                    .and_then(|l| l.split_whitespace().next().map(|s| s.to_string()))
            })
            .unwrap_or_else(|| "unknown".to_string()),
        None => head, // detached HEAD: HEAD contains the sha directly
    };
    let short: String = sha.chars().take(12).collect();
    match branch {
        Some(b) => format!("{} ({})", short, b),
        None => short,
    }
}

/// Load GT bin (same format as bench-hnsw): [u64 n_queries][u64 k][i64 data].
fn cli_load_ground_truth_bin(path: &Path) -> Result<(usize, usize, Vec<i64>), String> {
    use std::io::Read;
    let mut file =
        std::fs::File::open(path).map_err(|e| format!("Cannot open {}: {}", path.display(), e))?;
    let mut buf8 = [0u8; 8];
    file.read_exact(&mut buf8)
        .map_err(|e| format!("Read error: {}", e))?;
    let n_queries = usize::try_from(u64::from_le_bytes(buf8)).map_err(|_| "n_queries overflow")?;
    file.read_exact(&mut buf8)
        .map_err(|e| format!("Read error: {}", e))?;
    let k = usize::try_from(u64::from_le_bytes(buf8)).map_err(|_| "k overflow")?;
    let total = n_queries
        .checked_mul(k)
        .ok_or("overflow in n_queries*k")?;
    let mut data = vec![0i64; total];
    let bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut data);
    file.read_exact(bytes)
        .map_err(|e| format!("Read error: {}", e))?;
    Ok((n_queries, k, data))
}

/// Save GT bin in the same format: [u64 n_queries][u64 k][i64 data row-major].
fn cli_save_ground_truth_bin(path: &Path, ranking: &[Vec<usize>]) -> Result<(), String> {
    use std::io::Write;
    let n_queries = ranking.len();
    let k = ranking.first().map(|r| r.len()).unwrap_or(0);
    let mut out = std::fs::File::create(path)
        .map_err(|e| format!("Cannot create {}: {}", path.display(), e))?;
    out.write_all(&(n_queries as u64).to_le_bytes())
        .map_err(|e| e.to_string())?;
    out.write_all(&(k as u64).to_le_bytes())
        .map_err(|e| e.to_string())?;
    let flat: Vec<i64> = ranking
        .iter()
        .flat_map(|row| row.iter().map(|&i| i as i64))
        .collect();
    let bytes: &[u8] = bytemuck::cast_slice(&flat);
    out.write_all(bytes).map_err(|e| e.to_string())?;
    Ok(())
}

/// Exact CPU top-k brute force (rayon, set-ordered by distance). Used as the
/// honest CPU baseline in bench-gpu — NOTE: splatsdb::gpu::batch_search
/// silently dispatches to GPU when built with --features cuda, so calling it
/// as the "CPU" baseline measured the GPU twice (forensic finding R7).
#[allow(clippy::too_many_arguments)]
fn cli_cpu_topk(
    queries: &[f32],
    n_queries: usize,
    dataset: &[f32],
    n_vectors: usize,
    dim: usize,
    top_k: usize,
    metric: &str,
) -> Vec<Vec<usize>> {
    use rayon::prelude::*;
    (0..n_queries)
        .into_par_iter()
        .map(|qi| {
            let q = &queries[qi * dim..(qi + 1) * dim];
            let mut scored: Vec<(f32, usize)> = (0..n_vectors)
                .map(|vi| {
                    let v = &dataset[vi * dim..(vi + 1) * dim];
                    let d = if metric == "cosine" {
                        let dot: f32 = q.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
                        let qn: f32 = q.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-10);
                        let vn: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-10);
                        1.0 - dot / (qn * vn)
                    } else {
                        q.iter().zip(v.iter()).map(|(a, b)| (a - b) * (a - b)).sum::<f32>()
                    };
                    (d, vi)
                })
                .collect();
            scored.select_nth_unstable_by(top_k.min(n_vectors.saturating_sub(1)), |a, b| {
                a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
            });
            scored.truncate(top_k);
            scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            scored.into_iter().map(|(_, i)| i).collect()
        })
        .collect()
}

/// Compare two top-k result sets (unordered overlap == equality of sets).
fn cli_topk_sets_equal(
    a: &[Vec<usize>],
    b: &[Vec<usize>],
) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b.iter()).all(|(ra, rb)| {
        let sa: std::collections::HashSet<usize> = ra.iter().copied().collect();
        let sb: std::collections::HashSet<usize> = rb.iter().copied().collect();
        sa == sb
    })
}

/// recall@k of GPU results (index-only rows) vs GT indices.
fn cli_recall_at_k(
    results: &[Vec<usize>],
    gt_nq: usize,
    gt_k: usize,
    gt_idx: &[i64],
    top_k: usize,
) -> f64 {
    let effective_k = top_k.min(gt_k);
    let effective_n = results.len().min(gt_nq);
    if effective_k == 0 || effective_n == 0 {
        return 0.0;
    }
    let mut hits = 0usize;
    let mut total = 0usize;
    for (i, idxs) in results.iter().enumerate().take(effective_n) {
        let row = &gt_idx[i * gt_k..(i + 1) * gt_k];
        let gt_set: std::collections::HashSet<i64> =
            row[..effective_k].iter().copied().collect();
        total += effective_k;
        for &idx in idxs.iter().take(effective_k) {
            if gt_set.contains(&(idx as i64)) {
                hits += 1;
            }
        }
    }
    hits as f64 / total as f64
}
