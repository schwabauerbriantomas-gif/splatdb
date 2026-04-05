//! Index, ingest, and benchmark command handlers.

use std::path::PathBuf;

use splatdb::{SplatDBConfig, SplatStore};

use super::helpers::*;

pub fn cmd_index(data_dir: String, backend: String, config: SplatDBConfig, input: PathBuf, shard: String) {
    let vectors = match load_vectors_bin(&input) {
        Ok(v) => v,
        Err(e) => { eprintln!("Error: {}", e); std::process::exit(1); }
    };
    eprintln!("[splatdb] Loaded {} vectors ({}D) from {}", vectors.nrows(), vectors.ncols(), input.display());

    let persist = make_persistence(&data_dir, &backend, true)
        .unwrap_or_else(|e| { eprintln!("Storage error: {}", e); std::process::exit(1); });

    persist.save_vectors(&vectors, &shard)
        .unwrap_or_else(|e| { eprintln!("Save error: {}", e); std::process::exit(1); });

    let mut store = SplatStore::new(config);
    store.add_splat(&vectors);
    store.build_index();

    println!("{}", serde_json::json!({
        "status": "indexed",
        "shard": shard,
        "n_vectors": vectors.nrows(),
        "dim": vectors.ncols(),
        "n_active": store.n_active(),
        "entropy": (store.entropy() * 10000.0).round() / 10000.0,
    }));
}

pub fn cmd_ingest(config: SplatDBConfig, input: PathBuf, n_clusters: Option<usize>, seed: u64) {
    let vectors = match load_vectors_bin(&input) {
        Ok(v) => v,
        Err(e) => { eprintln!("Error: {}", e); std::process::exit(1); }
    };
    eprintln!("[splatdb] Ingesting {} vectors ({}D) via DatasetTransformer", vectors.nrows(), vectors.ncols());

    let mut store = SplatStore::new(config);
    let k = n_clusters.unwrap_or_else(|| (vectors.nrows() as f64).sqrt() as usize);
    let t0 = std::time::Instant::now();

    match store.ingest_with_transformer(&vectors, k, seed) {
        Ok((n_splats, compression, stats)) => {
            let elapsed = t0.elapsed();
            store.build_index();
            println!("{}", serde_json::json!({
                "status": "ingested",
                "method": "kmeans",
                "original_count": stats.original_count,
                "n_splats": n_splats,
                "compression_ratio": (compression * 10000.0).round() / 10000.0,
                "transform_time_s": (stats.transform_time_s * 10000.0).round() / 10000.0,
                "total_ingest_ms": elapsed.as_millis() as u64,
                "dim": vectors.ncols(),
            }));
        }
        Err(e) => { eprintln!("Ingest error: {}", e); std::process::exit(1); }
    }
}

pub fn cmd_ingest_hierarchical(config: SplatDBConfig, input: PathBuf, n_clusters: usize, min_cluster_size: usize, seed: u64) {
    let vectors = match load_vectors_bin(&input) {
        Ok(v) => v,
        Err(e) => { eprintln!("Error: {}", e); std::process::exit(1); }
    };
    eprintln!("[splatdb] Hierarchical ingest {} vectors ({}D)", vectors.nrows(), vectors.ncols());

    let mut store = SplatStore::new(config);
    let t0 = std::time::Instant::now();

    match store.ingest_hierarchical(&vectors, n_clusters, min_cluster_size, seed) {
        Ok((n_splats, compression, stats)) => {
            let elapsed = t0.elapsed();
            store.build_index();
            println!("{}", serde_json::json!({
                "status": "ingested",
                "method": "hierarchical_kmeans",
                "original_count": stats.original_count,
                "n_splats": n_splats,
                "compression_ratio": (compression * 10000.0).round() / 10000.0,
                "total_ingest_ms": elapsed.as_millis() as u64,
                "dim": vectors.ncols(),
            }));
        }
        Err(e) => { eprintln!("Ingest error: {}", e); std::process::exit(1); }
    }
}

pub fn cmd_ingest_leader(data_dir: String, backend: String, config: SplatDBConfig, input: PathBuf, target_clusters: usize, threshold: Option<f64>, seed: u64) {
    let vectors = match load_vectors_bin(&input) {
        Ok(v) => v,
        Err(e) => { eprintln!("Error: {}", e); std::process::exit(1); }
    };
    eprintln!("[splatdb] Leader clustering ingest {} vectors ({}D)", vectors.nrows(), vectors.ncols());

    let mut store = SplatStore::new(config);
    let t0 = std::time::Instant::now();

    match store.ingest_leader(&vectors, target_clusters, seed, threshold) {
        Ok((n_splats, compression, stats)) => {
            let elapsed = t0.elapsed();
            store.build_index();
            let persist = make_persistence(&data_dir, &backend, false).ok();
            if let Some(p) = persist {
                if let (Some(mu), Some(alpha), Some(kappa)) = (store.get_mu(), store.get_alpha(), store.get_kappa()) {
                    let n = store.n_active();
                    let vectors = mu.slice(ndarray::s![..n, ..]).to_owned();
                    if let Err(e) = p.save_vectors(&vectors, "default") {
                        eprintln!("Warning: save failed: {}", e);
                    }
                    if let Err(e) = p.save_energy_state(&vectors, &alpha.to_vec(), &kappa.to_vec()) {
                        eprintln!("Warning: energy save failed: {}", e);
                    }
                }
            }
            println!("{}", serde_json::json!({
                "status": "ingested",
                "method": "leader_clustering",
                "original_count": stats.original_count,
                "n_splats": n_splats,
                "compression_ratio": (compression * 10000.0).round() / 10000.0,
                "total_ingest_ms": elapsed.as_millis() as u64,
                "dim": vectors.ncols(),
                "threshold": threshold,
                "saved": true,
            }));
        }
        Err(e) => { eprintln!("Ingest error: {}", e); std::process::exit(1); }
    }
}

pub fn cmd_bench_gpu(n_vectors: usize, dim: usize, n_queries: usize, top_k: usize, metric: String) {
    #[cfg(not(feature = "cuda"))]
    {
        eprintln!("[splatdb] bench-gpu requires --features cuda");
        std::process::exit(1);
    }
    #[cfg(feature = "cuda")]
    {
    use std::time::Instant;
    use splatdb::gpu::cuda_kernel::GpuIndex;

    eprintln!("[splatdb] Generating {} vectors ({}D)...", n_vectors, dim);
    let mut rng = rand::thread_rng();
    let dataset: Vec<f32> = (0..n_vectors * dim)
        .map(|_| rand::Rng::gen_range(&mut rng, -1.0..1.0))
        .collect();
    let queries: Vec<f32> = (0..n_queries * dim)
        .map(|_| rand::Rng::gen_range(&mut rng, -1.0..1.0))
        .collect();

    eprintln!("[splatdb] CPU benchmark...");
    let t0 = Instant::now();
    let cpu_results = splatdb::gpu::batch_search(
        &queries, n_queries, &dataset, n_vectors, dim, top_k, &metric,
    );
    let cpu_ms = t0.elapsed().as_millis() as f64;

    let gpu_result = if let Some(mut gpu_idx) = GpuIndex::new() {
        eprintln!("[splatdb] Uploading {} vectors to GPU...", n_vectors);
        let t_upload = Instant::now();
        gpu_idx.upload_dataset(&dataset, n_vectors, dim);
        let upload_ms = t_upload.elapsed().as_millis() as f64;

        eprintln!("[splatdb] GPU query benchmark (dataset in VRAM)...");
        let t1 = Instant::now();
        let gpu_results = gpu_idx.topk_search(&queries, n_queries, top_k, &metric);
        let query_ms = t1.elapsed().as_millis() as f64;

        let gpu_results = match gpu_results {
            Some(r) => r,
            None => {
                eprintln!("[splatdb] GPU topk_search returned None");
                let t2 = Instant::now();
                let r = splatdb::gpu::batch_search(
                    &queries, n_queries, &dataset, n_vectors, dim, top_k, &metric,
                );
                let _ = query_ms + t2.elapsed().as_millis() as f64;
                r
            }
        };

        let matches = cpu_results.len() == gpu_results.len()
            && cpu_results.iter().zip(gpu_results.iter())
                .all(|(c, g)| c.0.first() == g.0.first());

        eprintln!("[splatdb] GPU/CPU results match: {}", matches);
        Some((upload_ms, query_ms))
    } else {
        None
    };

    let gpu_json = gpu_result.map(|(upload_ms, query_ms)| {
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
        })
    });

    println!("{}", serde_json::json!({
        "config": {
            "n_vectors": n_vectors,
            "dim": dim,
            "n_queries": n_queries,
            "top_k": top_k,
            "metric": metric,
            "dataset_mb": (n_vectors * dim * 4) as f64 / 1_048_576.0,
        },
        "cpu": {
            "total_ms": cpu_ms,
            "per_query_us": cpu_ms / n_queries as f64 * 1000.0,
            "qps": n_queries as f64 / (cpu_ms / 1000.0),
        },
        "gpu": gpu_json,
    }));
    } // cfg cuda
}

pub fn cmd_bench_gpu_ingest(n_vectors: usize, dim: usize, n_clusters: usize, n_queries: usize) {
    use std::time::Instant;

    eprintln!("[splatdb] Generating {} vectors ({}D)...", n_vectors, dim);
    let mut rng = rand::thread_rng();
    let raw: Vec<f32> = (0..n_vectors * dim)
        .map(|_| rand::Rng::gen_range(&mut rng, -1.0..1.0))
        .collect();
    let dataset = ndarray::Array2::from_shape_vec((n_vectors, dim), raw).unwrap();
    let queries: Vec<f32> = (0..n_queries * dim)
        .map(|_| rand::Rng::gen_range(&mut rng, -1.0..1.0))
        .collect();

    let mut ingest_cfg = SplatDBConfig::gpu(None);
    ingest_cfg.latent_dim = dim;
    ingest_cfg.max_splats = n_clusters * 2;
    ingest_cfg.device = "cpu".to_string();
    ingest_cfg.enable_cuda = false;
    ingest_cfg.enable_gpu_search = false;
    ingest_cfg.finalize();

    let mut store = SplatStore::new(ingest_cfg);
    eprintln!("[splatdb] Phase 1: DatasetTransformer ingest...");
    let t_ingest = Instant::now();
    let (n_splats, compression, stats) = store.ingest_with_transformer(&dataset, n_clusters, 42)
        .unwrap_or_else(|e| { eprintln!("Ingest error: {}", e); std::process::exit(1); });
    let ingest_ms = t_ingest.elapsed().as_millis() as f64;

    eprintln!("[splatdb] Phase 2: Building index...");
    let t_build = Instant::now();
    store.build_index();
    let build_ms = t_build.elapsed().as_millis() as f64;

    eprintln!("[splatdb] Phase 3: Search {} queries against {} splats...", n_queries, n_splats);
    let query_arr = ndarray::Array2::from_shape_vec((n_queries, dim), queries.clone()).expect("valid shape from args");
    let t_search = Instant::now();
    for i in 0..n_queries {
        let _ = store.find_neighbors(&query_arr.row(i), 10);
    }
    let search_ms = t_search.elapsed().as_millis() as f64;

    #[cfg(feature = "cuda")]
    let gpu_search: Option<serde_json::Value> = {
        if let Some(mut gpu_idx) = splatdb::gpu::cuda_kernel::GpuIndex::new() {
            let raw_flat: Vec<f32> = dataset.clone().into_raw_vec();
            eprintln!("[splatdb] Phase 4: GPU search on raw {} vectors...", n_vectors);
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
        } else { None }
    };
    #[cfg(not(feature = "cuda"))]
    let gpu_search: Option<serde_json::Value> = None;

    eprintln!("[splatdb] Phase 5: Fused search...");
    let n_fused = n_queries.min(10);
    let t_fused = Instant::now();
    for i in 0..n_fused {
        let _ = store.find_neighbors_fused(&query_arr.row(i), 10);
    }
    let fused_ms = t_fused.elapsed().as_millis() as f64;

    let total_pipeline = ingest_ms + build_ms + search_ms;

    println!("{}", serde_json::json!({
        "pipeline": "gpu_transformer",
        "config": {
            "n_vectors": n_vectors,
            "dim": dim,
            "n_clusters": n_clusters,
            "n_queries": n_queries,
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
    }));
}
