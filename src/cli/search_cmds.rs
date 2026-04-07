//! Search and query command handlers.

use std::io::Read;
use std::path::PathBuf;

use splatdb::SplatDBConfig;

use super::helpers::*;

pub fn cmd_search(
    data_dir: String,
    config: SplatDBConfig,
    query: String,
    k: usize,
    format: String,
) {
    let q = match parse_query(&query) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    };
    let (store, _) = load_or_create_store(&data_dir, &config);
    let results = store.find_neighbors(&q.view(), k);
    println!("{}", format_neighbors(&results, &format));
}

pub fn cmd_search_file(
    data_dir: String,
    config: SplatDBConfig,
    input: PathBuf,
    k: usize,
    format: String,
) {
    let q = match load_query_bin(&input) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    };
    let (store, _) = load_or_create_store(&data_dir, &config);
    let results = store.find_neighbors(&q.view(), k);
    println!("{}", format_neighbors(&results, &format));
}

pub fn cmd_fused_search(
    data_dir: String,
    config: SplatDBConfig,
    query: Option<String>,
    query_file: Option<PathBuf>,
    k: usize,
) {
    let q = resolve_query(query.as_deref(), query_file.as_ref());
    eprintln!("[splatdb] Query dim: {}", q.len());
    let (store, _) = load_or_create_store(&data_dir, &config);
    eprintln!("[splatdb] Store n_active: {}", store.n_active());
    let t0 = std::time::Instant::now();
    eprintln!("[splatdb] Starting search...");
    let results = store.find_neighbors(&q.view(), k);
    let elapsed = t0.elapsed();
    eprintln!("[splatdb] Search complete");
    let formatted = format_neighbors(&results, "json");
    eprintln!(
        "[splatdb] Fused search: {} results in {:?}",
        results.len(),
        elapsed
    );
    println!("{}", formatted);
}

pub fn cmd_hnsw_search(
    data_dir: String,
    dim: usize,
    max_splats: usize,
    query: Option<String>,
    query_file: Option<PathBuf>,
    k: usize,
) {
    let q = resolve_query(query.as_deref(), query_file.as_ref());
    let mut cfg = SplatDBConfig::advanced(None);
    cfg.latent_dim = dim;
    cfg.max_splats = max_splats;
    cfg.device = "cpu".to_string();
    cfg.enable_cuda = false;
    cfg.enable_gpu_search = false;
    cfg.finalize();

    let (mut store, _) = load_or_create_store(&data_dir, &cfg);
    eprintln!("[splatdb] HNSW enabled in config: {}", cfg.enable_hnsw);
    eprintln!("[splatdb] Store has HNSW: {}", store.has_hnsw());
    if !store.has_hnsw() {
        eprintln!("HNSW not enabled. Use 'advanced' or 'gpu' preset.");
        std::process::exit(1);
    }
    // HNSW was already loaded or built in load_or_create_store via build_index_if_needed.
    // Only rebuild as a fallback if somehow not built.
    if store.hnsw_is_built() {
        eprintln!("[splatdb] Using pre-built HNSW index");
    } else {
        eprintln!("[splatdb] HNSW index not built, building now...");
        store.build_index_with_save(Some(&data_dir));
    }
    let t0 = std::time::Instant::now();
    let results = store.find_neighbors_fused(&q.view(), k);
    let elapsed = t0.elapsed();
    eprintln!(
        "[splatdb] HNSW search: {} results in {:?}",
        results.len(),
        elapsed
    );
    println!("{}", format_neighbors(&results, "json"));
}

pub fn cmd_lsh_search(
    data_dir: String,
    dim: usize,
    max_splats: usize,
    query: Option<String>,
    query_file: Option<PathBuf>,
    k: usize,
) {
    let q = resolve_query(query.as_deref(), query_file.as_ref());
    let mut cfg = SplatDBConfig {
        latent_dim: dim,
        max_splats,
        enable_lsh: true,
        device: "cpu".to_string(),
        enable_cuda: false,
        ..Default::default()
    };
    cfg.finalize();

    let (mut store, _) = load_or_create_store(&data_dir, &cfg);
    if !store.has_lsh() {
        eprintln!("LSH not enabled.");
        std::process::exit(1);
    }
    store.build_index();
    let t0 = std::time::Instant::now();
    let results = store.find_neighbors_fused(&q.view(), k);
    let elapsed = t0.elapsed();
    eprintln!(
        "[splatdb] LSH search: {} results in {:?}",
        results.len(),
        elapsed
    );
    println!("{}", format_neighbors(&results, "json"));
}

pub fn cmd_quant_search(
    data_dir: String,
    backend: String,
    _dim: usize,
    query: String,
    top_k: usize,
) {
    use splatdb::quantization::{QuantConfig, QuantizedStore};

    let q_vec = match parse_query(&query) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    };

    let persist = make_persistence(&data_dir, &backend, false).ok();
    let vectors = if let Some(ref p) = persist {
        p.load_vectors("default").ok().flatten()
    } else {
        None
    };

    let vectors = match vectors {
        Some(v) => v,
        None => {
            eprintln!("Error: no vectors found. Run 'quant-index' first.");
            std::process::exit(1);
        }
    };

    let dim = vectors.ncols();
    let qcfg = QuantConfig::for_search(dim);
    let mut qstore = QuantizedStore::new(dim, qcfg).unwrap();
    qstore.add_batch(&vectors, 0);

    let t0 = std::time::Instant::now();
    let results = qstore.search(&q_vec.view(), top_k);
    let elapsed = t0.elapsed();

    let entries: Vec<serde_json::Value> = results
        .iter()
        .map(|(id, score)| {
            serde_json::json!({
                "id": id,
                "score": (score * 10000.0).round() / 10000.0,
            })
        })
        .collect();

    println!(
        "{}",
        serde_json::json!({
            "query_time_ms": elapsed.as_millis() as u64,
            "n_results": entries.len(),
            "results": entries,
        })
    );
}

/// Load binary ground truth: [u64 n_queries][u64 k][i64 data]
fn load_ground_truth_bin(path: &PathBuf) -> Result<(usize, usize, Vec<i64>), String> {
    let mut file = std::fs::File::open(path)
        .map_err(|e| format!("Cannot open {}: {}", path.display(), e))?;
    let mut buf8 = [0u8; 8];
    file.read_exact(&mut buf8)
        .map_err(|e| format!("Read error: {}", e))?;
    let n_queries = usize::try_from(u64::from_le_bytes(buf8))
        .map_err(|_| "n_queries overflow")?;
    file.read_exact(&mut buf8)
        .map_err(|e| format!("Read error: {}", e))?;
    let k = usize::try_from(u64::from_le_bytes(buf8))
        .map_err(|_| "k overflow")?;

    let total = n_queries.checked_mul(k).ok_or("overflow in n_queries*k")?;
    let mut data = vec![0i64; total];
    let bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut data);
    file.read_exact(bytes)
        .map_err(|e| format!("Read error: {}", e))?;
    Ok((n_queries, k, data))
}

pub fn cmd_bench_hnsw(
    train: PathBuf,
    queries: PathBuf,
    gt: Option<PathBuf>,
    dim: usize,
    k: usize,
    samples: Option<usize>,
    data_dir: Option<String>,
    max_splats: usize,
) {
    use super::helpers::load_vectors_bin;
    use std::time::Instant;

    eprintln!("[bench-hnsw] Loading training vectors from {}...", train.display());
    let t0 = Instant::now();
    let vectors = load_vectors_bin(&train).unwrap_or_else(|e| {
        eprintln!("Error loading training vectors: {}", e);
        std::process::exit(1);
    });
    let load_train_us = t0.elapsed().as_micros() as u64;
    eprintln!(
        "[bench-hnsw] Loaded {} vectors (dim={}) in {}ms",
        vectors.nrows(),
        vectors.ncols(),
        load_train_us / 1000,
    );

    // Create store with advanced preset (HNSW enabled)
    let dir = data_dir.unwrap_or_else(|| "./bench_temp/hnsw_bench".to_string());
    let mut cfg = SplatDBConfig::advanced(None);
    cfg.latent_dim = dim;
    cfg.max_splats = max_splats;
    cfg.device = "cpu".to_string();
    cfg.enable_cuda = false;
    cfg.enable_gpu_search = false;
    cfg.finalize();

    eprintln!("[bench-hnsw] Creating SplatStore with advanced preset (HNSW enabled)...");
    let mut store = splatdb::SplatStore::new(cfg.clone());

    // Add training vectors to store
    store.add_splat(&vectors);
    eprintln!("[bench-hnsw] Store has {} active splats", store.n_active());

    // Build or load HNSW index
    let t1 = Instant::now();
    let built = store.build_index_if_needed(&dir);
    let index_us = t1.elapsed().as_micros() as u64;
    if built {
        eprintln!("[bench-hnsw] Index ready in {}ms", index_us / 1000);
    } else {
        eprintln!("[bench-hnsw] Warning: no index built (no active splats?)");
    }

    // Load query vectors
    eprintln!("[bench-hnsw] Loading query vectors from {}...", queries.display());
    let all_queries = load_vectors_bin(&queries).unwrap_or_else(|e| {
        eprintln!("Error loading query vectors: {}", e);
        std::process::exit(1);
    });
    let n_total = all_queries.nrows();
    let n_samples = samples.unwrap_or(n_total).min(n_total);
    eprintln!(
        "[bench-hnsw] {} total queries, testing {}",
        n_total, n_samples,
    );

    // Load ground truth if provided
    let gt_data = gt.as_ref().and_then(|p| {
        match load_ground_truth_bin(p) {
            Ok((nq, gt_k, indices)) => {
                eprintln!(
                    "[bench-hnsw] Loaded ground truth: {} queries, k={}",
                    nq, gt_k,
                );
                Some((nq, gt_k, indices))
            }
            Err(e) => {
                eprintln!("[bench-hnsw] Warning: failed to load ground truth: {}", e);
                None
            }
        }
    });

    // Warm up with a few queries
    if n_samples > 0 {
        let _ = store.find_neighbors_fused(&all_queries.row(0), k);
    }

    // Run benchmark: measure search-only latency
    let mut latencies_us: Vec<u64> = Vec::with_capacity(n_samples);
    let mut all_predicted: Vec<Vec<usize>> = Vec::with_capacity(n_samples);

    eprintln!("[bench-hnsw] Running {} queries (k={})...", n_samples, k);
    let t_total = Instant::now();
    for i in 0..n_samples {
        let t_q = Instant::now();
        let results = store.find_neighbors_fused(&all_queries.row(i), k);
        let lat = t_q.elapsed().as_micros() as u64;
        latencies_us.push(lat);
        all_predicted.push(results.iter().map(|r| r.index).collect());
    }
    let total_search_us = t_total.elapsed().as_micros() as u64;

    // Sort latencies for percentiles
    latencies_us.sort_unstable();

    let p50 = latencies_us[n_samples * 50 / 100];
    let p95 = latencies_us[n_samples * 95 / 100];
    let p99 = latencies_us[n_samples * 99 / 100.min(n_samples)];
    let median = latencies_us[n_samples / 2];
    let total_search_s = total_search_us as f64 / 1_000_000.0;
    let qps = if total_search_s > 0.0 {
        n_samples as f64 / total_search_s
    } else {
        0.0
    };

    // Compute recall@K if ground truth available
    let recall = if let Some((gt_nq, gt_k, ref gt_indices)) = gt_data {
        let effective_k = k.min(gt_k);
        let effective_n = n_samples.min(gt_nq);
        let mut hits: usize = 0;
        let mut total_possible: usize = 0;

        for i in 0..effective_n {
            let gt_row_start = i * gt_k;
            let gt_set: std::collections::HashSet<i64> = gt_indices[gt_row_start..gt_row_start + effective_k]
                .iter()
                .copied()
                .collect();

            let pred = &all_predicted[i];
            total_possible += effective_k;
            for &idx in pred.iter().take(effective_k) {
                if gt_set.contains(&(idx as i64)) {
                    hits += 1;
                }
            }
        }
        hits as f64 / total_possible as f64
    } else {
        -1.0 // indicates no GT available
    };

    let recall_val = if recall < 0.0 {
        serde_json::Value::Null
    } else {
        serde_json::json!((recall * 10000.0).round() / 10000.0)
    };

    eprintln!("[bench-hnsw] Done. Total search: {:.2}s", total_search_s);

    let result = serde_json::json!({
        "bench": "hnsw",
        "config": {
            "n_train_vectors": vectors.nrows(),
            "dim": vectors.ncols(),
            "n_queries": n_samples,
            "k": k,
            "preset": "advanced",
        },
        "load_train_ms": load_train_us / 1000,
        "index_build_ms": index_us / 1000,
        "search": {
            "total_ms": total_search_us / 1000,
            "p50_us": p50,
            "p95_us": p95,
            "p99_us": p99,
            "median_us": median,
            "qps": (qps * 100.0).round() / 100.0,
        },
        "recall_at_k": recall_val,
    });

    println!("{}", serde_json::to_string_pretty(&result).unwrap());
}
