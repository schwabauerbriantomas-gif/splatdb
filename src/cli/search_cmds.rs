//! Search and query command handlers.

use std::path::PathBuf;

use m2m_vector_search::M2MConfig;

use super::helpers::*;

pub fn cmd_search(data_dir: String, config: M2MConfig, query: String, k: usize, format: String) {
    let q = match parse_query(&query) {
        Ok(v) => v,
        Err(e) => { eprintln!("Error: {}", e); std::process::exit(1); }
    };
    let (store, _) = load_or_create_store(&data_dir, &config);
    let results = store.find_neighbors(&q.view(), k);
    println!("{}", format_neighbors(&results, &format));
}

pub fn cmd_search_file(data_dir: String, config: M2MConfig, input: PathBuf, k: usize, format: String) {
    let q = match load_query_bin(&input) {
        Ok(v) => v,
        Err(e) => { eprintln!("Error: {}", e); std::process::exit(1); }
    };
    let (store, _) = load_or_create_store(&data_dir, &config);
    let results = store.find_neighbors(&q.view(), k);
    println!("{}", format_neighbors(&results, &format));
}

pub fn cmd_fused_search(data_dir: String, config: M2MConfig, query: Option<String>, query_file: Option<PathBuf>, k: usize) {
    let q = resolve_query(query.as_deref(), query_file.as_ref());
    eprintln!("[m2m] Query dim: {}", q.len());
    let (store, _) = load_or_create_store(&data_dir, &config);
    eprintln!("[m2m] Store n_active: {}", store.n_active());
    let t0 = std::time::Instant::now();
    eprintln!("[m2m] Starting search...");
    let results = store.find_neighbors(&q.view(), k);
    let elapsed = t0.elapsed();
    eprintln!("[m2m] Search complete");
    let formatted = format_neighbors(&results, "json");
    eprintln!("[m2m] Fused search: {} results in {:?}", results.len(), elapsed);
    println!("{}", formatted);
}

pub fn cmd_hnsw_search(data_dir: String, dim: usize, max_splats: usize, query: Option<String>, query_file: Option<PathBuf>, k: usize) {
    let q = resolve_query(query.as_deref(), query_file.as_ref());
    let mut cfg = M2MConfig::advanced(None);
    cfg.latent_dim = dim;
    cfg.max_splats = max_splats;
    cfg.device = "cpu".to_string();
    cfg.enable_cuda = false;
    cfg.enable_gpu_search = false;
    cfg.finalize();

    let (mut store, _) = load_or_create_store(&data_dir, &cfg);
    eprintln!("[m2m] HNSW enabled in config: {}", cfg.enable_hnsw);
    eprintln!("[m2m] Store has HNSW: {}", store.has_hnsw());
    if !store.has_hnsw() {
        eprintln!("HNSW not enabled. Use 'advanced' or 'gpu' preset.");
        std::process::exit(1);
    }
    store.build_index();
    let t0 = std::time::Instant::now();
    let results = store.find_neighbors_fused(&q.view(), k);
    let elapsed = t0.elapsed();
    eprintln!("[m2m] HNSW search: {} results in {:?}", results.len(), elapsed);
    println!("{}", format_neighbors(&results, "json"));
}

pub fn cmd_lsh_search(data_dir: String, dim: usize, max_splats: usize, query: Option<String>, query_file: Option<PathBuf>, k: usize) {
    let q = resolve_query(query.as_deref(), query_file.as_ref());
    let mut cfg = M2MConfig {
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
    eprintln!("[m2m] LSH search: {} results in {:?}", results.len(), elapsed);
    println!("{}", format_neighbors(&results, "json"));
}

pub fn cmd_quant_search(data_dir: String, backend: String, _dim: usize, query: String, top_k: usize) {
    use m2m_vector_search::quantization::{QuantConfig, QuantizedStore};

    let q_vec = match parse_query(&query) {
        Ok(v) => v,
        Err(e) => { eprintln!("Error: {}", e); std::process::exit(1); }
    };

    let persist = make_persistence(&data_dir, &backend, false).ok();
    let vectors = if let Some(ref p) = persist {
        p.load_vectors("default").ok().flatten()
    } else { None };

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

    let entries: Vec<serde_json::Value> = results.iter().map(|(id, score)| {
        serde_json::json!({
            "id": id,
            "score": (score * 10000.0).round() / 10000.0,
        })
    }).collect();

    println!("{}", serde_json::json!({
        "query_time_ms": elapsed.as_millis() as u64,
        "n_results": entries.len(),
        "results": entries,
    }));
}
