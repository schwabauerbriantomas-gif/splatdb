//! Data management command handlers (save, load, docs, backup, SOC, quant, status, GPU, serve, preset).

use std::path::PathBuf;

use splatdb::{SplatDBConfig, SplatStore};
use splatdb::quantization::{QuantConfig, QuantAlgorithm, QuantizedStore};

use super::helpers::*;

pub fn cmd_status(data_dir: String, max_splats: usize, config: SplatDBConfig, verbose: bool) {
    let (store, persist) = load_or_create_store(&data_dir, &config);
    let stats = store.get_statistics();

    let mut out = serde_json::json!({
        "n_active": stats.n_active,
        "embedding_dim": stats.embedding_dim,
        "entropy": (store.entropy() * 10000.0).round() / 10000.0,
        "max_splats": max_splats,
    });

    if verbose {
        if let Some(ref p) = persist {
            out["shards"] = serde_json::json!(p.list_shards());
        }
        let strategy = splatdb::interfaces::select_index_strategy(
            stats.n_active, stats.embedding_dim
        );
        out["index_strategy"] = serde_json::json!({
            "recommended": strategy.recommended,
            "reason": strategy.reason,
        });
    }

    println!("{}", serde_json::to_string_pretty(&out).expect("Failed to serialize status JSON"));
}

pub fn cmd_soc_check(data_dir: String, config: SplatDBConfig) {
    let (store, _) = load_or_create_store(&data_dir, &config);
    let mu = store.get_mu();
    let alpha = store.get_alpha();
    let kappa = store.get_kappa();

    if mu.is_none() || store.n_active() == 0 {
        println!("{}", serde_json::json!({"error": "No splats loaded"}));
        std::process::exit(1);
    }

    let energy_api = splatdb::ebm::energy_api::EBMEnergy::new();
    let mut soc = splatdb::ebm::soc::SOCEngine::new(energy_api, 0.7);
    soc.update_splats(mu.expect("mu checked above"), alpha.expect("alpha checked above"), kappa.expect("kappa checked above"));

    let report = soc.check_criticality();
    println!("{}", serde_json::json!({
        "state": format!("{:?}", report.state),
        "index": (report.index * 10000.0).round() / 10000.0,
        "energy_variance": (report.energy_variance * 10000.0).round() / 10000.0,
        "size_variance": (report.size_variance * 10000.0).round() / 10000.0,
        "needs_relaxation": report.needs_relaxation(),
        "needs_monitoring": report.needs_monitoring(),
    }));
}

pub fn cmd_soc_avalanche(data_dir: String, config: SplatDBConfig, seed: Option<usize>) {
    let (store, _) = load_or_create_store(&data_dir, &config);
    let mu = store.get_mu();
    let alpha = store.get_alpha();
    let kappa = store.get_kappa();

    if mu.is_none() || store.n_active() == 0 {
        println!("{}", serde_json::json!({"error": "No splats loaded"}));
        std::process::exit(1);
    }

    let energy_api = splatdb::ebm::energy_api::EBMEnergy::new();
    let mut soc = splatdb::ebm::soc::SOCEngine::new(energy_api, 0.7);
    soc.update_splats(mu.expect("mu checked above"), alpha.expect("alpha checked above"), kappa.expect("kappa checked above"));

    let result = soc.trigger_avalanche(seed);
    println!("{}", serde_json::json!({
        "affected_clusters": result.affected_clusters,
        "energy_released": (result.energy_released * 10000.0).round() / 10000.0,
        "duration_ms": (result.duration_ms * 1000.0).round() / 1000.0,
        "new_equilibrium": (result.new_equilibrium * 10000.0).round() / 10000.0,
    }));
}

pub fn cmd_soc_relax(data_dir: String, config: SplatDBConfig, iterations: usize) {
    let (store, _) = load_or_create_store(&data_dir, &config);
    let mu = store.get_mu();
    let alpha = store.get_alpha();
    let kappa = store.get_kappa();

    if mu.is_none() || store.n_active() == 0 {
        println!("{}", serde_json::json!({"error": "No splats loaded"}));
        std::process::exit(1);
    }

    let energy_api = splatdb::ebm::energy_api::EBMEnergy::new();
    let mut soc = splatdb::ebm::soc::SOCEngine::new(energy_api, 0.7);
    soc.update_splats(mu.expect("mu checked above"), alpha.expect("alpha checked above"), kappa.expect("kappa checked above"));

    let result = soc.relax(iterations);
    println!("{}", serde_json::json!({
        "initial_energy": (result.initial_energy * 10000.0).round() / 10000.0,
        "final_energy": (result.final_energy * 10000.0).round() / 10000.0,
        "energy_delta": (result.energy_delta * 10000.0).round() / 10000.0,
        "iterations": result.iterations,
        "improved": result.improved,
    }));
}

pub fn cmd_save(data_dir: String, config: SplatDBConfig, shard: String) {
    let (store, persist) = load_or_create_store(&data_dir, &config);
    let persist = match persist {
        Some(p) => p,
        None => { eprintln!("Error: no persistence layer"); std::process::exit(1); }
    };

    let mu = store.get_mu();
    let alpha = store.get_alpha();
    let kappa = store.get_kappa();

    if let (Some(mu), Some(alpha), Some(kappa)) = (mu, alpha, kappa) {
        let n = store.n_active();
        let vectors = mu.slice(ndarray::s![..n, ..]).to_owned();
        persist.save_vectors(&vectors, &shard)
            .unwrap_or_else(|e| { eprintln!("Save error: {}", e); std::process::exit(1); });
        persist.save_energy_state(&vectors, &alpha.to_vec(), &kappa.to_vec())
            .unwrap_or_else(|e| { eprintln!("Energy save error: {}", e); std::process::exit(1); });
        println!("{}", serde_json::json!({
            "status": "saved",
            "shard": shard,
            "n_vectors": n,
        }));
    } else {
        eprintln!("No data to save");
        std::process::exit(1);
    }
}

pub fn cmd_load(data_dir: String, backend: String, shard: String) {
    let persist = make_persistence(&data_dir, &backend, true)
        .unwrap_or_else(|e| { eprintln!("Storage error: {}", e); std::process::exit(1); });

    match persist.load_vectors(&shard) {
        Ok(Some(vectors)) => {
            println!("{}", serde_json::json!({
                "status": "loaded",
                "shard": shard,
                "n_vectors": vectors.nrows(),
                "dim": vectors.ncols(),
            }));
        }
        Ok(None) => {
            println!("{}", serde_json::json!({"status": "not_found", "shard": shard}));
        }
        Err(e) => {
            eprintln!("Load error: {}", e);
            std::process::exit(1);
        }
    }
}

pub fn cmd_list(data_dir: String, backend: String) {
    let persist = make_persistence(&data_dir, &backend, false)
        .unwrap_or_else(|e| { eprintln!("Storage error: {}", e); std::process::exit(1); });
    let shards = persist.list_shards();
    println!("{}", serde_json::json!({"shards": shards}));
}

pub fn cmd_doc_add(data_dir: String, backend: String, id: String, text: String, metadata: Option<String>) {
    let persist = make_persistence(&data_dir, &backend, true)
        .unwrap_or_else(|e| { eprintln!("Storage error: {}", e); std::process::exit(1); });
    let meta_json = metadata.as_deref()
        .and_then(|s| serde_json::from_str(s).ok());
    persist.save_metadata(&id, 0, 0, meta_json.as_ref(), Some(&text))
        .unwrap_or_else(|e| { eprintln!("Error: {}", e); std::process::exit(1); });
    println!("{}", serde_json::json!({"status": "ok", "id": id}));
}

pub fn cmd_doc_get(data_dir: String, backend: String, id: String) {
    let persist = make_persistence(&data_dir, &backend, false)
        .unwrap_or_else(|e| { eprintln!("Storage error: {}", e); std::process::exit(1); });
    match persist.get_metadata(&id) {
        Ok(Some(meta)) => println!("{}", serde_json::to_string_pretty(&meta).expect("Failed to serialize metadata")),
        Ok(None) => println!("{}", serde_json::json!({"status": "not_found", "id": id})),
        Err(e) => { eprintln!("Error: {}", e); std::process::exit(1); }
    }
}

pub fn cmd_doc_del(data_dir: String, backend: String, id: String) {
    let persist = make_persistence(&data_dir, &backend, true)
        .unwrap_or_else(|e| { eprintln!("Storage error: {}", e); std::process::exit(1); });
    match persist.soft_delete(&id) {
        Ok(true) => println!("{}", serde_json::json!({"status": "deleted", "id": id})),
        Ok(false) => println!("{}", serde_json::json!({"status": "not_found", "id": id})),
        Err(e) => { eprintln!("Error: {}", e); std::process::exit(1); }
    }
}

pub fn cmd_backup(data_dir: String, backend: String, output: PathBuf) {
    let persist = make_persistence(&data_dir, &backend, false)
        .unwrap_or_else(|e| { eprintln!("Storage error: {}", e); std::process::exit(1); });
    match persist.backup(output.to_str().unwrap_or("./backup")) {
        Ok(path) => println!("{}", serde_json::json!({"status": "backup_ok", "path": path})),
        Err(e) => { eprintln!("Backup error: {}", e); std::process::exit(1); }
    }
}

pub fn cmd_quant_index(input: PathBuf, bits: u8, algorithm: String, seed: u64) {
    let vectors = match load_vectors_bin(&input) {
        Ok(v) => v,
        Err(e) => { eprintln!("Error: {}", e); std::process::exit(1); }
    };
    let dim = vectors.ncols();
    eprintln!("[splatdb] Quantizing {} vectors ({}D, {}-bit)", vectors.nrows(), dim, bits);

    let algo = match algorithm.as_str() {
        "polar" => QuantAlgorithm::PolarQuant,
        _ => QuantAlgorithm::TurboQuant,
    };

    let projections = match algo {
        QuantAlgorithm::TurboQuant => (dim / 4).max(4),
        QuantAlgorithm::PolarQuant => 0,
    };

    let qcfg = QuantConfig { bits, projections, seed, algorithm: algo };
    let mut qstore = QuantizedStore::new(dim, qcfg).expect("Failed to create QuantizedStore");
    let t0 = std::time::Instant::now();
    let indexed = qstore.add_batch(&vectors, 0);
    let elapsed = t0.elapsed();

    let ratio = qstore.compression_ratio();
    let mem = qstore.memory_usage();
    let raw_mem = dim * 4 * indexed;

    println!("{}", serde_json::json!({
        "status": "quantized",
        "algorithm": algorithm,
        "bits": bits,
        "n_vectors": indexed,
        "dim": dim,
        "compression_ratio": (ratio * 100.0).round() / 100.0,
        "memory_quantized_bytes": mem,
        "memory_raw_bytes": raw_mem,
        "encode_time_ms": elapsed.as_millis() as u64,
    }));
}

pub fn cmd_quant_status(dim: usize) {
    let qcfg = QuantConfig::for_search(dim);
    let qstore = QuantizedStore::new(dim, qcfg).expect("Failed to create QuantizedStore");

    let cfg_4bit = QuantConfig::for_compression(dim);
    let qs_4bit = QuantizedStore::new(dim, cfg_4bit).expect("Failed to create QuantizedStore");

    let cfg_3bit = QuantConfig::for_max_compression(dim);
    let qs_3bit = QuantizedStore::new(dim, cfg_3bit).expect("Failed to create QuantizedStore");

    println!("{}", serde_json::json!( {
        "dim": dim,
        "presets": {
            "search_8bit": {
                "algorithm": "TurboQuant",
                "bits": 8,
                "compression_ratio": (qstore.compression_ratio() * 100.0).round() / 100.0,
                "bytes_per_vector": (qstore.compression_ratio().recip() * dim as f32 * 4.0) as usize,
            },
            "balanced_4bit": {
                "algorithm": "TurboQuant",
                "bits": 4,
                "compression_ratio": (qs_4bit.compression_ratio() * 100.0).round() / 100.0,
                "bytes_per_vector": (qs_4bit.compression_ratio().recip() * dim as f32 * 4.0) as usize,
            },
            "max_compression_3bit": {
                "algorithm": "PolarQuant",
                "bits": 3,
                "compression_ratio": (qs_3bit.compression_ratio() * 100.0).round() / 100.0,
                "bytes_per_vector": (qs_3bit.compression_ratio().recip() * dim as f32 * 4.0) as usize,
            },
        },
        "features": [
            "Data-oblivious: no training or calibration needed",
            "Deterministic: same (dim, bits, seed) always produces same codes",
            "Instant indexing: no offline training phase",
            "Unbiased inner product estimation",
        ],
    }));
}

pub fn cmd_gpu_info() {
    let cuda_available = splatdb::gpu::is_cuda_available();
    let gpu_info = splatdb::gpu::gpu_info();
    let device = splatdb::config::detect_device();

    println!("{}", serde_json::json!({
        "device": device,
        "cuda_available": cuda_available,
        "gpu_info": gpu_info,
        "cuda_feature": {
            "compiled": cfg!(feature = "cuda"),
        },
    }));
}

pub fn cmd_preset_info(preset: Option<String>) {
    let presets = vec![
        ("simple", SplatDBConfig::simple(None)),
        ("advanced", SplatDBConfig::advanced(None)),
        ("training", SplatDBConfig::training(None)),
        ("distributed", SplatDBConfig::distributed(None)),
        ("gpu", SplatDBConfig::gpu(None)),
    ];

    let filtered: Vec<_> = if let Some(ref name) = preset {
        let lower = name.to_lowercase();
        presets.into_iter().filter(|(n, _)| *n == lower).collect()
    } else {
        presets
    };

    let info: Vec<serde_json::Value> = filtered.iter().map(|(name, c)| {
        let mut store_config = c.clone();
        store_config.device = "cpu".to_string();
        store_config.enable_cuda = false;
        store_config.finalize();
        let store = SplatStore::new(store_config);
        serde_json::json!({
            "preset": name,
            "max_splats": c.max_splats,
            "latent_dim": c.latent_dim,
            "subsystems": {
                "quantization": c.enable_quantization,
                "hnsw": c.enable_hnsw,
                "lsh": c.enable_lsh,
                "semantic_memory": c.enable_semantic_memory,
                "graph": c.enable_graph,
                "cuda": c.enable_cuda,
                "gpu_search": c.enable_gpu_search,
            },
            "runtime": {
                "has_quantization": store.has_quantization(),
                "has_hnsw": store.has_hnsw(),
                "has_lsh": store.has_lsh(),
                "has_semantic_memory": store.has_semantic_memory(),
            },
        })
    }).collect();

    println!("{}", serde_json::to_string_pretty(&info).expect("Failed to serialize preset info"));
}

pub fn cmd_serve(port: u16) {
    println!("[splatdb] Starting API server on port {}...", port);
    let rt = tokio::runtime::Runtime::new()
        .expect("Failed to create tokio runtime");
    rt.block_on(async {
        if let Err(e) = splatdb::api_server::run_server(port).await {
            eprintln!("[splatdb] Server error: {}", e);
            std::process::exit(1);
        }
    });
}
