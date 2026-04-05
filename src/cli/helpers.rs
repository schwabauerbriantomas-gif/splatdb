//! Shared helpers for CLI command handlers.

use ndarray::{Array1, Array2};
use std::path::PathBuf;

use splatdb::{
    storage::persistence::{SplatDBPersistence, StorageBackend},
    SplatDBConfig, SplatStore,
};

pub fn make_config(dim: usize, max_splats: usize) -> SplatDBConfig {
    SplatDBConfig {
        latent_dim: dim,
        max_splats,
        ..Default::default()
    }
}

pub fn resolve_backend(name: &str) -> StorageBackend {
    match name {
        "json" | "json-file" => StorageBackend::JsonFile,
        "sqlite" => StorageBackend::Sqlite,
        _ => StorageBackend::Sqlite,
    }
}

pub fn make_persistence(
    data_dir: &str,
    backend: &str,
    rw: bool,
) -> Result<SplatDBPersistence, Box<dyn std::error::Error + Send + Sync>> {
    SplatDBPersistence::with_backend(data_dir, resolve_backend(backend), rw)
}

pub fn load_or_create_store(
    data_dir: &str,
    config: &SplatDBConfig,
) -> (SplatStore, Option<SplatDBPersistence>) {
    let persist = make_persistence(data_dir, "sqlite", true).ok();
    let mut store = SplatStore::new(config.clone());

    if let Some(ref p) = persist {
        if let Ok(Some(vectors)) = p.load_vectors("default") {
            store.add_splat(&vectors);
            store.build_index();
            eprintln!(
                "[splatdb] Loaded {} vectors from default shard",
                vectors.nrows()
            );
        }
    }

    (store, persist)
}

pub fn parse_query(s: &str) -> Result<Array1<f32>, String> {
    let vals: Result<Vec<f32>, _> = s.split(',').map(|v| v.trim().parse()).collect();
    match vals {
        Ok(v) => Ok(Array1::from(v)),
        Err(e) => Err(format!("Invalid query vector: {}", e)),
    }
}

pub fn resolve_query(query: Option<&str>, query_file: Option<&PathBuf>) -> Array1<f32> {
    match (query, query_file) {
        (Some(q), _) => parse_query(q).unwrap_or_else(|e| {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }),
        (_, Some(f)) => load_query_bin(f).unwrap_or_else(|e| {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }),
        (None, None) => {
            eprintln!("Error: provide --query or --query-file");
            std::process::exit(1);
        }
    }
}

pub fn load_query_bin(path: &PathBuf) -> Result<Array1<f32>, String> {
    use std::io::Read;
    let mut file =
        std::fs::File::open(path).map_err(|e| format!("Cannot open {}: {}", path.display(), e))?;
    let mut buf8 = [0u8; 8];
    file.read_exact(&mut buf8)
        .map_err(|e| format!("Read error: {}", e))?;
    let rows = u64::from_le_bytes(buf8) as usize;
    file.read_exact(&mut buf8)
        .map_err(|e| format!("Read error: {}", e))?;
    let cols = u64::from_le_bytes(buf8) as usize;

    if rows != 1 {
        return Err(format!("Query file must have exactly 1 row, got {}", rows));
    }

    let mut data = vec![0.0f32; cols];
    // SAFETY: `data` is a valid Vec<f32> with `cols` elements. Casting to &mut [u8] of length
    // `cols * 4` is valid because f32 has no padding (size == 4, align == 4) and the slice does
    // not outlive `data`. We read exactly `cols * 4` bytes, filling the entire buffer.
    let bytes: &mut [u8] =
        unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, cols * 4) };
    file.read_exact(bytes)
        .map_err(|e| format!("Read error: {}", e))?;
    Ok(Array1::from(data))
}

pub fn load_vectors_bin(path: &PathBuf) -> Result<Array2<f32>, String> {
    use std::io::Read;
    let mut file =
        std::fs::File::open(path).map_err(|e| format!("Cannot open {}: {}", path.display(), e))?;
    let mut buf8 = [0u8; 8];
    file.read_exact(&mut buf8)
        .map_err(|e| format!("Read error: {}", e))?;
    let rows = u64::from_le_bytes(buf8) as usize;
    file.read_exact(&mut buf8)
        .map_err(|e| format!("Read error: {}", e))?;
    let cols = u64::from_le_bytes(buf8) as usize;

    let mut data = vec![0.0f32; rows * cols];
    // SAFETY: `data` is a valid Vec<f32> with `rows * cols` elements. Casting to &mut [u8] of
    // length `data.len() * 4` is valid because f32 is 4 bytes with no padding. The slice does not
    // outlive `data` and we read exactly that many bytes.
    let bytes: &mut [u8] =
        unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, data.len() * 4) };
    file.read_exact(bytes)
        .map_err(|e| format!("Read error: {}", e))?;
    Array2::from_shape_vec((rows, cols), data).map_err(|e| format!("Shape error: {}", e))
}

pub fn format_neighbors(results: &[splatdb::splats::NeighborResult], format: &str) -> String {
    match format {
        "json" => {
            let entries: Vec<serde_json::Value> = results
                .iter()
                .map(|r| {
                    serde_json::json!({
                        "index": r.index,
                        "distance": (r.distance * 10000.0).round() / 10000.0,
                        "alpha": (r.alpha * 10000.0).round() / 10000.0,
                        "kappa": (r.kappa * 10000.0).round() / 10000.0,
                    })
                })
                .collect();
            serde_json::to_string_pretty(&entries).unwrap_or_else(|_| "[]".into())
        }
        "compact" => results
            .iter()
            .map(|r| format!("{}:{:.4}", r.index, r.distance))
            .collect::<Vec<_>>()
            .join(" "),
        _ => {
            let mut out = String::from("Index\tDistance\tAlpha\t\tKappa\n");
            out.push_str("—\t—\t—\t—\n");
            for r in results {
                out.push_str(&format!(
                    "{}\t{:.4}\t\t{:.4}\t\t{:.4}\n",
                    r.index, r.distance, r.alpha, r.kappa
                ));
            }
            out
        }
    }
}
