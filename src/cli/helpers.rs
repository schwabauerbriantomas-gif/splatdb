//! Shared helpers for CLI command handlers.

use ndarray::{Array1, Array2};
use std::path::PathBuf;

use splatsdb::{
    storage::persistence::{SplatsDBPersistence, StorageBackend},
    SplatStore, SplatsDBConfig,
};

pub fn make_config(dim: usize, max_splats: usize) -> SplatsDBConfig {
    SplatsDBConfig {
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
) -> Result<SplatsDBPersistence, Box<dyn std::error::Error + Send + Sync>> {
    SplatsDBPersistence::with_backend(data_dir, resolve_backend(backend), rw)
}

pub fn load_or_create_store(
    data_dir: &str,
    config: &SplatsDBConfig,
) -> (SplatStore, Option<SplatsDBPersistence>) {
    load_shard_store(data_dir, config, "default")
}

/// Load a store populated from a specific shard (issue #4).
///
/// `search`/`search-file` previously always read the "default" shard, so
/// vectors ingested with `index -s <name>` were silently invisible. This
/// variant loads the named shard; when the shard does not exist we fail
/// loudly instead of returning an empty (silent) result set.
pub fn load_shard_store(
    data_dir: &str,
    config: &SplatsDBConfig,
    shard: &str,
) -> (SplatStore, Option<SplatsDBPersistence>) {
    let persist = make_persistence(data_dir, "sqlite", true).ok();
    let mut store = SplatStore::new(config.clone());

    if let Some(ref p) = persist {
        match p.load_vectors(shard) {
            Ok(Some(vectors)) => {
                eprintln!(
                    "[splatsdb] Loaded {} vectors from shard '{}'",
                    vectors.nrows(),
                    shard
                );
                store.add_splat(&vectors);
                // Try loading persisted HNSW first; only build from scratch if needed
                store.build_index_if_needed(data_dir);
            }
            Ok(None) => {
                if shard != "default" {
                    let known = p.list_shards();
                    eprintln!(
                        "Error: shard '{}' not found. Available shards: {:?}",
                        shard, known
                    );
                    std::process::exit(2);
                }
            }
            Err(e) => {
                eprintln!("Error: failed to load shard '{}': {}", shard, e);
                std::process::exit(2);
            }
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
    let rows = usize::try_from(u64::from_le_bytes(buf8)).map_err(|_| "rows overflow")?;
    file.read_exact(&mut buf8)
        .map_err(|e| format!("Read error: {}", e))?;
    let cols = usize::try_from(u64::from_le_bytes(buf8)).map_err(|_| "cols overflow")?;

    if rows != 1 {
        return Err(format!("Query file must have exactly 1 row, got {}", rows));
    }

    // The header is untrusted input: validate element count and on-disk size
    // BEFORE allocating, so a hostile/truncated `.bin` fails with a clean
    // error instead of an OOM.
    let total = validate_bin_header(path, rows, cols)?;

    let mut data = vec![0.0f32; total];
    let bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut data);
    file.read_exact(bytes)
        .map_err(|e| format!("Read error: {}", e))?;
    Ok(Array1::from(data))
}

/// Validate the (untrusted) `rows`/`cols` header of a `.bin` vector file
/// BEFORE allocating for it.
///
/// Caps the element count at 1e9 (same limit as `persistence::load_vectors`)
/// and requires the file to actually contain the header-promised payload
/// (16-byte header + `rows*cols` little-endian f32), so a hostile or
/// truncated file fails with a clean `Result` error instead of an OOM or a
/// short read. Returns the validated element count.
fn validate_bin_header(path: &std::path::Path, rows: usize, cols: usize) -> Result<usize, String> {
    const MAX_ELEMS: u64 = 1_000_000_000;
    const HEADER_BYTES: u64 = 16; // rows: u64 LE + cols: u64 LE

    let total = (rows as u64)
        .checked_mul(cols as u64)
        .ok_or_else(|| "overflow in rows*cols".to_string())?;
    if total > MAX_ELEMS {
        return Err(format!(
            "Refusing excessive allocation: {}x{} = {} elements (cap {})",
            rows, cols, total, MAX_ELEMS
        ));
    }
    let expected = HEADER_BYTES
        + total
            .checked_mul(std::mem::size_of::<f32>() as u64)
            .ok_or_else(|| "overflow in expected file size".to_string())?;
    let actual = std::fs::metadata(path)
        .map_err(|e| format!("Cannot stat {}: {}", path.display(), e))?
        .len();
    if actual < expected {
        return Err(format!(
            "Truncated file {}: header promises {}x{} f32 ({} bytes including header), file has {} bytes",
            path.display(),
            rows,
            cols,
            expected,
            actual
        ));
    }
    Ok(total as usize)
}

pub fn load_vectors_bin(path: &PathBuf) -> Result<Array2<f32>, String> {
    use std::io::Read;
    let mut file =
        std::fs::File::open(path).map_err(|e| format!("Cannot open {}: {}", path.display(), e))?;
    let mut buf8 = [0u8; 8];
    file.read_exact(&mut buf8)
        .map_err(|e| format!("Read error: {}", e))?;
    let rows = usize::try_from(u64::from_le_bytes(buf8)).map_err(|_| "rows overflow")?;
    file.read_exact(&mut buf8)
        .map_err(|e| format!("Read error: {}", e))?;
    let cols = usize::try_from(u64::from_le_bytes(buf8)).map_err(|_| "cols overflow")?;

    // The header is untrusted input: validate element count and on-disk size
    // BEFORE allocating, so a hostile/truncated `.bin` fails with a clean
    // error instead of an OOM.
    let total = validate_bin_header(path, rows, cols)?;

    let mut data = vec![0.0f32; total];
    let bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut data);
    file.read_exact(bytes)
        .map_err(|e| format!("Read error: {}", e))?;
    Array2::from_shape_vec((rows, cols), data).map_err(|e| format!("Shape error: {}", e))
}

pub fn format_neighbors(results: &[splatsdb::splats::NeighborResult], format: &str) -> String {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "splatdb-cli-helpers-{}-{}-{}",
            tag,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    fn write_header_payload(path: &PathBuf, rows: u64, cols: u64, f32_elems: &[f32]) {
        let mut buf = Vec::new();
        buf.extend_from_slice(&rows.to_le_bytes());
        buf.extend_from_slice(&cols.to_le_bytes());
        buf.extend_from_slice(bytemuck::cast_slice(f32_elems));
        std::fs::write(path, &buf).unwrap();
    }

    #[test]
    fn load_vectors_bin_rejects_truncated_and_oversized_headers() {
        let dir = temp_dir("vectors-bin");
        std::fs::create_dir_all(&dir).unwrap();

        // 1) Alloc bomb: header promises 50_000 x 50_000 = 2.5e9 elements
        //    (over the 1e9 cap) with a 16-byte file -> clean error, no OOM.
        let p = dir.join("bomb.bin");
        write_header_payload(&p, 50_000, 50_000, &[]);
        let err = load_vectors_bin(&p).unwrap_err();
        assert!(
            err.contains("Refusing excessive allocation"),
            "got: {}",
            err
        );

        // 2) Truncated: header promises 2x3 f32 but payload is cut short.
        let p = dir.join("truncated.bin");
        write_header_payload(&p, 2, 3, &[1.0]);
        let err = load_vectors_bin(&p).unwrap_err();
        assert!(err.contains("Truncated file"), "got: {}", err);

        // 3) Happy path: full 2x3 payload loads with correct shape/values.
        let p = dir.join("ok.bin");
        write_header_payload(&p, 2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let v = load_vectors_bin(&p).unwrap();
        assert_eq!(v.dim(), (2, 3));
        assert_eq!(v[[1, 2]], 6.0);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_query_bin_rejects_truncated_and_oversized_headers() {
        let dir = temp_dir("query-bin");
        std::fs::create_dir_all(&dir).unwrap();

        // Alloc bomb: 1 x 2_000_000_000 f32 header, tiny file.
        let p = dir.join("bomb.bin");
        write_header_payload(&p, 1, 2_000_000_000, &[]);
        let err = load_query_bin(&p).unwrap_err();
        assert!(
            err.contains("Refusing excessive allocation"),
            "got: {}",
            err
        );

        // Truncated: 1x4 promised, only 2 f32 present.
        let p = dir.join("truncated.bin");
        write_header_payload(&p, 1, 4, &[1.0, 2.0]);
        let err = load_query_bin(&p).unwrap_err();
        assert!(err.contains("Truncated file"), "got: {}", err);

        // Happy path.
        let p = dir.join("ok.bin");
        write_header_payload(&p, 1, 3, &[0.5, 1.5, 2.5]);
        let q = load_query_bin(&p).unwrap();
        assert_eq!(q.len(), 3);
        assert_eq!(q[2], 2.5);

        std::fs::remove_dir_all(&dir).ok();
    }
}
