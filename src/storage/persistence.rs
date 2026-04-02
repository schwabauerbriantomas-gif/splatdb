//! Persistence layer — vectors (binary), metadata (pluggable backend), energy state, WAL.
//!
//! Supports any structured database via the `MetadataStore` trait.
//! Backend selection via `StorageBackend` enum or feature flags.

use ndarray::Array2;
use std::path::{Path, PathBuf};

use super::metadata_store::{DocumentRecord, MetadataStore};
use super::wal::WriteAheadLog;

/// Available storage backends.
#[derive(Debug, Clone, PartialEq)]
#[derive(Default)]
pub enum StorageBackend {
    /// SQLite — default, production-ready.
    #[default]
    Sqlite,
    /// JSON file — simple, no native deps. Good for dev/testing.
    JsonFile,
    /// Custom backend registered at runtime.
    Custom(String),
}


/// M2M Persistence manager.
///
/// Binary vector storage is always file-based (shards).
/// Metadata storage is pluggable via `MetadataStore` trait.
pub struct M2MPersistence {
    pub storage_path: PathBuf,
    pub wal: Option<WriteAheadLog>,
    meta: Box<dyn MetadataStore>,
}

impl M2MPersistence {
    /// Create with a specific backend.
    pub fn with_backend(storage_path: &str, backend: StorageBackend, enable_wal: bool) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let resolved = validate_path(storage_path)?;

        // Create directory structure
        for dir in &["data/vectors", "data/metadata", "index", "wal/checkpoint", "energy"] {
            std::fs::create_dir_all(resolved.join(dir))?;
        }

        let meta: Box<dyn MetadataStore> = match backend {
            StorageBackend::Sqlite => {
                let db_path = resolved.join("data").join("metadata").join("metadata.db");
                Box::new(super::sqlite_store::SqliteMetadataStore::new(db_path)?)
            }
            StorageBackend::JsonFile => {
                let json_path = resolved.join("data").join("metadata").join("metadata.json");
                Box::new(super::json_store::JsonMetadataStore::new(json_path)?)
            }
            StorageBackend::Custom(name) => {
                return Err(format!("Custom backend '{}' not registered. Use `with_custom_store()` instead.", name).into());
            }
        };

        let wal = if enable_wal {
            let wal_path = resolved.join("wal").join("wal.log");
            Some(WriteAheadLog::new(wal_path.to_str().unwrap_or(""), 50)?)
        } else {
            None
        };

        Ok(Self {
            storage_path: resolved,
            wal,
            meta,
        })
    }

    /// Create with default backend (SQLite).
    pub fn new(storage_path: &str, enable_wal: bool) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        Self::with_backend(storage_path, StorageBackend::Sqlite, enable_wal)
    }

    /// Create with a custom MetadataStore implementation.
    pub fn with_custom_store(storage_path: &str, meta_store: Box<dyn MetadataStore>, enable_wal: bool) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let resolved = validate_path(storage_path)?;
        for dir in &["data/vectors", "data/metadata", "index", "wal/checkpoint", "energy"] {
            std::fs::create_dir_all(resolved.join(dir))?;
        }
        let wal = if enable_wal {
            let wal_path = resolved.join("wal").join("wal.log");
            Some(WriteAheadLog::new(wal_path.to_str().unwrap_or(""), 50)?)
        } else {
            None
        };
        Ok(Self { storage_path: resolved, wal, meta: meta_store })
    }

    /// Get the active backend name.
    pub fn backend_name(&self) -> &'static str { self.meta.backend_name() }

    // ---- Vector Storage (file-based, unchanged) ----

    pub fn save_vectors(&self, vectors: &Array2<f32>, shard_name: &str) -> std::io::Result<()> {
        let path = self.storage_path.join("data").join("vectors").join(format!("{}.bin", shard_name));
        let mut file = std::fs::File::create(path)?;
        use std::io::Write;
        file.write_all(&(vectors.nrows() as u64).to_le_bytes())?;
        file.write_all(&(vectors.ncols() as u64).to_le_bytes())?;
        for row in vectors.rows() {
            let slice = row.as_slice().unwrap_or(&[]);
            // SAFETY: `slice` is a valid &[f32] from an ndarray row. Casting to &[u8] of length
            // `slice.len() * 4` is valid because f32 is 4 bytes with no padding and the slice
            // does not outlive the row reference.
            let bytes: &[u8] = unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, slice.len() * 4) };
            file.write_all(bytes)?;
        }
        Ok(())
    }

    pub fn load_vectors(&self, shard_name: &str) -> std::io::Result<Option<Array2<f32>>> {
        let path = self.storage_path.join("data").join("vectors").join(format!("{}.bin", shard_name));
        if !path.exists() { return Ok(None); }
        let mut file = std::fs::File::open(path)?;
        use std::io::Read;
        let mut buf8 = [0u8; 8];
        file.read_exact(&mut buf8)?;
        let rows = u64::from_le_bytes(buf8) as usize;
        file.read_exact(&mut buf8)?;
        let cols = u64::from_le_bytes(buf8) as usize;
        let mut data = vec![0.0f32; rows * cols];
        // SAFETY: `data` is a valid Vec<f32> with `rows * cols` elements. Casting to &mut [u8] of
        // length `data.len() * 4` is valid because f32 is 4 bytes with no padding. The slice does
        // not outlive `data` and we read exactly that many bytes.
        let bytes: &mut [u8] = unsafe {
            std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, data.len() * 4)
        };
        file.read_exact(bytes)?;
        Ok(Some(ndarray::Array2::from_shape_vec((rows, cols), data)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?))
    }

    pub fn list_shards(&self) -> Vec<String> {
        let dir = self.storage_path.join("data").join("vectors");
        std::fs::read_dir(&dir)
            .map(|entries| entries
                .filter_map(|e| e.ok())
                .filter(|e| e.path().extension().is_some_and(|ext| ext == "bin"))
                .filter_map(|e| e.path().file_stem().map(|s| s.to_string_lossy().into_owned()))
                .collect()
            )
            .unwrap_or_default()
    }

    // ---- Metadata (delegates to MetadataStore) ----

    pub fn save_metadata(&self, doc_id: &str, shard_idx: i64, vector_idx: i64, metadata: Option<&serde_json::Value>, document: Option<&str>) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_secs_f64();
        let record = DocumentRecord {
            id: doc_id.into(),
            shard_idx,
            vector_idx,
            metadata: metadata.cloned(),
            document: document.map(|s| s.into()),
            deleted: false,
            created_at: now,
            updated_at: now,
        };
        self.meta.upsert(&record)
    }

    pub fn get_metadata(&self, doc_id: &str) -> Result<Option<std::collections::HashMap<String, serde_json::Value>>, Box<dyn std::error::Error + Send + Sync>> {
        match self.meta.get(doc_id)? {
            Some(record) => Ok(Some(record.to_hashmap())),
            None => Ok(None),
        }
    }

    pub fn soft_delete(&self, doc_id: &str) -> Result<bool, Box<dyn std::error::Error + Send + Sync>> {
        self.meta.soft_delete(doc_id)
    }

    pub fn hard_delete(&self, doc_id: &str) -> Result<bool, Box<dyn std::error::Error + Send + Sync>> {
        self.meta.hard_delete(doc_id)
    }

    pub fn get_all_ids(&self, include_deleted: bool) -> Result<Vec<String>, Box<dyn std::error::Error + Send + Sync>> {
        self.meta.list_ids(include_deleted)
    }

    pub fn count_documents(&self, include_deleted: bool) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
        self.meta.count(include_deleted)
    }

    // ---- Energy State (file-based, unchanged) ----

    pub fn save_energy_state(&self, mu: &Array2<f32>, alpha: &[f32], kappa: &[f32]) -> std::io::Result<()> {
        let path = self.storage_path.join("energy").join("landscape.bin");
        let mut file = std::fs::File::create(path)?;
        use std::io::Write;
        file.write_all(&(mu.nrows() as u64).to_le_bytes())?;
        file.write_all(&(mu.ncols() as u64).to_le_bytes())?;
        for row in mu.rows() {
            let s = row.as_slice().unwrap_or(&[]);
            // SAFETY: `s` is a valid &[f32] from an ndarray row. Casting to &[u8] of length
            // `s.len() * 4` is valid because f32 is 4 bytes with no padding.
            let bytes: &[u8] = unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, s.len() * 4) };
            file.write_all(bytes)?;
        }
        file.write_all(&(alpha.len() as u64).to_le_bytes())?;
        // SAFETY: `alpha` is a valid &[f32]. Casting to &[u8] of length `alpha.len() * 4` is valid
        // because f32 is 4 bytes with no padding.
        let bytes: &[u8] = unsafe { std::slice::from_raw_parts(alpha.as_ptr() as *const u8, alpha.len() * 4) };
        file.write_all(bytes)?;
        file.write_all(&(kappa.len() as u64).to_le_bytes())?;
        // SAFETY: `kappa` is a valid &[f32]. Casting to &[u8] of length `kappa.len() * 4` is valid
        // because f32 is 4 bytes with no padding.
        let bytes: &[u8] = unsafe { std::slice::from_raw_parts(kappa.as_ptr() as *const u8, kappa.len() * 4) };
        file.write_all(bytes)?;
        Ok(())
    }

    // ---- Backup ----

    pub fn backup(&self, backup_path: &str) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        if backup_path.contains("..") {
            return Err("Path traversal detected".into());
        }
        self.meta.flush()?;
        let dest = PathBuf::from(backup_path);
        std::fs::create_dir_all(&dest)?;
        let ts = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_secs();
        let backup_dest = dest.join(format!("m2m_backup_{}", ts));
        copy_dir_recursive(&self.storage_path, &backup_dest)?;
        Ok(backup_dest.to_string_lossy().into_owned())
    }

    // ---- WAL ----

    pub fn checkpoint(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if let Some(ref wal) = self.wal { wal.checkpoint()?; }
        Ok(())
    }

    pub fn close(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.meta.flush()?;
        if let Some(ref wal) = self.wal { wal.close()?; }
        Ok(())
    }
}

fn validate_path(storage_path: &str) -> Result<PathBuf, Box<dyn std::error::Error + Send + Sync>> {
    if storage_path.contains("..") {
        return Err("Path traversal detected in storage_path".into());
    }
    Ok(PathBuf::from(storage_path).canonicalize().unwrap_or_else(|_| PathBuf::from(storage_path)))
}

fn copy_dir_recursive(src: &Path, dst: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let src_path = entry.path();
        let dst_path = dst.join(entry.file_name());
        if src_path.is_dir() {
            copy_dir_recursive(&src_path, &dst_path)?;
        } else {
            std::fs::copy(&src_path, &dst_path)?;
        }
    }
    Ok(())
}
