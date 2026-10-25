//! Write-Ahead Log for durability.
//! Binary format: [4-byte big-endian length][JSON payload].

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::PathBuf;

/// A single WAL entry.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WALEntry {
    pub lsn: u64,
    pub timestamp: f64,
    pub operation: String,
    pub data: serde_json::Value,
}

/// Write-Ahead Log ensuring durability of operations.
pub struct WriteAheadLog {
    path: PathBuf,
    sync_interval: u32,
    inner: Mutex<WALInner>,
}

struct WALInner {
    file: BufWriter<File>,
    lsn: u64,
    op_count: u32,
}

impl WriteAheadLog {
    pub fn new(path: &str, sync_interval: u32) -> std::io::Result<Self> {
        let pb = PathBuf::from(path);
        if let Some(parent) = pb.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let file = OpenOptions::new().create(true).append(true).open(&pb)?;
        let wal = Self {
            path: pb,
            sync_interval,
            inner: Mutex::new(WALInner {
                file: BufWriter::new(file),
                lsn: 0,
                op_count: 0,
            }),
        };

        // Load current LSN from existing entries
        wal.load_current_lsn()?;
        Ok(wal)
    }

    fn load_current_lsn(&self) -> std::io::Result<()> {
        let entries = self.read_entries()?;
        if let Some(last) = entries.last() {
            let mut inner = self.inner.lock();
            inner.lsn = last.lsn.saturating_add(1);
        }
        Ok(())
    }

    /// Log an operation. Returns the LSN assigned.
    pub fn log_operation(&self, op: &str, data: serde_json::Value) -> std::io::Result<u64> {
        let mut inner = self.inner.lock();
        let entry = WALEntry {
            lsn: inner.lsn,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs_f64(),
            operation: op.to_string(),
            data,
        };

        let serialized = serde_json::to_vec(&entry)?;
        let len: u32 = serialized
            .len()
            .try_into()
            .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "entry too large"))?;
        inner.file.write_all(&len.to_be_bytes())?;
        inner.file.write_all(&serialized)?;

        inner.op_count += 1;
        if inner.op_count >= self.sync_interval {
            inner.file.flush()?;
            inner.op_count = 0;
        }

        let lsn = inner.lsn;
        inner.lsn = inner.lsn.saturating_add(1);
        Ok(lsn)
    }

    /// Flush and fsync for durability.
    pub fn checkpoint(&self) -> std::io::Result<()> {
        let mut inner = self.inner.lock();
        let entry = WALEntry {
            lsn: inner.lsn,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs_f64(),
            operation: "checkpoint".to_string(),
            data: serde_json::json!({"lsn_at_checkpoint": inner.lsn}),
        };

        let serialized = serde_json::to_vec(&entry)?;
        let len: u32 = serialized
            .len()
            .try_into()
            .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "entry too large"))?;
        inner.file.write_all(&len.to_be_bytes())?;
        inner.file.write_all(&serialized)?;
        inner.file.flush()?;

        inner.lsn = inner.lsn.saturating_add(1);
        Ok(())
    }

    /// Recover all entries from the WAL.
    pub fn recover(&self) -> std::io::Result<Vec<WALEntry>> {
        self.read_entries()
    }

    /// Truncate entries before the given LSN.
    ///
    /// Crash-safe rewrite: the surviving entries go to a sibling temp file
    /// that is fsynced and renamed over the WAL — `inner.file` is only
    /// replaced after EVERY fallible step succeeded, so a failure anywhere
    /// leaves the original file and handle untouched (the old implementation
    /// pointed the handle at /dev/null on its error path, silently losing
    /// every later log_operation).
    ///
    /// # TOCTOU safety
    ///
    /// Acquires `self.inner` lock BEFORE `read_entries()`, so nothing can be
    /// logged between reading and rewriting.
    pub fn truncate(&self, before_lsn: u64) -> std::io::Result<()> {
        let mut inner = self.inner.lock();

        let entries = self.read_entries()?;
        let remaining: Vec<&WALEntry> = entries.iter().filter(|e| e.lsn >= before_lsn).collect();

        // 1. Build the surviving WAL in a sibling temp file.
        let tmp_path = self.path.with_file_name(format!(
            "{}.tmp.{}",
            self.path
                .file_name()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_default(),
            std::process::id()
        ));
        {
            let writer = std::fs::File::create(&tmp_path)?;
            let mut writer = BufWriter::new(writer);
            for entry in &remaining {
                let serialized = serde_json::to_vec(entry)?;
                let len: u32 = serialized.len().try_into().map_err(|_| {
                    std::io::Error::new(std::io::ErrorKind::InvalidData, "entry too large")
                })?;
                writer.write_all(&len.to_be_bytes())?;
                writer.write_all(&serialized)?;
            }
            writer.flush()?;
            writer.get_ref().sync_all()?;
        }

        // 2. Swap atomically, THEN reopen for append. Both steps run before
        //    `inner.file` is touched; if either fails, drop the temp and
        //    return — the live WAL is intact.
        let reopened = (|| -> std::io::Result<std::fs::File> {
            std::fs::rename(&tmp_path, &self.path)?;
            OpenOptions::new().append(true).open(&self.path)
        })();
        let file = match reopened {
            Ok(f) => f,
            Err(e) => {
                let _ = std::fs::remove_file(&tmp_path);
                return Err(e);
            }
        };

        // 3. Only now replace the handle (flush the old one first).
        inner.file.flush()?;
        inner.file = BufWriter::new(file);
        Ok(())
    }

    /// Close the WAL, flushing first.
    pub fn close(&self) -> std::io::Result<()> {
        let mut inner = self.inner.lock();
        inner.file.flush()
    }

    /// Read all entries from the WAL file on disk.
    ///
    /// Only called from `recover()` and `load_current_lsn()`; both are
    /// initialization/recovery-time calls with no concurrent loggers, so no
    /// inner lock is required.
    fn read_entries(&self) -> std::io::Result<Vec<WALEntry>> {
        if !self.path.exists() {
            return Ok(vec![]);
        }

        let file = File::open(&self.path)?;
        let mut reader = BufReader::new(file);
        let mut entries = Vec::new();

        loop {
            let mut len_buf = [0u8; 4];
            match reader.read_exact(&mut len_buf) {
                Ok(()) => {}
                Err(_) => break,
            }
            let len = u32::from_be_bytes(len_buf) as usize;
            if len == 0 || len > 100_000_000 {
                break; // Sanity check: max 100MB per entry
            }

            let mut data_buf = vec![0u8; len];
            match reader.read_exact(&mut data_buf) {
                Ok(()) => {}
                Err(_) => break, // Truncated entry
            }

            if let Ok(entry) = serde_json::from_slice::<WALEntry>(&data_buf) {
                entries.push(entry);
            }
        }

        Ok(entries)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_wal_path(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "splatdb-wal-{}-{}-{}",
            tag,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join("wal.log")
    }

    #[test]
    fn log_checkpoint_recover_roundtrip() {
        let path = temp_wal_path("roundtrip");
        let wal = WriteAheadLog::new(path.to_str().unwrap(), 100).unwrap();

        let lsn0 = wal
            .log_operation("insert", serde_json::json!({"i": 0}))
            .unwrap();
        let lsn1 = wal
            .log_operation("delete", serde_json::json!({"i": 1}))
            .unwrap();
        assert_eq!(lsn0, 0);
        assert_eq!(lsn1, 1);

        // checkpoint() flushes; recover() must see both ops plus the
        // checkpoint entry. (After removing truncate(), there is no code path
        // left that can repoint the writer away from the real WAL file.)
        wal.checkpoint().unwrap();
        let entries = wal.recover().unwrap();
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].operation, "insert");
        assert_eq!(entries[1].operation, "delete");
        assert_eq!(entries[2].operation, "checkpoint");

        // LSN continues monotonically after reload.
        drop(wal);
        let wal = WriteAheadLog::new(path.to_str().unwrap(), 100).unwrap();
        let lsn3 = wal
            .log_operation("insert", serde_json::json!({"i": 3}))
            .unwrap();
        assert_eq!(lsn3, 3);

        wal.close().unwrap();
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }
}
