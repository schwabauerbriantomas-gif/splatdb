//! Optimized data loader. Ported from splatsdb Python.

use ndarray::Array2;
use std::io::{Read, Write};

const MAX_LOAD_ELEMENTS: usize = 1_000_000_000;

/// Loads vector data from binary format (rows: u64, cols: u64, f32 data).
pub fn load_vectors_bin(path: &str) -> Result<Array2<f32>, String> {
    let mut file = std::fs::File::open(path).map_err(|e| format!("Cannot open {}: {}", path, e))?;
    let mut buf8 = [0u8; 8];

    file.read_exact(&mut buf8)
        .map_err(|e| format!("Read error: {}", e))?;
    let rows = usize::try_from(u64::from_le_bytes(buf8)).map_err(|_| "rows overflow")?;

    file.read_exact(&mut buf8)
        .map_err(|e| format!("Read error: {}", e))?;
    let cols = usize::try_from(u64::from_le_bytes(buf8)).map_err(|_| "cols overflow")?;

    let total = rows.checked_mul(cols).ok_or("overflow in rows*cols")?;
    if total > MAX_LOAD_ELEMENTS {
        return Err(format!(
            "Allocation too large: {} elements (max {})",
            total, MAX_LOAD_ELEMENTS
        ));
    }

    let mut data = vec![0.0f32; total];
    let bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut data);
    file.read_exact(bytes)
        .map_err(|e| format!("Read error: {}", e))?;
    Array2::from_shape_vec((rows, cols), data).map_err(|e| format!("Shape error: {}", e))
}

/// Saves vectors to binary format.
pub fn save_vectors_bin(path: &str, vectors: &Array2<f32>) -> Result<(), String> {
    let (rows, cols) = vectors.dim();
    let mut file =
        std::fs::File::create(path).map_err(|e| format!("Cannot create {}: {}", path, e))?;
    file.write_all(&rows.to_le_bytes())
        .map_err(|e| format!("Write error: {}", e))?;
    file.write_all(&cols.to_le_bytes())
        .map_err(|e| format!("Write error: {}", e))?;
    let bytes: &[u8] = bytemuck::cast_slice(vectors.as_slice().unwrap());
    file.write_all(bytes)
        .map_err(|e| format!("Write error: {}", e))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let vectors = Array2::from_shape_vec((2, 3), data).unwrap();
        save_vectors_bin("test_loader.bin", &vectors).unwrap();
        let loaded = load_vectors_bin("test_loader.bin").unwrap();
        assert_eq!(loaded.dim(), (2, 3));
        assert_eq!(&loaded, &vectors);
        std::fs::remove_file("test_loader.bin").ok();
    }
}
