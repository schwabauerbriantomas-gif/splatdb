//! Optimized data loader. Ported from splatdb Python.

use ndarray::Array2;
use std::io::{Read, Write};

/// Loads vector data from binary format (rows: u64, cols: u64, f32 data).
pub fn load_vectors_bin(path: &str) -> Result<Array2<f32>, String> {
    let mut file = std::fs::File::open(path).map_err(|e| format!("Cannot open {}: {}", path, e))?;
    let mut buf8 = [0u8; 8];

    file.read_exact(&mut buf8).map_err(|e| format!("Read error: {}", e))?;
    let rows = u64::from_le_bytes(buf8) as usize;

    file.read_exact(&mut buf8).map_err(|e| format!("Read error: {}", e))?;
    let cols = u64::from_le_bytes(buf8) as usize;

    let mut data = vec![0.0f32; rows * cols];
    // SAFETY: `data` is a valid Vec<f32> with `rows * cols` elements. Casting to &mut [u8] of
    // length `data.len() * 4` is valid because f32 is 4 bytes with no padding. The slice does not
    // outlive `data` and we read exactly that many bytes.
    let bytes: &mut [u8] = unsafe {
        std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, data.len() * 4)
    };
    file.read_exact(bytes).map_err(|e| format!("Read error: {}", e))?;
    Array2::from_shape_vec((rows, cols), data).map_err(|e| format!("Shape error: {}", e))
}

/// Saves vectors to binary format.
pub fn save_vectors_bin(path: &str, vectors: &Array2<f32>) -> Result<(), String> {
    let (rows, cols) = vectors.dim();
    let mut file = std::fs::File::create(path).map_err(|e| format!("Cannot create {}: {}", path, e))?;
    file.write_all(&rows.to_le_bytes()).map_err(|e| format!("Write error: {}", e))?;
    file.write_all(&cols.to_le_bytes()).map_err(|e| format!("Write error: {}", e))?;
    // SAFETY: `vectors` is a contiguous Array2<f32>. Casting its data pointer to & [u8] of length
    // `rows * cols * 4` is valid because f32 is 4 bytes with no padding and the slice does not
    // outlive the Array2 reference.
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(vectors.as_ptr() as *const u8, rows * cols * 4)
    };
    file.write_all(bytes).map_err(|e| format!("Write error: {}", e))?;
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
