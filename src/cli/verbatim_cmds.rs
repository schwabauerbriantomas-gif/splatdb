//! Verbatim Storage and AAAK Compression CLI commands.

use splatsdb::text_compression;
use splatsdb::verbatim::VerbatimResult;
use splatsdb::SplatsDBConfig;

use super::helpers::*;

// ── Verbatim Storage ──────────────────────────────────────────────────

pub fn cmd_verbatim_store(
    data_dir: String,
    _backend: String,
    id: String,
    text: String,
    category: Option<String>,
) {
    let config = SplatsDBConfig::default();
    let (store, persist) = load_or_create_store(&data_dir, &config);

    let dim = store.get_statistics().embedding_dim;
    let embedding = simcos_embed(&text, dim);
    let _arr = ndarray::Array2::from_shape_vec((1, dim), embedding).unwrap();
    let vector_idx = store.n_active() as i64;
    drop(store); // Release store before persistence operations

    // Store metadata with verbatim text
    if let Some(ref p) = persist {
        let mut meta_json = serde_json::json!({
            "verbatim": true,
        });
        if let Some(cat) = &category {
            meta_json["category"] = serde_json::json!(cat);
        }
        p.save_metadata(&id, 0, vector_idx, Some(&meta_json), Some(&text))
            .unwrap_or_else(|e| {
                eprintln!("Error storing document: {}", e);
                std::process::exit(1);
            });
    }

    let result = serde_json::json!({
        "id": id,
        "vector_index": vector_idx,
        "text_length": text.len(),
        "status": "stored_verbatim",
        "category": category,
    });
    println!("{}", serde_json::to_string_pretty(&result).unwrap());
}

pub fn cmd_verbatim_get(_data_dir: String, _backend: String, _id: String) {
    let persist = make_persistence(&_data_dir, &_backend, false).unwrap_or_else(|e| {
        eprintln!("Storage error: {}", e);
        std::process::exit(1);
    });

    match persist.get_metadata(&_id) {
        Ok(Some(meta)) => {
            let result = serde_json::json!({
                "id": _id,
                "metadata": meta,
            });
            println!("{}", serde_json::to_string_pretty(&result).unwrap());
        }
        Ok(None) => {
            eprintln!("Document '{}' not found", _id);
            std::process::exit(1);
        }
        Err(e) => {
            eprintln!("Error retrieving document: {}", e);
            std::process::exit(1);
        }
    }
}

pub fn cmd_verbatim_search(data_dir: String, config: SplatsDBConfig, query: String, k: usize) {
    let (mut store, persist) = load_or_create_store(&data_dir, &config);

    if store.n_active() == 0 {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "results": [],
                "query": query,
                "message": "No documents in store"
            }))
            .unwrap()
        );
        return;
    }

    let dim = store.get_statistics().embedding_dim;
    let embedding = simcos_embed(&query, dim);
    let query_vec = ndarray::Array1::from_vec(embedding);
    let neighbors = store.find_neighbors_fast(&query_vec.view(), k);

    let results: Vec<serde_json::Value> = neighbors
        .into_iter()
        .map(|n| {
            // Try to get document text from persistence
            let (doc_id, text, metadata) = if let Some(ref p) = persist {
                match p.find_doc_by_vector_idx(n.index as i64) {
                    Ok(Some(record)) => (
                        record.id,
                        record.document.unwrap_or_default(),
                        record.metadata,
                    ),
                    _ => (format!("vec_{}", n.index), String::new(), None),
                }
            } else {
                (format!("vec_{}", n.index), String::new(), None)
            };

            let result = VerbatimResult::new(doc_id, n.index, n.distance, text, metadata);
            result.to_json_value()
        })
        .collect();

    // Summary stats
    let high_count = results
        .iter()
        .filter(|r| r["confidence"].as_str() == Some("HIGH"))
        .count();
    let medium_count = results
        .iter()
        .filter(|r| r["confidence"].as_str() == Some("MEDIUM"))
        .count();
    let low_count = results
        .iter()
        .filter(|r| r["confidence"].as_str() == Some("LOW"))
        .count();

    let output = serde_json::json!({
        "query": query,
        "results": results,
        "summary": {
            "total": results.len(),
            "high_confidence": high_count,
            "medium_confidence": medium_count,
            "low_confidence": low_count,
            "reliable_count": high_count + medium_count,
        },
        "disclaimer": "LOW confidence results may be hallucinated — always verify against original source text",
    });

    println!("{}", serde_json::to_string_pretty(&output).unwrap());
}

// ── AAAK Compression ──────────────────────────────────────────────────

pub fn cmd_compress(text: String, verbose: bool) {
    let result = text_compression::compress(&text);

    let hex_data: String = result
        .binary_data
        .iter()
        .map(|b| format!("{:02x}", b))
        .collect();

    if verbose {
        let output = serde_json::json!({
            "original_text": text,
            "original_size": result.original_size,
            "semantic_text": result.semantic_text,
            "semantic_size": result.semantic_size,
            "semantic_ratio": (result.semantic_ratio * 100.0).round() / 100.0,
            "binary_size": result.binary_size,
            "binary_ratio": (result.compression_ratio * 100.0).round() / 100.0,
            "compression_data_hex": hex_data,
        });
        println!("{}", serde_json::to_string_pretty(&output).unwrap());
    } else {
        let output = serde_json::json!({
            "semantic_text": result.semantic_text,
            "original_size": result.original_size,
            "compressed_size": result.binary_size,
            "semantic_ratio": format!("{:.2}×", result.semantic_ratio),
            "total_ratio": format!("{:.2}×", result.compression_ratio),
            "data": hex_data,
        });
        println!("{}", serde_json::to_string_pretty(&output).unwrap());
    }
}

pub fn cmd_decompress(hex_data: String) {
    let binary_data: Vec<u8> = (0..hex_data.len())
        .step_by(2)
        .filter_map(|i| {
            let byte_str = &hex_data[i..i + 2];
            u8::from_str_radix(byte_str, 16).ok()
        })
        .collect();

    match text_compression::decompress(&binary_data) {
        Ok(text) => {
            let output = serde_json::json!({
                "text": text,
                "binary_size": binary_data.len(),
                "text_size": text.len(),
            });
            println!("{}", serde_json::to_string_pretty(&output).unwrap());
        }
        Err(e) => {
            eprintln!("Decompression error: {}", e);
            std::process::exit(1);
        }
    }
}

pub fn cmd_compress_bench(size: String) {
    let sample = match size.as_str() {
        "small" => {
            "The database administrator needs to update the configuration for the application environment. The information technology department manages the performance of the system."
        }
        "medium" => {
            "The database administrator needs to update the configuration for the application environment to improve performance. \
             The information technology department manages the performance of the database system and ensures that all applications \
             are running smoothly. The administrator should check the configuration settings and update them as needed. \
             Performance management is critical for the database to function properly. The application requires proper \
             configuration to operate efficiently in the production environment. Please update the database configuration \
             with respect to the performance requirements of the application."
        }
        "large" => {
            "The database administrator needs to update the configuration for the application environment to improve performance. \
             The information technology department manages the performance of the database system and ensures that all applications \
             are running smoothly. The administrator should check the configuration settings and update them as needed. \
             Performance management is critical for the database to function properly. The application requires proper \
             configuration to operate efficiently in the production environment. Please update the database configuration \
             with respect to the performance requirements of the application. \
             The government organization has requested that the corporation update its technology infrastructure. \
             The department of education at the university is working with the association to improve the experience \
             of students. The technical administrator should verify that the authentication and authorization \
             systems are configured correctly. The development environment should mirror the production configuration \
             as closely as possible. The management team needs to review the performance metrics and adjust the \
             parameters accordingly. The organization should implement proper password management and request \
             authentication for all transactions. The response time for the application should be monitored \
             and the reference values should be updated."
        }
        _ => {
            eprintln!("Unknown size '{}'. Use: small, medium, large", size);
            std::process::exit(1);
        }
    };

    let result = text_compression::compress(sample);

    let output = serde_json::json!({
        "size_category": size,
        "original_bytes": result.original_size,
        "semantic_bytes": result.semantic_size,
        "binary_bytes": result.binary_size,
        "semantic_ratio": format!("{:.2}×", result.semantic_ratio),
        "total_ratio": format!("{:.2}×", result.compression_ratio),
        "original_preview": &sample[..sample.len().min(100)],
        "semantic_preview": &result.semantic_text[..result.semantic_text.len().min(100)],
    });
    println!("{}", serde_json::to_string_pretty(&output).unwrap());
}

/// SimCos embedding — simple similarity-consistent embedding for CLI use.
fn simcos_embed(text: &str, dim: usize) -> Vec<f32> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let words: Vec<&str> = text.split_whitespace().collect();
    let mut trigrams = Vec::new();
    for w in &words {
        trigrams.push(format!("w:{}", w.to_lowercase()));
    }
    for i in 0..words.len().saturating_sub(1) {
        trigrams.push(format!(
            "wb:{}:{}",
            words[i].to_lowercase(),
            words[i + 1].to_lowercase()
        ));
    }

    let mut result = vec![0.0f32; dim];
    for tg in &trigrams {
        for band in 0..3 {
            let mut hasher = DefaultHasher::new();
            tg.hash(&mut hasher);
            band.hash(&mut hasher);
            let idx = (hasher.finish() as usize) % dim;
            let mut hasher2 = DefaultHasher::new();
            tg.hash(&mut hasher2);
            band.hash(&mut hasher2);
            (idx as u64).hash(&mut hasher2);
            let sign: f32 = if hasher2.finish().is_multiple_of(2) {
                1.0
            } else {
                -1.0
            };
            result[idx] += sign;
        }
    }

    let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in result.iter_mut() {
            *x /= norm;
        }
    }
    result
}
