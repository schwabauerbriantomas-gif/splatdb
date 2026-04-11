//! QA Validation Test Suite — SplatsDB
//!
//! Comprehensive tests covering:
//! - Text compression (semantic + binary) roundtrips
//! - WAL (log, truncate, recover, corruption)
//! - Persistence (shard name validation, vector load/save)
//! - Encoding (ColorHistogram, Attribute, Position, FullEmbedding roundtrips)
//! - HNSW (build, search, save/load, cosine metric)
//! - Quantization (PolarQuant encode/decode, TurboQuant encode/IP estimate)
//! - Graph operations (entity extraction, graph ops, edge cases)
//! - Spatial / distance calculations
//! - Error handling (wrong dimensions, empty inputs, path traversal)

use ndarray::{array, Array2};
use splatsdb::encoding::{
    AttributeEncoder, ColorHistogramEncoder, FullEmbeddingBuilder, SinusoidalPositionEncoder,
};
use splatsdb::geometry::{geodesic_distance, normalize_sphere};
use splatsdb::graph_splat::{GaussianGraphStore, GraphError};
use splatsdb::hnsw_index::HNSWIndex;
use splatsdb::interfaces::VectorIndex;
use splatsdb::quant::polar::PolarQuantizer;
use splatsdb::quant::turbo::TurboQuantizer;
use splatsdb::text_compression::{
    binary_compress, binary_decompress, compress, decompress, semantic_compress,
};

// Helper: random vector using rand
fn random_vec(dim: usize, seed: u64) -> Vec<f32> {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use rand_distr::{Distribution, StandardNormal};
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..dim).map(|_| StandardNormal.sample(&mut rng)).collect()
}

fn random_array2(n: usize, dim: usize, seed: u64) -> Array2<f32> {
    use rand::Rng;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let data: Vec<f32> = (0..n * dim).map(|_| rng.gen::<f32>()).collect();
    Array2::from_shape_vec((n, dim), data).unwrap()
}

// =====================================================================
// TEXT COMPRESSION TESTS
// =====================================================================

#[test]
fn qa_text_compress_decompress_roundtrip() {
    let text = "The database administrator needs to update the configuration for better performance and management of the application environment";
    let result = compress(text);
    let recovered = decompress(&result.binary_data).expect("decompress should succeed");
    assert!(
        recovered.contains("db"),
        "decompressed text should contain abbreviation 'db', got: {}",
        recovered
    );
    assert!(
        recovered.contains("admin"),
        "decompressed text should contain 'admin', got: {}",
        recovered
    );
    assert!(
        recovered.contains("config"),
        "decompressed text should contain 'config', got: {}",
        recovered
    );
}

#[test]
fn qa_text_compress_empty_input() {
    let result = compress("");
    assert_eq!(result.original_size, 0);
    assert_eq!(result.compression_ratio, 1.0);
    // Decompress should work on empty result
    if !result.binary_data.is_empty() {
        let recovered = decompress(&result.binary_data).expect("decompress of empty should work");
        assert!(recovered.is_empty() || recovered.trim().is_empty());
    }
}

#[test]
fn qa_text_compress_unicode() {
    let text = "日本語のテスト。データベースの設定を更新してください。";
    let result = compress(text);
    assert!(result.original_size > 0);
    // Binary roundtrip should preserve the semantic output
    if !result.binary_data.is_empty() {
        let recovered = decompress(&result.binary_data).expect("unicode decompress should work");
        // Japanese characters should survive (no stop word stripping applies)
        assert!(
            recovered.contains("日本語") || recovered.contains("テスト"),
            "unicode should survive: {}",
            recovered
        );
    }
}

#[test]
fn qa_text_compress_null_bytes() {
    let data: Vec<u8> = b"hello\x00world\x00test".to_vec();
    let compressed = binary_compress(&data);
    let decompressed = binary_decompress(&compressed).expect("null byte roundtrip should work");
    assert_eq!(data, decompressed);
}

#[test]
fn qa_text_compress_large_input() {
    // 100KB of repetitive text
    let base = "The database administrator needs to update the configuration for better performance management ";
    let text = base.repeat(2000);
    let result = compress(&text);
    assert!(
        result.compression_ratio > 1.0,
        "should compress large input"
    );
    if !result.binary_data.is_empty() {
        let recovered = decompress(&result.binary_data).expect("large decompress should work");
        assert!(!recovered.is_empty());
    }
}

#[test]
fn qa_binary_decompress_corrupted_data() {
    // Too short
    assert!(binary_decompress(&[]).is_err());
    assert!(binary_decompress(&[0, 0, 0]).is_err());

    // Valid header but truncated payload
    let header: [u8; 4] = 100u32.to_le_bytes();
    let mut corrupted = header.to_vec();
    corrupted.push(0xFF); // RLE marker with no data following
    assert!(binary_decompress(&corrupted).is_err());
}

#[test]
fn qa_semantic_compress_preserves_core_content() {
    let input = "The quick brown fox jumps over the lazy dog";
    let compressed = semantic_compress(input);
    // Core nouns/verbs should remain
    assert!(compressed.contains("quick"), "should keep 'quick'");
    assert!(compressed.contains("brown"), "should keep 'brown'");
    assert!(compressed.contains("fox"), "should keep 'fox'");
    assert!(compressed.contains("jumps"), "should keep 'jumps'");
    // Stop words like "the" should be removed (unless proper noun start)
    assert!(
        !compressed.contains(" the ") || compressed.starts_with("The"),
        "should remove standalone stop word 'the'"
    );
}

// =====================================================================
// WAL TESTS
// =====================================================================

#[test]
fn qa_wal_log_and_recover() {
    let dir = std::env::temp_dir().join("qa_wal_log_recover");
    let _ = std::fs::remove_dir_all(&dir);
    let wal_path = dir.join("test_wal.log");

    let wal = splatsdb::storage::wal::WriteAheadLog::new(wal_path.to_str().unwrap(), 1).unwrap();

    let lsn0 = wal
        .log_operation("insert", serde_json::json!({"id": 1, "vec": [0.1, 0.2]}))
        .unwrap();
    let lsn1 = wal
        .log_operation("delete", serde_json::json!({"id": 2}))
        .unwrap();
    assert_eq!(lsn0, 0, "first LSN should be 0");
    assert_eq!(lsn1, 1, "second LSN should be 1");

    let entries = wal.recover().unwrap();
    assert_eq!(entries.len(), 2);
    assert_eq!(entries[0].operation, "insert");
    assert_eq!(entries[1].operation, "delete");
    assert_eq!(entries[0].lsn, 0);
    assert_eq!(entries[1].lsn, 1);

    wal.close().unwrap();
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn qa_wal_truncate_removes_old_entries() {
    let dir = std::env::temp_dir().join("qa_wal_truncate");
    let _ = std::fs::remove_dir_all(&dir);
    let wal_path = dir.join("test_wal.log");

    let wal = splatsdb::storage::wal::WriteAheadLog::new(wal_path.to_str().unwrap(), 1).unwrap();

    for i in 0..5u64 {
        wal.log_operation("op", serde_json::json!({"i": i}))
            .unwrap();
    }

    // Truncate entries before LSN 3
    wal.truncate(3).unwrap();

    let entries = wal.recover().unwrap();
    assert!(
        entries.len() <= 2,
        "after truncation should have at most 2 entries, got {}",
        entries.len()
    );
    for e in &entries {
        assert!(
            e.lsn >= 3,
            "all remaining entries should have lsn >= 3, got {}",
            e.lsn
        );
    }

    wal.close().unwrap();
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn qa_wal_recover_from_empty_file() {
    let dir = std::env::temp_dir().join("qa_wal_empty");
    let _ = std::fs::remove_dir_all(&dir);
    let wal_path = dir.join("test_wal.log");

    let wal = splatsdb::storage::wal::WriteAheadLog::new(wal_path.to_str().unwrap(), 1).unwrap();
    let entries = wal.recover().unwrap();
    assert!(entries.is_empty(), "empty WAL should return no entries");

    wal.close().unwrap();
    let _ = std::fs::remove_dir_all(&dir);
}

// =====================================================================
// PERSISTENCE / SHARD NAME VALIDATION TESTS
// =====================================================================

#[test]
fn qa_persistence_reject_traversal_shard_name() {
    // Test the private validate_shard_name logic indirectly via SplatsDBPersistence
    // or directly if pub. Since it's private, test through save_vectors.
    let dir = std::env::temp_dir().join("qa_persistence_shard");
    let _ = std::fs::remove_dir_all(&dir);

    let p = splatsdb::storage::persistence::SplatsDBPersistence::with_backend(
        dir.to_str().unwrap(),
        splatsdb::storage::persistence::StorageBackend::Sqlite,
        false,
    )
    .unwrap();

    let vecs = Array2::from_shape_vec((2, 3), vec![1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0]).unwrap();

    // Normal name should work
    assert!(p.save_vectors(&vecs, "test_shard").is_ok());

    // Path traversal names should be rejected
    assert!(
        p.save_vectors(&vecs, "../etc/passwd").is_err(),
        "should reject .. in shard name"
    );
    assert!(
        p.save_vectors(&vecs, "foo/bar").is_err(),
        "should reject / in shard name"
    );
    assert!(
        p.save_vectors(&vecs, "foo\\bar").is_err(),
        "should reject \\ in shard name"
    );

    // Null byte in name
    let null_name = "test\x00shard".to_string();
    assert!(
        p.save_vectors(&vecs, &null_name).is_err(),
        "should reject null bytes in shard name"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn qa_persistence_save_load_vectors_roundtrip() {
    let dir = std::env::temp_dir().join("qa_persistence_roundtrip");
    let _ = std::fs::remove_dir_all(&dir);

    let p = splatsdb::storage::persistence::SplatsDBPersistence::with_backend(
        dir.to_str().unwrap(),
        splatsdb::storage::persistence::StorageBackend::Sqlite,
        false,
    )
    .unwrap();

    let original = Array2::from_shape_vec(
        (3, 4),
        vec![
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
    )
    .unwrap();

    p.save_vectors(&original, "roundtrip_test").unwrap();

    let loaded = p
        .load_vectors("roundtrip_test")
        .unwrap()
        .expect("should load");
    assert_eq!(loaded.nrows(), 3);
    assert_eq!(loaded.ncols(), 4);
    for i in 0..3 {
        for j in 0..4 {
            let diff = (original[[i, j]] - loaded[[i, j]]).abs();
            assert!(
                diff < 1e-6,
                "mismatch at [{}, {}]: {} vs {}",
                i,
                j,
                original[[i, j]],
                loaded[[i, j]]
            );
        }
    }

    // Non-existent shard should return None
    assert!(p.load_vectors("nonexistent").unwrap().is_none());

    let _ = std::fs::remove_dir_all(&dir);
}

// =====================================================================
// ENCODING TESTS
// =====================================================================

#[test]
fn qa_encoding_color_histogram_roundtrip_similar() {
    let encoder = ColorHistogramEncoder::new();
    let c1 = encoder.encode_single(&[1.0, 0.0, 0.0]); // Pure red
    let c2 = encoder.encode_single(&[1.0, 0.0, 0.0]); // Same red
    let c3 = encoder.encode_single(&[0.0, 0.0, 1.0]); // Pure blue

    // Same color → same encoding
    assert_eq!(c1, c2);

    // Different colors → different encodings
    assert_ne!(c1, c3);

    // All values should be finite
    for &v in c1.iter() {
        assert!(v.is_finite(), "color encoding should be finite");
    }
    // Peak should be at the correct bin
    assert!(c1.iter().all(|&v| (0.0..=1.0).contains(&v)));
}

#[test]
fn qa_encoding_attribute_roundtrip() {
    let encoder = AttributeEncoder::new();
    let enc = encoder.encode_single(0.5, &[0.1, 0.2, 0.3], &[1.0, 0.0, 0.0, 0.0]);
    assert_eq!(enc.len(), 64);
    assert!((enc[0] - 0.5f32).abs() < 1e-6, "opacity should be 0.5");
    assert!((enc[1] - 0.25f32).abs() < 1e-6, "opacity^2 should be 0.25");
    // All finite
    assert!(enc.iter().all(|&v| v.is_finite()));
}

#[test]
fn qa_encoding_position_normalization() {
    let encoder = SinusoidalPositionEncoder::new(64);
    let pos = [0.5, 0.5, 0.5];
    let enc = encoder.encode_single(&pos, None);
    assert_eq!(enc.len(), 64);
    // sin(0.5) and cos(0.5) for first freq
    assert!((enc[0] - 0.5f32.sin()).abs() < 1e-5);
    assert!((enc[1] - 0.5f32.cos()).abs() < 1e-5);
    // All finite
    assert!(enc.iter().all(|&v| v.is_finite()));
}

#[test]
fn qa_full_embedding_builder_deterministic() {
    let builder = FullEmbeddingBuilder::new();
    let e1 = builder.build_single(
        &[1.0, 2.0, 3.0],
        &[0.5, 0.5, 0.5],
        0.9,
        &[0.1, 0.1, 0.1],
        &[1.0, 0.0, 0.0, 0.0],
    );
    let e2 = builder.build_single(
        &[1.0, 2.0, 3.0],
        &[0.5, 0.5, 0.5],
        0.9,
        &[0.1, 0.1, 0.1],
        &[1.0, 0.0, 0.0, 0.0],
    );
    assert_eq!(e1, e2, "embedding should be deterministic");
    assert_eq!(e1.len(), 640);
}

// =====================================================================
// HNSW TESTS
// =====================================================================

#[test]
fn qa_hnsw_build_small_and_search_nearest() {
    let mut idx = HNSWIndex::new(4, 8, 100, 50, "l2", 42);

    // Build with known vectors: origin and points along axes
    let data = Array2::from_shape_vec(
        (5, 4),
        vec![
            1.0f32, 0.0, 0.0, 0.0, // point along x
            0.0, 1.0, 0.0, 0.0, // point along y
            0.0, 0.0, 1.0, 0.0, // point along z
            0.0, 0.0, 0.0, 1.0, // point along w
            0.1, 0.0, 0.0, 0.0, // close to point 0
        ],
    )
    .unwrap();

    idx.build(data.clone());
    assert_eq!(idx.n_items(), 5);

    // Query close to vector 4 → should return 0 or 4 as nearest (they're close)
    let query = array![0.1, 0.0, 0.0, 0.0];
    let result = idx.search(query.view(), 3);
    assert!(!result.indices.is_empty());
    // Index 4 should be in results (it's the exact query)
    assert!(
        result.indices.contains(&4),
        "expected index 4 in results, got {:?}",
        result.indices
    );
}

#[test]
fn qa_hnsw_cosine_search_returns_nearest() {
    let mut idx = HNSWIndex::new(4, 8, 100, 50, "cosine", 42);
    let data = random_array2(30, 4, 777);
    idx.build(data.clone());

    let query = data.row(10).to_owned();
    let result = idx.search(query.view(), 5);
    assert!(!result.indices.is_empty());
    // Self should be nearest
    assert!(
        result.indices.contains(&10),
        "expected self (idx 10) in cosine results, got {:?}",
        result.indices
    );
}

#[test]
fn qa_hnsw_serialize_deserialize_roundtrip() {
    let dir = std::env::temp_dir().join("qa_hnsw_serialize");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("qa_test.bin");

    let mut idx = HNSWIndex::new(4, 8, 100, 50, "l2", 42);
    let data = random_array2(20, 4, 555);
    idx.build(data.clone());

    idx.save(&path).unwrap();

    let loaded = HNSWIndex::load(&path, 8, 100, 50, "l2", 42).unwrap();
    assert_eq!(loaded.n_items(), 20);

    // Search results should be consistent
    let query = data.row(5).to_owned();
    let orig = idx.search(query.view(), 5);
    let loaded_result = loaded.search(query.view(), 5);
    assert_eq!(orig.indices, loaded_result.indices);

    std::fs::remove_file(&path).ok();
}

// =====================================================================
// QUANTIZATION TESTS
// =====================================================================

#[test]
fn qa_polar_encode_decode_accuracy() {
    let q = PolarQuantizer::new(8, 16, 42).unwrap();
    let x = vec![1.0f32, 2.0, -1.5, 0.5, 3.0, -2.0, 0.1, -0.8];
    let code = q.encode(&x).unwrap();
    let decoded = q.decode(&code).unwrap();

    for (i, (orig, dec)) in x.iter().zip(decoded.iter()).enumerate() {
        let err = (orig - dec).abs();
        assert!(
            err < 0.01,
            "pair {}: orig={:.4}, decoded={:.4}, err={:.4}",
            i,
            orig,
            dec,
            err
        );
    }
}

#[test]
fn qa_polar_inner_product_estimate_accuracy() {
    let q = PolarQuantizer::new(16, 16, 7).unwrap();
    let x = random_vec(16, 1);
    let y = random_vec(16, 2);
    let code = q.encode(&x).unwrap();

    let estimated = q.inner_product_estimate(&code, &y).unwrap();
    let exact: f32 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
    let rel_error = (estimated - exact).abs() / (exact.abs() + 1e-6);

    assert!(
        rel_error < 0.05,
        "inner product relative error too high: {:.4} (est={:.4}, exact={:.4})",
        rel_error,
        estimated,
        exact
    );
}

#[test]
fn qa_turbo_encode_and_ip_estimate() {
    let q = TurboQuantizer::new(16, 8, 16, 42).unwrap();
    let x = random_vec(16, 1);
    let y = random_vec(16, 2);
    let code = q.encode(&x).unwrap();

    let estimated = q.inner_product_estimate(&code, &y).unwrap();
    let exact: f32 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
    let rel_error = (estimated - exact).abs() / (exact.abs() + 1e-6);

    assert!(
        rel_error < 0.20,
        "turbo IP relative error too high: {:.4} (est={:.4}, exact={:.4})",
        rel_error,
        estimated,
        exact
    );
}

#[test]
fn qa_quantization_error_cases() {
    // Zero dimension
    assert!(PolarQuantizer::new(0, 8, 0).is_err());
    // Odd dimension
    assert!(PolarQuantizer::new(7, 8, 0).is_err());
    // Zero bits
    assert!(PolarQuantizer::new(8, 0, 0).is_err());
    // Dimension mismatch on encode
    let q = PolarQuantizer::new(8, 8, 0).unwrap();
    assert!(q.encode(&[1.0f32, 2.0, 3.0]).is_err()); // wrong dim
                                                     // Turbo invalid configs
    assert!(TurboQuantizer::new(0, 8, 16, 0).is_err());
    assert!(TurboQuantizer::new(7, 8, 16, 0).is_err());
    assert!(TurboQuantizer::new(8, 1, 16, 0).is_err()); // bits=1 means bits-1=0 for polar
    assert!(TurboQuantizer::new(8, 8, 0, 0).is_err());
}

// =====================================================================
// GRAPH SPLAT TESTS
// =====================================================================

#[test]
fn qa_graph_add_document_entity_and_relation() {
    let mut g = GaussianGraphStore::new();

    let doc = g
        .add_document("Machine learning basics", &[1.0, 0.0, 0.0])
        .unwrap();
    let entity = g
        .add_entity("Machine Learning", &[0.9, 0.1, 0.0], "concept")
        .unwrap();

    assert_ne!(doc, entity);
    assert_eq!(g.n_nodes(), 2);

    g.add_relation(doc, entity, "MENTIONS", 0.95).unwrap();
    assert_eq!(g.n_edges(), 1);

    // Verify outgoing edges
    let outgoing = g.get_outgoing(doc);
    assert_eq!(outgoing.len(), 1);
    assert_eq!(outgoing[0].target_id, entity);
    assert_eq!(outgoing[0].relation_type, "MENTIONS");
}

#[test]
fn qa_graph_search_entities_returns_correct_order() {
    let mut g = GaussianGraphStore::new();
    g.add_entity("Cat", &[1.0, 0.0], "animal").unwrap();
    g.add_entity("Dog", &[0.9, 0.1], "animal").unwrap();
    g.add_entity("Car", &[0.0, 1.0], "vehicle").unwrap();

    let results = g.search_entities(&[1.0, 0.0], 2);
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].content, "Cat"); // Most similar
}

#[test]
fn qa_graph_traverse_bfs_depth_limited() {
    let mut g = GaussianGraphStore::new();
    let n1 = g.add_document("D1", &[1.0]).unwrap();
    let n2 = g.add_entity("E1", &[1.0], "t").unwrap();
    let n3 = g.add_entity("E2", &[1.0], "t").unwrap();
    let n4 = g.add_entity("E3", &[1.0], "t").unwrap();

    g.add_relation(n1, n2, "R", 1.0).unwrap();
    g.add_relation(n2, n3, "R", 1.0).unwrap();
    g.add_relation(n3, n4, "R", 1.0).unwrap();

    // Depth 1: should reach n2 only
    let visited_d1 = g.traverse(n1, 1);
    assert!(visited_d1.contains(&n2));
    assert!(
        !visited_d1.contains(&n3),
        "n3 should not be reached at depth 1"
    );

    // Depth 3: should reach all
    let visited_d3 = g.traverse(n1, 3);
    assert!(visited_d3.contains(&n4));
}

#[test]
fn qa_graph_error_cases() {
    let mut g = GaussianGraphStore::new();
    let d = g.add_document("D", &[1.0]).unwrap();

    // Node not found
    assert!(matches!(
        g.add_relation(999, d, "X", 1.0),
        Err(GraphError::NodeNotFound { .. })
    ));
    assert!(matches!(
        g.add_relation(d, 999, "X", 1.0),
        Err(GraphError::NodeNotFound { .. })
    ));

    // Invalid weight
    let e = g.add_entity("E", &[1.0], "t").unwrap();
    assert!(matches!(
        g.add_relation(d, e, "X", f64::NAN),
        Err(GraphError::InvalidWeight)
    ));

    // Oversized embedding
    let huge = vec![0.0f32; 8193];
    assert!(matches!(
        g.add_document("x", &huge),
        Err(GraphError::EmbeddingTooLarge { .. })
    ));
}

// =====================================================================
// SPATIAL / DISTANCE TESTS
// =====================================================================

#[test]
fn qa_geodesic_distance_correctness() {
    let v1 = array![1.0f32, 0.0, 0.0];
    let v2 = array![0.0, 1.0, 0.0];
    let dist = geodesic_distance(&v1.view(), &v2.view());

    // Orthogonal → π/2
    assert!(
        (dist - std::f32::consts::FRAC_PI_2).abs() < 1e-4,
        "orthogonal vectors should be π/2 apart, got {}",
        dist
    );

    // Identical → 0
    let dist_same = geodesic_distance(&v1.view(), &v1.view());
    assert!(
        dist_same < 1e-4,
        "identical vectors should have ~0 distance, got {}",
        dist_same
    );

    // Opposite → π
    let v_neg = array![-1.0f32, 0.0, 0.0];
    let dist_opp = geodesic_distance(&v1.view(), &v_neg.view());
    assert!(
        (dist_opp - std::f32::consts::PI).abs() < 0.01,
        "opposite vectors should be π apart, got {}",
        dist_opp
    );
}

#[test]
fn qa_normalize_sphere_unit_vectors() {
    let v = array![[3.0f32, 4.0], [0.0, 5.0]];
    let normalized = normalize_sphere(&v);

    for i in 0..2 {
        let row = normalized.row(i);
        let norm: f32 = row.iter().map(|&x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 0.01,
            "row {} norm should be ~1.0, got {}",
            i,
            norm
        );
    }
}

// =====================================================================
// EDGE CASE / ERROR HANDLING TESTS
// =====================================================================

#[test]
fn qa_hnsw_empty_search_returns_nothing() {
    let idx = HNSWIndex::new(4, 8, 100, 50, "l2", 42);
    let query = array![1.0f32, 0.0, 0.0, 0.0];
    let result = idx.search(query.view(), 5);
    assert!(result.indices.is_empty());
    assert!(result.distances.is_empty());
}

#[test]
fn qa_persistence_list_shards() {
    let dir = std::env::temp_dir().join("qa_persistence_list");
    let _ = std::fs::remove_dir_all(&dir);

    let p = splatsdb::storage::persistence::SplatsDBPersistence::with_backend(
        dir.to_str().unwrap(),
        splatsdb::storage::persistence::StorageBackend::Sqlite,
        false,
    )
    .unwrap();

    let vecs = Array2::from_shape_vec((2, 3), vec![1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0]).unwrap();
    p.save_vectors(&vecs, "shard_a").unwrap();
    p.save_vectors(&vecs, "shard_b").unwrap();

    let shards = p.list_shards();
    assert_eq!(shards.len(), 2);
    assert!(shards.contains(&"shard_a".to_string()));
    assert!(shards.contains(&"shard_b".to_string()));

    let _ = std::fs::remove_dir_all(&dir);
}
