//! Integration tests for M2M Vector Search.

use splatdb::config::SplatDBConfig;
use splatdb::encoding::FullEmbeddingBuilder;
use splatdb::splats::SplatStore;
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand::Rng;

fn test_config() -> SplatDBConfig {
    let mut c = SplatDBConfig::default();
    c.max_splats = 2000;
    c.latent_dim = 64;
    c
}

fn random_normalized(n: usize, dim: usize, seed: u64) -> Array2<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut data = Array2::from_shape_fn((n, dim), |(_, _)| rng.gen::<f32>());
    for mut row in data.rows_mut() {
        let norm = row.dot(&row).sqrt().max(1e-10);
        row.mapv_inplace(|v: f32| v / norm);
    }
    data
}

#[test]
fn test_splat_store_workflow() {
    let mut store = SplatStore::new(test_config());

    // Add 1000 random vectors
    let data = random_normalized(1000, 64, 42);
    assert!(store.add_splat(&data));
    assert_eq!(store.n_active(), 1000);

    // Build index
    store.build_index();

    // Query nearest neighbors
    let query = data.row(0).to_owned();
    let results = store.find_neighbors(&query.view(), 10);

    assert_eq!(results.len(), 10);
    assert!(results[0].distance < 0.01, "First neighbor distance too high: {}", results[0].distance);
    assert_eq!(results[0].index, 0);

    // Distances should be increasing
    for i in 1..results.len() {
        assert!(results[i].distance >= results[i - 1].distance);
    }

    // Batch query
    let queries = data.slice(ndarray::s![..5, ..]).to_owned();
    let batch_results = store.find_neighbors_batch(&queries, 5);
    assert_eq!(batch_results.len(), 5);

    // Compact and entropy
    store.compact();
    assert_eq!(store.n_active(), 1000);
    let e = store.entropy();
    assert!(e > 0.0 && e <= 1.0, "Entropy out of range: {}", e);
}

#[test]
fn test_hrm2_end_to_end() {
    let config = test_config();
    let mut store = SplatStore::new(config);

    let data = random_normalized(500, 64, 123);
    store.add_splat(&data);
    store.build_index();

    let query = data.row(42).to_owned();
    let results = store.find_neighbors(&query.view(), 5);

    assert_eq!(results.len(), 5);
    assert_eq!(results[0].index, 42);
    assert!(results[0].distance < 0.01);

    // Outlier query
    let outlier = random_normalized(1, 64, 999);
    let outlier_results = store.find_neighbors(&outlier.row(0), 5);
    assert_eq!(outlier_results.len(), 5);
    for r in &outlier_results {
        assert!(r.distance > 0.0);
    }
}

#[test]
fn test_encoding_roundtrip() {
    let builder = FullEmbeddingBuilder::new();

    let positions = Array2::from_shape_fn((10, 3), |(i, j)| (i * 3 + j) as f32 * 0.1);
    let colors = Array2::from_shape_fn((10, 3), |(i, j)| ((i + j) as f32 * 0.05).sin().abs());
    let opacities = Array1::from_elem(10, 0.8f32);
    let scales = Array2::from_shape_fn((10, 3), |(i, j)| 0.1 + (i + j) as f32 * 0.01);
    let rotations = Array2::from_shape_fn((10, 4), |(i, j)| (i * 4 + j) as f32 * 0.25);

    let embeddings = builder.build(&positions, &colors, &opacities, &scales, &rotations);

    assert_eq!(embeddings.nrows(), 10);
    assert_eq!(embeddings.ncols(), 640);

    // Check no NaN or Inf
    for &val in embeddings.iter() {
        assert!(val.is_finite(), "Embedding contains NaN or Inf: {}", val);
    }

    // Check not all zeros
    let sum: f32 = embeddings.iter().map(|&v| v.abs()).sum();
    assert!(sum > 0.0, "Embeddings should not be all zeros");

    // Single embedding - check dimensions
    let single = builder.build_single(
        &[positions[[0, 0]], positions[[0, 1]], positions[[0, 2]]],
        &[colors[[0, 0]], colors[[0, 1]], colors[[0, 2]]],
        opacities[0],
        &[scales[[0, 0]], scales[[0, 1]], scales[[0, 2]]],
        &[rotations[[0, 0]], rotations[[0, 1]], rotations[[0, 2]], rotations[[0, 3]]],
    );
    assert_eq!(single.len(), 640);
    assert!(single.iter().all(|&v| v.is_finite()));
}
