//! Chaos tests for SplatsDB — stress, concurrency, edge cases, and WAL durability.

use std::sync::{Arc, Mutex};
use std::thread;

use ndarray::{Array1, Array2};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use splatsdb::config::SplatsDBConfig;
use splatsdb::splats::SplatStore;
use splatsdb::storage::wal::WriteAheadLog;

#[allow(clippy::field_reassign_with_default)]
fn chaos_config(dim: usize, max_splats: usize) -> SplatsDBConfig {
    let mut c = SplatsDBConfig::default();
    c.latent_dim = dim;
    c.max_splats = max_splats;
    c.enable_hnsw = false; // Keep simple for chaos tests
    c.enable_quantization = false;
    c.enable_lsh = false;
    c.enable_semantic_memory = false;
    c.enable_graph = false;
    c
}

fn random_vec(dim: usize, rng: &mut ChaCha8Rng) -> Vec<f32> {
    (0..dim).map(|_| rng.gen::<f32>()).collect()
}

fn random_normalized_vec(dim: usize, rng: &mut ChaCha8Rng) -> Vec<f32> {
    let mut v: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-10);
    for x in &mut v {
        *x /= norm;
    }
    v
}

// ─── Test 1: Concurrent stress test ───────────────────────────────

#[test]
fn chaos_concurrent_stress() {
    let dim = 64;
    let max_splats = 20000;
    let store = Arc::new(Mutex::new(SplatStore::new(chaos_config(dim, max_splats))));
    let n_threads = 10;
    let ops_per_thread = 1000;

    let mut handles = Vec::new();
    for t in 0..n_threads {
        let store = Arc::clone(&store);
        let handle = thread::spawn(move || {
            let mut rng = ChaCha8Rng::seed_from_u64(42 + t as u64);
            let mut insert_count = 0usize;
            let mut search_count = 0usize;
            let mut errors = 0usize;

            for _ in 0..ops_per_thread {
                let op: f32 = rng.gen();
                if op < 0.5 {
                    // Insert
                    let vec = random_normalized_vec(dim, &mut rng);
                    let mut s = store.lock().unwrap();
                    if s.insert(&vec).is_some() {
                        insert_count += 1;
                    }
                } else {
                    // Search
                    let query = random_vec(dim, &mut rng);
                    let s = store.lock().unwrap();
                    match s.search(&query, 5) {
                        results => {
                            if results.len() <= 5 {
                                search_count += 1;
                            } else {
                                errors += 1;
                            }
                        }
                    }
                }
            }
            (insert_count, search_count, errors)
        });
        handles.push(handle);
    }

    let mut total_inserts = 0;
    let mut total_searches = 0;
    let mut total_errors = 0;
    for h in handles {
        let (ins, sch, err) = h.join().expect("thread panicked");
        total_inserts += ins;
        total_searches += sch;
        total_errors += err;
    }

    eprintln!(
        "[chaos_concurrent] inserts={} searches={} errors={}",
        total_inserts, total_searches, total_errors
    );
    assert_eq!(total_errors, 0, "No search errors expected");
    assert!(total_inserts > 0, "Should have inserted some vectors");
    assert!(total_searches > 0, "Should have done some searches");

    let s = store.lock().unwrap();
    eprintln!(
        "[chaos_concurrent] final n_active={}/{}",
        s.n_active(),
        s.max_splats()
    );
}

// ─── Test 2: Memory pressure — fill to capacity ───────────────────

#[test]
fn chaos_memory_pressure() {
    let dim = 32;
    let max_splats = 10000;
    let mut store = SplatStore::new(chaos_config(dim, max_splats));
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    // Fill up to max capacity
    let batch_size = 1000;
    let mut total_inserted = 0;
    loop {
        let n = batch_size.min(max_splats - total_inserted);
        if n == 0 {
            break;
        }
        let data: Vec<f32> = (0..n * dim).map(|_| rng.gen::<f32>()).collect();
        let arr = Array2::from_shape_vec((n, dim), data).unwrap();
        if !store.add_splat(&arr) {
            break;
        }
        total_inserted += n;
    }

    assert_eq!(total_inserted, max_splats, "Should fill to max capacity");
    assert_eq!(store.n_active(), max_splats);

    // Try inserting one more — should fail
    let one_more = vec![0.0f32; dim];
    assert!(
        store.insert(&one_more).is_none(),
        "Should reject beyond capacity"
    );

    // Build index and search at capacity
    store.build_index();
    let query = random_vec(dim, &mut rng);
    let results = store.search(&query, 10);
    assert_eq!(results.len(), 10, "Should return k=10 results at capacity");

    eprintln!(
        "[chaos_memory_pressure] filled {}/{} vectors, search returned {}",
        total_inserted,
        max_splats,
        results.len()
    );
}

// ─── Test 3: Large vector stress (max dimension) ──────────────────

#[test]
fn chaos_large_vectors() {
    // Use a reasonably large dimension that won't blow memory
    let dim = 1024;
    let max_splats = 500;
    let mut store = SplatStore::new(chaos_config(dim, max_splats));
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    // Insert large-dim vectors
    for i in 0..200 {
        let vec = random_normalized_vec(dim, &mut rng);
        let idx = store.insert(&vec);
        assert!(idx.is_some(), "Insert {} should succeed", i);
    }
    assert_eq!(store.n_active(), 200);

    // Search with large vectors
    for _ in 0..50 {
        let query = random_vec(dim, &mut rng);
        let results = store.search(&query, 10);
        assert!(results.len() <= 10);
        for r in &results {
            assert!(r.distance.is_finite(), "Distance should be finite");
            assert_eq!(r.mu.len(), dim, "Result vector should have correct dim");
        }
    }

    eprintln!(
        "[chaos_large_vectors] dim={} n_active={} search OK",
        dim,
        store.n_active()
    );
}

// ─── Test 4: Rapid fire — 10000 sequential inserts then search ────

#[test]
fn chaos_rapid_fire() {
    let dim = 64;
    let max_splats = 15000;
    let mut store = SplatStore::new(chaos_config(dim, max_splats));
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    // Rapid sequential inserts
    let start = std::time::Instant::now();
    for i in 0..10000 {
        let vec = random_normalized_vec(dim, &mut rng);
        let idx = store
            .insert(&vec)
            .unwrap_or_else(|| panic!("Insert {} failed", i));
        assert_eq!(idx, i, "Index should be sequential");
    }
    let insert_time = start.elapsed();
    eprintln!(
        "[chaos_rapid_fire] 10000 inserts in {:?} ({:.0}/sec)",
        insert_time,
        10000.0 / insert_time.as_secs_f64()
    );

    assert_eq!(store.n_active(), 10000);

    // Build and search
    let start = std::time::Instant::now();
    store.build_index();
    let build_time = start.elapsed();
    eprintln!("[chaos_rapid_fire] build_index in {:?}", build_time);

    // Search for each 100th vector — should find itself as nearest
    let mut recall_count = 0;
    for i in (0..10000).step_by(100) {
        let _query = Array1::from_vec(
            (0..dim)
                .map(|j| {
                    let base = (i * dim + j) as f32 * 0.001;
                    base.sin()
                })
                .collect(),
        );
        // Actually fetch from store's mu
        let mu = store.get_mu().unwrap();
        let q = mu.row(i).to_owned();
        let results = store.find_neighbors(&q.view(), 1);
        if results[0].index == i {
            recall_count += 1;
        }
    }
    eprintln!("[chaos_rapid_fire] self-recall: {}/100", recall_count);
    assert!(
        recall_count >= 95,
        "Self-recall should be >= 95%, got {}%",
        recall_count
    );
}

// ─── Test 5: Interleaved operations ───────────────────────────────

#[test]
fn chaos_interleaved_operations() {
    let dim = 32;
    let max_splats = 5000;
    let mut store = SplatStore::new(chaos_config(dim, max_splats));
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    let mut inserted_ids: Vec<usize> = Vec::new();

    // Interleaved: insert, search, insert, search in pseudo-random order
    for round in 0..200 {
        let op = round % 5;
        match op {
            0..=2 => {
                // Insert a batch
                let n = (rng.gen::<usize>() % 20) + 1;
                let data: Vec<f32> = (0..n * dim).map(|_| rng.gen::<f32>()).collect();
                let arr = Array2::from_shape_vec((n, dim), data).unwrap();
                if store.add_splat(&arr) {
                    for i in 0..n {
                        inserted_ids.push(store.n_active() - n + i);
                    }
                }
            }
            3 => {
                // Search
                if !inserted_ids.is_empty() {
                    let query = random_vec(dim, &mut rng);
                    let results = store.search(&query, 5);
                    assert!(results.len() <= 5.min(inserted_ids.len()));
                    for r in &results {
                        assert!(r.distance.is_finite());
                    }
                }
            }
            4 => {
                // Compact
                store.compact();
            }
            _ => unreachable!(),
        }
    }

    // Final validation
    assert_eq!(store.n_active(), inserted_ids.len());
    let entropy = store.entropy();
    assert!((0.0..=1.0).contains(&entropy), "Entropy: {}", entropy);

    eprintln!(
        "[chaos_interleaved] n_active={} entropy={:.4}",
        store.n_active(),
        entropy
    );
}

// ─── Test 6: Edge cases ───────────────────────────────────────────

#[test]
fn chaos_edge_cases() {
    let dim = 64;
    let max_splats = 1000;
    let mut store = SplatStore::new(chaos_config(dim, max_splats));

    // 6a. Insert identical vectors
    let identical = vec![1.0f32; dim];
    for _ in 0..100 {
        assert!(store.insert(&identical).is_some());
    }
    assert_eq!(store.n_active(), 100);

    // Search with the identical vector — all distances should be ~0
    let results = store.search(&identical, 10);
    assert_eq!(results.len(), 10);
    for r in &results {
        assert!(
            r.distance < 1e-5,
            "Identical vector distance should be ~0, got {}",
            r.distance
        );
    }

    // 6b. Search with zero vector
    let zero_vec = vec![0.0f32; dim];
    let results = store.search(&zero_vec, 5);
    assert_eq!(results.len(), 5);
    for r in &results {
        assert!(
            r.distance.is_finite(),
            "Zero vector search should return finite distances"
        );
    }

    // 6c. Search with k=0 — should return empty
    let results = store.search(&identical, 0);
    assert!(results.is_empty());

    // 6d. Search with k > n_active
    let results = store.search(&identical, 200);
    assert_eq!(results.len(), 100, "Should return at most n_active results");

    // 6e. Search on empty store
    let empty_store = SplatStore::new(chaos_config(dim, 100));
    let results = empty_store.search(&identical, 5);
    assert!(results.is_empty());

    // 6f. Insert wrong dimension — should return None
    let wrong_dim = vec![0.0f32; dim + 1];
    assert!(store.insert(&wrong_dim).is_none());

    // 6g. NaN vector
    let mut nan_vec = vec![0.0f32; dim];
    nan_vec[0] = f32::NAN;
    // Insert allows NaN (it's just data), but search should handle it
    let _nan_insert = store.insert(&nan_vec);
    let results = store.search(&nan_vec, 5);
    // Results may contain NaN distances but should not panic
    assert_eq!(results.len(), 5);

    // 6h. Inf vector
    let mut inf_vec = vec![0.0f32; dim];
    inf_vec[0] = f32::INFINITY;
    let _inf_insert = store.insert(&inf_vec);
    let results = store.search(&inf_vec, 5);
    assert_eq!(results.len(), 5);

    // 6i. Very large values
    let large_vec: Vec<f32> = (0..dim).map(|i| (i as f32).exp()).collect();
    assert!(store.insert(&large_vec).is_some());

    // 6j. Very small values
    let small_vec: Vec<f32> = (0..dim).map(|i| -(i as f32).exp()).collect();
    assert!(store.insert(&small_vec).is_some());

    // 6k. Mixed positive/negative
    let mixed: Vec<f32> = (0..dim)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    assert!(store.insert(&mixed).is_some());

    eprintln!(
        "[chaos_edge_cases] all edge cases passed, n_active={}",
        store.n_active()
    );
}

// ─── Test 7: WAL under load ──────────────────────────────────────

#[test]
fn chaos_wal_under_load() {
    let dir = std::env::temp_dir().join("splatsdb_chaos_wal_test");
    std::fs::create_dir_all(&dir).unwrap();
    let wal_path = dir.join("chaos.wal");
    let path_str = wal_path.to_str().unwrap();

    // Create WAL
    let wal = WriteAheadLog::new(path_str, 10).expect("WAL create");
    let wal = Arc::new(Mutex::new(wal));

    // Concurrent writes
    let n_threads = 5;
    let writes_per_thread = 200;
    let mut handles = Vec::new();

    for t in 0..n_threads {
        let wal = Arc::clone(&wal);
        let handle = thread::spawn(move || {
            let mut lsn_list = Vec::new();
            for i in 0..writes_per_thread {
                let data = serde_json::json!({
                    "thread": t,
                    "op": i,
                    "payload": format!("data_{}_{}", t, i),
                });
                let lsn = wal
                    .lock()
                    .unwrap()
                    .log_operation("insert", data)
                    .expect("log_operation");
                lsn_list.push(lsn);
            }
            lsn_list
        });
        handles.push(handle);
    }

    let mut all_lsns = Vec::new();
    for h in handles {
        let lsns = h.join().expect("thread panicked");
        all_lsns.extend(lsns);
    }

    let total_entries = all_lsns.len();
    eprintln!("[chaos_wal] {} concurrent writes completed", total_entries);

    // Checkpoint
    wal.lock().unwrap().checkpoint().expect("checkpoint");

    // Recover all entries
    let entries = wal.lock().unwrap().recover().expect("recover");
    assert_eq!(
        entries.len(),
        total_entries + 1, // +1 for checkpoint
        "Should recover all entries + checkpoint"
    );

    // Verify LSNs are unique and sequential-ish
    let mut entry_lsns: Vec<u64> = entries.iter().map(|e| e.lsn).collect();
    entry_lsns.sort();
    entry_lsns.dedup();
    assert_eq!(
        entry_lsns.len(),
        total_entries + 1,
        "All LSNs should be unique"
    );

    // Truncate half the entries
    let mid_lsn = all_lsns[all_lsns.len() / 2];
    wal.lock().unwrap().truncate(mid_lsn).expect("truncate");

    // Verify truncation worked
    let remaining = wal
        .lock()
        .unwrap()
        .recover()
        .expect("recover after truncate");
    assert!(
        remaining.len() < total_entries,
        "Should have fewer entries after truncate"
    );

    // Write more after truncate
    for i in 0..50 {
        wal.lock()
            .unwrap()
            .log_operation("post_truncate", serde_json::json!({"i": i}))
            .expect("post-truncate write");
    }

    let final_entries = wal.lock().unwrap().recover().expect("final recover");
    let post_truncate_ops = final_entries
        .iter()
        .filter(|e| e.operation == "post_truncate")
        .count();
    assert_eq!(
        post_truncate_ops, 50,
        "Should have 50 post-truncate entries"
    );

    // Close
    wal.lock().unwrap().close().expect("close");

    // Cleanup
    std::fs::remove_dir_all(&dir).ok();

    eprintln!(
        "[chaos_wal] total={}, after_truncate={}, final={}",
        total_entries,
        remaining.len(),
        final_entries.len()
    );
}

// ─── Test 8: Compact under pressure ──────────────────────────────

#[test]
fn chaos_compact_heavy() {
    let dim = 32;
    let max_splats = 5000;
    let mut store = SplatStore::new(chaos_config(dim, max_splats));
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    // Insert a lot of vectors
    for _ in 0..3000 {
        let vec = random_normalized_vec(dim, &mut rng);
        store.insert(&vec);
    }
    assert_eq!(store.n_active(), 3000);

    // Compact on clean data should be a no-op
    let before = store.n_active();
    store.compact();
    let after = store.n_active();
    assert_eq!(
        after, before,
        "Compact on clean data should not change count"
    );

    // Build index and search
    store.build_index();
    let query = random_vec(dim, &mut rng);
    let results = store.search(&query, 10);
    assert!(results.len() <= 10);
    for r in &results {
        assert!(r.distance.is_finite());
    }

    eprintln!(
        "[chaos_compact] {} vectors, compact no-op, search OK",
        after
    );
}
