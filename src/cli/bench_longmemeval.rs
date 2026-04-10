//! LongMemEval benchmark — SplatDB native methodology
//!
//! Pipeline (same as production):
//!   1. Load session embeddings + query embeddings from binary
//!   2. Load metadata JSON (wing=session_group, answer_session_ids)
//!   3. Create SplatStore with HNSW (M=32, ef_construction=400)
//!   4. Insert all sessions, register spatial metadata (wing=question_idx)
//!   5. Build index (HNSW + norm caching)
//!   6. For each query:
//!      a. SpatialIndex.filter(wing=qi) → ~48 candidate indices
//!      b. find_neighbors_filtered(query, candidates, k) → exact re-rank
//!   7. Measure: session recall (answer session in top-k), latency, QPS

use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::Read;
use std::time::Instant;

use ndarray::Array1;

/// Session metadata entry
#[derive(serde::Deserialize)]
struct SessionMeta {
    i: usize,    // global vector index
    qi: usize,   // question index (= wing)
    sid: String, // session ID
}

/// Query metadata entry
#[derive(serde::Deserialize)]
struct QueryMeta {
    qi: usize,
    answer_sids: Vec<String>,
    qtype: String,
}

/// Full metadata
#[derive(serde::Deserialize)]
struct BenchMeta {
    dim: usize,
    n_sessions: usize,
    n_queries: usize,
    sessions: Vec<SessionMeta>,
    queries: Vec<QueryMeta>,
}

/// Read SplatDB binary vectors: u64 n + u64 dim + f32[n][dim]
fn read_vectors(path: &str) -> (usize, usize, Vec<f32>) {
    let mut f = fs::File::open(path).unwrap_or_else(|e| {
        eprintln!("Cannot open {}: {}", path, e);
        std::process::exit(1);
    });
    let mut buf8 = [0u8; 8];
    f.read_exact(&mut buf8).unwrap();
    let n = u64::from_le_bytes(buf8) as usize;
    f.read_exact(&mut buf8).unwrap();
    let dim = u64::from_le_bytes(buf8) as usize;
    let _nbytes = n * dim * 4;
    let mut data = vec![0.0f32; n * dim];
    let bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut data);
    f.read_exact(bytes).unwrap();
    (n, dim, data)
}

pub fn cmd_bench_longmemeval(
    sessions_path: String,
    queries_path: String,
    meta_path: String,
    k: usize,
    ef_search: usize,
    over_fetch: usize,
) {
    eprintln!("╔══════════════════════════════════════════════════════════════╗");
    eprintln!("║  LongMemEval — SplatDB Native Pipeline Benchmark           ║");
    eprintln!("╚══════════════════════════════════════════════════════════════╝");

    // ── Load metadata ──
    let meta: BenchMeta = serde_json::from_str(&fs::read_to_string(&meta_path).expect("read meta"))
        .expect("parse meta");
    eprintln!(
        "  Metadata: {} sessions, {} queries, dim={}",
        meta.n_sessions, meta.n_queries, meta.dim
    );

    // ── Load vectors ──
    eprintln!("[1/5] Loading vectors...");
    let (n_sess, dim_s, sess_data) = read_vectors(&sessions_path);
    let (n_q, dim_q, query_data) = read_vectors(&queries_path);
    assert_eq!(dim_s, meta.dim, "session dim mismatch");
    assert_eq!(dim_q, meta.dim, "query dim mismatch");
    assert_eq!(n_sess, meta.n_sessions, "session count mismatch");
    assert_eq!(n_q, meta.n_queries, "query count mismatch");
    eprintln!(
        "  Sessions: {} x {}d, Queries: {} x {}d",
        n_sess, dim_s, n_q, dim_q
    );

    // ── Build mappings ──
    // qi → list of global session indices
    let mut qi_to_sessions: HashMap<usize, Vec<usize>> = HashMap::new();
    // qi → set of answer session IDs
    let mut qi_to_answer_sids: HashMap<usize, HashSet<String>> = HashMap::new();
    // global_idx → session_id
    let mut idx_to_sid: HashMap<usize, String> = HashMap::new();
    // session_id → global_idx
    let mut sid_to_idx: HashMap<String, usize> = HashMap::new();

    for s in &meta.sessions {
        qi_to_sessions.entry(s.qi).or_default().push(s.i);
        idx_to_sid.insert(s.i, s.sid.clone());
        sid_to_idx.insert(s.sid.clone(), s.i);
    }
    for q in &meta.queries {
        qi_to_answer_sids.insert(q.qi, q.answer_sids.iter().cloned().collect());
    }

    // ── Build SplatStore with spatial index ──
    eprintln!("[2/5] Building SplatStore + SpatialIndex...");
    let t0 = Instant::now();

    let mut config = splatdb::config::SplatDBConfig::advanced(None);
    config.latent_dim = meta.dim;
    config.max_splats = meta.n_sessions + 1000;
    config.enable_hnsw = true;
    config.hnsw_m = 32;
    config.hnsw_ef_construction = 400;
    config.hnsw_ef_search = ef_search;
    config.hnsw_metric = "l2".to_string();
    config.over_fetch = over_fetch;

    let mut store = splatdb::splats::SplatStore::new(config.clone());

    // Insert all sessions
    for i in 0..n_sess {
        let vec: &[f32] = &sess_data[i * dim_s..(i + 1) * dim_s];
        store.insert(vec);
    }
    eprintln!(
        "  Inserted {} vectors in {:.1}s",
        n_sess,
        t0.elapsed().as_secs_f64()
    );

    // Build index (HNSW + HRM2)
    let t_build = Instant::now();
    store.build_index();
    let build_ms = t_build.elapsed().as_millis();
    eprintln!(
        "  Index built in {}ms ({:.1}s)",
        build_ms,
        build_ms as f64 / 1000.0
    );

    // Build spatial index: register each session with wing=question_idx
    let mut spatial = splatdb::spatial::SpatialIndex::new();
    for s in &meta.sessions {
        spatial.register_doc(
            &format!("sess_{}", s.i),
            Some(&format!("wing_{}", s.qi)), // wing = question group
            None,                            // room = auto from clustering
            None,                            // hall = not used here
        );
    }

    let total_setup_ms = t0.elapsed().as_millis();
    eprintln!(
        "  Total setup (insert + build + spatial): {:.1}s",
        total_setup_ms as f64 / 1000.0
    );

    // ── Run benchmark: spatial pre-filter → find_neighbors_filtered ──
    eprintln!(
        "[3/5] Running {} queries (spatial pre-filter → vector search)...",
        n_q
    );
    eprintln!(
        "  Method: SpatialIndex.filter(wing=qi) → find_neighbors_filtered(query, candidates, k)"
    );

    let k_values = [1, 3, 5, 10];
    let mut recall_counts: HashMap<usize, (usize, usize)> = HashMap::new(); // (hits, total)
    let mut per_type_recall: HashMap<String, HashMap<usize, (usize, usize)>> = HashMap::new();
    let mut latencies: Vec<f64> = Vec::with_capacity(n_q);

    let t_search = Instant::now();

    for q in &meta.queries {
        let qi = q.qi;
        let query_vec: &[f32] = &query_data[qi * dim_q..(qi + 1) * dim_q];
        let query_arr = Array1::from_vec(query_vec.to_vec());

        // STEP 1: Spatial pre-filter — get candidates for this wing
        let filter = splatdb::spatial::SpatialFilter {
            wing: Some(format!("wing_{}", qi)),
            room: None,
            hall: None,
        };
        let filtered_doc_ids = spatial.filter(&filter);
        let candidate_indices: Vec<usize> = filtered_doc_ids
            .iter()
            .filter_map(|id| {
                id.strip_prefix("sess_")
                    .and_then(|s| s.parse::<usize>().ok())
            })
            .collect();

        // STEP 2: Vector search on filtered candidates (exact re-rank)
        let t_q = Instant::now();
        let results =
            store.find_neighbors_filtered(&query_arr.view(), &candidate_indices, k.max(10));
        latencies.push(t_q.elapsed().as_micros() as f64 / 1000.0);

        // Build result set
        let result_indices: HashSet<String> = results
            .iter()
            .take(k)
            .map(|r| idx_to_sid.get(&r.index).cloned().unwrap_or_default())
            .collect();

        // Check recall at each k
        let answer_sids = qi_to_answer_sids.get(&qi).cloned().unwrap_or_default();
        let _hit = result_indices.iter().any(|sid| answer_sids.contains(sid));

        for &kv in &k_values {
            let result_k: HashSet<String> = results
                .iter()
                .take(kv)
                .map(|r| idx_to_sid.get(&r.index).cloned().unwrap_or_default())
                .collect();
            let hit_k = result_k.iter().any(|sid| answer_sids.contains(sid));
            let entry = recall_counts.entry(kv).or_insert((0, 0));
            if hit_k {
                entry.0 += 1;
            }
            entry.1 += 1;

            let pt = per_type_recall.entry(q.qtype.clone()).or_default();
            let pte = pt.entry(kv).or_insert((0, 0));
            if hit_k {
                pte.0 += 1;
            }
            pte.1 += 1;
        }

        if qi % 100 == 0 {
            eprintln!("  Query {}/{}...", qi, n_q);
        }
    }

    let search_ms = t_search.elapsed().as_millis();

    // ── Results ──
    eprintln!("\n╔══════════════════════════════════════════════════════════════╗");
    eprintln!("║  RESULTS — SplatDB Native Pipeline                         ║");
    eprintln!("╚══════════════════════════════════════════════════════════════╝");
    eprintln!("\n  Methodology:");
    eprintln!("    1. SpatialIndex.filter(wing=question_group) → ~48 candidates");
    eprintln!("    2. find_neighbors_filtered(query, candidates, k) → exact re-rank");
    eprintln!("    3. Check if answer session in top-k results");
    eprintln!(
        "\n  SplatDB config: HNSW M=32, ef_construction=400, ef_search={}, over_fetch={}x",
        ef_search, over_fetch
    );
    eprintln!("  Embeddings: all-MiniLM-L6-v2 ({}d, normalized)", meta.dim);

    eprintln!("\n  Session Recall:");
    for &kv in &k_values {
        let (hits, total) = recall_counts.get(&kv).copied().unwrap_or((0, 1));
        let pct = hits as f64 / total as f64 * 100.0;
        eprintln!("    Recall@{:2}: {}/{} = {:.1}%", kv, hits, total, pct);
    }

    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = latencies[latencies.len() / 2];
    let p95 = latencies[(latencies.len() as f64 * 0.95) as usize];
    let qps = n_q as f64 / (search_ms as f64 / 1000.0);

    eprintln!("\n  Speed:");
    eprintln!("    Index build:  {:.1}s", build_ms as f64 / 1000.0);
    eprintln!(
        "    Total search: {:.1}s ({} queries)",
        search_ms as f64 / 1000.0,
        n_q
    );
    eprintln!("    P50 latency:  {:.3}ms", p50);
    eprintln!("    P95 latency:  {:.3}ms", p95);
    eprintln!("    QPS:          {:.0}", qps);

    eprintln!("\n  Per-type Recall@{}:", k);
    let mut types_sorted: Vec<_> = per_type_recall.iter().collect();
    types_sorted.sort_by_key(|(t, _)| t.as_str());
    for (qtype, kmap) in &types_sorted {
        let (hits, total) = kmap.get(&k).copied().unwrap_or((0, 1));
        let pct = hits as f64 / total as f64 * 100.0;
        eprintln!("    {:<35}: {:3}/{:3} = {:.1}%", qtype, hits, total, pct);
    }

    // ── JSON output ──
    let mut output = serde_json::Map::new();
    output.insert(
        "benchmark".into(),
        serde_json::Value::String("LongMemEval-S-cleaned — SplatDB Native Pipeline".into()),
    );
    output.insert(
        "methodology".into(),
        serde_json::Value::String(
            "SpatialIndex.filter(wing) → find_neighbors_filtered (exact re-rank)".into(),
        ),
    );
    output.insert(
        "dim".into(),
        serde_json::Value::Number(serde_json::Number::from(meta.dim)),
    );
    output.insert(
        "n_sessions".into(),
        serde_json::Value::Number(serde_json::Number::from(meta.n_sessions)),
    );
    output.insert(
        "n_queries".into(),
        serde_json::Value::Number(serde_json::Number::from(meta.n_queries)),
    );

    let mut cfg = serde_json::Map::new();
    cfg.insert(
        "hnsw_m".into(),
        serde_json::Value::Number(serde_json::Number::from(32)),
    );
    cfg.insert(
        "hnsw_ef_construction".into(),
        serde_json::Value::Number(serde_json::Number::from(400)),
    );
    cfg.insert(
        "hnsw_ef_search".into(),
        serde_json::Value::Number(serde_json::Number::from(ef_search)),
    );
    cfg.insert(
        "over_fetch".into(),
        serde_json::Value::Number(serde_json::Number::from(over_fetch)),
    );
    cfg.insert("metric".into(), serde_json::Value::String("l2".into()));
    output.insert("splatdb_config".into(), serde_json::Value::Object(cfg));

    output.insert(
        "build_time_s".into(),
        serde_json::json!(build_ms as f64 / 1000.0),
    );
    output.insert(
        "search_time_s".into(),
        serde_json::json!(search_ms as f64 / 1000.0),
    );

    let mut recall_map = serde_json::Map::new();
    for &kv in &k_values {
        let (h, t) = recall_counts.get(&kv).copied().unwrap_or((0, 1));
        recall_map.insert(format!("@{}", kv), serde_json::json!(h as f64 / t as f64));
    }
    output.insert("recall".into(), serde_json::Value::Object(recall_map));

    output.insert("qps".into(), serde_json::json!(qps));
    output.insert("p50_ms".into(), serde_json::json!(p50));
    output.insert("p95_ms".into(), serde_json::json!(p95));

    let mut per_type = serde_json::Map::new();
    for (qt, kmap) in &types_sorted {
        let (h, t) = kmap.get(&k).copied().unwrap_or((0, 1));
        per_type.insert(
            (*qt).clone(),
            serde_json::json!({"recall": h as f64 / t as f64, "n": t}),
        );
    }
    output.insert("per_type".into(), serde_json::Value::Object(per_type));

    let json_str = serde_json::to_string_pretty(&serde_json::Value::Object(output)).unwrap();
    println!("{}", json_str);
}
