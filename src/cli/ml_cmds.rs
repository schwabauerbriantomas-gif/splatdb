//! CLI command handlers for entity extraction, data lake, and embedding evaluation.
//!
//! These wrap user-facing modules that previously had no CLI exposure.

// ── Entity Extraction ──────────────────────────────────────────────────────

pub fn cmd_extract_entities(text: String, min_score: f64) {
    use splatdb::entity_extractor::{ExtractorConfig, SplatDBEntityExtractor};

    let config = ExtractorConfig {
        use_structural_patterns: true,
        use_ngram_analysis: true,
        min_ngram_score: min_score,
    };
    let extractor = SplatDBEntityExtractor::with_config(config);
    let candidates = extractor.extract(&text);

    let entries: Vec<serde_json::Value> = candidates
        .iter()
        .filter(|c| c.score >= min_score)
        .map(|c| {
            serde_json::json!({
                "text": c.text,
                "entity_type": c.entity_type,
                "score": (c.score * 10000.0).round() / 10000.0,
                "count": c.count,
            })
        })
        .collect();

    println!(
        "{}",
        serde_json::json!({
            "n_entities": entries.len(),
            "entities": entries,
        })
    );
}

// ── Data Lake ──────────────────────────────────────────────────────────────

pub fn cmd_lake_list(data_dir: String) {
    use splatdb::data_lake::DataLake;

    let path = format!("{}/data_lake", data_dir);
    let mut lake = DataLake::new(&path);
    if let Err(e) = lake.load() {
        eprintln!("Warning: could not load lake: {}", e);
    }
    let entries: Vec<serde_json::Value> = lake
        .list()
        .iter()
        .map(|e| {
            serde_json::json!({
                "id": e.id,
                "name": e.name,
                "n_vectors": e.n_vectors,
                "dim": e.dim,
                "description": e.description,
            })
        })
        .collect();

    println!(
        "{}",
        serde_json::json!({
            "n_entries": entries.len(),
            "datasets": entries,
        })
    );
}

pub fn cmd_lake_register(
    data_dir: String,
    id: String,
    name: String,
    n_vectors: usize,
    dim: usize,
    description: Option<String>,
) {
    use splatdb::data_lake::DataLake;

    let path = format!("{}/data_lake", data_dir);
    let mut lake = DataLake::new(&path);
    if let Err(e) = lake.load() {
        eprintln!("Warning: could not load lake: {}", e);
    }
    lake.register(&id, &name, n_vectors, dim, description.as_deref());
    if let Err(e) = lake.save() {
        eprintln!("Error saving lake: {}", e);
        std::process::exit(1);
    }
    println!(
        "{}",
        serde_json::json!({
            "status": "registered",
            "id": id,
            "name": name,
        })
    );
}

// ── Embedding Evaluation ───────────────────────────────────────────────────

pub fn cmd_eval_embeddings(dim: usize, n_queries: usize) {
    use ndarray::Array2;
    use splatdb::evaluate_embeddings::EmbeddingEvaluator;

    // Generate synthetic embeddings for a quick evaluation demo
    let n = 100;
    let mut raw = Vec::with_capacity(n * dim);
    for i in 0..n {
        for j in 0..dim {
            let val = ((i * dim + j) as f32 * 0.01).sin();
            raw.push(val);
        }
    }
    let embeddings = Array2::from_shape_vec((n, dim), raw).expect("valid shape");
    let labels: Vec<String> = (0..n).map(|i| format!("label_{}", i % 5)).collect();

    let report = EmbeddingEvaluator::evaluate(&embeddings, &labels, n_queries);

    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "precision_at_1": (report.precision_at_1 * 10000.0).round() / 10000.0,
            "precision_at_5": (report.precision_at_5 * 10000.0).round() / 10000.0,
            "precision_at_10": (report.precision_at_10 * 10000.0).round() / 10000.0,
            "recall_at_1": (report.recall_at_1 * 10000.0).round() / 10000.0,
            "recall_at_5": (report.recall_at_5 * 10000.0).round() / 10000.0,
            "recall_at_10": (report.recall_at_10 * 10000.0).round() / 10000.0,
            "mrr": (report.mrr * 10000.0).round() / 10000.0,
            "ndcg_at_10": (report.ndcg_at_10 * 10000.0).round() / 10000.0,
            "n_queries": report.n_queries,
            "eval_time_ms": (report.eval_time_ms * 1000.0).round() / 1000.0,
        }))
        .expect("Failed to serialize eval report")
    );
}
