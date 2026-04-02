//! Embedding evaluation metrics.
//!
//! Computes precision, recall, MRR, NDCG, and embedding quality metrics.
//! Includes benchmark texts for standardized evaluation.
//! Ported from m2m-vector-search Python.

use ndarray::Array2;
use serde::Serialize;
use std::time::Instant;

/// Benchmark texts for standardized embedding evaluation.
pub const BENCHMARK_TEXTS: &[&str] = &[
    "Artificial intelligence transforms healthcare by enabling early disease detection.",
    "The quantum computer solved a complex optimization problem in seconds.",
    "Climate change models predict rising sea levels by 2050.",
    "Blockchain technology enables decentralized financial transactions.",
    "Neural architecture search automates the design of deep learning models.",
    "Genome sequencing reveals genetic mutations linked to rare diseases.",
    "Self-driving cars use lidar and cameras to navigate complex environments.",
    "Natural language generation produces human-like text from structured data.",
    "Federated learning enables model training across distributed datasets.",
    "Reinforcement learning from human feedback improves language model alignment.",
    "The attention mechanism allows transformers to capture long-range dependencies.",
    "Convolutional networks excel at spatial pattern recognition in images.",
    "Graph neural networks model relationships in social networks and molecules.",
    "Energy-efficient computing reduces the environmental impact of large AI models.",
    "Few-shot learning enables models to generalize from very few examples.",
    "Vector databases store high-dimensional embeddings for fast similarity search.",
    "Embedding space geometry affects the quality of nearest neighbor retrieval.",
    "Dimensionality reduction preserves semantic relationships while compressing vectors.",
    "Cross-modal retrieval finds images matching text descriptions and vice versa.",
    "Hyperparameter optimization improves model performance through systematic search.",
];

/// Embedding evaluation results.
#[derive(Debug, Clone, Serialize)]
pub struct EmbeddingEvalReport {
    pub precision_at_1: f64,
    pub precision_at_5: f64,
    pub precision_at_10: f64,
    pub recall_at_1: f64,
    pub recall_at_5: f64,
    pub recall_at_10: f64,
    pub mrr: f64,
    pub ndcg_at_10: f64,
    pub avg_similarity: f64,
    pub embedding_norm_std: f64,
    pub n_queries: usize,
    pub eval_time_ms: f64,
}

/// Comparison result between two embedding sets.
#[derive(Debug, Clone, Serialize)]
pub struct ComparisonReport {
    pub student_precision_at_10: f64,
    pub teacher_precision_at_10: f64,
    pub recall_preservation: f64,
    pub student_mrr: f64,
    pub teacher_mrr: f64,
    pub student_avg_sim: f64,
    pub teacher_avg_sim: f64,
}

/// Latency measurement result.
#[derive(Debug, Clone, Serialize)]
pub struct LatencyReport {
    pub n_queries: usize,
    pub total_time_ms: f64,
    pub avg_latency_us: f64,
    pub p50_latency_us: f64,
    pub p95_latency_us: f64,
    pub p99_latency_us: f64,
}

/// Evaluate embeddings quality.
pub struct EmbeddingEvaluator;

impl EmbeddingEvaluator {
    /// Evaluate retrieval quality given embeddings and labels.
    ///
    /// # Arguments
    /// * `embeddings` - Matrix [N, D] of embeddings
    /// * `labels` - Labels for each embedding
    /// * `n_queries` - Number of queries to evaluate (sampled from start)
    pub fn evaluate(embeddings: &Array2<f32>, labels: &[String], n_queries: usize) -> EmbeddingEvalReport {
        let start = Instant::now();
        let n = embeddings.nrows();
        let actual_queries = n_queries.min(n);

        let mut p1_sum = 0.0;
        let mut p5_sum = 0.0;
        let mut p10_sum = 0.0;
        let mut r1_sum = 0.0;
        let mut r5_sum = 0.0;
        let mut r10_sum = 0.0;
        let mut mrr_sum = 0.0;
        let mut ndcg_sum = 0.0;
        let mut sim_sum = 0.0;
        let mut sim_count = 0;

        for i in 0..actual_queries {
            let query = embeddings.row(i);
            let query_label = &labels[i];

            // Compute cosine similarities
            let mut sims: Vec<(usize, f64)> = Vec::with_capacity(n);
            for j in 0..n {
                if j == i { continue; }
                let other = embeddings.row(j);
                let dot: f32 = query.iter().zip(other.iter()).map(|(a, b)| a * b).sum();
                let norm_q: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
                let norm_o: f32 = other.iter().map(|x| x * x).sum::<f32>().sqrt();
                let sim = if norm_q > 0.0 && norm_o > 0.0 {
                    (dot / (norm_q * norm_o)) as f64
                } else {
                    0.0
                };
                sims.push((j, sim));
                sim_sum += sim;
                sim_count += 1;
            }

            // Sort by similarity descending
            sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Find how many have the same label
            let total_relevant: usize = labels.iter().filter(|l| **l == *query_label).count();
            if total_relevant == 0 { continue; }
            let relevant_minus_self = total_relevant.saturating_sub(1);

            // Compute metrics
            for (rank, (idx, _)) in sims.iter().enumerate() {
                let rank_0 = rank; // 0-indexed
                if labels[*idx] == *query_label {
                    if rank_0 == 0 { p1_sum += 1.0; }
                    if rank_0 < 5 { p5_sum += 1.0; }
                    if rank_0 < 10 { p10_sum += 1.0; }
                    if rank_0 == 0 { r1_sum += 1.0; }
                    if rank_0 < 5 { r5_sum += 1.0; }
                    if rank_0 < 10 { r10_sum += 1.0; }

                    // MRR
                    mrr_sum += 1.0 / (rank_0 + 1) as f64;

                    break; // Only count first relevant for MRR
                }
            }

            // NDCG@10
            let mut dcg = 0.0;
            let mut idcg = 0.0;
            let k10 = sims.iter().take(10);
            for (rank, (idx, _)) in k10.enumerate() {
                let rel = if labels[*idx].as_str() == *query_label { 1.0 } else { 0.0 };
                dcg += rel / (1.0 + rank as f64).ln();
            }
            // Ideal DCG
            let ideal_rels = std::cmp::min(10, relevant_minus_self);
            for i in 0..ideal_rels {
                idcg += 1.0 / (1.0 + i as f64).ln();
            }
            if idcg > 0.0 {
                ndcg_sum += dcg / idcg;
            } else {
                ndcg_sum = dcg; // if no relevant items, DCG is correct
            }
        }

        // Embedding norm statistics
        let mut norms = Vec::with_capacity(n);
        for i in 0..n {
            let row = embeddings.row(i);
            let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
            norms.push(norm as f64);
        }
        let norm_mean = norms.iter().sum::<f64>() / norms.len() as f64;
        let norm_var: f64 = norms.iter().map(|n| (n - norm_mean).powi(2)).sum::<f64>() / norms.len() as f64;
        let norm_std = norm_var.sqrt();

        let avg_sim = if sim_count > 0 { sim_sum / sim_count as f64 } else { 0.0 };

        EmbeddingEvalReport {
            precision_at_1: if actual_queries > 0 { p1_sum / actual_queries as f64 } else { 0.0 },
            precision_at_5: if actual_queries > 0 { p5_sum / (actual_queries as f64 * 5.0).min(actual_queries as f64) } else { 0.0 },
            precision_at_10: if actual_queries > 0 { p10_sum / (actual_queries as f64 * 10.0).min(actual_queries as f64) } else { 0.0 },
            recall_at_1: if actual_queries > 0 { r1_sum / actual_queries as f64 } else { 0.0 },
            recall_at_5: if actual_queries > 0 { r5_sum / actual_queries as f64 } else { 0.0 },
            recall_at_10: if actual_queries > 0 { r10_sum / actual_queries as f64 } else { 0.0 },
            mrr: if actual_queries > 0 { mrr_sum / actual_queries as f64 } else { 0.0 },
            ndcg_at_10: if actual_queries > 0 { ndcg_sum / actual_queries as f64 } else { 0.0 },
            avg_similarity: avg_sim,
            embedding_norm_std: norm_std,
            n_queries: actual_queries,
            eval_time_ms: start.elapsed().as_secs_f64() * 1000.0,
        }
    }

    /// Compare two sets of embeddings (student vs teacher) on retrieval quality.
    pub fn compare(
        student: &Array2<f32>,
        teacher: &Array2<f32>,
        labels: &[String],
        n_queries: usize,
        k: usize,
    ) -> ComparisonReport {
        let student_report = Self::evaluate(student, labels, n_queries);
        let teacher_report = Self::evaluate(teacher, labels, n_queries);

        // Recall preservation: fraction of teacher's top-k that appear in student's top-k
        let n = student.nrows();
        let actual_queries = n_queries.min(n);
        let mut preserved_count = 0usize;
        let mut total_relevant = 0usize;

        for i in 0..actual_queries {
            let s_query = student.row(i);
            let t_query = teacher.row(i);
            let query_label = &labels[i];

            // Teacher top-k
            let mut t_sims: Vec<(usize, f64)> = (0..n).filter(|&j| j != i).map(|j| {
                let sim = cosine_sim_rows(&t_query, &teacher.row(j));
                (j, sim)
            }).collect();
            t_sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let teacher_topk: Vec<usize> = t_sims.iter().take(k).map(|(idx, _)| *idx).collect();

            // Student top-k
            let mut s_sims: Vec<(usize, f64)> = (0..n).filter(|&j| j != i).map(|j| {
                let sim = cosine_sim_rows(&s_query, &student.row(j));
                (j, sim)
            }).collect();
            s_sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let student_topk: Vec<usize> = s_sims.iter().take(k).map(|(idx, _)| *idx).collect();

            let teacher_relevant: Vec<usize> = teacher_topk.iter().filter(|&&idx| labels[idx] == *query_label).copied().collect();
            let preserved = teacher_relevant.iter().filter(|&&idx| student_topk.contains(&idx)).count();
            preserved_count += preserved;
            total_relevant += teacher_relevant.len();
        }

        ComparisonReport {
            student_precision_at_10: student_report.precision_at_10,
            teacher_precision_at_10: teacher_report.precision_at_10,
            recall_preservation: if total_relevant > 0 { preserved_count as f64 / total_relevant as f64 } else { 0.0 },
            student_mrr: student_report.mrr,
            teacher_mrr: teacher_report.mrr,
            student_avg_sim: student_report.avg_similarity,
            teacher_avg_sim: teacher_report.avg_similarity,
        }
    }

    /// Measure search latency for an embedding set.
    pub fn measure_latency(embeddings: &Array2<f32>, k: usize, n_queries: usize) -> LatencyReport {
        let n = embeddings.nrows();
        let actual_queries = n_queries.min(n);
        let mut latencies: Vec<f64> = Vec::with_capacity(actual_queries);

        let start = Instant::now();
        for i in 0..actual_queries {
            let q_start = Instant::now();
            let query = embeddings.row(i);
            let mut sims: Vec<(usize, f64)> = (0..n).filter(|&j| j != i).map(|j| {
                let sim = cosine_sim_rows(&query, &embeddings.row(j));
                (j, sim)
            }).collect();
            sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let _topk: Vec<_> = sims.iter().take(k).collect();
            latencies.push(q_start.elapsed().as_micros() as f64);
        }
        let total_ms = start.elapsed().as_secs_f64() * 1000.0;

        latencies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let p50 = percentile(&latencies, 50.0);
        let p95 = percentile(&latencies, 95.0);
        let p99 = percentile(&latencies, 99.0);
        let avg = latencies.iter().sum::<f64>() / latencies.len().max(1) as f64;

        LatencyReport {
            n_queries: actual_queries,
            total_time_ms: total_ms,
            avg_latency_us: avg,
            p50_latency_us: p50,
            p95_latency_us: p95,
            p99_latency_us: p99,
        }
    }
}

fn cosine_sim_rows(a: &ndarray::ArrayView1<f32>, b: &ndarray::ArrayView1<f32>) -> f64 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|v| v * v).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|v| v * v).sum::<f32>().sqrt();
    let denom = na * nb;
    if denom < 1e-8 { 0.0 } else { dot as f64 / denom as f64 }
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() { return 0.0; }
    let idx = (p / 100.0 * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evaluate_clusters() {
        // 10 embeddings: 5 similar, 5 different
        let mut data = Vec::new();
        let mut labels = Vec::new();
        for _i in 0..5 {
            data.extend([1.0, 0.0, 0.0]); // cluster A
            labels.push("A".to_string());
        }
        for _i in 0..5 {
            data.extend([0.0, 1.0, 0.0]); // cluster B
            labels.push("B".to_string());
        }
        let emb = Array2::from_shape_vec((10, 3), data).unwrap();

        let report = EmbeddingEvaluator::evaluate(&emb, &labels, 5);
        assert!(report.precision_at_1 > 0.0);
        assert!(report.avg_similarity >= 0.0);
        assert_eq!(report.n_queries, 5);
    }

    #[test]
    fn test_norm_std() {
        let data = vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let emb = Array2::from_shape_vec((2, 3), data).unwrap();
        let labels = vec!["A".into(), "B".into()];
        let report = EmbeddingEvaluator::evaluate(&emb, &labels, 2);
        assert!(report.embedding_norm_std < 0.1);
        assert!(report.eval_time_ms >= 0.0);
    }

    #[test]
    fn test_compare() {
        let mut data = Vec::new();
        let mut labels = Vec::new();
        for _ in 0..5 { data.extend([1.0f32, 0.0, 0.0]); labels.push("A".into()); }
        for _ in 0..5 { data.extend([0.0f32, 1.0, 0.0]); labels.push("B".into()); }
        let emb = Array2::from_shape_vec((10, 3), data).unwrap();
        let report = EmbeddingEvaluator::compare(&emb, &emb, &labels, 5, 5);
        assert!(report.recall_preservation > 0.9, "Same embeddings should have ~1.0 preservation");
    }

    #[test]
    fn test_latency() {
        let mut data = Vec::new();
        for i in 0..20 { data.extend([i as f32, 0.0, 0.0]); }
        let emb = Array2::from_shape_vec((20, 3), data).unwrap();
        let report = EmbeddingEvaluator::measure_latency(&emb, 5, 5);
        assert_eq!(report.n_queries, 5);
        assert!(report.total_time_ms >= 0.0);
        assert!(report.p50_latency_us > 0.0);
    }
}
