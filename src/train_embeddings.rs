//! Embedding training infrastructure for SplatDB.
//! Knowledge distillation: Teacher -> Student embeddings.
//! Ported from splatdb Python.
//!
//! Note: Actual training requires a deep learning framework (candle, burn, tch).
//! This module provides the training pipeline, dataset generation, and evaluation.

use serde::{Deserialize, Serialize};

/// Training configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub train_size: usize,
    pub val_size: usize,
    pub embedding_dim: usize,
    pub matryoshka_dims: Vec<usize>,
    pub warmup_steps: usize,
    pub weight_decay: f64,
    pub seed: u64,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 10,
            batch_size: 32,
            learning_rate: 2e-5,
            train_size: 10_000,
            val_size: 1_000,
            embedding_dim: 640,
            matryoshka_dims: vec![64, 128, 256, 384, 512, 640],
            warmup_steps: 500,
            weight_decay: 0.01,
            seed: 42,
        }
    }
}

/// A training sample: text pair with teacher embeddings.
#[derive(Debug, Clone)]
pub struct TrainingSample {
    pub text_a: String,
    pub text_b: String,
    pub label: f32, // 1.0 = similar, 0.0 = dissimilar
    pub teacher_emb_a: Option<Vec<f32>>,
    pub teacher_emb_b: Option<Vec<f32>>,
}

/// Training epoch statistics.
#[derive(Debug, Clone, Default)]
pub struct EpochStats {
    pub epoch: usize,
    pub train_loss: f64,
    pub val_loss: f64,
    pub train_time_secs: f64,
    pub learning_rate: f64,
    pub samples_per_sec: f64,
}

/// Model evaluation metrics.
#[derive(Debug, Clone, Default)]
pub struct EvalMetrics {
    pub cosine_sim_mean: f64,
    pub cosine_sim_std: f64,
    pub recall_at_1: f64,
    pub recall_at_5: f64,
    pub recall_at_10: f64,
    pub ndcg_at_10: f64,
    pub mrr: f64,
}

/// Training checkpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingCheckpoint {
    pub epoch: usize,
    pub train_loss: f64,
    pub val_loss: f64,
    pub best_val_loss: f64,
    pub config: TrainingConfig,
}

/// Learning rate scheduler with warmup.
pub struct LRScheduler {
    base_lr: f64,
    warmup_steps: usize,
    current_step: usize,
    min_lr: f64,
}

impl LRScheduler {
    /// New.
    pub fn new(base_lr: f64, warmup_steps: usize) -> Self {
        Self { base_lr, warmup_steps, current_step: 0, min_lr: 1e-7 }
    }

    /// Linear warmup + cosine decay.
    pub fn step(&mut self) -> f64 {
        let lr = if self.current_step < self.warmup_steps {
            self.base_lr * (self.current_step + 1) as f64 / self.warmup_steps as f64
        } else {
            let progress = (self.current_step - self.warmup_steps) as f64 / 10_000.0;
            self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1.0 + (progress * std::f64::consts::PI).cos())
        };
        self.current_step += 1;
        lr
    }

    /// Current lr.
    pub fn current_lr(&self) -> f64 { self.base_lr }
    /// Reset.
    pub fn reset(&mut self) { self.current_step = 0; }
}

/// Training state tracker.
pub struct TrainingState {
    pub config: TrainingConfig,
    pub current_epoch: usize,
    pub best_val_loss: f64,
    pub history: Vec<EpochStats>,
    pub checkpoint: Option<TrainingCheckpoint>,
}

impl TrainingState {
    /// New.
    pub fn new(config: TrainingConfig) -> Self {
        Self { config, current_epoch: 0, best_val_loss: f64::MAX, history: Vec::new(), checkpoint: None }
    }

    /// Record epoch.
    pub fn record_epoch(&mut self, stats: EpochStats) {
        if stats.val_loss < self.best_val_loss {
            self.best_val_loss = stats.val_loss;
            self.checkpoint = Some(TrainingCheckpoint {
                epoch: stats.epoch,
                train_loss: stats.train_loss,
                val_loss: stats.val_loss,
                best_val_loss: self.best_val_loss,
                config: self.config.clone(),
            });
        }
        self.current_epoch = stats.epoch;
        self.history.push(stats);
    }

    /// Should stop early.
    pub fn should_stop_early(&self, patience: usize) -> bool {
        if self.history.len() < patience + 1 {
            return false;
        }
        let recent: Vec<f64> = self.history.iter().rev().take(patience + 1).map(|s| s.val_loss).collect();
        recent.windows(2).all(|w| w[0] <= w[1])
    }
}

/// Synthetic training data generator.
pub struct SyntheticDataset {
    templates_similar: Vec<String>,
    templates_dissimilar: Vec<String>,
    size: usize,
}

impl SyntheticDataset {
    /// New.
    pub fn new(size: usize) -> Self {
        let templates_similar: Vec<String> = vec![
            "Machine learning algorithms learn patterns from data".into(),
            "Neural networks are inspired by biological neurons".into(),
            "Vector databases store high-dimensional embeddings".into(),
            "Semantic search uses embedding similarity".into(),
            "Natural language processing analyzes text data".into(),
            "Knowledge distillation transfers teacher knowledge to student".into(),
            "Embedding models map text to vector space".into(),
            "Retrieval augmented generation combines search with generation".into(),
            "Transformer architecture revolutionized NLP".into(),
            "Attention mechanism weighs input importance".into(),
        ];
        let templates_dissimilar: Vec<String> = vec![
            "The recipe requires fresh basil and tomatoes".into(),
            "Soccer is the most popular sport in Brazil".into(),
            "The stock market closed higher yesterday".into(),
            "Jazz music originated in New Orleans".into(),
            "Photosynthesis converts sunlight to chemical energy".into(),
            "The museum exhibition opens next Tuesday".into(),
            "Ocean currents affect global climate patterns".into(),
            "Traditional ceramics require kiln firing above 1000 degrees".into(),
        ];
        Self { templates_similar, templates_dissimilar, size }
    }

    /// Generate training samples.
    pub fn generate(&self) -> Vec<TrainingSample> {
        let mut samples = Vec::with_capacity(self.size);
        for i in 0..self.size {
            if i % 2 == 0 {
                // Similar pair
                let idx = i % self.templates_similar.len();
                samples.push(TrainingSample {
                    text_a: self.templates_similar[idx].to_string(),
                    text_b: format!("{} is a key concept in AI", &self.templates_similar[idx][..20.min(self.templates_similar[idx].len())]),
                    label: 1.0,
                    teacher_emb_a: None,
                    teacher_emb_b: None,
                });
            } else {
                // Dissimilar pair
                let idx_a = i % self.templates_similar.len();
                let idx_b = i % self.templates_dissimilar.len();
                samples.push(TrainingSample {
                    text_a: self.templates_similar[idx_a].to_string(),
                    text_b: self.templates_dissimilar[idx_b].to_string(),
                    label: 0.0,
                    teacher_emb_a: None,
                    teacher_emb_b: None,
                });
            }
        }
        samples
    }
}

/// Evaluate embeddings using standard retrieval metrics.
pub fn evaluate_embeddings(
    query_embs: &[Vec<f32>],
    corpus_embs: &[Vec<f32>],
    relevant_ids: &[Vec<usize>],
) -> EvalMetrics {
    if query_embs.is_empty() || corpus_embs.is_empty() {
        return EvalMetrics::default();
    }

    let mut cosine_sims = Vec::new();
    let mut recall_1 = 0.0;
    let mut recall_5 = 0.0;
    let mut recall_10 = 0.0;
    let mut rr_sum = 0.0;
    let mut ndcg_sum = 0.0;
    let n_queries = query_embs.len().min(relevant_ids.len());

    for qi in 0..n_queries {
        let query = &query_embs[qi];
        let relevant = &relevant_ids[qi];

        // Compute similarities
        let scores: Vec<(usize, f64)> = corpus_embs.iter().enumerate()
            .map(|(ci, corpus)| {
                let sim = cosine_similarity(query, corpus);
                cosine_sims.push(sim);
                (ci, sim)
            })
            .collect();

        let mut sorted = scores.clone();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Recall@k
        let top_ids: Vec<usize> = sorted.iter().take(10).map(|(id, _)| *id).collect();
        if !top_ids.is_empty() && relevant.contains(&top_ids[0]) { recall_1 += 1.0; }
        if top_ids.len() > 4 && top_ids[..5].iter().any(|id| relevant.contains(id)) { recall_5 += 1.0; }
        if top_ids.iter().any(|id| relevant.contains(id)) { recall_10 += 1.0; }

        // MRR
        for (rank, (id, _)) in sorted.iter().enumerate() {
            if relevant.contains(id) {
                rr_sum += 1.0 / (rank + 1) as f64;
                break;
            }
        }

        // NDCG@10
        let dcg: f64 = sorted.iter().take(10).enumerate()
            .map(|(rank, (id, _))| {
                if relevant.contains(id) { 1.0 / (rank as f64 + 2.0).log2() } else { 0.0 }
            })
            .sum();
        let ideal_relevant = relevant.len().min(10);
        let idcg: f64 = (0..ideal_relevant).map(|r| 1.0 / (r as f64 + 2.0).log2()).sum();
        if idcg > 0.0 { ndcg_sum += dcg / idcg; }
    }

    let n = n_queries as f64;
    let mean_sim = cosine_sims.iter().sum::<f64>() / cosine_sims.len() as f64;
    let variance = cosine_sims.iter().map(|&s| (s - mean_sim).powi(2)).sum::<f64>() / cosine_sims.len() as f64;

    EvalMetrics {
        cosine_sim_mean: mean_sim,
        cosine_sim_std: variance.sqrt(),
        recall_at_1: recall_1 / n,
        recall_at_5: recall_5 / n,
        recall_at_10: recall_10 / n,
        ndcg_at_10: ndcg_sum / n,
        mrr: rr_sum / n,
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(&x, &y)| x as f64 * y as f64).sum();
    let norm_a: f64 = a.iter().map(|&v| v as f64 * v as f64).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|&v| v as f64 * v as f64).sum::<f64>().sqrt();
    let denom = norm_a * norm_b;
    if denom < 1e-8 { 0.0 } else { dot / denom }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lr_scheduler_warmup() {
        let mut scheduler = LRScheduler::new(0.01, 100);
        let lr1 = scheduler.step(); // step 0
        assert!(lr1 < 0.01);
        let lr100 = (0..99).map(|_| scheduler.step()).last().unwrap();
        assert!((lr100 - 0.01).abs() < 1e-6);
    }

    #[test]
    fn test_synthetic_dataset() {
        let ds = SyntheticDataset::new(100);
        let samples = ds.generate();
        assert_eq!(samples.len(), 100);
        assert_eq!(samples[0].label, 1.0);
        assert_eq!(samples[1].label, 0.0);
    }

    #[test]
    fn test_training_state() {
        let mut state = TrainingState::new(TrainingConfig::default());
        state.record_epoch(EpochStats { epoch: 1, train_loss: 0.5, val_loss: 0.3, ..Default::default() });
        assert_eq!(state.best_val_loss, 0.3);
        assert!(!state.should_stop_early(3));
    }

    #[test]
    fn test_evaluate_embeddings() {
        let query = vec![vec![1.0, 0.0, 0.0]];
        let corpus = vec![vec![0.9, 0.1, 0.0], vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 1.0]];
        let relevant = vec![vec![0]]; // first corpus item is relevant
        let metrics = evaluate_embeddings(&query, &corpus, &relevant);
        assert!(metrics.recall_at_1 > 0.9);
        assert!(metrics.recall_at_10 > 0.9);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let c = vec![0.0, 1.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);
        assert!(cosine_similarity(&a, &c).abs() < 1e-6);
    }
}
