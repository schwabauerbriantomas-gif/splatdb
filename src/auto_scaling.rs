//! Auto-Scaling for SplatDB Cluster.
//! Automatic scaling based on metrics, trends, and predictions.
//! Ported from splatdb Python.

use std::collections::HashMap;

/// Scaling direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalingDirection {
    ScaleUp,
    ScaleDown,
    None,
}

/// Scaling trigger type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalingTrigger {
    CpuThreshold,
    MemoryThreshold,
    LatencyThreshold,
    QpsThreshold,
    Predictive,
    Manual,
}

/// Metrics from a single node.
#[derive(Debug, Clone)]
pub struct NodeMetrics {
    pub node_id: String,
    pub cpu_percent: f64,
    pub memory_percent: f64,
    pub qps: f64,
    pub latency_ms: f64,
    pub active_queries: usize,
    pub uptime_seconds: f64,
    pub timestamp: f64,
}

impl NodeMetrics {
    /// New.
    pub fn new(node_id: &str) -> Self {
        Self {
            node_id: node_id.to_string(),
            cpu_percent: 0.0,
            memory_percent: 0.0,
            qps: 0.0,
            latency_ms: 0.0,
            active_queries: 0,
            uptime_seconds: 0.0,
            timestamp: now_secs(),
        }
    }
}

/// A scaling decision.
#[derive(Debug, Clone)]
pub struct ScalingDecision {
    pub action: ScalingDirection,
    pub trigger: ScalingTrigger,
    pub current_nodes: usize,
    pub target_nodes: usize,
    pub reason: String,
    pub metrics: HashMap<String, f64>,
}

/// Aggregated cluster statistics.
#[derive(Debug, Clone, Default)]
pub struct ClusterStats {
    pub nodes: usize,
    pub avg_cpu: f64,
    pub avg_memory: f64,
    pub total_qps: f64,
    pub avg_latency_ms: f64,
    pub total_active_queries: usize,
}

/// Metrics collector: aggregates per-node metrics and computes cluster stats.
pub struct MetricsCollector {
    history_size: usize,
    metrics_history: HashMap<String, Vec<NodeMetrics>>,
}

impl MetricsCollector {
    /// New.
    pub fn new(history_size: usize) -> Self {
        Self {
            history_size,
            metrics_history: HashMap::new(),
        }
    }

    /// Record metrics from a node.
    pub fn record(&mut self, metrics: NodeMetrics) {
        let node_id = metrics.node_id.clone();
        let history = self.metrics_history.entry(node_id).or_default();
        if history.len() >= self.history_size {
            history.remove(0);
        }
        history.push(metrics);
    }

    /// Get aggregated cluster statistics.
    pub fn get_cluster_stats(&self) -> ClusterStats {
        if self.metrics_history.is_empty() {
            return ClusterStats::default();
        }

        let mut cpus = Vec::new();
        let mut memories = Vec::new();
        let mut total_qps = 0.0;
        let mut latencies = Vec::new();
        let mut total_queries = 0usize;

        for history in self.metrics_history.values() {
            if let Some(latest) = history.last() {
                cpus.push(latest.cpu_percent);
                memories.push(latest.memory_percent);
                total_qps += latest.qps;
                latencies.push(latest.latency_ms);
                total_queries += latest.active_queries;
            }
        }

        let n = cpus.len() as f64;
        ClusterStats {
            nodes: self.metrics_history.len(),
            avg_cpu: cpus.iter().sum::<f64>() / n.max(1.0),
            avg_memory: memories.iter().sum::<f64>() / n.max(1.0),
            total_qps,
            avg_latency_ms: latencies.iter().sum::<f64>() / n.max(1.0),
            total_active_queries: total_queries,
        }
    }

    /// Detect metric trend: 'increasing', 'decreasing', or 'stable'.
    pub fn get_trend(&self, metric_name: &str, window: usize) -> Trend {
        let mut all_values: Vec<f64> = Vec::new();
        for history in self.metrics_history.values() {
            // Take last `window` items, maintaining chronological order
            let start = history.len().saturating_sub(window);
            for m in &history[start..] {
                let val = match metric_name {
                    "cpu" => Some(m.cpu_percent),
                    "memory" => Some(m.memory_percent),
                    "qps" => Some(m.qps),
                    "latency" => Some(m.latency_ms),
                    _ => None,
                };
                if let Some(v) = val {
                    all_values.push(v);
                }
            }
        }

        if all_values.len() < 3 {
            return Trend::Stable;
        }

        // Sort by time (already sorted since we recorded chronologically)
        let mid = all_values.len() / 2;
        let first_half: f64 = all_values[..mid].iter().sum::<f64>() / mid.max(1) as f64;
        let second_half: f64 =
            all_values[mid..].iter().sum::<f64>() / (all_values.len() - mid).max(1) as f64;
        let diff = second_half - first_half;
        let threshold = first_half.abs() * 0.1 + 0.01; // 10% change threshold, with minimum

        if diff > threshold {
            Trend::Increasing
        } else if diff < -threshold {
            Trend::Decreasing
        } else {
            Trend::Stable
        }
    }

    /// Get recent metrics for a specific node.
    pub fn get_node_history(&self, node_id: &str) -> &[NodeMetrics] {
        self.metrics_history
            .get(node_id)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }
}

/// Metric trend direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Trend {
    Increasing,
    Decreasing,
    Stable,
}

/// Auto-scaler for SplatDB Cluster.
///
/// Monitors metrics and makes scaling decisions automatically.
pub struct AutoScaler {
    min_nodes: usize,
    max_nodes: usize,
    scale_up_threshold: f64,
    scale_down_threshold: f64,
    cooldown_seconds: f64,
    enable_predictive: bool,
    metrics_collector: MetricsCollector,
    current_nodes: usize,
    last_scale_time: f64,
    scaling_history: Vec<ScalingDecision>,
}

impl AutoScaler {
    /// New.
    pub fn new(
        min_nodes: usize,
        max_nodes: usize,
        scale_up_threshold: f64,
        scale_down_threshold: f64,
    ) -> Self {
        Self {
            min_nodes,
            max_nodes,
            scale_up_threshold,
            scale_down_threshold,
            cooldown_seconds: 60.0,
            enable_predictive: false,
            metrics_collector: MetricsCollector::new(100),
            current_nodes: min_nodes,
            last_scale_time: 0.0,
            scaling_history: Vec::new(),
        }
    }

    /// Record metrics from a node.
    pub fn record_metrics(&mut self, metrics: NodeMetrics) {
        self.metrics_collector.record(metrics);
    }

    /// Evaluate scaling: check metrics and decide whether to scale.
    pub fn evaluate(&mut self) -> Option<ScalingDecision> {
        let stats = self.metrics_collector.get_cluster_stats();
        let now = now_secs();

        // Cooldown check
        if (now - self.last_scale_time) < self.cooldown_seconds {
            return None;
        }

        if stats.avg_cpu > self.scale_up_threshold && self.current_nodes < self.max_nodes {
            let decision = ScalingDecision {
                action: ScalingDirection::ScaleUp,
                trigger: ScalingTrigger::CpuThreshold,
                current_nodes: self.current_nodes,
                target_nodes: self.current_nodes + 1,
                reason: format!(
                    "CPU {}% > threshold {}%",
                    stats.avg_cpu as i32, self.scale_up_threshold as i32
                ),
                metrics: HashMap::new(),
            };
            self.apply_scaling(&decision);
            Some(decision)
        } else if stats.avg_cpu < self.scale_down_threshold && self.current_nodes > self.min_nodes {
            let decision = ScalingDecision {
                action: ScalingDirection::ScaleDown,
                trigger: ScalingTrigger::CpuThreshold,
                current_nodes: self.current_nodes,
                target_nodes: self.current_nodes - 1,
                reason: format!(
                    "CPU {}% < threshold {}%",
                    stats.avg_cpu as i32, self.scale_down_threshold as i32
                ),
                metrics: HashMap::new(),
            };
            self.apply_scaling(&decision);
            Some(decision)
        } else {
            // Predictive scaling
            if self.enable_predictive {
                let trend = self.metrics_collector.get_trend("cpu", 10);
                if trend == Trend::Increasing && self.current_nodes < self.max_nodes {
                    let decision = ScalingDecision {
                        action: ScalingDirection::ScaleUp,
                        trigger: ScalingTrigger::Predictive,
                        current_nodes: self.current_nodes,
                        target_nodes: self.current_nodes + 1,
                        reason: "CPU trend increasing (predictive)".into(),
                        metrics: HashMap::new(),
                    };
                    self.apply_scaling(&decision);
                    return Some(decision);
                }
            }
            None
        }
    }

    /// Get current node count.
    pub fn current_nodes(&self) -> usize {
        self.current_nodes
    }

    /// Get cluster stats.
    pub fn cluster_stats(&self) -> ClusterStats {
        self.metrics_collector.get_cluster_stats()
    }

    /// Get scaling history.
    pub fn scaling_history(&self) -> &[ScalingDecision] {
        &self.scaling_history
    }

    fn apply_scaling(&mut self, decision: &ScalingDecision) {
        self.current_nodes = decision.target_nodes;
        self.last_scale_time = now_secs();
        self.scaling_history.push(decision.clone());
    }
}

fn now_secs() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_decision_few_samples() {
        let mut scaler = AutoScaler::new(1, 10, 80.0, 30.0);
        // With cooldown active, should not scale even with high CPU
        let m = NodeMetrics {
            cpu_percent: 95.0,
            ..NodeMetrics::new("n1")
        };
        scaler.record_metrics(m);
        // First evaluate sets last_scale_time to now, so subsequent calls within cooldown return None
        let _ = scaler.evaluate();
        // Now within cooldown period
        assert!(scaler.evaluate().is_none());
    }

    #[test]
    fn test_scale_up_decision() {
        let mut scaler = AutoScaler::new(1, 10, 80.0, 30.0);
        scaler.last_scale_time = 0.0; // Clear cooldown
        for i in 0..5 {
            let m = NodeMetrics {
                cpu_percent: 95.0,
                ..NodeMetrics::new(&format!("n{}", i % 2))
            };
            scaler.record_metrics(m);
        }
        let decision = scaler.evaluate();
        assert!(decision.is_some());
        assert_eq!(decision.unwrap().action, ScalingDirection::ScaleUp);
    }

    #[test]
    fn test_scale_down_decision() {
        let mut scaler = AutoScaler::new(1, 10, 80.0, 30.0);
        scaler.current_nodes = 5;
        scaler.last_scale_time = 0.0; // Clear cooldown
        for i in 0..5 {
            let m = NodeMetrics {
                cpu_percent: 10.0,
                ..NodeMetrics::new(&format!("n{}", i))
            };
            scaler.record_metrics(m);
        }
        let decision = scaler.evaluate();
        assert!(decision.is_some());
        assert_eq!(decision.unwrap().action, ScalingDirection::ScaleDown);
    }

    #[test]
    fn test_trend_detection() {
        let mut collector = MetricsCollector::new(100);
        // Increasing CPU: 10, 20, 30, 40, 50, 60, 70, 80, 90, 100
        for i in 0..10 {
            let m = NodeMetrics {
                cpu_percent: 10.0 + i as f64 * 10.0,
                ..NodeMetrics::new("n1")
            };
            collector.record(m);
        }
        // With window=10, first half avg = 30, second half avg = 80
        // diff = 50, threshold = 30*0.1 = 3, diff > threshold => Increasing
        assert_eq!(collector.get_trend("cpu", 10), Trend::Increasing);
    }
}
