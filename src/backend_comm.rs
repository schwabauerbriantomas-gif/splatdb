//! Backend Communication Protocol — structured messaging, health checks,
//! performance metrics, and dead letter queue.
//! Ported from splatdb Python.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

/// Message types between backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BackendMsgType {
    SearchRequest,
    SearchResult,
    IndexRequest,
    IndexResult,
    HealthCheck,
    HealthResponse,
    MetricsReport,
    Error,
    Shutdown,
}

/// Priority levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Priority {
    Low = -1,
    Normal = 0,
    High = 1,
    Critical = 2,
}

/// Structured message between SplatDB backends.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendMessage {
    pub sender: String,
    pub receiver: String,
    pub msg_type: BackendMsgType,
    pub content: serde_json::Value,
    pub message_id: String,
    pub timestamp: f64,
    pub priority: Priority,
    pub ttl_seconds: f64,
    pub metadata: HashMap<String, serde_json::Value>,
    pub parent_id: Option<String>,
}

impl BackendMessage {
    /// Create a new message.
    pub fn new(
        sender: &str,
        receiver: &str,
        msg_type: BackendMsgType,
        content: serde_json::Value,
    ) -> Self {
        Self {
            sender: sender.to_string(),
            receiver: receiver.to_string(),
            msg_type,
            content,
            message_id: generate_id(),
            timestamp: now_secs(),
            priority: Priority::Normal,
            ttl_seconds: 60.0,
            metadata: HashMap::new(),
            parent_id: None,
        }
    }

    /// Set priority.
    pub fn with_priority(mut self, priority: Priority) -> Self {
        self.priority = priority;
        self
    }

    /// Set parent message ID for chaining.
    pub fn with_parent(mut self, parent_id: &str) -> Self {
        self.parent_id = Some(parent_id.to_string());
        self
    }

    /// Check if message has expired.
    pub fn is_expired(&self) -> bool {
        if self.ttl_seconds <= 0.0 {
            return false;
        }
        (now_secs() - self.timestamp) > self.ttl_seconds
    }

    /// Serialize to JSON.
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_default()
    }

    /// Deserialize from JSON.
    pub fn from_json(json: &str) -> Option<Self> {
        serde_json::from_str(json).ok()
    }
}

/// Backend health status.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendHealth {
    pub name: String,
    pub is_healthy: bool,
    pub last_heartbeat: f64,
    pub total_queries: usize,
    pub total_errors: usize,
    pub avg_latency_ms: f64,
    pub memory_usage_mb: f64,
    pub index_size: usize,
    pub error_message: String,
}

impl BackendHealth {
    /// New.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            is_healthy: true,
            last_heartbeat: now_secs(),
            total_queries: 0,
            total_errors: 0,
            avg_latency_ms: 0.0,
            memory_usage_mb: 0.0,
            index_size: 0,
            error_message: String::new(),
        }
    }
}

/// Performance metrics for a backend.
#[derive(Debug, Clone, Default)]
pub struct BackendMetrics {
    pub name: String,
    pub queries_per_second: f64,
    pub avg_latency_ms: f64,
    pub p50_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub p99_latency_ms: f64,
    pub error_rate: f64,
    pub total_queries: usize,
    pub total_errors: usize,
    pub uptime_seconds: f64,
}

/// Communication protocol between SplatDB backends.
///
/// Provides:
/// - Priority message bus
/// - Automatic health checks
/// - Performance metrics per backend
/// - Dead letter queue for failed messages
/// - Request/response tracking
pub struct BackendComm {
    max_history: usize,
    dead_letter_max: usize,
    _health_check_interval: f64,
    messages: VecDeque<BackendMessage>,
    dead_letter: VecDeque<BackendMessage>,
    backends: HashMap<String, f64>, // name -> registered_at
    health_status: HashMap<String, BackendHealth>,
    latency_samples: HashMap<String, VecDeque<f64>>,
    _last_health_check: f64,
    pending_requests: HashMap<String, BackendMessage>,
}

impl BackendComm {
    /// New.
    pub fn new(max_history: usize, dead_letter_max: usize, health_check_interval: f64) -> Self {
        Self {
            max_history,
            dead_letter_max,
            _health_check_interval: health_check_interval,
            messages: VecDeque::with_capacity(max_history),
            dead_letter: VecDeque::with_capacity(dead_letter_max),
            backends: HashMap::new(),
            health_status: HashMap::new(),
            latency_samples: HashMap::new(),
            _last_health_check: 0.0,
            pending_requests: HashMap::new(),
        }
    }

    /// Register a backend.
    pub fn register_backend(&mut self, name: &str) {
        self.backends.insert(name.to_string(), now_secs());
        self.latency_samples
            .insert(name.to_string(), VecDeque::with_capacity(1000));
        self.health_status
            .insert(name.to_string(), BackendHealth::new(name));
    }

    /// Unregister a backend.
    pub fn unregister_backend(&mut self, name: &str) {
        self.backends.remove(name);
        self.health_status.remove(name);
        self.latency_samples.remove(name);
    }

    /// Send a message to the bus.
    pub fn send(&mut self, msg: BackendMessage) -> String {
        let id = msg.message_id.clone();
        if self.messages.len() >= self.max_history {
            self.messages.pop_front();
        }
        self.messages.push_back(msg);
        id
    }

    /// Convenience: send a search request.
    pub fn send_search_request(
        &mut self,
        sender: &str,
        receiver: &str,
        query_id: &str,
        k: usize,
    ) -> String {
        let content = serde_json::json!({"query_id": query_id, "k": k});
        let msg = BackendMessage::new(sender, receiver, BackendMsgType::SearchRequest, content)
            .with_priority(Priority::High);
        self.send(msg)
    }

    #[allow(clippy::too_many_arguments)]
    /// Convenience: send a search result.
    pub fn send_search_result(
        &mut self,
        sender: &str,
        request_id: &str,
        query_id: &str,
        result_count: usize,
        latency_ms: f64,
        success: bool,
        error: &str,
    ) -> String {
        self.pending_requests.remove(request_id);
        let content = serde_json::json!({
            "query_id": query_id, "result_count": result_count,
            "latency_ms": latency_ms, "success": success, "error": error,
        });
        let msg = BackendMessage::new(sender, "supervisor", BackendMsgType::SearchResult, content)
            .with_parent(request_id);
        self.send(msg)
    }

    /// Receive messages for a receiver.
    pub fn receive(
        &self,
        receiver: &str,
        msg_type: Option<BackendMsgType>,
        min_priority: Option<Priority>,
    ) -> Vec<&BackendMessage> {
        let min_p = min_priority.unwrap_or(Priority::Low);
        self.messages
            .iter()
            .filter(|m| {
                (m.receiver == receiver || m.receiver == "all")
                    && msg_type.is_none_or(|t| m.msg_type == t)
                    && m.priority >= min_p
                    && !m.is_expired()
            })
            .collect()
    }

    /// Report an error.
    pub fn report_error(
        &mut self,
        sender: &str,
        error: &str,
        query_id: &str,
        severity: Priority,
    ) -> String {
        let content = serde_json::json!({"error": error, "query_id": query_id});
        let msg = BackendMessage::new(sender, "supervisor", BackendMsgType::Error, content)
            .with_priority(severity);
        self.send(msg)
    }

    /// Record latency sample for a backend.
    pub fn record_latency(&mut self, backend: &str, latency_ms: f64) {
        if let Some(samples) = self.latency_samples.get_mut(backend) {
            if samples.len() >= 1000 {
                samples.pop_front();
            }
            samples.push_back(latency_ms);
        }
    }

    /// Get health of a backend.
    pub fn get_health(&self, backend: &str) -> Option<&BackendHealth> {
        self.health_status.get(backend)
    }

    /// Update health of a backend.
    pub fn update_health(&mut self, backend: &str, health: BackendHealth) {
        self.health_status.insert(backend.to_string(), health);
    }

    /// Get health of all backends.
    pub fn get_all_health(&self) -> HashMap<String, serde_json::Value> {
        self.backends
            .keys()
            .filter_map(|name| {
                self.health_status.get(name).map(|h| {
                    (
                        name.clone(),
                        serde_json::json!({
                            "healthy": h.is_healthy,
                            "total_queries": h.total_queries,
                            "total_errors": h.total_errors,
                            "avg_latency_ms": h.avg_latency_ms,
                        }),
                    )
                })
            })
            .collect()
    }

    /// Calculate metrics for a backend.
    pub fn get_metrics(&self, backend: &str) -> BackendMetrics {
        let mut metrics = BackendMetrics {
            name: backend.to_string(),
            ..Default::default()
        };
        let health = self.health_status.get(backend);
        metrics.total_queries = health.map(|h| h.total_queries).unwrap_or(0);
        metrics.total_errors = health.map(|h| h.total_errors).unwrap_or(0);
        metrics.error_rate = if metrics.total_queries > 0 {
            metrics.total_errors as f64 / metrics.total_queries as f64
        } else {
            0.0
        };

        if let Some(samples) = self.latency_samples.get(backend) {
            let mut sorted: Vec<f64> = samples.iter().cloned().collect();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let n = sorted.len();
            if n > 0 {
                metrics.avg_latency_ms = sorted.iter().sum::<f64>() / n as f64;
                metrics.p50_latency_ms = sorted[(n as f64 * 0.50) as usize];
                metrics.p95_latency_ms = if n > 1 {
                    sorted[(n as f64 * 0.95) as usize]
                } else {
                    sorted[0]
                };
                metrics.p99_latency_ms = if n > 10 {
                    sorted[(n as f64 * 0.99) as usize]
                } else {
                    sorted[n - 1]
                };
                if metrics.avg_latency_ms > 0.0 {
                    metrics.queries_per_second = 1000.0 / metrics.avg_latency_ms;
                }
            }
        }
        metrics
    }

    /// Get dead letter queue contents.
    pub fn dead_letter_count(&self) -> usize {
        self.dead_letter.len()
    }

    /// Move expired/unprocessable messages to dead letter queue.
    pub fn flush_expired_to_dlq(&mut self) -> usize {
        let before = self.messages.len();
        self.messages.retain(|m| {
            if m.is_expired() && self.dead_letter.len() < self.dead_letter_max {
                self.dead_letter.push_back(m.clone());
                false
            } else {
                true
            }
        });
        before - self.messages.len()
    }

    /// Communication stats.
    pub fn stats(&self) -> CommStats {
        let mut type_counts: HashMap<String, usize> = HashMap::new();
        for m in &self.messages {
            *type_counts.entry(format!("{:?}", m.msg_type)).or_insert(0) += 1;
        }
        CommStats {
            total_messages: self.messages.len(),
            dead_letter_count: self.dead_letter.len(),
            registered_backends: self.backends.len(),
            pending_requests: self.pending_requests.len(),
            by_type: type_counts,
        }
    }

    /// Clear all state.
    pub fn clear(&mut self) {
        self.messages.clear();
        self.dead_letter.clear();
        self.pending_requests.clear();
    }
}

/// Communication statistics.
#[derive(Debug)]
pub struct CommStats {
    pub total_messages: usize,
    pub dead_letter_count: usize,
    pub registered_backends: usize,
    pub pending_requests: usize,
    pub by_type: HashMap<String, usize>,
}

fn now_secs() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

fn generate_id() -> String {
    use std::hash::{Hash, Hasher};
    use std::time::SystemTime;
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    SystemTime::now().hash(&mut hasher);
    format!("{:012x}", hasher.finish())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_send_receive() {
        let mut comm = BackendComm::new(1000, 100, 30.0);
        comm.register_backend("cpu");
        comm.send(BackendMessage::new(
            "test",
            "cpu",
            BackendMsgType::SearchRequest,
            serde_json::json!({"q": 1}),
        ));
        let msgs = comm.receive("cpu", None, None);
        assert_eq!(msgs.len(), 1);
    }

    #[test]
    fn test_search_request_response() {
        let mut comm = BackendComm::new(1000, 100, 30.0);
        let req_id = comm.send_search_request("supervisor", "cpu", "q1", 10);
        let res_id = comm.send_search_result("cpu", &req_id, "q1", 5, 2.5, true, "");
        assert!(!res_id.is_empty());
    }

    #[test]
    fn test_priority_filtering() {
        let mut comm = BackendComm::new(1000, 100, 30.0);
        comm.send(
            BackendMessage::new("a", "b", BackendMsgType::Error, serde_json::json!({}))
                .with_priority(Priority::Low),
        );
        comm.send(
            BackendMessage::new(
                "a",
                "b",
                BackendMsgType::SearchRequest,
                serde_json::json!({}),
            )
            .with_priority(Priority::High),
        );
        let high_only = comm.receive("b", None, Some(Priority::High));
        assert_eq!(high_only.len(), 1);
    }

    #[test]
    fn test_metrics() {
        let mut comm = BackendComm::new(1000, 100, 30.0);
        comm.register_backend("cpu");
        comm.record_latency("cpu", 5.0);
        comm.record_latency("cpu", 10.0);
        comm.record_latency("cpu", 15.0);
        let m = comm.get_metrics("cpu");
        assert!((m.avg_latency_ms - 10.0).abs() < 0.1);
    }

    #[test]
    fn test_json_roundtrip() {
        let msg = BackendMessage::new(
            "a",
            "b",
            BackendMsgType::HealthCheck,
            serde_json::json!({"node": "n1"}),
        );
        let json = msg.to_json();
        let parsed = BackendMessage::from_json(&json).unwrap();
        assert_eq!(parsed.sender, "a");
        assert_eq!(parsed.msg_type, BackendMsgType::HealthCheck);
    }
}
