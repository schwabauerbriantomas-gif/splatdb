//! Sync queue for offline actions and coordinator reconnection.
//! Ported from m2m-vector-search Python.

use std::time::Instant;

/// A pending action to be dispatched when connectivity restores.
#[derive(Debug, Clone)]
pub struct SyncAction {
    pub action_type: String,
    pub payload: String, // JSON payload
    pub queued_at: f64,
    pub retries: usize,
}

/// Queue for offline actions when coordinator is unreachable.
pub struct SyncQueue {
    queue: Vec<SyncAction>,
    flush_interval_secs: f64,
    last_flush: Option<Instant>,
    max_retries: usize,
}

impl SyncQueue {
    pub fn new(flush_interval_secs: f64) -> Self {
        Self {
            queue: Vec::new(),
            flush_interval_secs,
            last_flush: None,
            max_retries: 5,
        }
    }

    /// Queue an action for later dispatch.
    pub fn add_action(&mut self, action_type: &str, payload: &str) {
        self.queue.push(SyncAction {
            action_type: action_type.to_string(),
            payload: payload.to_string(),
            queued_at: now_secs(),
            retries: 0,
        });
    }

    /// Get a copy of all pending actions.
    pub fn get_pending(&self) -> &[SyncAction] {
        &self.queue
    }

    /// Number of pending actions.
    pub fn pending_count(&self) -> usize {
        self.queue.len()
    }

    /// Clear all pending actions.
    pub fn clear(&mut self) {
        self.queue.clear();
    }

    /// Check if it's time to flush based on interval.
    pub fn should_flush(&self) -> bool {
        match self.last_flush {
            Some(last) => last.elapsed().as_secs_f64() >= self.flush_interval_secs,
            None => true,
        }
    }

    /// Take all pending actions for dispatch.
    pub fn take_pending(&mut self) -> Vec<SyncAction> {
        self.last_flush = Some(Instant::now());
        std::mem::take(&mut self.queue)
    }

    /// Re-queue actions that failed to dispatch.
    pub fn requeue_failed(&mut self, mut actions: Vec<SyncAction>) {
        for action in &mut actions {
            action.retries += 1;
        }
        // Only re-queue actions that haven't exceeded max retries
        self.queue.extend(
            actions
                .into_iter()
                .filter(|a| a.retries <= self.max_retries),
        );
    }

    /// Remove actions that have been successfully dispatched.
    pub fn mark_flushed(&mut self, count: usize) {
        self.last_flush = Some(Instant::now());
        self.queue.drain(..count.min(self.queue.len()));
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
    fn test_add_and_take() {
        let mut q = SyncQueue::new(10.0);
        q.add_action("register", r#"{"doc":"d1"}"#);
        q.add_action("register", r#"{"doc":"d2"}"#);
        assert_eq!(q.pending_count(), 2);
        let actions = q.take_pending();
        assert_eq!(actions.len(), 2);
        assert_eq!(q.pending_count(), 0);
    }

    #[test]
    fn test_requeue_with_max_retries() {
        let mut q = SyncQueue::new(10.0);
        q.add_action("register", "{}");
        let actions = q.take_pending();
        // Simulate 6 retries
        let mut failed = actions;
        for a in &mut failed { a.retries = 6; }
        q.requeue_failed(failed);
        assert_eq!(q.pending_count(), 0); // exceeded max_retries=5
    }
}
