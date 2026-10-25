//! Sync queue for offline actions and coordinator reconnection.
//! Ported from splatsdb Python.

use std::time::Instant;

/// A pending action to be dispatched when connectivity restores.
#[derive(Debug, Clone)]
pub struct SyncAction {
    pub action_type: String,
    pub payload: String, // JSON payload
    pub queued_at: f64,
    pub retries: usize,
    /// Process-wide monotonic id: the identity used by `mark_flushed`
    /// (timestamps alone can collide for same-payload actions in one tick).
    pub seq: u64,
}

static NEXT_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

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
            seq: NEXT_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
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
    ///
    /// An action keeps being retried while it has seen fewer than
    /// `max_retries` attempts (the previous `<=` filter allowed one extra
    /// attempt — max_retries=5 gave 6 dispatches).
    pub fn requeue_failed(&mut self, mut actions: Vec<SyncAction>) {
        for action in &mut actions {
            action.retries += 1;
        }
        // Only re-queue actions that haven't exceeded max retries
        self.queue
            .extend(actions.into_iter().filter(|a| a.retries < self.max_retries));
    }

    /// Remove successfully dispatched actions, matched by identity.
    ///
    /// The previous count-based version drained from the FRONT of the queue,
    /// so actions re-queued by `requeue_failed` were silently dropped when
    /// they happened to sit at the front. Identity matching keeps them alive.
    /// Returns the number of actions actually removed.
    pub fn mark_flushed(&mut self, confirmed: &[SyncAction]) -> usize {
        self.last_flush = Some(Instant::now());
        // Identity = the monotonic `seq` assigned at queue time (payload+timestamp
        // pairs can collide when the same payload is queued twice in one tick).
        let confirmed_seqs: std::collections::HashSet<u64> =
            confirmed.iter().map(|a| a.seq).collect();
        let before = self.queue.len();
        self.queue.retain(|a| !confirmed_seqs.contains(&a.seq));
        before - self.queue.len()
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
        for a in &mut failed {
            a.retries = 6;
        }
        q.requeue_failed(failed);
        assert_eq!(q.pending_count(), 0); // exceeded max_retries=5
    }

    #[test]
    fn test_requeue_retry_budget_is_exactly_max_retries() {
        // max_retries = 5 → exactly 5 dispatch attempts in total.
        // The old `<=` filter allowed a 6th attempt.
        let mut q = SyncQueue::new(10.0);
        q.add_action("register", "{}");
        let mut dispatches = 0;
        while q.pending_count() > 0 {
            let batch = q.take_pending();
            dispatches += 1;
            q.requeue_failed(batch);
        }
        assert_eq!(
            dispatches, 5,
            "total dispatch attempts must equal max_retries"
        );
        assert_eq!(q.pending_count(), 0);
    }

    #[test]
    fn test_partial_flush_requeued_actions_survive() {
        // Regression: the old count-based mark_flushed drained from the FRONT,
        // so actions re-queued by requeue_failed were silently dropped when
        // they sat at the front of the queue.
        let mut q = SyncQueue::new(10.0);
        q.add_action("register", r#"{"doc":"a"}"#);
        q.add_action("register", r#"{"doc":"b"}"#);
        q.add_action("register", r#"{"doc":"c"}"#);
        assert_eq!(q.pending_count(), 3);

        let batch = q.take_pending();
        let confirmed: Vec<SyncAction> = batch
            .iter()
            .filter(|a| a.payload.contains(r#""a""#) || a.payload.contains(r#""c""#))
            .cloned()
            .collect();
        let failed: Vec<SyncAction> = batch
            .iter()
            .filter(|a| a.payload.contains(r#""b""#))
            .cloned()
            .collect();

        q.requeue_failed(failed);
        assert_eq!(q.pending_count(), 1);

        // Confirming the two successes must NOT remove the re-queued failure.
        let removed = q.mark_flushed(&confirmed);
        assert_eq!(removed, 0, "confirmed actions are no longer queued");
        assert_eq!(q.pending_count(), 1, "re-queued 'b' must survive");
        let pending = q.get_pending();
        assert!(pending[0].payload.contains(r#""b""#));
        assert_eq!(pending[0].retries, 1);
    }
}
