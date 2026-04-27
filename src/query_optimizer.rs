//! Query Optimizer with LRU Cache and Prefetching.
//!
//! Features:
//! - LRU cache for frequent queries
//! - Predictive prefetching based on patterns
//! - Query planning
//! - Result caching with TTL
//! - Adaptive caching based on memory
//!
//! Ported from splatsdb Python.

use std::collections::HashMap;

use ndarray::Array1;

/// A cache entry for query results.
#[derive(Debug, Clone)]
pub struct CacheEntry {
    pub query_hash: String,
    pub results_json: String,
    pub timestamp: f64,
    pub access_count: usize,
    pub last_access: f64,
    pub size_bytes: usize,
    pub ttl_seconds: f64,
}

impl CacheEntry {
    /// Is expired.
    pub fn is_expired(&self, now: f64) -> bool {
        if self.ttl_seconds <= 0.0 {
            return false;
        }
        (now - self.timestamp) > self.ttl_seconds
    }

    /// Touch.
    pub fn touch(&mut self, now: f64) {
        self.last_access = now;
        self.access_count += 1;
    }
}

/// A detected query pattern.
#[derive(Debug, Clone)]
pub struct QueryPattern {
    pub query_ids: Vec<String>,
    pub frequency: usize,
    pub last_seen: f64,
}

/// LRU Cache for query results.
pub struct QueryCache {
    entries: HashMap<String, CacheEntry>,
    access_order: Vec<String>,
    max_entries: usize,
    max_memory_bytes: usize,
    current_memory_bytes: usize,
    default_ttl: f64,
    hits: u64,
    misses: u64,
    evictions: u64,
}

fn now_secs() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

fn hash_query(query: &Array1<f32>, k: usize, filters_json: Option<&str>) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    for &v in query.iter() {
        v.to_bits().hash(&mut hasher);
    }
    k.hash(&mut hasher);
    if let Some(f) = filters_json {
        f.hash(&mut hasher);
    }
    format!("{:016x}", hasher.finish())
}

impl QueryCache {
    /// New.
    pub fn new(max_entries: usize, max_memory_mb: usize, default_ttl: f64) -> Self {
        Self {
            entries: HashMap::new(),
            access_order: Vec::new(),
            max_entries,
            max_memory_bytes: max_memory_mb * 1024 * 1024,
            current_memory_bytes: 0,
            default_ttl,
            hits: 0,
            misses: 0,
            evictions: 0,
        }
    }

    /// Get cached result if available and not expired.
    pub fn get(
        &mut self,
        query: &Array1<f32>,
        k: usize,
        filters_json: Option<&str>,
    ) -> Option<String> {
        let key = hash_query(query, k, filters_json);
        let now = now_secs();

        if let Some(entry) = self.entries.get_mut(&key) {
            if entry.is_expired(now) {
                self.evict_key(&key);
                self.misses += 1;
                return None;
            }

            let results = entry.results_json.clone();
            entry.touch(now);

            // Move to end of access order
            self.access_order.retain(|k| k != &key);
            self.access_order.push(key.clone());
            self.hits += 1;
            return Some(results);
        }

        self.misses += 1;
        None
    }

    /// Store a result in the cache.
    pub fn put(
        &mut self,
        query: &Array1<f32>,
        results_json: String,
        k: usize,
        filters_json: Option<&str>,
        ttl: Option<f64>,
    ) {
        let key = hash_query(query, k, filters_json);
        let size_bytes = results_json.len();
        let now = now_secs();

        // Evict until we have space
        while self.entries.len() >= self.max_entries
            || self.current_memory_bytes + size_bytes > self.max_memory_bytes
        {
            if !self.evict_oldest() {
                break;
            }
        }

        // Remove old entry if exists
        self.entries.remove(&key);

        let mut old_results = results_json;
        if let Some(old) = self.entries.get(&key) {
            old_results = old.results_json.clone();
        }

        self.entries.insert(
            key.clone(),
            CacheEntry {
                query_hash: key.clone(),
                results_json: old_results,
                timestamp: now,
                access_count: 0,
                last_access: now,
                size_bytes,
                ttl_seconds: ttl.unwrap_or(self.default_ttl),
            },
        );
        self.access_order.push(key);
        self.current_memory_bytes += size_bytes;
    }

    fn evict_key(&mut self, key: &str) {
        if let Some(entry) = self.entries.remove(key) {
            self.current_memory_bytes -= entry.size_bytes;
            self.access_order.retain(|k| k != key);
            self.evictions += 1;
        }
    }

    fn evict_oldest(&mut self) -> bool {
        if self.access_order.is_empty() {
            return false;
        }
        let oldest = self
            .access_order
            .first()
            .expect("access_order non-empty after is_empty check")
            .clone();
        self.evict_key(&oldest);
        true
    }

    /// Clear.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.access_order.clear();
        self.current_memory_bytes = 0;
    }

    /// Get stats.
    pub fn get_stats(&self) -> serde_json::Value {
        let total = self.hits + self.misses;
        let hit_rate = if total > 0 {
            self.hits as f64 / total as f64
        } else {
            0.0
        };
        serde_json::json!({
            "entries": self.entries.len(),
            "max_entries": self.max_entries,
            "memory_mb": self.current_memory_bytes as f64 / (1024.0 * 1024.0),
            "max_memory_mb": self.max_memory_bytes / (1024 * 1024),
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": hit_rate,
        })
    }
}

/// Predictive prefetcher based on query patterns.
pub struct QueryPrefetcher {
    window_size: usize,
    query_history: Vec<String>,
    patterns: HashMap<String, QueryPattern>,
}

impl QueryPrefetcher {
    /// New.
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            query_history: Vec::new(),
            patterns: HashMap::new(),
        }
    }

    /// Record a query in history.
    pub fn record_query(&mut self, query_hash: &str) {
        self.query_history.push(query_hash.to_string());

        if self.query_history.len() > self.window_size * 2 {
            let start = self.query_history.len() - self.window_size;
            self.query_history = self.query_history.split_off(start);
        }

        self.detect_patterns();
    }

    fn detect_patterns(&mut self) {
        if self.query_history.len() < 3 {
            return;
        }

        let now = now_secs();
        for seq_len in [3, 2] {
            if self.query_history.len() < seq_len {
                continue;
            }
            let start = self.query_history.len() - seq_len;
            let recent = &self.query_history[start..];
            let pattern_key = recent.join("|");

            self.patterns
                .entry(pattern_key)
                .and_modify(|p| {
                    p.frequency += 1;
                    p.last_seen = now;
                })
                .or_insert(QueryPattern {
                    query_ids: recent.to_vec(),
                    frequency: 1,
                    last_seen: now,
                });
        }
    }

    /// Predict next query based on patterns.
    pub fn predict_next(&self, current_hash: &str) -> Option<String> {
        let now = now_secs();
        let mut best: Option<&QueryPattern> = None;
        let mut best_score = 0.0f64;

        for pattern in self.patterns.values() {
            if pattern.query_ids.last().map(|s| s.as_str()) == Some(current_hash) {
                let recency = now - pattern.last_seen;
                let score = pattern.frequency as f64 / (1.0 + recency);
                if score > best_score {
                    best_score = score;
                    best = Some(pattern);
                }
            }
        }

        best.and_then(|p| {
            if p.query_ids.len() >= 2 {
                Some(p.query_ids[0].clone())
            } else {
                None
            }
        })
    }

    /// Get prefetch candidates.
    pub fn get_prefetch_candidates(&self, n: usize) -> Vec<String> {
        if self.query_history.is_empty() {
            return vec![];
        }

        let mut candidates = Vec::new();

        if let Some(last) = self.query_history.last() {
            if let Some(predicted) = self.predict_next(last) {
                candidates.push(predicted);
            }
        }

        // Add frequent queries from history
        let mut freq: HashMap<&str, usize> = HashMap::new();
        for h in &self.query_history {
            *freq.entry(h.as_str()).or_insert(0) += 1;
        }
        let mut freq_vec: Vec<_> = freq.into_iter().collect();
        freq_vec.sort_by_key(|b| std::cmp::Reverse(b.1));

        for (hash, _) in freq_vec {
            if !candidates.iter().any(|c| c == hash) {
                candidates.push(hash.to_string());
                if candidates.len() >= n {
                    break;
                }
            }
        }

        candidates
    }
}

/// Complete query optimizer with cache and prefetching.
pub struct QueryOptimizer {
    pub cache: QueryCache,
    pub prefetcher: Option<QueryPrefetcher>,
    pub enable_prefetch: bool,
    pub total_queries: u64,
    pub cache_hits: u64,
    pub prefetch_hits: u64,
}

impl QueryOptimizer {
    /// New.
    pub fn new(cache_entries: usize, cache_memory_mb: usize, enable_prefetch: bool) -> Self {
        Self {
            cache: QueryCache::new(cache_entries, cache_memory_mb, 300.0),
            prefetcher: if enable_prefetch {
                Some(QueryPrefetcher::new(10))
            } else {
                None
            },
            enable_prefetch,
            total_queries: 0,
            cache_hits: 0,
            prefetch_hits: 0,
        }
    }

    /// Get cached results or execute search function.
    pub fn execute_with_cache<F>(
        &mut self,
        query: &Array1<f32>,
        k: usize,
        filters_json: Option<&str>,
        search_fn: F,
    ) -> String
    where
        F: FnOnce(&Array1<f32>, usize) -> String,
    {
        self.total_queries += 1;

        if let Some(cached) = self.cache.get(query, k, filters_json) {
            self.cache_hits += 1;
            return cached.to_string();
        }

        let results = search_fn(query, k);
        self.cache
            .put(query, results.clone(), k, filters_json, None);

        if let Some(ref mut pf) = self.prefetcher {
            let key = hash_query(query, k, filters_json);
            pf.record_query(&key);
        }

        results
    }

    /// Get prefetch suggestions.
    pub fn get_prefetch_suggestions(&self, n: usize) -> Vec<String> {
        self.prefetcher
            .as_ref()
            .map(|pf| pf.get_prefetch_candidates(n))
            .unwrap_or_default()
    }

    /// Get optimizer statistics.
    pub fn get_stats(&self) -> serde_json::Value {
        let cache_hit_rate = if self.total_queries > 0 {
            self.cache_hits as f64 / self.total_queries as f64
        } else {
            0.0
        };
        serde_json::json!({
            "total_queries": self.total_queries,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": cache_hit_rate,
            "prefetch_hits": self.prefetch_hits,
            "prefetch_enabled": self.enable_prefetch,
            "cache": self.cache.get_stats(),
        })
    }

    /// Clear cache and history.
    pub fn clear(&mut self) {
        self.cache.clear();
        if let Some(ref mut pf) = self.prefetcher {
            pf.query_history.clear();
            pf.patterns.clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_put_get() {
        let mut cache = QueryCache::new(100, 10, 300.0);
        let q = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        cache.put(&q, r#"{"results": []}"#.to_string(), 10, None, None);

        let result = cache.get(&q, 10, None);
        assert!(result.is_some());
        assert_eq!(result.unwrap(), r#"{"results": []}"#);
    }

    #[test]
    fn test_cache_miss() {
        let mut cache = QueryCache::new(100, 10, 300.0);
        let q = Array1::from_vec(vec![1.0, 2.0]);
        assert!(cache.get(&q, 10, None).is_none());
    }

    #[test]
    fn test_cache_eviction() {
        let mut cache = QueryCache::new(2, 10, 300.0);
        let q1 = Array1::from_vec(vec![1.0]);
        let q2 = Array1::from_vec(vec![2.0]);
        let q3 = Array1::from_vec(vec![3.0]);

        cache.put(&q1, "r1".to_string(), 10, None, None);
        cache.put(&q2, "r2".to_string(), 10, None, None);
        cache.put(&q3, "r3".to_string(), 10, None, None);

        assert!(cache.get(&q1, 10, None).is_none()); // evicted
        assert!(cache.get(&q2, 10, None).is_some());
        assert!(cache.get(&q3, 10, None).is_some());
    }

    #[test]
    fn test_optimizer_execute() {
        let mut opt = QueryOptimizer::new(100, 10, false);
        let q = Array1::from_vec(vec![1.0, 2.0]);

        let r1 = opt.execute_with_cache(&q, 5, None, |_, _| "result1".to_string());
        assert_eq!(r1, "result1");
        assert_eq!(opt.cache_hits, 0);

        let r2 = opt.execute_with_cache(&q, 5, None, |_, _| "result2".to_string());
        assert_eq!(r2, "result1"); // cached
        assert_eq!(opt.cache_hits, 1);
    }

    #[test]
    fn test_prefetcher() {
        let mut pf = QueryPrefetcher::new(10);
        pf.record_query("a");
        pf.record_query("b");
        pf.record_query("a");
        pf.record_query("b");
        let candidates = pf.get_prefetch_candidates(3);
        assert!(!candidates.is_empty());
    }
}
