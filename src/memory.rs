//! Hierarchical memory management for Gaussian Splats.
//!
//! Three-tier memory architecture:
//! - Hot (VRAM): Frequently accessed splats
//! - Warm (RAM): Recently accessed splats
//! - Cold (Disk/HashMap): Infrequently accessed splats
//!
//! Ported from splatdb Python.

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::splat_types::GaussianSplat;

/// Configuration for memory management.
#[derive(Debug, Clone)]
pub struct MemoryConfig {
    /// Max splats in VRAM (hot tier)
    pub vram_limit: usize,
    /// Max splats in RAM (warm tier)
    pub ram_limit: usize,
    /// Evict when usage exceeds this fraction
    pub eviction_threshold: f64,
    /// Promote after N accesses
    pub access_threshold: usize,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            vram_limit: 100_000,
            ram_limit: 1_000_000,
            eviction_threshold: 0.8,
            access_threshold: 10,
        }
    }
}

/// Memory statistics.
#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    pub vram_usage: usize,
    pub ram_usage: usize,
    pub total_splats: usize,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub evictions: u64,
}

/// Manages hierarchical memory for Gaussian splats.
pub struct SplatMemoryManager {
    config: MemoryConfig,
    vram: HashMap<u64, GaussianSplat>,
    ram: HashMap<u64, GaussianSplat>,
    cold: HashMap<u64, GaussianSplat>,
    access_count: HashMap<u64, usize>,
    last_access: HashMap<u64, f64>,
    stats: MemoryStats,
}

impl SplatMemoryManager {
    /// New.
    pub fn new(config: MemoryConfig) -> Self {
        Self {
            config,
            vram: HashMap::new(),
            ram: HashMap::new(),
            cold: HashMap::new(),
            access_count: HashMap::new(),
            last_access: HashMap::new(),
            stats: MemoryStats::default(),
        }
    }

    /// Add splats to memory.
    pub fn add_splats(&mut self, splats: Vec<GaussianSplat>, to_cold: bool) {
        for splat in splats {
            let id = splat.id;
            if to_cold {
                self.cold.insert(id, splat);
            } else {
                self.ram.insert(id, splat);
            }
            self.access_count.insert(id, 0);
            self.last_access.insert(id, 0.0);
        }
        self.update_total();
    }

    /// Get a splat by ID with automatic tier management.
    pub fn get_splat(&mut self, splat_id: u64) -> Option<&GaussianSplat> {
        *self.access_count.entry(splat_id).or_insert(0) += 1;
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();
        self.last_access.insert(splat_id, now);

        // Check VRAM (hot)
        if self.vram.contains_key(&splat_id) {
            self.stats.cache_hits += 1;
            return self.vram.get(&splat_id);
        }

        // Check RAM (warm) — promote if frequently accessed
        if self.ram.contains_key(&splat_id) {
            self.stats.cache_hits += 1;
            let count = *self.access_count.get(&splat_id).unwrap_or(&0);
            if count >= self.config.access_threshold {
                self.promote_to_vram(splat_id);
            }
            return self.ram.get(&splat_id);
        }

        // Load from cold storage
        if self.cold.contains_key(&splat_id) {
            self.stats.cache_misses += 1;
            self.load_to_ram(splat_id);
            return self.ram.get(&splat_id);
        }

        None
    }

    /// Prefetch splats from cold to warm storage.
    pub fn prefetch_to_warm(&mut self, splat_ids: &[u64]) {
        for &id in splat_ids {
            if self.cold.contains_key(&id) && !self.ram.contains_key(&id) && !self.vram.contains_key(&id) {
                self.load_to_ram(id);
            }
        }
    }

    fn promote_to_vram(&mut self, splat_id: u64) {
        if !self.ram.contains_key(&splat_id) {
            return;
        }
        if self.vram.len() as f64 >= self.config.vram_limit as f64 * self.config.eviction_threshold {
            self.evict_from_vram();
        }
        if let Some(splat) = self.ram.remove(&splat_id) {
            self.vram.insert(splat_id, splat);
        }
        self.stats.vram_usage = self.vram.len();
        self.stats.ram_usage = self.ram.len();
    }

    fn load_to_ram(&mut self, splat_id: u64) {
        if !self.cold.contains_key(&splat_id) {
            return;
        }
        if self.ram.len() as f64 >= self.config.ram_limit as f64 * self.config.eviction_threshold {
            self.evict_from_ram();
        }
        if let Some(splat) = self.cold.get(&splat_id).cloned() {
            self.ram.insert(splat_id, splat);
        }
        self.stats.ram_usage = self.ram.len();
    }

    fn evict_from_vram(&mut self) {
        if self.vram.is_empty() {
            return;
        }
        let lru_id = self.vram
            .keys()
            .copied()
            .min_by_key(|&id| self.last_access.get(&id).copied().unwrap_or(0.0) as u64)
            .expect("vram non-empty guaranteed by is_empty guard above");

        if let Some(splat) = self.vram.remove(&lru_id) {
            self.ram.insert(lru_id, splat);
        }
        self.stats.evictions += 1;
        self.stats.vram_usage = self.vram.len();
        self.stats.ram_usage = self.ram.len();
    }

    fn evict_from_ram(&mut self) {
        if self.ram.is_empty() {
            return;
        }
        let lru_id = self.ram
            .keys()
            .copied()
            .min_by_key(|&id| self.last_access.get(&id).copied().unwrap_or(0.0) as u64)
            .expect("ram non-empty guaranteed by is_empty guard above");

        self.ram.remove(&lru_id);
        self.stats.evictions += 1;
        self.stats.ram_usage = self.ram.len();
    }

    /// Get memory statistics.
    pub fn get_stats(&self) -> &MemoryStats {
        &self.stats
    }

    /// Clear all memory tiers.
    pub fn clear(&mut self) {
        self.vram.clear();
        self.ram.clear();
        self.cold.clear();
        self.access_count.clear();
        self.last_access.clear();
        self.stats = MemoryStats::default();
    }

    fn update_total(&mut self) {
        self.stats.total_splats = self.cold.len() + self.ram.len() + self.vram.len();
        self.stats.vram_usage = self.vram.len();
        self.stats.ram_usage = self.ram.len();
    }

    /// Vram size.
    pub fn vram_size(&self) -> usize { self.vram.len() }
    /// Ram size.
    pub fn ram_size(&self) -> usize { self.ram.len() }
    /// Cold size.
    pub fn cold_size(&self) -> usize { self.cold.len() }
}

#[cfg(test)]
mod tests {
    use super::*;


    fn make_splat(id: u64) -> GaussianSplat {
        GaussianSplat {
            id,
            position: [0.0, 0.0, 0.0],
            color: [1.0, 1.0, 1.0],
            opacity: 1.0,
            scale: [1.0, 1.0, 1.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
        }
    }

    #[test]
    fn test_add_and_get() {
        let mut mgr = SplatMemoryManager::new(MemoryConfig::default());
        mgr.add_splats(vec![make_splat(1), make_splat(2)], true);
        assert_eq!(mgr.cold_size(), 2);
        let s = mgr.get_splat(1);
        assert!(s.is_some());
        assert_eq!(mgr.get_stats().cache_misses, 1);
    }

    #[test]
    fn test_promotion() {
        let config = MemoryConfig {
            access_threshold: 2,
            ..Default::default()
        };
        let mut mgr = SplatMemoryManager::new(config);
        mgr.add_splats(vec![make_splat(1)], false);
        // Access twice to trigger promotion
        let _ = mgr.get_splat(1);
        let _ = mgr.get_splat(1);
        assert!(mgr.vram.contains_key(&1));
    }

    #[test]
    fn test_clear() {
        let mut mgr = SplatMemoryManager::new(MemoryConfig::default());
        mgr.add_splats(vec![make_splat(1)], true);
        mgr.clear();
        assert_eq!(mgr.cold_size(), 0);
    }
}
