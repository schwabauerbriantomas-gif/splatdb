//! Energy-based router for SplatsDB Cluster.
//! Routes queries based on energy landscape of nodes.
//! Ported from splatsdb Python.

use std::collections::HashMap;

/// Node energy state for routing decisions.
#[derive(Debug, Clone)]
struct NodeEnergy {
    _node_id: String,
    last_energy: f64,
    updated_at: f64,
    weight: f64,
}

/// Routing strategy.
#[derive(Debug, Clone, Copy)]
pub enum RoutingStrategy {
    EnergyBalanced,
    RoundRobin,
    LeastLoaded,
    LocalityAware,
    Hybrid,
}

impl RoutingStrategy {
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Self {
        match s {
            "energy_balanced" => Self::EnergyBalanced,
            "round_robin" => Self::RoundRobin,
            "least_loaded" => Self::LeastLoaded,
            "locality_aware" => Self::LocalityAware,
            "hybrid" => Self::Hybrid,
            _ => Self::EnergyBalanced,
        }
    }
}

/// Energy-aware router configuration.
pub struct EnergyRouterConfig {
    pub enabled: bool,
    pub strategy: RoutingStrategy,
    pub fallback_strategy: RoutingStrategy,
    pub cache_energy: bool,
    pub cache_ttl_secs: f64,
    pub min_nodes: usize,
}

impl Default for EnergyRouterConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            strategy: RoutingStrategy::EnergyBalanced,
            fallback_strategy: RoutingStrategy::RoundRobin,
            cache_energy: true,
            cache_ttl_secs: 60.0,
            min_nodes: 2,
        }
    }
}

/// Energy-based router for distributing queries across cluster nodes.
///
/// Each node has an energy state. Lower energy = higher confidence = better
/// result potential. The router selects nodes probabilistically based on energy.
pub struct EnergyRouter {
    config: EnergyRouterConfig,
    node_energies: HashMap<String, NodeEnergy>,
    energy_cache: HashMap<String, f64>,
    cache_timestamps: HashMap<String, f64>,
    route_counts: HashMap<String, usize>,
    total_routes: usize,
    rr_index: usize,
}

impl EnergyRouter {
    pub fn new(config: EnergyRouterConfig) -> Self {
        Self {
            config,
            node_energies: HashMap::new(),
            energy_cache: HashMap::new(),
            cache_timestamps: HashMap::new(),
            route_counts: HashMap::new(),
            total_routes: 0,
            rr_index: 0,
        }
    }

    pub fn register_node(&mut self, node_id: &str, weight: f64) {
        self.node_energies.insert(
            node_id.to_string(),
            NodeEnergy {
                _node_id: node_id.to_string(),
                last_energy: 1.0,
                updated_at: now_secs(),
                weight,
            },
        );
        self.route_counts.insert(node_id.to_string(), 0);
    }

    pub fn remove_node(&mut self, node_id: &str) {
        self.node_energies.remove(node_id);
        self.route_counts.remove(node_id);
    }

    pub fn update_node_energy(&mut self, node_id: &str, energy: f64) {
        if let Some(ne) = self.node_energies.get_mut(node_id) {
            ne.last_energy = energy;
            ne.updated_at = now_secs();
        }
    }

    /// Route a query to the best node(s). Returns ordered list of node IDs.
    pub fn route(
        &mut self,
        query_hash: u64,
        online_nodes: &[String],
        load_metrics: Option<&HashMap<String, crate::cluster::health::LoadMetrics>>,
    ) -> Vec<String> {
        if online_nodes.is_empty() {
            return Vec::new();
        }

        if !self.config.enabled || online_nodes.len() < self.config.min_nodes {
            return online_nodes.to_vec();
        }

        match self.config.strategy {
            RoutingStrategy::EnergyBalanced => self.energy_route(query_hash, online_nodes),
            RoutingStrategy::RoundRobin => self.round_robin_route(online_nodes),
            RoutingStrategy::LeastLoaded => self.least_loaded_route(online_nodes, load_metrics),
            RoutingStrategy::LocalityAware => self.locality_route(online_nodes),
            RoutingStrategy::Hybrid => self.hybrid_route(online_nodes, load_metrics),
        }
    }

    fn energy_route(&mut self, query_hash: u64, nodes: &[String]) -> Vec<String> {
        let node_energies: Vec<(String, f64)> = nodes
            .iter()
            .map(|nid| {
                let energy = self.get_cached_energy(query_hash, nid);
                let weight = self
                    .node_energies
                    .get(nid)
                    .map(|ne| ne.weight)
                    .unwrap_or(1.0);
                (nid.clone(), energy * weight)
            })
            .collect();

        let energies: Vec<f64> = node_energies.iter().map(|(_, e)| *e).collect();
        let max_e = energies.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // Convert to probabilities (lower energy = higher probability)
        let probs: Vec<f64> = energies.iter().map(|e| max_e - e + 0.1).collect();
        let total: f64 = probs.iter().sum();

        let probs: Vec<f64> = if total > 0.0 {
            probs.iter().map(|p| p / total).collect()
        } else {
            vec![1.0 / nodes.len() as f64; nodes.len()]
        };

        // Deterministic selection based on query hash (no rand dependency)
        let hash_f = (query_hash as f64) % 1_000_000.0 / 1_000_000.0;
        let mut acc = 0.0;
        let mut selected_idx = 0;
        for (i, p) in probs.iter().enumerate() {
            acc += p;
            if hash_f < acc {
                selected_idx = i;
                break;
            }
        }

        let selected = nodes[selected_idx].clone();
        self.record_route(&selected);
        vec![selected]
    }

    fn round_robin_route(&mut self, nodes: &[String]) -> Vec<String> {
        if nodes.is_empty() {
            return Vec::new();
        }
        let idx = self.rr_index % nodes.len();
        self.rr_index += 1;
        let selected = nodes[idx].clone();
        self.record_route(&selected);
        vec![selected]
    }

    fn least_loaded_route(
        &mut self,
        nodes: &[String],
        load_metrics: Option<&HashMap<String, crate::cluster::health::LoadMetrics>>,
    ) -> Vec<String> {
        let metrics = match load_metrics {
            Some(m) => m,
            None => return self.round_robin_route(nodes),
        };

        let mut scored: Vec<(String, usize)> = nodes
            .iter()
            .map(|nid| {
                let load = metrics.get(nid).map(|m| m.active_queries).unwrap_or(0);
                (nid.clone(), load)
            })
            .collect();
        scored.sort_by_key(|(_, l)| *l);
        let selected = scored[0].0.clone();
        self.record_route(&selected);
        vec![selected]
    }

    fn locality_route(&mut self, nodes: &[String]) -> Vec<String> {
        let mut scored: Vec<(String, f64)> = nodes
            .iter()
            .map(|nid| {
                let energy = self
                    .node_energies
                    .get(nid)
                    .map(|ne| ne.last_energy)
                    .unwrap_or(1.0);
                let locality = 1.0 - energy.min(1.0);
                (nid.clone(), locality)
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let selected = scored[0].0.clone();
        self.record_route(&selected);
        vec![selected]
    }

    fn hybrid_route(
        &mut self,
        nodes: &[String],
        load_metrics: Option<&HashMap<String, crate::cluster::health::LoadMetrics>>,
    ) -> Vec<String> {
        let mut scores: Vec<(String, f64)> = nodes
            .iter()
            .map(|nid| {
                let ne = self.node_energies.get(nid);
                let energy_score = ne.map(|n| n.last_energy).unwrap_or(1.0);

                let load_score = load_metrics
                    .and_then(|m| m.get(nid))
                    .map(|m| (m.active_queries as f64 / 100.0).min(1.0))
                    .unwrap_or(0.0);

                let locality_score = energy_score;
                let latency_score = 0.0;

                let combined = 0.4 * energy_score
                    + 0.2 * load_score
                    + 0.3 * locality_score
                    + 0.1 * latency_score;
                (nid.clone(), combined)
            })
            .collect();
        scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let selected = scores[0].0.clone();
        self.record_route(&selected);
        vec![selected]
    }

    fn get_cached_energy(&mut self, query_hash: u64, node_id: &str) -> f64 {
        if !self.config.cache_energy {
            return self
                .node_energies
                .get(node_id)
                .map(|ne| ne.last_energy)
                .unwrap_or(1.0);
        }

        let cache_key = format!("{}_{}", query_hash, node_id);
        let now = now_secs();

        if let Some(&ts) = self.cache_timestamps.get(&cache_key) {
            if now - ts < self.config.cache_ttl_secs {
                return self.energy_cache.get(&cache_key).copied().unwrap_or(1.0);
            }
        }

        let energy = self
            .node_energies
            .get(node_id)
            .map(|ne| ne.last_energy)
            .unwrap_or(1.0);
        self.energy_cache.insert(cache_key.clone(), energy);
        self.cache_timestamps.insert(cache_key, now);
        energy
    }

    fn record_route(&mut self, node_id: &str) {
        *self.route_counts.entry(node_id.to_string()).or_insert(0) += 1;
        self.total_routes += 1;
    }

    pub fn get_routing_stats(&self) -> EnergyRouterStats {
        EnergyRouterStats {
            enabled: self.config.enabled,
            total_routes: self.total_routes,
            cache_size: self.energy_cache.len(),
            nodes_count: self.node_energies.len(),
            route_distribution: self.route_counts.clone(),
            node_energies: self
                .node_energies
                .iter()
                .map(|(k, v)| (k.clone(), v.last_energy))
                .collect(),
        }
    }

    pub fn clear_cache(&mut self) {
        self.energy_cache.clear();
        self.cache_timestamps.clear();
    }
}

/// Routing statistics.
#[derive(Debug)]
pub struct EnergyRouterStats {
    pub enabled: bool,
    pub total_routes: usize,
    pub cache_size: usize,
    pub nodes_count: usize,
    pub route_distribution: HashMap<String, usize>,
    pub node_energies: HashMap<String, f64>,
}

fn now_secs() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}
