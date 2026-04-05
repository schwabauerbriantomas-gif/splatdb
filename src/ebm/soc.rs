//! SOC (Self-Organized Criticality) Engine.
//! Detects instability and triggers avalanches for natural reorganization.

use ndarray::{Array1, Array2};
use std::collections::VecDeque;

use super::energy_api::EBMEnergy;

/// System criticality state.
#[derive(Debug, Clone, PartialEq)]
pub enum CriticalityState {
    Subcritical,
    Critical,
    Supercritical,
}

/// Report of system criticality.
#[derive(Debug, Clone)]
pub struct CriticalityReport {
    pub state: CriticalityState,
    pub index: f32,
    pub energy_variance: f32,
    pub size_variance: f32,
}

impl CriticalityReport {
    pub fn needs_relaxation(&self) -> bool {
        self.state == CriticalityState::Supercritical
    }
    pub fn needs_monitoring(&self) -> bool {
        matches!(
            self.state,
            CriticalityState::Critical | CriticalityState::Supercritical
        )
    }
}

/// Result of an avalanche reorganization.
#[derive(Debug, Clone)]
pub struct AvalancheResult {
    pub affected_clusters: usize,
    pub energy_released: f32,
    pub duration_ms: f32,
    pub new_equilibrium: f32,
}

/// Result of relaxation.
#[derive(Debug, Clone)]
pub struct RelaxationResult {
    pub initial_energy: f32,
    pub final_energy: f32,
    pub energy_delta: f32,
    pub iterations: usize,
    pub improved: bool,
}

struct Cluster {
    _center_idx: usize,
    energy: f32,
    size: usize,
    splat_indices: Vec<usize>,
}

/// SOC Engine for self-organized criticality.
pub struct SOCEngine {
    energy_api: EBMEnergy,
    critical_threshold: f32,
    clusters: Vec<Cluster>,
    splat_mu: Option<Array2<f32>>,
    splat_alpha: Option<Array1<f32>>,
    splat_kappa: Option<Array1<f32>>,
    avalanche_history: Vec<AvalancheResult>,
}

impl SOCEngine {
    pub fn new(energy_api: EBMEnergy, critical_threshold: f32) -> Self {
        Self {
            energy_api,
            critical_threshold,
            clusters: Vec::new(),
            splat_mu: None,
            splat_alpha: None,
            splat_kappa: None,
            avalanche_history: Vec::new(),
        }
    }

    /// Update splat data and rebuild internal clusters.
    pub fn update_splats(&mut self, mu: Array2<f32>, alpha: Array1<f32>, kappa: Array1<f32>) {
        let n = mu.nrows();
        self.splat_mu = Some(mu.clone());
        self.splat_alpha = Some(alpha.clone());
        self.splat_kappa = Some(kappa.clone());

        // Rebuild clusters (one per splat, simplified)
        self.clusters = (0..n)
            .map(|i| Cluster {
                _center_idx: i,
                energy: alpha[i],
                size: 1,
                splat_indices: vec![i],
            })
            .collect();

        // Update energy API
        self.energy_api.update_splats(mu, alpha, kappa);
    }

    /// Check current criticality state.
    pub fn check_criticality(&self) -> CriticalityReport {
        let energy_variance = self.compute_energy_variance();
        let size_variance = self.compute_size_variance();

        // Normalized criticality index
        let e_norm = energy_variance / (energy_variance + 1.0);
        let s_norm = size_variance / (size_variance + 1.0);
        let index = 0.6 * e_norm + 0.4 * s_norm;

        let state = if index < 0.3 {
            CriticalityState::Subcritical
        } else if index < 0.7 {
            CriticalityState::Critical
        } else {
            CriticalityState::Supercritical
        };

        CriticalityReport {
            state,
            index,
            energy_variance,
            size_variance,
        }
    }

    /// Trigger an avalanche of reorganization.
    pub fn trigger_avalanche(&mut self, seed_point: Option<usize>) -> AvalancheResult {
        let start = std::time::Instant::now();

        if self.clusters.is_empty() {
            return AvalancheResult {
                affected_clusters: 0,
                energy_released: 0.0,
                duration_ms: 0.0,
                new_equilibrium: 0.0,
            };
        }

        // Find seed: highest energy cluster or specified seed
        let start_idx = seed_point.unwrap_or_else(|| {
            self.clusters
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.energy.partial_cmp(&b.1.energy).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0)
        });

        // BFS avalanche
        let mut affected = Vec::new();
        let mut queue = VecDeque::new();
        let mut visited = vec![false; self.clusters.len()];
        let mut total_energy_released = 0.0f32;
        let max_cascade = 1000.min(self.clusters.len());

        queue.push_back(start_idx);
        visited[start_idx] = true;

        while let Some(idx) = queue.pop_front() {
            if affected.len() >= max_cascade {
                break;
            }

            let cluster_energy = self.clusters[idx].energy;
            if cluster_energy > self.critical_threshold {
                affected.push(idx);
                let released = cluster_energy * 0.3; // Release 30%
                total_energy_released += released;

                // Get neighbors (simplified: closest by mu distance)
                let neighbors = self.get_neighbor_indices(idx);

                let share = if neighbors.is_empty() {
                    0.0
                } else {
                    released / neighbors.len() as f32
                };
                for &n_idx in &neighbors {
                    self.clusters[n_idx].energy += share;
                    if !visited[n_idx] && self.clusters[n_idx].energy > self.critical_threshold {
                        visited[n_idx] = true;
                        queue.push_back(n_idx);
                    }
                }

                // Reorganize: reduce energy by 30%
                self.clusters[idx].energy *= 0.7;
            }
        }

        // Update splat_alpha to reflect relaxation
        if let Some(ref mut alpha) = self.splat_alpha {
            for &cluster_idx in &affected {
                for &splat_idx in &self.clusters[cluster_idx].splat_indices {
                    if splat_idx < alpha.len() {
                        alpha[splat_idx] *= 0.7;
                    }
                }
            }
        }

        // Update energy API
        if let (Some(mu), Some(alpha), Some(kappa)) =
            (&self.splat_mu, &self.splat_alpha, &self.splat_kappa)
        {
            self.energy_api
                .update_splats(mu.clone(), alpha.clone(), kappa.clone());
        }

        let new_eq = self.compute_equilibrium();
        let duration_ms = start.elapsed().as_secs_f32() * 1000.0;

        let result = AvalancheResult {
            affected_clusters: affected.len(),
            energy_released: total_energy_released,
            duration_ms,
            new_equilibrium: new_eq,
        };
        self.avalanche_history.push(result.clone());
        result
    }

    /// Relax the system toward lower energy.
    pub fn relax(&mut self, iterations: usize) -> RelaxationResult {
        let initial_energy = self.energy_api.free_energy();

        if self.splat_mu.is_none() || self.splat_alpha.is_none() {
            return RelaxationResult {
                initial_energy,
                final_energy: initial_energy,
                energy_delta: 0.0,
                iterations: 0,
                improved: false,
            };
        }

        for _ in 0..iterations {
            // Clone kappa for usage computation, then mutate alpha
            let kappa_clone = self.splat_kappa.clone();
            if let Some(ref mut alpha) = self.splat_alpha {
                let usage = kappa_clone.as_ref().unwrap().mapv(|k| (-k * 0.1).exp());
                for i in 0..alpha.len() {
                    alpha[i] *= 1.0 + usage[i] * 0.05;
                }
                let total: f32 = alpha.sum();
                if total > 0.0 {
                    alpha.mapv_inplace(|a| a / total);
                }
            }

            if let Some(ref mut kappa) = self.splat_kappa {
                kappa.mapv_inplace(|k| (k * 1.01).clamp(0.1, 100.0));
            }
        }

        // Update energy API
        if let (Some(mu), Some(alpha), Some(kappa)) =
            (&self.splat_mu, &self.splat_alpha, &self.splat_kappa)
        {
            self.energy_api
                .update_splats(mu.clone(), alpha.clone(), kappa.clone());
        }

        let final_energy = self.energy_api.free_energy();

        RelaxationResult {
            initial_energy,
            final_energy,
            energy_delta: final_energy - initial_energy,
            iterations,
            improved: final_energy < initial_energy,
        }
    }

    fn compute_energy_variance(&self) -> f32 {
        if self.clusters.is_empty() {
            return 0.0;
        }
        let energies: Vec<f32> = self.clusters.iter().map(|c| c.energy).collect();
        let mean = energies.iter().sum::<f32>() / energies.len() as f32;
        let variance = energies
            .iter()
            .map(|e| (e - mean) * (e - mean))
            .sum::<f32>()
            / energies.len() as f32;
        variance
    }

    fn compute_size_variance(&self) -> f32 {
        if self.clusters.is_empty() {
            return 0.0;
        }
        let sizes: Vec<f32> = self.clusters.iter().map(|c| c.size as f32).collect();
        let mean = sizes.iter().sum::<f32>() / sizes.len() as f32;
        let variance =
            sizes.iter().map(|s| (s - mean) * (s - mean)).sum::<f32>() / sizes.len() as f32;
        variance
    }

    fn compute_equilibrium(&self) -> f32 {
        if self.clusters.is_empty() {
            return 0.0;
        }
        self.clusters.iter().map(|c| c.energy).sum::<f32>() / self.clusters.len() as f32
    }

    fn get_neighbor_indices(&self, idx: usize) -> Vec<usize> {
        // Simplified: find closest clusters by mu distance
        let mu = match &self.splat_mu {
            Some(m) => m,
            None => return vec![],
        };
        if idx >= mu.nrows() {
            return vec![];
        }

        let center = mu.row(idx);
        let mut dists: Vec<(f32, usize)> = (0..mu.nrows())
            .filter(|&i| i != idx)
            .map(|i| {
                let diff = &mu.row(i) - &center;
                (diff.dot(&diff).sqrt(), i)
            })
            .collect();

        dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        dists.iter().take(10).map(|(_, i)| *i).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};

    fn make_engine() -> SOCEngine {
        SOCEngine::new(EBMEnergy::new(), 0.5)
    }

    fn make_engine_with_splats(n: usize) -> SOCEngine {
        let mut engine = SOCEngine::new(EBMEnergy::new(), 0.5);
        let mu = Array2::zeros((n, 3));
        let alpha = Array1::from_elem(n, 0.6f32);
        let kappa = Array1::from_elem(n, 1.0f32);
        engine.update_splats(mu, alpha, kappa);
        engine
    }

    #[test]
    fn test_soc_creation() {
        let engine = make_engine();
        assert!(engine.clusters.is_empty());
        assert_eq!(engine.critical_threshold, 0.5);
    }

    #[test]
    fn test_check_criticality_empty() {
        let engine = make_engine();
        let report = engine.check_criticality();
        assert_eq!(report.state, CriticalityState::Subcritical);
        assert_eq!(report.energy_variance, 0.0);
    }

    #[test]
    fn test_check_criticality_with_splats() {
        let engine = make_engine_with_splats(10);
        let report = engine.check_criticality();
        assert_eq!(report.state, CriticalityState::Subcritical);
        assert!(report.index < 0.3);
    }

    #[test]
    fn test_avalanche_empty() {
        let mut engine = make_engine();
        let result = engine.trigger_avalanche(None);
        assert_eq!(result.affected_clusters, 0);
        assert_eq!(result.energy_released, 0.0);
    }

    #[test]
    fn test_avalanche_with_splats() {
        let mut engine = make_engine_with_splats(5);
        let result = engine.trigger_avalanche(None);
        assert!(result.affected_clusters > 0);
        assert!(result.energy_released > 0.0);
        assert_eq!(engine.avalanche_history.len(), 1);
    }

    #[test]
    fn test_relax_no_splats() {
        let mut engine = make_engine();
        let result = engine.relax(10);
        assert!(!result.improved);
        assert_eq!(result.iterations, 0);
    }

    #[test]
    fn test_relax_with_splats() {
        let mut engine = make_engine_with_splats(5);
        let result = engine.relax(5);
        assert_eq!(result.iterations, 5);
    }

    #[test]
    fn test_criticality_report_methods() {
        let sub = CriticalityReport {
            state: CriticalityState::Subcritical,
            index: 0.1,
            energy_variance: 0.0,
            size_variance: 0.0,
        };
        assert!(!sub.needs_relaxation());
        assert!(!sub.needs_monitoring());

        let superc = CriticalityReport {
            state: CriticalityState::Supercritical,
            index: 0.8,
            energy_variance: 1.0,
            size_variance: 1.0,
        };
        assert!(superc.needs_relaxation());
        assert!(superc.needs_monitoring());
    }
}
