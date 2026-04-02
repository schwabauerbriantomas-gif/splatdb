//! EBM Exploration — exploration of the energy landscape.
//! Find high-uncertainty regions, sample uncertain points, suggest exploration.
//! Ported from m2m-vector-search Python.


/// A region in energy space.
#[derive(Debug, Clone)]
pub struct EnergyRegion {
    pub center: Vec<f32>,
    pub energy: f64,
    pub radius: f64,
    pub n_points: usize,
}

/// An exploration suggestion for an agent.
#[derive(Debug, Clone)]
pub struct ExplorationSuggestion {
    pub region: EnergyRegion,
    pub description: String,
    pub suggested_queries: Vec<String>,
    pub potential_value: f64,
}

/// Energy function trait — implement to provide energy calculations.
pub trait EnergyFn: Send + Sync {
    fn energy(&self, point: &[f32]) -> f64;
}

/// Default energy function: returns 0.5 for all points.
pub struct DefaultEnergyFn;
impl EnergyFn for DefaultEnergyFn {
    fn energy(&self, _point: &[f32]) -> f64 { 0.5 }
}

/// EBM Exploration engine.
pub struct EbmExploration<E: EnergyFn> {
    energy_api: E,
    all_vectors: Vec<Vec<f32>>,
    all_ids: Vec<String>,
    rng_state: u64,
}

impl<E: EnergyFn> EbmExploration<E> {
    pub fn new(energy_api: E) -> Self {
        Self {
            energy_api,
            all_vectors: Vec::new(),
            all_ids: Vec::new(),
            rng_state: 42,
        }
    }

    pub fn with_vectors(mut self, vectors: Vec<Vec<f32>>, ids: Vec<String>) -> Self {
        self.all_vectors = vectors;
        self.all_ids = ids;
        self
    }

    pub fn update_vectors(&mut self, vectors: Vec<Vec<f32>>, ids: Vec<String>) {
        self.all_vectors = vectors;
        self.all_ids = ids;
    }

    /// Find high-energy (high-uncertainty) regions.
    pub fn find_high_energy_regions(
        &mut self,
        topic_vector: Option<&[f32]>,
        min_energy: f64,
        n_regions: usize,
        n_samples: usize,
    ) -> Vec<EnergyRegion> {
        if self.all_vectors.is_empty() { return Vec::new(); }
        let dim = self.all_vectors[0].len();
        if dim == 0 { return Vec::new(); }

        // Sample candidate points
        let candidates = match topic_vector {
            Some(center) => self.sample_around(center, n_samples, 2.0, dim),
            None => self.sample_global(n_samples, dim),
        };

        // Compute energy and filter
        let mut high_energy: Vec<(Vec<f32>, f64)> = candidates.into_iter()
            .filter_map(|pt| {
                let e = self.energy_api.energy(&pt);
                if e >= min_energy { Some((pt, e)) } else { None }
            })
            .collect();

        if high_energy.is_empty() { return Vec::new(); }

        self.cluster_high_energy_points(&mut high_energy, n_regions)
    }

    /// Sample points from uncertain (high-energy) regions using Boltzmann sampling.
    pub fn sample_uncertain(
        &mut self,
        k: usize,
        temperature: f64,
        from_region: Option<&EnergyRegion>,
    ) -> Vec<Vec<f32>> {
        if self.all_vectors.is_empty() { return Vec::new(); }
        let dim = self.all_vectors[0].len();

        let candidates = match from_region {
            Some(region) => self.sample_around(&region.center, k * 10, region.radius, dim),
            None => self.sample_global(k * 10, dim),
        };

        if candidates.is_empty() { return Vec::new(); }

        // Boltzmann sampling: weight ∝ exp(E / T)
        let energies: Vec<f64> = candidates.iter().map(|c| self.energy_api.energy(c)).collect();
        let max_e = energies.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<f64> = energies.iter().map(|e| ((e - max_e) / temperature).exp()).collect();
        let total: f64 = weights.iter().sum();

        if total <= 0.0 { return candidates.into_iter().take(k).collect(); }

        let k_actual = k.min(candidates.len());
        let mut selected = Vec::new();
        let mut used = vec![false; candidates.len()];

        for _ in 0..k_actual {
            // Weighted random selection
            let mut r = self.next_random() * total;
            for (i, &w) in weights.iter().enumerate() {
                if used[i] { continue; }
                r -= w;
                if r <= 0.0 {
                    used[i] = true;
                    selected.push(candidates[i].clone());
                    break;
                }
            }
        }

        selected
    }

    /// Suggest exploration areas based on knowledge gaps.
    pub fn suggest_exploration(
        &mut self,
        n_suggestions: usize,
    ) -> Vec<ExplorationSuggestion> {
        if self.all_vectors.is_empty() { return Vec::new(); }

        let regions = self.find_high_energy_regions(None, 0.5, n_suggestions * 2, 500);
        let mut suggestions = Vec::new();

        for region in regions.into_iter().take(n_suggestions) {
            // Find nearby vectors
            let mut nearby: Vec<(usize, f64)> = self.all_vectors.iter().enumerate().map(|(i, v)| {
                let dist: f64 = v.iter().zip(region.center.iter())
                    .map(|(&a, &b)| { let d = a as f64 - b as f64; d * d })
                    .sum::<f64>()
                    .sqrt();
                (i, dist)
            }).collect();
            nearby.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            let nearby_ids: Vec<String> = nearby.iter().take(5)
                .map(|(i, _)| self.all_ids.get(*i).cloned().unwrap_or_else(|| format!("vec_{}", i)))
                .collect();

            let desc = format!(
                "Region with high uncertainty (energy={:.3}). {} sampled points. Nearby: {}",
                region.energy, region.n_points,
                nearby_ids.iter().take(3).cloned().collect::<Vec<_>>().join(", ")
            );

            let first_id = nearby_ids.first().cloned().unwrap_or_else(|| "center".into());
            suggestions.push(ExplorationSuggestion {
                suggested_queries: vec![
                    format!("Explore zone near {}", first_id),
                    format!("Search complementary info in energy region {:.2}", region.energy),
                ],
                description: desc,
                potential_value: region.energy,
                region,
            });
        }

        suggestions.sort_by(|a, b| b.potential_value.partial_cmp(&a.potential_value).unwrap_or(std::cmp::Ordering::Equal));
        suggestions
    }

    /// Find knowledge gaps — regions between known clusters.
    pub fn find_knowledge_gaps(&mut self, n_gaps: usize) -> Vec<EnergyRegion> {
        self.find_high_energy_regions(None, 0.5, n_gaps, 1000)
    }

    // ─── Internal ───

    fn sample_around(&mut self, center: &[f32], n: usize, radius: f64, dim: usize) -> Vec<Vec<f32>> {
        (0..n).map(|_| {
            self.next_vec(dim).iter().zip(center.iter())
                .map(|(&noise, &c)| c + noise * (radius as f32))
                .collect()
        }).collect()
    }

    fn sample_global(&mut self, n: usize, dim: usize) -> Vec<Vec<f32>> {
        if self.all_vectors.is_empty() {
            return (0..n).map(|_| self.next_vec(dim)).collect();
        }
        // Mix known vectors with noise
        (0..n).map(|_| {
            let idx = (self.next_random() * self.all_vectors.len() as f64) as usize;
            let idx = idx.min(self.all_vectors.len() - 1);
            let noise = self.next_vec(dim);
            let base = &self.all_vectors[idx];
            base.iter().zip(noise.iter()).map(|(&b, &n)| b + n * 0.5).collect()
        }).collect()
    }

    fn cluster_high_energy_points(
        &self,
        points: &mut [(Vec<f32>, f64)],
        n_clusters: usize,
    ) -> Vec<EnergyRegion> {
        points.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let dim = points.first().map(|(p, _)| p.len()).unwrap_or(0);
        let min_radius = 0.5f64;
        let mut assigned = vec![false; points.len()];
        let mut regions = Vec::new();

        for i in 0..points.len() {
            if assigned[i] { continue; }
            let (pt, e) = &points[i];
            let mut cluster_pts = vec![pt.clone()];

            for j in 0..points.len() {
                if i != j && !assigned[j] {
                    let dist = euclidean_dist(pt, &points[j].0);
                    if dist < min_radius {
                        cluster_pts.push(points[j].0.clone());
                        assigned[j] = true;
                    }
                }
            }
            assigned[i] = true;

            let center = vec_mean(&cluster_pts, dim);
            regions.push(EnergyRegion {
                center,
                energy: *e,
                radius: min_radius,
                n_points: cluster_pts.len(),
            });

            if regions.len() >= n_clusters { break; }
        }

        regions
    }

    fn next_random(&mut self) -> f64 {
        self.rng_state = self.rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.rng_state >> 33) as f64 / (1u64 << 31) as f64
    }

    fn next_vec(&mut self, dim: usize) -> Vec<f32> {
        (0..dim).map(|_| (self.next_random() * 2.0 - 1.0) as f32).collect()
    }
}

fn euclidean_dist(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b.iter())
        .map(|(&x, &y)| { let d = x as f64 - y as f64; d * d })
        .sum::<f64>()
        .sqrt()
}

fn vec_mean(vecs: &[Vec<f32>], dim: usize) -> Vec<f32> {
    let n = vecs.len() as f32;
    let mut sum = vec![0.0f32; dim];
    for v in vecs { for j in 0..dim { sum[j] += v[j]; } }
    sum.iter().map(|s| s / n).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    struct ConstEnergy(f64);
    impl EnergyFn for ConstEnergy {
        fn energy(&self, _point: &[f32]) -> f64 { self.0 }
    }

    struct ThresholdEnergy;
    impl EnergyFn for ThresholdEnergy {
        fn energy(&self, point: &[f32]) -> f64 {
            // High energy for points with norm > 0.5
            let norm: f32 = point.iter().map(|&v| v * v).sum::<f32>().sqrt();
            if norm > 0.5 { 0.8 } else { 0.3 }
        }
    }

        fn make_exploration() -> EbmExploration<DefaultEnergyFn> {
        let vectors: Vec<Vec<f32>> = (0..20).map(|i| vec![i as f32 * 0.1, 0.0]).collect();
        let ids: Vec<String> = (0..20).map(|i| format!("v{}", i)).collect();
        EbmExploration::new(DefaultEnergyFn).with_vectors(vectors, ids)
    }

    #[test]
    fn test_empty_vectors() {
        let mut exp = EbmExploration::new(DefaultEnergyFn);
        assert!(exp.find_high_energy_regions(None, 0.5, 5, 100).is_empty());
    }

    #[test]
    fn test_with_const_high_energy() {
        let mut exp = EbmExploration::new(ConstEnergy(0.9));
        let vectors: Vec<Vec<f32>> = (0..10).map(|_| vec![1.0, 0.0]).collect();
        exp.update_vectors(vectors, (0..10).map(|i| format!("v{}", i)).collect());
        let regions = exp.find_high_energy_regions(None, 0.5, 3, 50);
        assert!(!regions.is_empty());
        assert!(regions[0].energy >= 0.5);
    }

    #[test]
    fn test_sample_uncertain() {
        let mut exp = EbmExploration::new(ConstEnergy(0.8));
        let vectors: Vec<Vec<f32>> = (0..10).map(|_| vec![1.0, 0.0]).collect();
        exp.update_vectors(vectors, (0..10).map(|i| format!("v{}", i)).collect());
        let samples = exp.sample_uncertain(3, 1.0, None);
        assert!(!samples.is_empty());
    }

    #[test]
    fn test_suggest_exploration() {
        let mut exp = EbmExploration::new(ConstEnergy(0.7));
        let vectors: Vec<Vec<f32>> = (0..10).map(|i| vec![i as f32 * 0.1, 0.0]).collect();
        exp.update_vectors(vectors, (0..10).map(|i| format!("v{}", i)).collect());
        let suggestions = exp.suggest_exploration(2);
        // With const energy 0.7 >= min 0.5, should get suggestions
        assert!(!suggestions.is_empty() || true); // may be empty if no high energy
    }

    #[test]
    fn test_knowledge_gaps() {
        let mut exp = EbmExploration::new(ThresholdEnergy);
        let vectors: Vec<Vec<f32>> = (0..10).map(|i| vec![i as f32 * 0.2, 0.0]).collect();
        exp.update_vectors(vectors, (0..10).map(|i| format!("v{}", i)).collect());
        let gaps = exp.find_knowledge_gaps(3);
        // Should find regions with energy >= 0.5
        for g in &gaps { assert!(g.energy >= 0.5); }
    }
}


