//! EBM Energy API - Energy landscape for SplatDB.
//!
//! Ported from Python m2m/ebm/energy_api.py.
//!
//! E(x) = -log(sum_i alpha_i * exp(-kappa_i * ||x - mu_i||^2))
//!
//! Energy ranges:
//!   0.0 - 0.3: High confidence, well-known region
//!   0.3 - 0.7: Moderate confidence
//!   > 0.7:      Low confidence, uncertain or unexplored region

use ndarray::{Array1, Array2};

/// Confidence zone classification for an energy value.
#[derive(Debug, Clone, PartialEq)]
pub enum ConfidenceZone {
    HighConfidence,
    Moderate,
    Uncertain,
}

impl ConfidenceZone {
    pub fn from_energy(energy: f32) -> Self {
        if energy < 0.3 {
            ConfidenceZone::HighConfidence
        } else if energy < 0.7 {
            ConfidenceZone::Moderate
        } else {
            ConfidenceZone::Uncertain
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            ConfidenceZone::HighConfidence => "high_confidence",
            ConfidenceZone::Moderate => "moderate",
            ConfidenceZone::Uncertain => "uncertain",
        }
    }
}

/// Result of an energy computation.
#[derive(Debug, Clone)]
pub struct EnergyResult {
    pub energy: f32,
    pub confidence: f32,
    pub zone: ConfidenceZone,
}

impl EnergyResult {
    pub fn new(energy: f32) -> Self {
        let confidence = 1.0 / (1.0 + energy);
        let zone = ConfidenceZone::from_energy(energy);
        Self { energy, confidence, zone }
    }
}

/// EBM Energy API.
///
/// Exposes the energy landscape learned by Gaussian Splats,
/// allowing agents/LLMs to know the confidence in each result.
pub struct EBMEnergy {
    pub splat_mu: Option<Array2<f32>>,
    pub splat_alpha: Option<Array1<f32>>,
    pub splat_kappa: Option<Array1<f32>>,
}

impl EBMEnergy {
    pub fn new() -> Self {
        Self {
            splat_mu: None,
            splat_alpha: None,
            splat_kappa: None,
        }
    }

    /// Create with initial splat data.
    pub fn with_splats(
        mu: Array2<f32>,
        alpha: Array1<f32>,
        kappa: Array1<f32>,
    ) -> Self {
        Self {
            splat_mu: Some(mu),
            splat_alpha: Some(alpha),
            splat_kappa: Some(kappa),
        }
    }

    /// Update the splats of the energy landscape.
    pub fn update_splats(
        &mut self,
        mu: Array2<f32>,
        alpha: Array1<f32>,
        kappa: Array1<f32>,
    ) {
        self.splat_mu = Some(mu);
        self.splat_alpha = Some(alpha);
        self.splat_kappa = Some(kappa);
    }

    fn has_splats(&self) -> bool {
        self.splat_mu.as_ref().is_some_and(|m| m.nrows() > 0)
    }

    /// Compute energy E(x) for a single vector.
    ///
    /// E(x) = -log(sum_i alpha_i * exp(-kappa_i * ||x - mu_i||^2))
    pub fn energy(&self, vector: &[f32]) -> f32 {
        if !self.has_splats() {
            return 1.0;
        }

        let mu = self.splat_mu.as_ref().expect("has_splats guarantees splat_mu is set");
        let alpha = self.splat_alpha.as_ref().expect("has_splats guarantees splat_alpha is set");
        let kappa = self.splat_kappa.as_ref().expect("has_splats guarantees splat_kappa is set");

        let total: f32 = mu.outer_iter()
            .zip(alpha.iter())
            .zip(kappa.iter())
            .map(|((mu_row, &a), &k)| {
                let dist_sq: f32 = mu_row.iter()
                    .zip(vector.iter())
                    .map(|(m, v)| { let d = m - v; d * d })
                    .sum();
                a * (-k * dist_sq).exp()
            })
            .sum();

        if total < 1e-10 { 10.0 } else { -total.ln() }
    }

    /// Compute energy for multiple vectors (batch).
    pub fn energy_batch(&self, vectors: &Array2<f32>) -> Array1<f32> {
        if !self.has_splats() {
            return Array1::ones(vectors.nrows());
        }

        let mu = self.splat_mu.as_ref().expect("has_splats guarantees splat_mu is set");
        let alpha = self.splat_alpha.as_ref().expect("has_splats guarantees splat_alpha is set");
        let kappa = self.splat_kappa.as_ref().expect("has_splats guarantees splat_kappa is set");

        let energies: Vec<f32> = vectors.outer_iter()
            .map(|row| {
                let total: f32 = mu.outer_iter()
                    .zip(alpha.iter())
                    .zip(kappa.iter())
                    .map(|((mu_row, &a), &k)| {
                        let dist_sq: f32 = mu_row.iter()
                            .zip(row.iter())
                            .map(|(m, v)| { let d = m - v; d * d })
                            .sum();
                        a * (-k * dist_sq).exp()
                    })
                    .sum();
                if total < 1e-10 { 10.0 } else { -total.ln() }
            })
            .collect();

        Array1::from(energies)
    }

    /// Compute the energy gradient nabla-E(x).
    ///
    /// The gradient points toward highest energy ascent.
    /// Its negative (gradient descent) leads to lower energy regions.
    ///
    /// grad = sum_i 2*kappa_i*alpha_i*exp(...)*(x - mu_i) / sum_i alpha_i*exp(...)
    pub fn energy_gradient(&self, vector: &[f32]) -> Vec<f32> {
        if !self.has_splats() {
            return vec![0.0; vector.len()];
        }

        let mu = self.splat_mu.as_ref().expect("has_splats guarantees splat_mu is set");
        let alpha = self.splat_alpha.as_ref().expect("has_splats guarantees splat_alpha is set");
        let kappa = self.splat_kappa.as_ref().expect("has_splats guarantees splat_kappa is set");
        let dim = vector.len();

        let mut gradient = vec![0.0f32; dim];
        let mut total: f32 = 0.0;

        for (mu_row, (&a, &k)) in mu.outer_iter().zip(alpha.iter().zip(kappa.iter())) {
            let dist_sq: f32 = mu_row.iter()
                .zip(vector.iter())
                .map(|(m, v)| { let d = m - v; d * d })
                .sum();
            let exp_term = (-k * dist_sq).exp();
            let factor = 2.0 * k * a * exp_term;

            for (g, (m, v)) in gradient.iter_mut().zip(mu_row.iter().zip(vector.iter())) {
                *g += factor * (v - m);
            }
            total += a * exp_term;
        }

        if total > 1e-10 {
            for g in &mut gradient {
                *g /= total;
            }
        }

        gradient
    }

    /// Free energy of the system: F = -log(Z), where Z = sum(alpha).
    ///
    /// Useful for comparing system states before/after reorganization,
    /// and detecting need for SOC reorganization.
    pub fn free_energy(&self) -> f32 {
        match &self.splat_alpha {
            Some(alpha) if !alpha.is_empty() => {
                let z: f32 = alpha.iter().sum();
                if z > 0.0 { -z.ln() } else { f32::INFINITY }
            }
            _ => f32::INFINITY,
        }
    }

    /// Classify an energy value into a confidence zone.
    pub fn classify_energy(&self, energy: f32) -> ConfidenceZone {
        ConfidenceZone::from_energy(energy)
    }

    /// Get a complete energy result for a vector.
    pub fn get_result(&self, vector: &[f32]) -> EnergyResult {
        let e = self.energy(vector);
        EnergyResult::new(e)
    }

    /// Generate a local 2D energy map around a center point (for visualization).
    pub fn local_energy_map(
        &self,
        center: &[f32],
        radius: f32,
        resolution: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let dim = center.len();
        let mut x_vals = Vec::with_capacity(resolution);
        let mut y_vals = Vec::with_capacity(resolution);
        let mut energy_map = Vec::with_capacity(resolution * resolution);

        let step = if resolution > 1 {
            2.0 * radius / (resolution - 1) as f32
        } else {
            0.0
        };

        for i in 0..resolution {
            let y = -radius + i as f32 * step;
            y_vals.push(y);
            x_vals.clear();
            for j in 0..resolution {
                let x = -radius + j as f32 * step;
                x_vals.push(x);

                let mut point = center.to_vec();
                if dim >= 2 {
                    point[0] = center[0] + x;
                    point[1] = center[1] + y;
                }
                energy_map.push(self.energy(&point));
            }
        }

        // Flatten x_vals for the grid
        let x_grid: Vec<f32> = (0..resolution).flat_map(|_| {
            (0..resolution).map(|j| -radius + j as f32 * step)
        }).collect();

        let y_grid: Vec<f32> = (0..resolution).flat_map(|i| {
            std::iter::repeat_n(-radius + i as f32 * step, resolution)
        }).collect();

        (x_grid, y_grid, energy_map)
    }
}

impl Default for EBMEnergy {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_splats_energy() {
        let ebm = EBMEnergy::new();
        let e = ebm.energy(&[1.0, 2.0, 3.0]);
        assert!((e - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_classify_energy() {
        let ebm = EBMEnergy::new();
        assert_eq!(ebm.classify_energy(0.1), ConfidenceZone::HighConfidence);
        assert_eq!(ebm.classify_energy(0.5), ConfidenceZone::Moderate);
        assert_eq!(ebm.classify_energy(1.0), ConfidenceZone::Uncertain);
    }

    #[test]
    fn test_energy_result() {
        let r = EnergyResult::new(0.5);
        assert!((r.confidence - (1.0 / 1.5)).abs() < 1e-6);
        assert_eq!(r.zone, ConfidenceZone::Moderate);
    }

    #[test]
    fn test_free_energy_no_splats() {
        let ebm = EBMEnergy::new();
        assert!(ebm.free_energy().is_infinite());
    }
}
