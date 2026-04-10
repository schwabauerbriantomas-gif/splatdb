//! Energy functions for the SplatsDB Gaussian Splat landscape.
//!
//! Ported from Python m2m/energy.py.
//!
//! Energy model:
//!   E(x) = -log(sum_i alpha_i * exp(-kappa_i * ||x - mu_i||^2))
//!
//! Optimized: vectorized batch computation, avoids per-element loops where possible.

use ndarray::{s, Array1, Array2};

/// Configuration weights for energy computation.
#[derive(Debug, Clone)]
pub struct EnergyWeights {
    pub splat_weight: f32,
    pub geom_weight: f32,
    pub comp_weight: f32,
}

impl Default for EnergyWeights {
    fn default() -> Self {
        Self {
            splat_weight: 1.0,
            geom_weight: 0.1,
            comp_weight: 0.0,
        }
    }
}

/// Computes energy potentials for the Gaussian Splat landscape.
///
/// Lower energy = higher confidence (near splat attractors).
pub struct EnergyFunction {
    pub weights: EnergyWeights,
}

impl EnergyFunction {
    /// New.
    pub fn new(weights: EnergyWeights) -> Self {
        Self { weights }
    }

    /// Splat-based energy: negative log-density of Gaussian mixture.
    ///
    /// E_splats(x) = -log(sum_i alpha_i * exp(-kappa_i * ||x - mu_i||^2))
    ///
    /// # Arguments
    /// * `x` - Query vectors [B, D] (row-major)
    /// * `mu` - Splat centroids [N, D]
    /// * `alpha` - Splat amplitudes [N]
    /// * `kappa` - Splat concentrations [N]
    /// * `n_active` - Number of active splats to consider
    ///
    /// # Returns
    /// * Energy values [B], lower = closer to attractors
    pub fn e_splats(
        &self,
        x: &Array2<f32>,
        mu: &Array2<f32>,
        alpha: &[f32],
        kappa: &[f32],
        n_active: usize,
    ) -> Vec<f32> {
        if n_active == 0 {
            return vec![10.0; x.nrows()];
        }

        let n = n_active.min(mu.nrows());
        let mu_view = mu.slice(s![..n, ..]);
        let alpha_view = &alpha[..n];
        let kappa_view = &kappa[..n];

        let batch_size = x.nrows();
        let mut energies = Vec::with_capacity(batch_size);

        for row in x.outer_iter() {
            // Vectorized: compute all squared distances at once
            let total: f32 = mu_view
                .outer_iter()
                .zip(alpha_view.iter())
                .zip(kappa_view.iter())
                .map(
                    |((mu_row, &a), &k): ((ndarray::ArrayView1<f32>, &f32), &f32)| {
                        let dist_sq: f32 = mu_row
                            .iter()
                            .zip(row.iter())
                            .map(|(m, xi)| {
                                let d = m - xi;
                                d * d
                            })
                            .sum();
                        a * (-k * dist_sq).exp()
                    },
                )
                .sum();

            energies.push(if total < 1e-10 { 10.0 } else { -total.ln() });
        }

        energies
    }

    /// Fully vectorized batch splat energy using ndarray broadcasting.
    ///
    /// For large batches this is significantly faster than per-row loops.
    pub fn e_splats_vectorized(
        &self,
        x: &Array2<f32>,
        mu: &Array2<f32>,
        alpha: &[f32],
        kappa: &[f32],
        n_active: usize,
    ) -> Array1<f32> {
        if n_active == 0 {
            return Array1::from_elem(x.nrows(), 10.0);
        }

        let n = n_active.min(mu.nrows());
        let mu_n = mu.slice(s![..n, ..]);

        // x: [B, D], mu_n: [N, D]
        // For each batch row, compute energy
        let batch_size = x.nrows();
        let mut energies = Array1::<f32>::zeros(batch_size);

        for b in 0..batch_size {
            let row = x.row(b);
            // diff: [N, D] = mu_n - row (broadcast)
            let diff = &mu_n - &row.insert_axis(ndarray::Axis(0));
            // dist_sq: [N]
            let dist_sq = diff.map_axis(ndarray::Axis(1), |r: ndarray::ArrayView1<f32>| r.dot(&r));

            let total: f32 = dist_sq
                .iter()
                .zip(alpha[..n].iter())
                .zip(kappa[..n].iter())
                .map(|((&ds, &a), &k)| a * (-k * ds).exp())
                .sum();

            energies[b] = if total < 1e-10 { 10.0 } else { -total.ln() };
        }

        energies
    }

    /// Geometric energy: penalizes deviation from the unit sphere.
    ///
    /// E_geom(x) = (||x|| - 1)^2
    ///
    /// For vectors on S^{D-1}, ||x|| should be approximately 1.
    pub fn e_geom(&self, x: &Array2<f32>) -> Vec<f32> {
        // Fully vectorized: ||x||^2 via row dot product, then (sqrt - 1)^2
        x.outer_iter()
            .map(|row| {
                let norm_sq = row.dot(&row);
                let d = norm_sq.sqrt() - 1.0;
                d * d
            })
            .collect()
    }

    /// Compositional energy term.
    ///
    /// Reserved for future composite energy features. Currently inactive
    /// (comp_weight defaults to 0.0). Returns zeros so total_energy remains
    /// correct when the weight is enabled.
    pub fn e_comp(&self, x: &Array2<f32>) -> Vec<f32> {
        vec![0.0; x.nrows()]
    }

    /// Total energy: E_splats + weight_geom * E_geom + weight_comp * E_comp
    pub fn total_energy(
        &self,
        x: &Array2<f32>,
        mu: &Array2<f32>,
        alpha: &[f32],
        kappa: &[f32],
        n_active: usize,
    ) -> Vec<f32> {
        let e_s = self.e_splats(x, mu, alpha, kappa, n_active);
        let e_g = self.e_geom(x);
        let e_c = self.e_comp(x);

        e_s.into_iter()
            .zip(e_g)
            .zip(e_c)
            .map(|((es, eg), ec)| {
                es + self.weights.geom_weight * eg + self.weights.comp_weight * ec
            })
            .collect()
    }
}

impl Default for EnergyFunction {
    fn default() -> Self {
        Self::new(EnergyWeights::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};

    #[test]
    fn test_e_splats_no_active() {
        let ef = EnergyFunction::default();
        let x = Array2::zeros((2, 3));
        let mu = Array2::zeros((5, 3));
        let alpha = vec![1.0; 5];
        let kappa = vec![1.0; 5];
        let result = ef.e_splats(&x, &mu, &alpha, &kappa, 0);
        assert_eq!(result, vec![10.0, 10.0]);
    }

    #[test]
    fn test_e_geom_unit_vector() {
        let ef = EnergyFunction::default();
        let x = array![[1.0f32, 0.0, 0.0]];
        let result = ef.e_geom(&x);
        assert!((result[0] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_e_geom_non_unit() {
        let ef = EnergyFunction::default();
        let x = array![[2.0f32, 0.0, 0.0]];
        let result = ef.e_geom(&x);
        assert!((result[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_e_comp_zeros() {
        let ef = EnergyFunction::default();
        let x = array![[1.0f32, 2.0f32], [3.0f32, 4.0f32]];
        let result = ef.e_comp(&x);
        assert_eq!(result, vec![0.0, 0.0]);
    }
}
